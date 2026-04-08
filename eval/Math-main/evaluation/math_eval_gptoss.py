#!/usr/bin/env python3
"""
Math evaluation for GPT-OSS with optional attention interventions (SMI).
"""
import random
import os
import sys
import argparse
import time
import json
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl
from parser import extract_answer, parse_question, parse_ground_truth, choice_answer_clean
from data_loader import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interventions import (
    StepMomentumInjectionWrapper,
    BridgeGuardOEBWrapper,
)

def parse_args():
    parser = argparse.ArgumentParser(description='Math evaluation for GPT-OSS')
    parser.add_argument("--data_names", default="gsm8k,math500,aime24", type=str,
                       help="Comma-separated dataset names")
    parser.add_argument("--data_dir", default="data/datasets", type=str)
    parser.add_argument("--model_name_or_path", 
                       default="data/models/gpt-oss-20b", 
                       type=str)
    parser.add_argument("--output_dir", default="./outputs/gptoss_math", type=str)
    parser.add_argument("--prompt_type", default="harmony", type=str,
                       help="Prompt type: 'harmony' (recommended), 'tool-integrated', 'cot', etc.")
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int, 
                       help="-1 for full data")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--max_tokens_per_call", default=8192, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    
    # GPT-OSS specific arguments
    parser.add_argument("--reasoning-effort", type=str, default="low",
                       choices=["low", "medium", "high"],
                       help="Reasoning effort level for GPT-OSS")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for generation (currently only 1 supported)")
    parser.add_argument("--adapt_few_shot", action="store_true",
                       help="Few shot for multiple-choice questions, zero shot for others.")
    parser.add_argument("--use-smi", action="store_true",
                       help="Enable Step Momentum Injection (Module 2)")
    parser.add_argument("--smi-strength", type=float, default=0.2,
                       help="Momentum scale for SMI (single tunable parameter)")
    parser.add_argument("--use-oeb", action="store_true",
                       help="Enable shallow-layer Odds-Equal Bridge guard")
    parser.add_argument("--oeb-max-layer", type=int, default=8,
                       help="Maximum decoder layer (inclusive) for OEB (default: 8)")
    parser.add_argument("--oeb-layers", type=str, default="",
                       help="Comma separated shallow layers (e.g. '1,3,5,7'); empty = 0..max_layer")
    
    args = parser.parse_args()
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    effort_suffix = f"_re{args.reasoning_effort}"
    out_file_prefix = (
        f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}"
        f"_t{args.temperature}{effort_suffix}"
    )
    if args.use_smi:
        strength_tag = str(args.smi_strength).replace(".", "p")
        out_file_prefix += f"_smi_s{strength_tag}"
    if args.use_oeb:
        if args.oeb_layers.strip():
            layer_tag = "".join(
                str(int(x.strip()))
                for x in args.oeb_layers.split(",")
                if x.strip()
            )
        else:
            layer_tag = str(args.oeb_max_layer)
        out_file_prefix += f"_oeb_l{layer_tag}"
    # Simplify output path - remove double 'outputs' nesting
    output_dir = args.output_dir
    # Don't add 'outputs/' prefix if it's already an absolute path or contains 'outputs'
    if not output_dir.startswith('/') and 'outputs' not in output_dir:
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # deduplicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def load_gptoss_model(model_path):
    """Load GPT-OSS model and tokenizer"""
    print(f"Loading GPT-OSS model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='left'
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.eval()
    print(f"Model loaded successfully on {model.device}")
    
    return model, tokenizer


def strip_trailing_stop_tokens(text, stop_words):
    """
    Remove stop tokens only when they appear at the tail of the generation.
    This avoids accidentally truncating valid content that happens to contain
    stop markers in the middle (e.g., AIME final answers).
    """
    if not text:
        return text
    trimmed = text
    while True:
        trimmed = trimmed.rstrip()
        removed = False
        for token in stop_words:
            if trimmed.endswith(token):
                trimmed = trimmed[: -len(token)]
                removed = True
                break
        if not removed:
            break
    return trimmed.rstrip()


def generate_with_gptoss(
    model,
    tokenizer,
    prompts,
    args,
):
    """
    Generate outputs using GPT-OSS model (one at a time for simplicity)
    """
    # GPT-OSS stop tokens
    stop_words = ["<|endoftext|>", "<|return|>"]
    
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    
    outputs = []
    
    # Process one at a time (consistent with GPQA approach)
    for prompt in prompts:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=False)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        prompt_len = input_ids.shape[1]
        
        # Generate with exact same settings as GPQA
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_tokens_per_call,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,  # Same as GPQA: use <|return|> as pad
                eos_token_id=tokenizer.convert_tokens_to_ids(['<|return|>', '<|endoftext|>']),  # Same as GPQA
                use_cache=True,
                num_beams=1,  # Greedy, no beam search
            )
        
        # Decode
        generated_text = tokenizer.decode(
            generated_ids[0][prompt_len:], 
            skip_special_tokens=False
        )
        
        # Remove stop words
        generated_text = strip_trailing_stop_tokens(generated_text, stop_words)
        
        outputs.append(generated_text.strip())
    
    return outputs


def setup(args):
    """Setup model and run evaluation"""
    # Load model
    model, tokenizer = load_gptoss_model(args.model_name_or_path)
    
    smi_wrapper = None
    oeb_wrapper = None
    oeb_layers = None
    if args.oeb_layers.strip():
        try:
            parsed = {
                int(item.strip())
                for item in args.oeb_layers.split(",")
                if item.strip()
            }
        except ValueError:
            raise ValueError(f"Invalid --oeb-layers value: {args.oeb_layers}")
        oeb_layers = sorted(x for x in parsed if x >= 0)
    if args.use_smi:
        smi_wrapper = StepMomentumInjectionWrapper(
            tokenizer,
            strength=args.smi_strength,
        )
        smi_wrapper.apply(model)
    if args.use_oeb:
        oeb_wrapper = BridgeGuardOEBWrapper(
            tokenizer,
            max_layer=args.oeb_max_layer,
            layers=oeb_layers,
        )
        oeb_wrapper.apply(model)
    
    try:
        # Infer & eval
        data_list = args.data_names.split(",")
        results = []
        for data_name in data_list:
            results.append(
                main(
                    model,
                    tokenizer,
                    data_name,
                    args,
                    smi_wrapper=smi_wrapper,
                    oeb_wrapper=oeb_wrapper,
                )
            )

        # Add "avg" result to data_list and results
        data_list.append("avg")
        results.append({
            "acc": sum([result["acc"] for result in results]) / len(results),
        })

        # Print all results
        pad = max([len(data_name) for data_name in data_list])
        print("\n" + "=" * 80)
        print("FINAL RESULTS:")
        print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
        print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))
        print("=" * 80)
    finally:
        if smi_wrapper:
            smi_wrapper.remove()
        if oeb_wrapper:
            oeb_wrapper.remove()


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(
    model,
    tokenizer,
    data_name,
    args,
    *,
    smi_wrapper=None,
    oeb_wrapper=None,
):
    """Main evaluation loop for a single dataset"""
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print(f"Dataset: {data_name}")
    if len(processed_samples) > 0:
        print(f"Resume: Found {len(processed_samples)} completed samples")
    print(f"Remaining samples: {len(examples)}")
    if len(examples) == 0:
        print("All samples already processed! Skipping generation.")
        # Still need to evaluate and return metrics
        if len(processed_samples) > 0:
            _, result_json = evaluate(
                samples=processed_samples,
                data_name=data_name,
                prompt_type=args.prompt_type,
                execute=True,
            )
            print(f"\nResults for {data_name}:")
            print(f"  Accuracy: {result_json['acc']:.1f}%")
            print(f"  Saved to: {out_file}")
            return result_json
        else:
            return {"acc": 0.0, "num_samples": 0}
    
    if len(examples) > 0:
        print(f"First example: {examples[0]}")

    # Simplified: No Python execution, just text extraction
    # (This makes it as fast as GPQA evaluation)

    # Prepare samples
    samples = []
    for example in tqdm(examples, total=len(examples), desc="Preparing samples"):
        idx = example["idx"]

        # Parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans

        if idx == args.start:
            print("\n" + "=" * 50)
            print("FIRST QUESTION:")
            print(example["question"])
            print("=" * 50 + "\n")

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
        }

        # Add extra fields
        for key in [
            "level", "type", "unit", "solution_type", "choices", "solution",
            "ques_type", "ans_type", "answer_type", "dataset", "subfield",
            "filed", "theorem", "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # Simplified single-pass generation with real-time saving (like GPQA)
    start_time = time.time()
    total_samples = len(samples)
    
    print(f"\nGenerating completions for {total_samples} samples...")
    
    # Track completed samples for real-time saving
    completed_samples = []
    
    # Generate one sample at a time with progress bar and real-time saving
    for sample in tqdm(samples, desc=f"Evaluating {data_name}", unit="sample"):
        # Create chat prompt - just the question without additional instructions
        messages = [{"role": "user", "content": sample["question"]}]
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=args.reasoning_effort
        )
        
        # Generate
        outputs = generate_with_gptoss(
            model,
            tokenizer,
            [chat_prompt],
            args,
        )
        code = outputs[0].strip()
        
        # Remove stop words from code
        code_stop_words = ["<|endoftext|>", "<|return|>", "\n\n---"]
        code = strip_trailing_stop_tokens(code, code_stop_words)
        
        # Extract prediction immediately
        pred = extract_answer(code, data_name, use_last_number=True)
        
        # Handle multiple choice
        if sample["gt"] in ["A", "B", "C", "D", "E"] and pred not in ["A", "B", "C", "D", "E"]:
            pred = choice_answer_clean(code)
        elif is_multi_choice(sample["gt"]) and not is_multi_choice(pred):
            pred = "".join([c for c in pred if c in ["A", "B", "C", "D", "E"]])
        
        # Update sample with results
        sample.update({"code": [code], "pred": [pred], "report": [""]})
        if smi_wrapper:
            sample["smi_stats"] = smi_wrapper.collect_stats()
        completed_samples.append(sample)
        
        # Real-time save (append mode)
        if args.save_outputs:
            with open(out_file, 'a') as f:
                f.write(json.dumps(sample) + '\n')
    
    time_use = time.time() - start_time
    
    # All samples are now in completed_samples with predictions
    all_samples = completed_samples

    # Add processed samples from previous runs (resume support)
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # Outputs already saved in real-time during generation loop

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minute"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    # Save metrics
    metrics_file = out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(result_json, f, indent=4)
    
    print(f"\nResults for {data_name}:")
    print(f"  Accuracy: {result_json['acc']:.1f}%")
    print(f"  Time: {result_json['time_use_in_minute']}")
    print(f"  Saved to: {out_file}")
    
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)

