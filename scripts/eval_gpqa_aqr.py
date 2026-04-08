#!/usr/bin/env python3
"""
GPQA Diamond evaluation driver (baseline generation only).
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from tqdm import tqdm
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interventions import (
    StepMomentumInjectionWrapper,
    BridgeGuardOEBWrapper,
)
from src.model_config import (
    available_model_types,
    collect_eos_token_ids,
    resolve_model_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description='GPQA Diamond evaluation with interventions')
    parser.add_argument('--model-path', type=str, 
                       default='data/models/gpt-oss-20b',
                       help='Path to model')
    parser.add_argument(
        '--model-type',
        type=str,
        default='auto',
        choices=['auto', *available_model_types()],
        help='Model family (auto-detect, gpt-oss, deepseek-qwen)',
    )
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to evaluate (None = all)')
    parser.add_argument('--max-tokens', type=int, default=2000,
                       help='Maximum generation length')
    parser.add_argument('--output-dir', type=str, default='outputs/gpqa_eval',
                       help='Output directory')
    parser.add_argument('--reasoning-effort', type=str, default='low',
                       choices=['low', 'medium', 'high'],
                       help='Reasoning effort level (low=fast, medium=balanced, high=deep)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing results (skip completed samples)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results (ignore resume)')
    parser.add_argument('--use-smi', action='store_true',
                        help='Enable Step Momentum Injection (Module 2)')
    parser.add_argument('--smi-strength', type=float, default=0.5,
                        help='Momentum scale for SMI (single tunable parameter)')
    parser.add_argument('--use-oeb', action='store_true',
                        help='Enable shallow-layer Odds-Equal Bridge guard')
    parser.add_argument('--oeb-max-layer', type=int, default=None,
                        help='Maximum layer index (inclusive) for OEB (default: auto)')
    parser.add_argument('--oeb-layers', type=str, default='',
                        help='Comma separated layer indices (e.g. "1,3,5,7"); empty = use 0..max-layer')
    return parser.parse_args()


def load_gpqa_dataset(data_path='data/datasets/gpqa/gpqa_diamond.csv'):
    """Load GPQA Diamond dataset from local CSV"""
    df = pd.read_csv(data_path)
    # Convert to list of dicts for compatibility
    dataset = df.to_dict('records')
    return dataset


def format_question(sample):
    """Format question with options"""
    question = sample['Question']
    options = [
        sample['Correct Answer'],
        sample['Incorrect Answer 1'],
        sample['Incorrect Answer 2'],
        sample['Incorrect Answer 3']
    ]
    
    # Shuffle to randomize correct answer position
    import random
    correct_answer = sample['Correct Answer']
    random.shuffle(options)
    answer_letter = chr(65 + options.index(correct_answer))
    
    formatted = question + '\n'
    for i, opt in enumerate(options):
        formatted += f"{chr(65+i)}) {opt}\n"
    
    return formatted, answer_letter, options


def create_prompt(tokenizer, question_text, model_config, reasoning_effort='medium'):
    """Create prompt using official chat template"""
    # Use official chat template with reasoning_effort parameter
    messages = [
        {
        "role": "user",
            "content": question_text
        }
    ]
    
    template_kwargs = {
        "conversation": messages,
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if model_config.supports_reasoning_effort:
        template_kwargs["reasoning_effort"] = reasoning_effort
    prompt = tokenizer.apply_chat_template(**template_kwargs)
    
    return prompt


def extract_answer(response, model_config):
    """Extract answer letter from response"""
    final_section = response
    if model_config.key == "gpt-oss":
        if '<|channel|>final<|message|>' in response:
            final_section = response.split('<|channel|>final<|message|>')[-1]
        for stop_marker in ('<|return|>', '<|end|>'):
            if stop_marker in final_section:
                final_section = final_section.split(stop_marker)[0]
                break
    else:
        final_marker = model_config.final_start_marker
        if final_marker and final_marker in response:
            final_section = response.split(final_marker, 1)[-1]

    for eos_marker in model_config.eos_tokens:
        if eos_marker in final_section:
            final_section = final_section.split(eos_marker)[0]
            break
    
    # Pattern matching
    patterns = [
        r'[Tt]he\s+(?:answer|option)\s+is\s*:?\s*([A-D])',
        r'(?i:Answer)[\s:]*([A-D])',
        r'\(([A-D])\)',
        r'\b([A-D])\b(?=\s*(?:\.|$))',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, final_section)
        if match:
                    return match.group(1)
    
    # Fallback: last capital letter A-D
    matches = re.findall(r'\b([A-D])\b', response)
    if matches:
        return matches[-1]
    
    return None


def load_completed_results(details_file):
    """Load already completed results for resume"""
    if not os.path.exists(details_file):
        return [], set()
    
    completed_results = []
    completed_indices = set()
    
    try:
        with open(details_file, 'r') as f:
            for line in f:
                result = json.loads(line.strip())
                completed_results.append(result)
                completed_indices.add(result['idx'])
        
        return completed_results, completed_indices
    
    except Exception as e:
        print(f"Warning: Failed to load existing results, starting fresh")
        return [], set()


def main():
    args = parse_args()

    explicit_type = None if args.model_type == 'auto' else args.model_type
    model_config = resolve_model_config(
        model_path=args.model_path,
        explicit_type=explicit_type,
    )
    
    mode_bits = ["Baseline"]
    if args.use_smi:
        mode_bits.append("SMI")
    if args.use_oeb:
        mode_bits.append("OEB")
    mode_name = "+".join(mode_bits)
    # Print header
    print(f"\nGPQA Evaluation - {mode_name} - {args.reasoning_effort} reasoning")
    print(f"Samples: {args.num_samples if args.num_samples else 'all'}, Max tokens: {args.max_tokens}\n")
    
    # Load dataset
    dataset = load_gpqa_dataset()
    if args.num_samples:
        dataset = dataset[:min(args.num_samples, len(dataset))]
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    smi_wrapper = None
    oeb_wrapper = None
    if args.use_smi:
        smi_wrapper = StepMomentumInjectionWrapper(
            tokenizer,
            strength=args.smi_strength,
            model_config=model_config,
        )
        smi_wrapper.apply(model)
    oeb_layers = None
    if args.oeb_layers.strip():
        try:
            parsed = {
                int(item.strip())
                for item in args.oeb_layers.split(",")
                if item.strip() != ""
            }
        except ValueError:
            raise ValueError(f"Invalid --oeb-layers value: {args.oeb_layers}")
        oeb_layers = sorted(x for x in parsed if x >= 0)

    if args.use_oeb:
        resolved_oeb_max_layer = (
            args.oeb_max_layer
            if args.oeb_max_layer is not None
            else model_config.default_oeb_max_layer
        )
        oeb_wrapper = BridgeGuardOEBWrapper(
            tokenizer,
            max_layer=resolved_oeb_max_layer,
            layers=oeb_layers,
            model_config=model_config,
        )
        oeb_wrapper.apply(model)
    
    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)
    # Compose informative suffix (reasoning effort only)
    suffix_parts = ['baseline']
    if args.use_smi:
        suffix_parts.append('smi')
        strength_tag = str(args.smi_strength).replace('.', 'p')
        suffix_parts.append(f"s{strength_tag}")
    if args.use_oeb:
        suffix_parts.append('oeb')
        if oeb_layers:
            suffix_parts.append("l" + "".join(str(x) for x in oeb_layers))
        else:
            resolved_oeb_max_layer = (
                args.oeb_max_layer
                if args.oeb_max_layer is not None
                else model_config.default_oeb_max_layer
            )
            suffix_parts.append(f"l{resolved_oeb_max_layer}")
    # Always include reasoning effort to distinguish runs
    suffix_parts.append(f"re{args.reasoning_effort}")
    suffix = '_'.join(suffix_parts)
    details_file = os.path.join(args.output_dir, f'details_{suffix}.jsonl')
    summary_file = os.path.join(args.output_dir, f'summary_{suffix}.json')
    
    # Auto-detect resume if file exists (unless --overwrite specified)
    file_exists = os.path.exists(details_file)
    if file_exists and args.overwrite:
        print(f"Overwrite mode: deleting {details_file}\n")
        os.remove(details_file)
        if os.path.exists(summary_file):
            os.remove(summary_file)
        file_exists = False
    
    if file_exists and not args.resume:
        print(f"Resume: found {details_file}, continuing...\n")
        args.resume = True
    
    # Resume: Load completed results if requested or auto-detected
    if args.resume and file_exists:
        results, completed_indices = load_completed_results(details_file)
        correct_count = sum(1 for r in results if r.get('is_correct', False))
    else:
        results = []
        completed_indices = set()
        correct_count = 0
    
    # Evaluation
    total_samples = len(dataset)
    completed_count = len(completed_indices)
        
    if args.resume and completed_count > 0:
        print(f"Resuming: {completed_count} done, {total_samples - completed_count} remaining\n")
    
    # Use tqdm with initial value for better progress display
    pbar = tqdm(total=total_samples, desc="Evaluating", initial=completed_count)
    
    for idx, sample in enumerate(dataset):
        # Skip if already completed (resume mode)
        if idx in completed_indices:
            continue  # Don't update pbar - already counted in initial
        
        try:
            # Format question
            question_text, answer_letter, options = format_question(sample)
            prompt = create_prompt(tokenizer, question_text, model_config, args.reasoning_effort)
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
            prompt_len = inputs.input_ids.shape[1]
            
            # Generate
            with torch.no_grad():
                eos_token_ids = collect_eos_token_ids(tokenizer, model_config)
                pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

                generation_kwargs = dict(
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )
                if eos_token_ids:
                    generation_kwargs["eos_token_id"] = eos_token_ids if len(eos_token_ids) > 1 else eos_token_ids[0]

                outputs = model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=False)
            
            # Extract answer
            pred_letter = extract_answer(response, model_config)
            is_correct = (pred_letter == answer_letter)
            if is_correct:
                correct_count += 1
            
            result = {
                'idx': idx,
                'question': sample['Question'],
                'subdomain': sample.get('Subdomain', 'Unknown'),
                'options': options,
                'answer_letter': answer_letter,
                'pred_letter': pred_letter,
                'pred_response': response,
                'is_correct': is_correct,
            }
            if smi_wrapper:
                result['smi_stats'] = smi_wrapper.collect_stats()
            if oeb_wrapper:
                result['oeb_stats'] = oeb_wrapper.collect_stats()
            
            results.append(result)
            
            # Write to details file (clean version without stats)
            with open(details_file, 'a') as f:
                f.write(json.dumps(result) + '\n')
            
            pbar.update(1)  # Update progress bar after completing a sample
        
        except Exception as e:
            print(f"\nError on sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            pbar.update(1)
            continue
    
    pbar.close()
    
    # Calculate summary (detailed results already written incrementally)
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0.0
    
    summary = {
        'mode': mode_name,
        'total_samples': total,
        'correct': correct_count,
        'accuracy': accuracy,
        'max_tokens': args.max_tokens,
        'reasoning_effort': args.reasoning_effort,
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Results: {correct_count}/{total} correct ({accuracy*100:.1f}%)")
    print(f"Saved to: {details_file}")
    print(f"{'='*60}\n")

    if smi_wrapper:
        smi_wrapper.remove()
    if oeb_wrapper:
        oeb_wrapper.remove()


if __name__ == '__main__':
    main()
