#!/usr/bin/env python3
"""
Generate attention saliency maps for GPT-OSS-20B model.
Uses Grad×Activation (A×∇A) method to compute true saliency maps.
Analyzes different prompt types: Normal, Zero-shot CoT, Few-shot CoT.
Memory-optimized version with layer-wise processing.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from datasets import load_dataset
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LogNorm
from datetime import datetime
import re
import pickle
import json
from typing import Dict, List, Tuple, Any
import gc
import warnings
import argparse

# Suppress specific repetitive warnings
warnings.filterwarnings("ignore", message="Caching is incompatible with gradient checkpointing")
warnings.filterwarnings("ignore", message="The following generation flags are not valid")
warnings.filterwarnings("ignore", message="The attention mask is not set and cannot be inferred")
warnings.filterwarnings("ignore", message="`use_cache=True` is incompatible with gradient checkpointing")
warnings.filterwarnings("ignore", category=UserWarning)

from saliency_extractor import extract_saliency, extract_saliency_gptoss

class StopOnAnyToken(StoppingCriteria):
    """Custom stopping criteria that stops on any of the specified token IDs"""
    def __init__(self, stop_ids):
        self.stop_ids = set(stop_ids)

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1].item() in self.stop_ids

def get_special_id(tokenizer, tok: str):
    """More reliable way to get special token ID"""
    ids = tokenizer(tok, add_special_tokens=False).input_ids
    return ids[0] if ids else None

def configure_stop_conditions(tokenizer):
    """Configure proper stop conditions for GPT-OSS Harmony format"""
    # Get EOS token ID - prefer <|return|> for GPT-OSS, fallback to standard EOS
    return_token_id = get_special_id(tokenizer, "<|return|>")
    fallback = tokenizer.eos_token_id

    # Primary EOS token for generation - ONLY <|return|> should stop generation
    eos_token_id = return_token_id or fallback

    # Only stop on <|return|> - do NOT stop on <|end|> as it just ends analysis block
    stop_ids = [i for i in [return_token_id, fallback] if i is not None]

    # Create stopping criteria - only stop on <|return|>
    stopping_criteria = StoppingCriteriaList([StopOnAnyToken(stop_ids)])

    return eos_token_id, stop_ids, stopping_criteria

# Harmony format parsing based on actual tokens (flexible for incomplete sequences)
H_RE_COMPLETE = re.compile(
    r"<\|start\|>assistant<\|channel\|>analysis<\|message\|>(?P<ana>.*?)"
    r"<\|start\|>assistant<\|channel\|>final<\|message\|>(?P<fin>.*?)(?:<\|end\|>|<\|return\|>|$)",
    re.DOTALL
)

H_RE_ANALYSIS_ONLY = re.compile(
    r"<\|start\|>assistant<\|channel\|>analysis<\|message\|>(?P<ana>.*?)(?:<\|start\|>assistant<\|channel\|>final<\|message\|>|<\|end\|>|<\|return\|>|$)",
    re.DOTALL
)

def parse_harmony_blocks(full_text: str):
    """Parse analysis and final blocks from raw Harmony format text"""
    # Try complete pattern first
    m = H_RE_COMPLETE.search(full_text)
    if m:
        return m.group("ana"), m.group("fin")

    # Try analysis-only pattern for incomplete sequences
    m = H_RE_ANALYSIS_ONLY.search(full_text)
    if m:
        return m.group("ana"), None

    return None, None

def validate_harmony_from_raw(raw_text: str):
    """Validate Harmony format based on raw text with special tokens"""
    ana, fin = parse_harmony_blocks(raw_text)
    has_analysis = ana is not None and len(ana.strip()) > 0
    has_final = fin is not None and len(fin.strip()) > 0

    # Prefer complete analysis+final, but accept analysis-only for debugging
    ok = has_analysis and has_final  # Now require both for full validation
    return ok, {"has_analysis": has_analysis, "has_final": has_final}



def parse_harmony_sections(complete_sequence, tokenizer):
    """Parse Harmony sequence and return section boundaries for visualization"""
    sections = []

    # Find user question section
    user_match = re.search(r'<\|start\|>user<\|message\|>(.*?)<\|end\|>', complete_sequence, re.DOTALL)
    if user_match:
        user_start = complete_sequence.find('<|start|>user<|message|>')
        user_end = complete_sequence.find('<|end|>', user_start) + len('<|end|>')
        sections.append({
            'name': 'Question',
            'start_char': user_start,
            'end_char': user_end,
            'content': user_match.group(1).strip()
        })

    # Find analysis section
    analysis_match = re.search(r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>', complete_sequence, re.DOTALL)
    if analysis_match:
        analysis_start = complete_sequence.find('<|start|>assistant<|channel|>analysis<|message|>')
        analysis_end = complete_sequence.find('<|end|>', analysis_start) + len('<|end|>')
        sections.append({
            'name': 'Reasoning',
            'start_char': analysis_start,
            'end_char': analysis_end,
            'content': analysis_match.group(1).strip()
        })

    # Find final section
    final_match = re.search(r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)<\|return\|>', complete_sequence, re.DOTALL)
    if final_match:
        final_start = complete_sequence.find('<|start|>assistant<|channel|>final<|message|>')
        final_end = complete_sequence.find('<|return|>', final_start) + len('<|return|>')
        sections.append({
            'name': 'Answer',
            'start_char': final_start,
            'end_char': final_end,
            'content': final_match.group(1).strip()
        })

    # Convert character positions to token positions
    for section in sections:
        # Tokenize the prefix to get start token position
        prefix = complete_sequence[:section['start_char']]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        section['start_token'] = len(prefix_tokens)

        # Tokenize the section content to get end token position
        section_text = complete_sequence[section['start_char']:section['end_char']]
        section_tokens = tokenizer.encode(section_text, add_special_tokens=False)
        section['end_token'] = section['start_token'] + len(section_tokens)

    return sections

def filter_sequence_for_saliency(complete_sequence, tokenizer):
    """Filter sequence to focus on meaningful content using Harmony parsing."""
    # Parse Harmony blocks from the complete sequence
    analysis, final = parse_harmony_blocks(complete_sequence)

    # Extract user question from the sequence
    user_match = re.search(r'<\|start\|>user<\|message\|>(.*?)<\|end\|>', complete_sequence, re.DOTALL)
    user_question = user_match.group(1).strip() if user_match else ""

    # Build filtered sequence: Question + Analysis + Final Answer
    filtered_parts = []

    if user_question:
        filtered_parts.append(f"Question: {user_question}")

    if analysis and analysis.strip():
        filtered_parts.append(f"Reasoning: {analysis.strip()}")

    if final and final.strip():
        filtered_parts.append(f"Answer: {final.strip()}")

    # Join with clear separators
    filtered_sequence = "\n\n".join(filtered_parts)

    return filtered_sequence if filtered_sequence else complete_sequence

def extract_choice_answer(text, keys=("A", "B", "C", "D", "E")):
    """Extract multiple-choice answer (A/B/C/D/E) from response
    
    Args:
        text: Response text to extract from
        keys: Valid option keys (default: A-E)
    
    Returns:
        str: Single letter (A/B/C/D/E) or None
    """
    if not text:
        return None
    
    # Priority 1: Extract from <|channel|>final block
    final_match = re.search(r'<\|channel\|>final<\|message\|>(.*?)<\|(?:return|end)\|>', text, re.DOTALL)
    if final_match:
        final_text = final_match.group(1).strip()
    else:
        final_text = text
    
    # Priority 2: Look for explicit answer patterns
    patterns = [
        r'\b([A-E])\b',                    # Standalone letter
        r'[Aa]nswer[:\s]+([A-E])',        # "Answer: A" or "answer A"
        r'[Oo]ption\s+([A-E])',           # "option A"
        r'\b([A-E])\)',                   # "A)" format
        r'choose\s+([A-E])',              # "choose A"
        r'select\s+([A-E])',              # "select A"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, final_text, re.IGNORECASE)
        if matches:
            # Return the last match (usually the final answer)
            candidate = matches[-1].upper()
            if candidate in keys:
                return candidate
    
    return None


def extract_numerical_answer(text):
    """Extract answer from LaTeX response with nested brace support
    
    Supports multiple formats:
    1. \\boxed{answer} - standard MATH format
    2. <|channel|>final<|message|>...answer... - Harmony Chat format (GPT-OSS)
    3. LaTeX equation blocks with = sign
    """
    if not text:
        return None
    
    # Method 1: Find \boxed{...} with proper brace matching to handle nested braces like \frac{14}{3}
    boxed_pos = text.find('\\boxed{')
    if boxed_pos != -1:
        start = boxed_pos + len('\\boxed{')
        brace_count = 1
        i = start
        
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        
        if brace_count == 0:
            content = text[start:i-1].strip()
            # Clean up LaTeX spacing commands
            content = re.sub(r'\\[,;:!]', '', content)
            content = re.sub(r'\\quad|\\qquad', ' ', content)
            # Normalize LaTeX variants
            content = content.replace('\\dfrac', '\\frac')
            content = content.replace('\\tfrac', '\\frac')
            return content
    
    # Method 2: Extract from Harmony Chat final channel (for GPT-OSS)
    final_match = re.search(r'<\|channel\|>final<\|message\|>(.*?)<\|(?:return|end)\|>', text, re.DOTALL)
    if final_match:
        final_content = final_match.group(1).strip()
        
        # Extract content from LaTeX block \[ ... \] or \( ... \)
        latex_match = re.search(r'\\[\[\(]\s*(.*?)\s*\\[\]\)]', final_content, re.DOTALL)
        if latex_match:
            latex_content = latex_match.group(1).strip()
    else:
            # No LaTeX delimiters, use the whole content
            latex_content = final_content.strip()
        
        # Try to extract right-hand side of final equation
        # Pattern: expression = answer (possibly with line breaks)
        lines = latex_content.split('\n')
        for line in reversed(lines):  # Check from bottom up
            line = line.strip()
            if '=' in line:
                # Extract everything after the last = sign
                parts = line.split('=')
                answer = parts[-1].strip()
                # Clean up common LaTeX artifacts
                answer = re.sub(r'\\[,;:!]', '', answer)
                answer = re.sub(r'\\quad|\\qquad', ' ', answer)
                answer = re.sub(r'\\\\$', '', answer)  # Remove trailing \\
                if answer:  # Make sure we got something
                    return answer
        
        # If no equation found, return the whole content (cleaned)
        latex_content = re.sub(r'\\[,;:!]', '', latex_content)
        latex_content = re.sub(r'\\quad|\\qquad', ' ', latex_content)
        return latex_content.strip()
    
    # Method 3: Fallback - extract any LaTeX block and try equation pattern
    latex_block = re.search(r'\\[\[\(](.*?)\\[\]\)]', text, re.DOTALL)
    if latex_block:
        content = latex_block.group(1).strip()
        # Try to find final equation
        lines = content.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if '=' in line:
                parts = line.split('=')
                answer = parts[-1].strip()
                answer = re.sub(r'\\\\$', '', answer)
                if answer:
                    return answer
    
    return None


def compare_answers(predicted, ground_truth, dataset_name='math'):
    """Compare answers using symbolic equality (supports MATH, AIME24, and GPQA)"""
    if predicted is None or ground_truth is None:
        return False
    
    pred_str = str(predicted).strip()
    truth_str = str(ground_truth).strip()
    
    # GPQA-specific comparison (letter matching)
    if dataset_name == 'gpqa':
        # Simple case-insensitive letter comparison
        return pred_str.upper() == truth_str.upper()
    
    # AIME24-specific normalization (answer is typically an integer 0-999)
    if dataset_name == 'aime24':
        # Extract numeric value from \boxed{...} format
        def extract_aime_answer(s):
            # Try to extract from \boxed{}
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', s)
            if boxed_match:
                s = boxed_match.group(1)
            
            # Remove LaTeX and spaces
            s = s.replace('\\', '').replace(' ', '').replace(',', '')
            
            # Handle fractions: a/b
            if '/' in s:
                try:
                    parts = s.split('/')
                    return float(parts[0]) / float(parts[1])
                except (ValueError, TypeError, ZeroDivisionError, IndexError):
                    pass
            
            # Try direct numeric conversion
            try:
                return float(s)
            except (ValueError, TypeError):
                return s
        
        pred_val = extract_aime_answer(pred_str)
        truth_val = extract_aime_answer(truth_str)
        
        # Numeric comparison
        try:
            pred_num = float(pred_val)
            truth_num = float(truth_val)
            return abs(pred_num - truth_num) < 1e-6
        except (ValueError, TypeError):
            # String comparison as fallback
            return str(pred_val).lower() == str(truth_val).lower()
    
    # MATH dataset processing (original logic)
    # Normalize LaTeX variants
    def normalize_latex(s):
        s = s.replace('\\dfrac', '\\frac')
        s = s.replace('\\tfrac', '\\frac')
        s = s.replace('\\left(', '(').replace('\\right)', ')')
        s = s.replace('\\left[', '[').replace('\\right]', ']')
        return s
    
    pred_norm = normalize_latex(pred_str)
    truth_norm = normalize_latex(truth_str)
    
    # Direct string match
    if pred_norm == truth_norm:
            return True
    
    # Try sympy symbolic comparison
    try:
        from latex2sympy2 import latex2sympy
        import sympy
        
        pred_sym = latex2sympy(pred_norm)
        truth_sym = latex2sympy(truth_norm)
        
        if pred_sym == truth_sym:
            return True
        
        # Check after simplification
        try:
            diff = sympy.simplify(pred_sym - truth_sym)
            if diff == 0:
                return True
        except (ValueError, TypeError, AttributeError):
            pass
        
        # Numerical evaluation
        try:
            return abs(float(pred_sym) - float(truth_sym)) < 1e-6
        except (ValueError, TypeError):
            pass
    except (ImportError, ValueError, TypeError):
        pass
    
    # Fallback: normalized string comparison
    pred_clean = pred_norm.replace(' ', '').replace('\\', '').replace(',', '').lower()
    truth_clean = truth_norm.replace(' ', '').replace('\\', '').replace(',', '').lower()
    
    return pred_clean == truth_clean


def save_conversation_data(conversation_data: Dict, output_dir: str, problem_id: int, prompt_type: str):
    """Save conversation data as JSON"""
    os.makedirs(output_dir, exist_ok=True)

    json_filename = f"{prompt_type}_problem_{problem_id:02d}.json"
    json_path = os.path.join(output_dir, json_filename)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(conversation_data, f, indent=2, ensure_ascii=False)

    return json_path


def aggregate_samples(problem_dir, num_samples, ground_truth, dataset_name='math'):
    """
    Aggregate results from multiple samples for one problem
    
    Args:
        problem_dir: Problem directory path (e.g., outputs/.../math_001)
        num_samples: Number of samples
        ground_truth: Correct answer
        dataset_name: Dataset name for answer comparison
    
    Returns:
        dict: Metadata including accuracy, predictions, etc.
    """
    from pathlib import Path
    
    problem_path = Path(problem_dir)
    predictions = []
    correct_count = 0
    
    # Iterate through all samples
    for sample_idx in range(num_samples):
        sample_dir = problem_path / f"sample_{sample_idx:02d}"
        conversation_dir = sample_dir / "conversations"
        
        # Find conversation file
        conversation_files = list(conversation_dir.glob("*_problem_*.json"))
        if not conversation_files:
            predictions.append(None)
            continue
        
        # Read predicted answer
        with open(conversation_files[0], 'r', encoding='utf-8') as f:
            conv_data = json.load(f)
        
        predicted = conv_data.get('predicted_answer')
        predictions.append(predicted)
        
        # Compare answers using dataset-specific comparison
        if compare_answers(predicted, ground_truth, dataset_name=dataset_name):
            correct_count += 1
    
    accuracy = correct_count / num_samples if num_samples > 0 else 0.0
    
    # Save aggregated metadata
    metadata = {
        'num_samples': num_samples,
        'accuracy': accuracy,
        'num_correct': correct_count,
        'predictions': predictions,
        'ground_truth': ground_truth
    }
    
    metadata_path = problem_path / "aggregated_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return metadata


# Model configuration paths (override via --model-path CLI argument)
MODEL_PATHS = {
    'gpt-oss': "data/models/gpt-oss-20b",
}

def load_problems_from_gpqa_diamond(csv_path, limit=None, offset=0):
    """Load GPQA-diamond dataset from CSV
    
    Args:
        csv_path: Path to gpqa_diamond.csv
        limit: Maximum number of problems to load
        offset: Starting index
    
    Returns:
        list: Problem entries with format {"problem_id", "question", "options", "answer"}
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Adaptive column mapping
    question_col = None
    for col in ['Question', 'question', 'prompt', 'stem']:
        if col in df.columns:
            question_col = col
            break
    
    correct_col = None
    for col in ['Correct Answer', 'correct_answer', 'answer']:
        if col in df.columns:
            correct_col = col
            break
    
    # GPQA diamond has Incorrect Answer 1/2/3
    incorrect_cols = []
    for i in [1, 2, 3]:
        for col in [f'Incorrect Answer {i}', f'incorrect_answer_{i}']:
            if col in df.columns:
                incorrect_cols.append(col)
                break
    
    # Record ID
    id_col = None
    for col in ['Record ID', 'record_id', 'id', 'problem_id']:
        if col in df.columns:
            id_col = col
            break
    
    if question_col is None or correct_col is None or len(incorrect_cols) < 3:
        raise ValueError(f"Missing required columns in {csv_path}. Found columns: {list(df.columns)}")
    
    problems = []
    start_idx = offset
    end_idx = min(offset + limit, len(df)) if limit else len(df)
    
    for idx in range(start_idx, end_idx):
        row = df.iloc[idx]
        
        # Generate problem ID
        if id_col and pd.notna(row[id_col]):
            # Use record ID if available
            record_id = str(row[id_col]).replace('rec', '').strip()
            problem_id = f"gpqa_{record_id[:8]}"
        else:
            # Fallback to zero-padded index
            problem_id = f"gpqa_{idx:04d}"
        
        # Build options: A=Correct, B/C/D=Incorrect 1/2/3
        options = [
            {"key": "A", "text": str(row[correct_col]).strip()},
            {"key": "B", "text": str(row[incorrect_cols[0]]).strip()},
            {"key": "C", "text": str(row[incorrect_cols[1]]).strip()},
            {"key": "D", "text": str(row[incorrect_cols[2]]).strip()},
        ]
        
        problems.append({
            'problem_id': problem_id,
            'question': str(row[question_col]).strip(),
            'options': options,
            'answer': 'A'  # Correct answer is always A by design
        })
    
    return problems


def load_dataset_by_name(dataset_name='math', parquet_path=None, gpqa_path=None, data_dir=None):
    """Load dataset from local cache"""
    dataset_cache_dir = data_dir or os.environ.get("DATA_DIR", "data/datasets")

    if dataset_name == 'math':
        print("Loading MATH-500 dataset...")
        from datasets import Dataset
        math_path = f"{dataset_cache_dir}/MATH-500/test.jsonl"
        problems = []
        with open(math_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                problems.append({
                    'question': data['problem'],
                    'answer': data['answer'],
                })
        dataset = {'test': Dataset.from_list(problems)}
        print(f"Loaded {len(problems)} MATH problems")
    elif dataset_name == 'aime24':
        print("Loading AIME24 dataset from parquet...")
        from datasets import Dataset
        import pandas as pd
        
        if parquet_path is None:
            parquet_path = f"{dataset_cache_dir}/aime24/test-00000-of-00001 (3).parquet"
        
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            raise ValueError(f"Failed to load AIME24 parquet from {parquet_path}: {e}")
        
        # Adaptive column mapping
        id_candidates = ["id", "problem_id", "index"]
        question_candidates = ["question", "problem", "prompt"]
        answer_candidates = ["answer", "final_answer", "ground_truth", "label", "solution"]
        
        id_col = next((c for c in id_candidates if c in df.columns), None)
        question_col = next((c for c in question_candidates if c in df.columns), None)
        answer_col = next((c for c in answer_candidates if c in df.columns), None)
        
        if not question_col or not answer_col:
            raise ValueError(f"Cannot find required columns. Available: {df.columns.tolist()}")
        
        problems = []
        for idx, row in df.iterrows():
            # Get problem ID
            if id_col:
                prob_id = f"aime24_{row[id_col]:04d}"
            else:
                prob_id = f"aime24_{idx+1:04d}"
            
            # Extract answer from solution (AIME format is already \boxed{...})
            answer_text = str(row[answer_col])
            
            problems.append({
                'problem_id': prob_id,
                'question': str(row[question_col]),
                'answer': answer_text,
            })
        
        dataset = {'test': Dataset.from_list(problems)}
        print(f"Loaded {len(problems)} AIME24 problems")
    elif dataset_name == 'gpqa':
        print("Loading GPQA-diamond dataset from CSV...")
        from datasets import Dataset
        
        if gpqa_path is None:
            gpqa_path = f"{dataset_cache_dir}/gpqa/gpqa_diamond.csv"
        
        # Use the helper function to load GPQA
        problems = load_problems_from_gpqa_diamond(gpqa_path)
        dataset = {'test': Dataset.from_list(problems)}
        print(f"Loaded {len(problems)} GPQA-diamond problems")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset

def load_model_and_tokenizer(model_type='gpt-oss'):
    """Load model and tokenizer based on model type (unified loader)"""
    if model_type not in MODEL_PATHS:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: {list(MODEL_PATHS.keys())}")

    model_path = MODEL_PATHS[model_type]
    print(f"Loading {model_type} tokenizer and model from {model_path}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True
    )

    # GPT-OSS requires explicit chat template loading
    template_path = f"{model_path}/chat_template.jinja"
    with open(template_path, 'r', encoding='utf-8') as f:
        tokenizer.chat_template = f.read()
        print(f"Loaded chat template from: {template_path}")

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with unified configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )

    # Set to eval mode (no gradient checkpointing needed for generation)
    model.eval()

    print(f"Successfully loaded {model_type} model")
    return tokenizer, model

def load_model_and_data(dataset_name='math', model_type='gpt-oss', parquet_path=None, gpqa_path=None):
    """Load model and dataset (unified entry point)"""
    tokenizer, model = load_model_and_tokenizer(model_type)
    dataset = load_dataset_by_name(dataset_name, parquet_path=parquet_path, gpqa_path=gpqa_path)
    return tokenizer, model, dataset

def get_sample_problems(dataset, num_problems=4, dataset_name='math'):
    """Select first few problems from dataset (supports MATH, AIME24, and GPQA)"""
    problems = []
    test_data = dataset['test']

    for i in range(min(num_problems, len(test_data))):
        problem = test_data[i]
        
        # GPQA has problem_id and options fields
        if dataset_name == 'gpqa':
            problem_entry = {
                'id': i,
                'problem_id': problem.get('problem_id', f'gpqa_{i:04d}'),
            'question': problem['question'],
            'answer': problem['answer'],
                'options': problem.get('options', []),
            'index': i
            }
        # AIME24 has problem_id field
        elif dataset_name == 'aime24' and 'problem_id' in problem:
            problem_entry = {
                'id': i,
                'problem_id': problem['problem_id'],
                'question': problem['question'],
                'answer': problem['answer'],
                'index': i
            }
        else:
            problem_entry = {
                'id': i,
                'question': problem['question'],
                'answer': problem['answer'],
                'index': i
            }
        
        problems.append(problem_entry)

    return problems

def create_zero_shot_prompt(question):
    """Create zero-shot CoT prompt for MATH dataset"""
    return f"{question}"

def clean_token(token):
    """Clean tokens to remove special symbols for display and escape LaTeX characters"""
    cleaned = token.replace('Ġ', ' ')
    cleaned = cleaned.replace('âĢĻ', "'")
    cleaned = cleaned.replace('Ċ', '\\n')
    cleaned = cleaned.replace('ĉ', '\\t')

    # Escape LaTeX special characters for matplotlib
    # These characters have special meaning in LaTeX and need to be escaped
    latex_special_chars = {
        '$': '\\$',
        '%': '\\%',
        '&': '\\&',
        '#': '\\#',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '^': '\\textasciicircum{}',
        '\\': '\\textbackslash{}'
    }

    for char, escaped in latex_special_chars.items():
        cleaned = cleaned.replace(char, escaped)

    if len(cleaned) > 10:
        cleaned = cleaned[:8] + '..'
    return cleaned

def save_saliency_data(saliency_data, output_dir, problem_id, prompt_type):
    """Save complete saliency data as a single pickle file"""
    data_dir = os.path.join(output_dir, "saliency_data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Save entire saliency_data as a single pickle file
    pkl_file = os.path.join(data_dir, f"{prompt_type}_problem_{problem_id:02d}_saliency.pkl")
    
    with open(pkl_file, 'wb') as f:
        pickle.dump(saliency_data, f)
    
    return pkl_file

def generate_enhanced_saliency_heatmap(saliency_matrix, tokens, layer_idx, layer_type, output_path,
                                     prompt_type="", problem_id=0, step_size=1, format_valid=True,
                                     harmony_sections=None):
    """Generate enhanced saliency heatmap with content annotations and sampling info"""
    max_tokens = len(tokens)
    display_matrix = saliency_matrix[:max_tokens, :max_tokens]
    display_tokens = tokens[:max_tokens]

    # Calculate dynamic figure size
    base_size = 12
    token_factor = max_tokens / 40
    fig_width = max(16, base_size + token_factor * 4)
    fig_height = max(12, base_size + token_factor * 3)

    if max_tokens > 200:
        fig_width = max(fig_width, 24)
        fig_height = max(fig_height, 20)

    plt.figure(figsize=(fig_width, fig_height))

    # Improve visualization with LogNorm (logarithmic normalization)
    # Note: display_matrix already has causal mask applied (upper triangle is zero)

    # Get valid (non-zero, lower-triangle) values for color scale calculation
    valid_nonzero = display_matrix[display_matrix > 0]

    if len(valid_nonzero) > 0:
        # Use logarithmic normalization for better color distribution
        vmin = 0.0015  # Slightly increase vmin for darker colors
        vmax = np.percentile(valid_nonzero, 93)  

    sns.heatmap(
        display_matrix,
        xticklabels=display_tokens,
        yticklabels=display_tokens,
        cmap='Reds',
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cbar_kws={'label': 'Saliency Score (log scale)'},
        square=True,
            linewidths=0.02,
        cbar=True
    )
    else:
        # Fallback: original visualization without LogNorm
    sns.heatmap(
        display_matrix,
        xticklabels=display_tokens,
        yticklabels=display_tokens,
        cmap='Reds',
        cbar_kws={'label': 'Saliency Score (Grad×Activation)'},
        square=True,
        linewidths=0.02,
        cbar=True
    )

    # Enhanced title with format validation status
    format_status = "Valid Format" if format_valid else "Format Issues"
    sampling_info = f"Sampling: 1:{step_size}" if step_size > 1 else "No Sampling"

    title = f'GPT-OSS Layer {layer_idx:02d} ({layer_type}) - Enhanced Attention Saliency\n'
    title += f'Problem {problem_id} | {prompt_type.replace("_", " ").title()} | {format_status}\n'
    title += f'Content Focus: Q+Reasoning+Answer | {sampling_info} | Tokens: {max_tokens}'

    plt.title(title, fontsize=13, fontweight='bold', pad=25)
    plt.xlabel('Key Tokens (Filtered Content)', fontsize=12)
    plt.ylabel('Query Tokens (Filtered Content)', fontsize=12)

    # Add simple text annotations for sections (no colored backgrounds)
    if harmony_sections and len(harmony_sections) > 0:
        try:
            ax = plt.gca()
            for section in harmony_sections:
                section_name = section['name']
                # Adjust token positions for sampling
                start_pos = section['start_token'] // step_size
                end_pos = min(section['end_token'] // step_size, max_tokens)
                mid_pos = (start_pos + end_pos) / 2

                if start_pos < max_tokens and mid_pos < max_tokens - 2:
                    # Simple text label on the right side, no background color
                    ax.text(max_tokens + 0.5, mid_pos, section_name,
                           fontsize=9, ha='left', va='center', weight='bold',
                           color='black')
        except Exception as e:
            print(f"Warning: Could not add annotations: {e}")
            pass

    # Adjust tick formatting
    if max_tokens > 150:
        fontsize = 5
    elif max_tokens > 100:
        fontsize = 6
    else:
        fontsize = 7

    plt.xticks(rotation=45, ha='right', fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    plt.tight_layout(pad=2.0)

    # Save with metadata
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close()

def calculate_saliency_statistics(saliency_result, tokens):
    """Calculate comprehensive saliency statistics"""
    stats = {
        'first_token_saliency': [],
        'layer_stats': [],
        'tokens': tokens,
        'sequence_length': len(tokens),
        'sw_layers': [],
        'full_layers': []
    }
    
    for layer_idx, (layer_info, saliency_matrix) in enumerate(zip(
        saliency_result.layers_info, saliency_result.saliency_per_layer
    )):
        if saliency_matrix.dtype == torch.bfloat16:
            saliency_matrix = saliency_matrix.float()
        
        avg_saliency = saliency_matrix.mean(dim=1)[0].cpu().numpy()
        
        seq_len = avg_saliency.shape[0]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        avg_saliency = avg_saliency * causal_mask
        
        row_sums = avg_saliency.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        normalized_saliency = avg_saliency / row_sums
        
        first_token_sal = normalized_saliency[:, 0]
        avg_first_token_sal = np.mean(first_token_sal)
        
        saliency_entropy = -np.sum(normalized_saliency * np.log(normalized_saliency + 1e-10), axis=1)
        avg_entropy = np.mean(saliency_entropy)
        
        self_sal = np.diag(normalized_saliency)
        avg_self_sal = np.mean(self_sal)
        
        layer_stat = {
            'layer': layer_idx,
            'layer_type': layer_info.kind,
            'first_token_saliency': avg_first_token_sal,
            'saliency_entropy': avg_entropy,
            'self_saliency': avg_self_sal,
            'saliency_matrix': normalized_saliency,
            'heads': layer_info.heads
        }
        
        stats['layer_stats'].append(layer_stat)
        stats['first_token_saliency'].append(avg_first_token_sal)
        
        if layer_info.kind == 'SW':
            stats['sw_layers'].append(layer_idx)
        elif layer_info.kind == 'Full':
            stats['full_layers'].append(layer_idx)
    
    return stats

def generate_saliency_for_prompt(tokenizer, model, prompt, problem_id, prompt_type, output_base_dir, original_question, step_size_override=None, dataset_name="gsm8k", model_type="gpt-oss", original_answer="", saliency_mode="full", options=None):
    """Generate saliency maps for a specific prompt with complete conversation saving

    Args:
        saliency_mode: 'answer_only' (fastest), 'data_only' (fast, can do step analysis), 'full' (complete with plots)
        options: For GPQA, list of option dicts [{"key": "A", "text": "..."}]
    """

    gc.collect()
    torch.cuda.empty_cache()

    # Step 1: Generate model response to get complete input-output sequence
    
    # Build user content: question + options (if GPQA) + "Let's think step by step"
    if dataset_name == 'gpqa' and options:
        options_text = "\n".join([f"{opt['key']}) {opt['text']}" for opt in options])
        user_content = f"{prompt}\n\nOptions:\n{options_text}\n\nLet's think step by step"
    else:
        user_content = f"{prompt}\n\nLet's think step by step"

    # Build prompt based on model type
    if model_type == "gpt-oss":
        reasoning_effort = "low"
    messages = [
        {
            "role": "user",
                "content": user_content
        }
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
            reasoning_effort=reasoning_effort
    )
    # Configure stop conditions for GPT-OSS
        eos_token_id, stop_ids, stopping_criteria = configure_stop_conditions(tokenizer)
        max_new_tokens = 1100
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=3072)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Ensure attention mask is properly set
    if 'attention_mask' not in inputs:
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

    # Generate response with the model (deterministic generation for reproducibility)
    with torch.no_grad():
        # Temporarily suppress transformers logging
        import logging
        transformers_logger = logging.getLogger("transformers")
        original_level = transformers_logger.level
        transformers_logger.setLevel(logging.ERROR)

        try:
            # Greedy decoding for reproducibility
            generated_outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=max_new_tokens,
                do_sample=False, 
                temperature=None,
                top_k=None,
                top_p=None,
                eos_token_id=eos_token_id if model_type == "gpt-oss" else stop_ids,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                return_dict_in_generate=True,
                # output_scores=True,  # Removed: saves ~850MB, not needed for saliency
                use_cache=True  # Enable KV cache for efficient O(N) generation
            )
        finally:
            # Restore original logging level
            transformers_logger.setLevel(original_level)

    # Decode the generated response
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = generated_outputs.sequences[0][input_length:]
    generated_response_clean = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Create complete sequence for saliency analysis
    complete_sequence_raw = tokenizer.decode(generated_outputs.sequences[0], skip_special_tokens=False)

    # Model-specific response extraction
    if model_type == "gpt-oss":
        # Extract complete assistant response from Harmony format
        assistant_start = complete_sequence_raw.find('<|start|>assistant')
        if assistant_start != -1:
            generated_response_raw = complete_sequence_raw[assistant_start:]
            return_pos = generated_response_raw.find('<|return|>')
            if return_pos != -1:
                generated_response_raw = generated_response_raw[:return_pos + len('<|return|>')]
        else:
            generated_response_raw = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        
        # Filter out system messages for cleaner saliency visualization
        complete_sequence = complete_sequence_raw
        
        # Validate Harmony format (silent check)
        format_valid, _ = validate_harmony_from_raw(complete_sequence_raw)
        
        # Parse Harmony sections for visualization
        harmony_sections = parse_harmony_sections(complete_sequence_raw, tokenizer)

    # Extract predicted answer (dataset-specific)
    if dataset_name == 'gpqa':
        predicted_answer = extract_choice_answer(generated_response_raw)
    else:
        predicted_answer = extract_numerical_answer(generated_response_raw)
    
    # Compute correctness immediately
    is_correct = compare_answers(predicted_answer, original_answer, dataset_name)
    
    # Calculate output tokens count
    output_tokens = len(generated_tokens)
    
    # Save conversation data with minimal fields
    conversation_data = {
        "problem_id": problem_id,
        "question": original_question,
        "answer": original_answer,
        "response": generated_response_raw,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "output_tokens": output_tokens
    }

    # Save conversation data
    conversation_dir = os.path.join(output_base_dir, "conversations")
    json_path = save_conversation_data(conversation_data, conversation_dir, problem_id, prompt_type)
    print(f"Saved conversation: {json_path} | Predicted: {predicted_answer} | Correct: {is_correct}")

    # Critical memory cleanup: Free generation outputs before saliency computation
    # This prevents OOM for long sequences (AIME24 with 1200+ tokens)
    # Safe to delete because we already extracted and saved all needed data
    del generated_outputs  # Frees ~900MB-1GB (includes scores, metadata, internal buffers)
    del generated_tokens   # Frees token tensor
    if 'inputs' in locals():
        del inputs  # Frees input tensors
    gc.collect()
    torch.cuda.empty_cache()
    print(f"✓ Freed generation memory ({output_tokens} output tokens, est. ~{output_tokens * 0.7:.0f}MB saved)")

    # If answer_only mode, skip saliency map generation
    if saliency_mode == 'answer_only':
        print(f"Answer-only mode: Skipping saliency map generation for problem {problem_id}")
        return conversation_data

    # Filter sequence for meaningful saliency analysis
    filtered_sequence = filter_sequence_for_saliency(complete_sequence, tokenizer)

    display_step_size = step_size_override or 4

    # Prepare inputs for saliency analysis
    complete_inputs = tokenizer(filtered_sequence, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    complete_inputs = {k: v.to(device) for k, v in complete_inputs.items()}

    print(f"Computing saliency for {complete_inputs['input_ids'].shape[1]} tokens (display step: {display_step_size})...")

    # Switch to train mode for gradient computation
    model.train()
    # Enable gradient checkpointing only for saliency extraction (saves memory during backprop)
    model.gradient_checkpointing_enable()

    try:
        # Use unified saliency extraction with model type
        saliency_result = extract_saliency(
            model=model,
            inputs=complete_inputs,
            model_type=model_type,
            rule="ag",
            return_attn_probs=False,  # Save memory; not needed for step-level analysis
            compute_saliency_metrics=(saliency_mode == 'full')  # Only compute in full mode
        )

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("Memory error, trying with shorter sequence...")
            # Fallback: use original formatted prompt only
            fallback_inputs = tokenizer(formatted_prompt, return_tensors="pt", max_length=512, truncation=True)
            complete_inputs = {k: v.to(device) for k, v in fallback_inputs.items()}

            gc.collect()
            torch.cuda.empty_cache()

            saliency_result = extract_saliency(
                model=model,
                inputs=complete_inputs,
                model_type=model_type,
                rule="ag",
                return_attn_probs=False,  # Save memory
                compute_saliency_metrics=(saliency_mode == 'full')
            )

            # Update conversation data to reflect fallback
            conversation_data["note"] = "Used shorter sequence due to memory constraints"
        else:
            raise e

    # Keep tensors as-is (no early CPU move) to match previous behavior

    # Use complete sequence tokens for visualization
    raw_tokens = tokenizer.convert_ids_to_tokens(complete_inputs['input_ids'][0])
    formatted_tokens = [f"{idx}-{clean_token(token)}" for idx, token in enumerate(raw_tokens)]
    
    print(f"Processed {len(saliency_result.saliency_per_layer)} layers successfully")
    
    # Calculate statistics only in full mode (expensive and not needed for step analysis)
    if saliency_mode == 'full':
    saliency_stats = calculate_saliency_statistics(saliency_result, formatted_tokens)
    else:
        # Minimal stats for data_only mode
        saliency_stats = {'layer_stats': [{'layer_idx': i} for i in range(len(saliency_result.saliency_per_layer))]}
    
    output_dir = os.path.join(output_base_dir, f"problem_{problem_id:02d}")
    os.makedirs(output_dir, exist_ok=True)
    
    saliency_data = {
        'prompt': prompt,
        'tokens': formatted_tokens,
        'raw_tokens': raw_tokens,
        'saliency_result': saliency_result,
        'statistics': saliency_stats,
        'problem_id': problem_id,
        'prompt_type': prompt_type
    }
    
    data_file = save_saliency_data(saliency_data, output_base_dir, problem_id, prompt_type)
    
    print(f"Generating {len(saliency_result.saliency_per_layer)} saliency heatmaps (step size: {display_step_size})...")

    # Skip visualization if data_only mode (but keep saliency data for step analysis)
    if saliency_mode == 'data_only':
        print(f"Data-only mode: Skipping visualization for problem {problem_id}")
        print(f"Saliency data saved: {data_file} (can be used for step-level analysis)")
    elif saliency_mode == 'full':
        # Full mode: generate all visualizations
    for layer_idx, (layer_info, saliency_matrix) in enumerate(zip(
        saliency_result.layers_info, saliency_result.saliency_per_layer
    )):
        if saliency_matrix.dtype == torch.bfloat16:
            saliency_matrix = saliency_matrix.float()

        avg_saliency = saliency_matrix.mean(dim=1)[0].cpu().numpy()

        # Apply display sampling for visualization (not computation sampling)
        if display_step_size > 1:
            # Sample both dimensions to maintain attention pattern structure while reducing display complexity
            sampled_indices = list(range(0, avg_saliency.shape[0], display_step_size))
            avg_saliency = avg_saliency[np.ix_(sampled_indices, sampled_indices)]
            # Sample tokens accordingly for display
            sampled_tokens = [formatted_tokens[i] for i in sampled_indices if i < len(formatted_tokens)]
        else:
            sampled_tokens = formatted_tokens

        seq_len = avg_saliency.shape[0]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        avg_saliency = avg_saliency * causal_mask

            # Determine layer type based on model architecture
            if model_type == "gpt-oss":
                # GPT-OSS: Even layers (0,2,4...) = sliding_attention (SW), Odd layers = full_attention (Full)
                layer_type = "SW" if layer_idx % 2 == 0 else "Full"
            else:
                # Default to Full for unknown models
                layer_type = "Full"
            
            filename = f"problem_{problem_id:02d}_layer_{layer_idx:02d}_{layer_type}_saliency.png"
        filepath = os.path.join(output_dir, filename)

            # Generate enhanced heatmap with Harmony channel annotations
        generate_enhanced_saliency_heatmap(
                avg_saliency, sampled_tokens, layer_idx, layer_type, filepath,
                prompt_type=prompt_type, problem_id=problem_id, step_size=display_step_size,
                format_valid=format_valid, harmony_sections=harmony_sections
            )
    
    print(f"Completed problem {problem_id} ({prompt_type})")
    
    # Comprehensive memory cleanup
    del saliency_result

    # Clear any remaining variables that might hold tensors
    if 'complete_inputs' in locals():
        del complete_inputs
    if 'generated_outputs' in locals():
        del generated_outputs

    # Force comprehensive cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    return output_dir, formatted_tokens, len(saliency_stats['layer_stats']), saliency_stats

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="GPT-OSS Saliency Analysis with Enhanced Harmony Chat Format Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/generate_saliency_maps.py --problem_id 1
  python src/generate_saliency_maps.py --problem_id 2 --prompt_type few_shot_cot
  python src/generate_saliency_maps.py --problem_id 1 --step_size 8
  python src/generate_saliency_maps.py --problem_id 3 --output_dir outputs/custom
        """
    )

    parser.add_argument(
        '--problems',
        type=str,
        default='1',
        help='Comma-separated list or range of problem IDs (e.g., "1,2,3" or "1-10", default: 1)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='Number of samples to generate per problem (default: 10)'
    )

    parser.add_argument(
        '--step_size',
        type=int,
        default=4,
        choices=[1,2,3,4,5,8],
        help='Display sampling step size for visualization (default: 4)'
    )
    
    parser.add_argument(
        '--batch_mode',
        action='store_true',
        help='Enable batch mode to analyze multiple problems'
    )
    
    parser.add_argument(
        '--saliency_mode',
        type=str,
        default='full',
        choices=['answer_only', 'data_only', 'full'],
        help='''Saliency generation mode:
  - answer_only: Generate answers only (fastest, ~30s/sample)
  - data_only: Generate answers + saliency data without plots (fast, ~70s/sample, can do step analysis)
  - full: Generate answers + saliency data + plots (slowest, ~90s/sample)
Default: full'''
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='math',
        choices=['math', 'aime24', 'gpqa'],
        help='Dataset to use: math (MATH-500), aime24 (AIME 2024), or gpqa (GPQA-diamond). Default: math'
    )
    
    parser.add_argument(
        '--parquet-path',
        type=str,
        default=None,
        help='Path to parquet file (for aime24 dataset). Default: data/datasets/aime24/test-00000-of-00001 (3).parquet'
    )

    parser.add_argument(
        '--gpqa-path',
        type=str,
        default=None,
        help='Path to GPQA CSV file (for gpqa dataset). Default: data/datasets/gpqa/gpqa_diamond.csv'
    )

    return parser.parse_args()

def main():
    """Main function to generate saliency maps"""
    args = parse_arguments()

    print("Starting GPT-OSS Attention Saliency Analysis")
    print("=" * 60)

    run_batch_analysis(args)

def parse_problem_ids(problems_str):
    """Parse problem IDs from string supporting both comma-separated and ranges
    
    Examples:
        "1,2,3" -> [1, 2, 3]
        "1-5" -> [1, 2, 3, 4, 5]
        "1,3,5-7" -> [1, 3, 5, 6, 7]
    """
    problem_ids = []
    for part in problems_str.split(','):
        part = part.strip()
        if '-' in part:
            # Range format: "501-520"
            start, end = part.split('-')
            problem_ids.extend(range(int(start), int(end) + 1))
        else:
            # Single number
            problem_ids.append(int(part))
    return problem_ids


def run_batch_analysis(args):
    """Run batch analysis for multiple problems with multi-sampling (supports MATH, AIME24, and GPQA)"""
    problems = parse_problem_ids(args.problems)
    prompt_type = 'zero_shot_cot'
    num_samples = args.num_samples
    dataset_name = args.dataset

    print(f"Batch Analysis Configuration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Problems: {problems}")
    print(f"  Prompt Types: ['zero_shot_cot']")
    print(f"  Step Size: {args.step_size}")
    print(f"  Num Samples: {num_samples}")
    if dataset_name == 'aime24' and args.parquet_path:
        print(f"  Parquet Path: {args.parquet_path}")
    if dataset_name == 'gpqa' and args.gpqa_path:
        print(f"  GPQA Path: {args.gpqa_path}")
    print()

    # Load model and dataset
    print("Loading tokenizer and model...")
    tokenizer, model, dataset = load_model_and_data(dataset_name, model_type='gpt-oss', parquet_path=args.parquet_path, gpqa_path=args.gpqa_path)

    # Calculate the maximum problem ID needed
    max_problem_id = max(problems)
    all_problems = get_sample_problems(dataset, num_problems=max_problem_id + 1, dataset_name=dataset_name)

    # Auto-generate timestamp-based output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"outputs/sampling_saliency/gpt-oss_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    total_analyses = len(problems)
    completed_analyses = 0
    failed_analyses = 0

    print(f"============================================================")
    print(f"Processing Zero Shot Cot Prompts")
    print(f"============================================================")
    
    for problem_id in problems:
        print(f"--- Problem {problem_id} ({prompt_type}) ---")

    # Find the specific problem
    target_problem = None
        for problem in all_problems:
            if problem['id'] == problem_id:
            target_problem = problem
            break

    if target_problem is None:
            print(f"Error: Problem {problem_id} not found!")
            failed_analyses += 1
            continue
            
        # Use simplified naming for problem directory
        if dataset_name == 'aime24':
            # For AIME24, use the problem_id from dataset (e.g., "aime24_0060")
            problem_prefix = target_problem.get('problem_id', f"aime24_{problem_id:04d}")
        elif dataset_name == 'gpqa':
            # For GPQA, use the problem_id from dataset (e.g., "gpqa_06pn")
            problem_prefix = target_problem.get('problem_id', f"gpqa_{problem_id:04d}")
    else:
            problem_prefix = f"math_{problem_id:04d}"
        
        problem_dir = f"{base_output_dir}/{problem_prefix}"
        os.makedirs(problem_dir, exist_ok=True)

    question = target_problem['question']
        answer = target_problem.get('answer', '')
        options = target_problem.get('options', None)  # For GPQA
        current_prompt = create_zero_shot_prompt(question)
        
        # Multi-sampling loop
        problem_success = 0
        for sample_idx in range(num_samples):
            sample_output_dir = f"{problem_dir}/sample_{sample_idx:02d}"
            os.makedirs(sample_output_dir, exist_ok=True)
            
            print(f"  Sample {sample_idx+1}/{num_samples}")
            
            # Reset model state before each sample
            model.zero_grad(set_to_none=True)
            model.eval()

            try:
                result = generate_saliency_for_prompt(
                    tokenizer, model, current_prompt, problem_id, prompt_type,
                    sample_output_dir, question, args.step_size, dataset_name, 'gpt-oss', answer,
                    saliency_mode=args.saliency_mode, options=options
                )
                problem_success += 1
            except Exception as e:
                print(f"  Error: Sample {sample_idx+1} failed: {e}")
            
            # Memory cleanup after each sample
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        
        # Aggregate samples and compute accuracy
        print(f"  Aggregating {num_samples} samples for problem {problem_id}...")
        try:
            metadata = aggregate_samples(
                problem_dir=problem_dir,
                num_samples=num_samples,
                ground_truth=answer,
                dataset_name=dataset_name
            )
            
            print(f"  Problem {problem_id} Accuracy: {metadata['accuracy']*100:.1f}% "
                  f"({metadata['num_correct']}/{metadata['num_samples']})")
            
            if problem_success == num_samples:
                completed_analyses += 1
            else:
                failed_analyses += 1
    except Exception as e:
            print(f"  Error aggregating samples: {e}")
            failed_analyses += 1

        print()

    # Calculate overall accuracy from aggregated metadata
    from pathlib import Path
    base_path = Path(base_output_dir)
    
    # Choose glob pattern based on dataset
    if dataset_name == 'aime24':
        glob_pattern = "aime24_*"
    elif dataset_name == 'gpqa':
        glob_pattern = "gpqa_*"
    else:
        glob_pattern = "math_*"
    
    problem_dirs = sorted(base_path.glob(glob_pattern))
    
    total_correct = 0
    total_samples = 0
    problem_accuracies = []
    
    for prob_dir in problem_dirs:
        metadata_file = prob_dir / "aggregated_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                meta = json.load(f)
            total_correct += meta['num_correct']
            total_samples += meta['num_samples']
            problem_accuracies.append(meta['accuracy'])
    
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_problem_accuracy = sum(problem_accuracies) / len(problem_accuracies) if problem_accuracies else 0.0
    
    # Print summary
    print("============================================================")
    print("BATCH ANALYSIS SUMMARY")
    print("============================================================")
    print(f"Total problems analyzed: {len(problem_dirs)}")
    print(f"Total samples generated: {total_samples}")
    print(f"Overall accuracy: {overall_accuracy*100:.1f}% ({total_correct}/{total_samples})")
    print(f"Average problem accuracy: {avg_problem_accuracy*100:.1f}%")
    print(f"Failed problems: {failed_analyses}")
    print(f"Output directory: {base_output_dir}")
    print("============================================================")

if __name__ == "__main__":
    main()
