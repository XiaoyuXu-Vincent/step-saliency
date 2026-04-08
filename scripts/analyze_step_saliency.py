#!/usr/bin/env python3
"""
Step-level saliency analysis for GPT-OSS model.
Aggregates token-level saliency into step-level dependencies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pickle
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
import argparse

def extract_steps(response: str, question: str = '', model_type: str = 'gpt-oss') -> tuple:
    if model_type == 'gpt-oss':
        steps = []
        
        # Step 0: Question
        if question:
            steps.append(question)
        
        # Thinking steps (analysis channel)
        analysis_match = re.search(
            r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>', 
            response, re.DOTALL
        )
        thinking_count = 0
        if analysis_match:
            analysis_text = analysis_match.group(1).strip()
            thinking_steps = []
            
            # Split by sentence (period + space) to get finer granularity
            # This avoids splitting on decimal points like "3.14"
            sentences = re.split(r'\.\s+', analysis_text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                
                # Skip if too short (likely a number fragment)
                if len(sentence) < 15:
                    continue
                
                # Skip if it's just a number (like "5" from "3.5 kg")
                if sentence.replace('.', '').replace(',', '').replace('-', '').replace(' ', '').isdigit():
                    continue
                
                # Skip pure formatting
                if sentence.startswith('|') and sentence.count('|') > sentence.count('\n') * 2:
                    continue
                if sentence.startswith(('|--', '---')) and len(sentence) < 20:
                    continue
                
                # Add period back if missing (for consistency)
                if not sentence.endswith(('.', '!', '?', ':')):
                    sentence = sentence + '.'
                
                thinking_steps.append(sentence)
            
            steps.extend(thinking_steps)
            thinking_count = len(thinking_steps)
        
        # Summary steps (final channel)
        final_match = re.search(
            r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)<\|return\|>', 
            response, re.DOTALL
        )
        summary_count = 0
        if final_match:
            final_text = final_match.group(1).strip()
            summary_steps = []
            for step in final_text.split('\n\n'):
                step = step.strip()
                if len(step) < 10:
                    continue
                # Filter out pure markdown tables
                if step.startswith('|') and step.count('|') > step.count('\n') * 2:
                    continue
                # Filter out pure formatting separators
                if step.startswith(('|--', '---')) and len(step) < 20:
                    continue
                summary_steps.append(step)
            steps.extend(summary_steps)
            summary_count = len(summary_steps)
        
        boundaries = {
            'question': 1 if question else 0,
            'thinking': thinking_count,
            'summary': summary_count
        }
        
        return steps, boundaries
    
    # For other models
    return [], {'question': 0, 'thinking': 0, 'summary': 0}

def compute_step_saliency(saliency_map: np.ndarray, token_to_step: List[int], num_steps: int) -> np.ndarray:
    """Compute step x step saliency matrix"""
    T = saliency_map.shape[0]
    step_matrix = np.zeros((num_steps, num_steps))
    
    for query_step in range(num_steps):
        for key_step in range(num_steps):
            query_tokens = [i for i in range(T) if token_to_step[i] == query_step]
            key_tokens = [i for i in range(T) if token_to_step[i] == key_step]
            
            if query_tokens and key_tokens:
                sub_sal = saliency_map[np.ix_(query_tokens, key_tokens)]
                step_matrix[query_step, key_step] = sub_sal.mean()
    
    return step_matrix

def identify_key_computational_steps(steps: List[str]) -> List[int]:
    """Identify steps containing actual computations (not just any numbers)"""
    key_indices = []
    for i, step in enumerate(steps):
        # Skip Question (always step 0)
        if i == 0:
            continue
        
        # Pattern 1: Math equation with equals sign (e.g., "12/2 = 6", "3 × 4 = 12")
        if re.search(r'[=]', step) and re.search(r'\d+', step):
            # Exclude table headers/separators
            if not re.search(r'\|\s*-+\s*\|', step):  # Not a table separator
                key_indices.append(i)
                continue
        
        # Pattern 2: Fraction notation (\frac{...}{...})
        if re.search(r'\\frac\{.*?\d+.*?\}\{.*?\d+.*?\}', step):
            key_indices.append(i)
            continue
        
        # Pattern 3: Arithmetic in LaTeX (\[ ... \])
        if re.search(r'\\\[.*?\d+.*?[+\-×÷*/].*?\d+.*?\\\]', step, re.DOTALL):
            key_indices.append(i)
            continue
    
    return key_indices

def compute_step_metrics(step_matrix: np.ndarray, steps: List[str], boundaries: Dict = None) -> Dict:

    num_steps = len(steps)
    
    # ============================================================================
    # Row-wise normalization: convert to attention distribution
    # ============================================================================
    row_sums = step_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    step_matrix_norm = step_matrix / row_sums
    # Now: step_matrix_norm[i,j] = percentage of step i's attention to step j
    # Range: [0, 1], sum(row) = 1
    
    metrics = {
        'num_steps': num_steps,
        'final_heat_anomaly': 0.0,  # Ratio, normal ≈ 1
        'final_block_intensity': 0.0,  # Final region internal attention
        'analysis_block_intensity': 0.0,  # Analysis internal step dependency
    }
    
    if num_steps <= 1:
        return metrics
    
    # Parse boundaries
    if boundaries is None:
        boundaries = {'question': 1, 'thinking': 0, 'summary': 0}
    
    question_count = boundaries.get('question', 1)
    thinking_count = boundaries.get('thinking', 0)
    summary_count = boundaries.get('summary', 0)
    
    # Step indices
    question_end = question_count
    thinking_start = question_count
    thinking_end = question_count + thinking_count
    summary_start = question_count + thinking_count
    final_idx = num_steps - 1
    
    # ============================================================================
    # Final block heat anomaly
    # ============================================================================
    # Compare Final block intensity vs Analysis block intensity
    
    # Final block: Summary region attending to Summary region (lower-right corner)
    if summary_start < num_steps:
        final_block = step_matrix[summary_start:, summary_start:]
        final_block_intensity = float(np.mean(final_block)) if final_block.size > 0 else 0.0
    else:
        final_block_intensity = 0.0
    
    # Analysis block: Thinking region attending to Thinking region (middle diagonal)
    if thinking_count > 0 and thinking_end > thinking_start:
        analysis_block = step_matrix[thinking_start:thinking_end, thinking_start:thinking_end]
        analysis_block_intensity = float(np.mean(analysis_block)) if analysis_block.size > 0 else 0.0
    else:
        analysis_block_intensity = 0.0
    
    # Store both block intensities as independent metrics
    metrics['final_block_intensity'] = final_block_intensity
    metrics['analysis_block_intensity'] = analysis_block_intensity
    
    # Calculate anomaly ratio
    if analysis_block_intensity > 1e-10:  # Avoid division by zero
        metrics['final_heat_anomaly'] = float(final_block_intensity / analysis_block_intensity)
    else:
        metrics['final_heat_anomaly'] = 0.0
    
    return metrics

def plot_step_heatmap(step_matrix: np.ndarray, steps: List[str], save_path: str, title: str = "", boundaries: dict = None, 
                      global_vmin: float = None, global_vmax: float = None, threshold_percentile: float = 12.5):
    """Generate step saliency heatmap with Q/T/S region annotations
    
    Args:
        step_matrix: Step-level saliency matrix
        steps: List of step texts
        save_path: Output file path
        title: Optional plot title
        boundaries: Dict with 'question', 'thinking', 'summary' counts for region annotation
        global_vmin: Global minimum for unified color scale (log scale)
        global_vmax: Global maximum for unified color scale (log scale)
        threshold_percentile: Percentile threshold (10-15) for background masking
    """
    plt.figure(figsize=(14, 12))
    
    # Get valid (non-zero) values for color scale calculation
    valid_nonzero = step_matrix[step_matrix > 0]
    
    if len(valid_nonzero) > 0:
        # Apply threshold: mask values below threshold_percentile as background
        threshold_value = np.percentile(valid_nonzero, threshold_percentile)
        mask_matrix = step_matrix < threshold_value  # Boolean mask for background
        
        # Use global vmin/vmax if provided (for unified color scale), otherwise use local
        if global_vmin is not None and global_vmax is not None:
            vmin = global_vmin
            vmax = global_vmax
        else:
            # Local scaling as fallback
        vmin = 0.0000009   # 5th percentile for better contrast
        vmax = np.percentile(valid_nonzero, 95)   # 95th percentile
        
        # Use masked matrix for visualization
        # mask=True values will be shown as white background (removes "fog")
        sns.heatmap(
            step_matrix,
            xticklabels=[f"S{i+1}" for i in range(len(steps))],
            yticklabels=[f"S{i+1}" for i in range(len(steps))],
            cmap='Reds',  # Consistent red colormap
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cbar_kws={'label': 'Saliency Score (log scale)'},
            square=True,
            linewidths=0.02,
            cbar=True,
            mask=mask_matrix  # Mask low values as white background
        )
    else:
        # Fallback: linear scale for edge cases
        sns.heatmap(
            step_matrix + 1e-10,  # Avoid zeros
            xticklabels=[f"S{i+1}" for i in range(len(steps))],
            yticklabels=[f"S{i+1}" for i in range(len(steps))],
            cmap='Reds',
            cbar_kws={'label': 'Saliency Score'},
            square=True,
            linewidths=0.02,
            cbar=True
        )
    
    # Add Q/T/S boundary lines and annotations
    if boundaries:
        q_end = boundaries.get('question', 0)
        t_count = boundaries.get('thinking', 0)
        s_count = boundaries.get('summary', 0)
        t_end = q_end + t_count
        s_end = t_end + s_count
        
        ax = plt.gca()
        
        # Draw boundary lines
        if q_end > 0:
            ax.axhline(y=q_end, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
            ax.axvline(x=q_end, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
        
        if t_count > 0:
            ax.axhline(y=t_end, color='orange', linestyle='--', linewidth=2.5, alpha=0.8)
            ax.axvline(x=t_end, color='orange', linestyle='--', linewidth=2.5, alpha=0.8)
        
        # Add region labels on top (X-axis)
        y_pos = -1.5
        if q_end > 0:
            ax.text(q_end/2, y_pos, 'Q', ha='center', va='top', 
                   fontsize=16, fontweight='bold', color='red')
        if t_count > 0:
            ax.text((q_end + t_end)/2, y_pos, 'Thinking', ha='center', va='top',
                   fontsize=16, fontweight='bold', color='orange')
        if s_count > 0:
            ax.text((t_end + s_end)/2, y_pos, 'Summary', ha='center', va='top',
                   fontsize=16, fontweight='bold', color='green')
        
        # Add region labels on left (Y-axis)
        x_pos = -1.5
        if q_end > 0:
            ax.text(x_pos, q_end/2, 'Q', ha='right', va='center',
                   fontsize=16, fontweight='bold', color='red', rotation=90)
        if t_count > 0:
            ax.text(x_pos, (q_end + t_end)/2, 'Thinking', ha='right', va='center',
                   fontsize=16, fontweight='bold', color='orange', rotation=90)
        if s_count > 0:
            ax.text(x_pos, (t_end + s_end)/2, 'Summary', ha='right', va='center',
                   fontsize=16, fontweight='bold', color='green', rotation=90)
    
    plt.xlabel('Key Step', fontsize=12)
    plt.ylabel('Query Step', fontsize=12)
    if title:
        plt.title(title, fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def aggregate_step_saliency(steps: List[str], sal_data: dict, tokenizer, boundaries: dict = None) -> tuple:
  
    import torch
    
    tokens = sal_data.get('tokens', [])
    sr = sal_data.get('saliency_result')
    
    if not tokens or not sr or not sr.saliency_per_layer:
        return np.zeros((len(steps), len(steps))), {}, boundaries
    
    # Map tokens to steps (do this once)
    token_to_step = []
    for step_idx, step_text in enumerate(steps):
        step_tokens = tokenizer.encode(step_text, add_special_tokens=False)
        for _ in range(len(step_tokens)):
            if len(token_to_step) < len(tokens):
                token_to_step.append(step_idx)
    
    while len(token_to_step) < len(tokens):
        token_to_step.append(len(steps) - 1)
    token_to_step = token_to_step[:len(tokens)]
    
    # Compute step_matrix for each layer separately
    num_steps = len(steps)
    layer_step_matrices = []
    per_layer_anomaly = []
    per_layer_final_intensity = []
    per_layer_analysis_intensity = []
    
    for layer_idx, layer_tensor in enumerate(sr.saliency_per_layer):
        # Convert to numpy: (1, 64, seq_len, seq_len) -> (seq_len, seq_len)
        if isinstance(layer_tensor, torch.Tensor):
            layer_sal = layer_tensor.float().cpu().numpy()
        else:
            layer_sal = layer_tensor
        
        layer_sal_2d = np.mean(layer_sal, axis=(0, 1))  # Squeeze batch and head dims
        
        # Compute step_matrix for this layer
        step_matrix = compute_step_saliency(layer_sal_2d, token_to_step, num_steps)
        layer_step_matrices.append(step_matrix)
        
        # Compute metrics for this layer
        metrics = compute_step_metrics(step_matrix, steps, boundaries)
        per_layer_anomaly.append(metrics['final_heat_anomaly'])
        per_layer_final_intensity.append(metrics['final_block_intensity'])
        per_layer_analysis_intensity.append(metrics['analysis_block_intensity'])
    
    # Average the 24 step matrices (not averaging tensors)
    avg_step_matrix = np.mean(layer_step_matrices, axis=0)
    
    # Return data for per-layer analysis
    per_layer_data = {
        'layer_matrices': layer_step_matrices,
        'avg_matrix': avg_step_matrix,
        'anomaly_by_layer': per_layer_anomaly,
        'final_intensity_by_layer': per_layer_final_intensity,
        'analysis_intensity_by_layer': per_layer_analysis_intensity
    }
    
    return avg_step_matrix, per_layer_data, boundaries

def analyze_single_sample(sample_dir: Path, model_type: str, tokenizer, output_dir: Path):
   
    # Load conversation
    conv_files = list((sample_dir / 'conversations').glob('*.json'))
    if not conv_files:
        print(f"  Warning: No conversation file in {sample_dir}")
        return None
    
    with open(conv_files[0]) as f:
        conv_data = json.load(f)
        response = conv_data.get('full_response', '') or conv_data.get('response', '')
    
    # Load question (from conversation or parent directory)
    question = conv_data.get('question', '')
    if not question:
        # Try fetching from parent aggregated_metadata
        agg_meta = sample_dir.parent / 'aggregated_metadata.json'
        if agg_meta.exists():
            with open(agg_meta) as f:
                meta_data = json.load(f)
                question = meta_data.get('question', '')
    
    steps, boundaries = extract_steps(response, question, model_type)
    if not steps or len(steps) == 1:  # Only question, no thinking steps
        print(f"  Warning: No thinking steps in {sample_dir}")
        return None
    
    sal_files = list((sample_dir / 'saliency_data').glob('*_saliency.pkl'))
    if not sal_files:
        print(f"  Warning: No saliency data in {sample_dir}")
        return None
    
    with open(sal_files[0], 'rb') as f:
        sal_data = pickle.load(f, map_location='cpu') if hasattr(pickle, 'map_location') else pickle.load(f)
    
    # Call existing function to get per-layer data
    _, per_layer_data, _ = aggregate_step_saliency(steps, sal_data, tokenizer, boundaries)
    
    # Generate heatmap for each layer
    layer_matrices = per_layer_data['layer_matrices']  # List of 24 matrices
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # Unified color scale: compute global vmin/vmax across all layers (log colorbar)
    # ============================================================================
    all_valid_values = []
    for layer_matrix in layer_matrices:
        valid_nonzero = layer_matrix[layer_matrix > 0]
        if len(valid_nonzero) > 0:
            all_valid_values.extend(valid_nonzero.flatten())
    
    if len(all_valid_values) > 0:
        global_vmin = 0.0000009
        global_vmax = np.percentile(all_valid_values, 95)
        print(f"  Global color scale: vmin={global_vmin:.3e}, vmax={global_vmax:.3e}")
    else:
        global_vmin = None
        global_vmax = None
    
    print(f"  Generating 24 layer heatmaps for {sample_dir.name}...")
    for layer_idx, layer_matrix in enumerate(layer_matrices):
        output_path = output_dir / f"layer_{layer_idx:02d}_step_heatmap.png"
        plot_step_heatmap(
            layer_matrix, 
            steps, 
            str(output_path),
            title=f"Layer {layer_idx} Step Saliency",
            boundaries=boundaries,
            global_vmin=global_vmin,
            global_vmax=global_vmax,
            threshold_percentile=12.5
        )
   
    return {
        'anomaly_by_layer': per_layer_data['anomaly_by_layer'],
        'final_intensity_by_layer': per_layer_data['final_intensity_by_layer'],
        'analysis_intensity_by_layer': per_layer_data['analysis_intensity_by_layer'],
        'problem_id': sample_dir.parent.name,
        'sample_id': sample_dir.name
    }

def analyze_correct_vs_wrong(correct_dir: Path, wrong_dir: Path, model_type: str, output_dir: Path):

    tokenizer = AutoTokenizer.from_pretrained(args.model_path if hasattr(args, 'model_path') else 'data/models/gpt-oss-20b')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Correct vs Wrong Sample Comparison")
    print("="*60)
    
 
    correct_metrics = []
    print("\nProcessing correct samples:")
    for problem_dir in sorted(correct_dir.glob('*_*')):

        sample_dirs = sorted(problem_dir.glob('sample_*'))
        if sample_dirs:
            sample_dir = sample_dirs[0]
            print(f"  Processing {problem_dir.name}/{sample_dir.name}")
            sample_output = output_dir / 'correct' / problem_dir.name
            metrics = analyze_single_sample(sample_dir, model_type, tokenizer, sample_output)
            if metrics:
                correct_metrics.append(metrics)
    
 
    wrong_metrics = []
    print("\nProcessing wrong samples:")
    for problem_dir in sorted(wrong_dir.glob('*_*')):
        # Find the first sample directory
        sample_dirs = sorted(problem_dir.glob('sample_*'))
        if sample_dirs:
            sample_dir = sample_dirs[0]
            print(f"  Processing {problem_dir.name}/{sample_dir.name}")
            sample_output = output_dir / 'wrong' / problem_dir.name
            metrics = analyze_single_sample(sample_dir, model_type, tokenizer, sample_output)
            if metrics:
                wrong_metrics.append(metrics)
    
    print(f"\nValid correct samples: {len(correct_metrics)}")
    print(f"Valid wrong samples: {len(wrong_metrics)}")
    
    if not correct_metrics or not wrong_metrics:
        print("Warning: No correct or wrong samples found, cannot generate comparison plot")
        return
    

    correct_anomaly = np.array([m['anomaly_by_layer'] for m in correct_metrics])  # (N, 24)
    wrong_anomaly = np.array([m['anomaly_by_layer'] for m in wrong_metrics])
    correct_final_intensity = np.array([m['final_intensity_by_layer'] for m in correct_metrics])
    wrong_final_intensity = np.array([m['final_intensity_by_layer'] for m in wrong_metrics])
    correct_analysis_intensity = np.array([m['analysis_intensity_by_layer'] for m in correct_metrics])
    wrong_analysis_intensity = np.array([m['analysis_intensity_by_layer'] for m in wrong_metrics])
    

    def prepare_comparison_data(anomaly_array, final_intensity_array, analysis_intensity_array):

        avg_anomaly = np.mean(anomaly_array, axis=0)  # (24,)
        avg_final_intensity = np.mean(final_intensity_array, axis=0)  # (24,)
        avg_analysis_intensity = np.mean(analysis_intensity_array, axis=0)  # (24,)
        
        layers_data = []
        for layer_idx in range(24):
            layers_data.append({
                'final_heat_anomaly': avg_anomaly[layer_idx],
                'final_block_intensity': avg_final_intensity[layer_idx],
                'analysis_block_intensity': avg_analysis_intensity[layer_idx]
            })
        return layers_data
    
    
    correct_layers = prepare_comparison_data(correct_anomaly, correct_final_intensity, correct_analysis_intensity)
    wrong_layers = prepare_comparison_data(wrong_anomaly, wrong_final_intensity, wrong_analysis_intensity)
    
    plot_sampling_comparison(correct_layers, wrong_layers, output_dir, model_type,
                            high_label='Correct', low_label='Wrong')
    
    # Calculate statistics for Final Block Intensity
    from scipy import stats
    
    correct_final_mean = np.mean(correct_final_intensity)
    correct_final_std = np.std(correct_final_intensity)
    wrong_final_mean = np.mean(wrong_final_intensity)
    wrong_final_std = np.std(wrong_final_intensity)
    
    t_stat_final, p_value_final = stats.ttest_ind(
        correct_final_intensity.flatten(),
        wrong_final_intensity.flatten()
    )
    
    print("\n" + "="*60)
    print("Final Block Intensity (Summary region internal attention):")
    print("="*60)
    print(f"  Correct samples: {correct_final_mean:.6e} ± {correct_final_std:.6e}")
    print(f"  Wrong samples:   {wrong_final_mean:.6e} ± {wrong_final_std:.6e}")
    print(f"  Difference:      {wrong_final_mean - correct_final_mean:.6e} ({((wrong_final_mean - correct_final_mean) / correct_final_mean * 100):.1f}%)")
    print(f"  T-statistic:     {t_stat_final:.2f} (p={p_value_final:.4f})")
    if p_value_final < 0.05:
        print(f"  ✓ Statistically significant difference (p < 0.05)")
    else:
        print(f"  ✗ No significant difference (p >= 0.05)")
    print("="*60)
    
    # Calculate statistics for Analysis Block Intensity
    correct_analysis_mean = np.mean(correct_analysis_intensity)
    correct_analysis_std = np.std(correct_analysis_intensity)
    wrong_analysis_mean = np.mean(wrong_analysis_intensity)
    wrong_analysis_std = np.std(wrong_analysis_intensity)
    
    # T-test for Analysis Block Intensity
    t_stat, p_value = stats.ttest_ind(
        correct_analysis_intensity.flatten(),
        wrong_analysis_intensity.flatten()
    )
    
    print("\n" + "="*60)
    print("Analysis Block Intensity (Thinking region internal dependency):")
    print("="*60)
    print(f"  Correct samples: {correct_analysis_mean:.6f} ± {correct_analysis_std:.6f}")
    print(f"  Wrong samples:   {wrong_analysis_mean:.6f} ± {wrong_analysis_std:.6f}")
    print(f"  Difference:      {wrong_analysis_mean - correct_analysis_mean:.6f} ({((wrong_analysis_mean - correct_analysis_mean) / correct_analysis_mean * 100):.1f}%)")
    print(f"  T-statistic:     {t_stat:.2f} (p={p_value:.4f})")
    if p_value < 0.05:
        print(f"  ✓ Statistically significant difference (p < 0.05)")
    else:
        print(f"  ✗ No significant difference (p >= 0.05)")
    print("="*60 + "\n")
    
    analysis_results = {
        'metadata': {
            'num_correct': len(correct_metrics),
            'num_wrong': len(wrong_metrics),
            'model_type': model_type
        },
        'correct_samples': [m['problem_id'] + '/' + m['sample_id'] for m in correct_metrics],
        'wrong_samples': [m['problem_id'] + '/' + m['sample_id'] for m in wrong_metrics],
        'avg_metrics': {
            'correct': {
                'avg_anomaly_by_layer': np.mean(correct_anomaly, axis=0).tolist(),
                'avg_final_intensity_by_layer': np.mean(correct_final_intensity, axis=0).tolist(),
                'avg_analysis_intensity_by_layer': np.mean(correct_analysis_intensity, axis=0).tolist()
            },
            'wrong': {
                'avg_anomaly_by_layer': np.mean(wrong_anomaly, axis=0).tolist(),
                'avg_final_intensity_by_layer': np.mean(wrong_final_intensity, axis=0).tolist(),
                'avg_analysis_intensity_by_layer': np.mean(wrong_analysis_intensity, axis=0).tolist()
            }
        },
        'statistics': {
            'final_block_intensity': {
                'correct_mean': float(correct_final_mean),
                'correct_std': float(correct_final_std),
                'wrong_mean': float(wrong_final_mean),
                'wrong_std': float(wrong_final_std),
                't_statistic': float(t_stat_final),
                'p_value': float(p_value_final)
            },
            'analysis_block_intensity': {
                'correct_mean': float(correct_analysis_mean),
                'correct_std': float(correct_analysis_std),
                'wrong_mean': float(wrong_analysis_mean),
                'wrong_std': float(wrong_analysis_std),
                't_statistic': float(t_stat),
                'p_value': float(p_value)
            }
        }
    }
    
    with open(output_dir / 'comparison_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Comparison plot saved: {output_dir / f'sampling_comparison_{model_type}.png'}")
    print(f"✓ Analysis results saved: {output_dir / 'comparison_analysis.json'}")


def plot_sampling_comparison(high_acc_layers, low_acc_layers, output_dir, model_name, 
                             high_label='High Acc (≥70%)', low_label='Low Acc (≤30%)'):
    """Plot three metrics: Final Heat Anomaly, Final Block Intensity, and Analysis Block Intensity
    
    Args:
        high_acc_layers: High accuracy group layer data
        low_acc_layers: Low accuracy group layer data
        output_dir: Output directory
        model_name: Model name for title
        high_label: Label for high accuracy group
        low_label: Label for low accuracy group
    """
    num_layers = len(high_acc_layers)
    layers = list(range(num_layers))
    
    high_anomaly = [l['final_heat_anomaly'] for l in high_acc_layers]
    low_anomaly = [l['final_heat_anomaly'] for l in low_acc_layers]
    high_final = [l['final_block_intensity'] for l in high_acc_layers]
    low_final = [l['final_block_intensity'] for l in low_acc_layers]
    high_analysis = [l['analysis_block_intensity'] for l in high_acc_layers]
    low_analysis = [l['analysis_block_intensity'] for l in low_acc_layers]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16))
    
    # Plot 1: Final Heat Anomaly comparison
    ax1.plot(layers, high_anomaly, 'o-', label=high_label, color='#2ecc71', linewidth=2.5, markersize=8)
    ax1.plot(layers, low_anomaly, 's-', label=low_label, color='#e74c3c', linewidth=2.5, markersize=8)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Normal (ratio=1)')
    ax1.set_ylabel('Final Heat Anomaly (ratio)', fontsize=13, fontweight='bold')
    ax1.set_title('Final Heat Anomaly: Final/Analysis Ratio', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final Block Intensity comparison
    ax2.plot(layers, high_final, 'o-', label=high_label, color='#2ecc71', linewidth=2.5, markersize=8)
    ax2.plot(layers, low_final, 's-', label=low_label, color='#e74c3c', linewidth=2.5, markersize=8)
    ax2.set_ylabel('Final Block Intensity', fontsize=13, fontweight='bold')
    ax2.set_title('Final Block Intensity: Summary Region Internal Attention', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 3: Analysis Block Intensity comparison
    ax3.plot(layers, high_analysis, 'o-', label=high_label, color='#2ecc71', linewidth=2.5, markersize=8)
    ax3.plot(layers, low_analysis, 's-', label=low_label, color='#e74c3c', linewidth=2.5, markersize=8)
    ax3.set_xlabel('Layer Index', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Analysis Block Intensity', fontsize=13, fontweight='bold')
    ax3.set_title('Analysis Block Intensity: Thinking Region Internal Attention', fontsize=14, fontweight='bold', pad=15)
    ax3.legend(fontsize=11, loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    output_file = output_dir / f'sampling_comparison_{model_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Step-level saliency analysis - Compare correct vs wrong samples',
        epilog='Example: python analyze_step_saliency.py --correct-dir outputs/analysis_input/correct --wrong-dir outputs/analysis_input/wrong'
    )
    parser.add_argument('--model', default='gpt-oss', choices=['gpt-oss'],
                       help='Model to analyze (default: gpt-oss)')
    parser.add_argument('--model-path', type=str, default='data/models/gpt-oss-20b',
                       help='Path to model/tokenizer')
    parser.add_argument('--correct-dir', type=str, required=True,
                       help='Directory path for correct samples')
    parser.add_argument('--wrong-dir', type=str, required=True,
                       help='Directory path for wrong samples')
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer_path = args.model_path
    
    # Validate paths
    correct_path = Path(args.correct_dir)
    wrong_path = Path(args.wrong_dir)
    
    if not correct_path.exists():
        print(f"Error: Correct sample directory not found: {correct_path}")
        exit(1)
    if not wrong_path.exists():
        print(f"Error: Wrong sample directory not found: {wrong_path}")
        exit(1)
    
    # Run comparison analysis
    output_path = Path('outputs/comparison_analysis')
    analyze_correct_vs_wrong(correct_path, wrong_path, args.model, output_path)

