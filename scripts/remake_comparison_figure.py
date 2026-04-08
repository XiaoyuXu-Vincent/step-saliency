#!/usr/bin/env python3
"""
Redraw sampling comparison figure.
- Horizontal layout (1 row, 2 columns)
- Minimal text
- Clear units
- Unified colors (blue = correct, orange = wrong)
- Removed statistically problematic ratio plot
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============= Config =============
plt.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.edgecolor": "#cccccc",
    "xtick.color": "#666666",
    "ytick.color": "#666666",
})

COLOR_CORRECT = "#5a9adf"
COLOR_WRONG = "#f5b56a"

# ============= Read data =============
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--input", type=str, default="outputs/comparison_analysis/comparison_analysis.json")
_parser.add_argument("--output-dir", type=str, default="outputs/comparison_analysis")
_args = _parser.parse_args()

data_path = Path(_args.input)
with open(data_path, "r") as f:
    data = json.load(f)

correct_data = data["avg_metrics"]["correct"]
wrong_data = data["avg_metrics"]["wrong"]

final_correct = np.array(correct_data["avg_final_intensity_by_layer"]) * 1e6
final_wrong = np.array(wrong_data["avg_final_intensity_by_layer"]) * 1e6

analysis_correct = np.array(correct_data["avg_analysis_intensity_by_layer"]) * 1e6
analysis_wrong = np.array(wrong_data["avg_analysis_intensity_by_layer"]) * 1e6

num_layers = len(analysis_correct)
layers = np.arange(num_layers)

# ============= Create figure =============
fig, axes = plt.subplots(2, 1, figsize=(5, 6), sharey=False)
plt.subplots_adjust(hspace=0.35)

# ============= Subplot 1: Thinking Block (left) =============
ax = axes[0]
ax.plot(layers, analysis_correct, linewidth=0.7, 
        color=COLOR_CORRECT, label="Correct", alpha=0.75)
ax.plot(layers, analysis_wrong, linewidth=0.7, 
        color=COLOR_WRONG, label="Wrong", alpha=0.75)

ax.set_title("Analysis block", fontsize=10, pad=8, fontweight='normal')
ax.set_xlabel("Layers", fontsize=9)
ax.set_ylabel("Saliency (×1e−6)", fontsize=9)
ax.set_xlim(-0.5, num_layers - 0.5)
max_analysis = max(analysis_correct.max(), analysis_wrong.max())
ax.set_ylim(0.0, max_analysis * 1.15)
ax.set_xticks(np.arange(0, num_layers, 4))
ax.legend(loc="upper right", frameon=False, fontsize=8)
ax.yaxis.get_offset_text().set_visible(False)
ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.3)

# ============= Subplot 2: Summary Block (right) =============
ax = axes[1]
ax.plot(layers, final_correct, linewidth=0.7, 
        color=COLOR_CORRECT, label="Correct", alpha=0.75)
ax.plot(layers, final_wrong, linewidth=0.7, 
        color=COLOR_WRONG, label="Wrong", alpha=0.75)

ax.set_title("Final block", fontsize=10, pad=8, fontweight='normal')
ax.set_xlabel("Layers", fontsize=9)
ax.set_ylabel("Saliency (×1e−6)", fontsize=9)
ax.set_xlim(-0.5, num_layers - 0.5)
max_final = max(final_correct.max(), final_wrong.max())
ax.set_ylim(0.0, max_final * 1.15)
ax.set_xticks(np.arange(0, num_layers, 4))
ax.legend(loc="upper right", frameon=False, fontsize=8)
ax.yaxis.get_offset_text().set_visible(False)
ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.3)

# ============= Save =============
out_dir = Path(_args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
output_path = out_dir / "sampling_comparison_gpt-oss_v2.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved to: {output_path}")

output_path_original = out_dir / "sampling_comparison_gpt-oss.png"
plt.savefig(output_path_original, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Overwritten: {output_path_original}")

plt.close()

print("\n=== Summary ===")
print(f"Correct samples: {data['metadata']['num_correct']}")
print(f"Wrong samples: {data['metadata']['num_wrong']}")
print(f"Layers: {num_layers}")

