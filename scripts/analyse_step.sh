#!/bin/bash
# Step-level saliency comparison analysis

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT" && python scripts/analyze_step_saliency.py \
    --correct-dir outputs/analysis_input/correct \
    --wrong-dir outputs/analysis_input/wrong \
    --model gpt-oss \
    "$@"
