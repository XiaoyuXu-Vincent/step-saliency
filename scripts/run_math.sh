#!/bin/bash
# Math evaluation for GPT-OSS with StepFlow
# Usage: bash scripts/run_math.sh [--data_names aime25]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-data/models/gpt-oss-20b}"
DATA_DIR="${DATA_DIR:-data/datasets}"

cd "$PROJECT_ROOT/eval/Math-main/evaluation" && \
HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0 \
TOKENIZERS_PARALLELISM=false \
python math_eval_gptoss.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_names aime24 \
    --data_dir "$DATA_DIR" \
    --output_dir outputs/gptoss_math_dev \
    --reasoning-effort medium \
    --max_tokens_per_call 30000 \
    --temperature 0 \
    --seed 1 \
    --save_outputs \
    --num_shots 0 \
    --use-smi \
    --smi-strength 0.06 \
    --use-oeb \
    --oeb-layers 1,3,5 \
    "$@"
