#!/bin/bash
# GPQA-Diamond evaluation with StepFlow (SMI + OEB)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export TRANSFORMERS_NO_TORCHVISION=1

cd "$PROJECT_ROOT" && python scripts/eval_gpqa_aqr.py \
    --max-tokens 12000 \
    --reasoning-effort medium \
    --use-smi \
    --smi-strength 0.06 \
    --use-oeb \
    --oeb-layers 1,3,5,7 \
    "$@"
