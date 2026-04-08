#!/bin/bash
# LiveCodeBench evaluation with StepFlow

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export TRANSFORMERS_NO_TORCHVISION=1

cd "$PROJECT_ROOT" && python scripts/eval_livecodebench.py \
    --version-tag release_v1 \
    --test-mode all \
    --max-new-tokens 10000 \
    --execution-timeout 10 \
    --use-smi \
    --use-oeb \
    --oeb-layers 1,3,5,7 \
    --resume \
    "$@"
