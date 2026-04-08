# Step-Saliency & StepFlow

Official code for **"Reasoning Fails Where Step Flow Breaks"** (ACL 2026).

Step-Saliency is a step-level diagnostic that aggregates token-level attention-gradient saliency into step-to-step maps, revealing two depth-wise information-flow failure patterns in large reasoning models (LRMs): **Shallow Lock-in** and **Deep Decay**. StepFlow is a lightweight test-time intervention with two components -- **Odds-Equal Bridge (OEB)** for shallow layers and **Step Momentum Injection (SMI)** for deep layers -- that repairs these failure patterns and consistently improves accuracy without retraining.

## Project Structure

```
saliency/
  src/
    generate_saliency_maps.py   # Generate step-level saliency maps
    saliency_extractor.py       # Universal attention saliency extractor
    model_config.py             # Model configurations (GPT-OSS, DeepSeek, QwQ)
    interventions/
      attention_manager.py      # Attention hook manager for interventions
      bridge_guard_oeb.py       # Odds-Equal Bridge (OEB) implementation
      smi.py                    # Step Momentum Injection (SMI) implementation
      state_controller.py       # State tracking for channel segments
  scripts/
    eval_gpqa_aqr.py            # GPQA-Diamond evaluation
    eval_livecodebench.py       # LiveCodeBench evaluation
    analyze_step_saliency.py    # Step saliency analysis
    run_gpqa.sh                 # Run GPQA evaluation
    run_math.sh                 # Run MATH evaluation
    run_livecodebench.sh        # Run LiveCodeBench evaluation
  eval/
    Math-main/                  # MATH benchmark evaluation
```

## Installation

```bash
pip install -r requirements.txt
```

GPT-OSS model support requires a custom `transformers` build that includes `transformers.models.gpt_oss`. DeepSeek-R1-Distill and QwQ models work with the standard `transformers` package.

## Usage

### Generate Saliency Maps

```bash
python src/generate_saliency_maps.py \
    --model-path /path/to/model \
    --dataset math \
    --output-dir outputs/saliency
```

### Run StepFlow Evaluations

GPQA-Diamond:

```bash
bash scripts/run_gpqa.sh --model-path /path/to/model
```

MATH:

```bash
MODEL_PATH=/path/to/model bash scripts/run_math.sh
```

LiveCodeBench:

```bash
bash scripts/run_livecodebench.sh --model-path /path/to/model
```

### Hyperparameters

Default StepFlow configuration (Table 8 in paper):

| Model | OEB layers | SMI layers | tau_max | alpha |
|-------|-----------|-----------|---------|-------|
| R1-Distill-7B/14B/32B | bottom 1/4 | top 1/4 | 0.15 | 0.06 |
| GPT-OSS-20B | bottom 1/4 | top 1/4 | 0.15 | 0.06 |
| QwQ-32B | bottom 1/4 | top 1/4 | 0.15 | 0.06 |

Override via CLI:

```bash
--smi-strength 0.06       # SMI residual scale alpha
--oeb-layers 1,3,5,7     # Specific OEB layers
--tau-max 0.15            # OEB bridge mass upper bound
```

## Citation

```bibtex
@inproceedings{xu2026reasoning,
  title={Reasoning Fails Where Step Flow Breaks},
  author={Xu, Xiaoyu and Pan, Yulan and Yuan, Xiaosong and Shen, Zhihong and Su, Minghao and Su, Yuanhao and Zhang, Xiaofeng},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE).
