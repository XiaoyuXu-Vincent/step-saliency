from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass(frozen=True)
class ModelConfig:
    """Lightweight description of model-specific formatting conventions."""

    key: str
    analysis_start_marker: str
    final_start_marker: str
    eos_tokens: Sequence[str]
    supports_reasoning_effort: bool
    allow_system_prompt: bool
    default_temperature: float
    default_oeb_max_layer: int
    default_smi_min_layer: int
    num_layers: Optional[int] = None


_MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "gpt-oss": ModelConfig(
        key="gpt-oss",
        analysis_start_marker="<|channel|>analysis<|message|>",
        final_start_marker="<|channel|>final<|message|>",
        eos_tokens=("<|return|>", "<|end|>", "<|endoftext|>"),
        supports_reasoning_effort=True,
        allow_system_prompt=True,
        default_temperature=0.0,
        default_oeb_max_layer=15,
        default_smi_min_layer=48,
        num_layers=64,
    ),
    "deepseek-qwen": ModelConfig(
        key="deepseek-qwen",
        analysis_start_marker="<think>",
        final_start_marker="</think>",
        eos_tokens=("<｜end▁of▁sentence｜>",),
        supports_reasoning_effort=False,
        allow_system_prompt=False,
        default_temperature=0.6,
        default_oeb_max_layer=6,
        default_smi_min_layer=21,
        num_layers=28,
    ),
}

_DEFAULT_MODEL_TYPE = "gpt-oss"


def available_model_types() -> Sequence[str]:
    return tuple(_MODEL_CONFIGS.keys())


def get_model_config(model_type: str) -> ModelConfig:
    try:
        return _MODEL_CONFIGS[model_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Available: {', '.join(available_model_types())}"
        ) from exc


def detect_model_type(model_path: Union[str, Path]) -> str:
    """Infer model type from HuggingFace config metadata."""

    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return _DEFAULT_MODEL_TYPE

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
    except Exception:
        return _DEFAULT_MODEL_TYPE

    model_type = str(config.get("model_type", "")).lower()
    architectures = [str(a).lower() for a in config.get("architectures", [])]

    if "qwen2" in model_type or any("qwen" in arch for arch in architectures):
        return "deepseek-qwen"
    if "gpt_oss" in model_type or any("gptoss" in arch for arch in architectures):
        return "gpt-oss"
    return _DEFAULT_MODEL_TYPE


def resolve_model_config(
    *,
    model_path: Optional[Union[str, Path]] = None,
    explicit_type: Optional[str] = None,
) -> ModelConfig:
    """Return a concrete model configuration.

    Args:
        model_path: Directory containing `config.json` for auto-detection.
        explicit_type: Optional override supplied via CLI/config.
    """

    if explicit_type:
        return get_model_config(explicit_type)
    if model_path is None:
        return get_model_config(_DEFAULT_MODEL_TYPE)
    detected = detect_model_type(model_path)
    return get_model_config(detected)


def collect_eos_token_ids(tokenizer: Any, model_config: ModelConfig) -> List[int]:
    """Convert configured EOS markers into tokenizer-specific IDs."""

    token_ids: List[int] = []
    for marker in model_config.eos_tokens:
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(token_id, list):
            token_ids.extend(int(t) for t in token_id if isinstance(t, int) and t >= 0)
        elif isinstance(token_id, int) and token_id >= 0:
            token_ids.append(token_id)

    if not token_ids and tokenizer.eos_token_id is not None:
        token_ids.append(int(tokenizer.eos_token_id))

    # Preserve order but drop duplicates
    seen = set()
    unique_ids: List[int] = []
    for tok_id in token_ids:
        if tok_id not in seen:
            seen.add(tok_id)
            unique_ids.append(tok_id)
    return unique_ids

