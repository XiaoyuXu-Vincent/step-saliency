# saliency_extractor.py
# -----------------------------------------------------------
# Universal attention saliency extractor for decoder-only LLMs.
# Supports multiple model architectures including:
# - GPT-OSS-20B (24 layers, standard attention)
# - Qwen3-8B (36 layers, GQA - Grouped Query Attention)
#
# Works with HF-style models that return (attn_output, attn_probs, ...)
# from self-attention blocks when output_attentions=True.
# -----------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# Model configuration registry
MODEL_CONFIGS = {
    'gpt-oss': {
        'num_layers': 24,
        'attention_type': 'standard',
        'module_filters': ('self_attn', 'attn', 'attention'),
        'description': 'GPT-OSS-20B with standard Transformer attention'
    },
    'qwen3': {
        'num_layers': 36,
        'attention_type': 'gqa',  # Grouped Query Attention
        'module_filters': ('self_attn', 'attn', 'attention'),
        'description': 'Qwen3-8B with GQA (8 KV heads, 32 Q heads)'
    },
    'qwen3-4b': {
        'num_layers': 36,
        'attention_type': 'gqa',  # Grouped Query Attention
        'module_filters': ('self_attn', 'attn', 'attention'),
        'description': 'Qwen3-4B with GQA (8 KV heads, 32 Q heads)'
    },
    'qwen3-8b': {
        'num_layers': 36,
        'attention_type': 'gqa',  # Grouped Query Attention
        'module_filters': ('self_attn', 'attn', 'attention'),
        'description': 'Qwen3-8B with GQA (8 KV heads, 32 Q heads)'
    }
}


@dataclass
class LayerInfo:
    kind: str               # 'SW' | 'Full' | 'Unknown'
    heads: int
    seq_len: int


@dataclass
class GroupMetrics:
    bos_mean: float
    entropy: float
    self_mean: float
    bos_mean_sal: Optional[float] = None
    entropy_sal: Optional[float] = None
    self_mean_sal: Optional[float] = None


@dataclass
class SaliencyResult:
    layers_info: List[LayerInfo]
    saliency_per_layer: List[torch.Tensor]                # each [B,H,T,T]
    attn_probs_per_layer: Optional[List[torch.Tensor]]    # each [B,H,T,T]
    bos_mean: List[float]          # per-layer metrics on probs
    entropy: List[float]
    self_mean: List[float]
    bos_mean_sal: Optional[List[float]]   # per-layer metrics on saliency
    entropy_sal: Optional[List[float]]
    self_mean_sal: Optional[List[float]]
    by_kind: Dict[str, GroupMetrics]      # aggregated by SW / Full
    model_type: str                        # which model was used


# ----------------- helpers -----------------
def _maybe_force_attn_impl(model: nn.Module, attn_impl: Optional[str] = "eager"):
    """Force attention implementation to 'eager' for gradient computation"""
    cfg = getattr(model, "config", None)
    if attn_impl is None:
        return
    if cfg is not None:
        if hasattr(cfg, "attn_implementation"):
            cfg.attn_implementation = attn_impl
        if hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = attn_impl
    if hasattr(model, "attn_implementation"):
        setattr(model, "attn_implementation", attn_impl)


def _entropy(P: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute entropy of attention distribution"""
    P = P.clamp_min(eps)
    return -(P * P.log()).sum(dim=-1)  # [B,H,T]


def _bos_mean(P: torch.Tensor, bos_idx: int = 0) -> torch.Tensor:
    """Compute mean attention to BOS token"""
    result = P[..., bos_idx].mean()  # average over all dimensions
    return result


def _self_mean(P: torch.Tensor) -> torch.Tensor:
    """Compute mean self-attention (diagonal)"""
    T = P.shape[-1]
    eye = torch.eye(T, device=P.device).view(1, 1, T, T)
    result = (P * eye).sum(dim=-1).mean()  # average over all dimensions
    return result


def _metrics_list(tensors: List[torch.Tensor], bos_idx: int) -> Tuple[List[float], List[float], List[float]]:
    """Compute metrics for a list of attention tensors"""
    bos, ent, selfm = [], [], []
    for P in tensors:
        bos.append(float(_bos_mean(P, bos_idx)))
        ent.append(float(_entropy(P).mean()))
        selfm.append(float(_self_mean(P)))
    return bos, ent, selfm


def _detect_layer_kind_from_probs(P: torch.Tensor, eps: float = 1e-9) -> str:
    """
    Heuristic detection from attention probability maps.
    Full causal: row width grows with row index up to ~T (triangular).
    Sliding-window causal: row width saturates at a band width w << T.
    """
    Pm = P.mean(dim=(0, 1))  # [T,T]
    T = Pm.shape[-1]
    nz = (Pm > eps).to(torch.int32)
    widths = nz.sum(dim=-1).float()  # each row effective width
    mean_width = widths.mean().item()
    max_width = widths.max().item()

    if T >= 64:
        if max_width < 0.8 * T and mean_width < 0.6 * T:
            return "SW"
        if max_width >= 0.8 * T and mean_width >= 0.45 * T:
            return "Full"
    else:
        mid = T // 2
        if widths[-1] - widths[mid] < 0.1 * T:
            return "SW"
        if widths[-1] > 0.8 * T:
            return "Full"
    return "Unknown"


def _validate_model_type(model_type: str) -> Dict[str, Any]:
    """Validate and return model configuration"""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Unsupported model_type: {model_type}. "
            f"Supported types: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_type]


# ----------------- main API -----------------
def extract_saliency(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    *,
    model_type: str = 'gpt-oss',
    attn_impl: Optional[str] = "eager",
    include_modules_filters: Optional[Sequence[str]] = None,
    mode: str = "loss",                         # 'loss' or 'logit'
    labels: Optional[torch.Tensor] = None,
    target_pos: Optional[int] = None,
    target_token_id: Optional[int] = None,
    rule: str = "ag",                           # 'ag' | 'ag_signed' | 'g'
    bos_idx: int = 0,
    return_attn_probs: bool = True,
    compute_saliency_metrics: bool = True,
) -> SaliencyResult:
    """
    Extract attention saliency for decoder-only LLMs.
    
    Args:
        model: The language model (GPT-OSS, Qwen3, etc.)
        inputs: Dictionary with 'input_ids' and optionally 'attention_mask'
        model_type: Model architecture type ('gpt-oss' or 'qwen3')
        attn_impl: Attention implementation ('eager' for gradient computation)
        include_modules_filters: Module name filters (None = use model defaults)
        mode: 'loss' for next-token prediction or 'logit' for specific token
        labels: Labels for loss computation (None = use input_ids)
        target_pos: Position for logit mode
        target_token_id: Token ID for logit mode
        rule: Saliency rule ('ag' = abs(A*grad), 'ag_signed' = A*grad, 'g' = abs(grad))
        bos_idx: Index of BOS token for metrics
        return_attn_probs: Whether to return attention probabilities
        compute_saliency_metrics: Whether to compute saliency-based metrics
    
    Returns:
        SaliencyResult with per-layer saliency maps and statistics
    """
    # Validate model type and get configuration
    model_config = _validate_model_type(model_type)
    
    # Use model-specific filters if not provided
    if include_modules_filters is None:
        include_modules_filters = model_config['module_filters']
    
    model.eval()
    torch.set_grad_enabled(True)
    _maybe_force_attn_impl(model, attn_impl=attn_impl)

    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    # hook attention probabilities
    probs_list: List[torch.Tensor] = []
    handles: List[Any] = []

    def hook(module, inp, out):
        # expect out like (attn_output, attn_probs, ...)
        if isinstance(out, (tuple, list)) and len(out) >= 2 and torch.is_tensor(out[1]):
            p = out[1]
            # Only retain grad if the tensor requires gradients
            if p.requires_grad:
                p.retain_grad()
                probs_list.append(p)

    for name, m in model.named_modules():
        if any(tag in name for tag in include_modules_filters):
            try:
                handles.append(m.register_forward_hook(hook))
            except Exception:
                pass

    # forward (avoid .generate(); use direct forward for backprop stability)
    kwargs = dict(output_attentions=True, return_dict=True, use_cache=False)
    if mode == "loss":
        if labels is None:
            labels = inputs.get("labels", None) or inputs.get("input_ids", None)
        out = model(**inputs, labels=labels, **kwargs)
        if hasattr(out, "loss") and out.loss is not None:
            scalar = out.loss
        else:
            logits = out.logits
            x = inputs["input_ids"]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = x[:, 1:].contiguous()
            scalar = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
    else:
        out = model(**inputs, **kwargs)
        logits = out.logits
        assert target_pos is not None and target_token_id is not None, "logit mode needs target_pos & target_token_id"
        scalar = logits[0, int(target_pos), int(target_token_id)]

    # backward
    model.zero_grad(set_to_none=True)
    scalar.backward()

    # saliency & layer taxonomy
    saliency_per_layer: List[torch.Tensor] = []
    kept_probs: List[torch.Tensor] = []
    layers_info: List[LayerInfo] = []

    for p in probs_list:
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        if rule == "ag":
            s = (p * g).abs()
        elif rule == "ag_signed":
            s = (p * g)
        else:
            s = g.abs()
        saliency_per_layer.append(s.detach())
        if return_attn_probs:
            kept_probs.append(p.detach())
        
        # Determine layer kind based on model type
        if model_type.startswith('qwen3'):
            # Qwen3 models use full attention in all layers (no sliding window)
            kind = "Full"
        else:
            # For other models (e.g., gpt-oss), auto-detect from attention pattern
            kind = _detect_layer_kind_from_probs(p.detach())
        
        _, H, T, _ = p.shape
        layers_info.append(LayerInfo(kind=kind, heads=H, seq_len=T))

    # metrics
    bos, ent, selfm = _metrics_list(kept_probs if return_attn_probs else saliency_per_layer, bos_idx)
    bos_s = ent_s = selfm_s = None
    if compute_saliency_metrics:
        bos_s, ent_s, selfm_s = _metrics_list(saliency_per_layer, bos_idx)

    # group by kind
    idx_sw = [i for i, info in enumerate(layers_info) if info.kind == "SW"]
    idx_full = [i for i, info in enumerate(layers_info) if info.kind == "Full"]
    def _avg(xs: List[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    by_kind: Dict[str, GroupMetrics] = {}
    for tag, idxs in (("SW", idx_sw), ("Full", idx_full)):
        if not idxs:
            continue
        gm = GroupMetrics(
            bos_mean=_avg([bos[i] for i in idxs]),
            entropy=_avg([ent[i] for i in idxs]),
            self_mean=_avg([selfm[i] for i in idxs]),
            bos_mean_sal=_avg([bos_s[i] for i in idxs]) if bos_s is not None else None,
            entropy_sal=_avg([ent_s[i] for i in idxs]) if ent_s is not None else None,
            self_mean_sal=_avg([selfm_s[i] for i in idxs]) if selfm_s is not None else None,
        )
        by_kind[tag] = gm

    # cleanup hooks
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    # Critical memory cleanup to prevent accumulation
    model.zero_grad(set_to_none=True)

    # Clear intermediate variables that hold large tensors
    del probs_list
    if 'g' in locals():
        del g
    if 's' in locals():
        del s

    # Force garbage collection and CUDA cache clearing
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return SaliencyResult(
        layers_info=layers_info,
        saliency_per_layer=saliency_per_layer,
        attn_probs_per_layer=kept_probs if return_attn_probs else None,
        bos_mean=bos, entropy=ent, self_mean=selfm,
        bos_mean_sal=bos_s, entropy_sal=ent_s, self_mean_sal=selfm_s,
        by_kind=by_kind,
        model_type=model_type,
    )


# Backward compatibility: alias for old function name
def extract_saliency_gptoss(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    **kwargs
) -> SaliencyResult:
    """
    Backward compatibility wrapper for extract_saliency_gptoss.
    Automatically sets model_type='gpt-oss'.
    """
    # Remove model_type if provided to avoid conflict
    kwargs.pop('model_type', None)
    return extract_saliency(model, inputs, model_type='gpt-oss', **kwargs)


def brief_report(res: SaliencyResult) -> str:
    """Generate a brief text report of saliency results"""
    L = len(res.layers_info)
    heads = [info.heads for info in res.layers_info]
    kinds = [info.kind for info in res.layers_info]
    head_str = ",".join(map(str, heads[:6])) + ("..." if len(heads) > 6 else "")
    kind_str = ",".join(kinds[:12]) + ("..." if len(kinds) > 12 else "")
    lines = []
    lines.append(f"[Saliency] Model: {res.model_type}, L={L}, heads/layer={head_str}")
    lines.append(f"Kinds per layer: {kind_str}")
    lines.append(f"BOS(prob) per-layer (first 6): " + ", ".join(f"{x:.4f}" for x in res.bos_mean[:6]))
    lines.append(f"Entropy(prob) per-layer (first 6): " + ", ".join(f"{x:.3f}" for x in res.entropy[:6]))
    lines.append(f"Self(prob) per-layer (first 6): " + ", ".join(f"{x:.4f}" for x in res.self_mean[:6]))
    if res.bos_mean_sal is not None:
        lines.append(f"BOS(sal) per-layer (first 6): " + ", ".join(f"{x:.4f}" for x in res.bos_mean_sal[:6]))
    if res.by_kind:
        lines.append("---- Grouped by attention kind ----")
        for tag, gm in res.by_kind.items():
            lines.append(f"{tag}: BOS={gm.bos_mean:.4f}, H={gm.entropy:.3f}, SELF={gm.self_mean:.4f}")
            if gm.bos_mean_sal is not None:
                lines.append(f"{tag} (sal): BOS={gm.bos_mean_sal:.4f}, H={gm.entropy_sal:.3f}, SELF={gm.self_mean_sal:.4f}")
    return "\n".join(lines)




