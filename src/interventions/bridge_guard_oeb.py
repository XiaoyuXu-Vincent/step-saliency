from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

from ..model_config import ModelConfig, get_model_config
from .attention_manager import AttentionIntervention, AttentionState, manager
from .state_controller import get_generation_state_controller


class ChannelSegmentTracker:
    """Track which tokens belong to question/analysis/final segments."""

    _TAIL_LIMIT = 512

    def __init__(self, tokenizer, model_config: ModelConfig) -> None:
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.analysis_marker = model_config.analysis_start_marker
        self.final_marker = model_config.final_start_marker
        self.reset()

    def reset(self) -> None:
        self.labels = []
        self.tail_buffer = ""
        self.analysis_active = False
        self.final_active = False
        self.stream_cursor = 0
        self.prompt_len = 0

    def start_sequence(self, input_ids: torch.Tensor) -> None:
        self.reset()
        tokens = input_ids[0].tolist()
        self.prompt_len = len(tokens)
        self.stream_cursor = len(tokens)
        for token in tokens:
            self._append_token(token)

    def append_tokens(self, input_ids: torch.Tensor) -> None:
        tokens = input_ids[0].tolist()
        current_len = len(tokens)
        if current_len > self.stream_cursor:
            new_tokens = tokens[self.stream_cursor : current_len]
            self.stream_cursor = current_len
        else:
            new_tokens = tokens
            self.stream_cursor += len(new_tokens)
        for token in new_tokens:
            self._append_token(token)

    def _append_token(self, token: int) -> None:
        text = self.tokenizer.decode([token], skip_special_tokens=False)
        label = 2 if self.final_active else (1 if self.analysis_active else 0)
        self.labels.append(label)
        if not text:
            return
        self.tail_buffer = (self.tail_buffer + text)[-self._TAIL_LIMIT :]
        analysis_marker = self.analysis_marker
        if (
            analysis_marker
            and not self.analysis_active
            and not self.final_active
            and analysis_marker in self.tail_buffer
        ):
            self.analysis_active = True
        final_marker = self.final_marker
        if (
            final_marker
            and self.analysis_active
            and not self.final_active
            and final_marker in self.tail_buffer
        ):
            self.final_active = True
            self.analysis_active = False

    def build_masks(self, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.labels:
            return None
        labels = torch.tensor(self.labels, device=device)
        analysis_mask = (labels == 1).unsqueeze(0)
        final_mask = (labels == 2).unsqueeze(0)
        return analysis_mask.bool(), final_mask.bool()


class _ChannelStateListener:
    """Share channel masks through AttentionState."""

    def __init__(self, tokenizer, model_config: ModelConfig, *, on_new_sequence=None) -> None:
        self.tracker = ChannelSegmentTracker(tokenizer, model_config=model_config)
        self._on_new_sequence = on_new_sequence

    def __call__(
        self,
        state: AttentionState,
        input_ids: torch.Tensor,
        is_new_sequence: bool,
        model_kwargs: dict,
    ) -> None:
        if is_new_sequence:
            if self._on_new_sequence:
                self._on_new_sequence()
            self.tracker.start_sequence(input_ids)
        else:
            self.tracker.append_tokens(input_ids)
        masks = self.tracker.build_masks(input_ids.device)
        if masks is None:
            return
        analysis_mask, final_mask = masks
        state.analysis_mask = analysis_mask
        state.final_mask = final_mask
        if not state.prompt_len:
            state.prompt_len = self.tracker.prompt_len
        state.extras["channel_tracker"] = self.tracker


@dataclass
class BridgeGuardStats:
    triggers: int = 0
    total_tau_bridge: float = 0.0
    total_lambda_bridge: float = 0.0
    last_layer: Optional[int] = None
    last_position: Optional[int] = None
    last_tau_bridge: float = 0.0
    last_lambda_bridge: float = 0.0

    def reset(self) -> None:
        self.triggers = 0
        self.total_tau_bridge = 0.0
        self.total_lambda_bridge = 0.0
        self.last_layer = None
        self.last_position = None
        self.last_tau_bridge = 0.0
        self.last_lambda_bridge = 0.0

    def record(
        self,
        *,
        layer_idx: int,
        position_index: int,
        tau_bridge: float,
        lambda_bridge: float,
    ) -> None:
        self.triggers += 1
        self.total_tau_bridge += tau_bridge
        self.total_lambda_bridge += lambda_bridge
        self.last_layer = layer_idx
        self.last_position = position_index
        self.last_tau_bridge = tau_bridge
        self.last_lambda_bridge = lambda_bridge

    def summary(self) -> dict:
        avg_tau = self.total_tau_bridge / self.triggers if self.triggers else 0.0
        avg_lambda = (
            self.total_lambda_bridge / self.triggers if self.triggers else 0.0
        )
        return {
            "triggers": self.triggers,
            "avg_bridge_mass": avg_tau,
            "avg_lambda_bridge": avg_lambda,
            "last_layer": self.last_layer,
            "last_position": self.last_position,
            "last_tau_bridge": self.last_tau_bridge,
            "last_lambda_bridge": self.last_lambda_bridge,
        }


class BridgeGuardOEB(AttentionIntervention):
    """Odds-Equal Bridge guard implemented as KL projection (Eq. 5-8 in paper)."""

    def __init__(
        self,
        *,
        tracker: ChannelSegmentTracker,
        max_layer: int,
        stats: BridgeGuardStats,
        layers: Optional[Iterable[int]] = None,
        tau_max: float = 0.15,
    ) -> None:
        self.max_layer = max_layer
        self.stats = stats
        self.tracker = tracker
        self.tau_max = tau_max
        self.eps = 1e-6
        if layers is not None:
            normalized = sorted({int(x) for x in layers if x >= 0})
            self.allowed_layers = tuple(normalized)
        else:
            self.allowed_layers = None

    def on_pre_softmax(self, context, attn_logits: torch.Tensor) -> torch.Tensor:

        layer_idx = getattr(context.module, "layer_idx", None)
        if layer_idx is None:
            return attn_logits
        if self.allowed_layers is not None:
            if layer_idx not in self.allowed_layers:
                return attn_logits
        elif layer_idx > self.max_layer:
            return attn_logits

        tracker = self.tracker

        masks = tracker.build_masks(attn_logits.device)
        if masks is None:
            return attn_logits

        analysis_mask, final_mask = masks

        position_index = self._extract_position(context)
        if position_index is None:
            return attn_logits

        device = attn_logits.device
        seq_len = attn_logits.shape[-1]
        analysis_mask = analysis_mask.to(device)[..., :seq_len]
        final_mask = final_mask.to(device)[..., :seq_len]
        if position_index >= analysis_mask.shape[-1]:
            return attn_logits

        masks = self._select_masks(analysis_mask, final_mask, position_index)
        if masks is None:
            return attn_logits
        same_mask, bridge_mask = masks
        same_mask = same_mask[..., :seq_len]
        bridge_mask = bridge_mask[..., :seq_len]
        if not same_mask.any() or not bridge_mask.any():
            return attn_logits

        updated = self._apply_projection(
            context,
            attn_logits,
            same_mask,
            bridge_mask,
            layer_idx,
            position_index,
        )
        return updated

    def _apply_projection(
        self,
        context,
        attn_logits: torch.Tensor,
        same_mask: torch.Tensor,
        bridge_mask: torch.Tensor,
        layer_idx: int,
        position_index: int,
    ) -> torch.Tensor:
        logits = attn_logits
        dtype = logits.dtype

        mask_same = same_mask.bool()
        mask_bridge = bridge_mask.bool()
        mask_other = ~(mask_same | mask_bridge)

        mask_same_f = mask_same.float().to(dtype)
        mask_bridge_f = mask_bridge.float().to(dtype)
        mask_other_f = mask_other.float().to(dtype)

        mask_same_4d = mask_same_f[:, None, None, :]
        mask_bridge_4d = mask_bridge_f[:, None, None, :]
        mask_other_4d = mask_other_f[:, None, None, :]

        probs = torch.softmax(logits, dim=-1)
        p_same = (probs * mask_same_4d).sum(dim=-1)
        p_bridge = (probs * mask_bridge_4d).sum(dim=-1)
        p_other = (probs * mask_other_4d).sum(dim=-1)

        p_bridge_scalar = p_bridge.squeeze(-1)
        p_same_scalar = p_same.squeeze(-1)
        p_other_scalar = p_other.squeeze(-1)

        # Eq. 6: tau_B = min(sqrt(|B| / (|B| + |S|)), tau_max)
        bridge_counts = mask_bridge_f.sum(dim=-1).clamp_min(self.eps)
        same_counts = mask_same_f.sum(dim=-1).clamp_min(self.eps)
        ratio = bridge_counts / (bridge_counts + same_counts + self.eps)
        tau_bridge = torch.clamp(torch.sqrt(ratio), max=self.tau_max)
        tau_bridge = tau_bridge.unsqueeze(1).expand_as(p_bridge_scalar)

        # Algorithm 1 line 6: only apply when p_B < tau_B
        needs_adjustment = p_bridge_scalar < tau_bridge
        available = 1.0 - p_other_scalar
        feasible = (available > self.eps) & (p_same_scalar > self.eps)
        active = needs_adjustment & feasible
        if not torch.any(active):
            return logits

        # Ensure tau_bridge does not exceed available mass
        tau_bridge = torch.minimum(tau_bridge, available - self.eps)
        tau_bridge = torch.clamp(tau_bridge, min=self.eps)

        # Eq. 6: tau_S = 1 - p(O) - tau_B
        tau_same = available - tau_bridge

        # Eq. 8: KL projection via group-wise logit shifts
        lambda_bridge = torch.log((tau_bridge + self.eps) / (p_bridge_scalar + self.eps))
        lambda_same = torch.log((tau_same + self.eps) / (p_same_scalar + self.eps))

        lambda_bridge = torch.where(active, lambda_bridge, torch.zeros_like(lambda_bridge))
        lambda_same = torch.where(active, lambda_same, torch.zeros_like(lambda_same))

        # Eq. 7: z'_k = z_k + lambda_g
        logits = logits + lambda_bridge.unsqueeze(-1).unsqueeze(-1) * mask_bridge_4d
        logits = logits + lambda_same.unsqueeze(-1).unsqueeze(-1) * mask_same_4d

        active_indices = torch.nonzero(active, as_tuple=False)
        for b, h in active_indices.tolist():
            self.stats.record(
                layer_idx=layer_idx,
                position_index=position_index,
                tau_bridge=float(tau_bridge[b, h].item()),
                lambda_bridge=float(lambda_bridge[b, h].item()),
            )
        return logits

    @staticmethod
    def _extract_position(context) -> Optional[int]:
        cache_position = context.extra_kwargs.get("cache_position")
        tensor = cache_position or context.extra_kwargs.get("position_ids")
        if tensor is None:
            return None
        if tensor.dim() == 2:
            pos = tensor[0, -1]
        else:
            pos = tensor[-1]
        return int(pos.item())

    @staticmethod
    def _select_masks(
        analysis_mask: torch.Tensor,
        final_mask: torch.Tensor,
        position_index: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        batch = analysis_mask.shape[0]
        for b in range(batch):
            if analysis_mask[b, position_index]:
                return analysis_mask, final_mask
            if final_mask[b, position_index]:
                return final_mask, analysis_mask
        return None


class BridgeGuardOEBWrapper:
    """Helper to register the KL projection module."""

    def __init__(
        self,
        tokenizer,
        *,
        max_layer: Optional[int] = None,
        layers: Optional[Iterable[int]] = None,
        tau_max: float = 0.15,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        self.model_config = model_config or get_model_config("gpt-oss")
        resolved_max_layer = (
            max_layer
            if max_layer is not None
            else self.model_config.default_oeb_max_layer
        )
        self.tokenizer = tokenizer
        self.max_layer = resolved_max_layer
        self.stats = BridgeGuardStats()
        self.state_listener = _ChannelStateListener(
            tokenizer,
            self.model_config,
            on_new_sequence=self.stats.reset,
        )
        self.intervention = BridgeGuardOEB(
            tracker=self.state_listener.tracker,
            max_layer=self.max_layer,
            stats=self.stats,
            layers=layers,
            tau_max=tau_max,
        )

        self._model = None
        self._controller = None

    def apply(self, model) -> None:
        if self._model is not None:
            return
        self._model = model
        self._controller = get_generation_state_controller(model)
        self._controller.add_listener(self.state_listener)
        manager.register_intervention(self.intervention)

    def remove(self) -> None:
        if self._model is None:
            return
        manager.unregister_intervention(self.intervention)
        if self._controller is not None:
            self._controller.remove_listener(self.state_listener)
            self._controller = None
        self._model = None
        self.state_listener.tracker.reset()
        self.stats.reset()

    def collect_stats(self) -> Optional[dict]:
        return self.stats.summary()

