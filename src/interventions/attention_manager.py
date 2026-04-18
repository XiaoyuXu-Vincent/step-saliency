#!/usr/bin/env python3
"""
Unified attention intervention manager for GPT-OSS eager attention.

Supports registering multiple interventions  that can
modify attention logits or outputs without conflicting monkey patches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import importlib

import torch
import torch.nn.functional as F


def _load_gpt_oss_module():
    """Dynamically load the GPT-OSS modeling module.

    The module lives inside a custom ``transformers`` build that ships
    GPT-OSS support.  We import lazily so that the rest of the
    intervention framework can be loaded even when the custom
    ``transformers`` is not installed (e.g. for reading the code or
    running tests with a different backbone).
    """
    try:
        return importlib.import_module(
            "transformers.models.gpt_oss.modeling_gpt_oss"
        )
    except ModuleNotFoundError as exc:
        raise ImportError(
            "GPT-OSS model support requires a custom transformers build "
            "that includes 'transformers.models.gpt_oss'. "
            "Please install the correct transformers version."
        ) from exc


gpt_oss_module = None  # populated lazily on first patch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat K/V for GQA."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


@dataclass
class AttentionState:
    """State derived from input ids (e.g., question/analysis/final masks)."""

    analysis_mask: Optional[torch.Tensor] = None  # shape [batch, seq_len] (bool)
    final_mask: Optional[torch.Tensor] = None     # shape [batch, seq_len] (bool)
    question_mask: Optional[torch.Tensor] = None  # shape [batch, seq_len] (bool)
    prompt_len: int = 0
    extras: dict = field(default_factory=dict)


@dataclass
class AttentionContext:
    """Context passed to interventions."""

    module: torch.nn.Module
    query: torch.Tensor
    key_states: torch.Tensor
    value_states: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    extra_kwargs: dict
    state: Optional[AttentionState] = None
    scores: Optional[torch.Tensor] = None  # Softmax scores (w/o sinks, pre-dropout)


class AttentionIntervention:
    """Base class for interventions."""

    def on_register(self, manager: "AttentionPatchManager") -> None:
        """Called once when the intervention is registered."""

    def on_unregister(self, manager: "AttentionPatchManager") -> None:
        """Called once when the intervention is unregistered."""

    def on_pre_softmax(
        self, context: AttentionContext, attn_logits: torch.Tensor
    ) -> torch.Tensor:
        """Modify attention logits before sinks are added."""
        return attn_logits

    def on_post_softmax(
        self, context: AttentionContext, scores: torch.Tensor
    ) -> torch.Tensor:
        """Modify attention scores after softmax (no sinks, pre-dropout)."""
        return scores

    def on_output(
        self,
        context: AttentionContext,
        head_output: torch.Tensor,
        scores: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Modify per-head attention output (B, H, q_len, head_dim)."""
        return head_output


class AttentionPatchManager:
    """Coordinates attention interventions and patches eager_attention_forward."""

    def __init__(self) -> None:
        self._original_forward = None
        self._interventions: List[AttentionIntervention] = []
        self._state: Optional[AttentionState] = None

    # ------------------------------------------------------------------ State
    def set_state(self, state: Optional[AttentionState]) -> None:
        self._state = state

    def clear_state(self) -> None:
        self._state = None

    # ------------------------------------------------------------- Registration
    def register_intervention(self, intervention: AttentionIntervention) -> None:
        if intervention in self._interventions:
            return
        if self._original_forward is None:
            self._patch_forward()
        self._interventions.append(intervention)
        intervention.on_register(self)

    def unregister_intervention(self, intervention: AttentionIntervention) -> None:
        if intervention not in self._interventions:
            return
        self._interventions.remove(intervention)
        intervention.on_unregister(self)
        if not self._interventions and self._original_forward is not None:
            self._restore_forward()

    # ------------------------------------------------------------- Patching core
    def _patch_forward(self) -> None:
        global gpt_oss_module
        if self._original_forward is not None:
            return
        if gpt_oss_module is None:
            gpt_oss_module = _load_gpt_oss_module()
        self._original_forward = gpt_oss_module.eager_attention_forward

        def patched_forward(
            module: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            scaling: float,
            dropout: float = 0.0,
            **kwargs: Any,
        ):
            key_states = repeat_kv(key, module.num_key_value_groups)
            value_states = repeat_kv(value, module.num_key_value_groups)

            attn_logits = torch.matmul(
                query, key_states.transpose(2, 3)
            ) * scaling

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_logits = attn_logits + causal_mask

            context = AttentionContext(
                module=module,
                query=query,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                extra_kwargs=kwargs,
                state=self._state,
            )

            for intervention in self._interventions:
                attn_logits = intervention.on_pre_softmax(context, attn_logits)

            sinks = module.sinks.reshape(1, -1, 1, 1).expand(
                query.shape[0], -1, query.shape[-2], -1
            )
            combined_logits = torch.cat([attn_logits, sinks], dim=-1)
            combined_logits = combined_logits - combined_logits.max(
                dim=-1, keepdim=True
            ).values

            probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
            scores = probs[..., :-1]  # Drop sink column

            context.scores = scores
            for intervention in self._interventions:
                scores = intervention.on_post_softmax(context, scores)

            attn_weights = F.dropout(scores, p=dropout, training=module.training)
            head_output = torch.matmul(attn_weights, value_states)  # [B, H, q, D]

            for intervention in self._interventions:
                head_output = intervention.on_output(
                    context, head_output, scores, attn_weights
                )

            attn_output = head_output.transpose(1, 2).contiguous()
            return attn_output, attn_weights

        gpt_oss_module.eager_attention_forward = patched_forward

    def _restore_forward(self) -> None:
        if self._original_forward is None:
            return
        gpt_oss_module.eager_attention_forward = self._original_forward
        self._original_forward = None
        self.clear_state()


# Global singleton manager
manager = AttentionPatchManager()


