from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..model_config import ModelConfig, get_model_config
from .attention_manager import AttentionIntervention, AttentionState, manager


def _segment_analysis_steps(analysis_text: str, min_chars: int = 15) -> int:
    """Replicate step segmentation logic from scripts/analyze_step_saliency."""

    if not analysis_text:
        return 0

    steps = 0
    sentences = re.split(r"\.\s+", analysis_text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) < min_chars:
            continue
        sanitized = sentence.replace(".", "").replace(",", "").replace("-", "").replace(" ", "")
        if sanitized.isdigit():
            continue
        if sentence.startswith("|") and sentence.count("|") > sentence.count("\n") * 2:
            continue
        if sentence.startswith(("|--", "---")) and len(sentence) < 20:
            continue
        steps += 1
    return steps


@dataclass
class _StepMomentumWindow:
    start: int  # inclusive
    end: int    # exclusive


@dataclass
class StepMomentumStats:
    """Lightweight activity tracker for SMI."""

    triggers: int = 0
    total_window_tokens: int = 0
    total_norm: float = 0.0
    last_window: Optional[Tuple[int, int]] = None
    last_layer: Optional[int] = None
    last_position: Optional[int] = None
    last_norm: float = 0.0

    def reset(self) -> None:
        self.triggers = 0
        self.total_window_tokens = 0
        self.total_norm = 0.0
        self.last_window = None
        self.last_layer = None
        self.last_position = None
        self.last_norm = 0.0

    def record(
        self,
        *,
        layer_idx: int,
        position_index: int,
        window_start: int,
        window_end: int,
        injection_tensor: torch.Tensor,
    ) -> None:
        window_tokens = max(0, window_end - window_start)
        self.triggers += 1
        self.total_window_tokens += window_tokens
        self.last_window = (window_start, window_end)
        self.last_layer = layer_idx
        self.last_position = position_index
        norm = injection_tensor.norm().item()
        self.last_norm = norm
        self.total_norm += norm

    def summary(self) -> dict:
        avg_window = (
            self.total_window_tokens / self.triggers if self.triggers else 0.0
        )
        avg_norm = self.total_norm / self.triggers if self.triggers else 0.0
        last_window = list(self.last_window) if self.last_window else None
        return {
            "triggers": self.triggers,
            "avg_window_tokens": avg_window,
            "last_window": last_window,
            "last_layer": self.last_layer,
            "last_position": self.last_position,
            "last_norm": self.last_norm,
            "avg_norm": avg_norm,
        }


class StepMomentumTracker:
    """Tracks analysis steps and determines when to inject momentum."""

    _TAIL_LIMIT = 256

    def __init__(
        self,
        tokenizer,
        *,
        min_step_chars: int = 15,
        model_config: ModelConfig,
    ) -> None:
        self.tokenizer = tokenizer
        self.min_step_chars = min_step_chars
        self.model_config = model_config
        self.analysis_marker = model_config.analysis_start_marker
        self.final_marker = model_config.final_start_marker
        self.reset()

    def reset(self) -> None:
        self.prompt_len: int = 0
        self.generated_tokens: int = 0
        self.stream_cursor: int = 0
        self.analysis_active: bool = False
        self.final_started: bool = False
        self.analysis_text: str = ""
        self.tail_buffer: str = ""
        self.completed_steps: int = 0
        self.current_step_start: Optional[int] = None
        self.pending_step_start: Optional[int] = None
        self.prev_step_range: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------ Sequence
    def start_sequence(self, input_ids: torch.Tensor) -> None:
        self.reset()
        tokens = input_ids[0].tolist()
        self.prompt_len = len(tokens)
        self.stream_cursor = len(tokens)
        for token in tokens:
            text = self.tokenizer.decode([token], skip_special_tokens=False)
            self._update_tail(text)

    def append_tokens(self, input_ids: torch.Tensor) -> None:
        tokens = input_ids[0].tolist()
        current_len = len(tokens)
        if current_len > self.stream_cursor:
            new_tokens = tokens[self.stream_cursor : current_len]
        else:
            new_tokens = tokens
        self.stream_cursor = current_len

        for token in new_tokens:
            text = self.tokenizer.decode([token], skip_special_tokens=False)
            absolute_index = self.prompt_len + self.generated_tokens
            self.generated_tokens += 1
            self._update_tail(text)
            if not self.analysis_active or self.final_started:
                continue
            if text:
                self.analysis_text += text
            if self.current_step_start is None:
                self.current_step_start = absolute_index
            if "." in text:
                self._maybe_close_step(absolute_index)

    # ----------------------------------------------------------------- Utilities
    def _update_tail(self, fragment: str) -> None:
        if not fragment:
            return
        self.tail_buffer = (self.tail_buffer + fragment)[-self._TAIL_LIMIT :]
        analysis_marker = self.analysis_marker
        final_marker = self.final_marker
        if (
            analysis_marker
            and not self.analysis_active
            and analysis_marker in self.tail_buffer
        ):
            self.analysis_active = True
            self.final_started = False
            self.analysis_text = ""
            self.current_step_start = self.prompt_len + self.generated_tokens
            self.completed_steps = 0
        if (
            final_marker
            and self.analysis_active
            and not self.final_started
            and final_marker in self.tail_buffer
        ):
            self.final_started = True
            self.pending_step_start = None

    def _maybe_close_step(self, absolute_index: int) -> None:
        step_count = _segment_analysis_steps(self.analysis_text, self.min_step_chars)
        if step_count <= self.completed_steps:
            return
        self.completed_steps = step_count
        if self.current_step_start is not None and absolute_index + 1 > self.current_step_start:
            self.prev_step_range = (self.current_step_start, absolute_index + 1)
        else:
            self.prev_step_range = None
        self.pending_step_start = absolute_index + 1
        self.current_step_start = absolute_index + 1

    # ---------------------------------------------------------------- Momentum API
    def consume_pending_window(self, position_index: int, window_tokens: int) -> Optional[_StepMomentumWindow]:
        if self.pending_step_start is None:
            return None
        if position_index != self.pending_step_start:
            return None
        self.current_step_start = position_index
        self.pending_step_start = None
        if not self.prev_step_range:
            return None
        start, end = self.prev_step_range
        if end <= start:
            return None
        window_start = max(start, end - window_tokens)
        return _StepMomentumWindow(window_start, end)


class StepMomentumInjection(AttentionIntervention):
    """Injects value momentum from the previous step into the first token of the next."""

    def __init__(
        self,
        tracker: StepMomentumTracker,
        *,
        stats: StepMomentumStats,
        strength: float,
        window_tokens: int,
        min_layer: int,
    ) -> None:
        self.tracker = tracker
        self.stats = stats
        self.strength = strength
        self.window_tokens = window_tokens
        self.min_layer = min_layer

    def on_output(self, context, head_output, scores, attn_weights):
        if head_output.shape[2] != 1:
            return head_output

        layer_idx = getattr(context.module, "layer_idx", None)
        if layer_idx is None or layer_idx < self.min_layer:
            return head_output

        state = context.state
        if state is None or not state.extras:
            return head_output

        tracker = state.extras.get("smi_tracker")
        if tracker is None:
            return head_output

        position_index = self._extract_position(context)
        if position_index is None:
            return head_output

        window = tracker.consume_pending_window(position_index, self.window_tokens)
        if window is None:
            return head_output

        value_states = context.value_states
        seq_len = value_states.shape[2]
        start = max(0, min(window.start, seq_len))
        end = max(0, min(window.end, seq_len))
        if end <= start:
            return head_output

        prev_slice = value_states[:, :, start:end, :]
        if prev_slice.numel() == 0:
            return head_output

        momentum = prev_slice.mean(dim=2, keepdim=True)
        injection = momentum * self.strength
        self.stats.record(
            layer_idx=layer_idx,
            position_index=position_index,
            window_start=start,
            window_end=end,
            injection_tensor=injection,
        )
        return head_output + injection

    @staticmethod
    def _extract_position(context) -> Optional[int]:
        cache_position = context.extra_kwargs.get("cache_position")
        if cache_position is not None:
            tensor = cache_position
        else:
            tensor = context.extra_kwargs.get("position_ids")
        if tensor is None:
            return None
        if tensor.dim() == 2:
            pos = tensor[0, -1]
        else:
            pos = tensor[-1]
        return int(pos.item())


class StepMomentumInjectionWrapper:
    """High-level helper to attach Step Momentum Injection to a model."""

    def __init__(
        self,
        tokenizer,
        *,
        strength: float = 0.06,
        min_layer: Optional[int] = None,
        window_tokens: int = 4,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.strength = strength
        self.model_config = model_config or get_model_config("gpt-oss")
        self.min_layer = (
            min_layer
            if min_layer is not None
            else self.model_config.default_smi_min_layer
        )
        self.window_tokens = window_tokens

        self.stats = StepMomentumStats()
        self.tracker = StepMomentumTracker(
            tokenizer,
            model_config=self.model_config,
        )
        self.intervention = StepMomentumInjection(
            self.tracker,
            stats=self.stats,
            strength=strength,
            window_tokens=window_tokens,
            min_layer=min_layer,
        )

        self._model = None
        self._orig_prepare_inputs = None

    # ------------------------------------------------------------------ Lifecycle
    def apply(self, model) -> None:
        if self._model is not None:
            return
        self._model = model
        self._orig_prepare_inputs = model.prepare_inputs_for_generation

        def patched_prepare_inputs(inner_self, input_ids, **model_kwargs):
            past = model_kwargs.get("past_key_values")
            is_new_sequence = past is None
            if not is_new_sequence and isinstance(past, (tuple, list)):
                is_new_sequence = len(past) == 0
            if not is_new_sequence and hasattr(past, "get_seq_length"):
                try:
                    is_new_sequence = past.get_seq_length() == 0
                except Exception:
                    is_new_sequence = False
            if is_new_sequence:
                self.stats.reset()
                self.tracker.start_sequence(input_ids)
            else:
                self.tracker.append_tokens(input_ids)

            state = AttentionState(
                analysis_mask=None,
                final_mask=None,
                prompt_len=self.tracker.prompt_len,
                extras={
                    "smi_tracker": self.tracker,
                    "smi_stats": self.stats,
                },
            )
            manager.set_state(state)
            return self._orig_prepare_inputs(input_ids, **model_kwargs)

        model.prepare_inputs_for_generation = patched_prepare_inputs.__get__(model, type(model))
        manager.register_intervention(self.intervention)

    def remove(self) -> None:
        if self._model is None:
            return

        manager.unregister_intervention(self.intervention)

        if self._orig_prepare_inputs is not None:
            self._model.prepare_inputs_for_generation = self._orig_prepare_inputs

        self._model = None
        self._orig_prepare_inputs = None
        self.tracker.reset()

    def collect_stats(self) -> Optional[dict]:
        return self.stats.summary()


