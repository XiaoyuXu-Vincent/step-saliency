from __future__ import annotations

from typing import Callable, List, Optional

import torch

from .attention_manager import AttentionState, manager


def _is_new_sequence(past_kv) -> bool:
    if past_kv is None:
        return True
    if isinstance(past_kv, (tuple, list)):
        return len(past_kv) == 0
    if hasattr(past_kv, "get_seq_length"):
        try:
            return past_kv.get_seq_length() == 0
        except Exception:
            return False
    return False


class GenerationStateController:
    """Central hook to populate AttentionState before generation steps."""

    def __init__(self, model) -> None:
        self.model = model
        self.listeners: List[
            Callable[[AttentionState, torch.Tensor, bool, dict], None]
        ] = []
        self._orig_prepare_inputs = None
        self._prompt_len: int = 0

    # ------------------------------------------------------------------ Listener API
    def add_listener(
        self,
        listener: Callable[[AttentionState, torch.Tensor, bool, dict], None],
    ) -> None:
        if listener in self.listeners:
            return
        self.listeners.append(listener)
        if self._orig_prepare_inputs is None:
            self._patch_model()

    def remove_listener(
        self,
        listener: Callable[[AttentionState, torch.Tensor, bool, dict], None],
    ) -> None:
        if listener in self.listeners:
            self.listeners.remove(listener)
        if not self.listeners and self._orig_prepare_inputs is not None:
            self._restore_model()

    # ------------------------------------------------------------------ Patching
    def _patch_model(self) -> None:
        if self._orig_prepare_inputs is not None:
            return
        self._orig_prepare_inputs = self.model.prepare_inputs_for_generation

        controller = self

        def patched_prepare_inputs(inner_self, input_ids, **model_kwargs):
            past_kv = model_kwargs.get("past_key_values")
            is_new = _is_new_sequence(past_kv)
            if is_new:
                controller._prompt_len = input_ids.shape[-1]

            state = AttentionState(
                analysis_mask=None,
                final_mask=None,
                prompt_len=controller._prompt_len,
                extras={},
            )

            # Snapshot listeners in case the list mutates during iteration
            for listener in list(controller.listeners):
                listener(state, input_ids, is_new, model_kwargs)

            manager.set_state(state)
            return controller._orig_prepare_inputs(input_ids, **model_kwargs)

        self.model.prepare_inputs_for_generation = patched_prepare_inputs.__get__(
            self.model, type(self.model)
        )

    def _restore_model(self) -> None:
        if self._orig_prepare_inputs is None:
            return
        self.model.prepare_inputs_for_generation = self._orig_prepare_inputs
        self._orig_prepare_inputs = None
        self._prompt_len = 0


def get_generation_state_controller(model) -> GenerationStateController:
    """Get or create the controller associated with a model."""

    attr = "_generation_state_controller"
    controller: Optional[GenerationStateController] = getattr(model, attr, None)
    if controller is None:
        controller = GenerationStateController(model)
        setattr(model, attr, controller)
    return controller




