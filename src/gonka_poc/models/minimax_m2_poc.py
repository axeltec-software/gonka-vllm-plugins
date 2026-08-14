"""MiniMax-M2 with PoC transforms baked in BEFORE compilation.

Registered over the stock architecture name from ``plugin.register()`` via the
public ``ModelRegistry`` surface, so vLLM compiles the WRAPPED model: the
Householder layer wrappers and the seeded-router gate wrappers become part of
the compiled graph — exactly how the 0.20 in-tree branch ran (attach before
compile), which is the bit-reference for golden trajectories.

Chat safety: with the PoC mask all-zero every wrapper is an exact identity
(``torch.where(False, t, x) -> x``); the 0.20 branch shipped the same wrappers
under chat traffic (gsm8k-verified).
"""
import os

import torch

from vllm.model_executor.models.minimax_m2 import MiniMaxM2ForCausalLM

from gonka_poc.poc.native import attach_native_poc

# Buffer cap: prefill rows of the largest decode chunk (nonces * seq_len).
POC_NATIVE_MAX_ROWS = int(os.environ.get("POC_NATIVE_MAX_ROWS", str(128 * 256)))
# 0.20 engine default (poc_route_window=16). PROCESS constant: the window is
# baked into the compiled graph at first compilation (dynamo freezes the
# global) — request-time switching is impossible by construction, exactly
# like the 0.20 engine arg. Change via env before start, never per request.
POC_ROUTE_WINDOW_DEFAULT = int(os.environ.get("POC_ROUTE_WINDOW", "16"))


class MiniMaxM2ForCausalLMPoC(MiniMaxM2ForCausalLM):
    """Wrappers attach at the END of load_weights: after the checkpoint is
    mapped (wrapping earlier renames parameters to ``layers.N.inner.*`` and
    breaks weight loading) and before the first forward (vLLM compiles the
    forward lazily on first call — so the compiled graph still contains the
    wrappers, the 0.20 bit path)."""

    def load_weights(self, weights):
        out = super().load_weights(weights)
        vllm_config = getattr(self, "vllm_config", None)
        hidden = (int(vllm_config.model_config.get_hidden_size())
                  if vllm_config is not None else
                  int(self.config.hidden_size))
        try:
            p = next(self.parameters())
            device, dtype = p.device, p.dtype
        except StopIteration:  # pragma: no cover
            device, dtype = torch.device("cuda"), torch.bfloat16
        max_rows = POC_NATIVE_MAX_ROWS
        if vllm_config is not None:
            try:
                max_rows = max(max_rows,
                               int(vllm_config.scheduler_config
                                   .max_num_batched_tokens))
            except Exception:  # pragma: no cover — config shape drift
                pass
        attach_native_poc(self, hidden, max_rows, device, dtype,
                          POC_ROUTE_WINDOW_DEFAULT)
        return out
