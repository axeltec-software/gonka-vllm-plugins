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
POC_ROUTE_WINDOW_DEFAULT = int(os.environ.get("POC_ROUTE_WINDOW", "256"))


class MiniMaxM2ForCausalLMPoC(MiniMaxM2ForCausalLM):

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        hidden = int(vllm_config.model_config.get_hidden_size())
        try:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
        except StopIteration:  # pragma: no cover
            device, dtype = torch.device("cuda"), torch.bfloat16
        attach_native_poc(self, hidden, POC_NATIVE_MAX_ROWS, device, dtype,
                          POC_ROUTE_WINDOW_DEFAULT)
