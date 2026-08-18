"""DeepSeek-family models with PoC transforms baked in before compilation.

Same discipline as :mod:`.minimax_m2_poc`: the registry override makes vLLM
construct OUR subclass, and the wrappers attach at the END of ``load_weights``
(earlier renames parameters, later misses the lazily-compiled graph).

One subclass covers the whole family — Kimi K-series checkpoints declare the
DeepSeek architecture in their HF config, so they resolve to the same class.
The grouped-router support (two-stage seeded forcing)
lives in gpu_random/native and keys off n_group/topk_group discovered at
attach time, so nothing here is model-specific beyond the base class.
"""
import os

import torch

from gonka_poc.poc.native import attach_native_poc

POC_NATIVE_MAX_ROWS = int(os.environ.get("POC_NATIVE_MAX_ROWS", str(128 * 256)))
POC_ROUTE_WINDOW_DEFAULT = int(os.environ.get("POC_ROUTE_WINDOW", "256"))


def _attach_after_load(self, out):
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


def build_poc_subclasses():
    """Yield (architecture_name, subclass) for every DeepSeek-family base
    class present in this vLLM build. Import errors are per-architecture:
    a build that lacks V4 still registers V3."""
    candidates = [
        ("DeepseekV3ForCausalLM",
         "vllm.model_executor.models.deepseek_v2", "DeepseekV3ForCausalLM"),
        ("DeepseekV2ForCausalLM",
         "vllm.model_executor.models.deepseek_v2", "DeepseekV2ForCausalLM"),
        ("DeepseekV4ForCausalLM",
         "vllm.model_executor.models.deepseek_v4", "DeepseekV4ForCausalLM"),
    ]
    import importlib
    for arch, mod_name, cls_name in candidates:
        try:
            base = getattr(importlib.import_module(mod_name), cls_name)
        except (ImportError, AttributeError):
            continue

        def load_weights(self, weights, _base=base):
            return _attach_after_load(self, _base.load_weights(self, weights))

        sub = type(f"{cls_name}PoC", (base,), {
            "__doc__": f"{cls_name} + PoC wrappers attached after weight "
                       f"load (see module docstring).",
            "load_weights": load_weights,
        })
        yield arch, sub
