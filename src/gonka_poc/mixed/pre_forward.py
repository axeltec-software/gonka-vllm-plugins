# SPDX-License-Identifier: Apache-2.0
"""The pre-forward callback — the single consumer of the residual's
``GPUModelRunner.pre_forward_hooks`` seam (kaitakuai/vllm @ mixed/pre-forward-hooks).

Contract with the residual (enforced by convention + smoke test):
  * called once per engine step, right before the forward, with
    (runner, scheduler_output, input_ids, positions, inputs_embeds,
    attn_metadata);
  * buffer-writes only — no allocation, no data-dependent control flow.
    CUDA-graph capture goes through _dummy_run and never sees hooks, so
    anything fed to the model must live in persistent, address-stable
    buffers (PoCNativeState) that captured graphs read by address.

This revision is the WORKABILITY skeleton for the B300 smoke: it proves the
seam (hook fires every step, sees real step context, adds no measurable
overhead) and records evidence. The mixed row-layout delivery
(nonce-sorted synthetic embeds + reflection mask via state.set_rows /
set_routing / set_embeds) lands next, gated on the E1/E2 experiments.
"""
import os

# Smoke evidence, drained via PoCWorkerExtension.mixed_hook_stats RPC.
_STATS = {
    "calls": 0,
    "last_num_tokens": -1,
    "last_num_reqs": -1,
    "has_attn_metadata": False,
    "errors": 0,
}


def poc_pre_forward(runner, scheduler_output, input_ids, positions,
                    inputs_embeds, attn_metadata):
    """Skeleton hook: observe-only (cheap int writes; no GPU work, no sync)."""
    try:
        _STATS["calls"] += 1
        if scheduler_output is not None:
            _STATS["last_num_tokens"] = int(
                getattr(scheduler_output, "total_num_scheduled_tokens", -1))
            _STATS["last_num_reqs"] = len(
                getattr(scheduler_output, "num_scheduled_tokens", ()) or ())
        _STATS["has_attn_metadata"] = attn_metadata is not None
    except Exception:  # noqa: BLE001 — the hook must never kill a step
        _STATS["errors"] += 1


def stats() -> dict:
    return dict(_STATS)


def install(runner) -> bool:
    """Idempotently register the hook on a runner. Returns True if installed."""
    if os.environ.get("POC_MIXED_PRE_FORWARD", "0") != "1":
        return False
    hooks = getattr(runner, "pre_forward_hooks", None)
    if hooks is None:
        raise RuntimeError(
            "gonka_poc.mixed: runner has no pre_forward_hooks — the residual "
            "build lacks the mixed/pre-forward-hooks patch (kaitakuai/vllm); "
            "refusing to fall back to private-internals patching")
    if poc_pre_forward not in hooks:
        hooks.append(poc_pre_forward)
    return True
