"""Decode-PoC loop for the plugin architecture — replaces the 0.20 engine
mixing with a self-contained step loop on the worker.

One chunk = ``len(nonces)`` sequences, each holding ``seq_len + max_tokens``
KV tokens for the whole trajectory:

    prefill (seq_len synthetic embeds, fresh buffer — decision #12)
      -> snap step 0
      -> for t in 1..max_tokens:
           embeds_t = f(prev_k)            (chained, on-device)
           forward 1 token/row             (KV grows into slot seq_len+t-1)
           pick(t) -> project -> snap -> mismatch/teacher-force -> prev_k

Migration rules honoured (NOTES.md):
  * buffers are address-stable, the step body is pure tensor code — CAPTURE-
    READY; the initial revision replays it eagerly step-by-step, the capture
    of the step graph is the first perf iteration, not a redesign;
  * the prefill chunk must fit the pre-sized MoE workspace: chunk ≤
    ``POC_DECODE_PREFILL_CHUNK`` (default 128 nonces × 256 tokens = 32768);
  * shared engine buffers are never resized here (v0.1.3 lesson);
  * fresh prefill embeds always (decision #12 — golden parity), the legacy
    KV-scratch quirk stays in the prefill-only path of tag v0.1.x.

Consensus arithmetic lives in gpu_random/sphere/decode_chain (layer A,
bit-parity-tested); this module only orchestrates it.
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

import torch

from .gpu_random import (
    decode_base_seeds,
    generate_decode_inputs_gpu,
    generate_inputs,
    random_pick_indices,
    random_pick_indices_gpu,
)
from .sphere import (
    SPHERE_DIM,
    get_sphere_codebook,
    project_to_sphere,
    snap_with_margin,
)
from .decode_chain import count_mismatch, keep_q_step, next_prev_k
from .native import attach_native_poc

logger = logging.getLogger(__name__)

_MARGIN_TAU = float(os.environ.get("VLLM_POC_MARGIN_TAU", "0") or "0")
# Compiled path is the default (0.20 bit reference); eager only as a debug
# fallback behind the flag (migration rule: eager is not an execution mode).
_SKIP_COMPILED = os.environ.get("POC_DECODE_SKIP_COMPILED", "0") == "1"
# Per-step wall-clock breakdown (metadata/embeds/forward/snap) logged once per
# chunk; costs one cuda sync per step — diagnostics only, keep off in prod.
_PROFILE = os.environ.get("POC_DECODE_PROFILE", "0") == "1"
POC_DECODE_PREFILL_CHUNK = int(os.environ.get("POC_DECODE_PREFILL_CHUNK", "128"))


@torch.inference_mode()
def execute_poc_decode(
    worker,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    max_tokens: int,
    hidden_size: int,
    route_window: int,
    enforced_k_steps: Optional[Dict[int, List[int]]] = None,
    debug: bool = False,
    va_steps: int = 0,
    per_nonce_reflection: bool = False,
    borrowed_block_ids: Optional[List[int]] = None,
    borrowed_stripe: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Run one decode-PoC chunk on a V1 worker. Returns artifacts on the last
    PP rank of the TP driver; None elsewhere.

    Validation mode = ``enforced_k_steps`` given (teacher forcing, decision:
    the enforced k seeds the next step; every step an independent check).
    """
    import torch.distributed as dist
    from vllm.distributed import get_pp_group, get_tp_group
    from vllm.distributed.communication_op import broadcast_tensor_dict
    from vllm.forward_context import set_forward_context
    from gonka_poc._compat import current as _compat_current
    from .poc_model_runner import _create_v1_attn_metadata

    device = worker.device
    dtype = worker.model_config.dtype
    model = worker.model_runner.model
    vllm_config = worker.vllm_config
    batch = len(nonces)
    alloc_len = seq_len + max_tokens

    tp_group = get_tp_group()
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
        if tp_group.rank_in_group == 0:
            broadcast_tensor_dict({
                "poc_decode_go": True, "seq_len": seq_len,
                "max_tokens": max_tokens, "hidden_size": hidden_size,
                "nonces": nonces, "route_window": route_window,
                "debug": debug, "va_steps": va_steps,
                "per_nonce_reflection": per_nonce_reflection,
                "enforced": enforced_k_steps or {},
                "has_borrowed": borrowed_block_ids is not None,
                "borrowed_block_ids": borrowed_block_ids or [],
                "borrowed_stripe": int(borrowed_stripe or 0),
            }, src=0)
        else:
            bd = broadcast_tensor_dict(src=0)
            seq_len = int(bd["seq_len"]); max_tokens = int(bd["max_tokens"])
            hidden_size = int(bd["hidden_size"]); nonces = list(bd["nonces"])
            route_window = int(bd["route_window"]); debug = bool(bd["debug"])
            va_steps = int(bd["va_steps"])
            per_nonce_reflection = bool(bd["per_nonce_reflection"])
            enforced_k_steps = {int(k): v for k, v in bd["enforced"].items()} or None
            if bd.get("has_borrowed"):
                borrowed_block_ids = list(bd["borrowed_block_ids"])
                borrowed_stripe = int(bd["borrowed_stripe"])
            else:
                borrowed_block_ids = None; borrowed_stripe = None
            batch = len(nonces); alloc_len = seq_len + max_tokens

    pp_group = get_pp_group()
    if pp_group.world_size > 1:
        raise RuntimeError("decode-PoC: PP > 1 not supported in this revision")

    # --- model wrappers: pre-compile (registry class) or late attach -------
    # The registry-wrapped model carries state from construction (wrappers are
    # INSIDE the compiled graph — 0.20 bit path). Late attach remains as a
    # fallback for unregistered architectures; its wrappers are only seen by
    # the eager path.
    state = getattr(model, "_poc_native_state", None)
    if state is None:
        max_rows = POC_DECODE_PREFILL_CHUNK * seq_len
        state = attach_native_poc(model, hidden_size, max_rows, device, dtype,
                                  route_window)
    elif int(route_window) != int(getattr(state, "route_window", route_window)):
        # The window is frozen into the compiled graph at first compilation
        # (0.20 semantics: an engine arg, not a request knob) — a mismatched
        # request must fail loud, silently serving the frozen window would
        # produce consensus-invalid artifacts.
        raise ValueError(
            f"route_window={route_window} requested but the process was "
            f"compiled with {state.route_window}; set POC_ROUTE_WINDOW and "
            f"restart")
    if per_nonce_reflection:
        raise NotImplementedError(
            "per_nonce_reflection: per-row reflection buffers deferred "
            "(golden scope: off)")

    # --- chain state (address-stable, capture-ready) ------------------------
    codebook = get_sphere_codebook().to(device=device)
    base_seeds = decode_base_seeds(block_hash, public_key, nonces, device)
    prev_k = torch.zeros(batch, dtype=torch.int64, device=device)
    k_steps = torch.full((batch, max_tokens + 1), -1, dtype=torch.int64,
                         device=device)
    margin_steps = torch.zeros(batch, max_tokens + 1, device=device)
    n_nan = torch.zeros(batch, dtype=torch.int64, device=device)
    mismatch = torch.zeros(batch, dtype=torch.int64, device=device)
    validating = enforced_k_steps is not None
    if validating:
        ref_rows = [enforced_k_steps[n] for n in nonces]
        L = min(min(len(r) for r in ref_rows), max_tokens + 1)
        reference = torch.tensor([r[:L] for r in ref_rows], dtype=torch.int64,
                                 device=device)
    q_kept: Dict[int, torch.Tensor] = {}

    def snap_rows(last_hidden: torch.Tensor, sph_idx: torch.Tensor, step: int):
        """hidden [B,H] + pick idx [B,SPHERE_DIM] -> обновить цепочку шага."""
        nonlocal prev_k
        q = project_to_sphere(torch.gather(last_hidden.float(), 1, sph_idx))
        k, bad, margin = snap_with_margin(q, codebook)
        k_steps[:, step] = k
        margin_steps[:, step] = margin
        n_nan.add_(bad.to(torch.int64))
        if validating and step < reference.shape[1]:
            ref = reference[:, step]
            mismatch.add_(count_mismatch(k, ref, margin, _MARGIN_TAU))
            prev_k = next_prev_k(k, ref)
        else:
            prev_k = next_prev_k(k, None)
        if keep_q_step(step, debug, va_steps > 0, va_steps):
            q_kept[step] = q.detach().to(torch.float16).cpu()

    # ------------------------------------------------------------ prefill --
    if batch * seq_len > POC_DECODE_PREFILL_CHUNK * seq_len:
        raise ValueError(
            f"decode-PoC chunk {batch} nonces exceeds prefill workspace bound "
            f"{POC_DECODE_PREFILL_CHUNK} (MoE workspace is pre-sized; do not "
            f"resize after capture)")

    positions_pf = torch.arange(seq_len, dtype=torch.int64,
                                device=device).repeat(batch)
    attn_pf, slots_pf = _create_v1_attn_metadata(
        batch, seq_len, device, worker, positions_pf,
        borrowed_block_ids=borrowed_block_ids,
        borrowed_stripe=borrowed_stripe,
        alloc_len=alloc_len)
    embeds_pf = generate_inputs(block_hash, public_key, nonces,
                                dim=hidden_size, seq_len=seq_len,
                                device=device, dtype=dtype)

    if not getattr(state, "has_embed_patch", False):
        raise RuntimeError(
            "decode-PoC: embed_tokens patch missing — the compiled engine "
            "path needs in-model embedding swap (eager is not a mode)")

    # ENGINE signature: input_ids tensor + inputs_embeds None — the exact
    # shape @support_torch_compile was compiled under. Synthetic embeds are
    # staged into state and swapped in by the embed patch INSIDE the graph.
    ids_pf = torch.zeros(batch * seq_len, dtype=torch.int64, device=device)
    state.set_rows(block_hash, batch * seq_len)
    state.set_routing(block_hash, nonces, seq_len, 0)
    state.set_embeds(embeds_pf.view(-1, hidden_size))

    with set_forward_context(attn_pf, vllm_config,
                             num_tokens=batch * seq_len,
                             slot_mapping=slots_pf,
                             skip_compiled=_SKIP_COMPILED):
        hidden = model(input_ids=ids_pf, positions=positions_pf,
                       intermediate_tensors=None, inputs_embeds=None)
    if isinstance(hidden, tuple):
        hidden = hidden[0]
    last_pf = hidden.view(batch, seq_len, -1)[:, -1, :]
    sph0 = random_pick_indices(block_hash, public_key, nonces,
                               hidden_size, SPHERE_DIM, device)
    snap_rows(last_pf, sph0, 0)

    # ------------------------------------------------------------- decode --
    positions_step = torch.empty(batch, dtype=torch.int64, device=device)
    steps_t = torch.empty(batch, dtype=torch.int64, device=device)
    ids_step = torch.zeros(batch, dtype=torch.int64, device=device)
    prof = [0.0, 0.0, 0.0, 0.0] if _PROFILE else None  # meta/embeds/fwd/snap
    for t in range(1, max_tokens + 1):
        if prof is not None:
            torch.cuda.synchronize(); _t0 = time.perf_counter()
        steps_t.fill_(t)
        positions_step.fill_(seq_len + t - 1)
        embeds_t = generate_decode_inputs_gpu(base_seeds, prev_k, steps_t,
                                              hidden_size, device).view(
                                                  batch, hidden_size).to(dtype)
        if prof is not None:
            torch.cuda.synchronize(); _t1 = time.perf_counter()
            prof[1] += _t1 - _t0; _t0 = _t1
        attn_t, slots_t = _create_v1_attn_metadata(
            batch, 1, device, worker, positions_step,
            borrowed_block_ids=borrowed_block_ids,
            borrowed_stripe=borrowed_stripe,
            alloc_len=alloc_len, decode_step=seq_len + t - 1)
        if prof is not None:
            torch.cuda.synchronize(); _t1 = time.perf_counter()
            prof[0] += _t1 - _t0; _t0 = _t1
        state.set_rows(block_hash, batch)
        state.set_routing(block_hash, nonces, 1, t)
        state.set_embeds(embeds_t)
        with set_forward_context(attn_t, vllm_config, num_tokens=batch,
                                 slot_mapping=slots_t,
                                 skip_compiled=_SKIP_COMPILED):
            h = model(input_ids=ids_step, positions=positions_step,
                      intermediate_tensors=None, inputs_embeds=None)
        if isinstance(h, tuple):
            h = h[0]
        if prof is not None:
            torch.cuda.synchronize(); _t1 = time.perf_counter()
            prof[2] += _t1 - _t0; _t0 = _t1
        sph_t = random_pick_indices_gpu(base_seeds, prev_k, steps_t,
                                        hidden_size, SPHERE_DIM, device)
        snap_rows(h.view(batch, -1), sph_t, t)
        if prof is not None:
            torch.cuda.synchronize()
            prof[3] += time.perf_counter() - _t0
    if prof is not None:
        tot = sum(prof) or 1.0
        logger.info(
            "PoC decode profile (%d steps, batch %d): metadata %.1fms/step "
            "(%.0f%%), embeds %.1fms (%.0f%%), forward %.1fms (%.0f%%), "
            "snap %.1fms (%.0f%%)", max_tokens, batch,
            1e3*prof[0]/max_tokens, 100*prof[0]/tot,
            1e3*prof[1]/max_tokens, 100*prof[1]/tot,
            1e3*prof[2]/max_tokens, 100*prof[2]/tot,
            1e3*prof[3]/max_tokens, 100*prof[3]/tot)

    state.clear()

    if tp_group.rank_in_group != 0:
        return None

    # -------------------------------------------------------------- emit --
    from .data import encode_vector
    k_host = k_steps.cpu().tolist()
    nan_host = n_nan.cpu().tolist()
    mism_host = mismatch.cpu().tolist()
    artifacts = []
    for i, nonce in enumerate(nonces):
        art = {
            "nonce": nonce,
            "vector_b64": "",
            "k_points_steps": k_host[i],
            "n_sphere_mismatches": mism_host[i] if validating else -1,
            "n_nan_steps": nan_host[i],
        }
        if q_kept:
            art["sph_values_steps"] = [
                encode_vector(q_kept[s][i].numpy()) if s in q_kept else ""
                for s in range(max_tokens + 1)
            ]
        artifacts.append(art)
    return {"artifacts": artifacts,
            "steps_total": batch * (max_tokens + 1),
            "mismatch_total": int(sum(mism_host)) if validating else -1}
