"""In-model PoC transforms — module WRAPPERS, never forward hooks.

Ported from the 0.20 in-tree branch (``vllm/poc/native.py`` @ 5c1d09f55e92)
into the plugin, trimmed to what the plugin-side decode loop actually needs:

  * decoder-layer forward MONKEYPATCH — per-layer Householder reflection of
    hidden+residual on PoC rows (identity for non-PoC rows via the mask);
  * MoE gate forward MONKEYPATCH — REPLACES router logits with deterministic
    seeded logits on PoC rows (consensus: routing must not read the
    noise-prone hidden state);
  * ``PoCNativeState``   — address-stable buffers (reflection vectors, row
    mask, per-layer route bases, shared step buffer) updated IN PLACE, so a
    captured CUDA graph reads live values (migration rule: eager is not an
    execution mode; the step function must be capture-ready from day one).

Dropped relative to 0.20 (inventory DROP verdicts, roles covered elsewhere):
``PoCEmbeddingWrapper`` (the loop feeds ``inputs_embeds`` directly),
``PoCSnapWrapper`` (the loop snaps the returned hidden inside its own step
function), TP-rank divergence assertion (single-driver RPC path).

CONSENSUS: the arithmetic here (reflection formula, seeded-logit selection,
seed strings via ``gpu_random``) defines k-trajectories. Any change needs a
coordinated re-collection.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
from torch import nn

from .gpu_random import (
    expert_logits_from_base,
    generate_householder_vector,
    _seed_from_string,
    route_base_seed,
)

logger = logging.getLogger(__name__)


def _reflect(x: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked Householder: rows where mask is True -> x - 2*(x·v)*v; else x.

    0.20: native.py:60-66. Per-row independent, static-shape (no data-dependent
    control flow) — capture-safe.
    """
    dot = (x * v).sum(-1, keepdim=True)
    transformed = x - 2.0 * dot * v
    return torch.where(mask, transformed, x)


class PoCNativeState:
    """Address-stable per-model transform state (0.20: native.py:229-…).

    max_rows sizes every buffer: the largest token-row count a PoC forward can
    carry (prefill chunk: nonces*seq_len; decode step: nonces). Buffers are
    updated in place; a captured graph reads live values.
    """

    def __init__(self, num_layers: int, hidden_size: int, max_rows: int,
                 device: torch.device, dtype: torch.dtype):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.max_rows = max_rows
        self.device = device
        # Reflection vectors are BROADCAST [1, hidden]: one chunk carries one
        # block_hash, so a per-token-row [rows, hidden] buffer (12+ GiB at the
        # 128-nonce prefill chunk) is pure waste. per_nonce_reflection would
        # need per-row vectors — deferred (not in the golden scope), guarded
        # in set_rows.
        self.vectors: List[torch.Tensor] = [
            torch.zeros(1, hidden_size, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.mask = torch.zeros(max_rows, 1, device=device, dtype=torch.bool)
        self.route_step = torch.zeros(max_rows, dtype=torch.int64, device=device)
        self._route_base: List[torch.Tensor] = []
        self.router_meta: List[tuple] = []
        self._hh_cache: dict = {}
        self._rows_key = None
        self._route_key = None

    # -- reflection vectors ------------------------------------------------
    def _hh_vectors(self, block_hash: str, nonce: Optional[int], dtype):
        key = (block_hash, nonce)
        vs = self._hh_cache.get(key)
        if vs is None:
            if len(self._hh_cache) > 64:
                self._hh_cache.clear()
            suffix = "" if nonce is None else f"_nonce{nonce}"
            vs = [
                generate_householder_vector(
                    f"{block_hash}{suffix}_layer_{i}_householder",
                    self.hidden_size, self.device).to(dtype)
                for i in range(self.num_layers)
            ]
            self._hh_cache[key] = vs
        return vs

    def set_rows(self, block_hash: Optional[str], n_rows: int,
                 refl_nonce: Optional[int] = None,
                 per_nonce: bool = False) -> None:
        """Broadcast reflection vectors for ONE block_hash + mask first n_rows.

        block_hash None -> full identity (mask off). per_nonce reflection needs
        per-row vectors — not implemented in this revision (golden scope has it
        off); fail loud rather than silently mis-derive.
        """
        if per_nonce:
            raise NotImplementedError(
                "per_nonce_reflection needs per-row reflection buffers; "
                "excluded from this revision (golden scope: off)")
        key = (block_hash, refl_nonce, n_rows)
        if key == self._rows_key:
            return
        if block_hash is None:
            self.mask.zero_()
        else:
            self.mask[:n_rows].fill_(True)
            self.mask[n_rows:].zero_()
            dtype = self.vectors[0].dtype
            vs = self._hh_vectors(block_hash, refl_nonce, dtype)
            for li in range(self.num_layers):
                self.vectors[li].copy_(vs[li].unsqueeze(0))
        self._rows_key = key

    # -- seeded routing ----------------------------------------------------
    def set_routing(self, block_hash: str, nonces: List[int],
                    tokens_per_nonce: int, step: int) -> None:
        """Refresh seeded-router state for one chunk.

        Row layout: nonce-major, ``tokens_per_nonce`` consecutive rows per
        nonce (prefill: seq_len rows/nonce, decode step: 1 row/nonce). Base
        seed sha256 is hashed once per (hash, nonce, layer) and expanded on
        device; the step folds in on-GPU inside the wrapper.
        """
        if not self._route_base:
            return
        n = len(nonces) * tokens_per_nonce
        base_key = (block_hash, tuple(nonces), tokens_per_nonce)
        if base_key != self._route_key:
            for li, buf in enumerate(self._route_base):
                vals = torch.tensor(
                    [_seed_from_string(route_base_seed(block_hash, nz, li))
                     for nz in nonces],
                    dtype=torch.int64, device=self.device)
                buf[:n].copy_(vals.repeat_interleave(tokens_per_nonce))
            self._route_key = base_key
        self.route_step[:n].fill_(int(step))

    def clear(self) -> None:
        """Identity for everything (defensive; engine paths never see us)."""
        self.mask.zero_()
        self._rows_key = None


def _find_decoder_layers(model: nn.Module) -> nn.Module:
    """Locate the decoder layer owner generically: the module whose ``.layers``
    ModuleList is the LONGEST one in the tree (the transformer stack). The
    previous "deepest" heuristic could latch onto a nested short list (seen on
    MiniMax-M2.7: 28 inner modules wrapped instead of the 62 decoder layers,
    which also left every MoE gate undiscovered)."""
    best, best_name = None, None
    for name, m in model.named_modules():
        layers = getattr(m, "layers", None)
        if isinstance(layers, nn.ModuleList) and (
                best is None or len(layers) > len(best.layers)):
            best, best_name = m, name
    if best is None:
        raise RuntimeError("PoC native: no decoder .layers ModuleList found")
    logger.info("PoC native: decoder stack at '%s' (%d layers)",
                best_name or "<root>", len(best.layers))
    return best


def attach_native_poc(model: nn.Module, hidden_size: int, max_rows: int,
                      device, dtype, route_window: int) -> "PoCNativeState":
    """Bake PoC transforms by MONKEYPATCHING the bound ``forward`` of each
    decoder layer and each MoE gate — the modules, parameter names and the
    ``@support_torch_compile`` signature inspection stay intact (wrapping in a
    new nn.Module renamed params to ``.inner.*`` and broke both weight loading
    and compile). The patched forward reads live state buffers, so it captures
    into the compiled graph exactly like the 0.20 attach-before-compile path.
    Idempotent; chat rows (mask False) are exact identity.
    """
    from .gpu_random import set_route_window
    set_route_window(route_window)

    if getattr(model, "_poc_native_state", None) is not None:
        return model._poc_native_state

    owner = _find_decoder_layers(model)
    layers = list(owner.layers)
    state = PoCNativeState(len(layers), hidden_size, max_rows, device, dtype)

    for li, layer in enumerate(layers):
        _patch_layer_forward(layer, state, li)
        moe = _find_layer_moe(layer)
        if moe is None:
            continue
        n_exp = int(moe.experts.global_num_experts)
        top_k = int(moe.experts.top_k)
        route_base = torch.zeros(max_rows, dtype=torch.int64, device=device)
        state._route_base.append(route_base)
        state.router_meta.append((n_exp, top_k))
        _patch_gate_forward(moe, state, route_base, n_exp, top_k)

    logger.info("PoC native attached: %d layers patched, %d MoE gates seeded, "
                "route window %d", len(layers), len(state.router_meta),
                route_window)
    model._poc_native_state = state
    return state


def _find_layer_moe(layer: nn.Module):
    return next(
        (m for m in layer.modules()
         if hasattr(m, "gate") and hasattr(m, "experts")
         and hasattr(getattr(m, "experts"), "top_k")
         and not getattr(m, "_poc_gate_patched", False)),
        None)


def _patch_layer_forward(layer: nn.Module, state: "PoCNativeState", li: int):
    """Reflect the layer's (hidden, residual) output on PoC rows, in place."""
    orig = layer.forward

    def forward(*args, **kwargs):
        out = orig(*args, **kwargs)
        vbuf, mask = state.vectors[li], state.mask
        if isinstance(out, tuple) and len(out) == 2:
            hidden, residual = out
            n = hidden.shape[0]
            hidden = _reflect(hidden, vbuf, mask[:n])
            if residual is not None:
                residual = _reflect(residual, vbuf, mask[:n])
            return hidden, residual
        hidden = out[0] if isinstance(out, tuple) else out
        n = hidden.shape[0]
        hidden = _reflect(hidden, vbuf, mask[:n])
        return (hidden, *out[1:]) if isinstance(out, tuple) else hidden

    layer.forward = forward


def _patch_gate_forward(moe: nn.Module, state: "PoCNativeState",
                        route_base: torch.Tensor, n_experts: int, top_k: int):
    """Override the MoE gate's logits with seeded logits on PoC rows."""
    gate = moe.gate
    orig = gate.forward
    step_buf = state.route_step
    mask = state.mask

    def forward(*args, **kwargs):
        out = orig(*args, **kwargs)
        logits = out[0] if isinstance(out, tuple) else out
        n = logits.shape[0]
        forced = expert_logits_from_base(
            route_base[:n], step_buf[:n], n_experts, top_k,
            logits.device).to(logits.dtype)
        logits = torch.where(mask[:n], forced, logits)
        return (logits, *out[1:]) if isinstance(out, tuple) else logits

    gate.forward = forward
    moe._poc_gate_patched = True
