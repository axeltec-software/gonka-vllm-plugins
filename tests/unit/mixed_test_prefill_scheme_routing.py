# SPDX-License-Identifier: Apache-2.0
"""The prefill scheme must run on the model's NATURAL MoE router.

Decode forces a seeded router because the sphere snap turns a ULP-level expert
flip into a flipped k that then chains (MiniMax-M2.7 honest floor 18.4% ->
1.4%, KB seeded-routing). Prefill has no snap and no chain, and the deployed
fleet validates it on an untouched router.

One mask cannot express that: a prefill row IS a PoC row (the reflection
applies) but keeps the natural router. ``route_mask`` vetoes ``mask`` -- the
gate forces only where both are set -- and defaults to all-True, so a pure
decode round never writes it.

Keyed on the SCHEME, never the phase: a decode request's own prefill step snaps
k0 and must stay seeded.
"""
import torch
from types import SimpleNamespace

import pytest

from gonka_poc.mixed import bridge as bridge_mod
from gonka_poc.mixed.native import PoCNativeState, PoCRouterWrapper
from gonka_poc.poc.gpu_random import random_pick_indices

CPU = torch.device("cpu")


def _state(max_tokens=8, layers=2, hidden=16):
    return PoCNativeState(layers, hidden, max_tokens, CPU, torch.float32)


# ------------------------------------------------------------- the veto buffer

def test_the_veto_defaults_open_so_a_decode_round_never_writes_it():
    """A normal round leaves the buffer untouched; `mask` alone decides."""
    st = _state()
    assert st.route_mask.all(), "veto must start open"
    assert st._last_force is None
    st.set_mask(torch.tensor([True, True, False, False]))
    assert st.route_mask.all(), "set_mask must not touch the veto"


def test_set_route_mask_none_is_the_decode_round_and_restores_the_veto():
    st = _state()
    st.set_route_mask([False, True, True, True])
    assert not st.route_mask[0].item()
    st.set_route_mask(None)                       # next round is pure decode
    assert st.route_mask.all()
    assert st._last_force is None


def test_set_route_mask_vetoes_only_the_named_rows():
    st = _state()
    st.set_mask(torch.tensor([True, True, True]))
    st.set_route_mask([False, True, False])       # prefill, decode, prefill
    assert st.route_mask[:3].tolist() == [False, True, False]


def test_the_scatter_is_memoized_but_never_stale():
    """pinned_to_device page-locks a fresh host buffer; rebuilding it every
    step measured 12-56us, about the cost of the whole seeded-routing refresh.
    Skip when the mapping is unchanged -- and never when it changes."""
    st = _state(max_tokens=4)
    st.set_route_mask([False, True, True])
    assert st.route_mask[:3].tolist() == [False, True, True]

    st.route_mask[1] = False                      # poison; a real rescatter repairs it
    st.set_route_mask([False, True, True])        # same mapping -> skipped
    assert st.route_mask[1].item() is False, "the memo did not skip"

    st.set_route_mask([True, False, True])        # changed -> rescatter from all-True
    assert st.route_mask[:3].tolist() == [True, False, True]


def test_a_shorter_mapping_does_not_leave_a_veto_behind():
    """Rows past the new mapping must go back to open, not keep an old False."""
    st = _state(max_tokens=8)
    st.set_route_mask([False, False, False, False])
    st.set_route_mask([False])
    assert st.route_mask[1:].all(), "a stale veto survived a shorter batch"


# ------------------------------------------------------- the router seam itself

class _Gate(torch.nn.Module):
    """Stands in for an MoE gate: returns fixed, recognisable logits."""

    def __init__(self, n_rows, n_experts):
        super().__init__()
        self.natural = torch.arange(
            n_rows * n_experts, dtype=torch.float32).view(n_rows, n_experts)

    def forward(self, *a, **k):
        return self.natural.clone()


def _wrap(st, gate, n_experts, n_rows):
    w = PoCRouterWrapper(gate, torch.arange(n_rows, dtype=torch.int64),
                         st.route_step, n_experts, 2, st.mask, st.route_mask)
    w._inner_call = gate.forward
    return w


def test_router_forces_decode_rows_only():
    """Two nonces in one batch: one prefill-scheme, one decode-scheme, plus a
    chat row. Only the decode row may leave with replaced logits."""
    n_experts, n_rows = 8, 3
    st = _state(max_tokens=n_rows)
    gate = _Gate(n_rows, n_experts)
    w = _wrap(st, gate, n_experts, n_rows)

    # rows: 0 = chat, 1 = PoC prefill scheme, 2 = PoC decode scheme
    st.set_mask(torch.tensor([False, True, True]))
    st.set_route_mask([False, False, True])

    out, natural = w.forward(), gate.natural
    assert torch.equal(out[0], natural[0]), "chat row was routed"
    assert torch.equal(out[1], natural[1]), \
        "prefill-scheme row had its MoE router forced -- artifacts diverge from 3.0.16"
    assert not torch.equal(out[2], natural[2]), \
        "decode-scheme row was NOT forced -- k0 is exposed to expert flips"


def test_the_open_veto_reproduces_the_previous_behaviour():
    """Veto untouched, so `mask` alone decides: what the gate did before."""
    n_experts, n_rows = 8, 3
    st = _state(max_tokens=n_rows)
    gate = _Gate(n_rows, n_experts)
    w = _wrap(st, gate, n_experts, n_rows)

    st.set_mask(torch.tensor([False, True, True]))
    st.set_route_mask(None)

    out, natural = w.forward(), gate.natural
    assert torch.equal(out[0], natural[0])
    assert not torch.equal(out[1], natural[1])
    assert not torch.equal(out[2], natural[2])


def test_a_chat_only_step_forces_nothing_even_with_the_veto_open():
    """set_mask(None) leaves the veto open; `mask` excludes the chat rows."""
    n_experts, n_rows = 8, 4
    st = _state(max_tokens=n_rows)
    gate = _Gate(n_rows, n_experts)
    w = _wrap(st, gate, n_experts, n_rows)

    st.set_mask(None)
    assert st.route_mask.all()
    assert torch.equal(w.forward(), gate.natural)


def test_a_prefill_only_round_forces_nothing():
    n_experts, n_rows = 8, 4
    st = _state(max_tokens=n_rows)
    gate = _Gate(n_rows, n_experts)
    w = _wrap(st, gate, n_experts, n_rows)

    st.set_mask(torch.tensor([True, True, True, True]))
    st.set_route_mask([False, False, False, False])
    assert torch.equal(w.forward(), gate.natural)


# ------------------------------------------------------------- the pick seed

def test_prefill_pick_seed_drops_the_decode_salt():
    """The one place the two derivations disagree on shared math."""
    kw = dict(block_hash="bh", public_key="pk", nonces=[41, 42],
              dim=64, k=12, device=CPU)
    pre = random_pick_indices(**kw, prefill_vector=True)
    dec = random_pick_indices(**kw, prefill_vector=False)
    assert pre.shape == dec.shape == (2, 12)
    assert not torch.equal(pre, dec), \
        "prefill_vector did not change the seed -- v0.1.x compat is not restored"
    assert torch.equal(pre, random_pick_indices(**kw, prefill_vector=True))
    assert not torch.equal(pre[0], pre[1])


# ------------------------------------------- the bridge builds the veto per row

class _Native:
    def __init__(self):
        self.route = "unset"
        self.refl = None

    def set_embeds(self, e): pass
    def set_mask(self, m): pass
    def set_row_block_hashes(self, hashes, refl): self.refl = refl
    def set_route_mask(self, force): self.route = force
    def set_routing(self, h, n, s): pass


def _params(nonce, poc_decode, per_nonce_reflection=False):
    return SimpleNamespace(block_hash="bh", nonce=nonce, poc_decode=poc_decode,
                           per_nonce_reflection=per_nonce_reflection)


def _run_bridge(monkeypatch, metadata, n_rows):
    br = bridge_mod.PoCRunnerBridge.__new__(bridge_mod.PoCRunnerBridge)
    br.runner = SimpleNamespace()
    br.native = _Native()
    br._step = {"poc_req_ids": ["a", "b"]}
    br._reqs = {}
    monkeypatch.setattr(
        bridge_mod.mixed_decode, "build_unified_mixed_batch_inputs",
        lambda *a, **k: (torch.zeros(n_rows, 4), None, None, metadata))
    br.pre_forward(SimpleNamespace(), torch.zeros(n_rows), n_rows,
                   batch_view=(0, []))
    return br.native


def test_bridge_builds_no_list_for_a_pure_decode_round(monkeypatch):
    """Every round today: no per-row list built, no scatter."""
    native = _run_bridge(monkeypatch, [
        {"start_idx": 0, "length": 1, "poc_params": _params(41, True)},
        {"start_idx": 1, "length": 1, "poc_params": _params(42, True)},
    ], 2)
    assert native.route is None, \
        "a pure decode round must not materialise a per-row veto list"


def test_bridge_vetoes_prefill_rows_when_the_batch_mixes_schemes(monkeypatch):
    """Two nonces in flight, one per scheme."""
    native = _run_bridge(monkeypatch, [
        {"start_idx": 0, "length": 2, "poc_params": _params(41, False)},
        {"start_idx": 2, "length": 1, "poc_params": _params(42, True)},
    ], 3)
    assert native.route == [False, False, True], \
        "the veto must follow the SCHEME, not the row count or the phase"


def test_bridge_ignores_per_nonce_reflection_on_the_prefill_scheme(monkeypatch):
    """The old chain cannot send this flag, so it must not reach a prefill row."""
    native = _run_bridge(monkeypatch, [
        {"start_idx": 0, "length": 1,
         "poc_params": _params(41, False, per_nonce_reflection=True)},
        {"start_idx": 1, "length": 1,
         "poc_params": _params(42, True, per_nonce_reflection=True)},
    ], 2)
    assert native.refl == [None, 42], \
        "prefill row must keep the per-block reflection seed"


# ------------------------------------------------------------------ on device

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_veto_on_device_and_stable_across_rearm():
    """Two things CPU cannot cover: the pinned bool H2D copy, and address
    stability across rounds (the captured graph reads this buffer live, so a
    reallocation would leave replay reading stale or freed memory)."""
    dev = torch.device("cuda")
    st = PoCNativeState(2, 32, 8, dev, torch.float16)
    assert st.route_mask.all()
    addr = st.route_mask.data_ptr()

    gate = _Gate(3, 8)
    gate.natural = gate.natural.to(dev).half()
    w = PoCRouterWrapper(gate, torch.arange(3, dtype=torch.int64, device=dev),
                         st.route_step, 8, 2, st.mask, st.route_mask)
    w._inner_call = gate.forward

    st.set_mask(torch.tensor([False, True, True], device=dev))
    st.set_route_mask([False, False, True])
    assert st.route_mask[:3].tolist() == [False, False, True]

    out = w.forward()
    assert torch.equal(out[0], gate.natural[0])           # chat
    assert torch.equal(out[1], gate.natural[1])           # prefill scheme
    assert not torch.equal(out[2], gate.natural[2])       # decode scheme

    st.set_route_mask(None)                               # back to a decode round
    assert st.route_mask.data_ptr() == addr, "route_mask was reallocated"
    assert st.route_mask.all()
    assert not torch.equal(w.forward()[1], gate.natural[1]), \
        "replay read a stale veto"


def test_bridge_reopens_the_veto_when_a_step_carries_no_poc_metadata(monkeypatch):
    """A closed veto must not survive into a later decode step."""
    native = _run_bridge(monkeypatch, [], 2)
    assert native.route is None, "a metadata-less step must reopen the veto"


def test_a_realistic_mixed_batch_maps_every_row(monkeypatch):
    """Chat rows interleaved with both schemes, prefill rows spanning tokens."""
    native = _run_bridge(monkeypatch, [
        {"start_idx": 2, "length": 4, "poc_params": _params(41, False)},
        {"start_idx": 6, "length": 1, "poc_params": _params(42, True)},
        {"start_idx": 7, "length": 4, "poc_params": _params(43, False)},
        {"start_idx": 11, "length": 1, "poc_params": _params(44, True)},
    ], 12)
    assert native.route == [
        False, False,                    # rows 0-1: chat, never forced
        False, False, False, False,      # nonce 41, prefill scheme
        True,                            # nonce 42, decode scheme
        False, False, False, False,      # nonce 43, prefill scheme
        True,                            # nonce 44, decode scheme
    ]


def test_the_veto_is_a_pure_function_of_the_row_schemes():
    """Determinism check (TP proxy, not a TP test): each rank builds this
    buffer from the same poc_params, so prior state must not leak into it."""
    st_a, st_b = _state(max_tokens=6), _state(max_tokens=6)
    force = [False, True, False, True, True, False]

    st_a.set_route_mask(force)
    # a rank that saw a different history must still land on the same buffer
    st_b.set_route_mask([True] * 6)
    st_b.set_route_mask(None)
    st_b.set_route_mask(force)

    assert torch.equal(st_a.route_mask, st_b.route_mask)


# --------------------------------------------- the invariant the whole PR rests on

def test_a_prefill_row_is_still_reflected_while_its_router_stays_natural():
    """Why a SECOND mask and not a narrower first one: a prefill row still
    gets the Householder reflection, only its router is vetoed. Dropping the
    reflection gives a meaningless artifact; seeding the router gives one the
    deployed fleet cannot reproduce."""
    from gonka_poc.mixed.native import PoCLayerWrapper

    rows, hidden, n_experts = 3, 16, 8
    st = _state(max_tokens=rows, layers=1, hidden=hidden)
    st.set_mask(torch.tensor([False, True, True]))   # chat, prefill, decode
    st.set_route_mask([False, False, True])

    # --- reflection: applies to BOTH PoC rows, not the chat row
    st.vectors[0][:rows] = torch.nn.functional.normalize(
        torch.randn(rows, hidden, generator=torch.Generator().manual_seed(3)), dim=-1)
    x = torch.randn(rows, hidden, generator=torch.Generator().manual_seed(4))
    inner = torch.nn.Identity()
    lw = PoCLayerWrapper(inner, st.vectors[0], st.mask)
    lw._inner_call = lambda *a, **k: x.clone()
    reflected = lw.forward()

    assert torch.equal(reflected[0], x[0]), "chat row was reflected"
    assert not torch.equal(reflected[1], x[1]), \
        "prefill-scheme row was NOT reflected -- its artifact is not a proof"
    assert not torch.equal(reflected[2], x[2]), "decode row was not reflected"

    # --- routing: vetoed on the prefill row only
    logits = torch.randn(rows, n_experts, generator=torch.Generator().manual_seed(5))
    gate = _Gate(rows, n_experts)
    gate.natural = logits
    rw = _wrap(st, gate, n_experts, rows)
    routed = rw.forward()

    assert torch.equal(routed[0], logits[0]), "chat row was routed"
    assert torch.equal(routed[1], logits[1]), \
        "prefill-scheme row was routed -- artifacts diverge from 3.0.16"
    assert not torch.equal(routed[2], logits[2]), "decode row was not routed"


def test_the_gate_tuple_return_shape_is_preserved():
    """Real MoE gates hand back a tuple; the veto must not change that path."""
    rows, n_experts = 4, 8
    st = _state(max_tokens=rows)
    logits = torch.randn(rows, n_experts, generator=torch.Generator().manual_seed(6))
    extra = torch.arange(rows)

    class _TupleGate(torch.nn.Module):
        def forward(self, *a, **k):
            return (logits.clone(), extra.clone())

    g = _TupleGate()
    w = PoCRouterWrapper(g, torch.arange(rows, dtype=torch.int64),
                         st.route_step, n_experts, 2, st.mask, st.route_mask)
    w._inner_call = g.forward

    st.set_mask(torch.tensor([False, True, True, True]))
    st.set_route_mask([False, False, True, True])
    out = w.forward()

    assert isinstance(out, tuple) and len(out) == 2
    assert torch.equal(out[1], extra), "the gate's extra outputs were disturbed"
    assert torch.equal(out[0][0], logits[0])          # chat
    assert torch.equal(out[0][1], logits[1])          # prefill scheme
    assert not torch.equal(out[0][2], logits[2])      # decode scheme


def test_a_mapping_longer_than_the_buffer_fails_loudly():
    """Truncating would leave tail rows seeded on the prefill scheme."""
    st = _state(max_tokens=4)
    with pytest.raises(ValueError, match="route veto"):
        st.set_route_mask([True] * 5)


def test_the_memo_stores_a_copy_not_the_caller_s_list():
    """A caller mutating its own list must not alias the memo."""
    st = _state(max_tokens=4)
    shared = [False, True, True]
    st.set_route_mask(shared)
    assert st.route_mask[:3].tolist() == [False, True, True]

    shared[0] = True                      # mutate the SAME object, then resubmit
    st.set_route_mask(shared)
    assert st.route_mask[:3].tolist() == [True, True, True], \
        "the memo aliased the caller's list and skipped a real change"


def test_a_full_step_sequence_keeps_both_buffers_consistent():
    """The sequence pre_forward runs, repeated, across a shape change."""
    st = _state(max_tokens=8, layers=2, hidden=16)
    st._route_base.append(torch.zeros(8, dtype=torch.int64))
    st.router_meta.append((8, 2))

    for _ in range(3):                                  # steady decode round
        st.set_mask(torch.tensor([False, True, True]))
        st.set_route_mask(None)
        st.set_routing(["bh", "bh", "bh"], [1, 2, 3], [0, 1, 1])
        assert st.route_mask.all()
        assert st.mask[:3].tolist() == [False, True, True]

    for _ in range(3):                                  # mixed round
        st.set_mask(torch.tensor([True, True]))
        st.set_route_mask([False, True])
        st.set_routing(["bh", "bh"], [4, 5], [0, 0])
        assert st.route_mask[:2].tolist() == [False, True]

    st.set_mask(None)                                   # chat-only step
    st.set_route_mask(None)
    assert not st.mask.any()
    assert st.route_mask.all()
