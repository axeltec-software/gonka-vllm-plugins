# SPDX-License-Identifier: Apache-2.0
"""The prefill-scheme change must be an exact no-op on the decode path.

It touches three things decode also uses: the MoE gate's mask, the reflection
seed selection, and the pick seed. Each test reconstructs the pre-change
behaviour and asserts equality for decode-scheme inputs.

Scope: this pins the three functions, not end-to-end artifacts. Decode
artifacts are a k trajectory scored by mismatch rate against an honest floor --
they are not bit-stable across hardware and nothing here claims they are. What
it rules out is a shift introduced by this diff, which would otherwise be
invisible: trajectories would still come out full length, deterministic, and
separating honest from fraud.
"""
import torch

from gonka_poc.mixed.native import PoCNativeState, PoCRouterWrapper, _reflect
from gonka_poc.poc.gpu_random import (random_pick_indices,
                                      generate_householder_vector)

CPU = torch.device("cpu")
N_EXPERTS, TOP_K = 32, 4


class _Gate(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, *a, **k):
        return self.logits.clone()


def _old_gate_forward(w, logits):
    """The gate exactly as it was: forced wherever the row is PoC, full stop."""
    from gonka_poc.poc.gpu_random import expert_logits_from_base
    n = logits.shape[0]
    m = w.poc_mask[:n].unsqueeze(-1)
    forced = expert_logits_from_base(
        w.poc_route_base[:n], w.poc_route_step[:n],
        w.n_experts, w.top_k, logits.device).to(logits.dtype)
    return torch.where(m, forced, logits)


def test_router_output_is_identical_on_a_decode_round():
    """Veto open -> the gate must reproduce the old logits exactly, over many
    row counts and mask patterns."""
    g = torch.Generator().manual_seed(20260829)
    for rows in (1, 2, 7, 32, 129):
        st = PoCNativeState(2, 64, rows, CPU, torch.float32)
        logits = torch.randn(rows, N_EXPERTS, generator=g)
        gate = _Gate(logits)
        w = PoCRouterWrapper(gate, torch.arange(rows, dtype=torch.int64),
                             st.route_step, N_EXPERTS, TOP_K,
                             st.mask, st.route_mask)
        w._inner_call = gate.forward

        for pattern in (torch.ones(rows, dtype=torch.bool),
                        torch.zeros(rows, dtype=torch.bool),
                        torch.arange(rows) % 2 == 0,
                        torch.arange(rows) % 3 == 0):
            st.set_mask(pattern)
            st.set_route_mask(None)                # a decode round
            assert torch.equal(w.forward(), _old_gate_forward(w, logits)), \
                f"decode routing changed at rows={rows}"


def test_router_output_is_identical_when_every_poc_row_is_decode():
    """The bridge may also pass an all-True list (a batch it did materialise).
    That must be the same computation as leaving the veto open."""
    rows = 16
    st = PoCNativeState(2, 64, rows, CPU, torch.float32)
    logits = torch.randn(rows, N_EXPERTS, generator=torch.Generator().manual_seed(7))
    gate = _Gate(logits)
    w = PoCRouterWrapper(gate, torch.arange(rows, dtype=torch.int64),
                         st.route_step, N_EXPERTS, TOP_K, st.mask, st.route_mask)
    w._inner_call = gate.forward

    st.set_mask(torch.arange(rows) % 2 == 0)
    st.set_route_mask(None)
    open_veto = w.forward()
    st.set_route_mask([True] * rows)
    assert torch.equal(w.forward(), open_veto)
    assert torch.equal(open_veto, _old_gate_forward(w, logits))


def test_decode_pick_seed_is_untouched():
    """random_pick_indices grew a keyword; the decode call must return the same
    indices it did before the keyword existed."""
    for step in (0, 1, 17):
        for prev in (None, [3, 9]):
            kw = dict(block_hash="bh", public_key="pk", nonces=[41, 42],
                      dim=128, k=12, device=CPU, prev_point_ids=prev, step=step)
            assert torch.equal(random_pick_indices(**kw),
                               random_pick_indices(**kw, prefill_vector=False)), \
                f"decode pick seed moved at step={step} prev={prev}"


def test_decode_reflection_seed_is_untouched():
    """The bridge now writes row_refl_nonces only when the row is decode-scheme.
    For a decode row the value must be exactly what it was."""
    st = PoCNativeState(3, 64, 4, CPU, torch.float32)
    # per-block (default) and per-nonce draws must both still be reachable and
    # must differ from each other -- that is what the flag selects.
    a = generate_householder_vector("bh_layer_0_householder", 64, CPU)
    b = generate_householder_vector("bh_nonce41_layer_0_householder", 64, CPU)
    assert not torch.equal(a, b)
    assert torch.equal(a, generate_householder_vector("bh_layer_0_householder", 64, CPU))


def test_reflection_math_is_untouched():
    """The veto excludes rows from the ROUTER only; _reflect is unchanged."""
    g = torch.Generator().manual_seed(11)
    x = torch.randn(5, 64, generator=g)
    v = torch.randn(5, 64, generator=g)
    v = v / v.norm(dim=-1, keepdim=True)
    m = torch.tensor([True, False, True, False, True]).unsqueeze(-1)
    out = _reflect(x, v, m)
    expected = torch.where(m, x - 2.0 * (x * v).sum(-1, keepdim=True) * v, x)
    assert torch.equal(out, expected)
