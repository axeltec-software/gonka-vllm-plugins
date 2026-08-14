# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the ported native wrappers (layer B plumbing).

A tiny fake model (decoder .layers with a .gate/.experts MoE inside) exercises
attach discovery, mask identity, the Householder reflection and the seeded
router override — no vLLM, no GPU.
"""
import torch
from torch import nn

from gonka_poc.poc import gpu_random
from gonka_poc.poc.native import attach_native_poc

H = 32


class FakeExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_num_experts = 16
        self.top_k = 2


class FakeGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(H, 16, bias=False)

    def forward(self, x):
        return self.lin(x), None      # GateLinear returns (logits, bias)


class FakeMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = FakeGate()
        self.experts = FakeExperts()

    def forward(self, x):
        logits, _ = self.gate(x)
        return logits


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = FakeMoE()

    def forward(self, hidden, residual=None):
        return hidden + 1.0, residual


class FakeCore(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList(FakeLayer() for _ in range(n_layers))


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeCore()


def test_attach_discovery_and_idempotence():
    m = FakeModel()
    st = attach_native_poc(m, H, max_rows=8, device=torch.device("cpu"),
                           dtype=torch.float32, route_window=256)
    assert len(st.router_meta) == 2 and st.router_meta[0] == (16, 2)
    assert all(getattr(l.moe, "_poc_gate_patched", False)
               for l in m.model.layers)
    again = attach_native_poc(m, H, 8, torch.device("cpu"),
                              torch.float32, 256)
    assert again is st  # idempotent


def test_layer_wrapper_identity_when_masked_off():
    m = FakeModel()
    st = attach_native_poc(m, H, 4, torch.device("cpu"), torch.float32, 256)
    st.clear()
    x = torch.randn(4, H)
    out, _ = m.model.layers[0](x, None)
    assert torch.equal(out, x + 1.0)  # exact identity path


def test_layer_wrapper_reflects_poc_rows():
    m = FakeModel()
    st = attach_native_poc(m, H, 4, torch.device("cpu"), torch.float32, 256)
    bh = "deadbeef" * 8
    st.set_rows(bh, 2)  # first 2 rows PoC
    x = torch.randn(4, H)
    out, _ = m.model.layers[0](x, None)
    base = x + 1.0
    v = gpu_random.generate_householder_vector(
        f"{bh}_layer_0_householder", H, torch.device("cpu"))
    expect0 = base[0] - 2.0 * (base[0] @ v) * v
    assert torch.allclose(out[0], expect0, atol=1e-6)
    assert torch.equal(out[2], base[2])          # row beyond n_rows untouched


def test_router_wrapper_forces_seeded_logits():
    m = FakeModel()
    st = attach_native_poc(m, H, 4, torch.device("cpu"), torch.float32, 256)
    bh = "deadbeef" * 8
    st.set_rows(bh, 4)
    st.set_routing(bh, [0, 1, 2, 3], 1, 5)
    x = torch.randn(4, H)
    logits = m.model.layers[0].moe(x)
    # expected: seeded logits for (bh, nonce, step=5, layer=0)
    for row, nonce in enumerate([0, 1, 2, 3]):
        exp = gpu_random.seeded_experts(bh, nonce, 5, 0, 16, 2,
                                        torch.device("cpu"))
        got = torch.topk(logits[row], 2).indices
        assert torch.equal(torch.sort(got).values,
                           torch.sort(exp).values), f"row {row}"


def test_per_nonce_reflection_guarded():
    m = FakeModel()
    st = attach_native_poc(m, H, 2, torch.device("cpu"), torch.float32, 256)
    import pytest as _pt
    with _pt.raises(NotImplementedError):
        st.set_rows("deadbeef" * 8, 2, per_nonce=True)
