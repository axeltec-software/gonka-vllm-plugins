# SPDX-License-Identifier: Apache-2.0
"""The prefill scheme must derive exactly as the shipped MLNode image does.

Verified against the 3.0.16 bundle (gonka_poc 0.1.3): generate_inputs,
apply_haar_rotation and generate_householder_vector already match bit for bit;
random_pick_indices was the single divergence, because the decode chain salts
its seed with the step. Prefill must not carry that salt — a chain that asks
for the prefill proof gets the artifact the fleet validates today.

The reference formula is spelled out here rather than imported, so this runs
everywhere instead of skipping like the cross-version parity suite.
"""
import pytest
import torch

from gonka_poc.poc import gpu_random as gr

BH, PK, DEV = "block-parity", "pk-parity", torch.device("cpu")


def _v01x_pick(nonce, dim, k):
    """gonka_poc 0.1.3: seed string carries no decode salt."""
    seed = gr._seed_from_string(f"{BH}_{PK}_nonce_{nonce}_pick_{k}")
    idx = torch.arange(dim, dtype=torch.int32).unsqueeze(0)
    scores = gr._batched_murmur3_32(
        idx, torch.tensor([seed], dtype=torch.int64).unsqueeze(1))
    return torch.topk(-scores, k=k, largest=True, sorted=False, dim=1).indices[0]


@pytest.mark.parametrize("nonce", [0, 7, 4242])
def test_prefill_pick_matches_the_shipped_image(nonce):
    got = gr.random_pick_indices(BH, PK, [nonce], 512, 12, DEV,
                                 prefill_vector=True)[0]
    assert torch.equal(torch.sort(got).values,
                       torch.sort(_v01x_pick(nonce, 512, 12)).values)


def test_decode_pick_keeps_its_own_salt():
    """The decode chain is a different derivation and must stay one, step 0
    included, or its golden trajectory moves."""
    decode = gr.random_pick_indices(BH, PK, [0], 512, 12, DEV)[0]
    assert not torch.equal(torch.sort(decode).values,
                           torch.sort(_v01x_pick(0, 512, 12)).values)
