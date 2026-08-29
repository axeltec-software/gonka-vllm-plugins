# SPDX-License-Identifier: Apache-2.0
"""Both schemes reach the engine the same way: one request per nonce.

The prefill scheme used to leave the serving pipeline entirely
(``collective_rpc`` -> a hand-built forward over KV blocks 0..N, with every
in-flight request aborted first). It now rides the scheduler like any other
request, so it mixes with live chat instead of stopping it.

What separates the two schemes is per-row state inside the forward, not the
path: ``poc_decode`` carries the scheme to the runner, which vetoes the seeded
MoE router and drops the decode salt from the pick seed for a prefill row
(see mixed_test_prefill_scheme_routing).
"""
import asyncio
from types import SimpleNamespace

import pytest

from gonka_poc.poc.generate_queue import compute_nonce_artifacts


class _Engine:
    """Records what the queue submits, and replies like a finished PoC row."""

    def __init__(self, vector="dmVjdG9y", k_points=()):
        self.calls = []
        self._vector = vector
        self._k_points = list(k_points)

    def generate(self, *, prompt, sampling_params, poc_params, request_id,
                 priority):
        self.calls.append(poc_params)

        async def _stream():
            yield SimpleNamespace(
                finished=True,
                poc_output={"nonce": poc_params.nonce,
                            "vector_b64": self._vector,
                            "k_points_steps": self._k_points},
            )
        return _stream()


def _run(engine, **kw):
    return asyncio.run(compute_nonce_artifacts(
        engine, [41, 42], "bh", "pk", 7, 256, 12, **kw))


def test_prefill_goes_through_the_engine_not_collective_rpc():
    """Two nonces, prefill scheme: two engine requests, no RPC, no abort."""
    eng = _Engine()
    arts = _run(eng, poc_decode=False, max_tokens=0)

    assert [a["nonce"] for a in arts] == [41, 42]
    assert [a["vector_b64"] for a in arts] == ["dmVjdG9y"] * 2
    assert len(eng.calls) == 2, "one engine request per nonce"


def test_the_scheme_reaches_the_runner_on_the_request():
    """poc_decode is what the bridge reads to veto the router, so it has to be
    carried per request rather than inferred from the phase or a launch flag."""
    eng = _Engine()
    _run(eng, poc_decode=False, max_tokens=0)
    assert all(p.poc_decode is False for p in eng.calls)
    assert all(p.max_tokens == 0 for p in eng.calls)

    eng = _Engine(vector="", k_points=[3, 9, 4])
    _run(eng, poc_decode=True, max_tokens=32)
    assert all(p.poc_decode is True for p in eng.calls)
    assert all(p.max_tokens == 32 for p in eng.calls)


def test_both_schemes_carry_the_same_nonce_identity():
    """block_hash / public_key / nonce are the derivation's seed material; the
    path change must not perturb what the runner receives."""
    eng = _Engine()
    _run(eng, poc_decode=False, max_tokens=0)
    assert [(p.block_hash, p.public_key, p.nonce, p.seq_len, p.k_dim)
            for p in eng.calls] == [("bh", "pk", 41, 256, 12),
                                    ("bh", "pk", 42, 256, 12)]


def test_a_decode_request_returns_its_trajectory():
    eng = _Engine(vector="", k_points=[3, 9, 4])
    arts = _run(eng, poc_decode=True, max_tokens=32)
    assert [a["k_points_steps"] for a in arts] == [[3, 9, 4]] * 2


def test_a_nonce_that_yields_no_artifact_is_dropped_not_faked():
    class _Empty(_Engine):
        def generate(self, **kw):
            self.calls.append(kw["poc_params"])

            async def _stream():
                yield SimpleNamespace(finished=True, poc_output=None)
            return _stream()

    eng = _Empty()
    assert _run(eng, poc_decode=False, max_tokens=0) == []
    assert len(eng.calls) == 2
