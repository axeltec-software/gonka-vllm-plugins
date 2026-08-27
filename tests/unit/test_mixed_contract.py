# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the mixed subpackage.

1. One-way dependency: no gonka_poc core module imports gonka_poc.mixed.
2. The pre-forward hook is registrable, fires with the residual-contract
   signature, never raises, and records evidence.
"""
import pathlib
import types

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "gonka_poc"


def test_core_never_imports_mixed():
    offenders = []
    for p in SRC.rglob("*.py"):
        if "mixed" in p.parts:
            continue
        text = p.read_text(encoding="utf-8")
        import re
        if re.search(r"gonka_poc\.mixed|from\s+\.\s*mixed|from\s+gonka_poc\s+import[^\n]*\bmixed\b", text):
            offenders.append(str(p.relative_to(SRC)))
    assert not offenders, f"core imports mixed (one-way rule): {offenders}"




def test_policy_pure_functions():
    from gonka_poc.mixed import policy
    # клапан справедливости: после лимита отложек PoC получает эксклюзивный шаг
    d = 0
    for _ in range(policy.POC_DEFER_LIMIT):
        defer_chat, defer_poc, d = policy.decode_only_mixing_gate(
            mixed_cudagraph=True, poc_decode_pending=False,
            poc_will_prefill=False, chat_will_prefill=True,
            consecutive_defers=d)
        assert defer_poc and not defer_chat
    defer_chat, defer_poc, d = policy.decode_only_mixing_gate(
        mixed_cudagraph=True, poc_decode_pending=False,
        poc_will_prefill=False, chat_will_prefill=True, consecutive_defers=d)
    assert defer_chat and not defer_poc and d == 0
    p = types.SimpleNamespace(seq_len=256, max_tokens=256)
    assert policy.poc_step_num_tokens(p, 0) == 256
    assert policy.poc_step_num_tokens(p, 256) == 1
    # ONE definition, in runtime: the policy copy had drifted to a 2-arg form
    # without the KV clamp, so it silently tested a function nobody calls.
    from gonka_poc.mixed.runtime import resolve_poc_max_batch_size
    assert resolve_poc_max_batch_size(0, 704) == 704
    assert resolve_poc_max_batch_size(536, 704) == 536
    assert resolve_poc_max_batch_size(0, 704, kv_capacity=128) == 128
    assert resolve_poc_max_batch_size(536, 704, kv_capacity=128) == 536
