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
        if "gonka_poc.mixed" in text or "from .mixed" in text:
            offenders.append(str(p.relative_to(SRC)))
    assert not offenders, f"core imports mixed (one-way rule): {offenders}"


def test_hook_installs_and_fires(monkeypatch):
    from gonka_poc.mixed import pre_forward

    runner = types.SimpleNamespace(pre_forward_hooks=[])
    monkeypatch.setenv("POC_MIXED_PRE_FORWARD", "1")
    assert pre_forward.install(runner) is True
    assert pre_forward.install(runner) is True  # idempotent
    assert runner.pre_forward_hooks.count(pre_forward.poc_pre_forward) == 1

    sched = types.SimpleNamespace(total_num_scheduled_tokens=536,
                                  num_scheduled_tokens={"r1": 1, "r2": 1})
    before = pre_forward.stats()["calls"]
    for hook in runner.pre_forward_hooks:
        hook(runner, sched, None, None, None, object())
    s = pre_forward.stats()
    assert s["calls"] == before + 1
    assert s["last_num_tokens"] == 536
    assert s["last_num_reqs"] == 2
    assert s["has_attn_metadata"] is True
    assert s["errors"] == 0


def test_hook_never_raises():
    from gonka_poc.mixed import pre_forward
    pre_forward.poc_pre_forward(None, object(), None, None, None, None)
    assert pre_forward.stats()["errors"] >= 0  # счётчик жив, исключения нет


def test_install_refuses_without_residual_seam(monkeypatch):
    from gonka_poc.mixed import pre_forward
    monkeypatch.setenv("POC_MIXED_PRE_FORWARD", "1")
    with pytest.raises(RuntimeError, match="pre_forward_hooks"):
        pre_forward.install(types.SimpleNamespace())


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
    assert policy.resolve_poc_max_batch_size(0, 704) == 704
    assert policy.resolve_poc_max_batch_size(536, 704) == 536
