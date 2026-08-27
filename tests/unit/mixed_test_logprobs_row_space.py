# SPDX-License-Identifier: Apache-2.0
"""Chat logprobs must live in the same row space as the sampled tokens.

In a mixed step the sampler only ever sees chat rows, so everything it
returns is compacted to 0..len(chat)-1, while ``sampled_token_ids`` is
scattered back to full batch width with PoC slots zeroed. The runner then
reads ``output.sampled_token_ids`` and ``output.logprobs`` in parallel by
request index — so two different row spaces hand a chat request the logprobs
belonging to whichever row follows a PoC row.

That is the renumbering incident again (the one the natural-order rule was
written for), moved onto the other tensor, and it only shows when a chat
request actually asks for logprobs — which the default never does.
"""
import sys
from collections import namedtuple
from types import ModuleType, SimpleNamespace

import pytest
import torch

from gonka_poc.mixed.bridge import PoCRunnerBridge

_LP = namedtuple("LogprobsTensors",
                 "logprob_token_ids logprobs selected_token_ranks")
_SO = namedtuple("SamplerOutput", "sampled_token_ids logprobs_tensors")


@pytest.fixture(autouse=True)
def _stub_vllm_outputs(monkeypatch):
    """bridge imports these lazily; give it shape-compatible stubs so this
    stays a pure-CPU unit test."""
    mod = ModuleType("vllm.v1.outputs")
    mod.LogprobsTensors = _LP
    mod.SamplerOutput = _SO
    monkeypatch.setitem(sys.modules, "vllm.v1.outputs", mod)
    yield


def _lp(rows, k=3):
    """Distinct per-row values so a misplaced row is visible, not plausible."""
    base = torch.arange(rows, dtype=torch.float32).unsqueeze(1)
    return _LP(
        logprob_token_ids=(base.to(torch.int32) + 100).repeat(1, k),
        logprobs=base.repeat(1, k),
        selected_token_ranks=base.squeeze(1).to(torch.int32),
    )


def test_chat_logprobs_land_on_their_own_request_rows():
    # Batch of 4: PoC at 0 and 2, chat at 1 and 3. A PoC row sits FIRST, so a
    # compacted row space is off by one from the very first chat request.
    chat_rows = [1, 3]
    idx = torch.tensor(chat_rows, dtype=torch.long)
    out = PoCRunnerBridge._scatter_logprobs(_lp(len(chat_rows)), idx, n_full=4)

    # chat row 1 keeps the sampler's row 0, chat row 3 keeps row 1
    assert out.logprobs[1].tolist() == [0.0, 0.0, 0.0]
    assert out.logprobs[3].tolist() == [1.0, 1.0, 1.0]
    assert out.selected_token_ranks.tolist() == [0, 0, 0, 1]
    # PoC rows are present but empty — never read, never shifting anyone
    assert out.logprobs[0].tolist() == [0.0, 0.0, 0.0]
    assert out.logprobs[2].tolist() == [0.0, 0.0, 0.0]
    assert out.logprobs.shape[0] == 4


def test_row_space_matches_the_token_tensor_width():
    """Both tensors the runner zips must be n_full rows, or the zip is wrong."""
    idx = torch.tensor([2], dtype=torch.long)
    out = PoCRunnerBridge._scatter_logprobs(_lp(1), idx, n_full=5)
    assert out.logprobs.shape[0] == 5
    assert out.logprob_token_ids.shape[0] == 5
    assert out.selected_token_ranks.shape[0] == 5


def test_absent_logprobs_stay_absent():
    assert PoCRunnerBridge._scatter_logprobs(
        None, torch.tensor([0], dtype=torch.long), n_full=3) is None


def test_multi_row_per_request_is_refused_not_guessed():
    """Spec decode emits >1 row per request; placing them needs a row map this
    path does not carry. Refusing beats silently mis-indexing."""
    idx = torch.tensor([1, 3], dtype=torch.long)
    with pytest.raises(RuntimeError, match="logprob rows"):
        PoCRunnerBridge._scatter_logprobs(_lp(4), idx, n_full=4)


def test_sampled_tokens_and_logprobs_agree_row_for_row(monkeypatch):
    """The pairing the runner actually performs, with a PoC row going first.

    ``output.sampled_token_ids`` and ``output.logprobs`` are zipped by request
    index downstream, so this asserts on the pair, not on either tensor alone:
    whatever token row 2 gets, row 2's logprobs must belong to the same chat
    request.
    """
    import gonka_poc.mixed.runtime as md

    monkeypatch.setattr(md, "slice_sampling_metadata",
                        lambda sm, rows, device: rows)

    # rows: 0 = PoC, 1 and 2 = chat
    batch = SimpleNamespace(req_ids=["poc-a", "chat-x", "chat-y"], num_reqs=3)

    def sampler(logits, sampling_metadata):
        n = len(sampling_metadata)          # chat rows only, compacted
        tokens = torch.tensor([[7000 + i] for i in range(n)], dtype=torch.int32)
        return SimpleNamespace(sampled_token_ids=tokens, logprobs_tensors=_lp(n))

    b = PoCRunnerBridge(SimpleNamespace(
        device=torch.device("cpu"), input_batch=batch, sampler=sampler))
    b._step = {"poc_req_ids": {"poc-a"}}

    out = b.sample_chat_rows(logits=None, sampling_metadata=None)

    # chat-x is batch row 1 and was the sampler's row 0; chat-y is row 2 / 1.
    assert out.sampled_token_ids[1].tolist() == [7000]
    assert out.logprobs_tensors.selected_token_ranks[1].item() == 0
    assert out.sampled_token_ids[2].tolist() == [7001]
    assert out.logprobs_tensors.selected_token_ranks[2].item() == 1
    # the PoC row occupies its natural slot in BOTH tensors
    assert out.sampled_token_ids[0].tolist() == [0]
    assert out.logprobs_tensors.selected_token_ranks[0].item() == 0
    assert out.sampled_token_ids.shape[0] == 3
    assert out.logprobs_tensors.logprobs.shape[0] == 3
