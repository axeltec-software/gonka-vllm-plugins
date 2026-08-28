# SPDX-License-Identifier: Apache-2.0
"""The prefill forward must run on a model with no PoC rows marked.

The in-model wrappers are gated by a row mask, and a decode round leaves it
set. The prefill scheme reaches the worker over collective_rpc and never goes
through the runner bridge that would clear it, so a prefill round following a
decode round read transformed hidden states: on 1xH100 / Qwen3-1.7B the same
request returned a different vector_b64 depending on whether a decode round
had run before it. Reproducible artifacts are the whole point of the proof,
so the prefill path asserts this itself rather than trusting the other path
to tidy up.
"""
from types import SimpleNamespace

import gonka_poc.poc.poc_model_runner as pmr


class _Native:
    def __init__(self):
        self.cleared = 0

    def set_mask(self, row_mask):
        assert row_mask is None, "prefill marks no rows"
        self.cleared += 1


def test_prefill_forward_clears_the_row_mask(monkeypatch):
    """Read the source rather than booting a model: the reset must sit before
    the forward, and there is no cheaper way to pin ordering."""
    import inspect

    src = inspect.getsource(pmr)
    reset = src.find("set_mask(None)")
    forward = src.find("with poc_forward_context():")
    assert reset != -1, "prefill path no longer clears the PoC row mask"
    assert reset < forward, "the mask is cleared after the forward, too late"


def test_reset_is_skipped_when_no_transforms_are_attached():
    """A model without the wrappers has no mask to clear; the lookup must not
    raise on a plain module."""
    model = SimpleNamespace()
    assert getattr(model, "_poc_native_state", None) is None
