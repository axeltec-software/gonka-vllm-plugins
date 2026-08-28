# SPDX-License-Identifier: Apache-2.0
"""``params.scheme`` decides which proof a request gets, and prefill is default.

Both schemes live in one process, so the choice has to be explicit and it has
to default to the one the deployed fleet already validates: a chain that knows
nothing about decode must keep receiving the artifacts it has always received.
"""
import pytest

from gonka_poc.poc.routes import PoCParamsModel


def test_absent_scheme_means_prefill():
    """The compatibility guarantee: an old chain sends no scheme at all."""
    assert PoCParamsModel(model="m", seq_len=256).scheme == "prefill"


@pytest.mark.parametrize("scheme", ["prefill", "decode"])
def test_both_schemes_are_accepted(scheme):
    assert PoCParamsModel(model="m", seq_len=256, scheme=scheme).scheme == scheme


def test_unknown_scheme_is_rejected():
    """Never guess: an unrecognised proof name must fail the request, not
    silently fall back to a derivation the caller did not ask for."""
    with pytest.raises(Exception):
        PoCParamsModel(model="m", seq_len=256, scheme="sphere")
