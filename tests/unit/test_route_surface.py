# SPDX-License-Identifier: Apache-2.0
"""The PoC HTTP surface the network node calls.

These paths are a contract with MLNode's Go proxy, not an internal detail:
the node discovers capabilities and drives rounds through them, and it ships
on its own release cadence. Dropping one is a breaking change for every node
already deployed, which is exactly how ``/versions`` — the ADR-0015 §6
feature-detection handshake — went missing when KV-lease borrowing was
removed: the mechanism behind the flag went away, and the flag went with it.

Removing a path here is a deliberate act that needs an MLNode-side plan.
Adding one is free. So this pins existence and method only, never payload
shape (the artifact fields are additive and covered elsewhere).
"""
import pytest

pytest.importorskip("vllm")

from gonka_poc.poc.routes import router  # noqa: E402

# path -> method, as MLNode calls them.
REQUIRED = {
    "/api/v1/pow/init/generate": "POST",
    "/api/v1/pow/generate": "POST",
    "/api/v1/pow/generate/{request_id}": "GET",
    "/api/v1/pow/versions": "GET",
    "/api/v1/pow/status": "GET",
    "/api/v1/pow/stop": "POST",
}


def _surface():
    return {r.path: r.methods for r in router.routes}


@pytest.mark.parametrize("path,method", sorted(REQUIRED.items()))
def test_chain_facing_route_is_served(path, method):
    surface = _surface()
    assert path in surface, (
        f"{path} is gone; deployed MLNode versions still call it. Removing a "
        f"chain-facing path needs a node-side plan, not just a deletion."
    )
    assert method in surface[path], (
        f"{path} no longer accepts {method} (accepts {sorted(surface[path])})"
    )


def test_versions_reports_both_components_and_the_coexistence_flag():
    """MLNode reads this to decide whether it may keep serving during a round.

    Mixed decode-PoC makes coexistence unconditional — PoC and chat share the
    scheduler — so the flag is now a constant rather than a borrow probe. It
    must still be present and true, or a node that reads it falls back to the
    abort-everything regime and stops serving inference for the round.
    """
    import asyncio

    from gonka_poc.poc.routes import get_versions

    body = asyncio.run(get_versions(None))
    assert body["poc_validation_inference"] is True
    assert body["vllm_version"]
    assert body["gonka_poc_version"]
