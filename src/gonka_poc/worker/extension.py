"""PoCWorkerExtension -- mixed into the vLLM V1 GPU Worker via ``--worker-extension-cls``.

Diagnostics only. PoC nonces ride the serving pipeline as engine requests
(routes -> generate(poc_params=...) -> scheduler admission -> runner bridge),
so nothing here is on the artifact path.

How vLLM wires this in (verified on the 0.25.1 line):
    ``vllm/v1/worker/worker_base.py:263-284`` (WorkerWrapperBase.init_worker)
    resolves the qualname, asserts no attribute collisions with the concrete
    Worker, then appends the class to ``worker_class.__bases__``. There is NO
    __init__ -- methods just become attributes on the live Worker.

Inside any method, ``self`` is the live GPU Worker. Return only
msgpack-serialisable values -- no tensors.
"""
import logging

from typing import Any, Dict

logger = logging.getLogger(__name__)


class PoCWorkerExtension:
    """RPC surface for PoC-enabled workers (see module docstring)."""

    def poc_worker_info(self) -> dict:
        """Liveness/identity probe for operators and tests."""
        return {
            "rank": getattr(self, "rank", -1),
            "device": str(getattr(self, "device", "?")),
        }


__all__ = ["PoCWorkerExtension"]
