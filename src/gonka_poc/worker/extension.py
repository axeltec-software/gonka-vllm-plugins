"""PoCWorkerExtension -- mixed into the vLLM V1 GPU Worker via ``--worker-extension-cls``.

Activation: pass the mixed extension subclass (see the mixed subpackage) as
``--worker-extension-cls``; this base carries only the shared surface.

How vLLM wires this in (verified on the 0.25.1 line):
    ``vllm/v1/worker/worker_base.py:263-284`` (WorkerWrapperBase.init_worker)
    resolves the qualname, asserts no attribute collisions with the concrete
    Worker, then appends the class to ``worker_class.__bases__``. There is NO
    __init__ -- methods just become attributes on the live Worker.

Inside any method, ``self`` is the live GPU Worker (self.model_runner,
self.device, self.rank, self.vllm_config).

In mixed mode PoC work does NOT enter through collective_rpc: nonces ride the
serving pipeline as engine requests (routes -> generate(poc_params=...) ->
scheduler admission -> runner bridge). This base class only carries
introspection helpers shared by the mixed extension; keep method names
prefixed ``poc_``/``mixed_`` -- vLLM asserts no attribute collisions at
init_worker time, and return values must be msgpack-serialisable (no tensors).
"""
import logging

logger = logging.getLogger(__name__)


class PoCWorkerExtension:
    """Base RPC surface for PoC-enabled workers (see module docstring)."""

    def poc_worker_info(self) -> dict:
        """Liveness/identity probe for operators and tests."""
        return {
            "rank": getattr(self, "rank", -1),
            "device": str(getattr(self, "device", "?")),
        }


__all__ = ["PoCWorkerExtension"]
