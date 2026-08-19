# SPDX-License-Identifier: Apache-2.0
"""Mixed-PoC (chat + PoC in one forward) — EXPERIMENTAL subpackage.

Off unless explicitly enabled at launch. Depends one-way on the gonka_poc
core (core never imports gonka_poc.mixed — pinned by
tests/unit/test_mixed_contract.py). Engine coupling is exactly two seams:

  * ``GPUModelRunner.pre_forward_hooks`` — a 6-line residual extension
    (kaitakuai/vllm branch mixed/pre-forward-hooks) delivering per-step
    context right before the forward;
  * ``--scheduler-cls`` — admission policy subclass (documented-risk seam:
    upstream disclaims scheduler-interface stability; revalidated per minor).

Everything else rides the same public points the core already uses.
"""

from gonka_poc.mixed.admission import PoCAdmission  # noqa: E402
from gonka_poc.mixed.bridge import PoCRunnerBridge  # noqa: E402
