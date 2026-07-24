# ADR-0014 — Residual vLLM fork as permanent infrastructure (local stub)

**Status:** Accepted (amends ADR-0013 §Layer-3). Originated as
mlnode-foundry ADR-0014 ([canonical copy](https://github.com/kaitakuai/mlnode-foundry/blob/main/docs/adr/0014-residual-fork-permanent-infra.md), public); this file is
authoritative for this repo.
This file is a 1-page summary so an offline reader of the `gonka-poc` repo
can find the rationale for the two-artifact (plugin + thin fork) shipping
model without leaving the package.

**Date:** 2026-06-16 (canonical); local stub 2026-06-17.
**Owners:** @baychak
**Amends:** [ADR-0013](ADR-0013-poc-gate-ordering.md).

## Context (why this file exists)

`README.md` cites ADR-0014 from two places:

- Top-of-file status banner (`MIGRATION_FROM_FORK.md` + ADR-0014 explain
  the two-artifact relationship).
- "Why two artifacts" section (`see ADR-0014 in this repo's docs/adr/`).

`tests/gonka/README.md` also cites it when explaining why
`Dockerfile.quick` is not part of this repo. Without
a local file, those citations are dead links. This stub closes the link-rot.
The full options-considered narrative (status-quo, monkey-patch sampler,
full-fork rebase, thin-fork-permanent) lives in the original mlnode-foundry
ADR ([public](https://github.com/kaitakuai/mlnode-foundry/blob/main/docs/adr/0014-residual-fork-permanent-infra.md)) and is not duplicated here.

## Decision (the part the plugin's shipping model depends on)

Gonka PoC ships as **two artifacts** on purpose:

1. **`gonka-poc` plugin** (this repo) — everything reachable through vLLM's
   public extension surfaces:
   - `vllm.general_plugins` entry point.
   - `--worker-extension-cls gonka_poc.worker.PoCWorkerExtension` (the
     official `collective_rpc` surface).
   - `gonka-vllm-serve` composition wrapper around `build_app` (no source
     patches).
2. **`kaitakuai/vllm@poc-sampler-residual-vX.YY`** — a thin (6-commit,
   ~150-line) fork holding the sampler-stack residual: enforced-token
   sampling, per-request `logprobs_mode`, structured-output graceful
   degradation. Those touch private vLLM internals (`vllm/v1/sample/*`,
   `vllm/v1/structured_output/*`, `vllm/v1/worker/gpu_input_batch.py`)
   with no public hook today.

The original ADR-0013 plan was to upstream the sampler hooks to
`vllm-project/vllm` (Layer 3) and retire the fork. ADR-0014 marks that
exit strategy as **DEFERRED — no owner assigned for the upstream track**.
No `vllm-project` PR is in flight and none is scheduled; until someone
owns that track, the thin fork is treated as **permanent infrastructure**
rather than a temporary bridge, and the shipping model must not assume the
fork disappears.

Status note: this decision was made under kaitakuai ownership
(2026-06-16). Whoever owns the package next may well have an acceptance
channel into `vllm-project` that was unavailable then — re-open the
Layer-3 question rather than inheriting this deferral as settled.

## What this means operationally

- Per vLLM minor: rebuild the residual fork as
  `vllm==0.23.0+gonka.samplerN` (mechanical hours, not days). REBASE.md on
  the residual branch documents the cherry-pick order.
- Contract tests on the residual branch (`tests/contract/`) pin the
  private sampler surface so upstream drift fires a CI alert before the
  manual rebase.
- Plugin (`gonka-poc`) and residual wheel are coupled by version:
  `pip install gonka-poc==X` requires `vllm==0.23.Y+gonka.samplerZ`.
- Once an upstream PR ever lands that retires part of the residual, that
  part migrates into the plugin and the corresponding compat shim and
  fork commit go away. The shipping model survives the migration —
  consumers still install one plugin.

## Provenance

Originated as mlnode-foundry ADR-0014 ([public](https://github.com/kaitakuai/mlnode-foundry/blob/main/docs/adr/0014-residual-fork-permanent-infra.md)); this file is
authoritative for this repo.

## Links

- Residual fork branch: <https://github.com/kaitakuai/vllm/tree/poc-sampler-residual-v0.23>
- Companion ADR: [ADR-0013](ADR-0013-poc-gate-ordering.md) (PoC gate
  ordering contract)
- Migration guide: `MIGRATION_FROM_FORK.md` (Section 3 — per-commit fork
  inventory and the upstream-PR backlog)
