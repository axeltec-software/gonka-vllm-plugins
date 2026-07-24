# `.github/workflows/`

CI for `gonka-poc`. All workflows run on `ubuntu-latest` and are CPU-only —
GPU validation happens downstream in the deployment pipeline.

## Workflows

### `contract-tests.yml`

Read-only API-drift detector against the `vllm` versions we target.

- **What it does.** Installs `vllm` (pinned per matrix entry), installs
  `gonka-poc` with the `[test]` extra, runs `pytest tests/contract tests/unit`. The
  contract suite only imports vllm internals and asserts the shapes
  (classes, signatures, attribute names) the plugin binds to. No model is
  loaded; no CUDA is touched.
- **Triggers.**
  - `push` to `main`
  - `pull_request` against any branch
  - `workflow_dispatch` (manual)
  - Skipped automatically when the diff is docs-only (`**/*.md`, `docs/**`,
    `LICENSE`).
- **Matrix.** `vllm == 0.23.0` and `vllm == 0.25.1` — published-on-PyPI
  versions only (vllm RC tags exist on GitHub but are never published to
  PyPI). Add a matrix entry when a new supported version ships.
- **Manual trigger.**
  - GitHub UI: *Actions → contract-tests → Run workflow → pick a ref*.
  - CLI: `gh workflow run contract-tests.yml --ref <branch>`.

#### `smoke-help` job (same workflow, same matrix)

Real-wheel composition smoke: verifies the `gonka-vllm-serve` console
script resolves to our entrypoint, imports the entrypoint/worker/`poc.*`
modules against the installed vllm wheel, and checks the
`vllm.general_plugins` entry-point loads and invokes idempotently.
Catches packaging and import-path bugs the contract suite cannot see.

### `grep-lint.yml`

Process gate running `tools/grep_lint.py` (pure stdlib, no install step).
Fails PRs that import `vllm.v1.*` outside `src/gonka_poc/_compat/` (the
only blessed channel for upstream internals) or cite an `ADR-NNNN` with
no matching file under `docs/adr/`.

## What to do if contract tests fail

See the "two hard rules" in [CONTRIBUTING.md](../../CONTRIBUTING.md):
a failing contract test means the surface moved, so the fix belongs in
`_compat/`, not in the assertion.
