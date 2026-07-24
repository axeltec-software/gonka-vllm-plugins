# Contributing

This package is a **plugin**, not a fork. It installs on a stock vLLM wheel and
reaches into vLLM only through documented extension points. Most of the rules
below exist to keep it that way, because the moment the plugin depends on vLLM
internals in an undisciplined way, every vLLM minor becomes a porting project.

## The two hard rules

**1. Private vLLM APIs live in `_compat/` and nowhere else.**

Anything under `vllm.v1.*` is a private surface that upstream renames without
notice. Import it only inside `src/gonka_poc/_compat/v0_XX.py`, behind a function
the rest of the package calls. `tools/grep_lint.py` enforces this in CI.

When vLLM adds a minor, add a new `_compat/v0_XX.py` and register it in the
dispatch table in `_compat/__init__.py`. The dispatcher raises on an unsupported
minor on purpose — a silent fallback would produce wrong results instead of a
clear error.

**2. A failing contract test means the code moved, not the test.**

`tests/contract/` pins the exact private surfaces the plugin depends on. These
tests are deliberately brittle: they exist to fail loudly when upstream
refactors something we rely on. When one fails, fix `_compat/` (or the calling
code) to match the new surface. Do not relax the assertion to make CI green —
that converts a loud failure into a silent one, which is the failure mode the
suite was written to prevent.

## Consensus-critical code

The PoC forward path is consensus-critical: a prover and a validator running
different hardware must produce bit-identical vectors. That covers the seeded
RNG, the Householder/Haar scoring, the nonce partitioning, and the vector
encoding.

Consequences for changes in `src/gonka_poc/poc/`:

- Refactors that look purely cosmetic can still change floating-point results.
  Reordering operations, fusing loops, or "deduplicating" two near-identical
  helpers are all capable of moving the last bits.
- `_murmur3_32` / `_batched_murmur3_32` and `_normal` / `_batched_normal` are
  kept separate **on purpose**. Do not unify them without a cross-validator
  bit-compatibility run.
- The nonce partitioning formula
  (`nonce = node_id + group_id*n_nodes + x*(n_groups*n_nodes)`) is a
  network-wide contract. Changing it breaks disjoint coverage across nodes.
- The pseudo-input-ids derivation is pinned by frozen reference vectors in
  `tests/unit/test_v4_metadata_layout.py`. If that test fails, the change is
  wrong unless the whole network agrees to a new convention.

If a change touches this path, say so in the PR description and state how it was
verified (ideally: identical nonces from the same seed, before and after).

## Running the tests

```bash
pip install -e '.[test]'
pytest tests/unit tests/contract      # CPU-only, no GPU, no engine start
python3 tools/grep_lint.py            # process gates
```

`tests/gonka/` holds live GPU tests. They need a running engine and real
weights, so they are not part of the CPU suite; see `tests/gonka/README.md`.

CI runs the CPU suite against every supported vLLM minor. Both minors must pass:
a change that fixes one and breaks the other is not done.

## Pull requests

Describe *why*, not just *what* — the reasoning behind a compat shim is usually
worth more than the shim. If a change is a workaround for an upstream bug, link
the upstream issue so the workaround can be removed when it is fixed.

Decisions with lasting consequences go in `docs/adr/` and are summarised in
`docs/decision-log.md`.
