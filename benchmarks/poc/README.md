# PoC benchmarks & reports

Client-side measurement and reporting kit for mixed decode-PoC. Everything
talks to a running server over HTTP (`vllm serve <model>` with this plugin
installed) and writes role-tagged JSONs under `runs/`; the report tools are
pure renderers over those files.

| tool | role |
|---|---|
| `collect.py` | generate / validate rounds (separation data, full provenance) |
| `perfomance_nonces.py` | PoC steps/s and chat tok/s under load |
| `quality_gsm8k.py` | chat accuracy with concurrent PoC (co-existence) |
| `analyze.py` | offline separation + perf tables from runs/*.json |
| `report.py` | self-contained HTML report per model from runs/ |
| `simplify_report.py` | the condensed one-page report |
| `thresholds.py` | artifact-derived acceptance lines (k + vector), AUC |
| `vector_separation.py` | offline vector-channel analysis |
| `pair_report.sh` | end-to-end honest/fraud 2×2 for a model pair |
| `run_model_report.sh` | full per-model measurement session |
| `calibrate_tau.sh` | margin-gate τ calibration (re-validates per candidate τ) |

Verdict-constant context: the production working point (count metric,
τ=0.010, p_mismatch=0.021 @ batch 200) was calibrated on the worst fleet
pair and independently verified; see the wiki's verdict-metric calibration
entry. `calibrate_tau.sh` re-derives τ for a new (model, GPU) pair.
