# Migration from `mb/feat/port-pocv2-vllm-0.23.0` fork

This document is a **historical record** of the 2026-06 port: it records the
disposition of every commit on the source branch
(branch `mb/feat/port-pocv2-vllm-0.23.0` of the `kaitakuai/vllm` fork).
References to `mlnode-foundry` below name the deployment repo those commits
moved to at port time; the plugin has **no build or runtime dependency** on
it — deployment tooling is out of scope for this package. Three buckets:

1. **Plugin** -- ported into this package (no upstream edits).
2. **Foundry profile** -- belongs in `mlnode-foundry` (deployment defaults,
   Dockerfile overlays, CI workflows). Listed below with the destination
   file path so the foundry repo PR knows exactly what to receive.
3. **Stays in fork until Layer 3** -- private-API surfaces vLLM does not
   expose; requires upstream PR for a hook before we can move to the
   plugin.

## 1. Plugin commits (already in this scaffold)

*This table records the original port disposition; some destination modules
were later refactored away (see git history).*

| SHA | Subject | Destination in plugin |
|-----|---------|-----------------------|
| `3efa985f8` | feat(poc): import PoC v2 module from 0.15.1 fork | `src/gonka_poc/poc/*` |
| `2547cbc95` | feat(poc): stronger RNG via concat-murmur (upstream PR kaitakuai/vllm#30) | `src/gonka_poc/poc/{config,engine_patch,generate_queue,gpu_random,manager,poc_model_runner,routes}.py` |
| `99a372d4e` | safer kv cache reuse (kv-reuse part; Dockerfile portion -> Section 2) | `src/gonka_poc/poc/poc_model_runner.py` |
| `9ec7ab432` fix compilation skip | `src/gonka_poc/poc/poc_model_runner.py` (already final state) |
| `4a4c921f0` add scratchpad (revert) | `src/gonka_poc/poc/poc_model_runner.py` (already final state) |
| `623ef37d7` feat(api): integrate PoC router and priority gating | `src/gonka_poc/entrypoint/{api_router,gating}.py` + reuse of `src/gonka_poc/poc/routes.py` |
| `8f30fd4e2` chore(api): return token id as numeric string from _get_decoded_token | REPLACED by `src/gonka_poc/poc/routes.py` serialiser (do NOT port the `_get_decoded_token` edit -- emit numeric ids from PoC routes only) |
| `582f087a5` fix(poc): restore seq_lens_cpu_upper_bound kwarg for MLA attention (kaitakuai/vllm#9) | `src/gonka_poc/_compat/v0_23.py::build_common_attention_metadata` |
| `d16c2127d` test(gonka): port PoC v2 live and unit tests from 0.15.1 fork | `tests/gonka/*` |

The fork commit `99a372d4e` ("safer kv cache reuse") also bumps the
Dockerfile base image -- the Dockerfile part is **foundry**, see Section 2.

### Not ported to plugin

| SHA | Subject | Rationale |
|-----|---------|-----------|
| `15ee09f11` feat(models): add Qwen3MoeForCausalLMConfig with PoC custom_ops defaults | No real PoC-specific `custom_ops` exist for Qwen3MoE today -- the fork's `MODELS_CONFIG_MAP` entry was an aspirational scaffold with no concrete ops. When a real op is needed it will be added explicitly under a fresh module + wired from `plugin.register()`; carrying an empty seed invites confusion. |

## 2. Foundry-profile commits (land in `mlnode-foundry`)

All paths below are relative to `mlnode-foundry/`. None of these belong in
the plugin.

| SHA | Subject | mlnode-foundry destination |
|-----|---------|----------------------------|
| `75ad0684b` | chore(config): raise OpenAI API default max_num_batched_tokens | `profiles/gonka-poc/engine-args.yaml` (add key `max_num_batched_tokens`) |
| `02f74e99a` | chore(docker): Dockerfile.quick overlay for vllm-openai:v0.19.0 | `profiles/gonka-poc/Dockerfile.overlay` (initial overlay -- base image, `VLLM_ALLOW_INSECURE_SERIALIZATION=1`, FlashInfer NVFP4/FP8 env) |
| `fc77f468f` | chore(docker): bump Dockerfile.quick base to vllm-openai:v0.20.0 | `profiles/gonka-poc/Dockerfile.overlay` (base image bump) |
| `c66d29e03` | chore(config): raise default gpu_memory_utilization to 0.925 for OpenAI API | `profiles/gonka-poc/engine-args.yaml` (`gpu_memory_utilization: 0.925`) |
| `a2ffe9c2b` | bake attention_backend, logprobs_mode, compilation_config, max_num_batched_tokens defaults | `profiles/gonka-poc/engine-args.yaml` (`attention_backend: FLASHINFER`, `logprobs_mode: processed_logprobs`) |
| `cb02223b2` | Fix attention backend (move to EngineArgs.attention_backend) | `profiles/gonka-poc/engine-args.yaml` (same key; ensure CLI shape) |
| `c92077746` | fix default args -- wire --attention-backend CLI default to FLASHINFER | `profiles/gonka-poc/launch.sh` (CLI flag) OR `engine-args.yaml` |
| `03f74653b` | hardcode dtype auto | `profiles/gonka-poc/engine-args.yaml` (`dtype: auto`) -- do NOT patch `vllm/config/model.py` |
| `9e6d2735f` | chore(docker): bump Dockerfile.quick base to vllm-openai:v0.23.0 | `profiles/gonka-poc/Dockerfile.overlay` (base image bump) |
| `99a372d4e` *(Dockerfile bump portion only)* | safer kv cache reuse (Dockerfile bump only) | `profiles/gonka-poc/Dockerfile.overlay` |
| `423a5a591` | ci: add build-stage1 workflow for the vLLM overlay image (kaitakuai/vllm#10) | `.github/workflows/build-stage1.yml` (in `mlnode-foundry` -- includes cosign, SLSA, SBOM signing) |

### Foundry-profile target file layout (recommended)

```
mlnode-foundry/
  profiles/gonka-poc/
    Dockerfile.overlay      # base = vllm/vllm-openai:v0.23.0-cu129
    engine-args.yaml        # gpu_memory_utilization, dtype, attention_backend,
                            # logprobs_mode, max_num_batched_tokens
    launch.sh               # gonka-vllm-serve entrypoint + per-arch CLI flags
  .github/workflows/
    build-stage1.yml        # ported from 423a5a591
```

The overlay Dockerfile should `RUN pip install gonka-poc==<pinned>` on top of
the stock vllm-openai image, then set
`ENV VLLM_ALLOW_INSECURE_SERIALIZATION=1` and PoC env defaults.

## 3. Stays in fork until Layer 3 (upstream hook required)

These commits touch private vLLM surfaces with no plugin extension point.
Keep them as a **thin patch series** on a `kaitakuai/vllm` fork branch
(`kaitakuai/vllm @ poc-sampler-residual-v0.23`) and rebuild the wheel for
each vllm minor bump. Move to plugin only after an upstream PR adds a hook.

| SHA | Subject | Touched files | Rationale |
|-----|---------|---------------|-----------|
| `1e6913a38` | feat(sampling): add per-request logprobs_mode and enforced_token_ids fields | `vllm/sampling_params.py`, `vllm/v1/sample/metadata.py` | Sampler-stack data fields -- no plugin hook. ADR fork-residual. |
| `e35461e0e` | feat(sampler): add need_processed_logprobs and .sample() wrapper | `vllm/v1/sample/ops/topk_topp_sampler.py` | Sampler op signature change; not pluggable. |
| `81abd50f5` | feat(sampler): port PoC v2 mixed-mode sampling and enforced tokens | `vllm/v1/sample/sampler.py` | Core enforced-token override; sampler-stack residual. |
| `95dce5242` | feat(worker): port InputBatch enforced-tokens and logprobs-mode bookkeeping | `vllm/v1/worker/gpu_input_batch.py`, `vllm/v1/worker/gpu_model_runner.py` | InputBatch is private worker state -- not reachable via `worker_extension_cls`. |
| `1a328700e` | fix(sampler): thread need_processed_logprobs through forward_xpu | `vllm/v1/sample/ops/topk_topp_sampler.py` | Continuation of sampler-op surgery. |
| `8f30fd4e2` (the `_get_decoded_token` part) | chore(api): return token id as numeric string | `vllm/...` -- **DROPPED**; replaced by plugin serialiser | Already documented in Section 1; the fork edit itself stays dropped, **not** carried as a residual. |
| `4996d5af7` | feat(structured-output): graceful degradation on grammar token rejection | `vllm/v1/structured_output/__init__.py`, `vllm/v1/structured_output/backend_xgrammar.py` | Private xgrammar internals; ADR layer-3-deferred. |

### Upstream-PR backlog (to retire each item from the fork)

Upstreaming is DEFERRED-INDEFINITELY per ADR-0014 (the fork is permanent
infrastructure; this backlog records what WOULD retire each item; revisit
under gonka-ai ownership).

1. **Sampler-stack hook**: PR upstream adding a `LogitsProcessor`-style
   per-request `enforced_token_ids` slot + `logprobs_mode` enum exposed on
   `SamplingParams`. Once merged, port `1e6913a38` / `81abd50f5` /
   `e35461e0e` / `1a328700e` to the plugin.
2. **InputBatch hook**: PR upstream exposing `per_request_extras: dict` on
   `InputBatch` with a documented refresh point. Retires `95dce5242`.
3. **Structured-output hook**: PR upstream adding a callback registry on
   `StructuredOutputManager` for grammar rejection. Retires the
   structured-output residual.

When all three land upstream the plugin becomes the single source of truth
and the `kaitakuai/vllm` fork can be archived (archive only after one
quarter of clean hardware validation on stock wheels).
