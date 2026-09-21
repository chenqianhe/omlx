# Hy-MT2 v1 validation record

Validation date: 2026-09-21. Implementation base: main
`d94d4cec989019344c2c4df592975e133f18255b`.

## Automated regression

The focused prefill, scheduler, engine, settings and admin suites passed:
**2,301 tests**, with 3 deselected by the repository's default test configuration.
The run includes real small `llama`, `qwen2`, `qwen3` and `hy_v3` models, affine
quantization and HyV3 expert gate/up fusion, scalar versus batched decode,
unequal-length cache extraction, cancellation, memory pressure, transactional
handoff failures, MTP UID rollback, persistence and model override isolation.
Adapter tests additionally verify that geometry/memory planning works without
importing MLX and rejects unsupported argument schemas.

```sh
python -m pytest -q \
  tests/test_prefill*.py tests/test_batched_prefill*.py \
  tests/test_scheduler*.py tests/test_engine_core.py tests/test_engine_pool.py \
  tests/test_settings.py tests/test_model_settings.py \
  tests/test_model_settings_profiles.py tests/test_admin_model_settings.py \
  tests/test_admin_profiles_api.py tests/test_admin_new_profile_expose_as_model.py
```

New prefill modules, benchmark and related new test files pass Ruff;
`git diff --check` passes. Existing unrelated lint findings in the large
integration modules are outside this change. This is a focused regression run,
not a claim that the entire repository test suite passed.

## Complete-checkpoint engine smoke

Checkpoint: `QianheChen/Hy-MT2-30B-A3B-APEX-Imatrix-I-Quality-MLX`.
Its `config.json` SHA-256 was
`7b6c814aaa6440a9b8b207a9bed421e6b021aaf268b4b512c6736bf3546188de`;
the smoke did not change it. Runtime versions: MLX 0.32.2, MLX-LM 0.32.0
(repository pin `872ae88d1fac77350db23c8c04fe8dd372a9e3e8`), MLX-VLM 0.7.1.
Dependencies were provided in an isolated test directory.

Normal `BatchedEngine.start()` loaded the checkpoint and its standard post-load
patches. Capability inspection accepted its 48 plain-KV layers, 128 routed
experts, top-8 routing and shared expert. Four concurrent cold text requests had
126, 127, 129 and 130 input tokens, requesting translation of “The meeting starts
tomorrow.” with differing reference notes. Generation used temperature 0 and
natural EOS with a 32-token cap. Prefix caching was disabled, prefill step size
was 256, and the process/MLX memory limit was 30 GiB.

One warmup plus three counterbalanced rounds at each width produced 48 completed
requests. All ended normally with zero cached tokens, batch failures, discarded
or requeued tokens. Width 2 formed two groups per trial; width 4 formed one.
The engine shut down cleanly. After separating the model adapters and shared
test fixtures, a second 48-request engine smoke also completed successfully
with actual W2/W4 groups and zero batch failures; the timing table below retains
the initial run. The highest cumulative MLX allocator peak observed
was 19.95 GiB; this is not total process memory or a per-width memory comparison.

| Maximum width | Median four-request completion time, excluding warmup |
| --- | ---: |
| 1 | 0.8723 s |
| 2 | 0.7121 s |
| 4 | 0.6567 s |

These are observations from a short engine smoke, **not a release performance
benchmark**. The run bypassed HTTP, used only four closely related short prompts
and short completions, and did not compare unchanged main. Width 2 also generated
one fewer output token per trial. Do not extrapolate these figures to general
translation workloads, longer decode, other quantizations, or other MoE models.

## Output difference and diagnosis

Width 4 matched width 1 text for all four prompts in every round. At width 2,
the 129-token prompt consistently returned “会议明天开始。” whereas width 1
returned “会议将于明天开始。”. These are semantically similar in this example,
but they are not token-identical and no corpus-level quality conclusion follows.

A teacher-forced diagnostic compared the affected row's scalar prefix with:

1. The new runtime's two-row, 128-token forward and extracted scalar KV.
2. An ordinary MLX-LM `KVCache` receiving the same native `2 × 128` forward,
   followed by direct first-row slicing and scalar decode, bypassing the new
   extraction adapter and scheduler.

All 48 layers' extracted valid KV arrays had **maximum absolute difference 0**
between these two batched paths. Their top-token logits matched across all five
teacher-forced positions. Both reproduced the ranking flip after “会议”:

| Path | Logit for “将于” | Logit for “明天” |
| --- | ---: | ---: |
| Scalar prefix | 29.8851 | 29.5837 |
| New batched prefix | 29.6557 | 29.8042 |
| Native batched prefix + direct slice | 29.6557 | 29.8042 |

This control supports a batch-shape numerical effect in the loaded model/runtime,
rather than a new cache extraction or scheduler handoff defect. It does not
identify a particular kernel or establish a universal numerical-error bound.
Small-model tolerance checks cannot replace full-checkpoint quality evaluation.

## Release assessment

The implementation is ready for **opt-in code review**, with width 1 remaining
the default. Before recommending default enablement or publishing speedup claims,
run varied real translation corpora with quality scoring, sustained and staggered
HTTP workloads, prefix-cache traffic, and longer inputs/outputs on the target
hardware. Compare widths 1/2/4 and unchanged main with controlled cache state.
No HTTP serving, long-duration soak or translation quality benchmark is claimed
by this validation record.
