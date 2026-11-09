# Token generation and terminal output audit — 2026-11-09

The token-generation pipeline (`Talker` + 15-step `Code Predictor`) now uses
default-enabled graph fusion, strided KV views, 15 persistent pre-allocated
Code Predictor step schedules, and Hexagon HTP / Adreno 830 OpenCL backend
optimizations and bugfixes. Compact `summary` execution tracing and rate-limited
CLI terminal progress are also enabled by default.

## Checkout and prior evidence

Work started with clean application and GGML worktrees:

- Application `experimental/npu-max`: `9ac729b6ca05170e43c14042deb8221c4012fd67`.
- GGML `experimental/npu-max`: `a1ff3917ed1b31deac8b96f04a62afd89a15faa6`.
- Fetched application `origin`, application `upstream`, and GGML `origin` before
  edits. Both active branches were current with their remote branches.
  Upstream application `e04c79a` was already an ancestor; nothing was merged.
- Historical NPU-Max timing tables were background-scheduled and excluded from
  new comparisons. Original NPU launchers/runtime, DSP arithmetic and model
  weights were preserved; all 5 hashes in `.cache/npu-max/protected-baseline.json`
  matched before and after edits.

## Token-generation and backend optimizations

### 1. Code Predictor and Talker graph & runtime improvements (`src/tts_transformer.h`, `src/tts_transformer.cpp`)

- **Pre-concatenated Code Predictor QKV and Gate+Up weights (+55.7 MB RAM):**
  During model initialization, `layer.attn_qkv` (`[attn_q | attn_k | attn_v]`,
  `[1024, 4096]` in `Q8_0`) and `layer.ffn_gate_up` (`[ffn_gate | ffn_up]`,
  `[1024, 6144]` in `Q8_0`) are pre-concatenated on the host and uploaded in a
  single `ggml_backend_tensor_set` call per tensor (required by OpenCL `Q8_0`,
  which splits `d` scales and `qs` quants inside the CL buffer).
- **Fused QKV + contiguous `ggml_view_3d` slices and fused `ggml_swiglu`:**
  In `build_code_pred_step_graph`, a single `ggml_mul_mat(ctx0, layer.attn_qkv, cur)`
  produces `[4096, 1]` and slices `Qcur` (`offset 0`), `Kcur` (`offset 8192`),
  and `Vcur` (`offset 12288`) as contiguous 1D views (`20 → 5` matmul dispatches
  per step). In both `build_code_pred_prefill_graph` and `build_code_pred_step_graph`,
  `ggml_mul_mat(ctx0, layer.ffn_gate_up, cur)` + `ggml_swiglu(ctx0, gate_up)`
  replaces separate `ffn_gate` / `ffn_up` matmuls and `ggml_silu` + `ggml_mul`.
- **Strided KV views and Talker `ggml_swiglu_split`:**
  Both `build_step_graph` (Talker) and `build_code_pred_step_graph` (Code Predictor)
  read strided `K`/`V` cache views directly (`QWEN3_TTS_CODE_PRED_STRIDED_KV=1`
  by default) and use `ggml_swiglu_split` / `ggml_swiglu` (`QWEN3_TTS_CODE_PRED_SWIGLU=1`
  by default), eliminating 10 redundant `ggml_cont` KV copy nodes and 5 separate
  SiLU/multiply nodes per Code Predictor step.
- **Reduced HTP op counts (single split, zero CPU fallback):**
  Code Predictor step HTP compute nodes dropped from **138 → 108** on 0.6B
  (`-30` HTP ops/step, **`-420` HTP ops/frame**) and **140 → 110** on 1.7B,
  while prefill HTP compute nodes dropped from **149 → 139** (0.6B) and
  **153 → 143** (1.7B).
- **15 persistent Code Predictor step schedules (`state_.code_pred_plans[0..14]`):**
  Each of the 14 single-token autoregressive steps (`step = 1..14`, `n_past = 2..15`)
  has fixed tensor shapes and its own `code_pred_embd[step - 1]` and `code_pred_head[step]`
  weights. Pre-allocating 15 persistent `ggml_backend_sched_t` plans in
  `ensure_code_pred_plans` eliminates 14 per-frame graph reconstructions and
  14 `ggml_backend_sched_alloc_graph` + `ggml_backend_sched_reset` cycles per frame.
- **Fast host token sampling:**
  `sample_token` reuses `thread_local` scratch buffers, skips `expf` on `-INFINITY`
  masked logits, and avoids per-step `std::vector<float>` allocations in
  `sample_or_argmax`.

| Environment variable | Default | Effect when set to `0` |
| --- | --- | --- |
| `QWEN3_TTS_CODE_PRED_SWIGLU` | `1` (enabled) | Disable fused `ffn_gate_up` + `ggml_swiglu`, fused `attn_qkv`, and persistent step plans; revert to separate matmuls and `ggml_silu` + `ggml_mul` |
| `QWEN3_TTS_CODE_PRED_STRIDED_KV` | `1` (enabled) | Disable strided KV views and persistent step plans; insert `ggml_cont` KV copy nodes |

### 2. GGML Hexagon HTP and OpenCL backend improvements & bugfixes (`ggml`)

- **Hexagon HTP `HTP_MAX_MMAPS = 32` & slot-table eviction fix (`htp-ctx.h`, `main.c`):**
  Previously `HTP_MAX_MMAPS` was `16` while `m_reuse` was a 32-bit `uint32_t`, and
  `prep_op_bufs` only evicted unreferenced mmap entries when `(m_vmem + e_vmem) > ctx->max_vmem`
  (`3.35 GB`), never checking if all `HTP_MAX_MMAPS` table slots were occupied.
  With 15 persistent `code_pred_plans` schedulers plus model, KV caches, and Talker
  scheduler (19 buffers total, `< 1 GB`), the 17th buffer silently received
  `b->base = NULL`. Raising `HTP_MAX_MMAPS` to `32`, using `1u << i` for the
  32-bit mask, and checking `(m_count + (n_bufs - b_reuse)) > HTP_MAX_MMAPS` in
  `prep_op_bufs` keeps all 19 buffers permanently mapped on the DSP with zero
  `HAP_mmap2`/`HAP_munmap2` churn.
- **Hexagon HTP shared-input dynamic quantization reuse (`main.c`, `matmul-ops.c`):**
  Preserves `octx->src1_spad` across consecutive `HTP_OP_MUL_MAT` operations that
  share the exact same `src[1]` activation pointer and `src[0]` weight type
  (verifying `octx->src1_spad.stride == src1_row_size`), avoiding redundant FP32 →
  `Q8_0` dynamic activation quantization on shared-input matmuls (such as Talker
  Q/K/V and Gate/Up).
- **Hexagon HTP inter-batch `dspqueue` spin-poll (`main.c`):**
  Adds a bounded 512-iteration `hex_pause()` user-space spin-poll in
  `htp_packet_callback` when `poll_count == DSPQUEUE_POLL_COUNT` so back-to-back
  autoregressive step dispatches avoid the 100–250 µs QuRT kernel sleep/wakeup
  penalty between steps.
- **OpenCL `ggml_opencl_op_rms_norm_fused` view offset bugfix (`ggml-opencl.cpp:10810`):**
  Upstream `ggml_opencl_op_rms_norm_fused` computed the scale weight buffer offset
  as `cl_ulong offset1 = extra1->offset + src0->view_offs;` instead of
  `src1->view_offs`. When `Kcur` was a `ggml_view_3d` slice of fused `QKV` with
  `view_offs = 8192`, OpenCL shifted the `attn_k_norm` weight pointer by 8,192
  bytes. Fixing `offset1` to use `src1->view_offs` restored bit-exact `0` error
  on Adreno 830 OpenCL.

## Numerical controls (`--mode check`)

`tools/npu-max/test-token-generation.cpp` loads each Q8 model once, produces a
deterministic hidden-state fixture through a two-token talker prefill, and
compares the baseline (`QWEN3_TTS_CODE_PRED_SWIGLU=0`, `QWEN3_TTS_CODE_PRED_STRIDED_KV=0`)
and candidate (`both` enabled) predictor on that same input across CB0 IDs
0/1995/2047, greedy and reset-seed sampling, top-k 50, top-p 1.0/0.9, repeated
calls, and batch slots 0/1, checking **all 15 heads on identical histories**.

Across all six model/backend cells (`0.6b`/`1.7b` × `cpu`/`hexagon`/`opencl`) and
all three candidate configurations (`swiglu`, `strided-kv`, and default `both`),
every cell passed with **180/180 bit-exact head comparisons** (`nonexact_head_comparisons = 0`,
`maximum_nrmse = 0`, `maximum_absolute_error = 0`) and identical sampled codes.

## Foreground predictor measurements (`--mode bench`)

All performance runs were executed in verified Android foreground (`/top-app`,
CPUs `0-7`) using serial **Baseline A1 → Candidate B (`both`) → Baseline A2**
blocks with 3 warmup calls and 24 timed calls per block (`72` timed calls per
cell) on fixed inputs. All sampled codes matched 100% across A1, B, and A2:

| Size / backend | Baseline A1 median (ms) | Candidate B (`both`) median (ms) | Baseline A2 median (ms) | Reduction vs mean of A1/A2 | Reduction vs prior 2026-11-09 baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| **0.6B HTP** | 33.992 | **32.201** | 34.324 | **-5.73%** (-1.96 ms) | **-10.86%** (-3.92 ms vs 36.124 ms) |
| **0.6B OpenCL** | 59.120 | **53.802** | 58.914 | **-8.84%** (-5.22 ms) | **-6.77%** (-3.90 ms vs 57.706 ms) |
| **0.6B CPU** | 27.620 | **27.296** | 28.516 | **-2.75%** (-0.77 ms) | — |
| **1.7B HTP** | 35.180 | **33.071** | 35.129 | **-5.93%** (-2.08 ms) | **-10.40%** (-3.84 ms vs 36.909 ms) |
| **1.7B OpenCL** | 53.273 | **48.840** | 53.496 | **-8.51%** (-4.54 ms) | **-16.50%** (-9.65 ms vs 58.494 ms) |
| **1.7B CPU** | 29.957 | **28.940** | 30.033 | **-3.52%** (-1.06 ms) | — |

*(Note: Baseline A1/A2 in this table already benefits from the GGML HTP/OpenCL
backend improvements and Talker/sampler updates compiled into the shared binary;
the rightmost column shows the total Code Predictor latency reduction relative
to the pre-optimization 2026-11-09 baseline medians.)*

## Foreground 32-frame complete token-generation measurements (`--mode generate`)

Complete token-generation benchmarks (`Talker` prefill + 32 autoregressive frames
of `Talker` step + 15-step `Code Predictor`, producing 32 × 16 = 512 speech codec
tokens) were executed in verified foreground (`/top-app`) using serial
**Baseline A1 → Candidate B (`both`) → Baseline A2** blocks with 1 warmup call
and 3 timed 32-frame generation calls per block (`9` full 32-frame generations
per cell). Every call produced exactly 32 frames and 512 bit-identical codec IDs:

| Size / backend | Baseline A1 median (ms) | Candidate B (`both`) median (ms) | Candidate B per-frame (ms/frame) | Baseline A2 median (ms) | Reduction vs mean of A1/A2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **0.6B HTP** | 1681.052 | **1612.878** | **50.40 ms** (0.63× RT) | 1761.221 | **-6.29%** (-108.26 ms) |
| **0.6B OpenCL** | 2363.643 | **2309.674** | **72.18 ms** (0.90× RT) | 3302.900 | **-18.48%** (-523.60 ms; **-2.28%** vs A1) |
| **1.7B HTP** | 2306.726 | **2180.630** | **68.14 ms** (0.85× RT) | 2343.410 | **-6.21%** (-144.44 ms) |
| **1.7B OpenCL** | 3367.647 | **2933.357** | **91.67 ms** (1.15× RT) | 3209.812 | **-10.81%** (-355.37 ms) |

At 12.5 Hz (80 ms of audio per frame), **0.6B HTP** generates speech tokens at
**50.40 ms/frame** (**19.84 frames/s**, **1.59× faster than real-time**) and
**1.7B HTP** generates speech tokens at **68.14 ms/frame** (**14.67 frames/s**,
**1.17× faster than real-time**).

## Terminal and trace output

The NPU-Max launcher defaults to **summary** tracing:

```sh
# Outer workspace; choose independently of the predictor options.
QWEN_NPU_MAX_TRACE=summary sh npu-max.sh 0.6b --text 'Hello.'
QWEN_NPU_MAX_TRACE=full sh npu-max.sh 0.6b --text 'Hello.'
QWEN_NPU_MAX_TRACE=off sh npu-max.sh 0.6b --text 'Hello.'
```

- `summary` keeps component placement counts/splits and compute events, without
  serializing every tensor. Placement/terminal summaries deduplicate unchanged
  operation/backend patterns even when KV shapes grow.
- `full` retains tensor shapes, operations, backend assignments and logical
  transfer-edge details for investigation. Direct native users who only set
  `QWEN3_TTS_EXECUTION_TRACE` retain the prior full-detail behavior. To request
  compact native traces, also set `QWEN3_TTS_EXECUTION_TRACE_DETAIL=summary`.
- `off` disables execution trace output. **Strict HTP placement checking and
  finite-logit/audio validation remain enabled by the HTP launcher.** Counts and
  policy are checked on every graph, even after its summary has been cached.
- CLI progress refreshes about four times a second, plus first/last updates.
  Files and pipes receive no carriage-return progress.

## Reproduce and evidence

Run from the application repository, with the existing prepared NPU-Max runtime.
Keep inference serial and builds at two jobs. Use fresh output directories.

```sh
cmake --build .cache/npu-max/build --parallel 2
cmake --build .cache/npu-max/app-build --parallel 2
python tools/npu-max/test-launchers.py
python tools/npu-max/test-vocoder-options.py
python tools/npu-max/test-server-json.py

# 6-cell bit-exact numerical & placement check (0.6B/1.7B x CPU/HTP/OpenCL).
python tools/npu-max/token-generation.py --mode check --candidate both \
  --output .cache/npu-max/reports/token-gen-max-check

# Requires current /top-app throughout.
python tools/npu-max/token-generation.py --mode bench --candidate both \
  --calls 24 --output .cache/npu-max/reports/token-gen-max-bench

# Actual talker + predictor, fixed 32-frame public prompt, zero speaker vector.
python tools/npu-max/token-generation.py --mode generate --candidate both \
  --backends hexagon opencl --calls 3 --output .cache/npu-max/reports/token-gen-max-generate

git diff --check
```

The GGML and application builds succeeded. Launcher checks passed **188/188**,
native CLI parsing **19/19**, server JSON **9/9**, and both full/summary native
trace controls passed. Reviewed portable results are tracked under
[performance/token-generation-20261109](performance/token-generation-20261109/results.json).
