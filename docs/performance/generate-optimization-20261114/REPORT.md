# Qwen3-TTS generation optimization: measured prompt-memory overhead

2026-11-14 — Samsung Galaxy S25 Ultra / Snapdragon 8 Elite / Hexagon v79 / Adreno 830 / Termux.

## Executive finding

The best low-risk opportunity found was **a zero-copy CPU vocabulary view**, rather than persistent Talker graphs. The Talker builds and schedules a graph each frame, but a fresh 96-frame profile measured only 17.9 ms of combined Talker graph construction/allocation in 6,822 ms of synthesis. Meanwhile, text-prompt preparation took 458.3 ms. Full ICL prompt assembly performs seven text projections, and the original scheduler copied the entire **622,329,856-byte F16 text vocabulary** from its HTP buffer into a CPU buffer for each CPU `GET_ROWS` fallback. Exposing a CPU view of the existing HTP shared-memory table removes those full-table copies while preserving CPU lookup and HTP projection math. No new weight allocation is needed.

The selected change preserves the November 13 Mimi optimization and all HTP arithmetic, including the scratchpad layout/alias guards. It does not claim to solve the remaining steady-state autoregressive bottleneck. The final short and long **actual ICL** measurements, with codec-ID/float-audio/WAV equality checks, follow below. Long-run reference preprocessing varies enough that conclusions must be based on the recorded totals, not the isolated projection saving.

## 1. Baseline, repository and runtime

- Application starting HEAD: `3b620ac93e5836fe6194c6e845695cdf6ba6253a`, branch `npu-max-icl-fix`, clean before source edits. This is the validated November 13 baseline, **including** the Mimi SIMD convolution optimization. No comparison here uses the slower pre-Mimi baseline.
- GGML submodule: `717597a2aa208a78b773a146544f4d28825ff08b`, branch `fix/htp-src1-spad-layout-guard`, clean and unchanged throughout.
- Recent application history: `3b620ac` Mimi convolutions; `723f077` listening-validation notes; `c183525` corrected ICL scratchpad reuse; `38b8208` predictor fusions/persistent plans; `0bff39a` reference cache and callback/JSON fixes. Read `AGENTS.md`, `docs/FORK_AUDIT.md`, `docs/TOKEN_GENERATION_AUDIT.md`, the November 13 report and the ICL regression report before edits.
- Existing application Release build: `.cache/npu-max/app-build`, `QWEN3_TTS_TIMING=ON`, `QWEN3_TTS_MIMI_CONV_SIMD=ON`, `QWEN3_TTS_KV_F32=OFF`, `QWEN3_TTS_NPU_TESTS=ON`, Oryon-1 CPU tuning. Rebuilt with two jobs.
- Existing GGML Release runtime: `.cache/npu-max/build`; Hexagon v79 and Adreno OpenCL enabled; SPHAL FastRPC; HMX attention FP16 exp approximation disabled. No GGML, DSP skeleton, SDK or frozen `.cache/npu` rebuild/modification.
- `tools/npu-max/native-env.sh`: HOSTBUF=0, NHVX=6, OPPOLL=1, project host/DSP library paths, vendor OpenCL ICD. Actual tests force HTP transformer/ECAPA/vocoder, MAX vocoder lowering, strict graph placement and finite output validation.
- All five protected-baseline hashes were verified before work and again at delivery. No model server was started, stopped or replaced.

### Existing architecture and confirmed behavior

`TTSTransformer::forward_step` rebuilds its Talker graph and calls scheduler allocation/reset each frame. Attention dimensions and KV views grow with `n_past`; allocation can reuse memory, so reset is not proof of fresh DMA allocation every frame. The Code Predictor already has 15 persistent graphs/schedulers, fused QKV/GateUp projections, strided KV views and host codec-embedding caches. Neither feature is presented as new work.

At 12.5 Hz, every generated frame requires one Talker completion plus one predictor two-token prefill and fourteen dependent predictor steps. CPU sampling/embedding feedback separates these steps. HTP uses shared DDR, packed operation batches and `dspqueue`, not one FastRPC syscall per tensor operation. `ggml_backend_hexagon_graph_compute` flushes pending work; scheduler synchronization calls another flush. Most recorded flushes are **empty** and do not imply a new DSP dispatch. Arbitrarily deleting synchronization without respecting producer/consumer dependencies would be unsafe.

Talker and predictor KV are F16 in this build. There is no implemented quantized KV path in this application. Generation preallocates capacity from prefill plus requested maximum frames. Direct forward APIs can extend/copy the cache, but that physical extension path is not exercised by the full-synthesis benchmarks. The long test exercises growing **logical** KV views and sustained decode beyond 1,024 positions.

The main Talker/predictor, ECAPA and lowered vocoder graphs remain on HTP. `text_projection` intentionally has one CPU `GET_ROWS` plus five HTP MLP operations and two splits. This is an existing strict-placement exception. Mimi, text tokenization, FFT preprocessing, sampling and I/O remain host work. The text table is **F16 even in the Q8_0 model file**; a Q8 model filename is not the table's tensor type. HTP's current `ggml_hexagon_supported_get_rows` accepts F32 source tensors only.

## 2. Read-only profiling before implementation

A fresh foreground CLI profile used the original `3b620ac` executable, 0.6B Q8 model, two host threads, six HVX workers, 96 generated frames, greedy sampling, existing Release TIMING instrumentation and summary placement/aggregate transfer tracing. This initial **plain-TTS diagnostic** selected the target; it is not a substitute for the mandatory ICL benchmarks below.

| Measured component | ms | Interpretation |
| --- | ---: | --- |
| Prompt embedding/prefill preparation | 458.3 | Mostly repeated vocabulary transfer/projection calls |
| Actual Talker prefill forward | 37.3 | 36.4 ms compute |
| Talker steps, all 95 calls | 1,298.1 | 8.3 build + 9.6 alloc + 1,278.9 compute + 0.9 data |
| Predictor, 96 frames / 1,440 graphs | 3,253.7 | 3,228.4 compute; 19.3 one-time plan allocation; 5.3 data/sampling |
| Frame embedding lookup/sum | 0.6 | Existing host embedding cache is effective |
| Other generation overhead | 2.8 | No evidence of a large unaccounted host loop |
| Generate | 5,054 stage / 5,050.8 internal | Timer boundaries differ slightly |
| Decode | 1,768 | Separate lowered HTP vocoder stage |
| Synthesis Total | 6,822 | 7.68 seconds generated; synthesis RTF 0.8883 |

The compute timer includes host backend planning, dispatch, completion waits and actual DSP execution; it must not be labeled pure kernel time. CPU data/sampling is small for this greedy diagnostic. Sampling at nonzero temperature is separately exercised in a seeded correctness control. Per-op DSP profiling, described below, is diagnostic and excluded from A/B timings.

Aggregate counters in this initial run recorded 1,557 dispatched batches, 198,827 operations, 7,818 flush calls / 6,277 empty flushes, and **2,636,677,184 logical tensor-get bytes**. In short ICL, seven vocabulary copies produce **4,504,396,864 logical get bytes**. After placement changes, the short ICL get count is **148,087,872 bytes**, exactly **4,356,308,992 bytes (7 × table size) fewer**. The final view preserves the original weight upload and allocation. Earlier separate-CPU-storage candidates reduced upload accounting but are excluded from the final results. Batch dispatches and dispatched operations are unchanged (1,573 / 210,713 in the 96-frame ICL fixture). There is no hidden reduction in model arithmetic or predictor steps.

Counters measure software tensor API extents, **not physical DDR traffic or energy**. The underlying F16 get operation is a memcpy. Placement's logical transfer-edge summary can report zero because scheduler allocation rewrites inputs to copies; it is not authoritative evidence of no transfer. The aggregate set/get counters and source copy path establish this bottleneck.

## 3. Ranked opportunities

| Rank | Opportunity | Evidence / expected end-to-end effect | Difficulty | Memory cost | Numerical/audio risk |
| --- | --- | --- | --- | --- | --- |
| 1, implemented | Expose the HTP-unsupported F16 text vocabulary through a CPU view | Measured hundreds of ms per request; final full-synthesis results below; fixed prompt cost, diminishing share on long audio | Low–medium | Non-owning wrapper and one metadata tensor; existing shared DDR stays the sole weight allocation | Low: identical lookup and MLP operations, exact controls |
| 2, future | Cache backend-side HTP operation lowering/plans for persistent graphs | Predictor execution dominates; host planning and queue overhead are inside compute. Needs finer unperturbed profiling before a speed forecast | Medium–high, GGML change | Cached descriptors / lifetimes; measure | Medium: graph addresses, copies, alias guards and invalidation |
| 3, future | Reduce complete predictor dispatch/sampling boundaries | 15 sequential graphs/frame; potential steady-state gain, but preserving top-k/top-p/RNG requires substantial device orchestration | High, likely DSP work | Extra code/state | High unless teacher-forced logits and seeded codes remain correct |
| 4, deferred | Persistent/bucketed Talker graphs | At most 17.9 ms saved on the diagnostic, about 0.26% of full synthesis; larger buckets add masked attention work | Medium–high | Bucket plans and padded KV/masks | Medium: dynamic offsets, padded reads, long-context correctness |
| 5, deferred | Remove redundant empty sync calls | Thousands of empty flushes, but they are not thousands of RPC round trips. No measured worthwhile isolated saving | Medium if backend-specific | Minimal | Medium if synchronization ordering changes |
| 6, deferred | More application vector/buffer reuse or sampling tuning | Greedy data/sampling is ~5 ms across the whole diagnostic predictor; limited headroom | Low | Small scratch buffers | Low–medium if RNG order changes |
| 7, research | Quantized KV / alternative attention / lower precision | Long Talker compute rises with context; bandwidth reduction may matter. No new speed/quality evidence | High | Potential reduction | High; preserve existing HMX and ICL fixes |

Already present: predictor plans/fusions, strided KV, shared DDR, mapped-buffer slot capacity fixes, corrected scratchpad quantization reuse, HTP vocoder lowering/chunking, and the Mimi optimization. Larger vocoder windows affect decoding/audio boundaries and are outside this generation change.

## 4. Implementation and source references

### Vocabulary residency and loading

- `src/tts_transformer.h::tts_transformer_model::text_embd_buffer`: owns an optional **non-owning CPU buffer wrapper**, not the weights themselves.
- `src/tts_transformer.cpp::TTSTransformer::load_tensor_data`: loads/uploads every original weight exactly as before. For aligned F16 text tables on HTP only, it queries actual `GET_ROWS` support with metadata-only probe tensors. If unsupported, `ggml_backend_cpu_buffer_from_ptr` wraps the original shared-DDR pointer; a duplicate metadata tensor exposes the same bytes to the CPU scheduler. The original weight tensor stays in `model_.tensors`, backed by the original model buffer. Tests verify that original and alias data pointers are identical and their buffer wrappers differ.
- This is safe for the selected format because HTP F16 set/get is plain memcpy; its bytes already have the CPU's F16 layout. **Repacked quantized formats are excluded**, as are other backends and HTP-supported gathers. CPU alignment is checked before creating the view. There is no memory-mapped-file lifetime, quantization, weight-format or repacking change.
- `TTSTransformer::project_text_tokens` pins the five projection MLP operations to HTP when the CPU view is active. Without this, scheduler CPU placement propagates into the MLP. The initial unpinned candidate failed exactness/strict-placement controls and was rejected. The accepted graph retains one CPU gather and five HTP operations.
- `free_transformer_model` frees and nulls the CPU wrapper **before** the model buffer owning the bytes. `ggml_backend_cpu_buffer_from_ptr_i.free_buffer` is NULL in the pinned GGML revision; freeing the wrapper never frees the underlying shared allocation. Read-only embedding lookup does not write those weights.
- `QWEN3_TTS_HTP_HOST_TEXT_EMBD=0` retains the original graph input. Default enables the view only for eligible, unsupported F16 gathers. CPU/OpenCL, quantized text embeddings, unaligned data and supported HTP formats retain their original behavior. The setting is read at model load; rebuild C++ clients for the added internal model field.

### Reference timing and controls

`src/qwen3_tts.cpp::Qwen3TTS::synthesize_with_voice` adds a timing-only `[ICL reference]` line reporting ECAPA speaker encoding and Mimi encoding separately when `print_timing` is enabled. Public result structures and C ABI are unchanged. Mimi timing includes cache/copy bookkeeping around that block; all measured requests are cold-reference cache misses.

`tools/npu-max/test-htp-text-embedding.cpp` exercises original → CPU-view → original placement on a real model, comparing every projected float bit-for-bit for single, special, repeated, last-vocabulary and multi-token inputs. It also checks empty projection, repeated unload/reload and failed subsequent load. It passed on 0.6B HTP, 1.7B HTP and 0.6B CPU; the final shared-memory-view version was rechecked on both HTP sizes. This is a numerical/lifecycle test, not a microbenchmark speed claim.

`tools/npu-max/test-generation-synthesis.cpp` invokes full `Qwen3TTS::synthesize_with_voice` with actual reference WAV + transcript. It fixes the transformer RNG seed, retains generated codec IDs, reference codec IDs, original float audio and PCM16 WAV, checks finite/non-silent output and exact frame/sample counts, and reports model-load / synthesis wall / API stage times. It is built through the existing NPU tools CMake setup.

The baseline synthesis tool links the archived original `3b620ac` transformer archive and original headers, with the original orchestration source recompiled with **only the same reference timing print**. This avoids mixing old archive layout with the newly extended C++ model structure, and preserves the November 13 Mimi implementation. Both tools use the same unchanged GGML/DSP and component libraries. The original CLI executable is also retained; it supplied the initial diagnostic.

No changes touch `ggml/`, DSP arithmetic, attention precision, cache layout, model weights, scratchpad guards, sampling or vocoder chunk sizes. The application Release rebuild and targeted controls passed. Temporary unpinned and separate-CPU-storage candidates are not the delivered implementation; their raw evidence is retained and excluded from final performance claims.

## 5. Actual ICL benchmark method

All final comparisons use 0.6B Q8 weights + the existing F16/F32 tokenizer, HTP pipeline, 2 host threads, NHVX=6, F16 KV, seed **20261114**, main/sub-talker temperature **0**, top-k 50, top-p 1, English, and default 100-frame vocoder windows with 16-frame replay. No progress/audio playback or server. Every trial starts a new process and re-encodes the reference; OS disk caches are uncontrolled. Output retention/WAV writing and teardown are included in external process wall.

- **Short:** same 3-second cropped `smalldon.wav` fixture and excerpt transcript used on November 13; same short target sentence; 96 generated frames / **7.68 seconds**. Reference encodes to 38 frames.
- **Long:** original existing **23.684-second** `smalldon.wav` and full transcript from `butterverse_longform_queue.sh`; a 486-word neutral target passage; 768 generated frames / **61.44 seconds**. Reference encodes to 297 frames. The fixed frame cap truncates the longer target deliberately to keep workload equal.
- The long streaming-layout prefill has 307 positions (9 framing + 298 reference-codec positions); logical KV length grows to 1,074, with 1,083 positions reserved. These values follow from the observed reference frame count and unchanged prefill builder. Cache view growth and >1,024-position decode are covered; physical cache reallocation is not.
- Runtime monitor samples child cpuset/cgroup, allowed CPUs, RSS and `aoss-0` temperature roughly every 100 ms. Valid timing trials must remain `/top-app`, CPUs 0–7. One earlier long control (`long-A2`) spent 178 samples under `/foreground`, CPUs 0–5; **all its timing numbers are excluded**. Its raw failure record remains retained.
- `Total` in `tts_result` is synthesis only. It includes lazy ECAPA load, speaker/Mimi conditioning, generation, decoding and surrounding orchestration; it excludes initial model load, reference WAV read/resampling before the inner call, output serialization and final teardown. External process wall includes those. RTF is processing seconds divided by **generated** audio duration after reference-prefix trimming, not total decoded reference-plus-generated audio.
- A separate-CPU-storage candidate showed higher process high-water memory during Mimi encoding. It was replaced with the non-owning view, retaining original allocation/loading. All earlier relocation/direct-read candidate timings are diagnostic only; final view results are explicitly distinguished below.

## 6. Final timing, correctness and memory results

| Fixture / trial | Speaker ms | Mimi ms | Generate ms | Decode ms | Synthesis Total ms | External wall ms | Synthesis RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short-A3 | 51 | 1,051 | 5,410 | 2,639 | 9,177 | 11,073 | 1.1949 |
| short-view-B | 53 | 1,065 | 4,686 | 2,733 | 8,571 | 10,776 | 1.1160 |
| short-A4 | 54 | 1,054 | 5,402 | 2,690 | 9,236 | 11,480 | 1.2026 |
| long-A3 | 976 | 8,516 | 44,423 | 22,840 | 76,789 | 78,763 | 1.2498 |
| long-view-B | 1,006 | 7,987 | 42,837 | 22,133 | 74,004 | 76,249 | 1.2045 |
| long-A4 | 960 | 8,164 | 44,185 | 23,098 | 76,449 | 78,853 | 1.2443 |

Comparisons use the mean of the two original controls for each fixture:

- **Short** (7.68 s generated): synthesis **9.2065 → 8.571 s**, saving **635.5 ms / 6.90%**. Synthesis RTF **1.1988 → 1.1160**. External wall **11.276 → 10.776 s** (4.44% lower), process RTF **1.4683 → 1.4031**. Generate **5.406 → 4.686 s**.
- **Long** (61.44 s generated): synthesis **76.6190 → 74.004 s**, saving **2,615.0 ms / 3.41%**. Synthesis RTF **1.2471 → 1.2045**. External wall **78.808 → 76.249 s** (3.25% lower), process RTF **1.2827 → 1.2410**. Generate **44.304 → 42.837 s**.

The short total saving is stable relative to its close baseline controls. The observed long total improves as well, but its magnitude is **not wholly attributable to the vocabulary view**: unchanged Mimi and vocoder phases were faster in that candidate trial too. Dynamic clocks, cache/thermal history and unmeasured operating-system effects remain. The source/counter evidence establishes removal of the full-table copies, and measured projection compute falls by hundreds of ms; it does not establish a universal 3.4% sustained-throughput gain. One candidate trial per final bracket gives no statistical confidence interval. Earlier invalid/background and superseded-candidate results remain in `results.json`, clearly excluded.

### Memory and ownership

The original HTP model buffer remains **1,374,950,400 bytes** on 0.6B and **2,484,589,568 bytes** on 1.7B in both modes. The CPU wrapper advertises the table's 622,329,856-byte span but owns **zero weight bytes**. Its pointer is identical to the original F16 tensor's pointer. Additional storage is only a wrapper and tensor metadata. The CPU scheduler no longer allocates/copies a second complete table for prompt assembly.

| Trial | Model load ms | Sampled peak RSS MiB | Application stage-boundary RSS peak MiB | aoss-0 min–max °C |
| --- | ---: | ---: | ---: | --- |
| short-A3 | 1678.7 | 1227.2 | 1226.8 | 44.4–71.4 |
| short-view-B | 1949.1 | 719.4 | 633.0 | 39.3–70.2 |
| short-A4 | 1955.5 | 1227.5 | 1227.1 | 41.3–69.5 |
| long-A3 | 1595.7 | 1246.3 | 1244.1 | 46.7–74.5 |
| long-view-B | 1973.4 | 1019.7 | 649.8 | 32.4–75.3 |
| long-A4 | 2097.3 | 1246.1 | 1243.9 | 39.3–72.6 |

These are process RSS observations, not full-device memory accounting or DSP allocation telemetry. The 100-ms sampler can miss brief peaks, and the application's RSS peak is sampled only at stage boundaries. On Android this source's `phys` log is actually VmHWM, not Apple-style physical footprint. No reduction in **total model weight bytes** is claimed. The earlier separate-CPU-storage candidate's larger process high-water memory was a reason to replace it with the view. Final short sampled RSS is roughly 719 MiB versus 1,227 MiB baseline; long sampled RSS roughly 1,020 MiB versus 1,246 MiB baseline.

### Correctness and tested scope

All final short A/B/A outputs are bit-identical across **1,536 generated codec IDs**, **608 reference codec IDs**, **184,320 float samples** and the full WAV file. All final long A/B/A outputs are bit-identical across **12,288 generated IDs**, **4,752 reference IDs**, **1,474,560 float samples** and the full WAV. A separate seed-20261114 / temperature-0.9 control also matches all 128 generated IDs, float samples and WAV bytes over eight frames. The latter is a correctness control, not a speed benchmark.

Both output fixtures are valid mono 24 kHz PCM16 WAVs. Short PCM RMS is 0.0177525584, peak 0.2610473633; long PCM RMS is 0.0310012273, peak 0.2872009277. The harness rejects empty, non-finite and silent float output and mismatched frame/sample counts. Exact float and waveform equality gives no evidence of an audio-quality regression on these fixtures; no new subjective listening claim is made.

Exact short float SHA-256: `3db04194cf120825e4dd3eb76b972256240249d436ae2649a6bcf21047f687d4`.
Exact long float SHA-256: `418df1acf6869f1cd52de99748371173c1e67b7e41c70be633f8832447fc082e`.
The full token/WAV/reference/model/library/binary hashes are in adjacent `results.json`.

The original November 13 short reference and prompt produce the **same PCM** as that validated run (PCM payload SHA-256 `6f42537140a14fed4c7362b60cd7bf9d1c1184cadcce56ec157a5b5c56f37c4c`). The benchmark's original archive is `3b620ac`, not the pre-Mimi archive. November 13 absolute timing is historical context; only the freshly measured controls establish this change's delta.

## 7. DSP execution versus host dispatch

The first attempt to use `GGML_HEXAGON_PROFILE=1` through the CLI did not expose the per-op records because `src/main.cpp::ggml_log_filter` suppresses debug output. The full-synthesis validation tool keeps default logging, so a separate **eight-frame actual ICL** run captured those records with PROFILE=1. Its times are **excluded from performance comparisons**.

Within Generate, the instrumented run recorded **137 DSP batches**, summed host batch intervals **531,715 µs**, and summed DSP operator times **513,724 µs**. Approximate operator composition:

| DSP op group | Count | Sum ms | Share of operator time |
| --- | ---: | ---: | ---: |
| MUL_MAT | 4,832 | 422.222 | 82.19% |
| CONT | 1,004 | 35.856 | 6.98% |
| FLASH_ATTN_EXT | 756 | 17.020 | 3.31% |
| RMS_NORM+MUL | 3,650 | 12.325 | 2.40% |
| ROPE | 1,760 | 7.753 | 1.51% |
| CPY | 1,760 | 4.052 | 0.79% |
| SWIGLU | 796 | 3.730 | 0.73% |

The roughly 18-ms difference between summed host batch intervals and DSP op time is **not an isolated FastRPC cost**. The host interval begins inside queue preparation; scheduler work, graph lowering, descriptor preparation and CPU sampling exist outside it, and per-op logging perturbs execution. The full instrumented Generate stage is 1,272 ms, including text-table copies and logging; subtracting kernel sums from it would mix unrelated overhead. This diagnostic supports matmuls as the main DSP work and shows why empty-flush counts alone do not justify synchronization changes. A clean host planner/queue profile at long KV lengths is the next discriminating measurement.

## 8. Reproduction and next steps

Run from `/data/data/com.termux/files/home/qwen3-tts-workspace/repos/Markon101-qwen3-tts.cpp`:

```sh
cmake --build .cache/npu-max/app-build --parallel 2
QWEN3_TTS_MAX_STRICT=1 sh tools/npu-max/native-env.sh \
  .cache/npu-max/app-build/tools/npu/npu-max-test-htp-text-embedding \
  ../../models/qwen3-tts-0.6b-q8_0.gguf hexagon

# Local retained full-synthesis comparisons; wrapper requires /top-app at start
# and records placement throughout. Substitute fresh output labels.
python .cache/generate-opt-20261114/bench.py repeat-short-A \
  .cache/generate-opt-20261114/before-synthesis short
python .cache/generate-opt-20261114/bench.py repeat-short-B \
  .cache/npu-max/app-build/tools/npu/npu-max-test-generation-synthesis short
python .cache/generate-opt-20261114/bench.py repeat-long-A \
  .cache/generate-opt-20261114/before-synthesis long
python .cache/generate-opt-20261114/bench.py repeat-long-B \
  .cache/npu-max/app-build/tools/npu/npu-max-test-generation-synthesis long
```

The tracked synthesis tool can be used without the local Python wrapper. Set the HTP/strict/FFT/MAX/validation environment listed in each result's `environment`, then invoke it through `native-env.sh` with:

```text
MODEL TOKENIZER REFERENCE_WAV REF_TRANSCRIPT_FILE TARGET_TEXT_FILE OUTPUT_PREFIX MAX_FRAMES SEED TEMPERATURE
```

Use the same executable with `QWEN3_TTS_HTP_HOST_TEXT_EMBD=0` as an ongoing original-placement control; change the flag before loading the model. The archived baseline additionally preserves the exact prior transformer implementation. Target passages are retained beside this report; reference WAVs/transcripts remain private in the workspace. All exact commands are retained in `results.json`.

Recommended next work:

1. Profile HTP graph lowering/descriptor preparation separately from DSP execution, without per-op logging in the timing phase. Compare the first and late steady decode positions at matched KV lengths, and identify whether persistent backend descriptor plans offer enough end-to-end headroom to justify a GGML change.
2. Preserve the `717597a2` scratchpad layout/alias guard, HMX long-context fixes and raw teacher-forced controls. If cached backend plans are attempted, validate buffer/shape/stride lifetime and invalidation across prefill, predictor steps, model reload and batch slots; never cache activation quantization merely by pointer identity.
3. Examine long-context attention/KV traffic with same-state CPU/HTP controls. Quantized KV or alternative attention is separate numerical research; do not infer safe speech from allocations or a successful short prompt.
4. Consider device-side sampling only with full seeded top-k/top-p controls and demonstrated savings from removing completion boundaries. Optimizing the tiny host sampler alone cannot remove fifteen dependent decoder evaluations.
5. Keep Talker graph buckets low priority unless a future profile changes the measured headroom. Do not accept extra masked attention work or memory for a fraction-of-percent build/alloc saving.

Limitations: two specific ICL fixtures, one final candidate run per bracket, one handset, foreground monitored but clocks/thermal state not fixed. The long sample lasts 61.44 seconds, not the prior 120 seconds, and is capped before finishing the whole target passage. Physical KV extension, other model families/voices, 1.7B full ICL, OpenCL speed and background speed were not benchmarked. No energy claim or broad all-HTP process claim is made. Existing reference caches and server amortization are unchanged.

Raw evidence is retained in `.cache/generate-opt-20261114/`: original executable/archives/headers, original-source timing-only orchestration build, exact fixture texts, all intermediate/rejected/final logs, per-line timestamps, runtime placement/RSS/temperature samples, raw codec IDs and float/WAV outputs, profiler summaries, build logs and protected checks. Large/private artifacts and all WAVs remain ignored. The full reference transcript is not committed. The application/GGML source worktrees and shared-storage export are checked at delivery.

## Source navigation

- `src/tts_transformer.h:190` — Non-owning wrapper field.
- `src/tts_transformer.cpp:998` — Eligible shared-memory view and support/alignment guards.
- `src/tts_transformer.cpp:1467` — Preserved HTP projection placement.
- `src/tts_transformer.cpp:4802` — Wrapper-before-owner cleanup.
- `src/qwen3_tts.cpp:1225` — Separate reference encoding timer.
- `ggml/src/ggml-hexagon/ggml-hexagon.cpp:3121` — Backend F32-only GET_ROWS support.
- `ggml/src/ggml-backend.cpp:2285` — Non-owning CPU wrapper: free_buffer is NULL.
- `ggml/src/ggml-hexagon/ggml-hexagon.cpp:1713` — HTP F16 default memcpy get path.
