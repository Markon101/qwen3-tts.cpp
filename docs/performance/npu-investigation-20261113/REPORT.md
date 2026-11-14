# Qwen3-TTS Snapdragon v79 investigation

Date: 2026-11-13. Device: Samsung Galaxy S25 Ultra, Snapdragon 8 Elite, Adreno 830, Hexagon v79. This report separates source findings, historical evidence, user-supplied timings, and fresh measurements. Lower real-time factor (RTF = processing seconds / output audio seconds) is better.

## 1. Executive summary and current fork architecture

The fork already has extensive HTP acceleration. Adding generic NPU selection, fused Code Predictor projections, persistent Code Predictor graphs, or HTP vocoder offloading would duplicate existing work. The clearest overlooked work is **CPU Mimi reference encoding before ICL generation**: its scalar convolutions are outside every printed component timer, while included in the ICL synthesis Total. Optimizing this path is a low-risk way to reduce cloned-request latency without changing DSP kernels, weights, sampling, or vocoder chunk boundaries.

**Critical timing correction:** the supplied 211,110 ms Total appears to be the CLI's `tts_result.t_total_ms`. That is synthesis time, not process wall time. Initial model loading/repacking, final backend teardown, and WAV output happen outside it. They cannot account for a gap *inside* that Total. If the supplied Total was instead measured externally, those costs would also contribute. The original command, log, model size, reference duration, and trace were not provided; no exact retrospective split of 79,955 ms is possible. This audit does not fabricate one.

### Checkout and build provenance

- Repository: `/data/data/com.termux/files/home/qwen3-tts-workspace/repos/Markon101-qwen3-tts.cpp` (workspace root itself is not a Git repository).
- Initial application HEAD: `723f077`, branch `npu-max-icl-fix`; clean, including untracked-file inspection.
- GGML submodule: `717597a2aa208a78b773a146544f4d28825ff08b`, branch `fix/htp-src1-spad-layout-guard`, clean.
- `723f077`: records user listening validation of the ICL fix. `c183525`: pins corrected HTP scratchpad reuse and ICL regression evidence. `38b8208`: fused QKV/GateUp, persistent predictor plans, compact tracing. `0bff39a`: strict server JSON, ICL callback trimming, voice reference cache, portable CPU tuning. `ac1e339`: controllable vocoder windows. `24bf404`: HTP graph placement and channels-first vocoder lowering.
- Existing Release application build: `.cache/npu-max/app-build`; `QWEN3_TTS_TIMING=ON`, `QWEN3_TTS_NPU_TESTS=ON`, `QWEN3_TTS_KV_F32=OFF`, `QWEN3_TTS_AARCH64_MCPU=auto`. Actual flags include `-O3 -DNDEBUG -mcpu=oryon-1+fp16+fp16fml+dotprod+i8mm+bf16`.
- Linked GGML build: `.cache/npu-max/build`, Release; `GGML_HEXAGON=ON`, architectures `v79`, `GGML_HEXAGON_USE_SPHAL=ON`, `GGML_HEXAGON_FA_EXP2_HF=OFF`; `GGML_OPENCL=ON`, Adreno kernels ON, embedded kernels ON, OpenCL profiling OFF. SDK 6.6.0.0 / Hexagon Tools 19.0.07. Application CMake consumes a separately built GGML; its GPU detection messages alone do not prove HTP availability.
- All five `.cache/npu-max/protected-baseline.json` hashes matched before work. Frozen `.cache/npu` binaries/launchers are separate from `.cache/npu-max` and must stay untouched.

### Backend and runtime map

`tools/npu-max/launch.py` selects explicit component backends. `src/gguf_loader.cpp::init_preferred_backend` resolves `hexagon` to `HTP0`, `opencl` to `GPUOpenCL`, and fails when an explicitly requested device is unavailable. Named component backends bypass the automatic shared GPU singleton. Non-CPU components have CPU scheduler fallbacks; strict placement checks reject CPU/other compute nodes in the main HTP graphs, with explicit exceptions for CPU `GET_ROWS` during prompt embedding/text projection assembly.

| Stage | Normal `htp` profile | Remaining host work / boundaries |
| --- | --- | --- |
| Tokenize | CPU BPE (`text_tokenizer.cpp`) | Small text vectors; Q8 text vocabulary `GET_ROWS` remains CPU (text projection MLP on HTP); ICL tokenization is currently not timed |
| Encode speaker | HTP GGML graph (`audio_tokenizer_encoder.cpp::encode`, convolution lowering) | WAV read/resample, CPU FFT/mel preprocessing, model load before Encode timer |
| Encode ICL reference | **CPU** (`mimi_encoder.cpp::encode`) | Scalar causal Conv1D, CPU GGML matmuls, norms/RoPE/softmax, scalar RVQ; not represented by printed Encode |
| Generate Talker | HTP (`TTSTransformer::forward_prefill`, `forward_step`) | Graph build/alloc each Talker step, host sampling, hidden/logit reads, next-frame embeddings |
| Generate 15-step predictor | HTP, 15 persistent plans | One 2-token prefill plus 14 dependent single-token graphs; host logit reads/sampling and input uploads between steps |
| Decode | HTP lowered graph (`AudioTokenizerDecoder`) | Build/schedule once per chunk, code uploads, output read, overlap replay, ICL reference prefix decoding/trimming |

`mixed` selects HTP transformer/ECAPA and OpenCL decoder with MAX vocoder lowering disabled. `opencl` selects OpenCL components; Mimi remains CPU in every profile. `cpu` is explicit. Automatic preference without selectors is IGPU → GPU → ACCEL → CPU; do not assume that is HTP.

`tools/npu-max/native-env.sh` sets the host libraries in `LD_LIBRARY_PATH`, DSP skeleton in `ADSP_LIBRARY_PATH`, vendor OpenCL ICD, `GGML_HEXAGON_HOSTBUF=0`, NHVX=6, OPPOLL=1, PROFILE=0 and an OpenCL binary cache. It unsets LD_PRELOAD. These are FastRPC/ggml HTP kernels, not QNN graph compilation. Vendor FastRPC is loaded through the SPHAL path (`ggml/src/ggml-hexagon/htp-drv.cpp`); the project supplies `libggml-htp-v79.so`.

## 2. Stage-by-stage bottleneck analysis: supplied 120-second run

| Item | Supplied ms | RTF contribution | Share of synthesis Total |
| --- | ---: | ---: | ---: |
| Tokenize | 0 | 0 | 0% |
| Encode | 757 | 0.0063 | 0.36% |
| Generate | 88,971 | 0.7414 | 42.14% |
| Decode | 41,427 | 0.3452 | 19.62% |
| Unassigned remainder | 79,955 | 0.6663 | 37.87% |
| Total | 211,110 | 1.7593 | 100% |

These are user-supplied numbers, not reproduced measurements. 1,500 frames at 12.5 Hz represents 120 seconds. Cloned output after reference trimming can be shorter than decoded full-code duration.

### Generate: 59.31 ms per generated frame

`src/tts_transformer.cpp::ensure_code_pred_plans` stores 15 separate metadata buffers, graphs, I/O tensor handles and preallocated schedulers, keyed by KV buffer identity, batch slot and fusion flags. `predict_codes_autoregressive` uses them when either default optimization flag is enabled. Setting both `QWEN3_TTS_CODE_PRED_SWIGLU=0` and `QWEN3_TTS_CODE_PRED_STRIDED_KV=0` selects the rebuild baseline. Persistent plans already remove predictor graph allocation on every substep; adding that optimization again cannot explain a new gain.

`forward_step` **does rebuild** the Talker graph and invokes sched_alloc_graph/reset every frame. Its attention views grow with `n_past`; persistent metadata alone is not sufficient to reuse an allocated plan with different dimensions. Buffer memory can be retained by the scheduler, so a scheduler reset does not necessarily imply a fresh DMA allocation each frame. Existing TIMING counters separate graph build, alloc, compute and data. The old AGENTS.md performance proportions predate the fusions and must not be treated as current measurements.

Talker KV is F16 by default, F32 behind `QWEN3_TTS_KV_F32`; no quantized KV path is implemented here. Generation reserves context based on prefill and max frames, while direct forward calls can extend by 512 positions with a cache copy. Predictor KV uses 16 positions and is cleared per frame; strided views remove redundant contiguous K/V copies. Host-cached codec embedding rows avoid repeatedly unpacking REPACK weights to feed autoregression. Fused predictor QKV/GateUp weights add approximately 55.7 MB resident RAM (historical audit).

There are approximately **24,000 graph completions** for 1,500 frames (one Talker plus 15 predictor graphs each, plus setup). `ggml_backend_hexagon_graph_compute` converts operations to HTP batches and flushes; `ggml_backend_hexagon_synchronize` also calls flush. This is not one FastRPC syscall per GGML op: operations use shared-memory `dspqueue` batches. Host sampling makes the 15 predictor graph completions sequential. Read/write API byte counts are logical shared-DDR accesses, not measured physical PCIe-style transfers. Shared DDR avoids a separate GPU-style physical-memory domain but does not remove dispatch, cache coherence, copies or completion waits.

HTP `HTP_MAX_MMAPS=32`, interbatch spin-poll, fused SwiGLU, strided KV and dynamic input quantization reuse are already implemented. **Preserve the 717597a2 layout/alias guard:** pointer/type/stride-only activation reuse caused real ICL divergence. Removing it to gain speed is unsafe. Persistent plans need compatible buffer mapping capacity; slot eviction was already fixed.

### Decode: existing HTP acceleration, substantial graph size and overlap

`src/audio_tokenizer_decoder.cpp::prepare_max_weights` uploads transformed constants once. `max_conv` lowers causal Conv1D into per-tap matmuls/views/concats, tiling at **1,024 rows**. `max_depthwise_conv` uses per-tap elementwise operations. `max_transpose_conv` uses polyphase tap matrices rather than unsupported native transposed convolution. Snake and attention are lowered to supported GGML operations. Channels-first layouts reduce transposes. Thus native Conv1D support in the backend is not a prerequisite for the current HTP vocoder.

`AudioTokenizerDecoder::decode` recursively splits long sequences. Default window=100 total frames, left context=16, step=84 new frames. Values 17..256 are already exposed through CLI/C++/server (`--vocoder-chunk-frames`). A 1,500-frame non-ICL input invokes 18 chunk graphs and computes 1,772 effective frames, approximately **18.1% repeated frame work**. ICL additionally prepends reference frames, decodes them, and trims their samples. Relative to 1,500 generated frames it therefore costs more; exact count depends on reference length. Callback decoding can use still smaller emitted chunks (default 12), increasing replay further.

Each chunk rebuilds/schedules a large vocoder graph; full 256-frame controls historically reached 43,048 HTP compute nodes. `decode` already prints `[Vocoder Breakdown] build_graph / alloc / inputs / compute / tensor_get / total`. Dynamic tail lengths rebuild graphs; they do not intrinsically cause CPU fallback when the strict HTP lowering is used. Scheduler splits and trace placement, rather than a banner, establish placement. CPU fallback may occur under the legacy launcher or other profiles and must be checked on that run.

Window=256 reduces the example to 7 graphs and 1,596 effective frames (6.4% replay). It reduces repeated frame work by about 9.9% relative to window=100, **not a measured 9.9% Decode speedup**: attention cost, memory and graph construction also change. Chunking already exists; persistent vocoder state does not. Overlap is an approximation, not full-history equivalence, and changing windows changes numerical audio. Historical same-code controls reported around 0.003 HTP-vs-CPU waveform NRMSE but appreciable chunk/full-history differences. Do not change the default on timing alone.

### The missing 79,955 ms: timer boundaries matter

`src/main.cpp::main` loads all initial models **before** synthesis. `run_once` writes/resamples the WAV **after** synthesis returns and prints the stored Total. C++ object destruction/backend frees happen later. These costs affect external process wall time but not this Total.

`Qwen3TTS::synthesize_with_voice(samples)` starts Total before `ensure_encoder_loaded`. That lazily loads the speaker encoder from the TTS GGUF. Encode starts only afterward and measures ECAPA extraction. The subsequent **Mimi encode, reference tokenization, cache comparison/copy and some bookkeeping have no stage timer**. Generate starts immediately around `transformer_.generate_icl`. Decode times `audio_decoder_.decode`; ICL trimming/copy and validation happen around/outside it. On the ICL path `Tokenize` remains default 0 even when tokenization ran. This makes CPU Mimi the leading explanation for a large in-Total residual, rather than initial HVX repacking or final teardown. Fresh substage evidence is recorded below; the original exact 80-second attribution remains unproven.

Mimi `apply_causal_conv1d` loops time → output channel → kernel → input channel with scalar accumulation. Large strided weight reads and serial multiply/accumulate dependencies dominate candidate CPU work. `run_transformer` also creates a fresh GGML context and copies weights/input for each projection and attention-head matrix multiply; it creates a CPU backend once per reference but no explicit persistent worker pool in this path. Norms, attention mixing and RVQ include host loops. Reference cost scales with **reference audio**, not 120-second generated output length.

`Qwen3TTS` already caches the most recent speaker embedding and Mimi codes for references ≤30 seconds using model paths plus exact normalized-sample comparison. A repeated in-process request can skip Mimi. CLI processes do not share this cache. `--server` already keeps model/runtime state resident, but no server was started in this audit.

### Loading, memory and teardown outside synthesis

- All available models resolve under Termux private HOME (`../../models`), not `/sdcard` FUSE; export reports/audio to `/sdcard`, keep model assets private for inference.
- `TTSTransformer::load_tensor_data` uses `fopen/fseeko/fread` and a reusable heap buffer per tensor, then uploads. Fused predictor constants reread their component tensor ranges. It is not mmap based.
- Shared `load_tensor_data_from_file` (ECAPA, decoder, Mimi) maps the file and calls `ggml_backend_tensor_set` into an allocated backend buffer, then unmaps. Despite comments suggesting CPU zero-copy, the implementation still **copies into that buffer**; it does not retain file-mapped weight pointers.
- HTP `ggml_backend_hexagon_buffer_set_tensor` converts supported quantized types to tiled REPACK layout (e.g. Q8_0 → Q8x4x2). No durable prepacked-weight cache is implemented. F16/F32 vocoder weights also have separately prepared constants. mmap alone does not remove repacking or transformed constant preparation.
- `ggml_hexagon_buffer` allocates shared DDR via `rpcmem_alloc2` (fallback `rpcmem_alloc`), gets a DMA-BUF FD and calls `fastrpc_mmap`; freeing unmaps/frees. Scheduler reuse limits repeated allocation; it must be measured rather than assumed.
- `release_preferred_backend` reference-counts automatic backends, while named ones are released directly. HTP session lifecycle is registry/device scoped; `ggml_backend_free` alone is not evidence that every device session was recreated. `ggml_hexagon_session::release` stops the interface, destroys opbatch/queue, closes dspqueue and FastRPC handle. There is no timing proving an 80-second teardown here.
- Existing `GGML_HEXAGON_TRANSFER_TRACE` counters expose logical set/get/repack bytes, dispatches and flush/empty-flush counts. They explicitly do **not** measure physical DDR traffic. PROFILE/VERBOSE traces perturb timing; collect placement/profiling separately from the matched performance run.

## 3. Implemented vs potential improvements matrix

Expected gains below are conditional estimates or ceilings, not promises. Rank considers leverage *and* low quality risk for this request.

| Rank | Optimization | Already implemented? | Expected RTF impact | Complexity | Quality risk |
| --- | --- | --- | --- | --- | --- |
| 1 | Vectorize independent Mimi convolution output channels, preserve reduction order | Implemented by this commit | Measured 29.3% synthesis reduction / 0.514 RTF reduction on the 7.68-second ICL fixture; benefit depends on reference cost; no benefit to cached/non-ICL requests | Low–medium | Low with bit-exact tensor/code controls |
| 2 | Reuse loaded process and existing reference cache | Yes, server + one-reference cache | Can avoid cold loads/repack and Mimi entirely on repeated requests; not a new code optimization | Low operationally | Low; validate request/cache isolation |
| 3 | Persistent Mimi matmul context/CPU pool; vectorized RVQ | No | Remaining reference-only work after convolution fix; profile before prioritizing | Medium | Low–medium numerical risk |
| 4 | Persistent stateful vocoder (attention + conv + upsampler history) | No | Remove ~18% repeated frame work at default window, plus setup; cannot assume proportional speedup | High | High until chunk/full parity |
| 5 | Larger vocoder windows (e.g. 200/256) | Yes, opt-in | Roughly 10% less repeated frame work at 256 in the 1,500-frame example; actual wall speed uncertain | Low use, medium validation | Medium: boundaries/attention history differ |
| 6 | Talker graph reuse with fixed-capacity views/masks or shape buckets | No; predictor reuse already done | Only measured Talker build/alloc portion; increasing masked work can erase gain | Medium–high | Medium: KV offsets/masks must stay correct |
| 7 | Remove host predictor sampling/embedding round trips | No | Up to thousands of completion boundaries saved; DSP sampling must reproduce seed/top-k/top-p | High | Medium–high |
| 8 | Persistent mapped/prepacked weight cache | No durable cache; in-process residency exists | Startup-only; cannot reduce an in-synthesis ICL residual attributed to Mimi | Medium–high | Medium: layout/version/hash invalidation |
| 9 | Quantized KV, speculative or fewer RVQ steps, lower precision vocoder | No | Potential steady-state gains, unmeasured | High | High; not suitable first change |
| — | Explicit HTP selection, NHVX=6, HMX attention, fusions, 15 predictor plans, FFT ECAPA, strict placement, HTP vocoder lowering | Yes | Preserve existing gains; no new speed claim | Existing | Keep ICL guard and negative controls |

## 4. Recommended optimization roadmap

### Qwen3-TTS

1. Land the independent-channel Mimi convolution optimization with portable scalar fallback and `QWEN3_TTS_MIMI_CONV_SIMD=OFF` control; retain exact scalar accumulation order, bias, causal padding, stride/dilation and output lengths. Compare original/candidate full reference codec IDs and complete WAV samples, not merely RMS.
2. Add explicit future timing fields for model-load phases, lazy ECAPA load, Mimi conv/transformer/downsample/RVQ, ICL tokenization, prefix trimming, WAV I/O and teardown. Keep synthesis Total separate from externally measured process wall. Updating the public/C API timing schema is separate work.
3. Profile current corrected HTP binary at representative long KV lengths. Aggregate existing TIMING and vocoder breakdowns; capture strict full placement and dispatch counters separately. Keep the 717597a2 src1 layout guard. Optimize Talker setup only if counters show material cost.
4. Benchmark persistent process/repeated exact-reference requests separately from cold process runs. Never silently start a server; use the existing server only when explicitly desired. For true cold disk claims distinguish OS page cache from process startup; this audit does not drop caches.
5. Evaluate optional larger vocoder windows on retained identical codes, memory, first-audio delay and listening at each boundary. Then implement real persistent causal state before claiming streaming/full equivalence or attempting Generate/Decode overlap on the same DSP.
6. Consider persistent Mimi CPU worker/context reuse and RVQ after the first fix; move Mimi to GGML/HTP only with layerwise CPU oracle controls. Do not assume NPU quantized convolutions retain current reference codes.

### Porting patterns to MOSS / transcribe.cpp

A local implementation exists at `/data/data/com.termux/files/home/projects/transcribe.cpp`, `src/arch/moss/{encoder,decoder,weights,model}.cpp`; inspected read-only. It is a different repository, with an existing untracked `scripts/transcribe_recordings.py` left untouched. `docs/porting/families/moss.md` describes Whisper-Medium audio encoder → 4× merge/VQAdaptor MLP → Qwen3-0.6B decoder. This is **not** a TTS RVQ predictor or waveform vocoder; do not port their 15-plan schedule/chunk semantics blindly.

1. Audit that checkout's own GGML revision, CMake/backend selection and weight loader before importing any backend patches. Reuse explicit named HTP selection, CPU fallback, strict observed-placement traces, shared-DDR allocation and lifetime discipline. Follow its `transcribe-backend.h` safe teardown/API exception guards.
2. Lower Whisper Conv1D and adaptor matmuls to supported tiled HTP operations; keep mel preprocessing on CPU initially and measure it separately. Enforce actual matmul row/type/stride bounds; bucket/pad encoder lengths with correct masks, not arbitrary truncation. Keep normalization eps, GELU and attention numerics correct.
3. Port Qwen3 decoder projection fusions/strided KV and device-resident weights after tensor-level parity. Decoder lengths vary: static shape plans require mask/KV-offset validation. TTS predictor fixed-16-token caches do not generalize directly.
4. Use the corrected HTP shared-input quantization guard and mmap-slot capacity rules; preserve src0/dst extent/stride checks and no-alias requirement. Validate long audio-prefill and several post-prefill decode positions, where the TTS ICL bug appeared.
5. Build `transcribe-cli`, run its cheap MOSS preflight, then `uv run scripts/validate.py all --family moss --variant moss-transcribe-diarize` with CPU/reference tensor dumps. Gate on transcript tokens, WER/CER, timestamp and speaker-marker parsing on English/Chinese and long clips, not nonempty output alone. Only then run matched foreground cold/warm timing. No MOSS code changes or performance claims are part of this commit.

## 5. Fresh verification and A/B measurements

Read-only source audit completed before application source edits. Initial protected hashes matched. Initial runtime was `/top-app`, later changed to `/moderate`; foreground was requested for performance measurement. Report export was tested before implementation. The initial report deliberately precedes the chosen code change; final numerical and timing evidence is appended here after verification.

### Implemented change

`src/mimi_encoder.cpp::MimiEncoder::apply_causal_conv1d` now packs weights in eight-output-channel blocks and accumulates eight independent outputs together. This exposes SIMD to the compiler without changing the kernel/input-channel reduction order. No fast-math, thread creation, model format change, DSP patch, quantization or chunk-size change is involved. Input channels divisible by 16 use the new path; irregular input channels, the initial one-channel convolution and remaining output channels retain the original loop. The local packed buffer holds one convolution's weights and is released on return; this trades temporary memory/packing for much less scalar accumulation overhead.

The code preserves **both** original compiler behaviors on this Release configuration: pointwise contiguous weights use rounded products followed by ordered adds, while strided kernel weights use FMA. The initial naive lane-vectorization failed exactness controls because it fused the former; the final implementation separates that product/add. Exact rounding is compiler/flags-sensitive, so do not claim cross-compiler bit identity without rerunning the oracle. CMake option `QWEN3_TTS_MIMI_CONV_SIMD` defaults ON; OFF compiles the original scalar loop. Tests explicitly validate both builds.

### Controlled short end-to-end A1 → B → A2

Original: preserved executable from `723f077`. Candidate: rebuilt current Release implementation, same corrected GGML shared runtime, same weights, same native environment. Each trial starts a new CLI process; OS file cache was **not** dropped. Serial runs, two host threads, NHVX=6, fixed greedy Talker and sub-talker sampling, max 96 generated frames, default vocoder window=100, three-second reference cut from existing `smalldon.wav`. Output is **96 frames / 7.68 seconds**, mono 24 kHz PCM16. The reference has 38 Mimi frames / 608 codec IDs. The prompt is “Hello. This is a short test of clear speech, with a steady pace and a natural voice.” Reference transcript is a short approximate matching excerpt; these are deterministic arithmetic controls, not a listening evaluation of transcription alignment.

| Trial | Tokenize ms | Encode ms | Generate ms | Decode ms | Synthesis Total ms | Process wall ms | In-Total residual ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original A1 | 0 | 57 | 5,559 | 2,754 | 13,490 | 16,415.2 | 5,120 |
| Candidate B | 0 | 52 | 5,659 | 2,708 | **9,515** | **11,982.4** | **1,096** |
| Original A2 | 0 | 56 | 5,484 | 2,722 | 13,431 | 15,912.7 | 5,169 |

Relative to mean original Total **13,460.5 ms**, candidate saves **3,945.5 ms (29.31%)**. Synthesis RTF falls from **1.7527 to 1.2389**. Mean external process wall falls from **16,163.95 to 11,982.44 ms (25.87%)**, but that comparison also includes startup/cache variability; the measured synthesis reduction is the cleaner result. Generate/Decode are effectively unchanged within run variation; the change accelerates work outside their timers. These are three short runs, not a sustained or 120-second speed claim. Original A1/A2 Total differ by only 59 ms; no statistical confidence interval is implied by one candidate trial.

Foreground was verified every ~100 ms in the child process: **163/163 A1, 119/119 B, 158/158 A2 samples were `/top-app`, CPUs 0–7**. User foreground confirmation was received during final verification. Earlier failed start attempts refused `/moderate` without launching inference. Diagnostic standalone Mimi runs occurred in `/foreground-boost`, CPUs 0–7, and are separated below. Thermal zone 0 (`aoss-0`) ranged A1 37.0–67.2°C, B 45.9–71.0°C, A2 45.9–71.4°C; these are raw sensor readings, not battery temperature or a thermal-stability guarantee. Battery temperature was permission-denied. No sustained cooling policy or energy measurement was attempted.

### What the fresh evidence says about missing time

Timestamped stdout/stderr shows A1 model loading **2,690 ms**, B **2,251 ms**, outside synthesis Total. Lazy speaker model loading is A1 **37 ms**, B **40 ms**, inside Total but outside Encode. The gap between `synth/start` (after ECAPA) and `synth/after-tokenize` is **5,080 ms A1 vs 1,056 ms B**. It contains Mimi reference encode and very small ICL tokenization/setup. The overall unassigned in-Total remainder drops from **5,120 to 1,096 ms**. This independently localizes the improvement to omitted ICL preprocessing, rather than Generate or backend teardown.

A separate stage harness, using original vs candidate encoder archives on the same reference, reported:

| Mimi substage | Original ms | Candidate ms |
| --- | ---: | ---: |
| Model load | 615.023 | 398.050 |
| SEANet convolutions | 4,806.679 | 765.921 |
| CPU transformer | 149.739 | 120.637 |
| Downsample convolution | 35.479 | 6.601 |
| RVQ | 128.517 | 128.986 |

These standalone timings are **diagnostic single calls**, not the controlled end-to-end speed estimate. All saved conv, transformer and downsample tensors are bit-identical, as are all 608 final codec IDs. In the fresh end-to-end A1 log, WAV writing takes roughly 1 ms at line-timestamp resolution; the wrapper returns about 0.19 s after the last log. That last interval includes teardown and runner/polling overhead, not an isolated `ggml_backend_free` timer. The original 120-second run still cannot be decomposed exactly without its command/reference/log. This audit did not rerun 1,500 frames or extrapolate the 29.3% result to them.

### Observed operation placement and dispatches

Summary placement traces from the short ICL runs show:

| Graph | HTP compute nodes | CPU compute nodes | Splits |
| --- | ---: | ---: | ---: |
| Speaker encoder | 760 | 0 | 1 |
| Text projection | 5 | **1 (`GET_ROWS`)** | **2** |
| Talker prefill | 815 | 0 | 1 |
| Talker step | 675 | 0 | 1 |
| Predictor prefill | 139 | 0 | 1 |
| Predictor step | 108 | 0 | 1 |
| Vocoder chunk 1 | 14,580 | 0 | 1 |
| Vocoder tail | 11,620 | 0 | 1 |

This confirms HTP compute placement, not all-HTP process execution. CPU text embedding lookup is explicitly allowed in `audit_graph_placement`; it creates a real scheduler boundary during prompt assembly. Mimi, BPE, FFT, sampling and I/O sit outside these accelerator graphs. Counts depend on model/shape; these are 0.6B figures.

The 96-frame run records 1,344 predictor steps, 96 predictor prefills, 95 Talker steps, 3 Talker-prefill calls, 7 text projections, 2 vocoder chunks and 1 speaker-encoder call: **1,548 audited graph computes**. HTP aggregate counters for A1 report **1,573 dispatched batches / 210,713 operations**, **7,850 flush calls (6,302 empty)**, 4,204 logical tensor sets / 1,763 gets, and 243 repack uploads (613,875,712 logical bytes). Tensor-get logical accounting includes large scheduler prompt-assembly transfers, so it must not be reduced to tiny logits alone. No per-physical-DDR byte claim follows. Repack gets are zero. The profiling flags were left OFF because existing strict placement, compute traces and aggregate counters were sufficient and avoid per-op profiling perturbation.

### Validation and reproducibility

- Release application rebuild succeeded with two jobs; separately configured scalar-OFF test build succeeded.
- `tools/npu-max/test-mimi-conv.cpp`: **7,560 bit-exact cases**, covering output-channel blocks/tails, irregular input channels, kernel sizes 1/3/7/8/10/12/16, stride/dilation, bias/no-bias, padding/no-padding, plus clearing an empty output. Uses explicit failure returns, including under `-DNDEBUG`.
- Targeted CTest `mimi_conv_test` passed in both SIMD-ON and SIMD-OFF configurations. The full model/reference-dependent CTest suite was not run for this focused CPU kernel change.
- Real-reference conv/transformer/downsample tensors and all final Mimi codec IDs match the preserved original archive exactly.
- A1/B/A2 output PCM is bit-identical: SHA-256 **`6f42537140a14fed4c7362b60cd7bf9d1c1184cadcce56ec157a5b5c56f37c4c`**. Each file has 184,320 samples, RMS **0.01775256**, peak **0.26104736**, and valid 24 kHz mono PCM16 format. Float vocoder outputs were checked finite by the unchanged runtime validation. No fresh listening claim; exact equality establishes no waveform change on this fixture.
- All five protected baseline hashes match after work. Application source only; GGML submodule unchanged. No server was started, stopped or replaced. Generated audio, reference crops, raw tensors, old executables and profiling logs remain in ignored `.cache/` only.

Run from the fork root:

```sh
cmake --build .cache/npu-max/app-build --parallel 2
ctest --test-dir .cache/npu-max/app-build -R '^mimi_conv_test$' --output-on-failure

# Repeat the exact retained short experiment on this handset, serially.
# Requires the preserved private evidence directory and foreground placement.
python .cache/investigation-20261113/run_ab.py A3 \
  .cache/investigation-20261113/before-qwen3-tts-cli
python .cache/investigation-20261113/run_ab.py B2 \
  .cache/npu-max/app-build/qwen3-tts-cli

# Portable build-time scalar control (use a separate build directory).
cmake -S . -B .cache/investigation-20261113/scalar-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_BUILD_DIR="$PWD/.cache/npu-max/build" \
  -DQWEN3_TTS_MIMI_CONV_SIMD=OFF -DQWEN3_TTS_NPU_TESTS=ON -DQWEN3_TTS_TIMING=ON
cmake --build .cache/investigation-20261113/scalar-build \
  --target npu-max-test-mimi-conv --parallel 2
ctest --test-dir .cache/investigation-20261113/scalar-build \
  -R '^mimi_conv_test$' --output-on-failure
```

Invoke Termux `ctest` directly: wrapping it in `native-env.sh` selects Android system `libjsoncpp` through LD_LIBRARY_PATH and failed with missing `_ZTIN4Json9ExceptionE`. Direct CTest works because the built test executable already has its runtime search paths. This is a test-runner environment issue, not a kernel failure.

Portable curated measurements and hashes: `docs/performance/npu-investigation-20261113/results.json`. Private full evidence: `.cache/investigation-20261113/` (A1/B/A2 logs, line timestamps, runtime samples, placement/transfer JSONL, model/binary/library hashes, fixture and runner, scalar/candidate stage harness and tensor dumps, build/test logs, before/after protected checks). No test WAV artifacts are committed. Other compilers, model sizes, non-ICL modes, cached server requests and sustained long-form speed were not newly measured; the modified Mimi path is shared independently of Talker model size.
