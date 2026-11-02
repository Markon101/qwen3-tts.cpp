# Fork audit and repeatable checks

This is the entry point for an agent continuing work on this fork. Start with
repository identity, choose checks for the component being changed, and preserve
raw evidence. A past pass does not certify another checkout or binary.

Last audit: **2026-11-02 UTC**. Scope: the Android NPU-Max fork, larger controllable
vocoder windows, launch routing, numerical continuity and source-level improvement
opportunities. This was not an exhaustive security audit or a speed benchmark.

## Identity and evidence

- Application starting commit: `24bf4043b58bdc4cb32f431f4e1f89a1bcd16fd4`,
  branch `experimental/npu-max`; application and GGML worktrees started clean.
- GGML: `a1ff3917ed1b31deac8b96f04a62afd89a15faa6`.
- Upstream `CHE3MZ/qwen3-tts.cpp` latest default-branch commit checked through
  GitHub on this date: [`e04c79a`](https://github.com/CHE3MZ/qwen3-tts.cpp/commit/e04c79a2b37be8b4696e918e728edd5e7541a379).
  That commit was already present locally; no upstream merge was performed.
- Device: Android/Termux, Snapdragon 8 Elite / Adreno 830 / HTP v79.
  Compiler: Clang 21.1.8. Inference checks used two CPU threads or six HVX workers.
- Local raw evidence: `.cache/improvement-audit-20261102/`. This includes build
  logs, complete placement traces, raw F32 outputs, original source snapshots,
  the pre-change CLI/pipeline executables, and reproduction probes.
- Portable evidence: [audit results](performance/fork-audit-20261102/results.json),
  [launcher checks](performance/fork-audit-20261102/launchers.json),
  [CLI parsing checks](performance/fork-audit-20261102/cli-validation.json).
  The results file records hashes and scopes. Raw models/audio are not Git-tracked.

The frozen `.cache/npu` baseline is distinct from `.cache/npu-max`. Check all
entries in `.cache/npu-max/protected-baseline.json` before and after work. Never
replace the baseline merely to test an NPU-Max change.

## Controllable vocoder windows

The rebuilt NPU-Max launcher and native CLI accept:

```sh
# Run from the workspace that contains npu-max.sh and models/.
sh npu-max.sh 1.7b --vocoder-chunk-frames 200 \
  -r reference.wav --ref-text 'The reference transcript.' \
  -t 'The new sentence to synthesize.' -o speech.wav
```

Use any integer **17 through 256**; omission retains **100**. This sets the total
number of codec frames admitted to one vocoder graph, including up to **16 frames
of replayed left context**. It does not change sampling or `--max-tokens`.
At 1,920 samples/frame and 24 kHz, windows of 100, 200 and 256 represent 8.00,
16.00 and 20.48 seconds. For longer inputs, the step is window minus 16:
84, 184 or 240 new frames per chunk. The existing chunk schedule is preserved.

Why 256: two 2x upsamplers feed pointwise matrix multiplications with up to
`4 * window` activation rows. The current HTP backend permits 1,024 rows.
Raising the cap further requires tiling those operations and checking graph
capacity, allocation sizes, backend placement and numerical output. A bounded
channel width alone does not prove that arbitrarily large windows will work.

Tradeoffs and interfaces:

- Larger windows reduce history resets and repeated overlap, while increasing
  temporary memory and work before a graph returns. No speedup is promised.
- The overlap approximates full history. Decoder attention sees all previous
  frames within its graph; replaying only 16 frames cannot preserve that state.
  A larger window is not a guarantee of better perceived speech.
- C++ callers set `tts_params::vocoder_chunk_frames`, or pass the fourth argument
  to `AudioTokenizerDecoder::decode`. Rebuild C++ clients against the changed
  headers. The C API parameter structure/ABI was not extended.
- `--server --vocoder-chunk-frames 200` sets the process default for every request,
  and individual JSON requests may also pass `"vocoder_chunk_frames": 200` (validated
  in `17..256` per request without mutating the server default).
- Callback emission size remains controlled independently by
  `set_audio_chunk_callback(callback, chunk_frames)`. Its default is still 12.
  The emission size is capped at `vocoder_chunk_frames - 16`; changing the graph
  limit alone does not increase an explicitly smaller callback size.
- The separate experimental `tools/npu-max/pipeline.py` still uses its own
  100-frame replay window. It is not the normal launcher path.
- The preserved `npu.sh` baseline executable predates this flag. Use `npu-max.sh`
  or the rebuilt native CLI, rather than assuming an older binary supports it.

Invalid values fail before CLI model loading. Integer overflow is rejected rather
than wrapping into a valid frame count. Streaming output is checked for expected
sample count and, when audio validation is enabled, finite samples before delivery.

## Checks performed in this audit

| Check | Result and scope |
| --- | --- |
| Repository state, submodule, remotes, active inference processes | Inspected before edits; no active inference job found |
| Protected baseline hashes | All 5 matched before and after edits (`protected-baseline.json`) |
| CMake application build | Passed with two build jobs; existing GGML/DSP runtime reused |
| Launcher checks | 184/184; fake installation, both model sizes and all four profiles; not inference |
| Native CLI option parsing | 19/19; endpoints, invalid bounds, fractional/junk values, overflow, empty/missing value; no model load |
| QAIC generated-interface cache | 7/7; did not rebuild the DSP or run a performance sweep |
| HTP same-code chunk tests | Passed: 100/200/256 windows, a full 256-frame graph and 257-frame chunked input |
| HTP placement | Every observed vocoder compute graph stayed on HTP, CPU=0, other=0, splits=1 (9 distinct graphs, 14 compute calls, max 43,048 HTP compute nodes) |
| CPU & OpenCL same-code chunk tests | Passed on both `cpu` and `opencl` across 100/200/256 windows, 257-frame split, and callback checks |
| Cross-backend waveform parity | At windows 100/200/256, HTP vs CPU NRMSE is `0.00293–0.00304`, OpenCL vs CPU is `0.00261–0.00278`, and HTP vs OpenCL is `0.00153–0.00160` |
| Old-default compatibility | Bit-exact SHA-256 match against pre-edit binary (8 frames, `647a2320...`) and pre-edit decoder source at default 100 (256 frames, `c47c909b...`) |
| Callback delivery & ICL prefix trimming | Matched offline samples bit-for-bit across HTP, CPU, and OpenCL; ICL `trim_prefix_frames` verified for both single-shot and streaming callback paths |
| Invalid C++ parameters | Rejected before graph execution; includes direct decoder and callback paths |
| Server RFC 8259 JSON parser (`test-server-json.py`) | 9/9 passed: `\uXXXX` UTF-8 decoding, UTF-16 surrogate pairs, key-in-value isolation, nested objects, strict `vocoder_chunk_frames` validation, malformed JSON rejection |
| Listening, sustained speed, battery use | Not performed; no new claims for these outcomes |

The waveform probe repeats a saved **16-frame** speech-code sequence to exercise
256/257-frame shapes without sampling new tokens. It checks implementation and
continuity, not natural long-form prosody. The test deliberately loads only the
vocoder; it does not claim a fresh end-to-end test of every voice/model mode.

For HTP, relative waveform error against one full 256-frame decode was
**0.206252** with window 100 and **0.182902** with window 200 (CPU: **0.206152** and
**0.182560**; OpenCL: **0.206363** and **0.183078**). After their respective
first boundaries, the HTP errors were **0.248456** and **0.328602**. These are different
suffix intervals, not a matched listening comparison. The larger window reduced
whole-waveform error on this probe, but did not establish uniformly smaller
boundary error. The old assertion that 16-frame overlap guarantees zero pitch or
phase discontinuity is unsupported. Preserve these negative results.

> [!IMPORTANT]
> **Background scheduling caveat for all performance tests:** All historical
> performance tests in [docs/NPU_MAX_BACKEND_AUDIT.md](NPU_MAX_BACKEND_AUDIT.md)
> (as well as the ordinary timing instrumentation in this audit's raw logs) were
> taken while Termux was running in the **background**. On Android, background
> scheduling restricts host CPU core placement (`cpuset`) and clock frequencies,
> which throttles CPU reference execution, host-side token orchestration, and GPU
> dispatch overhead while Hexagon HTP continues executing on the dedicated DSP
> under explicit HAP performance votes. Do not cite background-recorded latencies
> or relative CPU/OpenCL speedup ratios as foreground benchmarks; any speed claim
> requires a fresh, verified foreground (`/top-app`) measurement.

## Repeatable workflow

Run commands from the fork root unless a block says otherwise. Store each new
run in a fresh evidence directory. Keep checks sequential on the phone and use
at most two build jobs. Do not run a speed or sustained-load experiment without
fresh foreground approval from the user; correctness checks are a separate task.

### 1. Establish the checkout and device state

```sh
pwd
git status --short --branch
git rev-parse HEAD
git remote -v
git submodule status
git -C ggml status --short --branch
ps -ef | rg 'qwen3-tts|npu-max|cmake|ninja|benchmark'
cat /proc/self/cpuset
rg 'Cpus_allowed_list' /proc/self/status
head -6 /proc/meminfo
df -h .
```

Read `AGENTS.md`, this file, `docs/NPU_MAX_BACKEND_AUDIT.md`, the relevant component
source and the newest evidence. Check existing changes and jobs before building.
Verify the executable actually used by the launcher; a branch name or banner is
not sufficient provenance. Hash model/tokenizer, executable, host libraries and
DSP skeleton for a new numerical or performance comparison.

### 2. Build and run checks that do not need models

For an existing prepared NPU-Max runtime:

```sh
cmake --build .cache/npu-max/app-build --parallel 2
python tools/npu-max/test-launchers.py
python tools/npu-max/test-vocoder-options.py
python tools/npu-max/test-server-json.py
python tools/npu-max/test-qaic-cache.py
git diff --check
```

For a clean installation, use `sh tools/npu-max/build.sh` after reviewing its SDK
requirements. It is more work than an incremental application build. Never point
it at `.cache/npu` to save time. The QAIC cache test checks hashes and invalidation;
it does not establish correctness of changed DSP arithmetic.

### 3. Run numerical vocoder checks

The chunk harness requires the tokenizer GGUF and a saved raw little-endian I32
file: 16 codec IDs per row, each in `[0, 2047]`. The output directory must exist.
In the current workspace, the following seed is available; on another checkout,
substitute a known saved sequence and record its hash.

```sh
mkdir -p .cache/my-vocoder-check/htp
QWEN3_TTS_EXECUTION_TRACE="$PWD/.cache/my-vocoder-check/htp/placement.jsonl" \
  sh tools/npu-max/native-env.sh \
  .cache/npu-max/app-build/tools/npu/npu-max-test-vocoder-chunks \
  ../../models/qwen3-tts-tokenizer-f16-f32.gguf \
  .cache/npu-max/pipeline/20261028T103343-6866/codes.i32 \
  .cache/my-vocoder-check/htp hexagon \
  > .cache/my-vocoder-check/htp/result.jsonl \
  2> .cache/my-vocoder-check/htp/run.log
```

Repeat sequentially with `cpu` or `opencl` as the last argument and a separate
output directory when those backends are affected. CPU tests can take several
minutes. Inspect the exit status and final `pass` record, not just the existence of
an audio file. Compare saved outputs for the **same** input codes and window.
Do not compare independently sampled waveforms as an arithmetic parity test.

The harness covers:

- Default versus explicit 100 on short input, bit for bit.
- End-to-end decoder orchestration parameter forwarding at windows 100/200/256.
- Exact sample counts (`frames * 1920`) and finite audio.
- Input one frame longer than the maximum graph window, including the final tail.
- Full-graph versus overlap-decoded waveform differences, with raw F32 retained.
- Callback sample order, concatenation, final marker and cancellation prefix.
- Callback graph bounds even when the requested emission size is 1,024 frames.
- Invalid window rejection without synthesis.

For changes to convolution lowering, also run
`npu-max-test-vocoder-lowering --backend cpu` and direct HTP controls. For unary,
masking or attention changes, run the matching NPU tests before end-to-end speech.
Preserve non-aligned row/tail cases and the previous long-context attention fix.

### 4. Test the product paths relevant to the change

A routing test is not a model test. For a release, maintain separate results for:

| Axis | Required cases |
| --- | --- |
| Size | 0.6B and 1.7B; speaker embedding dimensions must match |
| Voice | No reference, x-vector reference, saved embedding, ICL reference + transcript |
| Other model types | CustomVoice and VoiceDesign with the corresponding model assets |
| Backend | HTP, mixed, OpenCL, CPU; trace actual placement |
| Length | Short, window boundary minus/at/plus one, multiple windows and a partial tail |
| Interface | One-shot CLI, repeated server requests, C++ callback, C API where affected |
| Controls | Fixed seed/greedy codes, same-code decode, CPU oracle, old binary/source baseline |
| Failures | Bad files, wrong dimensions, invalid flags, backend failure, output failure and cancellation |

Check ICL trimming separately from vocoder chunk counts. Do not claim cloning
quality from finite PCM alone. Listen to matched speech examples at boundaries
and over long utterances, retaining user judgments separately from metrics.
The generic CTest suite requires reference binaries and model names that may not
exist in a new workspace; inspect the commands/assets before running it.

### 5. Only then test speed, when authorized

Use cold and warm requests separately. Record foreground/cpuset, temperatures,
clocks, memory, input/output identities, sampling settings, frame counts and
component timings. Use serial matched trials and fixed work. Record first audio
separately from total completion. A larger window can change both. Do not infer
energy savings from GPU temperature or assume that parallel HTP processes gain
throughput: they share physical resources. Stop expanding a test after relevant
checks pass unless a new failure or unresolved concern justifies more work.

### 6. Finish with a durable handoff

Record commands, exit codes, hashes, build flags, tested backends and untested
paths in a tracked report. Keep large/private inputs in the local artifact folder,
with hashes and a format description. Verify protected-baseline hashes again,
inspect `git diff --check` and `git status`, and make new reports/tests tracked.
State whether files are staged, committed or pushed. Do not imply a local edit
has been published.

## Resolved source improvements and remaining opportunities

### Resolved in this audit

1. **Replaced the server's ad-hoc JSON extraction (`src/main.cpp`, `tools/npu-max/test-server-json.py`).**
   Top-level JSON requests are now parsed with an RFC 8259 lexer that decodes
   `\uXXXX` and UTF-16 surrogate pairs to UTF-8 (`{"text":"Caf\u00e9 \u4f60\u597d"}` ->
   `Café 你好`), isolates top-level keys from string contents and nested composites
   (`{"note":"text","junk":"wrong","text":"correct"}` -> `correct`), validates
   per-request `"vocoder_chunk_frames"` in `17..256`, rejects malformed JSON, and
   escapes control characters `< 0x20` in `json_escape`.
2. **Fixed ICL streaming callback trimming (`src/qwen3_tts.h`, `src/qwen3_tts.cpp`).**
   `decode_codes_streaming` now accepts `trim_prefix_frames` (`n_ref_frames` on the
   ICL path). When `audio_chunk_callback_` is registered, streaming starts at
   `base = trim_prefix_frames` while still using up to `VOCODER_LEFT_CONTEXT` (`16`)
   preceding reference frames as causal warmup, so callbacks never receive the
   reference audio prefix and concatenated callback samples match `result.audio`
   bit-for-bit (`icl_trim_callback_bit_exact: true` in `test-vocoder-chunks.cpp`).
3. **Cached reference conditioning for repeated voices (`src/qwen3_tts.h`, `src/qwen3_tts.cpp`).**
   `Qwen3TTS` caches the extracted ECAPA speaker embedding and Mimi ICL reference
   codec frames for a single active reference voice (`<= 30 s`), keyed by loaded
   model/tokenizer paths and exact `memcmp` of normalized `ref_samples`, and
   invalidated on `unload_models()` or any reference change.
4. **Made AArch64 CPU tuning configurable and compiler-checked (`CMakeLists.txt`).**
   Added `QWEN3_TTS_AARCH64_MCPU` (`auto` default, `oryon-1`, `native`, or `generic`)
   guarded by `check_cxx_compiler_flag` so non-Oryon AArch64 targets and compilers
   fall back cleanly.

### Remaining architectural opportunities

1. **Preserve decoder state across chunks.** A persistent attention KV cache and
   correct convolution/upsampler history could remove the 16-frame overlap
   approximation and replay. Establish chunk/full waveform parity before pursuing
   overlapping pipeline stages.
2. **Optimize the code predictor after profiling the current build in foreground.**
   It makes 15 dependent predictions per frame. Inspect graph setup, host
   sampling/embedding transfers and arithmetic separately. Larger vocoder windows
   do not accelerate this autoregressive loop.
3. **Run foreground (`/top-app`) performance benchmarks when authorized.** Because
   all historical timing tables in `docs/NPU_MAX_BACKEND_AUDIT.md` were recorded in
   the background, re-measure cold, warm, and sustained throughput only after
   confirming `/proc/self/cpuset` is `/top-app` throughout the run.
