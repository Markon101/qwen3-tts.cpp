# ICL voice-cloning divergence: root cause and fix — 2026-11-12

Status: **root cause identified; fix verified by probe, by an end-to-end render,
and by a user listening pass on long-form ICL (2026-11-12). Pushed to `origin`;
no measured performance claim.**

This supersedes the 2026-11-10 conclusion that `HTP_MAX_MMAPS 16 -> 32` alone
caused the ICL decode divergence. That conclusion is **refuted** below.

## Symptom under investigation

With the 2026-11-09 token-generation optimizations (`38b8208` application,
`78a07870` GGML), reference-conditioned ICL cloning produced garbled, silent or
overlong speech on HTP, while plain 0.6B HTP TTS worked. The divergence appeared
at the first autoregressive decode step.

## Method

Probe: `tools/npu/test-clone-prefill compare` (tracked source). Real-input mode,
fixtures frozen from a CPU `record` run on 2026-11-09:

- speaker x-vector, 1024 floats (`spk-cpu.bin`, 4100 bytes)
- reference codec frames: 206 frames x 16 codebooks (`ref-codes.i32`)
- 216 prefill positions, target 49 tokens, decode steps at `n_past` 216..223
- gate: cosine > 0.99 and NRMSE < 0.15 per tensor against the frozen CPU record

Model `qwen3-tts-0.6b-q8_0.gguf`, HTP backend forced by the probe, 2 host
threads. Each run installs one `libggml-htp-v79.so` and reruns the identical
comparison. Only the DSP library changes between rows; the application binary is
rebuilt once at `38b8208` and held fixed.

All runs are numerical parity checks. **None of them is a timing measurement**,
and none of them tests speech quality.

## Results

`real-step-0-hidden` cosine vs the frozen CPU record; `pass` is the probe's own
verdict over all 8 decode steps and all prefill/forward tensors.

| # | Application | DSP library (variant) | step-0 cosine | Verdict |
| - | --- | --- | ---: | --- |
| 1 | `9ac729b` (known good) | `a1ff3917` baseline | 0.999132795 | pass |
| 2 | `38b8208` | `a1ff3917` baseline | 0.999132795 | pass |
| 3 | `38b8208` | `78a07870` as published (32 slots) | 0.056268274 | **fail** |
| 4 | `38b8208` | orphaned `e1pad` build | 0.999132795 | pass |
| 5 | `38b8208` | orphaned `d1lock` build | 0.999132795 | pass |
| 6 | `38b8208` | V2: 16 slots + eviction fix, spad reuse on, spin-poll on | 0.216278412 | **fail** |
| 7 | `38b8208` | V4: V2 with `src1_spad` reuse disabled | 0.999132795 | pass |
| 8 | `38b8208` | V5: V2 with `src1_spad` reuse layout-guarded (16 slots) | 0.999132795 | pass |
| 9 | `38b8208` | V6: 32 slots + eviction fix + guarded reuse (the fix) | 0.999132795 | pass |

Rows 1, 2, 4, 5, 7, 8 and 9 are bit-identical to the 2026-11-09 CPU record, so
the passing builds are indistinguishable on this probe. Rows 3 and 6 fail on all
eight decode steps while prefill and the 216-position forward pass still match
(cosine 0.99999), i.e. the defect is confined to single-token decode.

Raw outputs: `compare-*.txt` in this directory, plus the phase-2 matrix in the
local `.cache/icl-debug/phase2/` (gitignored, regenerable with `run-probe.sh`).

## Conclusions

- **OBSERVED** — The mmap slot count is not causal. Row 6 fails with
  `HTP_MAX_MMAPS 16`, and rows 4/5 pass. The 2026-11-10 "mmap32 reproduces it"
  result is explained by that build also carrying the `src1_spad` change.
- **OBSERVED** — The `dspqueue` spin-poll is not causal (rows 7/8/9 keep it and
  pass).
- **OBSERVED** — The Hexagon `src1_spad` dynamic-quantization reuse is causal.
  Disabling it (row 7) restores parity; re-enabling it with the original guard
  (row 6) reproduces the failure.
- **OBSERVED** — The failure is deterministic: row 6 reproduces the 2026-11-09
  `no-swiglu` run to nine significant digits despite a different application
  graph configuration. This is not a race or a scheduling artefact.
- **HYPOTHESIS (mechanism)** — The original guard accepts a reuse when the next
  op consumes the same `src1` tensor pointer, the same `src0` type and the same
  row stride. It does not compare `src0`/`dst` extents, although the VTCM spad
  layout (full versus 16-row chunked, and `size_per_thread`) is selected from
  those extents. Two matmuls that share an activation but differ in weight/output
  shape therefore reuse a block quantized for a different layout. Row 9's guard
  removes exactly those cases and restores bit-exact parity.
- **NOT ESTABLISHED** — How often the guarded reuse still fires. With fused
  QKV/GateUp there may be few eligible consecutive pairs, so the optimization's
  remaining value is unquantified and would need a foreground measurement.
- **NOT ESTABLISHED** — Measured speed. The user reports the fixed build is
  faster than the previous one, but no timing was recorded in this work, so no
  number is claimed and no comparison table exists.
- **PARTLY ESTABLISHED** — Speech quality: confirmed by the user for 0.6B HTP ICL
  long-form on 2026-11-12. The probe itself reports
  `speech_quality_tested: false` by design and cannot substitute for that.

## Fix

GGML branch `fix/htp-src1-spad-layout-guard`, one commit (`717597a2`) on top of
`78a07870`, touching only `htp/main.c`:

- keep the `src1_spad` reuse, but only when the previous and current ops have
  identical `src0` and `dst` `ne[]`/`nb[]` descriptors, the previous spad is
  populated, and the previous `dst` does not alias the reused `src1` buffer.

`HTP_MAX_MMAPS 32` and the `prep_op_bufs` slot-eviction fix are retained
unchanged: the slot count is not the cause of this divergence, but the 15
persistent Code Predictor plans need the larger table.

Application branch `npu-max-icl-fix` updates the submodule pointer and carries
this report. Nothing is pushed.

## End-to-end gate

Same command, same text, 96-token cap, launcher-equivalent environment. The
application must set the launcher's HTP profile variables, in particular
`QWEN3_TTS_MAX_VOCODER=1`; without it the vocoder graph splits 90 ways, falls
back to CPU and fails with `Vocoder decode failed: Failed to compute graph`. That
failure mode is an invocation error and reproduces on the known-good build, so
do not read it as a regression.

| Build | Result | Audio |
| --- | --- | --- |
| `9ac729b` + `a1ff3917` (control) | exit 0, vocoder splits 1 | 2.48 s, peak 0.254, RMS 0.0371 |
| `npu-max-icl-fix` + V6 (fix) | exit 0, vocoder splits 1 | 4.08 s, peak 0.337, RMS 0.0416 |

Both are non-silent. The duration difference is expected under the default
sampling settings and says nothing about correctness either way.

## User validation

On 2026-11-12 the user rendered long-form Butterverse takes through
`butterverse_longform_queue.sh` (0.6B, `htp`, ICL reference + transcript) on the
fixed build and reported that long-form generation and ICL voice cloning work as
expected, and that the build is faster than the one it replaces.

Treat that as a **user report, not a measurement**: no timing was recorded and no
speed claim is made here. It does establish that the reported symptom — garbled,
silent or overlong ICL output — is gone in the configuration the user actually
uses, which the probe alone could not show.

## Reproduce

```sh
cd repos/Markon101-qwen3-tts.cpp
sh .cache/icl-debug/phase2/run-probe.sh <label> [path/to/libggml-htp-v79.so]
```

The probe needs the fixture directory at mode 0700 and writes `htp-*` dumps into
it; `run-probe.sh` prints the installed library hash at the start of every run.

End-to-end:

```sh
QWEN3_TTS_TRANSFORMER_BACKEND=hexagon QWEN3_TTS_ENCODER_BACKEND=hexagon \
QWEN3_TTS_DECODER_BACKEND=hexagon QWEN3_TTS_ENCODER_FFT=1 \
QWEN3_TTS_MAX_VOCODER=1 QWEN3_TTS_MAX_STRICT=1 \
QWEN3_TTS_VALIDATE_AUDIO=1 QWEN3_TTS_VALIDATE_LOGITS=1 \
  sh tools/npu-max/native-env.sh .cache/npu-max/app-build/qwen3-tts-cli \
  -m ../../models/qwen3-tts-0.6b-q8_0.gguf \
  --tokenizer ../../models/qwen3-tts-tokenizer-f16-f32.gguf \
  --threads 2 --language english -r ../../smalldon.wav --ref-text "$REF" \
  -t 'Hello. This is a short end to end check.' --max-tokens 96 -o out.wav
```

## Open items

- Coverage not yet exercised with the fixed library: plain non-ICL TTS, 1.7B,
  OpenCL/`mixed`, server path, batch.
- No measured performance comparison exists. If the speed gain matters for a
  release note, run a foreground (`/top-app`) matched comparison against
  `9ac729b`; do not cite a number without one.
- `experimental/npu-max` (application and GGML) is fast-forwarded to the fix, so a
  fresh checkout no longer carries the defect. The broken commits remain in
  history: application `38b8208`, GGML `78a07870`.
