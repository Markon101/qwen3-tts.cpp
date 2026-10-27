# NPU launcher audit — 2026-10-27

The NPU shell entry points now share one mode handler for both model sizes. The short workspace command remains `sh npu.sh`; a portable root `npu.sh` is tracked in the repository, and the workspace copy delegates to it through a tracked template. The runtime/DSP inference algorithms and performance settings were not changed by this audit.

## Corrections

- Extraction-only mode no longer receives demo text. `-r ref.wav --embedding-out speaker.bin` extracts without synthesizing.
- Explicit reference, embedding, named-speaker, instruction or model options suppress an optional demo embedding. The 1024-float 0.6B preset is not injected into 1.7B.
- Conflicting reference/embedding inputs and transcript-without-reference requests fail before loading models. This prevents silent voice-input precedence errors.
- Saved embedding files receive a cheap structure check before model startup; the native runtime remains responsible for the model-width and finite-value checks.
- `--ref-text-file` safely expands a nonempty UTF-8 transcript into the CLI reference text, preserving quoted values and supporting UTF-8 BOM files.
- Server mode keeps stdin/stdout intact and rejects CLI fields that the native JSON protocol would ignore. The individual JSON `success` fields remain authoritative.
- Model directory overrides, spaces and caller-relative file paths are preserved. Help is available without build/model assets.
- SIGINT, SIGTERM and SIGHUP are forwarded to the synthesis process group and the child is reaped. Signals return 130/143/129 respectively; ordinary nonzero child exit codes are preserved.
- The measurement runner is reused in the launcher's existing Python interpreter. No extra Python process, model import, temperature abort or cooldown was added to the startup path.

## Checks

`tools/npu/test-launchers.py` passed **79/79 inference-free checks** against isolated fake installations, including paths with spaces and a help-only installation without binaries/models. The fake CLI performs no SDK initialization, model loading or inference. The real telemetry reader is retained. Checks include every voice mode, transcript files, malformed embeddings, conflicting arguments, empty/flag-like text values, custom model directories, stdout JSON cleanliness, exit-code propagation and child cleanup for all three forwarded signals.

Separate real smoke runs loaded the installed **0.6B and 1.7B** models with HTP0 generation and GPUOpenCL decoding. For each model:

1. Extracted the correct embedding width (1024/2048) with no synthesis.
2. Served serial JSON requests for no-reference speech, saved embedding, fresh reference and reference-text ICL.
3. Returned an expected error for a deliberately wrong-width embedding, then successfully served the following request.
4. Completed a one-shot non-streaming ICL request.
5. Completed the workspace shortcut's transcript-file ICL request.
6. Completed a final server request through the shared single-interpreter launcher/runner path.

Smoke outputs were deliberately short and checked for valid, nonempty audio with the native raw-logit/audio guards enabled. They establish mode readiness, not a new listening-quality or speed result. Existing longer listening checks are recorded separately in the voice/attention reports.

[routing-checks.json](routing-checks.json), [real-mode-summary.json](real-mode-summary.json) and [provenance.json](provenance.json) retain the checked outcomes and source/log hashes. Recordings, extracted embeddings and generated audio remain private under ignored `.cache/`. Native speaker/style availability still depends on the selected model variant; the real audit used the installed Base models. One-shot playback argument routing was checked without starting playback.

## Tracking

All repository shell launchers are tracked. The outer phone workspace is not itself a Git checkout. Its six top-level launcher contents are nevertheless versioned: the current NPU shortcut is an exact copy of `tools/npu/workspace-npu.sh.in`, and the five older OpenCL helpers have exact restoration snapshots in `tools/workspace-launchers/`. Their hashes are listed in that directory's manifest. The older helpers still refer to their retained private build/experiment layouts; canonical portable launchers are provided separately.

Use [the mode guide](../../NPU_LAUNCHERS.md) for short commands and JSON protocol details. No model, SDK or compiled binary was added to Git.
