# NPU launchers

The same launcher handles both installed Base model sizes and all supported voice input modes. It selects HTP0 generation and OpenCL decoding, validates raw logits/audio, and logs temperatures/clocks without added thermal limits. Existing DSP thread, polling and quantization settings are retained.

From the existing phone workspace, the short commands remain:

```sh
sh npu.sh 0.6b
sh npu.sh 1.7b
```

The repository also has a tracked root `npu.sh`. It defaults to 0.6B, English and a short demo when no text was supplied. It respects `QWEN_MODELS_DIR`, uses caller-relative input paths and keeps spaces in filenames intact. The existing workspace shortcut supplies its known 0.6B demo embedding only for the matching model. A fresh clone uses the normal no-reference path unless a model-specific demo embedding is explicitly configured. The 0.6B preset is never reused automatically for 1.7B.

For an external model directory:

```sh
export QWEN_MODELS_DIR=/path/to/models
```

The underlying `tools/npu/run-tts.sh` requires a model size and forwards explicit CLI input without adding demo text/output. Both entries use the same mode validation. `--help` works before models/build assets are installed; when the native CLI/runtime is present it also shows native CLI options.

## Voice modes

Examples use short placeholder paths; substitute your own files. Each code line is a complete command.

```sh
sh npu.sh 0.6b --text "Hello."
sh npu.sh 1.7b -r voice.wav --text "Hello."
sh npu.sh 1.7b -r voice.wav --ref-text-file voice.txt --text "Hello."
sh npu.sh 0.6b -r voice.wav --ref-text "Words in the clip." --text "Hello."
sh npu.sh 1.7b --embedding-in speaker.bin --text "Hello."
```

Named-speaker and instruction flags pass through unchanged; their availability depends on the loaded model variant. The real audit used the installed Base models.

`--ref-text-file` reads nonempty UTF-8 text, including common UTF-8 BOM files, and passes it as `--ref-text`. The reference text must describe the words in the reference recording. Supplying both text-source options is rejected. Without reference text, `-r` performs speaker-only cloning; with matching reference text it enables ICL.

An existing embedding must come from the matching model: the supplied 0.6B model uses 1024 floats and 1.7B uses 2048. The launcher checks the saved-file structure, and the runtime checks its model width and finite values. A WAV is not a saved embedding. The launcher rejects simultaneous `--reference` and `--embedding-in` to prevent the CLI from silently ignoring a reference.

## Embedding extraction

```sh
sh npu.sh 0.6b -r voice.wav --embedding-out speaker.bin
sh npu.sh 1.7b -r voice.wav --embedding-out speaker17.bin
```

These commands extract only, with no injected demo text or speech generation. Add explicit nonempty `--text` if you intentionally want synthesis and embedding saving together. `--output` selects the generated WAV path. `--non-streaming` remains available for the corresponding prefill mode, including ICL.

For one-shot playback, `--speak` passes through to the native CLI without injecting a WAV output path. Add `--output` if you also want a file. Playback itself depends on the platform's native audio support; the audit verified its argument routing without starting playback.

## Serial JSON requests

```sh
sh npu.sh 1.7b --server
```

Read JSON lines from stdin and write one JSON response per request to stdout. Model loading and telemetry messages stay on stderr. Text, voice and output fields belong in each request, for example:

```json
{"text":"Hello.","reference":"voice.wav","ref_text":"Words in the clip.","language":"english","max_tokens":128,"output":"speech.wav"}
```

Use `embedding_in` instead of `reference`/`ref_text` for a saved embedding, or omit voice fields for no-reference speech. CLI voice/text/output flags alongside `--server` are rejected because the native server does not use them as request defaults. `--non-streaming`, `--threads` and sampling parameters can still set process defaults. The transcript-file convenience is for one-shot CLI input; JSON clients send the actual `ref_text` string.

Each response's `success` field is authoritative. A failed request can be followed by a successful request, and the process may exit zero at EOF even if an earlier request failed. Callers must inspect responses. This is the CLI's serial stdin/stdout protocol.

Ctrl+C, SIGTERM and SIGHUP are forwarded to the child process group, and the runner reaps its child. Ordinary nonzero child exit codes are preserved. Diagnostic queue-only operation flags are cleared for the normal NPU launcher so it actually computes the request.

## Git tracking and verification

All repository shell launchers are tracked. The outer phone workspace itself is outside Git. Its active NPU shortcut matches the tracked template `tools/npu/workspace-npu.sh.in`, while the five older workspace OpenCL launchers are preserved as exact content snapshots under `tools/workspace-launchers/`. See that directory's manifest for each filename/hash. The portable canonical fast launchers remain at the repository root and under `scripts/android/`.

The [launcher audit](performance/launcher-audit-20261027/REPORT.md) separates inference-free routing/signal checks from real 0.6B/1.7B mode smoke runs. Models, recordings, embeddings, generated audio and the SDK are excluded from Git.
