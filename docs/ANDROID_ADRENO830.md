# Samsung Adreno 830 / Termux profile

This branch contains the measured Snapdragon 8 Elite changes and tracks the matching GGML fork. The profile was studied on a Samsung S25 Ultra; it is not an automatic cross-device tuner.

## Build

Install the normal Termux C++/CMake/Ninja/Python/OpenCL headers and loader prerequisites, then clone with the submodule:

```sh
git clone --recurse-submodules https://github.com/Markon101/qwen3-tts.cpp.git
cd qwen3-tts.cpp
sh scripts/android/build-adreno.sh
```

The script builds the pinned GGML CPU/OpenCL libraries and the application in Release mode. It does not install packages, change system/vendor files or download models. Set `BUILD_JOBS` to choose build concurrency. No hardware execution or performance check is part of the build command.

## Models and launchers

Keep model assets outside Git. Set `QWEN_MODELS_DIR` to an existing model directory or use ignored `models/` with these filenames:

- `qwen3-tts-0.6b-q8_0.gguf`
- `qwen3-tts-1.7b-q8_0.gguf`
- `qwen3-tts-tokenizer-f16-f32.gguf`

```sh
QWEN_MODELS_DIR=/path/to/models ./run-tts-fast-0.6b.sh \
  --text 'Hello from my phone.' --reference /path/to/reference.wav --language english --output speech.wav
```

The 0.6B default generates on two CPU threads pinned to cores 6–7 and decodes on OpenCL. `QWEN_TTS_MODE=gpu` selects the measured GPU alternative. The 1.7B default keeps generation on GPU because the 0.6B thread/placement result did not generalize convincingly. Additional CLI arguments pass through.

The first process compiles and caches kernels. Later starts can reuse the program-binary cache; first-install startup is not the cache-populated startup reported in the measurements. The cache is local and driver/source/options keyed.

## Measurement-only logging

Current launchers impose no temperature aborts or cooldown waits. They log CPU/GPU sensors, frequency policies, available utilization counters, scheduling state and process I/O at roughly 100 ms intervals under `.cache/tts-runs/`. Android continues to manage device frequency and thermal behavior. Reported frequency is not a cycle-accurate residency measurement.

`QWEN3_TTS_VALIDATE_AUDIO=1` rejects non-finite decoded audio before PCM saving. This is a numerical-integrity check, not a thermal limit. Optional research controls are `QWEN3_TTS_SEED` (reset per request) and `QWEN3_TTS_CODES_DIR` (retain private raw speech codes in an existing directory).

The recording prototype is disabled in these launchers. The optional two-token kernel is enabled only for the 0.6B GPU profile. The causal-mask correction is part of the pinned GGML source.

## Evidence and limitations

Read [the measured results](performance/adreno830/REPORT.md). The measurement-only fixed 64-frame comparison produced finite, non-silent audio and observed 7.293 → 5.458 seconds median synthesis time. Hybrid generation changed seeded speech codes; this is not a bit-identical quality claim. Long-run energy and thermal superiority are unproven.

Models, reference voices, generated audio, raw speech codes, cached programs and compiled libraries are intentionally absent from this source fork. The original research artifacts remain in the local workspace. The public evidence contains reviewed numeric measurements and clock/temperature traces.
