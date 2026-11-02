# AGENTS.md

Coding conventions and architecture guide for AI agents working on this codebase.

## Fork audit and continuation

Before changing the Android/NPU-Max fork, read [docs/FORK_AUDIT.md](docs/FORK_AUDIT.md).
It records the current controls, reusable check commands, tested scopes, larger
vocoder-window option, known bugs, and evidence paths. Select checks relevant to
the change; do not interpret old results as verification of a new checkout.

## Project Overview

`qwen3-tts.cpp` is a pure C++17 implementation of the Qwen3-TTS text-to-speech pipeline using GGML. It converts text to speech through four stages: tokenization, speaker encoding, transformer code generation, and vocoder decoding.

The implementation supports all three model variants and both model sizes from the Python reference:
- **Base model** (0.6B / 1.7B): voice cloning via ECAPA-TDNN x-vector speaker embedding
- **CustomVoice model** (0.6B / 1.7B): named speaker selection + optional emotion/style instruct text
- **VoiceDesign model** (0.6B / 1.7B): free-form natural-language voice description

## Repository Structure

```
qwen3-tts.cpp/
  src/                          # C++ source files
    main.cpp                    # CLI entry point
    qwen3_tts.{h,cpp}          # Full pipeline orchestration
    tts_transformer.{h,cpp}    # TTS transformer (talker + code predictor)
    text_tokenizer.{h,cpp}     # BPE text tokenizer
    audio_tokenizer_encoder.{h,cpp}  # ECAPA-TDNN speaker encoder
    audio_tokenizer_decoder.{h,cpp}  # WavTokenizer vocoder
    mimi_encoder.{h,cpp}       # Mimi audio encoder (ICL voice cloning)
    gguf_loader.{h,cpp}        # GGUF model loading
    qwen3tts_c_api.{h,cpp}    # C FFI bindings
    coreml_code_predictor.h    # CoreML code predictor interface
    coreml_code_predictor.mm   # CoreML implementation (Apple only)
    coreml_code_predictor_stub.cpp  # No-op stub for non-Apple builds
  tests/                        # Component tests
    test_codebook.cpp
    test_vq_only.cpp
    test_tokenizer.cpp
    test_encoder.cpp
    test_transformer.cpp        # Deterministic reference comparison
    test_decoder.cpp
    test_batch.cpp              # Batch inference correctness + KV isolation
    test_mimi_encoder.cpp       # Mimi encoder codec code match vs Python
    bench_batch.cpp             # Batch throughput benchmark
  scripts/                      # Python utilities
    convert_tts_to_gguf.py      # HuggingFace -> GGUF converter (TTS model)
    convert_tokenizer_to_gguf.py # HuggingFace -> GGUF converter (vocoder)
    setup_pipeline_models.py    # One-shot download + convert all models
    generate_deterministic_reference.py  # Generate Python reference data
    compare_e2e.py              # End-to-end Python vs C++ comparison
    run_all_tests.sh            # Test runner
  tools/                        # Build scripts and model setup tools
    build-scripts/              # CMake + Ninja build wrappers (build.bat / build.sh)
    model-downloader/           # Pre-built GGUF downloader (no torch required)
    model-converter/            # Interactive download + local conversion wizard
  reference/                    # Reference data (*.bin gitignored, *.json tracked)
  models/                       # GGUF models (gitignored)
  CMakeLists.txt
```

## Build System

- **CMake 3.14+** with C++17
- GGML is vendored under `./ggml` and linked from `./ggml/build/src`
- Build GGML first: `cmake -S ggml -B ggml/build -DGGML_METAL=ON && cmake --build ggml/build -j4`
- Build project: `cmake -S . -B build && cmake --build build -j4`
- Timing build: `cmake -S . -B build -DQWEN3_TTS_TIMING=ON && cmake --build build -j4`
- GGML headers are in `./ggml/include`

## Coding Conventions

### C++ Style

- C++17 standard, no exceptions, no RTTI
- Use `fprintf(stderr, ...)` for logging, not `std::cerr`
- Error handling: methods return `bool`, error details stored in `error_msg_` member
- Memory: GGML contexts own tensor memory; use `ggml_free()` for cleanup
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for macros
- Header guards: `#pragma once`
- All public types in `qwen3_tts` namespace

### GGML Patterns

Every forward pass follows this pattern:

```cpp
// 1. Build computation graph
struct ggml_cgraph * gf = build_xxx_graph(...);

// 2. Allocate graph memory
ggml_backend_sched_alloc_graph(state_.sched, gf);

// 3. Set input tensors
struct ggml_tensor * inp = ggml_graph_get_tensor(gf, "input_name");
ggml_backend_tensor_set(inp, data, 0, size);

// 4. Compute
ggml_backend_sched_graph_compute(state_.sched, gf);

// 5. Get output tensors
struct ggml_tensor * out = ggml_graph_get_tensor(gf, "output_name");
ggml_backend_tensor_get(out, output_data, 0, size);

// 6. Reset scheduler
ggml_backend_sched_reset(state_.sched);
```

Important: `ggml_cast` to F32 is needed before `ggml_mul_mat` when weight tensors are F16 (specifically `ffn_down` in both talker and code predictor layers).

Backend initialization and scheduling notes:

- Use `init_preferred_backend()` (`src/gguf_loader.cpp`) to select backend in order: `IGPU -> GPU -> ACCEL -> CPU`
- If the selected runtime backend is not CPU, add a CPU backend as scheduler fallback (`backend_cpu`) when calling `ggml_backend_sched_new(...)`
- Decoder follows the same backend preference; load decoder weights with `GGML_BACKEND_DEVICE_TYPE_IGPU` preference for Metal-first execution

### Model Architecture

The TTS transformer has two sub-models:

1. **Talker** — 28-layer Qwen2 transformer (1024 hidden, 16 heads, 8 KV heads, 128 head_dim)
   - Input: prefill embedding or step embedding (float32, [1, 1024])
   - Output: hidden states + codec logits via `codec_head`

2. **Code Predictor** — 5-layer transformer (same attention config)
   - Input: talker hidden state + codebook-0 embedding (2-token prefill)
   - Output: 15 codebook predictions (autoregressive, one per step)
   - Has its own separate KV cache (max 16 tokens)

### Prefill Embedding Structure (10 positions for single-word input)

```
Pos 0:   text_projection(<|im_start|>)
Pos 1:   text_projection(assistant)
Pos 2:   text_projection(\n)
Pos 3:   tts_pad + codec_embd(think_id)
Pos 4:   tts_pad + codec_embd(think_bos_id)
Pos 5:   tts_pad + codec_embd(language_id)
Pos 6:   tts_pad + codec_embd(think_eos_id)
Pos 7:   tts_pad + speaker_embedding
Pos 8:   tts_bos + codec_embd(pad_id)
Pos 9+:  text_projection(text_token[i]) + codec_embd(bos_id or pad_id)
```

This structure must mirror the Python pipeline exactly.

### Special Token IDs

```
tts_bos = 151672, tts_eos = 151673, tts_pad = 151671
codec_bos = 2149, codec_eos = 2150, codec_pad = 2148
codec_think = 2154, codec_think_bos = 2156, codec_think_eos = 2157
codec_nothink = 2155
english_language_id = 2050
```

These are defaults for the 0.6B-Base model. The converter now writes all token IDs and the speaker/language tables to GGUF, so they are read from the file at runtime.

### Key Files to Understand

- `tts_transformer.cpp` — Core file (~3400 lines). Contains `generate()`, `generate_icl()`, `generate_from_prefill()`, `build_prefill_graph()`, `build_prefill_graph_icl()`, `build_prefill_graph_instruct()`, `forward_prefill()`, `forward_step()`, `predict_codes_autoregressive()`, and all GGML graph builders.
- `qwen3_tts.cpp` — Pipeline orchestration (~900 lines). Dispatches to the right synthesis path based on model type (base/custom_voice/voice_design). Handles model discovery, generation_config.json loading, and all public APIs.
- `text_tokenizer.cpp` — BPE tokenizer with `encode_for_tts()`, `encode_ref_text()`, `encode_instruct()`.
- `CMakeLists.txt` — Build configuration.

## Testing

### Reference Data

Deterministic reference data is generated by `scripts/generate_deterministic_reference.py` using float32 Python inference with greedy decoding. Files go in `reference/det_*.bin` (gitignored) and `reference/det_*.json` (tracked).

### Test Strategy

- `test_transformer` loads reference data and compares C++ output at each stage
- Pass criteria: prefill logits cosine > 0.99; speech codes partially match (F16 precision causes divergence)
- E2E comparison (`compare_e2e.py`): checks both pipelines produce valid non-silent audio with similar duration; waveform correlation is informational only

### Running Tests

```bash
bash scripts/run_all_tests.sh           # Full suite
./build/test_transformer --ref-dir reference/  # Transformer only
```

## Git Conventions

- Conventional commits: `feat(scope):`, `fix(scope):`, `docs:`
- Scopes: `transformer`, `vocoder`, `test`, `timing`
- One logical change per commit
- Do not commit model files (*.gguf) or reference binaries (*.bin)

## Known Limitations

- F16 model weights cause autoregressive divergence vs Python's float32 — speech codes differ but audio is perceptually equivalent
- M-RoPE uses 1D positions — matches Python's actual behaviour: all 3 M-RoPE dimensions use the same sequential counter for TTS (the 3D structure only diverges for vision inputs, which TTS never has). No quality impact.
- **Batch inference** — implemented as round-robin sequential decode across N utterances per frame step. Functionally correct; not true data-parallel GPU batch inference (all N entries share one KV cache slot each, decoded one at a time per step). See `synthesize_batch()` in `qwen3_tts.cpp` and `generate_batch()` in `tts_transformer.cpp`.
- **ICL voice cloning** — fully implemented. The Mimi audio encoder is loaded from the tokenizer GGUF and achieves 98.9% code match vs Python (F16) / 100% (F32). `generate_icl()` is used when `--ref-text` is provided alongside `--reference`.
- **Abort callback** — wired to the CPU backend only via `ggml_backend_cpu_set_abort_callback`. Has no effect on GPU backends (Vulkan/CUDA/Metal) during heavy compute steps — those schedulers do not support mid-graph abort. On GPU builds, aborting is only possible between frame steps (checked in the generate loop).
- `--top-p` is parsed and applied in both the main talker and code predictor sampling
- Top-level CMake expects vendored GGML at `./ggml`

## Performance Profile

The code predictor is the primary bottleneck (~71% of generation time) because it runs 15 sequential forward passes per frame (1 prefill + 14 autoregressive steps). The talker accounts for ~27%. Graph build/alloc and data I/O are negligible (<1%).
