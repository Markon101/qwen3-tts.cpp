# Architecture & Feature Reference

Complete technical reference for `qwen3-tts.cpp`. Covers the full pipeline,
all model variants, every feature, the C API surface, and internal implementation
details for contributors and integrators.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline](#pipeline)
3. [Model Variants & Sizes](#model-variants--sizes)
4. [Source Layout](#source-layout)
5. [Build System](#build-system)
6. [Model Files](#model-files)
7. [Synthesis Modes](#synthesis-modes)
8. [Sampling Parameters](#sampling-parameters)
9. [Callbacks](#callbacks)
10. [Batch Inference](#batch-inference)
11. [C API](#c-api)
12. [GGUF Format](#gguf-format)
13. [Known Limitations](#known-limitations)

---

## Overview

`qwen3-tts.cpp` is a pure C++17 inference runtime for
[Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) using
[GGML](https://github.com/ggml-org/ggml) as the compute backend. It requires
no Python or PyTorch at runtime.

Supported hardware: CPU (any), Vulkan (cross-platform GPU), CUDA (NVIDIA),
Metal (Apple — CoreML code predictor optional).

---

## Pipeline

```
Text input
    │
    ▼
┌─────────────────────────────────────┐
│  1. Text Tokenizer (BPE)            │  src/text_tokenizer.{h,cpp}
│     Qwen2 tiktoken vocabulary       │
│     151,936 token vocab             │
└──────────────────┬──────────────────┘
                   │  token IDs
                   ▼
┌─────────────────────────────────────┐
│  2. Speaker Encoder (ECAPA-TDNN)    │  src/audio_tokenizer_encoder.{h,cpp}
│     Reference WAV → 1024-dim        │
│     x-vector speaker embedding      │  (optional — Base model only)
│     Res2Net + SE blocks + ASP pool  │
└──────────────────┬──────────────────┘
                   │  speaker embedding [1024]
                   ▼
┌─────────────────────────────────────┐
│  3. TTS Transformer                 │  src/tts_transformer.{h,cpp}
│                                     │
│   ┌─────────────────────────────┐   │
│   │  Talker (28L Qwen2)         │   │
│   │  GQA: 16 heads / 8 KV heads │   │
│   │  RoPE θ=1,000,000           │   │
│   │  Flash attention (decode)   │   │
│   │  KV cache (F16 default)     │   │
│   │  Outputs: CB0 logits +      │   │
│   │           hidden states     │   │
│   └──────────────┬──────────────┘   │
│                  │ hidden [hidden_size]
│   ┌──────────────▼──────────────┐   │
│   │  Code Predictor (5L)        │   │
│   │  15 autoregressive steps    │   │
│   │  per frame (CB1–CB15)       │   │
│   │  Separate KV cache          │   │
│   └──────────────┬──────────────┘   │
└──────────────────┼──────────────────┘
                   │  speech codes [N_frames × 16]
                   ▼
┌─────────────────────────────────────┐
│  4. Vocoder (WavTokenizer)          │  src/audio_tokenizer_decoder.{h,cpp}
│     VQ lookup → pre-transformer     │
│     → ConvNeXt upsample × 2        │
│     → Snake + ConvTranspose × 4    │
│     Upsample: 8×5×4×3 = 480×       │
│     Output: 24 kHz mono PCM        │
└─────────────────────────────────────┘
```

**Frame rate:** 12 Hz (one codec frame = ~83 ms of audio).
**Latency:** First audio available after full synthesis completes (non-streaming
vocoder). Streaming chunked decode is available via `set_audio_chunk_callback()`.

---

## Model Variants & Sizes

### Variants

| Variant | What it does | Key difference |
|---------|-------------|----------------|
| **Base** | Voice cloning from reference audio; anonymous synthesis without reference | ECAPA-TDNN speaker encoder present |
| **CustomVoice** | ~30 named speaker presets (e.g. Vivian, Ryan); optional style instruct | Speaker ID lookup table in GGUF |
| **VoiceDesign** | Free-form voice description in natural language | Instruct-conditioned prefill |

All three variants support `--reference` voice cloning (x-vector path). Only
**Base** supports full **ICL** (in-context learning) cloning via `--ref-text`.

### Sizes

| Parameter | 0.6B | 1.7B |
|-----------|------|------|
| Talker hidden | 1024 | **2048** |
| Talker intermediate | 3072 | **6144** |
| Talker layers | 28 | 28 |
| Code predictor hidden | 1024 | 1024 |
| Code predictor intermediate | 3072 | 3072 |
| Code predictor layers | 5 | 5 |
| Attention heads / KV heads | 16 / 8 | 16 / 8 |
| Head dim | 128 | 128 |
| Codec vocab | 3072 | 3072 |
| Codebooks | 16 | 16 |
| GGUF size (F16) | ~1.75 GB | ~3.86 GB |
| GGUF size (Q8_0) | ~1.0 GB | ~2.1 GB |

For 1.7B the talker hidden size (2048) is projected to code predictor hidden
size (1024) via `small_to_mtp_projection` (linear layer stored in GGUF as
`code_pred.proj.weight`).

---

## Source Layout

```
src/
  main.cpp                       CLI entry point + server mode
  qwen3_tts.{h,cpp}             Pipeline orchestration — primary public API
  tts_transformer.{h,cpp}       Core LM: talker + code predictor + KV cache + sampling
  text_tokenizer.{h,cpp}        BPE tokenizer (reads vocab from GGUF)
  audio_tokenizer_encoder.{h,cpp}  ECAPA-TDNN speaker encoder
  audio_tokenizer_decoder.{h,cpp}  WavTokenizer vocoder
  mimi_encoder.{h,cpp}          Mimi audio encoder (ICL voice cloning)
  gguf_loader.{h,cpp}           GGUF I/O, mmap, shared GPU backend singleton
  qwen3tts_c_api.{h,cpp}       C FFI bindings (primary integration surface)
  coreml_code_predictor.h       CoreML code predictor interface
  coreml_code_predictor.mm      CoreML implementation (compiled on Apple only)
  coreml_code_predictor_stub.cpp  No-op stub for non-Apple builds

tests/
  test_transformer.cpp          Deterministic prefill + speech code reference comparison
  test_batch.cpp                Batch inference correctness + KV isolation
  test_decoder.cpp              Vocoder L2/correlation vs Python reference
  test_tokenizer.cpp            BPE encode/decode round-trip
  test_encoder.cpp              Speaker encoder x-vector extraction
  test_mimi_encoder.cpp         Mimi encoder codec code match vs Python
  test_codebook.cpp             VQ codebook round-trip
  test_vq_only.cpp              VQ-only encode/decode
  bench_batch.cpp               Batch throughput benchmark

scripts/
  convert_tts_to_gguf.py        HuggingFace safetensors → TTS GGUF
  convert_tokenizer_to_gguf.py  HuggingFace safetensors → Tokenizer/Vocoder GGUF
  setup_pipeline_models.py      One-shot download + convert all models
  generate_deterministic_reference.py  Generate Python reference data for tests
  compare_e2e.py                End-to-end Python vs C++ comparison

tools/
  build-scripts/build.bat       Windows CMake + Ninja build
  build-scripts/build.sh        macOS/Linux CMake + Ninja build
  model-downloader/download.py  Pre-built GGUF downloader (no torch required)
  model-downloader/download.bat Windows wrapper
  model-downloader/download.sh  macOS/Linux wrapper
  model-converter/setup_models.py   Interactive download+convert wizard (Python)
  model-converter/separate/    Per-component download scripts

docs/
  architecture.md               This file
  porting.md                    Guide for forking to a new TTS model
  tensor_mapping.md             HuggingFace → GGUF tensor name reference
  benchmarks/OPTIMIZATION.md   Performance analysis and bottleneck notes
  handoff.md                    Session continuity notes (internal)
```

---

## Build System

CMake 3.14+ with C++17. GGML is vendored as a submodule at `./ggml`.

### Build options

| CMake flag | Default | Effect |
|------------|---------|--------|
| `QWEN3_TTS_KV_F32` | OFF | Use F32 KV cache (2× RAM, bit-exact with Python) |
| `QWEN3_TTS_TIMING` | OFF | Enable detailed per-stage timing instrumentation |
| `QWEN3_TTS_COREML` | ON | Build CoreML code predictor bridge (Apple only) |
| `QWEN3_TTS_AARCH64_MCPU` | `auto` | AArch64 release CPU tuning: `auto`, `oryon-1`, `native`, or `generic`; compiler support is checked |

### GPU backends

Detected automatically from `ggml/build/`:

| Backend | Detection | Platform |
|---------|-----------|----------|
| Metal | `ggml/build/src/ggml-metal/` | macOS only |
| CUDA | `ggml/build/src/ggml-cuda/` | Linux/Windows, NVIDIA |
| Vulkan | `ggml/build/src/ggml-vulkan/` | Cross-platform |

Build with GPU:
```bat
REM Windows + Vulkan
cmake -S ggml -B ggml/build -DGGML_VULKAN=ON
cmake --build ggml/build
cmake -S . -B build-ninja -G Ninja
cmake --build build-ninja --config Release
```

### Build output

All executables output to `build-ninja/` (Ninja) or `build/` (Make/MSVC).
On Windows, GGML DLLs are automatically staged alongside the executables.

---

## Model Files

Two GGUF files are required:

| File | Contains | Converted by |
|------|----------|-------------|
| `qwen3-tts-{size}-{quant}.gguf` | Talker + code predictor + speaker encoder | `convert_tts_to_gguf.py` |
| `qwen3-tts-tokenizer-{tok}-{mimi}.gguf` | Vocoder decoder + Mimi encoder | `convert_tokenizer_to_gguf.py` |

The tokenizer GGUF is shared across all model variants and sizes — download once.

### Tokenizer GGUF naming

The tokenizer filename encodes two independent precision choices:

| Part | Controls | Affects |
|------|----------|---------|
| `{tok}` (first tag) | Vocoder decoder weights | Audio output quality for all synthesis |
| `{mimi}` (second tag) | Mimi encoder weights | ICL voice cloning accuracy only |

| Filename | Vocoder | Mimi | Use case |
|----------|---------|------|----------|
| `qwen3-tts-tokenizer-f16-f32.gguf` | F16 | F32 | **Recommended for ICL** — bit-exact cloning |
| `qwen3-tts-tokenizer-f16-f16.gguf` | F16 | F16 | Good general use, no ICL quality loss in practice |
| `qwen3-tts-tokenizer-f16-q8_0.gguf` | F16 | Q8_0 | Smallest Mimi, not recommended for ICL |
| `qwen3-tts-tokenizer-f32-f32.gguf` | F32 | F32 | No benefit over f16-f32, double size |

> **Note:** F32 vocoder has no quality benefit over F16 — the source weights are BF16.
> Q8_0 is not available for the vocoder `--type` — 3D convolution weights in the
> WavTokenizer cannot be quantized to Q8_0 (non-2D tensors are unsupported).
> The Mimi encoder precision only matters for ICL voice cloning (`--ref-text` mode).
> F32 Mimi = 100% code match vs Python; F16 Mimi = 98.9% match; Q8_0 Mimi = 94.3%.

### Supported quantization

| Type | Status | Notes |
|------|--------|-------|
| F16 | ✅ | Recommended — full precision |
| F32 | ✅ | Same quality as F16 (source is BF16), double size |
| Q8_0 | ✅ | Near-lossless, ~43% smaller |
| Q6_K–Q2_K | ❌ | Not yet supported by Python converter |

### GGUF metadata namespace

All keys use the `qwen3-tts.*` prefix. Key metadata stored:
- Model variant (`qwen3-tts.model_type`): `base`, `custom_voice`, `voice_design`
- Model size (`qwen3-tts.model_size`): `0.6b`, `1.7b`
- All special token IDs (codec_pad, codec_bos, think tokens, etc.)
- Speaker name → ID table (CustomVoice)
- Language name → ID table
- Speaker encoder dimensions and sample rate

### Auto-discovery

`Qwen3TTS::load_models(dir)` scans for GGUFs in priority order:

**TTS model** (Q8_0 preferred → F16 fallback, 1.7B preferred → 0.6B):
```
qwen3-tts-1.7b-q8_0.gguf → qwen3-tts-1.7b-f16.gguf → qwen3-tts-0.6b-q8_0.gguf → qwen3-tts-0.6b-f16.gguf → ...
```

**Tokenizer** (f16 vocoder preferred, f32 Mimi preferred for best ICL):
```
qwen3-tts-tokenizer-f16-f32.gguf  ← auto-selected if present (best ICL)
qwen3-tts-tokenizer-f16-f16.gguf
qwen3-tts-tokenizer-f16-q8_0.gguf
qwen3-tts-tokenizer-f32-f32.gguf
qwen3-tts-tokenizer-f32-f16.gguf
qwen3-tts-tokenizer-f32-q8_0.gguf
qwen3-tts-tokenizer-f16.gguf      ← legacy single-tag (backward compat)
qwen3-tts-tokenizer-f32.gguf
qwen3-tts-tokenizer-q8_0.gguf
qwen3-tts-tokenizer.gguf
```

Pass a direct `.gguf` file path to bypass auto-selection entirely.

---

## Synthesis Modes

### Basic synthesis

No reference audio. Base models produce a neutral anonymous voice.
CustomVoice models use the default or specified named speaker.

```bash
./qwen3-tts-cli -m models -t "Hello world" -o out.wav
```

### Voice cloning (x-vector)

Reference WAV → ECAPA-TDNN encoder → 1024-dim speaker embedding → synthesis.
Works with any Base model. Reference audio: 3–30 seconds recommended.

```bash
./qwen3-tts-cli -m models -t "Hello" -r ref.wav -o clone.wav
```

> **Warning:** Reference clips over 30 seconds significantly increase encoding
> time (ECAPA-TDNN runs full attention over the mel spectrogram).

### ICL voice cloning (in-context learning)

Full ICL: reference audio codes prepended to the TTS context. Requires:
- `--reference ref.wav`
- `--ref-text "Transcript of reference audio"`
- A tokenizer GGUF with the Mimi encoder present

Produces higher quality cloning than x-vector alone, especially for prosody.
ICL with F16 achieves 98.9% code match vs Python; F32 Mimi achieves 100%.

```bash
./qwen3-tts-cli -m models -t "Hello" -r ref.wav --ref-text "The reference." -o icl.wav
```

ICL is only available for **Base** model (the Python reference only tests 1.7B;
0.6B ICL works but quality is lower).

### Named speaker (CustomVoice)

```bash
./qwen3-tts-cli -m models -t "Hello" --speaker Vivian -o vivian.wav
./qwen3-tts-cli -m models --list-speakers   # show available speakers
```

### Voice design (VoiceDesign, 1.7B only)

```bash
./qwen3-tts-cli -m models -t "Hello" --instruct "Calm, warm female voice" -o out.wav
```

### Style/emotion instruct (CustomVoice 1.7B)

```bash
./qwen3-tts-cli -m models -t "Hello!" --speaker Vivian --instruct "Excited tone" -o out.wav
```

### Pre-computed speaker embeddings

Save an embedding once, reuse without re-encoding:

```bash
# Save
./qwen3-tts-cli -m models -r ref.wav --embedding-out voice.bin

# Reuse
./qwen3-tts-cli -m models -t "Hello" --embedding-in voice.bin -o out.wav
```

### Server mode

Load model once, process JSON requests from stdin, write responses to stdout:

```bash
./qwen3-tts-cli -m models --server
```

Request format (one JSON object per line):
```json
{"text": "Hello world", "output": "out.wav", "temperature": 0.9}
```

Response format:
```json
{"success": true, "output": "out.wav", "duration_s": 1.23, "t_total_ms": 450}
```

Supported fields: `text`, `output`, `reference`, `ref_text`, `embedding_in`,
`speaker`, `instruct`, `language`, `temperature`, `top_k`, `top_p`,
`repetition_penalty`, `max_tokens`, `vocoder_chunk_frames`. The latter must be an
integer from 17 to 256; it overrides the process default for that request.
Malformed JSON is rejected. Check each response's `success` field: a failed request
does not necessarily stop the server or make its eventual EOF exit nonzero.

### Streaming prefill modes

`--non-streaming`: all text tokens fed at once in one prefill block (matches
Python `non_streaming_mode=True`). Slightly different output cadence but
same quality. Default is streaming mode (interleaved text + codec tokens).

### Output resampling

```bash
./qwen3-tts-cli -m models -t "Hello" --output-rate 48000 -o hello_48k.wav
```

Native output is 24 kHz. 48 kHz upsampling uses Kaiser-windowed sinc
interpolation. Upsampling does not add information above 12 kHz Nyquist.

---

## Sampling Parameters

All parameters apply to the **main talker** (codebook 0 prediction). Sub-talker
parameters (code predictor, CB1–CB15) can be set independently with `--sub-*`
flags or inherit from main talker when set to `-1`.

| Parameter | CLI flag | Default | Notes |
|-----------|----------|---------|-------|
| Temperature | `--temperature` | 0.9 | 0 = greedy. **Greedy produces silent audio** — codec LMs collapse without noise |
| Top-k | `--top-k` | 50 | 0 = disabled |
| Top-p | `--top-p` | 1.0 | Nucleus sampling. 1.0 = disabled |
| Min-p | `--min-p` | 0.0 | Keep tokens where prob ≥ min_p × max_prob. 0 = disabled |
| Repetition penalty | `--repetition-penalty` | 1.05 | 1.0 = disabled |
| Frequency penalty | `--frequency-penalty` | 0.0 | Subtracts `freq_pen × token_count` from logit |
| Presence penalty | `--presence-penalty` | 0.0 | Flat subtract if token appeared at all |
| DRY multiplier | `--dry-multiplier` | 0.0 | N-gram repetition penalty scale. 0.8 is a good start |
| Dynamic temperature | `--dyntemp-range` | 0.0 | Half-range of entropy-adaptive temperature |
| Sub-talker temp | `--sub-temperature` | -1 | -1 = inherit main temperature |
| Sub-talker top-k | `--sub-top-k` | -1 | -1 = inherit main top-k |
| Sub-talker top-p | `--sub-top-p` | -1 | -1 = inherit main top-p |
| Max tokens | `--max-tokens` | 4096 | Maximum codec frames to generate |

### Special token IDs (0.6B Base defaults, read from GGUF at runtime)

| Token | ID |
|-------|-----|
| `tts_bos` | 151672 |
| `tts_eos` | 151673 |
| `tts_pad` | 151671 |
| `codec_pad` | 2148 |
| `codec_bos` | 2149 |
| `codec_eos` | 2150 |
| `codec_think` | 2154 |
| `codec_nothink` | 2155 |
| `codec_think_bos` | 2156 |
| `codec_think_eos` | 2157 |
| `english_language` | 2050 |

---

## Callbacks

All callbacks are set on the `Qwen3TTS` object before calling `synthesize()`.

### Progress callback

Called once per generated codec frame (~83 ms of audio).

```cpp
tts.set_progress_callback([](int tokens_done, int tokens_max) -> int {
    fprintf(stderr, "\r%d/%d frames", tokens_done, tokens_max);
    return 0;  // return non-zero to request early stop
});
```

### Per-frame logits callback

Called once per frame, after CB0 is sampled, before CB1–CB15 prediction.
Provides raw talker logits for codebook 0.

```cpp
tts.set_logits_callback([](int32_t frame_idx,
                            const float * logits, int32_t n_logits,
                            int32_t cb0_token) -> int {
    // inspect logits[0..n_logits-1]
    return 0;
});
```

### Streaming audio chunk callback

Delivers decoded PCM as it is produced, without waiting for full synthesis.
`chunk_frames` controls how many codec frames are batched per callback (default 12 = ~1 second). The emission size is capped by the vocoder graph window minus its 16-frame left context. ICL callbacks exclude the reference prefix. See [vocoder window controls and limits](FORK_AUDIT.md#controllable-vocoder-windows).

```cpp
tts.set_audio_chunk_callback([](const float * samples, int32_t n_samples,
                                 int32_t sample_rate, int is_last) -> int {
    // write samples to audio output
    return 0;  // return non-zero to abort remaining synthesis
}, /*chunk_frames=*/12);
```

### Abort callback

Cancels synthesis on user request. **Important limitation:** on GPU backends
(Vulkan/CUDA/Metal) the abort has no effect during heavy graph compute steps —
those schedulers do not support mid-graph cancellation. On GPU builds, synthesis
can only be interrupted between codec frame steps. On CPU-only builds,
cancellation fires per graph node for near-instant response.

```cpp
tts.set_abort_callback([](void *) -> bool {
    return user_requested_stop;
}, nullptr);
```

---

## Batch Inference

Process N texts with a shared speaker embedding. Each text generates
independently (no cross-attention between entries).

**Implementation note:** This is sequential round-robin decode — for each
codec frame step, all N entries are processed one at a time in a loop. It is
not true data-parallel GPU batching. Memory usage scales with N.

```cpp
std::vector<std::string> texts = {"Hello", "World", "How are you"};
auto results = tts.synthesize_batch(texts, speaker_embedding.data(), params);
```

Per-entry instruct text (VoiceDesign/CustomVoice):
```cpp
std::vector<std::string> instructs = {"calm", "excited", ""};
auto results = tts.synthesize_batch(texts, embedding, params, &instructs);
```

Via C API:
```c
Qwen3TtsResult ** results = qwen3_tts_synthesize_batch_ex(
    tts, texts, n_texts, embedding, emb_size, &params, instructs);
// free with:
qwen3_tts_free_batch_results(results, n_texts);
```

---

## C API

`src/qwen3tts_c_api.h` is the primary integration surface for non-C++ consumers
(Go, Rust, Python ctypes, Nim, Swift, etc.). See the header for full documentation.

### Lifecycle

```c
// Fill params with defaults first
Qwen3TtsParams params;
qwen3_tts_default_params(&params);

// Create engine (loads models from directory)
Qwen3Tts * tts = qwen3_tts_create("models/", /*n_threads=*/0);
if (!tts) {
    fprintf(stderr, "%s\n", qwen3_tts_get_last_create_error());
    return 1;
}

// Synthesize
Qwen3TtsResult * result = qwen3_tts_synthesize_ex(tts, "Hello world", &params);
if (result->success) {
    qwen3_tts_save_wav("out.wav", result->audio.samples,
                        result->audio.n_samples, result->audio.sample_rate);
}
qwen3_tts_free_result(result);

// Destroy
qwen3_tts_destroy(tts);
```

### Thread safety

One `Qwen3Tts*` handle is **not thread-safe**. For parallel synthesis, create
one handle per worker thread. Model weights are loaded independently per handle
(no shared state between handles).

### Key function groups

| Group | Functions |
|-------|-----------|
| Lifecycle | `qwen3_tts_create`, `qwen3_tts_create_with_config`, `qwen3_tts_destroy`, `qwen3_tts_unload` |
| Simple synthesis | `qwen3_tts_synthesize`, `qwen3_tts_synthesize_with_voice_file`, `qwen3_tts_synthesize_with_embedding` |
| Extended synthesis | `qwen3_tts_synthesize_ex` and `_ex` variants (returns timing + memory stats) |
| Batch | `qwen3_tts_synthesize_batch`, `qwen3_tts_synthesize_batch_ex`, `qwen3_tts_free_batch_results` |
| Speaker embedding | `qwen3_tts_extract_embedding_file`, `qwen3_tts_save_embedding`, `qwen3_tts_load_embedding` |
| Speech codes | `qwen3_tts_synthesize_codes`, `qwen3_tts_decode_codes` |
| Callbacks | `qwen3_tts_set_progress_callback`, `qwen3_tts_set_abort_callback`, `qwen3_tts_set_logits_callback`, `qwen3_tts_set_audio_chunk_callback` |
| Model info | `qwen3_tts_model_type`, `qwen3_tts_model_size`, `qwen3_tts_list_speakers`, `qwen3_tts_list_languages`, `qwen3_tts_has_mimi_encoder` |
| Audio I/O | `qwen3_tts_save_wav`, `qwen3_tts_load_wav`, `qwen3_tts_resample` |

### Codes-only path

Synthesize speech codes without decoding to audio. Useful for caching,
offline processing, or vocoder swapping:

```c
int32_t n_cb;
int32_t n_frames = qwen3_tts_synthesize_codes(tts, text, &params,
                                               NULL, 0, &n_cb); // size query
int32_t * codes = malloc(n_frames * n_cb * sizeof(int32_t));
qwen3_tts_synthesize_codes(tts, text, &params, codes, n_frames, &n_cb);

// Later, decode:
Qwen3TtsResult * r = qwen3_tts_decode_codes(tts, codes, n_frames, n_cb, &params);
```

---

## GGUF Format

### TTS model GGUF (`qwen3-tts-*.gguf`)

Tensor namespaces:
- `talker.*` — Main 28-layer transformer (text embeddings, attention, FFN, codec head)
- `code_pred.*` — 5-layer code predictor (per-codebook embeddings/heads, layers)
- `spk_enc.*` — ECAPA-TDNN speaker encoder (conv, res2net, SE, ASP, FC)

Key metadata keys:
```
qwen3-tts.model_type          "base" | "custom_voice" | "voice_design"
qwen3-tts.model_size          "0.6b" | "1.7b"
qwen3-tts.codec.pad_id        2148
qwen3-tts.codec.bos_id        2149
qwen3-tts.codec.eos_id        2150
qwen3-tts.speakers.names      string array
qwen3-tts.speakers.ids        int32 array
qwen3-tts.languages.names     string array
qwen3-tts.languages.ids       int32 array
qwen3-tts.speaker_encoder.embedding_length  1024
qwen3-tts.speaker_encoder.sample_rate       24000
```

### Tokenizer GGUF (`qwen3-tts-tokenizer-*.gguf`)

Tensor namespaces:
- `tok_dec.*` — WavTokenizer vocoder decoder
- `mimi_enc.*` — Mimi audio encoder (ICL path)

Mimi encoder metadata:
```
mimi.hidden_size              512
mimi.num_hidden_layers        8
mimi.num_attention_heads      8
mimi.codebook_size            2048
mimi.num_quantizers           32
mimi.upsampling_ratios        [8, 6, 5, 4]
```

---

## Known Limitations

| Limitation | Status | Notes |
|------------|--------|-------|
| K-quants (Q6_K–Q2_K) | Not supported | Python gguf library doesn't implement them. C++ runtime fully supports loading them if generated elsewhere |
| Abort on GPU backends | Partial | Abort callback fires between frame steps only; no mid-graph cancel on Vulkan/CUDA/Metal |
| Batch = true GPU parallelism | No | Round-robin sequential per frame step, not data-parallel |
| M-RoPE | 1D approximation | The talker uses M-RoPE (multi-dimensional) but for TTS all three position dimensions (temporal, height, width) are set to the same sequential counter — identical to Python's behaviour (`position_ids = cache_position.expand(3, ...)` with all rows equal). No quality impact. |
| ICL slow on CPU | Expected | Mimi encoder always runs on CPU even on GPU builds (its transformer uses plain scalar C++); ~13s for a 6.8s reference clip |
| Greedy decoding (temperature=0) | Silent audio | Codec LMs collapse to silence/pad tokens without sampling noise. Use temperature ≥ 0.1 |
| F32 1.7B conversion | OOM on <16 GB RAM | 1.7B F32 requires ~7.7 GB on disk and ~12–15 GB peak RAM during conversion |
| CoreML code predictor | Apple only | Gracefully no-ops on non-Apple builds |
