#pragma once

#include "text_tokenizer.h"
#include "tts_transformer.h"
#include "audio_tokenizer_encoder.h"
#include "audio_tokenizer_decoder.h"
#include "mimi_encoder.h"
#include "ggml-backend.h"   /* for ggml_abort_callback */

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <cstdint>
#include <thread>

namespace qwen3_tts {

// ============================================================
// Model type constants  (mirrors Python tts_model_type)
// ============================================================
static constexpr const char * MODEL_TYPE_BASE         = "base";
static constexpr const char * MODEL_TYPE_CUSTOM_VOICE = "custom_voice";
static constexpr const char * MODEL_TYPE_VOICE_DESIGN = "voice_design";

// ============================================================
// Generation parameters
// ============================================================
struct tts_params {
    // Maximum audio tokens to generate per synthesis
    int32_t max_audio_tokens = 4096;

    // Main-talker sampling
    float   temperature      = 0.9f;   // 0 = greedy
    float   top_p            = 1.0f;
    int32_t top_k            = 50;     // 0 = disabled
    float   repetition_penalty = 1.05f;

    // Sub-talker (code predictor CB1-15) sampling
    // -1.0 / -1 means "inherit from main-talker values above"
    float   subtalker_temperature = -1.0f;
    int32_t subtalker_top_k       = -1;
    float   subtalker_top_p       = -1.0f;  // -1.0 = inherit from main top_p

    // min_p sampling: keep tokens where prob >= min_p * max_prob.
    // More principled than top-p for TTS codec tokens. 0.0 = disabled.
    float   min_p = 0.0f;

    // Frequency penalty: subtract (freq_penalty * token_count) from logit.
    // Reduces probability proportionally to how often a token has appeared.
    // 0.0 = disabled. Recommended range: 0.0–0.5 for codec tokens.
    float   frequency_penalty = 0.0f;

    // Presence penalty: subtract presence_penalty from logit if token appeared at all.
    // Flat penalty regardless of count. 0.0 = disabled.
    float   presence_penalty = 0.0f;

    // DRY (Don't Repeat Yourself) n-gram repetition penalty.
    // Penalises tokens that would extend an already-seen token sequence.
    // dry_multiplier: penalty scale (0.0 = disabled, 0.8 is a good start)
    // dry_base:       exponential growth per extra repeated token (default 1.75)
    // dry_allowed_length: min repeated n-gram length before penalising (default 2)
    // dry_penalty_last_n: how many recent tokens to scan (-1 = all generated, 0 = disabled)
    float   dry_multiplier      = 0.0f;
    float   dry_base            = 1.75f;
    int32_t dry_allowed_length  = 2;
    int32_t dry_penalty_last_n  = -1;

    // Dynamic temperature: scale temperature by entropy of the distribution.
    // When enabled, temperature is adapted per token: low entropy → lower temp,
    // high entropy → higher temp, preventing both over-confidence and chaos.
    // dyntemp_range: half-range of temperature variation around `temperature`.
    //   effective_temp = temperature ± dyntemp_range  (0.0 = disabled)
    // dyntemp_exponent: shaping exponent (1.0 = linear, default).
    float   dyntemp_range    = 0.0f;
    float   dyntemp_exponent = 1.0f;

    // Language: "auto", "english", "chinese", "japanese", "korean",
    //            "russian", "german", "french", "spanish", "italian", …
    // Can also be a raw integer codec ID (e.g. "2050" for English).
    std::string language = "auto";

    // Named speaker (CustomVoice model only), e.g. "Vivian", "Ryan"
    std::string speaker;

    // Style / emotion instruction text (VoiceDesign or CustomVoice models)
    std::string instruct;

    // Voice cloning: path to reference WAV or empty for no cloning
    std::string reference_audio;

    // ICL mode — use reference audio codes in addition to speaker embedding.
    // Requires reference_audio to be set AND ref_text to be non-empty.
    bool icl_mode = false;

    // Reference text transcript (required for ICL mode)
    std::string ref_text;

    // Non-streaming text input mode (matches Python non_streaming_mode=True).
    // When true, all text tokens are fed at once in the prefill followed by
    // codec_bos; trailing = tts_pad. Produces slightly different output cadence.
    // Default false = streaming mode (interleaved text + codec).
    bool non_streaming_mode = false;

    // Threading — default to hardware concurrency (capped at 8 to avoid diminishing returns)
    int32_t n_threads = (int32_t)std::min((unsigned)8u, std::thread::hardware_concurrency());

    // Logging
    bool print_progress = false;
    bool print_timing   = false;  // default OFF — matches C API default; set true explicitly if needed
};

// ============================================================
// Result
// ============================================================
struct tts_result {
    std::vector<float> audio;
    int32_t sample_rate = 24000;
    bool    success     = false;
    std::string error_msg;

    int64_t t_tokenize_ms = 0;
    int64_t t_encode_ms   = 0;
    int64_t t_generate_ms = 0;
    int64_t t_decode_ms   = 0;
    int64_t t_total_ms    = 0;

    uint64_t mem_rss_start_bytes  = 0;
    uint64_t mem_rss_end_bytes    = 0;
    uint64_t mem_rss_peak_bytes   = 0;
    uint64_t mem_phys_start_bytes = 0;
    uint64_t mem_phys_end_bytes   = 0;
    uint64_t mem_phys_peak_bytes  = 0;
};

// ============================================================
// Callbacks
// ============================================================

// Progress callback: (tokens_generated, max_tokens)
// Return non-zero to request early stop (best-effort).
using tts_progress_callback_t = std::function<int(int, int)>;

// Per-frame logits callback.
// Called once per generated frame, after CB0 sampling, before CB1-15 prediction.
//   frame_idx:        0-based frame index
//   cb0_logits:       float[codec_vocab_size] — talker raw logits for codebook 0
//   cb0_logits_size:  number of elements in cb0_logits (codec_vocab_size, typically 3072)
//   cb0_token:        the token that was sampled for codebook 0
// Return non-zero to stop generation early (best-effort).
using tts_logits_callback_t = std::function<int(
    int32_t frame_idx,
    const float * cb0_logits, int32_t cb0_logits_size,
    int32_t cb0_token)>;

// Streaming audio chunk callback.
// Called with decoded PCM audio each time a chunk of frames has been decoded.
//   samples:      float32 PCM, 24 kHz mono, normalized [-1, 1]
//   n_samples:    number of samples in this chunk
//   sample_rate:  always 24000
//   is_last:      1 if this is the final chunk
// Return non-zero to abort (best-effort — remaining synthesis is cancelled).
using tts_audio_chunk_callback_t = std::function<int(
    const float * samples, int32_t n_samples,
    int32_t sample_rate, int is_last)>;

// ============================================================
// Main TTS class
// ============================================================
class Qwen3TTS {
public:
    Qwen3TTS();
    ~Qwen3TTS();

    // ---- Loading -------------------------------------------------------

    // Load models from a directory.  Supports both 0.6B and 1.7B checkpoints.
    // Auto-detects available GGUF files using the patterns:
    //   qwen3-tts-*.gguf  (transformer + speaker encoder)
    //   qwen3-tts-tokenizer-*.gguf  (vocoder)
    bool load_models(const std::string & model_dir);

    // Same as load_models() but uses an explicit tokenizer GGUF path instead of
    // auto-discovering it alongside the model. Useful when you want to compare
    // different tokenizer precisions (e.g. f16-f16 vs f16-f32 Mimi) without
    // renaming files.
    bool load_models(const std::string & model_dir,
                     const std::string & tokenizer_path);

    // Unload all models and free GPU/CPU buffers.
    // The object remains valid — call load_models() again to reload.
    void unload_models();

    // Optionally load generation defaults from a JSON file.
    // JSON keys: do_sample, temperature, top_k, top_p, repetition_penalty,
    //            subtalker_dosample, subtalker_temperature, subtalker_top_k,
    //            max_new_tokens
    bool load_generation_config(const std::string & json_path);

    // ---- High-level synthesis APIs -------------------------------------

    // Synthesize text using the current model type and params.
    // This is the recommended entry point — it dispatches to the correct
    // internal path based on model type and params.
    tts_result synthesize(const std::string & text,
                          const tts_params  & params = tts_params());

    // Convenience: voice-clone from a WAV file path (x-vector-only mode)
    tts_result synthesize_with_voice(const std::string & text,
                                      const std::string & reference_audio,
                                      const tts_params  & params = tts_params());

    // Convenience: voice-clone from raw float32 samples (24 kHz, mono)
    tts_result synthesize_with_voice(const std::string & text,
                                      const float * ref_samples, int32_t n_ref_samples,
                                      const tts_params  & params = tts_params());

    // Batch synthesis: process N texts simultaneously, sharing the same speaker embedding.
    // Each entry in the result vector corresponds to one text.
    // instruct_per_entry: optional per-entry instruct text (VoiceDesign/CustomVoice).
    //   When non-null, must have the same size as texts. Empty strings = no instruct for that entry.
    // Returns empty vector on failure (check get_error()).
    std::vector<tts_result> synthesize_batch(
        const std::vector<std::string> & texts,
        const float * speaker_embedding,
        const tts_params  & params = tts_params(),
        const std::vector<std::string> * instruct_per_entry = nullptr);

    // Convenience: synthesize with a pre-computed speaker embedding vector
    tts_result synthesize_with_embedding(const std::string & text,
                                          const std::vector<float> & embedding,
                                          const tts_params  & params = tts_params());

    // Convenience overload accepting a raw pointer + size
    tts_result synthesize_with_embedding(const std::string & text,
                                          const float * embedding, int32_t embedding_size,
                                          const tts_params  & params = tts_params());

    // ---- Speaker embedding utilities -----------------------------------

    // Extract speaker embedding from a WAV file
    bool extract_speaker_embedding(const std::string & reference_audio,
                                   std::vector<float> & embedding);

    // Extract speaker embedding from raw samples (24 kHz, mono)
    bool extract_speaker_embedding(const float * ref_samples, int32_t n_ref_samples,
                                   std::vector<float> & embedding,
                                   const tts_params  & params = tts_params());

    // Save / load speaker embedding to/from a binary file
    bool save_speaker_embedding(const std::string & path,
                                const std::vector<float> & embedding);
    bool load_speaker_embedding(const std::string & path,
                                std::vector<float> & embedding);

    // ---- Model introspection -------------------------------------------

    // Returns "base", "custom_voice", "voice_design", or ""
    std::string get_model_type() const;

    // Returns "0.6b", "1.7b", or ""
    std::string get_model_size() const;

    // Returns list of supported speaker names (CustomVoice models only)
    std::vector<std::string> get_supported_speakers() const;

    // Returns list of supported language names
    std::vector<std::string> get_supported_languages() const;

    // Resolve a language name/alias to its codec token ID.
    // Returns -1 if not found (caller should use english_language_id).
    int32_t resolve_language_id(const std::string & language_name) const;

    // Resolve a speaker name to its codec token ID (-1 if not found)
    int32_t resolve_speaker_id(const std::string & speaker_name) const;

    // ---- Misc ----------------------------------------------------------
    void set_progress_callback(tts_progress_callback_t callback);

    // Abort callback — cancels synthesis mid-graph compute.
    // fn(data) returns true → abort. Pass nullptr to clear.
    // NOTE: On GPU backends (Vulkan, CUDA, Metal) this has no effect during
    // heavy graph compute steps — those schedulers do not support mid-graph
    // cancellation. On GPU builds, synthesis can only be interrupted between
    // codec frame steps (checked in the generate loop). On CPU-only builds,
    // the callback fires per graph node and cancels immediately.
    void set_abort_callback(ggml_abort_callback fn, void * userdata);
    void clear_abort_callback();

    // Per-frame logits callback — called once per generated codec frame.
    // Provides CB0 logits (talker output) and the sampled CB0 token.
    // Return non-zero from the callback to stop generation early.
    void set_logits_callback(tts_logits_callback_t callback);

    // Streaming audio chunk callback — delivers decoded PCM as it is produced
    // without waiting for the full synthesis to complete.
    // chunk_frames controls how many codec frames are batched before decoding.
    // 0 = use default (12 frames = ~1 second of audio at 12 Hz).
    void set_audio_chunk_callback(tts_audio_chunk_callback_t callback,
                                   int32_t chunk_frames = 0);

    // Synthesize and return raw speech codes without decoding to audio.
    // codes_out: [n_frames * n_codebooks] row-major int32_t.
    // n_codebooks_out: filled with n_codebooks (always 16).
    // Returns number of frames, or -1 on failure.
    int32_t synthesize_codes(const std::string & text,
                              std::vector<int32_t> & codes_out,
                              int32_t & n_codebooks_out,
                              const tts_params & params = tts_params());

    // Convenience: synthesize codes with voice reference.
    int32_t synthesize_codes_with_voice(const std::string & text,
                                         const std::string & reference_audio,
                                         std::vector<int32_t> & codes_out,
                                         int32_t & n_codebooks_out,
                                         const tts_params & params = tts_params());

    // Convenience: synthesize codes with pre-computed embedding.
    int32_t synthesize_codes_with_embedding(const std::string & text,
                                             const float * embedding,
                                             int32_t embedding_size,
                                             std::vector<int32_t> & codes_out,
                                             int32_t & n_codebooks_out,
                                             const tts_params & params = tts_params());

    // Decode previously obtained speech codes to audio (e.g. from synthesize_codes).
    // codes: [n_frames * n_codebooks] row-major int32_t.
    tts_result decode_speech_codes(const int32_t * codes,
                                    int32_t n_frames,
                                    int32_t n_codebooks,
                                    const tts_params & params = tts_params());

    const std::string & get_error() const { return error_msg_; }
    bool is_loaded() const { return models_loaded_; }

    // Returns the speaker-row width required by the loaded transformer.
    // Before loading, falls back to available encoder configuration/metadata.
    // Callers must query this instead of reusing an embedding across model sizes.
    int32_t get_embedding_dim() const;

    // True when the Mimi encoder was loaded from the tokenizer GGUF (ICL path).
    bool has_mimi_encoder() const { return mimi_encoder_loaded_; }

private:
    // Core internal synthesis that all public paths funnel into.
    // If codes_only=true, stops after generation (no vocoder) and populates result.audio
    // with an empty vector; the raw speech_codes are returned via codes_out.
    tts_result synthesize_internal(const std::string & text,
                                   const float * speaker_embedding,
                                   const tts_params & params,
                                   tts_result & result,
                                   std::vector<int32_t> * codes_out = nullptr);

    // Decode codes in streaming chunks, calling audio_chunk_callback_ for each.
    // Falls through to a single-shot decode when no chunk callback is registered.
    bool decode_codes_streaming(const std::vector<int32_t> & codes,
                                 tts_result & result,
                                 const tts_params & params);

    // Encoder lazy-load helper
    bool ensure_encoder_loaded(const tts_params & params, tts_result * result = nullptr);

    // Decode audio codes to waveform (lazy-loads decoder in low-mem mode)
    bool decode_codes(const std::vector<int32_t> & codes,
                       tts_result & result,
                       const tts_params & params);

    // Translate tts_params language string to codec token ID
    int32_t params_language_id(const tts_params & params) const;

    TextTokenizer        tokenizer_;
    TTSTransformer       transformer_;
    AudioTokenizerEncoder audio_encoder_;
    AudioTokenizerDecoder audio_decoder_;
    MimiEncoder           mimi_encoder_;   // for ICL reference-audio encoding

    bool models_loaded_      = false;
    bool encoder_loaded_     = false;
    bool transformer_loaded_ = false;
    bool decoder_loaded_     = false;
    bool mimi_encoder_loaded_= false;
    bool low_mem_mode_       = false;

    // From GGUF metadata (qwen3-tts.speaker_encoder.embedding_length).
    int32_t speaker_embedding_dim_ = 1024;
    // From GGUF metadata (qwen3-tts.speaker_encoder.sample_rate).
    int32_t encoder_sample_rate_   = 24000;

    std::string error_msg_;
    std::string tts_model_path_;
    std::string decoder_model_path_;

    // generation_config.json defaults
    struct gen_defaults {
        bool    do_sample          = true;
        float   temperature        = 0.9f;
        int32_t top_k              = 50;
        float   top_p              = 1.0f;
        float   repetition_penalty = 1.05f;
        float   subtalker_temp     = -1.0f;
        int32_t subtalker_top_k    = -1;
        float   subtalker_top_p    = 1.0f;
        int32_t max_new_tokens     = 2048;
        bool    loaded             = false;
    } gen_defaults_;

    tts_progress_callback_t    progress_callback_;
    tts_logits_callback_t      logits_callback_;
    tts_audio_chunk_callback_t audio_chunk_callback_;
    int32_t                    audio_chunk_frames_ = 12;  // ~1 s at 12 Hz

    // Stored so abort works even if set before transformer reload (low-mem mode).
    ggml_abort_callback abort_cb_      = nullptr;
    void *              abort_cb_data_ = nullptr;
};

// ============================================================
// Standalone WAV utilities
// ============================================================
bool load_audio_file(const std::string & path,
                     std::vector<float> & samples,
                     int & sample_rate);

bool save_audio_file(const std::string & path,
                     const std::vector<float> & samples,
                     int sample_rate);

// Resample float32 PCM from input_rate to output_rate using Kaiser-windowed sinc.
// Same resampler used internally for reference audio input.
void resample_audio(const float * input, int32_t n_input,
                    int32_t input_rate, int32_t output_rate,
                    std::vector<float> & output);

bool save_speaker_embedding(const std::string & path,
                             const std::vector<float> & embedding);

bool load_speaker_embedding(const std::string & path,
                             std::vector<float> & embedding);

} // namespace qwen3_tts
