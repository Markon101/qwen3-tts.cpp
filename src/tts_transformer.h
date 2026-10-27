#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "coreml_code_predictor.h"

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <functional>
#include <random>
#include <unordered_set>
#ifdef QWEN3_TTS_TIMING
#include <chrono>
#endif

namespace qwen3_tts {

#ifdef QWEN3_TTS_TIMING
struct tts_timing {
    // Prefill phase
    double t_prefill_build_ms = 0;      // build_prefill_graph (embedding lookups, text projection)
    double t_prefill_forward_ms = 0;    // forward_prefill total
    double t_prefill_graph_build_ms = 0;  // build_prefill_forward_graph
    double t_prefill_graph_alloc_ms = 0;  // sched_alloc_graph
    double t_prefill_compute_ms = 0;      // sched_graph_compute
    double t_prefill_data_ms = 0;         // tensor_set + tensor_get + reset

    // Talker forward_step totals (accumulated across all frames)
    double t_talker_forward_ms = 0;       // total time in forward_step()
    double t_talker_graph_build_ms = 0;   // build_step_graph
    double t_talker_graph_alloc_ms = 0;   // sched_alloc_graph
    double t_talker_compute_ms = 0;       // sched_graph_compute
    double t_talker_data_ms = 0;          // tensor_set + tensor_get + reset

    // Code predictor totals (accumulated across all frames)
    double t_code_pred_ms = 0;            // total predict_codes_autoregressive
    double t_code_pred_init_ms = 0;       // init/clear KV cache + CB0 embed lookup
    double t_code_pred_prefill_ms = 0;    // code pred prefill (2-token, per frame)
    double t_code_pred_steps_ms = 0;      // code pred autoregressive steps (14 steps, per frame)
    double t_code_pred_graph_build_ms = 0;  // graph build (prefill + steps combined)
    double t_code_pred_graph_alloc_ms = 0;  // sched_alloc_graph
    double t_code_pred_compute_ms = 0;      // sched_graph_compute
    double t_code_pred_data_ms = 0;         // tensor_set + tensor_get + reset
    double t_code_pred_coreml_ms = 0;       // CoreML predictor compute + I/O

    // Embed lookups in generate() loop
    double t_embed_lookup_ms = 0;

    int32_t n_frames = 0;
    double t_generate_total_ms = 0;
};
#endif

#define QWEN3_TTS_MAX_NODES 16384

// TTS Transformer configuration (Qwen2-based Talker)
struct tts_transformer_config {
    // Text embedding
    int32_t text_vocab_size = 151936;
    int32_t text_embd_dim = 2048;
    
    // Talker transformer
    int32_t hidden_size = 1024;
    int32_t n_layers = 28;
    int32_t n_attention_heads = 16;
    int32_t n_key_value_heads = 8;
    int32_t intermediate_size = 3072;
    int32_t head_dim = 128;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    
    // M-RoPE sections [time, freq, channel] = [24, 20, 20]
    // NOTE: currently unused — reserved for future M-RoPE support
    
    // Codec vocabulary
    int32_t codec_vocab_size = 3072;  // talker.codec_embd/codec_head
    int32_t n_codebooks = 16;
    
    // Code predictor
    int32_t code_pred_layers = 5;
    int32_t code_pred_vocab_size = 2048;  // Per-codebook vocab
    // Code predictor hidden/intermediate (may differ from talker for 1.7B model)
    // 0.6B: code_pred_hidden_size=1024 (same as talker)
    // 1.7B: code_pred_hidden_size=1024 (talker=2048), code_pred_intermediate_size=3072 (talker=6144)
    int32_t code_pred_hidden_size = 0;        // 0 = inherit from hidden_size
    int32_t code_pred_intermediate_size = 0;  // 0 = inherit from intermediate_size
    
    // Special codec tokens
    int32_t codec_pad_id = 2148;
    int32_t codec_bos_id = 2149;
    int32_t codec_eos_id = 2150;

    int32_t tts_bos_token_id = 151672;
    int32_t tts_eos_token_id = 151673;
    int32_t tts_pad_token_id = 151671;

    int32_t codec_think_id = 2154;
    int32_t codec_nothink_id = 2155;
    int32_t codec_think_bos_id = 2156;
    int32_t codec_think_eos_id = 2157;

    int32_t english_language_id = 2050;

    // Model variant metadata (read from GGUF)
    std::string model_type;   // "base", "custom_voice", "voice_design"
    std::string model_size;   // "0.6b", "1.7b", etc.

    // Named speaker table: speaker_name (lowercase) -> codec token ID
    std::map<std::string, int32_t> spk_id;

    // Dialect override: speaker_name (lowercase) -> dialect language name (lowercase)
    // e.g. "speaker_cantonese" -> "cantonese"
    // When set, the dialect's language_id overrides the requested language_id.
    std::map<std::string, std::string> spk_is_dialect;

    // Language name table: language_name (lowercase) -> codec token ID
    std::map<std::string, int32_t> codec_language_id;
};

// Transformer layer weights
struct transformer_layer {
    struct ggml_tensor * attn_norm = nullptr;
    
    struct ggml_tensor * attn_q = nullptr;
    struct ggml_tensor * attn_k = nullptr;
    struct ggml_tensor * attn_v = nullptr;
    struct ggml_tensor * attn_output = nullptr;
    struct ggml_tensor * attn_q_norm = nullptr;
    struct ggml_tensor * attn_k_norm = nullptr;
    
    struct ggml_tensor * ffn_norm = nullptr;
    
    struct ggml_tensor * ffn_gate = nullptr;
    struct ggml_tensor * ffn_up = nullptr;
    struct ggml_tensor * ffn_down = nullptr;
};

// TTS Transformer model weights
struct tts_transformer_model {
    tts_transformer_config config;
    
    // Text embedding and projection
    struct ggml_tensor * text_embd = nullptr;      // [text_embd_dim, text_vocab_size]
    struct ggml_tensor * text_proj_fc1 = nullptr;  // [text_embd_dim, text_embd_dim]
    struct ggml_tensor * text_proj_fc1_bias = nullptr;
    struct ggml_tensor * text_proj_fc2 = nullptr;  // [text_embd_dim, hidden_size]
    struct ggml_tensor * text_proj_fc2_bias = nullptr;
    
    // Codec embedding (for autoregressive input)
    struct ggml_tensor * codec_embd = nullptr;     // [hidden_size, codec_vocab_size]
    
    // Talker transformer layers
    std::vector<transformer_layer> layers;
    
    // Final RMSNorm
    struct ggml_tensor * output_norm = nullptr;    // [hidden_size]
    
    // Codec head (for first codebook prediction)
    struct ggml_tensor * codec_head = nullptr;     // [hidden_size, codec_vocab_size]
    
     // Code predictor layers
     std::vector<transformer_layer> code_pred_layers;
     
     // Code predictor output norm (final RMS norm before lm_head)
     struct ggml_tensor * code_pred_output_norm = nullptr;  // [hidden_size]
     
    // Code predictor projection: projects talker hidden_size → code_pred_hidden_size
    // Only used when code_pred_hidden_size differs from hidden_size (e.g. 1.7B model)
    struct ggml_tensor * code_pred_proj = nullptr;    // [code_pred_hidden_size, hidden_size]
    struct ggml_tensor * code_pred_proj_bias = nullptr;  // [code_pred_hidden_size]

    // Code predictor per-codebook embeddings and heads (15 codebooks, 0 uses talker output)
    // code_pred_embd is in talker space (cfg.hidden_size), projected by code_pred_proj if needed
    std::vector<struct ggml_tensor *> code_pred_embd;  // [hidden_size, code_pred_vocab_size] x 15
    std::vector<struct ggml_tensor *> code_pred_head;  // [code_pred_hidden_size, code_pred_vocab_size] x 15
    
    // GGML context for tensor metadata
    struct ggml_context * ctx = nullptr;
    
    // Backend buffer for weights
    ggml_backend_buffer_t buffer = nullptr;
    
    // Tensor name to tensor mapping
    std::map<std::string, struct ggml_tensor *> tensors;

    // Host-side raw cache for codec_embd and code_pred_embd[0..14] to avoid GPU roundtrips/splits
    std::map<const struct ggml_tensor *, std::vector<uint8_t>> host_embd_raw;
};

// KV cache for autoregressive generation
struct tts_kv_cache {
    std::vector<struct ggml_tensor *> k_cache;
    std::vector<struct ggml_tensor *> v_cache;
    
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    
    int32_t n_ctx = 0;
    int32_t n_used = 0;
    int32_t head_dim = 128;
    int32_t n_kv_heads = 8;
    int32_t n_layers = 28;
    int32_t n_batch = 1;
};

// TTS Transformer state
struct tts_transformer_state {
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> compute_meta;

    tts_kv_cache cache;           // Talker KV cache (28 layers)
    tts_kv_cache code_pred_cache; // Code predictor KV cache (5 layers)

    // Persistent thread pool — avoids per-compute thread create/destroy overhead.
    // One thread is created per physical core (capped at n_threads).
    struct ggml_threadpool * threadpool     = nullptr;
    int32_t                  threadpool_n   = 0;  // n_threads the pool was created for

    // Abort callback — set by set_abort_callback(); passed into ggml_backend_sched_graph_compute.
    // Returning true from the callback cancels the current graph compute.
    ggml_abort_callback abort_cb      = nullptr;
    void *              abort_cb_data = nullptr;

    // Eval callback — fires for every graph node; use for debugging/profiling.
    ggml_backend_sched_eval_callback eval_cb      = nullptr;
    void *                           eval_cb_data = nullptr;
};

// TTS Transformer class
class TTSTransformer {
public:
    void set_seed(uint32_t seed) { rng_.seed(seed); }

    TTSTransformer();
    ~TTSTransformer();
    
    // Load model from GGUF file
    bool load_model(const std::string & model_path);

    // Release all model/runtime resources
    void unload_model();
    
    // Initialize KV cache
    bool init_kv_cache(int32_t n_ctx, int32_t n_batch = 1);
    
    // Clear KV cache
    void clear_kv_cache();

    // Set number of CPU threads for inference (default: 4).
    // Has no effect if using a GPU backend (Metal/CUDA/Vulkan).
    void set_n_threads(int32_t n_threads);
    
    // Initialize code predictor KV cache (5 layers, max 16 context)
    bool init_code_pred_kv_cache(int32_t n_ctx, int32_t n_batch = 1);
    
    // Clear code predictor KV cache
    void clear_code_pred_kv_cache();
    
    bool forward_prefill(const float * prefill_embd, int32_t n_tokens,
                         int32_t n_past, std::vector<float> & output,
                         std::vector<float> * logits_out = nullptr,
                         int32_t batch_idx = 0);

    bool forward_step(const float * step_embd, int32_t n_past,
                      std::vector<float> & output,
                      std::vector<float> * hidden_out = nullptr,
                      int32_t batch_idx = 0);
    
    // Run code predictor autoregressively to generate 15 codes (codebooks 1-15)
    // hidden: hidden states from talker [hidden_size]
    // codebook_0_token: the codebook 0 token (used to create 2-token prefill input)
    // output: generated codes for codebooks 1-15 [15]
    // batch_idx: which batch slot to use for KV cache (default 0)
    bool predict_codes_autoregressive(const float * hidden, int32_t codebook_0_token, 
                                       std::vector<int32_t> & output,
                                       float temperature = 0.9f,
                                       int32_t top_k = 50,
                                       float top_p = 1.0f,
                                       int32_t batch_idx = 0);
    
    // Generate speech codes autoregressively
    // text_tokens: input text token IDs [n_tokens]
    // speaker_embd: speaker embedding [hidden_size]
    // max_len: maximum number of frames to generate
    // output: generated speech codes [n_frames, n_codebooks]
    // subtalker_* params control independent sampling for the code predictor sub-model
    bool generate(const int32_t * text_tokens, int32_t n_tokens,
                  const float * speaker_embd, int32_t max_len,
                  std::vector<int32_t> & output,
                  int32_t language_id = 2050,
                  float repetition_penalty = 1.05f,
                  float temperature = 0.9f,
                  int32_t top_k = 50,
                  float top_p = 1.0f,
                  float subtalker_temperature = -1.0f,
                  int32_t subtalker_top_k = -1,
                  bool non_streaming_mode = false,
                  float subtalker_top_p = -1.0f);

    bool generate_icl(const int32_t * text_tokens, int32_t n_tokens,
                      const float * speaker_embd,
                      const int32_t * ref_text_tokens, int32_t n_ref_text_tokens,
                      const int32_t * ref_codes, int32_t n_ref_frames,
                      int32_t max_len,
                      std::vector<int32_t> & output,
                      int32_t language_id = 2050,
                      float repetition_penalty = 1.05f,
                      float temperature = 0.9f,
                      int32_t top_k = 50,
                      float top_p = 1.0f,
                      float subtalker_temperature = -1.0f,
                      int32_t subtalker_top_k = -1,
                      bool non_streaming_mode = false,
                      float subtalker_top_p = -1.0f);

    // Batch generation: process N independent texts simultaneously.
    // Each output vector receives codes for one utterance.
    // When instruct_tokens is provided per entry, build_prefill_graph_instruct is used.
    bool generate_batch(
        const int32_t * const * text_tokens,     // [n_batch] arrays, each [n_tokens_b]
        const int32_t * n_tokens_per_batch,       // [n_batch]
        const float * const * speaker_embds,      // [n_batch] arrays, each [hidden_size] (nullptr allowed)
        int32_t n_batch,
        int32_t max_len,
        std::vector<std::vector<int32_t>> & outputs,  // [n_batch]
        const int32_t * language_ids = nullptr,   // [n_batch], nullptr = all 2050
        const int32_t * const * instruct_tokens = nullptr,  // [n_batch], nullptr = no instruct
        const int32_t * n_instruct_tokens = nullptr,          // [n_batch], ignored when instruct_tokens is null
        float repetition_penalty = 1.05f,
        float temperature = 0.9f,
        int32_t top_k = 50,
        float top_p = 1.0f,
        float subtalker_temperature = -1.0f,
        int32_t subtalker_top_k = -1,
        float subtalker_top_p = -1.0f,
        bool non_streaming_mode = false);

    bool generate_from_prefill(const std::vector<float> & prefill_embd,
                                const std::vector<float> & trailing_text_hidden,
                                const std::vector<float> & tts_pad_embed,
                                int32_t max_len,
                                std::vector<int32_t> & output,
                                float repetition_penalty = 1.05f,
                                float temperature = 0.9f,
                                int32_t top_k = 50,
                                float top_p = 1.0f,
                                float subtalker_temperature = -1.0f,
                                int32_t subtalker_top_k = -1,
                                float subtalker_top_p = -1.0f);

    // Set extended sampling parameters used by the generate loop.
    // Call before generate() / generate_icl() / generate_from_prefill().
    void set_extended_sampling(
        float   min_p              = 0.0f,
        float   frequency_penalty  = 0.0f,
        float   presence_penalty   = 0.0f,
        float   dry_multiplier     = 0.0f,
        float   dry_base           = 1.75f,
        int32_t dry_allowed_length = 2,
        int32_t dry_penalty_last_n = -1,
        float   dyntemp_range      = 0.0f,
        float   dyntemp_exponent   = 1.0f) {
        ext_min_p              = min_p;
        ext_frequency_penalty  = frequency_penalty;
        ext_presence_penalty   = presence_penalty;
        ext_dry_multiplier     = dry_multiplier;
        ext_dry_base           = dry_base;
        ext_dry_allowed_length = dry_allowed_length;
        ext_dry_penalty_last_n = dry_penalty_last_n;
        ext_dyntemp_range      = dyntemp_range;
        ext_dyntemp_exponent   = dyntemp_exponent;
    }
    // CB0 logits are ready, before sampling.  Return non-zero to stop early.
    // Pass nullptr to clear.
    // Callback signature: (frame_idx, cb0_logits, cb0_logits_size, cb0_token) → int
    using logits_cb_t = std::function<int(int32_t, const float *, int32_t, int32_t)>;
    void set_logits_callback(logits_cb_t cb) { logits_cb_ = std::move(cb); }
    void clear_logits_callback() { logits_cb_ = nullptr; }

    // Abort callback — cancels the current graph compute mid-flight.
    // callback returns true → abort. Pass nullptr to clear.
    void set_abort_callback(ggml_abort_callback cb, void * userdata) {
        state_.abort_cb      = cb;
        state_.abort_cb_data = userdata;
        // Wire into the CPU backend using the vendored ggml API
        ggml_backend_t cpu_backend = state_.backend_cpu ? state_.backend_cpu : nullptr;
        if (!cpu_backend && state_.backend) {
            ggml_backend_dev_t dev = ggml_backend_get_device(state_.backend);
            if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU)
                cpu_backend = state_.backend;
        }
        if (cpu_backend) {
            ggml_backend_cpu_set_abort_callback(cpu_backend, cb, userdata);
        }
    }
    void clear_abort_callback() { set_abort_callback(nullptr, nullptr); }

    // Eval callback — fires for every graph node during compute.
    // When ask==true, return true to observe this node's output tensor.
    // When ask==false, the computed tensor is passed in; return false to cancel.
    void set_eval_callback(ggml_backend_sched_eval_callback cb, void * userdata) {
        state_.eval_cb      = cb;
        state_.eval_cb_data = userdata;
        if (state_.sched) {
            ggml_backend_sched_set_eval_callback(state_.sched, cb, userdata);
        }
    }
    void clear_eval_callback() { set_eval_callback(nullptr, nullptr); }

    const tts_transformer_config & get_config() const { return model_.config; }

    const std::string & get_error() const { return error_msg_; }

    // Prefill embedding builders — public so qwen3_tts.cpp can call them
    bool build_prefill_graph(const int32_t * text_tokens, int32_t n_tokens,
                             const float * speaker_embd, int32_t language_id,
                             std::vector<float> & prefill_embd,
                             std::vector<float> & trailing_text_hidden,
                             std::vector<float> & tts_pad_embed,
                             bool non_streaming_mode = false);

    bool build_prefill_graph_icl(const int32_t * text_tokens, int32_t n_tokens,
                                  const float * speaker_embd, int32_t language_id,
                                  const int32_t * ref_text_tokens, int32_t n_ref_text_tokens,
                                  const int32_t * ref_codes, int32_t n_ref_frames,
                                  std::vector<float> & prefill_embd,
                                  std::vector<float> & trailing_text_hidden,
                                  std::vector<float> & tts_pad_embed,
                                  bool non_streaming_mode = false);

    bool build_prefill_graph_instruct(const int32_t * text_tokens, int32_t n_tokens,
                                       const float * speaker_embd, int32_t language_id,
                                       const int32_t * instruct_tokens, int32_t n_instruct_tokens,
                                       std::vector<float> & prefill_embd,
                                       std::vector<float> & trailing_text_hidden,
                                       std::vector<float> & tts_pad_embed,
                                       bool non_streaming_mode = false);

private:
    bool try_init_coreml_code_predictor(const std::string & model_path);
    bool predict_codes_autoregressive_coreml(const float * hidden, int32_t codebook_0_token,
                                             std::vector<int32_t> & output,
                                             float temperature,
                                             int32_t top_k,
                                             float top_p);

    // Internal graph builders
    struct ggml_cgraph * build_prefill_forward_graph(int32_t n_tokens, int32_t n_past, int32_t batch_idx = 0);
    struct ggml_cgraph * build_step_graph(int32_t n_past, int32_t batch_idx = 0);

    bool project_text_tokens(const int32_t * text_tokens, int32_t n_tokens,
                             std::vector<float> & output);

    bool lookup_embedding_rows(struct ggml_tensor * embedding, const int32_t * token_ids,
                               int32_t n_tokens, const char * input_name,
                               const char * output_name, std::vector<float> & output);
    bool lookup_single_embedding_row(struct ggml_tensor * embedding, int32_t token_id,
                                     float * out_row);
    
    // Build computation graph for single-step autoregressive code predictor
    // n_past: number of tokens already in KV cache (0-14)
    // generation_step: which codebook we're predicting (0-14)
    struct ggml_cgraph * build_code_pred_step_graph(int32_t n_past, int32_t generation_step, int32_t batch_idx = 0);
    
    // Build computation graph for 2-token prefill of code predictor
    // Processes [past_hidden, codec_embd(codebook_0_token)] together
    struct ggml_cgraph * build_code_pred_prefill_graph(int32_t batch_idx = 0);
    
    // Grow talker KV cache while preserving positions [0, n_positions_to_copy).
    bool extend_kv_cache(int32_t new_ctx, int32_t n_positions_to_copy);

    // Generic KV cache extension — operates on any tts_kv_cache via a reinit callback.
    // Used by extend_kv_cache (talker) and can be used for code_pred_cache if needed.
    bool extend_kv_cache_impl(tts_kv_cache & cache, int32_t new_ctx,
                               int32_t n_positions_to_copy,
                               std::function<bool(int32_t, int32_t)> reinit_fn);

    // Parse hyperparameters from GGUF
    bool parse_config(struct gguf_context * ctx);
    
    // Create tensor structures
    bool create_tensors(struct gguf_context * ctx);
    
    // Load tensor data from file using an already-acquired backend handle.
    // Caller owns the backend lifetime; this function does NOT release it.
    bool load_tensor_data(const std::string & path, struct gguf_context * ctx,
                          ggml_backend_t backend);
    
    tts_transformer_model model_;
    tts_transformer_state state_;
    std::string error_msg_;
    
    // Cached hidden states from last forward pass
    std::vector<float> last_hidden_;
    std::vector<ggml_fp16_t> embd_row_fp16_scratch_;
    std::mt19937 rng_{std::random_device{}()};
    CoreMLCodePredictor coreml_code_predictor_;
    bool use_coreml_code_predictor_ = false;
    std::string coreml_code_predictor_path_;
    bool skip_ggml_code_pred_layers_ = false;

    // Optional per-frame logits callback (set via set_logits_callback)
    logits_cb_t logits_cb_;

    // Extended sampling parameters — set by qwen3_tts.cpp before calling generate().
    // These extend the basic temperature/top_k/top_p/rep_penalty params that are
    // passed directly to generate(). Default values disable all extensions.
    float   ext_min_p             = 0.0f;
    float   ext_frequency_penalty = 0.0f;
    float   ext_presence_penalty  = 0.0f;
    float   ext_dry_multiplier    = 0.0f;
    float   ext_dry_base          = 1.75f;
    int32_t ext_dry_allowed_length = 2;
    int32_t ext_dry_penalty_last_n = -1;
    float   ext_dyntemp_range     = 0.0f;
    float   ext_dyntemp_exponent  = 1.0f;

#ifdef QWEN3_TTS_TIMING
    tts_timing * timing_ = nullptr;
#endif
};

// Free model resources
void free_transformer_model(tts_transformer_model & model);

// Free KV cache resources
void free_tts_kv_cache(tts_kv_cache & cache);

} // namespace qwen3_tts
