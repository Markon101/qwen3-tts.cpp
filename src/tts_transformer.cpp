#include "tts_transformer.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <cstdlib>
#include <cctype>
#include <sys/stat.h>

namespace qwen3_tts {

// Validate raw graph outputs before sampler masks intentionally introduce -INF.
static bool validate_logits(const std::vector<float> & values, const char * stage,
                            std::string & error) {
    const char * enabled = getenv("QWEN3_TTS_VALIDATE_LOGITS");
    if (!enabled || enabled[0] != '1') return true;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            error = std::string("Non-finite raw logits in ") + stage +
                    " at index " + std::to_string(i);
            return false;
        }
    }
    return true;
}

// KV cache element type: F16 (default, half RAM) or F32 (bit-exact, 2× RAM).
// Controlled by -DQWEN3_TTS_KV_F32 compile flag.
#ifdef QWEN3_TTS_KV_F32
static constexpr ggml_type QWEN3_TTS_KV_TYPE = GGML_TYPE_F32;
#else
static constexpr ggml_type QWEN3_TTS_KV_TYPE = GGML_TYPE_F16;
#endif

TTSTransformer::TTSTransformer() = default;

TTSTransformer::~TTSTransformer() {
    unload_model();
}

void TTSTransformer::unload_model() {
    free_tts_kv_cache(state_.cache);
    free_tts_kv_cache(state_.code_pred_cache);
    free_transformer_model(model_);

    coreml_code_predictor_.unload();
    use_coreml_code_predictor_ = false;
    coreml_code_predictor_path_.clear();
    skip_ggml_code_pred_layers_ = false;

    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.threadpool) {
        ggml_threadpool_free(state_.threadpool);
        state_.threadpool     = nullptr;
        state_.threadpool_n   = 0;
    }
    if (state_.backend) {
        release_preferred_backend(state_.backend);
        state_.backend = nullptr;
    }
    if (state_.backend_cpu) {
        ggml_backend_free(state_.backend_cpu);
        state_.backend_cpu = nullptr;
    }

    state_.compute_meta.clear();
    last_hidden_.clear();
    embd_row_fp16_scratch_.clear();
}

// Parse QWEN3_TTS_USE_COREML env var: returns true if CoreML is explicitly disabled.
// "0", "false", "off", "no" (case-insensitive) → disabled.  Empty / unset → not disabled.
static bool coreml_env_disabled() {
    const char * val = std::getenv("QWEN3_TTS_USE_COREML");
    if (!val || val[0] == '\0') return false;
    std::string s = val;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return s == "0" || s == "false" || s == "off" || s == "no";
}

// Resolve the CoreML model path: QWEN3_TTS_COREML_MODEL env override, or
// <model_dir>/coreml/code_predictor.mlpackage next to the GGUF file.
// Only used on Apple platforms — wrapped to suppress unused-function warning on other builds.
#if defined(__APPLE__)
static std::string coreml_model_path(const std::string & model_path) {
    const char * override_env = std::getenv("QWEN3_TTS_COREML_MODEL");
    if (override_env && override_env[0] != '\0') return override_env;
    size_t slash = model_path.find_last_of("/\\");
    const std::string model_dir = (slash == std::string::npos) ? "." : model_path.substr(0, slash);
    return model_dir + "/coreml/code_predictor.mlpackage";
}
#endif

bool TTSTransformer::load_model(const std::string & model_path) {
    unload_model();

    skip_ggml_code_pred_layers_ = false;
#if defined(__APPLE__)
    if (!coreml_env_disabled()) {
        const std::string coreml_path = coreml_model_path(model_path);
        struct stat st = {};
        if (stat(coreml_path.c_str(), &st) == 0) {
            // CoreML package found next to the model — skip GGML code-predictor weights.
            skip_ggml_code_pred_layers_ = true;
        } else {
            // Explicit opt-in (env set but file not found yet) — keep strict mode
            // so the error surfaces at try_init_coreml_code_predictor time.
            const char * use_coreml_env = std::getenv("QWEN3_TTS_USE_COREML");
            if (use_coreml_env && use_coreml_env[0] != '\0') {
                skip_ggml_code_pred_layers_ = true;
            }
        }
    }
#endif

    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };
    
    struct gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        error_msg_ = "Failed to open GGUF file: " + model_path;
        return false;
    }
    
    if (!parse_config(ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    if (!create_tensors(ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    // Acquire the backend before loading tensor data so we only call
    // init_preferred_backend once and avoid a ref-count double-increment.
    state_.backend = init_preferred_backend("TTSTransformer", &error_msg_);
    if (!state_.backend) {
        free_transformer_model(model_);
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    if (!load_tensor_data(model_path, ctx, state_.backend)) {
        free_transformer_model(model_);
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    gguf_free(ctx);
    if (meta_ctx) ggml_free(meta_ctx);
    
    ggml_backend_dev_t device = ggml_backend_get_device(state_.backend);
    const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
    fprintf(stderr, "  TTSTransformer backend: %s\n", device_name);

    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        state_.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!state_.backend_cpu) {
            error_msg_ = "Failed to initialize CPU fallback backend for TTSTransformer";
            return false;
        }
    }
    
    std::vector<ggml_backend_t> backends;
    backends.push_back(state_.backend);
    if (state_.backend_cpu) {
        backends.push_back(state_.backend_cpu);
    }
    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), QWEN3_TTS_MAX_NODES, false, false);
    // op_offload=false: use lazy buffer allocation instead of pre-reservation.
    // Pre-reservation (true) scales O(n_kv²) for large contexts and causes OOM
    // allocation failures (>17 GB) when n_kv exceeds ~4096.
    if (!state_.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }

    // Wire eval callback if it was set before load (unlikely but safe)
    if (state_.eval_cb) {
        ggml_backend_sched_set_eval_callback(state_.sched, state_.eval_cb, state_.eval_cb_data);
    }

    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_TTS_MAX_NODES + ggml_graph_overhead());

    if (!try_init_coreml_code_predictor(model_path)) {
        return false;
    }

    // ---- Metal shader warmup ------------------------------------------------
    // On Apple Silicon, GGML compiles Metal shaders JIT as the KV cache grows —
    // kernel_mul_mv_f16_f32_nsg=N variants compile at n_kv thresholds ~32/64/128/256.
    // Running the step graph at increasing KV sizes forces all variants to compile
    // upfront so no shader compilation happens during real synthesis.
#if defined(__APPLE__)
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(state_.backend);
        if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            fprintf(stderr, "  Warming up Metal shader cache...\n");

            // KV sizes that trigger new nsg variant compilation
            const int warmup_kv_sizes[] = {1, 32, 64, 128, 256, 512};
            const int n_warmup = (int)(sizeof(warmup_kv_sizes) / sizeof(warmup_kv_sizes[0]));

            for (int wi = 0; wi < n_warmup; ++wi) {
                int kv = warmup_kv_sizes[wi];
                if (!init_kv_cache(kv + 1, 1) || !init_code_pred_kv_cache(kv + 2, 1)) break;

                // Prefill graph (only needed once — covers mul_mm, get_rows, soft_max)
                if (wi == 0) {
                    struct ggml_cgraph * gf = build_prefill_forward_graph(1, 0, 0);
                    if (gf && ggml_backend_sched_alloc_graph(state_.sched, gf)) {
                        struct ggml_tensor * t = ggml_graph_get_tensor(gf, "inp_prefill_embd");
                        struct ggml_tensor * p = ggml_graph_get_tensor(gf, "inp_pos");
                        if (t) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
                        if (p) ggml_backend_tensor_memset(p, 0, 0, ggml_nbytes(p));
                        audited_graph_compute("transformer", state_.sched, gf, &error_msg_);
                        ggml_backend_sched_reset(state_.sched);
                    }
                }

                // Step graph at this KV size — triggers nsg variant for this tier
                struct ggml_cgraph * gf = build_step_graph(kv, 0);
                if (gf && ggml_backend_sched_alloc_graph(state_.sched, gf)) {
                    struct ggml_tensor * t = ggml_graph_get_tensor(gf, "inp_step_embd");
                    struct ggml_tensor * p = ggml_graph_get_tensor(gf, "inp_pos");
                    if (t) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
                    if (p) ggml_backend_tensor_memset(p, 0, 0, ggml_nbytes(p));
                    audited_graph_compute("transformer", state_.sched, gf, &error_msg_);
                    ggml_backend_sched_reset(state_.sched);
                }

                free_tts_kv_cache(state_.cache);
                free_tts_kv_cache(state_.code_pred_cache);
            }
            fprintf(stderr, "  Metal shader cache warmed up.\n");
        }
    }
#endif

    return true;
}

bool TTSTransformer::try_init_coreml_code_predictor(const std::string & model_path) {
    use_coreml_code_predictor_ = false;
    coreml_code_predictor_path_.clear();

    if (coreml_env_disabled()) {
        return true;
    }

#if !defined(__APPLE__)
    {
        (void)model_path;  // only used in the Apple branch below
        const char * use_coreml_env = std::getenv("QWEN3_TTS_USE_COREML");
        if (use_coreml_env && use_coreml_env[0] != '\0') {
            fprintf(stderr, "  CoreML code predictor requested but this build is not on Apple platform\n");
        }
    }
    return true;
#else
    const std::string coreml_path = coreml_model_path(model_path);

    if (!coreml_code_predictor_.load(coreml_path, model_.config.n_codebooks - 1)) {
        if (skip_ggml_code_pred_layers_) {
            error_msg_ = "CoreML code predictor load failed in strict mode: " + coreml_code_predictor_.get_error();
            return false;
        } else {
            fprintf(stderr, "  CoreML code predictor load failed: %s\n",
                    coreml_code_predictor_.get_error().c_str());
            fprintf(stderr, "  Falling back to GGML code predictor\n");
            return true;
        }
    }

    use_coreml_code_predictor_ = true;
    coreml_code_predictor_path_ = coreml_path;
    fprintf(stderr, "  CoreML code predictor enabled: %s\n", coreml_code_predictor_path_.c_str());
    return true;
#endif
}

bool TTSTransformer::parse_config(struct gguf_context * ctx) {
    auto get_u32_any = [&](std::initializer_list<const char *> keys, int32_t default_val) -> int32_t {
        for (const char * key : keys) {
            int64_t idx = gguf_find_key(ctx, key);
            if (idx >= 0) {
                return (int32_t)gguf_get_val_u32(ctx, idx);
            }
        }
        return default_val;
    };
    
    auto get_f32_any = [&](std::initializer_list<const char *> keys, float default_val) -> float {
        for (const char * key : keys) {
            int64_t idx = gguf_find_key(ctx, key);
            if (idx >= 0) {
                return gguf_get_val_f32(ctx, idx);
            }
        }
        return default_val;
    };
    
    auto & cfg = model_.config;
    cfg.text_vocab_size = get_u32_any({
        "qwen3-tts.text.vocab_size",
        "qwen3-tts.text_vocab_size",
    }, 151936);
    cfg.text_embd_dim = get_u32_any({
        "qwen3-tts.text.embedding_dim",
        "qwen3-tts.text_hidden_size",
    }, 2048);
    cfg.hidden_size = get_u32_any({
        "qwen3-tts.talker.embedding_length",
        "qwen3-tts.embedding_length",
    }, 1024);
    cfg.n_layers = get_u32_any({
        "qwen3-tts.talker.block_count",
        "qwen3-tts.block_count",
    }, 28);
    cfg.n_attention_heads = get_u32_any({
        "qwen3-tts.talker.attention.head_count",
        "qwen3-tts.attention.head_count",
    }, 16);
    cfg.n_key_value_heads = get_u32_any({
        "qwen3-tts.talker.attention.head_count_kv",
        "qwen3-tts.attention.head_count_kv",
    }, 8);
    cfg.intermediate_size = get_u32_any({
        "qwen3-tts.talker.feed_forward_length",
        "qwen3-tts.feed_forward_length",
    }, 3072);
    cfg.head_dim = get_u32_any({
        "qwen3-tts.talker.attention.key_length",
        "qwen3-tts.attention.key_length",
    }, 128);
    cfg.rms_norm_eps = get_f32_any({
        "qwen3-tts.talker.attention.layer_norm_rms_epsilon",
        "qwen3-tts.attention.layer_norm_rms_epsilon",
    }, 1e-6f);
    cfg.rope_theta = get_f32_any({
        "qwen3-tts.talker.rope.freq_base",
        "qwen3-tts.rope.freq_base",
    }, 1000000.0f);

    cfg.codec_vocab_size = get_u32_any({
        "qwen3-tts.talker.codec_vocab_size",
        "qwen3-tts.vocab_size",
    }, 3072);
    cfg.n_codebooks = get_u32_any({
        "qwen3-tts.talker.num_codebooks",
        "qwen3-tts.num_code_groups",
    }, 16);

    cfg.code_pred_layers = get_u32_any({
        "qwen3-tts.code_pred.layer_count",
        "qwen3-tts.code_predictor.layer_count",
    }, 5);
    cfg.code_pred_vocab_size = get_u32_any({
        "qwen3-tts.code_pred.vocab_size",
        "qwen3-tts.code_predictor.vocab_size",
    }, 2048);
    cfg.code_pred_hidden_size = get_u32_any({
        "qwen3-tts.code_pred.embedding_length",
        "qwen3-tts.code_predictor.embedding_length",
    }, 0);
    cfg.code_pred_intermediate_size = get_u32_any({
        "qwen3-tts.code_pred.feed_forward_length",
        "qwen3-tts.code_predictor.feed_forward_length",
    }, 0);
    // Fall back to talker values if not specified (0.6B model)
    if (cfg.code_pred_hidden_size == 0) cfg.code_pred_hidden_size = cfg.hidden_size;
    if (cfg.code_pred_intermediate_size == 0) cfg.code_pred_intermediate_size = cfg.intermediate_size;

    cfg.codec_pad_id = get_u32_any({
        "qwen3-tts.codec.pad_id",
    }, 2148);
    cfg.codec_bos_id = get_u32_any({
        "qwen3-tts.codec.bos_id",
    }, 2149);
    cfg.codec_eos_id = get_u32_any({
        "qwen3-tts.codec.eos_id",
        "qwen3-tts.codec.eos_token_id",
    }, 2150);

    cfg.tts_bos_token_id = get_u32_any({
        "qwen3-tts.tts_bos_token_id",
        "qwen3-tts.tts.bos_token_id",
        "qwen3-tts.tts.bos_id",
    }, 151672);
    cfg.tts_eos_token_id = get_u32_any({
        "qwen3-tts.tts_eos_token_id",
        "qwen3-tts.tts.eos_token_id",
        "qwen3-tts.tts.eos_id",
    }, 151673);
    cfg.tts_pad_token_id = get_u32_any({
        "qwen3-tts.tts_pad_token_id",
        "qwen3-tts.tts.pad_token_id",
        "qwen3-tts.tts.pad_id",
    }, 151671);

    cfg.codec_think_id = get_u32_any({
        "qwen3-tts.codec.think_id",
        "qwen3-tts.codec_think_id",
    }, 2154);
    cfg.codec_nothink_id = get_u32_any({
        "qwen3-tts.codec.nothink_id",
        "qwen3-tts.codec_nothink_id",
    }, 2155);
    cfg.codec_think_bos_id = get_u32_any({
        "qwen3-tts.codec.think_bos_id",
        "qwen3-tts.codec_think_bos_id",
    }, 2156);
    cfg.codec_think_eos_id = get_u32_any({
        "qwen3-tts.codec.think_eos_id",
        "qwen3-tts.codec_think_eos_id",
    }, 2157);

    cfg.english_language_id = get_u32_any({
        "qwen3-tts.language.english_id",
        "qwen3-tts.codec.language.english_id",
        "qwen3-tts.language_id",
    }, 2050);

    // Model type and size (optional — written by updated converter)
    {
        int64_t idx = gguf_find_key(ctx, "qwen3-tts.model_type");
        if (idx >= 0) {
            cfg.model_type = gguf_get_val_str(ctx, idx);
        }
        idx = gguf_find_key(ctx, "qwen3-tts.model_size");
        if (idx >= 0) {
            cfg.model_size = gguf_get_val_str(ctx, idx);
        }
    }

    // Helper: read an integer from a GGUF array at index i.
    // Handles UINT32, INT32, UINT64, INT64 element types robustly.
    auto read_arr_int = [&](int64_t arr_idx, size_t i) -> int32_t {
        const void * data = gguf_get_arr_data(ctx, arr_idx);
        if (!data) return -1;
        enum gguf_type elem_type = gguf_get_arr_type(ctx, arr_idx);
        switch (elem_type) {
            case GGUF_TYPE_INT32:  return ((const int32_t  *)data)[i];
            case GGUF_TYPE_UINT32: return (int32_t)(((const uint32_t *)data)[i]);
            case GGUF_TYPE_INT64:  return (int32_t)(((const int64_t  *)data)[i]);
            case GGUF_TYPE_UINT64: return (int32_t)(((const uint64_t *)data)[i]);
            case GGUF_TYPE_INT16:  return (int32_t)(((const int16_t  *)data)[i]);
            case GGUF_TYPE_UINT16: return (int32_t)(((const uint16_t *)data)[i]);
            case GGUF_TYPE_INT8:   return (int32_t)(((const int8_t   *)data)[i]);
            case GGUF_TYPE_UINT8:  return (int32_t)(((const uint8_t  *)data)[i]);
            default: return -1;
        }
    };

    // Speaker ID table: "qwen3-tts.speakers.names" (string array)
    //                   "qwen3-tts.speakers.ids"   (int array)
    {
        int64_t names_idx = gguf_find_key(ctx, "qwen3-tts.speakers.names");
        int64_t ids_idx   = gguf_find_key(ctx, "qwen3-tts.speakers.ids");
        if (names_idx >= 0 && ids_idx >= 0) {
            size_t n = gguf_get_arr_n(ctx, names_idx);
            for (size_t i = 0; i < n; ++i) {
                const char * name = gguf_get_arr_str(ctx, names_idx, i);
                int32_t id = read_arr_int(ids_idx, i);
                if (name && id >= 0) {
                    std::string lower = name;
                    for (char & c : lower) c = (char)tolower((unsigned char)c);
                    cfg.spk_id[lower] = id;
                }
            }
        }
    }

    // Language table: "qwen3-tts.languages.names" (string array)
    //                 "qwen3-tts.languages.ids"   (int array)
    {
        int64_t names_idx = gguf_find_key(ctx, "qwen3-tts.languages.names");
        int64_t ids_idx   = gguf_find_key(ctx, "qwen3-tts.languages.ids");
        if (names_idx >= 0 && ids_idx >= 0) {
            size_t n = gguf_get_arr_n(ctx, names_idx);
            for (size_t i = 0; i < n; ++i) {
                const char * name = gguf_get_arr_str(ctx, names_idx, i);
                int32_t id = read_arr_int(ids_idx, i);
                if (name && id >= 0) {
                    std::string lower = name;
                    for (char & c : lower) c = (char)tolower((unsigned char)c);
                    cfg.codec_language_id[lower] = id;
                }
            }
        }
        // "auto" means no language token (uses codec_nothink_id path, language_id=-1)
        // Do NOT add it to the table so resolve_language_id("auto") returns -1 via its own check
    }

    // spk_is_dialect table: "qwen3-tts.speakers.dialect_names" (string array of speaker names)
    //                        "qwen3-tts.speakers.dialect_langs" (string array of language names)
    // e.g. speaker "cantonese_voice" -> dialect "cantonese"
    {
        int64_t spk_idx  = gguf_find_key(ctx, "qwen3-tts.speakers.dialect_names");
        int64_t lang_idx = gguf_find_key(ctx, "qwen3-tts.speakers.dialect_langs");
        if (spk_idx >= 0 && lang_idx >= 0) {
            size_t n = gguf_get_arr_n(ctx, spk_idx);
            for (size_t i = 0; i < n; ++i) {
                const char * spk_name  = gguf_get_arr_str(ctx, spk_idx,  i);
                const char * lang_name = gguf_get_arr_str(ctx, lang_idx, i);
                if (spk_name && lang_name) {
                    std::string spk_lower  = spk_name;
                    std::string lang_lower = lang_name;
                    for (char & c : spk_lower)  c = (char)tolower((unsigned char)c);
                    for (char & c : lang_lower) c = (char)tolower((unsigned char)c);
                    cfg.spk_is_dialect[spk_lower] = lang_lower;
                }
            }
        }
    }

    return true;
}

bool TTSTransformer::create_tensors(struct gguf_context * ctx) {
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    const auto & cfg = model_.config;
    
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_msg_ = "Failed to create GGML context";
        return false;
    }
    
    model_.layers.resize(cfg.n_layers);
    model_.code_pred_layers.resize(cfg.code_pred_layers);
    model_.code_pred_embd.resize(cfg.n_codebooks - 1);
    model_.code_pred_head.resize(cfg.n_codebooks - 1);
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);
        
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        int n_dims = 0;
        
        if (strstr(name, "spk_enc.") || strstr(name, "tok_")) {
            continue;
        }
        
        if (strstr(name, "talker.text_embd.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.text_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc1.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.text_embd_dim;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc1.bias")) {
            ne[0] = cfg.text_embd_dim;
            n_dims = 1;
        } else if (strstr(name, "talker.text_proj.fc2.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.hidden_size;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc2.bias")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "talker.codec_embd.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.codec_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.codec_head.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.codec_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.output_norm.weight")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "talker.blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "talker.blk.%d.", &layer_idx) == 1 && 
                layer_idx >= 0 && layer_idx < cfg.n_layers) {
                
                if (strstr(name, "attn_norm.weight")) {
                    ne[0] = cfg.hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "attn_q_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_k_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_q.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_attention_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_k.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_v.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_output.weight")) {
                    ne[0] = cfg.n_attention_heads * cfg.head_dim;
                    ne[1] = cfg.hidden_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_norm.weight")) {
                    ne[0] = cfg.hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "ffn_gate.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_up.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_down.weight")) {
                    ne[0] = cfg.intermediate_size;
                    ne[1] = cfg.hidden_size;
                    n_dims = 2;
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.blk.")) {
            if (skip_ggml_code_pred_layers_) {
                continue;
            }
            int layer_idx = -1;
            if (sscanf(name, "code_pred.blk.%d.", &layer_idx) == 1 && 
                layer_idx >= 0 && layer_idx < cfg.code_pred_layers) {
                
                const int cp_hs = cfg.code_pred_hidden_size;
                const int cp_ff = cfg.code_pred_intermediate_size;
                
                if (strstr(name, "attn_norm.weight")) {
                    ne[0] = cp_hs;
                    n_dims = 1;
                } else if (strstr(name, "attn_q_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_k_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_q.weight")) {
                    ne[0] = cp_hs;
                    ne[1] = cfg.n_attention_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_k.weight")) {
                    ne[0] = cp_hs;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_v.weight")) {
                    ne[0] = cp_hs;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_output.weight")) {
                    ne[0] = cfg.n_attention_heads * cfg.head_dim;
                    ne[1] = cp_hs;
                    n_dims = 2;
                } else if (strstr(name, "ffn_norm.weight")) {
                    ne[0] = cp_hs;
                    n_dims = 1;
                } else if (strstr(name, "ffn_gate.weight")) {
                    ne[0] = cp_hs;
                    ne[1] = cp_ff;
                    n_dims = 2;
                } else if (strstr(name, "ffn_up.weight")) {
                    ne[0] = cp_hs;
                    ne[1] = cp_ff;
                    n_dims = 2;
                } else if (strstr(name, "ffn_down.weight")) {
                    ne[0] = cp_ff;
                    ne[1] = cp_hs;
                    n_dims = 2;
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.codec_embd.")) {
            int cb_idx = -1;
            if (sscanf(name, "code_pred.codec_embd.%d.weight", &cb_idx) == 1 &&
                cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                // codec_embeddings are stored in talker space (2048-dim for 1.7B)
                // and projected to code_pred_hidden_size by code_pred_proj
                ne[0] = cfg.hidden_size;
                ne[1] = cfg.code_pred_vocab_size;
                n_dims = 2;
            } else {
                continue;
            }
         } else if (strstr(name, "code_pred.lm_head.")) {
              if (skip_ggml_code_pred_layers_) {
                  continue;
              }
              int cb_idx = -1;
              if (sscanf(name, "code_pred.lm_head.%d.weight", &cb_idx) == 1 &&
                  cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                  ne[0] = cfg.code_pred_hidden_size;
                  ne[1] = cfg.code_pred_vocab_size;
                  n_dims = 2;
              } else {
                  continue;
              }
          } else if (strstr(name, "code_pred.output_norm.weight")) {
              if (skip_ggml_code_pred_layers_) {
                  continue;
              }
              ne[0] = cfg.code_pred_hidden_size;
              n_dims = 1;
          } else if (strstr(name, "code_pred.proj.weight")) {
              if (skip_ggml_code_pred_layers_) {
                  continue;
              }
              // code_pred_proj: [code_pred_hidden_size, hidden_size]
              ne[0] = cfg.hidden_size;
              ne[1] = cfg.code_pred_hidden_size;
              n_dims = 2;
          } else if (strstr(name, "code_pred.proj.bias")) {
              if (skip_ggml_code_pred_layers_) {
                  continue;
              }
              ne[0] = cfg.code_pred_hidden_size;
              n_dims = 1;
          } else {
              continue;
          }
        
        struct ggml_tensor * tensor = ggml_new_tensor(model_.ctx, type, n_dims, ne);
        if (!tensor) {
            error_msg_ = "Failed to create tensor: " + std::string(name);
            return false;
        }
        ggml_set_name(tensor, name);
        model_.tensors[name] = tensor;
        
        if (strstr(name, "talker.text_embd.weight")) {
            model_.text_embd = tensor;
        } else if (strstr(name, "talker.text_proj.fc1.weight")) {
            model_.text_proj_fc1 = tensor;
        } else if (strstr(name, "talker.text_proj.fc1.bias")) {
            model_.text_proj_fc1_bias = tensor;
        } else if (strstr(name, "talker.text_proj.fc2.weight")) {
            model_.text_proj_fc2 = tensor;
        } else if (strstr(name, "talker.text_proj.fc2.bias")) {
            model_.text_proj_fc2_bias = tensor;
        } else if (strstr(name, "talker.codec_embd.weight")) {
            model_.codec_embd = tensor;
        } else if (strstr(name, "talker.codec_head.weight")) {
            model_.codec_head = tensor;
        } else if (strstr(name, "talker.output_norm.weight")) {
            model_.output_norm = tensor;
        } else if (strstr(name, "talker.blk.")) {
            int layer_idx = -1;
            sscanf(name, "talker.blk.%d.", &layer_idx);
            if (layer_idx >= 0 && layer_idx < cfg.n_layers) {
                auto & layer = model_.layers[layer_idx];
                if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        } else if (strstr(name, "code_pred.blk.")) {
            int layer_idx = -1;
            sscanf(name, "code_pred.blk.%d.", &layer_idx);
            if (layer_idx >= 0 && layer_idx < cfg.code_pred_layers) {
                auto & layer = model_.code_pred_layers[layer_idx];
                if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        } else if (strstr(name, "code_pred.codec_embd.")) {
            int cb_idx = -1;
            sscanf(name, "code_pred.codec_embd.%d.weight", &cb_idx);
            if (cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                model_.code_pred_embd[cb_idx] = tensor;
            }
         } else if (strstr(name, "code_pred.lm_head.")) {
             int cb_idx = -1;
             sscanf(name, "code_pred.lm_head.%d.weight", &cb_idx);
             if (cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                 model_.code_pred_head[cb_idx] = tensor;
             }
         } else if (strstr(name, "code_pred.output_norm.weight")) {
              model_.code_pred_output_norm = tensor;
          } else if (strstr(name, "code_pred.proj.weight")) {
              model_.code_pred_proj = tensor;
          } else if (strstr(name, "code_pred.proj.bias")) {
              model_.code_pred_proj_bias = tensor;
          }
      }
     
     return true;
 }

bool TTSTransformer::load_tensor_data(const std::string & path, struct gguf_context * ctx,
                                       ggml_backend_t backend) {
    // backend is owned by the caller (load_model via state_.backend) — do NOT release it here.
    model_.buffer = ggml_backend_alloc_ctx_tensors(model_.ctx, backend);
    if (!model_.buffer) {
        error_msg_ = "Failed to allocate tensor buffer";
        return false;
    }
    
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        error_msg_ = "Failed to open file for reading: " + path;
        return false;
    }
    
    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::vector<uint8_t> read_buf;
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);
        
        auto it = model_.tensors.find(name);
        if (it == model_.tensors.end()) {
            continue;
        }
        
        struct ggml_tensor * tensor = it->second;
        size_t nbytes = ggml_nbytes(tensor);
        
        read_buf.resize(nbytes);
        
        if (
#ifdef _WIN32
            _fseeki64(f, (int64_t)(data_offset + offset), SEEK_SET) != 0
#else
            fseeko(f, (off_t)(data_offset + offset), SEEK_SET) != 0
#endif
        ) {
            error_msg_ = "Failed to seek to tensor data: " + std::string(name);
            fclose(f);
            return false;
        }
        
        if (fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            error_msg_ = "Failed to read tensor data: " + std::string(name);
            fclose(f);
            return false;
        }
        
        ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);

        if (tensor == model_.codec_embd || strstr(name, "code_pred.codec_embd.") != nullptr) {
            model_.host_embd_raw[tensor] = read_buf;
        }
    }
    
    fclose(f);
    
    return true;
}

bool TTSTransformer::extend_kv_cache(int32_t new_ctx, int32_t n_positions_to_copy) {
    // Delegates to the generic implementation operating on the talker KV cache.
    return extend_kv_cache_impl(
        state_.cache, new_ctx, n_positions_to_copy,
        [this](int32_t ctx, int32_t batch) { return init_kv_cache(ctx, batch); });
}

bool TTSTransformer::extend_kv_cache_impl(
        tts_kv_cache & cache, int32_t new_ctx, int32_t n_positions_to_copy,
        std::function<bool(int32_t, int32_t)> reinit_fn) {
    if (new_ctx <= cache.n_ctx) {
        return true;
    }
    if (!cache.ctx || cache.k_cache.empty() || !cache.k_cache[0]) {
        return reinit_fn(new_ctx, cache.n_batch > 0 ? cache.n_batch : 1);
    }

    const int32_t old_ctx   = cache.n_ctx;
    const int32_t n_batch   = cache.n_batch > 0 ? cache.n_batch : 1;
    const int32_t n_layers  = cache.n_layers;
    const int32_t n_copy    = std::min(n_positions_to_copy, old_ctx);
    const size_t  nb_pos    = cache.k_cache[0]->nb[2];

    if (n_copy > 0 && nb_pos > 0) {
        std::vector<std::vector<uint8_t>> k_saved((size_t)n_layers);
        std::vector<std::vector<uint8_t>> v_saved((size_t)n_layers);
        const size_t bytes_per_batch = (size_t)n_copy * nb_pos;

        for (int il = 0; il < n_layers; ++il) {
            k_saved[il].resize(bytes_per_batch * (size_t)n_batch);
            v_saved[il].resize(bytes_per_batch * (size_t)n_batch);
            for (int b = 0; b < n_batch; ++b) {
                const size_t batch_off = (n_batch > 1)
                    ? (size_t)b * cache.k_cache[il]->nb[3] : 0;
                const size_t dst_off   = (size_t)b * bytes_per_batch;
                ggml_backend_tensor_get(cache.k_cache[il],
                                        k_saved[il].data() + dst_off,
                                        batch_off, bytes_per_batch);
                ggml_backend_tensor_get(cache.v_cache[il],
                                        v_saved[il].data() + dst_off,
                                        batch_off, bytes_per_batch);
            }
        }

        const int32_t n_used_saved = cache.n_used;
        if (!reinit_fn(new_ctx, n_batch)) {
            return false;
        }

        // After reinit, cache now refers to the newly-allocated cache — reuse
        // the same reference since reinit_fn wrote into the same struct.
        for (int il = 0; il < n_layers; ++il) {
            for (int b = 0; b < n_batch; ++b) {
                const size_t batch_off = (n_batch > 1)
                    ? (size_t)b * cache.k_cache[il]->nb[3] : 0;
                const size_t src_off   = (size_t)b * bytes_per_batch;
                ggml_backend_tensor_set(cache.k_cache[il],
                                        k_saved[il].data() + src_off,
                                        batch_off, bytes_per_batch);
                ggml_backend_tensor_set(cache.v_cache[il],
                                        v_saved[il].data() + src_off,
                                        batch_off, bytes_per_batch);
            }
        }
        cache.n_used = std::max(n_used_saved, n_copy);
        fprintf(stderr, "  [info] KV cache extended %d -> %d (preserved %d positions)\n",
                old_ctx, new_ctx, n_copy);
        return true;
    }

    if (!reinit_fn(new_ctx, n_batch)) {
        return false;
    }
    fprintf(stderr, "  [info] KV cache extended %d -> %d (empty cache)\n", old_ctx, new_ctx);
    return true;
}

bool TTSTransformer::init_kv_cache(int32_t n_ctx, int32_t n_batch) {
    const auto & cfg = model_.config;
    
    free_tts_kv_cache(state_.cache);
    
    state_.cache.n_ctx = n_ctx;
    state_.cache.n_used = 0;
    state_.cache.head_dim = cfg.head_dim;
    state_.cache.n_kv_heads = cfg.n_key_value_heads;
    state_.cache.n_layers = cfg.n_layers;
    state_.cache.n_batch = n_batch;
    
    const size_t n_tensors = cfg.n_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    state_.cache.ctx = ggml_init(params);
    if (!state_.cache.ctx) {
        error_msg_ = "Failed to create KV cache context";
        return false;
    }
    
    state_.cache.k_cache.resize(cfg.n_layers);
    state_.cache.v_cache.resize(cfg.n_layers);
    
    for (int il = 0; il < cfg.n_layers; ++il) {
        if (n_batch <= 1) {
            // 3D cache for single batch (backward compatible)
            state_.cache.k_cache[il] = ggml_new_tensor_3d(
                state_.cache.ctx, QWEN3_TTS_KV_TYPE,
                cfg.head_dim, cfg.n_key_value_heads, n_ctx);
            ggml_format_name(state_.cache.k_cache[il], "k_cache_%d", il);
            
            state_.cache.v_cache[il] = ggml_new_tensor_3d(
                state_.cache.ctx, QWEN3_TTS_KV_TYPE,
                cfg.head_dim, cfg.n_key_value_heads, n_ctx);
            ggml_format_name(state_.cache.v_cache[il], "v_cache_%d", il);
        } else {
            // 4D cache for batch inference [head_dim, n_kv_heads, n_ctx, n_batch]
            const int64_t ne[4] = {cfg.head_dim, cfg.n_key_value_heads, n_ctx, n_batch};
            state_.cache.k_cache[il] = ggml_new_tensor(state_.cache.ctx, QWEN3_TTS_KV_TYPE, 4, ne);
            ggml_format_name(state_.cache.k_cache[il], "k_cache_%d", il);
            
            state_.cache.v_cache[il] = ggml_new_tensor(state_.cache.ctx, QWEN3_TTS_KV_TYPE, 4, ne);
            ggml_format_name(state_.cache.v_cache[il], "v_cache_%d", il);
        }
    }
    
    state_.cache.buffer = ggml_backend_alloc_ctx_tensors(state_.cache.ctx, state_.backend);
    if (!state_.cache.buffer) {
        error_msg_ = "Failed to allocate KV cache buffer";
        return false;
    }
    // Zero the KV cache: Metal kernels may read past the logical KV length
    // (padded flash-attention variants), and recycled buffer memory can hold
    // garbage that poisons attention with NaN. Deterministic zeros are safe.
    ggml_backend_buffer_clear(state_.cache.buffer, 0);
    
    return true;
}

void TTSTransformer::clear_kv_cache() {
    state_.cache.n_used = 0;
    // Scrub stale K/V (including any NaN from a previously poisoned session):
    // padded flash-attention reads can touch beyond the logical length, so
    // the buffer must hold deterministic zeros at every generation start.
    if (state_.cache.buffer) {
        ggml_backend_buffer_clear(state_.cache.buffer, 0);
    }
}

void TTSTransformer::set_n_threads(int32_t n_threads) {
    if (n_threads <= 0) return;

    // Determine which CPU backend handle to configure
    ggml_backend_t cpu_backend = state_.backend_cpu ? state_.backend_cpu : nullptr;
    if (!cpu_backend && state_.backend) {
        ggml_backend_dev_t dev = ggml_backend_get_device(state_.backend);
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu_backend = state_.backend;
        }
    }

    if (cpu_backend) {
        ggml_backend_cpu_set_n_threads(cpu_backend, n_threads);

        // Create or recreate the persistent threadpool when thread count changes.
        // This eliminates the OS thread create/destroy overhead for every
        // graph compute — critical for TTS which does hundreds of small computes.
        if (state_.threadpool_n != n_threads) {
            if (state_.threadpool) {
                ggml_threadpool_free(state_.threadpool);
                state_.threadpool = nullptr;
            }
            struct ggml_threadpool_params tp_params = ggml_threadpool_params_default(n_threads);
            state_.threadpool = ggml_threadpool_new(&tp_params);
            if (state_.threadpool) {
                ggml_backend_cpu_set_threadpool(cpu_backend, state_.threadpool);
                state_.threadpool_n = n_threads;
            }
        }
    }
}

bool TTSTransformer::init_code_pred_kv_cache(int32_t n_ctx, int32_t n_batch) {
    const auto & cfg = model_.config;
    
    free_tts_kv_cache(state_.code_pred_cache);
    
    state_.code_pred_cache.n_ctx = n_ctx;
    state_.code_pred_cache.n_used = 0;
    state_.code_pred_cache.head_dim = cfg.head_dim;
    state_.code_pred_cache.n_kv_heads = cfg.n_key_value_heads;
    state_.code_pred_cache.n_layers = cfg.code_pred_layers;
    state_.code_pred_cache.n_batch = n_batch;
    
    const size_t n_tensors = cfg.code_pred_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    state_.code_pred_cache.ctx = ggml_init(params);
    if (!state_.code_pred_cache.ctx) {
        error_msg_ = "Failed to create code predictor KV cache context";
        return false;
    }
    
    state_.code_pred_cache.k_cache.resize(cfg.code_pred_layers);
    state_.code_pred_cache.v_cache.resize(cfg.code_pred_layers);
    
    for (int il = 0; il < cfg.code_pred_layers; ++il) {
        if (n_batch <= 1) {
            state_.code_pred_cache.k_cache[il] = ggml_new_tensor_3d(
                state_.code_pred_cache.ctx, QWEN3_TTS_KV_TYPE,
                cfg.head_dim, cfg.n_key_value_heads, n_ctx);
            ggml_format_name(state_.code_pred_cache.k_cache[il], "code_pred_k_cache_%d", il);
            
            state_.code_pred_cache.v_cache[il] = ggml_new_tensor_3d(
                state_.code_pred_cache.ctx, QWEN3_TTS_KV_TYPE,
                cfg.head_dim, cfg.n_key_value_heads, n_ctx);
            ggml_format_name(state_.code_pred_cache.v_cache[il], "code_pred_v_cache_%d", il);
        } else {
            const int64_t ne[4] = {cfg.head_dim, cfg.n_key_value_heads, n_ctx, n_batch};
            state_.code_pred_cache.k_cache[il] = ggml_new_tensor(
                state_.code_pred_cache.ctx, QWEN3_TTS_KV_TYPE, 4, ne);
            ggml_format_name(state_.code_pred_cache.k_cache[il], "code_pred_k_cache_%d", il);
            
            state_.code_pred_cache.v_cache[il] = ggml_new_tensor(
                state_.code_pred_cache.ctx, QWEN3_TTS_KV_TYPE, 4, ne);
            ggml_format_name(state_.code_pred_cache.v_cache[il], "code_pred_v_cache_%d", il);
        }
    }
    
    state_.code_pred_cache.buffer = ggml_backend_alloc_ctx_tensors(state_.code_pred_cache.ctx, state_.backend);
    if (!state_.code_pred_cache.buffer) {
        error_msg_ = "Failed to allocate code predictor KV cache buffer";
        return false;
    }
    // Same reasoning as the talker KV cache: deterministic zeros instead of
    // recycled buffer garbage that padded kernels may read.
    ggml_backend_buffer_clear(state_.code_pred_cache.buffer, 0);
    
    return true;
}

void TTSTransformer::clear_code_pred_kv_cache() {
    state_.code_pred_cache.n_used = 0;
}

bool TTSTransformer::lookup_embedding_rows(struct ggml_tensor * embedding, const int32_t * token_ids,
                                           int32_t n_tokens, const char * input_name,
                                           const char * output_name, std::vector<float> & output) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!embedding) {
        error_msg_ = "Embedding tensor not found";
        return false;
    }
    if (n_tokens <= 0) {
        output.clear();
        return true;
    }

    const int32_t embd_dim = (int32_t) embedding->ne[0];
    if (model_.host_embd_raw.find(embedding) != model_.host_embd_raw.end() ||
        (n_tokens <= 32 &&
         (embedding->type == GGML_TYPE_F16 || embedding->type == GGML_TYPE_F32))) {
        output.resize((size_t) embd_dim * n_tokens);
        for (int32_t t = 0; t < n_tokens; ++t) {
            if (!lookup_single_embedding_row(embedding, token_ids[t],
                                             output.data() + (size_t) t * embd_dim)) {
                return false;
            }
        }
        return true;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, input_name);
    ggml_set_input(inp_tokens);

    struct ggml_tensor * rows = ggml_get_rows(ctx0, embedding, inp_tokens);
    rows = ggml_cast(ctx0, rows, GGML_TYPE_F32);
    ggml_set_name(rows, output_name);
    ggml_set_output(rows);

    ggml_build_forward_expand(gf, rows);

    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate embedding lookup graph";
        ggml_free(ctx0);
        return false;
    }

    struct ggml_tensor * inp = ggml_graph_get_tensor(gf, input_name);
    ggml_backend_tensor_set(inp, token_ids, 0, n_tokens * sizeof(int32_t));

    if (audited_graph_compute("embedding_lookup", state_.sched, gf, &error_msg_) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute embedding lookup graph";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }
    // Synchronous scheduler compute already synchronizes all backends.

    struct ggml_tensor * out = ggml_graph_get_tensor(gf, output_name);
    if (!out) {
        error_msg_ = "Failed to find embedding lookup output tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    output.resize((size_t)embedding->ne[0] * n_tokens);
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));

    ggml_backend_sched_reset(state_.sched);
    ggml_free(ctx0);
    return true;
}

bool TTSTransformer::lookup_single_embedding_row(struct ggml_tensor * embedding, int32_t token_id,
                                                 float * out_row) {
    if (!embedding) {
        error_msg_ = "Embedding tensor not found";
        return false;
    }
    if (!out_row) {
        error_msg_ = "Embedding output row is null";
        return false;
    }

    const int64_t embd_dim = embedding->ne[0];
    const int64_t vocab_size = embedding->ne[1];
    if (token_id < 0 || token_id >= vocab_size) {
        error_msg_ = "Embedding token ID out of range";
        return false;
    }

    const size_t row_offset = (size_t) token_id * embedding->nb[1];
    auto host_it = model_.host_embd_raw.find(embedding);
    if (host_it != model_.host_embd_raw.end()) {
        const uint8_t * row_ptr = host_it->second.data() + row_offset;
        if (embedding->type == GGML_TYPE_F32) {
            memcpy(out_row, row_ptr, (size_t) embd_dim * sizeof(float));
            return true;
        }
        const auto * traits = ggml_get_type_traits(embedding->type);
        if (traits && traits->to_float) {
            traits->to_float(row_ptr, out_row, embd_dim);
            return true;
        }
    }

    if (embedding->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(embedding, out_row, row_offset, (size_t) embd_dim * sizeof(float));
        return true;
    }
    if (embedding->type == GGML_TYPE_F16) {
        embd_row_fp16_scratch_.resize((size_t) embd_dim);
        ggml_backend_tensor_get(embedding, embd_row_fp16_scratch_.data(),
                                row_offset, (size_t) embd_dim * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < embd_dim; ++i) {
            out_row[i] = ggml_fp16_to_fp32(embd_row_fp16_scratch_[i]);
        }
        return true;
    }

    std::vector<int32_t> single_token = { token_id };
    std::vector<float> single_out;
    if (!lookup_embedding_rows(embedding, single_token.data(), 1,
                               "inp_compat_embed", "out_compat_embed", single_out)) {
        return false;
    }
    memcpy(out_row, single_out.data(), (size_t) embd_dim * sizeof(float));
    return true;
}

bool TTSTransformer::project_text_tokens(const int32_t * text_tokens, int32_t n_tokens,
                                         std::vector<float> & output) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (n_tokens <= 0) {
        output.clear();
        return true;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_text_tokens");
    ggml_set_input(inp_tokens);

    struct ggml_tensor * cur = ggml_get_rows(ctx0, model_.text_embd, inp_tokens);
    cur = ggml_mul_mat(ctx0, model_.text_proj_fc1, cur);
    cur = ggml_add(ctx0, cur, model_.text_proj_fc1_bias);
    cur = ggml_silu(ctx0, cur);
    cur = ggml_mul_mat(ctx0, model_.text_proj_fc2, cur);
    cur = ggml_add(ctx0, cur, model_.text_proj_fc2_bias);

    ggml_set_name(cur, "text_proj_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate text projection graph";
        ggml_free(ctx0);
        return false;
    }

    struct ggml_tensor * inp = ggml_graph_get_tensor(gf, "inp_text_tokens");
    ggml_backend_tensor_set(inp, text_tokens, 0, n_tokens * sizeof(int32_t));

    if (audited_graph_compute("text_projection", state_.sched, gf, &error_msg_) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute text projection graph";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }
    // Synchronous scheduler compute already synchronizes all backends.

    struct ggml_tensor * out = ggml_graph_get_tensor(gf, "text_proj_out");
    if (!out) {
        error_msg_ = "Failed to find text projection output tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    output.resize((size_t)model_.config.hidden_size * n_tokens);
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));

    ggml_backend_sched_reset(state_.sched);
    ggml_free(ctx0);
    return true;
}

bool TTSTransformer::build_prefill_graph(const int32_t * text_tokens, int32_t n_tokens,
                                         const float * speaker_embd, int32_t language_id,
                                         std::vector<float> & prefill_embd,
                                         std::vector<float> & trailing_text_hidden,
                                         std::vector<float> & tts_pad_embed,
                                         bool non_streaming_mode) {
    if (!text_tokens) {
        error_msg_ = "text_tokens is null";
        return false;
    }
    if (n_tokens < 4) {
        error_msg_ = "Need at least 4 text tokens for prefill";
        return false;
    }

    const auto & cfg = model_.config;
    const int32_t hidden_size = cfg.hidden_size;

    int32_t special_tokens[3] = {
        cfg.tts_bos_token_id,
        cfg.tts_eos_token_id,
        cfg.tts_pad_token_id,
    };

    std::vector<float> special_proj;
    if (!project_text_tokens(special_tokens, 3, special_proj)) {
        return false;
    }

    std::vector<float> tts_bos_embed(hidden_size);
    std::vector<float> tts_eos_embed(hidden_size);
    tts_pad_embed.resize(hidden_size);
    memcpy(tts_bos_embed.data(), special_proj.data() + 0 * hidden_size, hidden_size * sizeof(float));
    memcpy(tts_eos_embed.data(), special_proj.data() + 1 * hidden_size, hidden_size * sizeof(float));
    memcpy(tts_pad_embed.data(), special_proj.data() + 2 * hidden_size, hidden_size * sizeof(float));

    // Role tokens: first 3 of text_tokens (<|im_start|>, assistant, \n)
    std::vector<float> role_embed;
    if (!project_text_tokens(text_tokens, 3, role_embed)) {
        return false;
    }

    // Codec prefill tokens: [think_id, think_bos_id, language_id, think_eos_id]
    // (or nothink variant when language_id < 0 i.e. "auto")
    std::vector<int32_t> codec_prefill_tokens;
    if (language_id < 0) {
        codec_prefill_tokens = {
            cfg.codec_nothink_id,
            cfg.codec_think_bos_id,
            cfg.codec_think_eos_id,
        };
    } else {
        codec_prefill_tokens = {
            cfg.codec_think_id,
            cfg.codec_think_bos_id,
            language_id,
            cfg.codec_think_eos_id,
        };
    }

    std::vector<float> codec_prefill_embed;
    if (!lookup_embedding_rows(model_.codec_embd, codec_prefill_tokens.data(),
                               (int32_t)codec_prefill_tokens.size(),
                               "inp_codec_prefill_tokens", "codec_prefill_rows",
                               codec_prefill_embed)) {
        return false;
    }

    int32_t codec_tail_tokens[2] = { cfg.codec_pad_id, cfg.codec_bos_id };
    std::vector<float> codec_tail_embed;
    if (!lookup_embedding_rows(model_.codec_embd, codec_tail_tokens, 2,
                               "inp_codec_tail_tokens", "codec_tail_rows",
                               codec_tail_embed)) {
        return false;
    }

    const bool has_speaker = (speaker_embd != nullptr);
    const int32_t codec_input_len = (int32_t)codec_prefill_tokens.size() + (has_speaker ? 1 : 0) + 2;
    std::vector<float> codec_input_embedding((size_t)codec_input_len * hidden_size);

    {
        int32_t dst = 0;
        memcpy(codec_input_embedding.data(), codec_prefill_embed.data(), codec_prefill_embed.size() * sizeof(float));
        dst += (int32_t)codec_prefill_tokens.size();
        if (has_speaker) {
            memcpy(codec_input_embedding.data() + (size_t)dst * hidden_size,
                   speaker_embd, hidden_size * sizeof(float));
            ++dst;
        }
        memcpy(codec_input_embedding.data() + (size_t)dst * hidden_size,
               codec_tail_embed.data(), codec_tail_embed.size() * sizeof(float));
    }

    // codec_bos_embed = last row of codec_input_embedding
    const float * codec_bos_embed_ptr = codec_input_embedding.data() + (size_t)(codec_input_len - 1) * hidden_size;

    // ----------------------------------------------------------------
    // NON-STREAMING MODE  (non_streaming_mode=True in Python)
    // All text tokens fed at once overlaid with codec_pad, then codec_bos.
    //
    // Python:
    //   talker_input_embed = talker_input_embed[:, :-1]  # remove last token
    //   text_proj_all = cat([text_proj(text[3:-5]), tts_eos_embed], dim=1) + codec_pad * T
    //   bos_part      = tts_pad + codec_bos
    //   full_embed    = cat([role, overlay_prefix, text_proj_all, bos_part])
    //   trailing      = tts_pad_embed   (single vector)
    // ----------------------------------------------------------------
    if (non_streaming_mode) {
        // text_tokens layout:
        //   [0..2]   = role (<|im_start|>, assistant, \n)
        //   [3]      = first text token
        //   [4..n-6] = body text tokens
        //   [n-5..n-1] = trailing special tokens (<|im_end|>, \n, <|im_start|>, assistant, \n)
        //
        // We want: text_proj(tokens[3 .. n-6]) + tts_eos overlaid with codec_pad
        const int32_t body_start = 3;
        const int32_t body_end   = std::max(body_start, n_tokens - 5); // exclusive
        const int32_t body_len   = body_end - body_start;              // may be 0

        // Project body text tokens + tts_eos
        std::vector<float> body_proj;
        if (body_len > 0) {
            if (!project_text_tokens(text_tokens + body_start, body_len, body_proj)) {
                return false;
            }
        }
        // body_plus_eos = [body_proj | tts_eos_embed]  length = body_len + 1
        const int32_t text_part_len = body_len + 1;
        std::vector<float> text_part((size_t)text_part_len * hidden_size, 0.0f);
        if (body_len > 0) {
            memcpy(text_part.data(), body_proj.data(), body_proj.size() * sizeof(float));
        }
        memcpy(text_part.data() + (size_t)body_len * hidden_size, tts_eos_embed.data(), hidden_size * sizeof(float));

        // Overlay each text position with codec_pad
        std::vector<int32_t> pad_tokens(text_part_len, cfg.codec_pad_id);
        std::vector<float> pad_embeds;
        if (!lookup_embedding_rows(model_.codec_embd, pad_tokens.data(), text_part_len,
                                   "inp_ns_pad", "ns_pad_rows", pad_embeds)) {
            return false;
        }
        for (int32_t t = 0; t < text_part_len; ++t) {
            float * row = text_part.data() + (size_t)t * hidden_size;
            const float * pad = pad_embeds.data() + (size_t)t * hidden_size;
            for (int32_t h = 0; h < hidden_size; ++h) row[h] += pad[h];
        }

        // codec_plus_overlay (prefix): same as streaming, length = codec_input_len - 1
        // (think tokens + optional speaker) overlaid with tts_pad/tts_bos
        const int32_t prefix_len = codec_input_len - 1;
        std::vector<float> prefix_overlay((size_t)prefix_len * hidden_size);
        for (int32_t t = 0; t < prefix_len; ++t) {
            const float * overlay = (t == prefix_len - 1) ? tts_bos_embed.data() : tts_pad_embed.data();
            const float * codec_row = codec_input_embedding.data() + (size_t)t * hidden_size;
            float * out = prefix_overlay.data() + (size_t)t * hidden_size;
            for (int32_t h = 0; h < hidden_size; ++h) out[h] = overlay[h] + codec_row[h];
        }

        // Final tts_pad + codec_bos position
        std::vector<float> bos_part(hidden_size);
        for (int32_t h = 0; h < hidden_size; ++h) {
            bos_part[h] = tts_pad_embed.data()[h] + codec_bos_embed_ptr[h];
        }

        // Full prefill = [role(3) | prefix_overlay | text_part | bos_part]
        const int32_t total_len = 3 + prefix_len + text_part_len + 1;
        prefill_embd.resize((size_t)total_len * hidden_size);
        int32_t off = 0;
        memcpy(prefill_embd.data() + (size_t)off * hidden_size, role_embed.data(), role_embed.size() * sizeof(float));
        off += 3;
        memcpy(prefill_embd.data() + (size_t)off * hidden_size, prefix_overlay.data(), prefix_overlay.size() * sizeof(float));
        off += prefix_len;
        memcpy(prefill_embd.data() + (size_t)off * hidden_size, text_part.data(), text_part.size() * sizeof(float));
        off += text_part_len;
        memcpy(prefill_embd.data() + (size_t)off * hidden_size, bos_part.data(), hidden_size * sizeof(float));

        // Trailing = single tts_pad_embed row
        trailing_text_hidden = tts_pad_embed;
        return true;
    }

    // ----------------------------------------------------------------
    // STREAMING MODE (non_streaming_mode=False, the default)
    // Text tokens interleaved with codec sequence.
    // This is the original implementation.
    // ----------------------------------------------------------------
    const int32_t codec_plus_overlay_len = codec_input_len - 1;
    std::vector<float> codec_plus_overlay((size_t)codec_plus_overlay_len * hidden_size);
    for (int32_t t = 0; t < codec_plus_overlay_len; ++t) {
        const float * overlay = (t == codec_plus_overlay_len - 1)
            ? tts_bos_embed.data()
            : tts_pad_embed.data();
        const float * codec_row = codec_input_embedding.data() + (size_t)t * hidden_size;
        float * out_row = codec_plus_overlay.data() + (size_t)t * hidden_size;
        for (int32_t h = 0; h < hidden_size; ++h) {
            out_row[h] = overlay[h] + codec_row[h];
        }
    }

    std::vector<float> first_text_embed;
    if (!project_text_tokens(text_tokens + 3, 1, first_text_embed)) {
        return false;
    }

    std::vector<float> first_text_plus_codec_bos(hidden_size);
    for (int32_t h = 0; h < hidden_size; ++h) {
        first_text_plus_codec_bos[h] = first_text_embed[h] + codec_bos_embed_ptr[h];
    }

    const int32_t prefill_len = 3 + codec_plus_overlay_len + 1;
    prefill_embd.resize((size_t)prefill_len * hidden_size);
    memcpy(prefill_embd.data(), role_embed.data(), role_embed.size() * sizeof(float));
    memcpy(prefill_embd.data() + (size_t)3 * hidden_size,
           codec_plus_overlay.data(), codec_plus_overlay.size() * sizeof(float));
    memcpy(prefill_embd.data() + (size_t)(prefill_len - 1) * hidden_size,
           first_text_plus_codec_bos.data(), hidden_size * sizeof(float));

    const int32_t trailing_token_count = std::max(0, n_tokens - 9);
    std::vector<float> trailing_text_proj;
    if (trailing_token_count > 0) {
        if (!project_text_tokens(text_tokens + 4, trailing_token_count, trailing_text_proj)) {
            return false;
        }
    }

    const int32_t trailing_len = trailing_token_count + 1;
    trailing_text_hidden.resize((size_t)trailing_len * hidden_size);
    if (trailing_token_count > 0) {
        memcpy(trailing_text_hidden.data(), trailing_text_proj.data(), trailing_text_proj.size() * sizeof(float));
    }
    memcpy(trailing_text_hidden.data() + (size_t)(trailing_len - 1) * hidden_size,
           tts_eos_embed.data(), hidden_size * sizeof(float));

    return true;
}

struct ggml_cgraph * TTSTransformer::build_prefill_forward_graph(int32_t n_tokens, int32_t n_past, int32_t batch_idx) {
    const auto & cfg = model_.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.n_layers;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    struct ggml_tensor * inp_prefill_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, n_tokens);
    ggml_set_name(inp_prefill_embd, "inp_prefill_embd");
    ggml_set_input(inp_prefill_embd);
    
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    struct ggml_tensor * cur = inp_prefill_embd;
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];
        
        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
        
        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);
        
        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }
        
        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }
        
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        struct ggml_tensor * k_cache = state_.cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.cache.v_cache[il];
        
        // Batch offset for 4D cache [head_dim, n_kv_heads, n_ctx, n_batch]
        const size_t batch_off_k = (ggml_n_dims(k_cache) >= 4) ? (size_t)batch_idx * k_cache->nb[3] : 0;
        const size_t batch_off_v = (ggml_n_dims(v_cache) >= 4) ? (size_t)batch_idx * v_cache->nb[3] : 0;
        
        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2],
            n_past * k_cache->nb[2] + batch_off_k);
        
        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2],
            n_past * v_cache->nb[2] + batch_off_v);
        
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        
        int n_kv = n_past + n_tokens;
        
        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_kv,
            k_cache->nb[1], k_cache->nb[2],
            batch_off_k);
        
        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_kv,
            v_cache->nb[1], v_cache->nb[2],
            batch_off_v);
        
        // prefill: use soft_max_ext + causal mask (multi-token, causal masking required).
        // Flash attention is used for single-token decode steps only (build_step_graph).
        // For multi-token prefill, ggml_flash_attn_ext requires a carefully laid-out
        // F16 causal mask that depends on the KV cache position encoding — kept as
        // a future improvement when the exact layout contract is verified.
        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_diag_mask_inf(ctx0, KQ, n_past);
        KQ = ggml_soft_max_ext(ctx0, KQ, nullptr, KQscale, 0.0f);
        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, n_head * head_dim, n_tokens);
        
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;
        
        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
        gate = ggml_silu(ctx0, gate);
        
        cur = ggml_mul(ctx0, gate, up);
        
        // F16 weights need an explicit cast to F32 before mul_mat on CPU.
        // K-quant types (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0) are handled
        // natively by ggml_mul_mat — casting them wastes memory and time.
        struct ggml_tensor * ffn_down_w = (layer.ffn_down->type == GGML_TYPE_F16)
            ? ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32)
            : layer.ffn_down;
        cur = ggml_mul_mat(ctx0, ffn_down_w, cur);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;
    
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model_.output_norm);
    ggml_set_name(cur, "hidden_states");
    ggml_set_output(cur);

    struct ggml_tensor * logits = ggml_mul_mat(ctx0, model_.codec_head, cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    
    ggml_build_forward_expand(gf, logits);
    
    ggml_free(ctx0);
    
    return gf;
}

struct ggml_cgraph * TTSTransformer::build_step_graph(int32_t n_past, int32_t batch_idx) {
    const auto & cfg = model_.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.n_layers;
    const int n_tokens = 1;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    struct ggml_tensor * inp_step_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, 1);
    ggml_set_name(inp_step_embd, "inp_step_embd");
    ggml_set_input(inp_step_embd);
    
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    struct ggml_tensor * cur = inp_step_embd;
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];
        
        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
        
        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);
        
        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }
        
        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }
        
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        struct ggml_tensor * k_cache = state_.cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.cache.v_cache[il];
        
        const size_t batch_off_k = (ggml_n_dims(k_cache) >= 4) ? (size_t)batch_idx * k_cache->nb[3] : 0;
        const size_t batch_off_v = (ggml_n_dims(v_cache) >= 4) ? (size_t)batch_idx * v_cache->nb[3] : 0;
        
        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2],
            n_past * k_cache->nb[2] + batch_off_k);
        
        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2],
            n_past * v_cache->nb[2] + batch_off_v);
        
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        
        int n_kv = n_past + n_tokens;
        
        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_kv,
            k_cache->nb[1], k_cache->nb[2],
            batch_off_k);
        
        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_kv,
            v_cache->nb[1], v_cache->nb[2],
            batch_off_v);
        
        // flash attention: Q[head_dim, n_tokens, n_head], K/V[head_dim, n_kv, n_head_kv]
        // V is already non-transposed in the KV cache — exactly what flash_attn_ext needs
        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));
        struct ggml_tensor * KQV = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, KQscale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(KQV, GGML_PREC_F32);
        // flash_attn_ext output: [head_dim, n_head, n_tokens] -> reshape to [n_head*head_dim, n_tokens]
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, n_head * head_dim, n_tokens);
        
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;
        
        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
        gate = ggml_silu(ctx0, gate);
        
        cur = ggml_mul(ctx0, gate, up);
        
        // F16 weights need an explicit cast to F32 before mul_mat on CPU.
        // K-quant types (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0) are handled
        // natively by ggml_mul_mat — casting them wastes memory and time.
        struct ggml_tensor * ffn_down_w = (layer.ffn_down->type == GGML_TYPE_F16)
            ? ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32)
            : layer.ffn_down;
        cur = ggml_mul_mat(ctx0, ffn_down_w, cur);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;
    
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model_.output_norm);
    ggml_set_name(cur, "hidden_states");
    ggml_set_output(cur);
    
    struct ggml_tensor * logits = ggml_mul_mat(ctx0, model_.codec_head, cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    
    ggml_build_forward_expand(gf, logits);
    
    ggml_free(ctx0);
    
    return gf;
}

struct ggml_cgraph * TTSTransformer::build_code_pred_prefill_graph(int32_t batch_idx) {
    const auto & cfg = model_.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.code_pred_layers;
    const int n_tokens = 2;
    // Code predictor may have different hidden_size than talker (1.7B model)
    const int cp_hs = cfg.code_pred_hidden_size;
    const int talker_hs = cfg.hidden_size;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
    
    // Input: past_hidden from talker [talker_hs] (talker's hidden_size)
    struct ggml_tensor * inp_hidden = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, talker_hs);
    ggml_set_name(inp_hidden, "inp_hidden");
    ggml_set_input(inp_hidden);
    
    // Input: codebook 0 token embedding [talker_hs] (from talker's codec_embd)
    struct ggml_tensor * inp_cb0_embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, talker_hs);
    ggml_set_name(inp_cb0_embd, "inp_cb0_embd");
    ggml_set_input(inp_cb0_embd);
    
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    
    // Project talker-space inputs to code predictor space using small_to_mtp_projection
    struct ggml_tensor * hidden_2d = ggml_reshape_2d(ctx0, inp_hidden, talker_hs, 1);
    struct ggml_tensor * cb0_2d = ggml_reshape_2d(ctx0, inp_cb0_embd, talker_hs, 1);
    struct ggml_tensor * cur;
    if (model_.code_pred_proj) {
        // 1.7B: project from talker_hs to cp_hs
        struct ggml_tensor * proj_hidden = ggml_mul_mat(ctx0, model_.code_pred_proj, hidden_2d);
        if (model_.code_pred_proj_bias) {
            proj_hidden = ggml_add(ctx0, proj_hidden, model_.code_pred_proj_bias);
        }
        struct ggml_tensor * proj_cb0 = ggml_mul_mat(ctx0, model_.code_pred_proj, cb0_2d);
        if (model_.code_pred_proj_bias) {
            proj_cb0 = ggml_add(ctx0, proj_cb0, model_.code_pred_proj_bias);
        }
        cur = ggml_concat(ctx0, proj_hidden, proj_cb0, 1);
    } else {
        // 0.6B: same dimension, no projection
        cur = ggml_concat(ctx0, hidden_2d, cb0_2d, 1);
    }
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.code_pred_layers[il];
        
        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
        
        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);
        
        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }
        
        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }
        
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        struct ggml_tensor * k_cache = state_.code_pred_cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.code_pred_cache.v_cache[il];
        
        // Batch offset for 4D cache
        const size_t batch_off_k_cp = (ggml_n_dims(k_cache) >= 4) ? (size_t)batch_idx * k_cache->nb[3] : 0;
        const size_t batch_off_v_cp = (ggml_n_dims(v_cache) >= 4) ? (size_t)batch_idx * v_cache->nb[3] : 0;
        
        // Store at position 0 (prefill starts fresh)
        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2],
            batch_off_k_cp);
        
        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2],
            batch_off_v_cp);
        
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        
        // code predictor prefill: use soft_max_ext + causal mask (2-token, mask required)
        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        struct ggml_tensor * K = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
        struct ggml_tensor * V = ggml_permute(ctx0, Vcur, 0, 2, 1, 3);
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_diag_mask_inf(ctx0, KQ, 0);
        KQ = ggml_soft_max_ext(ctx0, KQ, nullptr, KQscale, 0.0f);
        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, n_head * head_dim, n_tokens);
        
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;
        
        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
        gate = ggml_silu(ctx0, gate);
        
        cur = ggml_mul(ctx0, gate, up);
        
        struct ggml_tensor * ffn_down_w = (layer.ffn_down->type == GGML_TYPE_F16)
            ? ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32)
            : layer.ffn_down;
        cur = ggml_mul_mat(ctx0, ffn_down_w, cur);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
     cur = inpL;
     
     cur = ggml_rms_norm(ctx0, cur, eps);
     cur = ggml_mul(ctx0, cur, model_.code_pred_output_norm);
     
      struct ggml_tensor * last_hidden = ggml_view_2d(ctx0, cur, cp_hs, 1,
                                                       cur->nb[1], cur->nb[1]);
     
     struct ggml_tensor * logits = ggml_mul_mat(ctx0, model_.code_pred_head[0], last_hidden);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    
    ggml_build_forward_expand(gf, logits);
    
    ggml_free(ctx0);
    
    return gf;
}

struct ggml_cgraph * TTSTransformer::build_code_pred_step_graph(int32_t n_past, int32_t generation_step, int32_t batch_idx) {
    const auto & cfg = model_.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.code_pred_layers;
    const int n_tokens = 1;
    // Code predictor may have different hidden_size than talker (1.7B model)
    const int talker_hs = cfg.hidden_size;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
    
    // Input in talker space: step 0 = hidden state, step 1+ = host-looked-up codec_embedding
    struct ggml_tensor * inp_hidden = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, talker_hs);
    ggml_set_name(inp_hidden, "inp_hidden");
    ggml_set_input(inp_hidden);
    
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    
    struct ggml_tensor * cur = ggml_reshape_2d(ctx0, inp_hidden, talker_hs, 1);
    // Project from talker space to code predictor space (if different)
    if (model_.code_pred_proj) {
        cur = ggml_mul_mat(ctx0, model_.code_pred_proj, cur);
        if (model_.code_pred_proj_bias) {
            cur = ggml_add(ctx0, cur, model_.code_pred_proj_bias);
        }
    }
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.code_pred_layers[il];
        
        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
        
        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);
        
        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }
        
        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }
        
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        struct ggml_tensor * k_cache = state_.code_pred_cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.code_pred_cache.v_cache[il];
        
        const size_t batch_off_k_cs = (ggml_n_dims(k_cache) >= 4) ? (size_t)batch_idx * k_cache->nb[3] : 0;
        const size_t batch_off_v_cs = (ggml_n_dims(v_cache) >= 4) ? (size_t)batch_idx * v_cache->nb[3] : 0;
        
        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2],
            n_past * k_cache->nb[2] + batch_off_k_cs);
        
        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2],
            n_past * v_cache->nb[2] + batch_off_v_cs);
        
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        
        int n_kv = n_past + n_tokens;
        
        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_kv,
            k_cache->nb[1], k_cache->nb[2],
            batch_off_k_cs);
        
        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_kv,
            v_cache->nb[1], v_cache->nb[2],
            batch_off_v_cs);
        
        // flash attention: Q[head_dim, n_tokens, n_head], K/V[head_dim, n_kv, n_head_kv]
        // V is already non-transposed in the KV cache — exactly what flash_attn_ext needs
        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));
        struct ggml_tensor * KQV = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, KQscale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(KQV, GGML_PREC_F32);
        // flash_attn_ext output: [head_dim, n_head, n_tokens] -> reshape to [n_head*head_dim, n_tokens]
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, n_head * head_dim, n_tokens);
        
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;
        
        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
        gate = ggml_silu(ctx0, gate);
        
        cur = ggml_mul(ctx0, gate, up);
        
        // F16 weights need an explicit cast to F32 before mul_mat on CPU.
        // K-quant types (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0) are handled
        // natively by ggml_mul_mat — casting them wastes memory and time.
        struct ggml_tensor * ffn_down_w = (layer.ffn_down->type == GGML_TYPE_F16)
            ? ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32)
            : layer.ffn_down;
        cur = ggml_mul_mat(ctx0, ffn_down_w, cur);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
     cur = inpL;
     
     cur = ggml_rms_norm(ctx0, cur, eps);
     cur = ggml_mul(ctx0, cur, model_.code_pred_output_norm);
     
     struct ggml_tensor * logits = ggml_mul_mat(ctx0, model_.code_pred_head[generation_step], cur);
     ggml_set_name(logits, "logits");
     ggml_set_output(logits);
     
     ggml_build_forward_expand(gf, logits);
    
    ggml_free(ctx0);
    
    return gf;
}

bool TTSTransformer::forward_prefill(const float * prefill_embd, int32_t n_tokens,
                                     int32_t n_past, std::vector<float> & output,
                                     std::vector<float> * logits_out,
                                     int32_t batch_idx) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!prefill_embd) {
        error_msg_ = "prefill_embd is null";
        return false;
    }
    if (n_tokens <= 0) {
        error_msg_ = "n_tokens must be > 0";
        return false;
    }

    // Chunk large prefills to avoid excessive peak memory in the GGML scheduler.
    // The graph node count is fixed (~38 ops/layer × 28 layers ≈ 1069 nodes regardless
    // of n_tokens), but the scheduler's internal buffer allocation scales with tensor
    // sizes (e.g. KQ attention matrix is [n_kv, n_tokens] per head per layer).
    // For large ICL prefills (90+ tokens) this caused ggml_backend_sched_alloc_graph
    // to stall or fail silently. Processing in chunks of 16 keeps peak allocations small.
    static constexpr int32_t PREFILL_CHUNK_SIZE = 16;
    if (n_tokens > PREFILL_CHUNK_SIZE) {
        const int32_t H = model_.config.hidden_size;
        // Ensure KV cache is large enough for the full prefill upfront.
        if (state_.cache.n_ctx == 0) {
            const int32_t min_ctx = std::max<int32_t>(256, n_past + n_tokens + 16);
            if (!init_kv_cache(min_ctx)) return false;
        }
        if (n_past + n_tokens > state_.cache.n_ctx) {
            const int32_t new_ctx = n_past + n_tokens + 512;
            if (!extend_kv_cache(new_ctx, n_past)) {
                error_msg_ = "Context length exceeded and failed to extend KV cache";
                return false;
            }
        }
        // Process in chunks. Only the last chunk needs logits_out.
        int32_t pos = n_past;
        int32_t remaining = n_tokens;
        output.resize((size_t)n_tokens * H);
        while (remaining > 0) {
            const int32_t chunk = std::min(remaining, PREFILL_CHUNK_SIZE);
            const bool is_last  = (remaining == chunk);
            const float * chunk_embd = prefill_embd + (size_t)(n_tokens - remaining) * H;
            std::vector<float> chunk_out;
            std::vector<float> * chunk_logits = is_last ? logits_out : nullptr;
            if (!forward_prefill(chunk_embd, chunk, pos, chunk_out, chunk_logits, batch_idx)) {
                return false;
            }
            // Copy chunk hidden states into the output buffer at the right offset.
            const size_t off = (size_t)(n_tokens - remaining) * H;
            memcpy(output.data() + off, chunk_out.data(), (size_t)chunk * H * sizeof(float));
            pos       += chunk;
            remaining -= chunk;
        }
        // last_hidden_ was set by the final chunk's forward_prefill call — already correct.
        return true;
    }

    if (state_.cache.n_ctx == 0) {
        const int32_t min_ctx = std::max<int32_t>(256, n_past + n_tokens + 16);
        if (!init_kv_cache(min_ctx)) {
            return false;
        }
    }
    
    if (n_past + n_tokens > state_.cache.n_ctx) {
        const int32_t new_ctx = n_past + n_tokens + 512;
        if (!extend_kv_cache(new_ctx, n_past)) {
            error_msg_ = "Context length exceeded and failed to extend KV cache";
            return false;
        }
    }
    
#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_cgraph * gf = build_prefill_forward_graph(n_tokens, n_past, batch_idx);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_tensor * inp_prefill = ggml_graph_get_tensor(gf, "inp_prefill_embd");
    if (inp_prefill) {
        ggml_backend_tensor_set(inp_prefill, prefill_embd, 0,
                                (size_t)n_tokens * model_.config.hidden_size * sizeof(float));
    }
    
    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        std::vector<int32_t> positions(n_tokens);
        for (int i = 0; i < n_tokens; ++i) {
            positions[i] = n_past + i;
        }
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, n_tokens * sizeof(int32_t));
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (audited_graph_compute("talker_prefill", state_.sched, gf, &error_msg_) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    // Synchronous scheduler compute already synchronizes all backends.
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    struct ggml_tensor * hidden = ggml_graph_get_tensor(gf, "hidden_states");
    if (!hidden) {
        error_msg_ = "Failed to find hidden_states tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    output.resize(n_tokens * model_.config.hidden_size);
    ggml_backend_tensor_get(hidden, output.data(), 0, output.size() * sizeof(float));
    
    last_hidden_.resize(model_.config.hidden_size);
    ggml_backend_tensor_get(hidden, last_hidden_.data(), 
                           (n_tokens - 1) * model_.config.hidden_size * sizeof(float),
                           model_.config.hidden_size * sizeof(float));

    if (logits_out) {
        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }

        logits_out->resize(model_.config.codec_vocab_size);
        ggml_backend_tensor_get(logits, logits_out->data(),
                                (n_tokens - 1) * model_.config.codec_vocab_size * sizeof(float),
                                model_.config.codec_vocab_size * sizeof(float));
    }
    
    {
        const int32_t new_used = n_past + n_tokens;
        if (new_used > state_.cache.n_used) state_.cache.n_used = new_used;
    }
    
    if (logits_out && !validate_logits(*logits_out, "talker prefill", error_msg_)) {
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    ggml_backend_sched_reset(state_.sched);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    return true;
}

bool TTSTransformer::forward_step(const float * step_embd, int32_t n_past,
                                  std::vector<float> & output,
                                  std::vector<float> * hidden_out,
                                  int32_t batch_idx) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!step_embd) {
        error_msg_ = "step_embd is null";
        return false;
    }

    if (state_.cache.n_ctx == 0) {
        const int32_t min_ctx = std::max<int32_t>(256, n_past + 1 + 16);
        if (!init_kv_cache(min_ctx)) {
            return false;
        }
    }

    if (n_past + 1 > state_.cache.n_ctx) {
        const int32_t new_ctx = state_.cache.n_ctx + 512;
        if (!extend_kv_cache(new_ctx, n_past)) {
            error_msg_ = "Context length exceeded and failed to extend KV cache";
            return false;
        }
    }
    
#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_cgraph * gf = build_step_graph(n_past, batch_idx);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
    static bool printed_step_splits = false;
    if (!printed_step_splits) {
        fprintf(stderr, "  [Talker Step] Graph splits: %d\n",
                ggml_backend_sched_get_n_splits(state_.sched));
        std::map<std::string, int> cpu_ops;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            struct ggml_tensor * node = ggml_graph_node(gf, i);
            if (ggml_backend_sched_get_tensor_backend(state_.sched, node) == state_.backend_cpu) {
                cpu_ops[ggml_op_name(node->op)]++;
            }
        }
        fprintf(stderr, "  [Talker Step] Ops placed on CPU:\n");
        for (auto & kv : cpu_ops) {
            fprintf(stderr, "    - %s: %d\n", kv.first.c_str(), kv.second);
        }
        printed_step_splits = true;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_tensor * inp_step = ggml_graph_get_tensor(gf, "inp_step_embd");
    if (inp_step) {
        ggml_backend_tensor_set(inp_step, step_embd, 0,
                                model_.config.hidden_size * sizeof(float));
    }
    
    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        int32_t pos = n_past;
        ggml_backend_tensor_set(inp_pos, &pos, 0, sizeof(int32_t));
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (audited_graph_compute("talker_step", state_.sched, gf, &error_msg_) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    // Synchronous scheduler compute already synchronizes all backends.
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    struct ggml_tensor * hidden = ggml_graph_get_tensor(gf, "hidden_states");

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (hidden) {
        last_hidden_.resize(model_.config.hidden_size);
        ggml_backend_tensor_get(hidden, last_hidden_.data(), 0, 
                               model_.config.hidden_size * sizeof(float));
        if (hidden_out) {
            *hidden_out = last_hidden_;
        }
    }
    
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        error_msg_ = "Failed to find logits tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    output.resize(model_.config.codec_vocab_size);
    ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));
    if (!validate_logits(output, "talker step", error_msg_)) {
        ggml_backend_sched_reset(state_.sched);
        return false;
    }

    
    {
        const int32_t new_used = n_past + 1;
        if (new_used > state_.cache.n_used) state_.cache.n_used = new_used;
    }
    
    ggml_backend_sched_reset(state_.sched);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    return true;
}

static int32_t argmax(const float * data, int32_t n) {
    int32_t max_idx = 0;
    float max_val = data[0];
    for (int32_t i = 1; i < n; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

bool TTSTransformer::predict_codes_autoregressive_coreml(const float * hidden,
                                                         int32_t codebook_0_token,
                                                         std::vector<int32_t> & output,
                                                         float temperature,
                                                         int32_t top_k,
                                                         float top_p) {
    if (!use_coreml_code_predictor_ || !coreml_code_predictor_.is_loaded()) {
        error_msg_ = "CoreML code predictor is not loaded";
        return false;
    }

    const auto & cfg = model_.config;
    const int32_t n_steps = cfg.n_codebooks - 1;

    output.resize(n_steps);
    std::vector<float> logits_data(cfg.code_pred_vocab_size);
    std::vector<float> code_probs(cfg.code_pred_vocab_size);
    std::vector<float> seq_embd((size_t)16 * cfg.hidden_size, 0.0f);

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

    auto sample_or_argmax = [&](float * logits_ptr, int32_t vocab_size) -> int32_t {
        if (temperature <= 0.0f) {
            return argmax(logits_ptr, vocab_size);
        }

        for (int32_t i = 0; i < vocab_size; ++i) {
            logits_ptr[i] /= temperature;
        }

        if (top_k > 0 && top_k < vocab_size) {
            std::vector<std::pair<float, int32_t>> scored(vocab_size);
            for (int32_t i = 0; i < vocab_size; ++i) {
                scored[i] = {logits_ptr[i], i};
            }
            std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                    return a.first > b.first;
                });
            float threshold = scored[top_k - 1].first;
            for (int32_t i = 0; i < vocab_size; ++i) {
                if (logits_ptr[i] < threshold) {
                    logits_ptr[i] = -INFINITY;
                }
            }
        }

        if (top_p > 0.0f && top_p < 1.0f) {
            float max_logit = *std::max_element(logits_ptr, logits_ptr + vocab_size);
            std::vector<std::pair<float, int32_t>> sp(vocab_size);
            double sum_p = 0.0;
            for (int32_t i = 0; i < vocab_size; ++i) {
                float prob = expf(logits_ptr[i] - max_logit);
                sp[i] = {prob, i};
                sum_p += prob;
            }
            for (auto & x : sp) x.first /= (float)sum_p;
            std::sort(sp.begin(), sp.end(),
                [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                    return a.first > b.first;
                });
            float cumul = 0.0f;
            int32_t keep_until = vocab_size;
            for (int32_t k = 0; k < vocab_size; ++k) {
                cumul += sp[k].first;
                if (cumul >= top_p) {
                    keep_until = k + 1;
                    break;
                }
            }
            for (int32_t k = keep_until; k < vocab_size; ++k) {
                logits_ptr[sp[k].second] = -INFINITY;
            }
        }

        float max_logit = *std::max_element(logits_ptr, logits_ptr + vocab_size);
        double sum = 0.0;
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = expf(logits_ptr[i] - max_logit);
            sum += code_probs[i];
        }
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = (float)(code_probs[i] / sum);
        }

        std::discrete_distribution<int32_t> dist(code_probs.begin(), code_probs.begin() + vocab_size);
        return dist(rng_);
    };

    memcpy(seq_embd.data(), hidden, (size_t)cfg.hidden_size * sizeof(float));
    if (!lookup_single_embedding_row(model_.codec_embd, codebook_0_token,
                                     seq_embd.data() + cfg.hidden_size)) {
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_code_pred_init_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    for (int32_t step = 0; step < n_steps; ++step) {
        if (step > 0) {
            float * dst = seq_embd.data() + (size_t)(step + 1) * cfg.hidden_size;
            if (!lookup_single_embedding_row(model_.code_pred_embd[step - 1], output[step - 1], dst)) {
                return false;
            }
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!coreml_code_predictor_.predict_step(step, seq_embd.data(), step + 2, cfg.hidden_size, logits_data)) {
            error_msg_ = "CoreML predictor step failed: " + coreml_code_predictor_.get_error();
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (timing_) timing_->t_code_pred_compute_ms += dt_ms;
        if (timing_) timing_->t_code_pred_coreml_ms += dt_ms;
#endif

        if ((int32_t)logits_data.size() != cfg.code_pred_vocab_size) {
            error_msg_ = "CoreML predictor returned unexpected logits size";
            return false;
        }
        output[step] = sample_or_argmax(logits_data.data(), cfg.code_pred_vocab_size);

#ifdef QWEN3_TTS_TIMING
        if (timing_) {
            if (step == 0) {
                timing_->t_code_pred_prefill_ms += dt_ms;
            } else {
                timing_->t_code_pred_steps_ms += dt_ms;
            }
        }
#endif
    }

    return true;
}

// Forward declaration — defined later in this file before generate()
static int32_t sample_token(
    std::vector<float> & logits, int32_t vocab_size, int32_t eos_id,
    int32_t suppress_start, const std::unordered_set<int32_t> & gen_tokens,
    float repetition_penalty, float temperature, int32_t top_k, float top_p,
    std::vector<float> & probs, std::mt19937 & rng);

// Forward declaration for the struct-based overload
struct sample_token_params;
static int32_t sample_token(
    std::vector<float> & logits,
    const std::unordered_set<int32_t> & gen_tokens,
    const std::vector<int32_t> * token_history,
    const std::unordered_map<int32_t,int32_t> * token_counts,
    std::vector<float> & probs,
    std::mt19937 & rng,
    const sample_token_params & p);

bool TTSTransformer::predict_codes_autoregressive(const float * hidden, int32_t codebook_0_token,
                                                   std::vector<int32_t> & output,
                                                   float temperature, int32_t top_k, float top_p,
                                                   int32_t batch_idx) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    const auto & cfg = model_.config;

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

    if (use_coreml_code_predictor_ && coreml_code_predictor_.is_loaded()) {
        if (predict_codes_autoregressive_coreml(hidden, codebook_0_token, output, temperature, top_k, top_p)) {
            return true;
        }
        if (skip_ggml_code_pred_layers_) {
            return false;
        }
        fprintf(stderr, "  CoreML code predictor failed, falling back to GGML: %s\n", error_msg_.c_str());
        use_coreml_code_predictor_ = false;
    }
    
    if (state_.code_pred_cache.n_ctx < 16 ||
        state_.code_pred_cache.n_batch != state_.cache.n_batch) {
        if (!init_code_pred_kv_cache(16, state_.cache.n_batch)) {
            return false;
        }
    }
    clear_code_pred_kv_cache();
    
    output.resize(15);
    std::vector<float> logits_data(cfg.code_pred_vocab_size);
    std::vector<float> code_probs(cfg.code_pred_vocab_size);
    std::unordered_set<int32_t> empty_set;  // no rep penalty for sub-talker

    // Helper: sample from code predictor logits using temperature + top-k + top-p
    // sub-talker does not use suppress_tokens or repetition_penalty
    auto sample_or_argmax = [&](float * logits_ptr, int32_t vocab_size) -> int32_t {
        std::vector<float> lv(logits_ptr, logits_ptr + vocab_size);
        return sample_token(lv, vocab_size, -1 /*no eos*/, vocab_size /*suppress none*/,
                            empty_set, 1.0f /*no rep penalty*/,
                            temperature, top_k, top_p /*sub-talker top_p*/,
                            code_probs, rng_);
    };
    
    std::vector<float> cb0_embd(cfg.hidden_size);
    if (!lookup_single_embedding_row(model_.codec_embd, codebook_0_token, cb0_embd.data())) {
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_code_pred_init_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    // Prefill with 2 tokens [past_hidden, cb0_embd]
    {
#ifdef QWEN3_TTS_TIMING
        auto t_pf_start = clk::now();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_cgraph * gf = build_code_pred_prefill_graph(batch_idx);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
            error_msg_ = "Failed to allocate code predictor prefill graph";
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_tensor * inp_hidden = ggml_graph_get_tensor(gf, "inp_hidden");
        if (inp_hidden) {
            ggml_backend_tensor_set(inp_hidden, hidden, 0, cfg.hidden_size * sizeof(float));
        }
        
        struct ggml_tensor * inp_cb0_embd = ggml_graph_get_tensor(gf, "inp_cb0_embd");
        if (inp_cb0_embd) {
            ggml_backend_tensor_set(inp_cb0_embd, cb0_embd.data(), 0, cfg.hidden_size * sizeof(float));
        }
        
        struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
        if (inp_pos) {
            int32_t positions[2] = {0, 1};
            ggml_backend_tensor_set(inp_pos, positions, 0, 2 * sizeof(int32_t));
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (audited_graph_compute("code_predictor_prefill", state_.sched, gf, &error_msg_) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute code predictor prefill graph";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
    // Synchronous scheduler compute already synchronizes all backends.
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
        
        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor in prefill";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        ggml_backend_tensor_get(logits, logits_data.data(), 0, 
                                 cfg.code_pred_vocab_size * sizeof(float));
        
        if (!validate_logits(logits_data, "code predictor prefill", error_msg_)) {
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
        output[0] = sample_or_argmax(logits_data.data(), cfg.code_pred_vocab_size);
        
        ggml_backend_sched_reset(state_.sched);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (timing_) timing_->t_code_pred_prefill_ms += std::chrono::duration<double, std::milli>(t1 - t_pf_start).count();
#endif
    }
    
    // Generate 14 more tokens autoregressively
#ifdef QWEN3_TTS_TIMING
    auto t_steps_start = clk::now();
#endif
    std::vector<float> step_embd(cfg.hidden_size);
    for (int step = 1; step < 15; ++step) {
        int32_t n_past = step + 1;

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_cgraph * gf = build_code_pred_step_graph(n_past, step, batch_idx);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
            error_msg_ = "Failed to allocate code predictor step graph";
            return false;
        }
        static bool printed_codepred_splits = false;
        if (!printed_codepred_splits) {
            fprintf(stderr, "  [CodePred Step] Graph splits: %d\n",
                    ggml_backend_sched_get_n_splits(state_.sched));
            std::map<std::string, int> cpu_ops;
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                struct ggml_tensor * node = ggml_graph_node(gf, i);
                if (ggml_backend_sched_get_tensor_backend(state_.sched, node) == state_.backend_cpu) {
                    cpu_ops[ggml_op_name(node->op)]++;
                }
            }
            fprintf(stderr, "  [CodePred Step] Ops placed on CPU:\n");
            for (auto & kv : cpu_ops) {
                fprintf(stderr, "    - %s: %d\n", kv.first.c_str(), kv.second);
            }
            printed_codepred_splits = true;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!lookup_single_embedding_row(model_.code_pred_embd[step - 1], output[step - 1], step_embd.data())) {
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
        struct ggml_tensor * inp_hidden = ggml_graph_get_tensor(gf, "inp_hidden");
        if (inp_hidden) {
            ggml_backend_tensor_set(inp_hidden, step_embd.data(), 0, cfg.hidden_size * sizeof(float));
        }
        
        struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
        if (inp_pos) {
            int32_t pos = n_past;
            ggml_backend_tensor_set(inp_pos, &pos, 0, sizeof(int32_t));
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (audited_graph_compute("code_predictor_step", state_.sched, gf, &error_msg_) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute code predictor step graph";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
    // Synchronous scheduler compute already synchronizes all backends.
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
        
        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        ggml_backend_tensor_get(logits, logits_data.data(), 0, 
                                 cfg.code_pred_vocab_size * sizeof(float));
        
        if (!validate_logits(logits_data, "code predictor step", error_msg_)) {
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
        output[step] = sample_or_argmax(logits_data.data(), cfg.code_pred_vocab_size);
        
        ggml_backend_sched_reset(state_.sched);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    }
#ifdef QWEN3_TTS_TIMING
    if (timing_) timing_->t_code_pred_steps_ms += std::chrono::duration<double, std::milli>(clk::now() - t_steps_start).count();
#endif
    
    return true;
}

// ---------------------------------------------------------------------------
// Shared sampling helper: suppress → rep-penalty → temperature → top_k →
// top_p → softmax → sample. Called from generate(), generate_icl(),
// generate_from_prefill().
// Returns the sampled token index.
// ---------------------------------------------------------------------------
// Extended token sampler — supports all sampling strategies adopted from llama.cpp.
// Parameters beyond the original set default to "disabled" (0.0f or -1) so
// existing call-sites that pass fewer args are unaffected.
struct sample_token_params {
    int32_t vocab_size       = 0;
    int32_t eos_id           = -1;
    int32_t suppress_start  = INT32_MAX;   // suppress [suppress_start, vocab_size) except eos
    float   repetition_penalty = 1.0f;     // HuggingFace multiplicative rep penalty
    float   frequency_penalty  = 0.0f;     // subtract freq_penalty * count from logit
    float   presence_penalty   = 0.0f;     // subtract presence_penalty if token ever appeared
    float   temperature        = 1.0f;     // 0 = greedy
    float   dyntemp_range      = 0.0f;     // dynamic temperature half-range; 0 = disabled
    float   dyntemp_exponent   = 1.0f;     // dynamic temperature shaping exponent
    int32_t top_k              = 0;        // 0 = disabled
    float   top_p              = 1.0f;     // 1.0 = disabled
    float   min_p              = 0.0f;     // 0.0 = disabled; keep tokens >= min_p * max_prob
    float   dry_multiplier     = 0.0f;     // DRY penalty scale; 0 = disabled
    float   dry_base           = 1.75f;    // DRY exponential growth base
    int32_t dry_allowed_length = 2;        // min n-gram length before DRY penalises
    int32_t dry_penalty_last_n = -1;       // context window for DRY (-1 = all tokens)
    int32_t min_new_tokens     = 0;        // ban EOS until n_generated reaches this (0 = off)
    int32_t n_generated        = 0;        // tokens generated so far, set by caller each step
};

static int32_t sample_token(
        std::vector<float> & logits,
        const std::unordered_set<int32_t> & gen_tokens,      // for rep/freq/presence penalties
        const std::vector<int32_t> * token_history,           // for DRY; nullptr = disabled
        const std::unordered_map<int32_t,int32_t> * token_counts, // token -> count for freq/presence
        std::vector<float> & probs,
        std::mt19937 & rng,
        const sample_token_params & p) {

    const int32_t V = p.vocab_size;

    // 1. Suppress range [suppress_start, V) except EOS
    for (int32_t i = p.suppress_start; i < V; ++i) {
        if (i != p.eos_id) logits[i] = -INFINITY;
    }

    // 1b. min_new_tokens: EOS banned until enough tokens were generated
    if (p.min_new_tokens > 0 && p.n_generated < p.min_new_tokens &&
        p.eos_id >= 0 && p.eos_id < V) {
        logits[p.eos_id] = -INFINITY;
    }

    // 2. Repetition + frequency + presence penalties
    if (p.repetition_penalty != 1.0f || p.frequency_penalty != 0.0f || p.presence_penalty != 0.0f) {
        for (int32_t tok : gen_tokens) {
            if (tok < 0 || tok >= V) continue;
            // Repetition penalty (HuggingFace multiplicative style)
            if (p.repetition_penalty != 1.0f) {
                if (logits[tok] > 0.0f) logits[tok] /= p.repetition_penalty;
                else                    logits[tok] *= p.repetition_penalty;
            }
            // Frequency penalty (additive, proportional to count)
            if (p.frequency_penalty != 0.0f && token_counts) {
                auto it = token_counts->find(tok);
                if (it != token_counts->end())
                    logits[tok] -= p.frequency_penalty * (float)it->second;
            }
            // Presence penalty (flat additive if token appeared at all)
            if (p.presence_penalty != 0.0f)
                logits[tok] -= p.presence_penalty;
        }
    }

    // 3. DRY (Don't Repeat Yourself) n-gram penalty — from llama.cpp
    // Penalises any token that would extend a pattern already seen in the context.
    if (p.dry_multiplier != 0.0f && token_history && !token_history->empty()) {
        const auto & hist = *token_history;
        int32_t n_hist = (int32_t)hist.size();
        int32_t scan_len = (p.dry_penalty_last_n < 0)
                          ? n_hist
                          : (int32_t)std::min((int32_t)n_hist, p.dry_penalty_last_n);

        // For each candidate token, find the longest suffix in recent history
        // that matches the end of history + candidate token
        for (int32_t tok = 0; tok < V; ++tok) {
            if (!std::isfinite(logits[tok])) continue;
            // Build candidate sequence: last dry_allowed_length-1 tokens + tok
            int32_t max_match = 0;
            for (int32_t i = 1; i < scan_len; ++i) {
                if (hist[n_hist - 1 - (i - 1)] != tok) {
                    // Check if the suffix ending at position (n_hist-1-i) matches
                    // the last (match_len) tokens
                    continue;
                }
                // hist[n_hist-1-i] == tok: potential match of length 1+
                int32_t match = 1;
                while (match < p.dry_allowed_length && match <= i &&
                       hist[n_hist - 1 - (i + match - 1)] ==
                       hist[n_hist - 1 - (match - 1)]) {
                    ++match;
                }
                if (match > max_match) max_match = match;
            }
            if (max_match >= p.dry_allowed_length) {
                // Exponential penalty: multiplier * base^(match - allowed_length)
                float penalty = p.dry_multiplier *
                    powf(p.dry_base, (float)(max_match - p.dry_allowed_length));
                logits[tok] -= penalty;
            }
        }
    }

    // 4. Greedy if temperature == 0
    if (p.temperature <= 0.0f) {
        return argmax(logits.data(), V);
    }

    // 5. Dynamic temperature — adapt temp based on distribution entropy
    float eff_temp = p.temperature;
    if (p.dyntemp_range > 0.0f) {
        // Compute softmax entropy at base temperature
        float mx = *std::max_element(logits.data(), logits.data() + V);
        double sum_e = 0.0;
        for (int32_t i = 0; i < V; ++i) sum_e += expf((logits[i] - mx) / p.temperature);
        float entropy = 0.0f;
        for (int32_t i = 0; i < V; ++i) {
            float pi = expf((logits[i] - mx) / p.temperature) / (float)sum_e;
            if (pi > 1e-10f) entropy -= pi * logf(pi);
        }
        float max_entropy = logf((float)V);
        float norm_entropy = (max_entropy > 0.0f) ? (entropy / max_entropy) : 0.5f;
        float t_factor = powf(norm_entropy, p.dyntemp_exponent);
        eff_temp = p.temperature - p.dyntemp_range + 2.0f * p.dyntemp_range * t_factor;
        if (eff_temp < 1e-5f) eff_temp = 1e-5f;
    }

    // 6. Temperature scaling
    for (int32_t i = 0; i < V; ++i) logits[i] /= eff_temp;

    // 7. Top-k filtering
    if (p.top_k > 0 && p.top_k < V) {
        std::vector<std::pair<float,int32_t>> sc(V);
        for (int32_t i = 0; i < V; ++i) sc[i] = {logits[i], i};
        std::partial_sort(sc.begin(), sc.begin() + p.top_k, sc.end(),
            [](const std::pair<float,int32_t>&a, const std::pair<float,int32_t>&b){
                return a.first > b.first;
            });
        float thr = sc[p.top_k - 1].first;
        for (int32_t i = 0; i < V; ++i) if (logits[i] < thr) logits[i] = -INFINITY;
    }

    // 8. Top-p (nucleus) filtering
    if (p.top_p > 0.0f && p.top_p < 1.0f) {
        float mx = *std::max_element(logits.data(), logits.data() + V);
        std::vector<std::pair<float,int32_t>> sp(V);
        double sum_p = 0.0;
        for (int32_t i = 0; i < V; ++i) {
            float prob = expf(logits[i] - mx);
            sp[i] = {prob, i};
            sum_p += prob;
        }
        for (auto & x : sp) x.first /= (float)sum_p;
        std::sort(sp.begin(), sp.end(),
            [](const std::pair<float,int32_t>&a, const std::pair<float,int32_t>&b){
                return a.first > b.first;
            });
        // Walk the sorted list accumulating probability mass.
        // Once the cumulative sum reaches top_p, suppress all remaining tokens.
        // This avoids recomputing probabilities (and the float rounding mismatch
        // between the two passes that the old cutoff approach had).
        float cumul = 0.0f;
        int32_t keep_until = V; // exclusive index: tokens [0, keep_until) are kept
        for (int32_t k = 0; k < V; ++k) {
            cumul += sp[k].first;
            if (cumul >= p.top_p) {
                keep_until = k + 1; // include this token, drop everything after
                break;
            }
        }
        // Build a set of suppressed indices
        for (int32_t k = keep_until; k < V; ++k) {
            logits[sp[k].second] = -INFINITY;
        }
    }

    // 9. min_p filtering: keep tokens where prob >= min_p * max_prob
    if (p.min_p > 0.0f) {
        float mx = *std::max_element(logits.data(), logits.data() + V);
        double sum_e = 0.0;
        for (int32_t i = 0; i < V; ++i) sum_e += expf(logits[i] - mx);
        // max_prob = exp(mx - mx) / sum_e = 1 / sum_e, which is the probability
        // of the highest-logit token (the argmax always has logit == mx after
        // the shift, so its unnormalised weight is exp(0) = 1).
        float max_prob = 1.0f / (float)sum_e;
        float threshold = p.min_p * max_prob;
        for (int32_t i = 0; i < V; ++i) {
            float prob = expf(logits[i] - mx) / (float)sum_e;
            if (prob < threshold) logits[i] = -INFINITY;
        }
    }

    // 10. Softmax → sample
    float mx = *std::max_element(logits.data(), logits.data() + V);
    if (!std::isfinite(mx)) return 0;  // all suppressed, safe fallback
    double sum = 0.0;
    probs.resize(V);
    for (int32_t i = 0; i < V; ++i) {
        probs[i] = expf(logits[i] - mx);
        sum += probs[i];
    }
    for (int32_t i = 0; i < V; ++i) probs[i] = (float)(probs[i] / sum);
    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    return dist(rng);
}

// ---------------------------------------------------------------------------
// Legacy wrapper — preserves the old call signature used throughout the file.
// New code should use the struct-based overload directly.
// ---------------------------------------------------------------------------
static int32_t sample_token(
        std::vector<float> & logits,
        int32_t vocab_size,
        int32_t eos_id,
        int32_t suppress_start,
        const std::unordered_set<int32_t> & gen_tokens,
        float repetition_penalty,
        float temperature,
        int32_t top_k,
        float top_p,
        std::vector<float> & probs,
        std::mt19937 & rng) {
    sample_token_params p;
    p.vocab_size        = vocab_size;
    p.eos_id            = eos_id;
    p.suppress_start    = suppress_start;
    p.repetition_penalty = repetition_penalty;
    p.temperature       = temperature;
    p.top_k             = top_k;
    p.top_p             = top_p;
    return sample_token(logits, gen_tokens, nullptr, nullptr, probs, rng, p);
}

bool TTSTransformer::generate(const int32_t * text_tokens, int32_t n_tokens,
                               const float * speaker_embd, int32_t max_len,
                               std::vector<int32_t> & output,
                               int32_t language_id,
                               float repetition_penalty,
                               float temperature,
                               int32_t top_k,
                               float top_p,
                               float subtalker_temperature,
                               int32_t subtalker_top_k,
                               bool non_streaming_mode,
                               float subtalker_top_p) {
#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    tts_timing timing = {};
    auto t_gen_start = clk::now();
    auto t0 = t_gen_start, t1 = t_gen_start;
    timing_ = &timing;
#endif

    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
#ifdef QWEN3_TTS_TIMING
        timing_ = nullptr;
#endif
        return false;
    }
    if (!text_tokens) {
        error_msg_ = "text_tokens is null";
#ifdef QWEN3_TTS_TIMING
        timing_ = nullptr;
#endif
        return false;
    }
    if (n_tokens < 4) {
        error_msg_ = "Need at least 4 text tokens for generation";
#ifdef QWEN3_TTS_TIMING
        timing_ = nullptr;
#endif
        return false;
    }
    if (max_len <= 0) {
        output.clear();
        return true;
    }
    
    const auto & cfg = model_.config;

    // Resolve subtalker sampling parameters (-1 means inherit from main talker)
    const float sub_temp = (subtalker_temperature < 0.0f) ? temperature : subtalker_temperature;
    const int32_t sub_top_k = (subtalker_top_k < 0) ? top_k : subtalker_top_k;
    const float sub_top_p = (subtalker_top_p < 0.0f) ? top_p : subtalker_top_p;

    std::vector<float> prefill_embd;
    std::vector<float> trailing_text_hidden;
    std::vector<float> tts_pad_embed;

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!build_prefill_graph(text_tokens, n_tokens, speaker_embd, language_id,
                             prefill_embd, trailing_text_hidden, tts_pad_embed,
                             non_streaming_mode)) {
#ifdef QWEN3_TTS_TIMING
        timing_ = nullptr;
#endif
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    timing.t_prefill_build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    const int32_t prefill_len = (int32_t)(prefill_embd.size() / cfg.hidden_size);
    const int32_t trailing_len = (int32_t)(trailing_text_hidden.size() / cfg.hidden_size);

    const int32_t required_ctx = prefill_len + max_len + 8;
    if (state_.cache.n_ctx < required_ctx || state_.cache.n_ctx > std::max<int32_t>(required_ctx * 2, 512)) {
        if (!init_kv_cache(required_ctx)) {
#ifdef QWEN3_TTS_TIMING
            timing_ = nullptr;
#endif
            return false;
        }
    }
    clear_kv_cache();
    
    std::vector<float> hidden_out;
    std::vector<float> logits;

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!forward_prefill(prefill_embd.data(), prefill_len, 0, hidden_out, &logits)) {
#ifdef QWEN3_TTS_TIMING
        timing_ = nullptr;
#endif
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    timing.t_prefill_forward_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    output.clear();
    output.reserve(max_len * cfg.n_codebooks);
    
    int32_t n_past = prefill_len;
    std::vector<int32_t> frame_codes(cfg.n_codebooks);
    std::unordered_set<int32_t> generated_cb0_tokens;
    const int32_t suppress_start = cfg.codec_vocab_size - 1024;
    
    std::vector<float> probs(cfg.codec_vocab_size);
    std::vector<float> step_embd(cfg.hidden_size, 0.0f);
    std::vector<float> embd_row(cfg.hidden_size);

    // Token history and counts for extended sampling (freq/presence/DRY penalties)
    std::vector<int32_t>            cb0_token_history;
    std::unordered_map<int32_t,int32_t> cb0_token_counts;

    // Build the sampling params struct once — reused every frame
    sample_token_params stp;
    stp.vocab_size         = cfg.codec_vocab_size;
    stp.eos_id             = cfg.codec_eos_id;
    stp.suppress_start     = suppress_start;
    stp.repetition_penalty = repetition_penalty;
    stp.frequency_penalty  = ext_frequency_penalty;
    stp.presence_penalty   = ext_presence_penalty;
    stp.temperature        = temperature;
    stp.dyntemp_range      = ext_dyntemp_range;
    stp.dyntemp_exponent   = ext_dyntemp_exponent;
    stp.top_k              = top_k;
    stp.top_p              = top_p;
    stp.min_p              = ext_min_p;
    stp.dry_multiplier     = ext_dry_multiplier;
    stp.dry_base           = ext_dry_base;
    stp.dry_allowed_length = ext_dry_allowed_length;
    stp.dry_penalty_last_n = ext_dry_penalty_last_n;
    stp.min_new_tokens     = 2;

    for (int frame = 0; frame < max_len; ++frame) {
        stp.n_generated = frame;
        int32_t next_token = sample_token(
            logits, generated_cb0_tokens,
            (ext_dry_multiplier != 0.0f) ? &cb0_token_history : nullptr,
            (ext_frequency_penalty != 0.0f || ext_presence_penalty != 0.0f) ? &cb0_token_counts : nullptr,
            probs, rng_, stp);

        if (next_token == cfg.codec_eos_id) {
            break;
        }
        
        frame_codes[0] = next_token;
        generated_cb0_tokens.insert(next_token);

        // Track token history and counts for DRY / frequency / presence penalties
        cb0_token_history.push_back(next_token);
        cb0_token_counts[next_token]++;

        // Fire per-frame logits callback — caller can inspect raw CB0 logits
        // and the sampled token, and return non-zero to stop generation early.
        if (logits_cb_) {
            int stop = logits_cb_((int32_t)frame, logits.data(),
                                  (int32_t)logits.size(), next_token);
            if (stop) break;
        }
        
#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        std::vector<int32_t> codes_1_15;
        if (!predict_codes_autoregressive(last_hidden_.data(), frame_codes[0], codes_1_15, sub_temp, sub_top_k, sub_top_p)) {
#ifdef QWEN3_TTS_TIMING
            timing_ = nullptr;
#endif
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        timing.t_code_pred_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

        for (int cb = 1; cb < cfg.n_codebooks; ++cb) {
            frame_codes[cb] = codes_1_15[cb - 1];
        }
        
        for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
            output.push_back(frame_codes[cb]);
        }
        if (frame_cb_ && !frame_cb_(frame_codes.data(), cfg.n_codebooks)) {
            error_msg_ = "Complete-frame callback failed";
#ifdef QWEN3_TTS_TIMING
            timing_ = nullptr;
#endif
            return false;
        }

#ifdef QWEN3_TTS_TIMING
        timing.n_frames = frame + 1;
#endif

        if (frame + 1 >= max_len) {
            break;
        }

        std::fill(step_embd.begin(), step_embd.end(), 0.0f);

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!lookup_single_embedding_row(model_.codec_embd, frame_codes[0], embd_row.data())) {
#ifdef QWEN3_TTS_TIMING
            timing_ = nullptr;
#endif
            return false;
        }
        for (int32_t h = 0; h < cfg.hidden_size; ++h) {
            step_embd[h] = embd_row[h];
        }

        for (int cb = 1; cb < cfg.n_codebooks; ++cb) {
            int32_t code_token = frame_codes[cb];
            if (!lookup_single_embedding_row(model_.code_pred_embd[cb - 1], code_token, embd_row.data())) {
#ifdef QWEN3_TTS_TIMING
                timing_ = nullptr;
#endif
                return false;
            }
            for (int32_t h = 0; h < cfg.hidden_size; ++h) {
                step_embd[h] += embd_row[h];
            }
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        timing.t_embed_lookup_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

        const float * trailing_row = (frame < trailing_len)
            ? trailing_text_hidden.data() + (size_t)frame * cfg.hidden_size
            : tts_pad_embed.data();
        for (int32_t h = 0; h < cfg.hidden_size; ++h) {
            step_embd[h] += trailing_row[h];
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!forward_step(step_embd.data(), n_past, logits)) {
#ifdef QWEN3_TTS_TIMING
            timing_ = nullptr;
#endif
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        timing.t_talker_forward_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
        
        n_past++;
    }
    
#ifdef QWEN3_TTS_TIMING
    timing.t_generate_total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_gen_start).count();
    timing_ = nullptr;
    const auto & t = timing;
    int nf = t.n_frames;
    fprintf(stderr, "\n=== Detailed Generation Timing (%d frames) ===\n", nf);
    fprintf(stderr, "\n  Prefill:\n");
    fprintf(stderr, "    Build graph:      %8.1f ms\n", t.t_prefill_build_ms);
    fprintf(stderr, "    Forward total:    %8.1f ms\n", t.t_prefill_forward_ms);
    fprintf(stderr, "      Graph build:    %8.1f ms\n", t.t_prefill_graph_build_ms);
    fprintf(stderr, "      Graph alloc:    %8.1f ms\n", t.t_prefill_graph_alloc_ms);
    fprintf(stderr, "      Compute:        %8.1f ms\n", t.t_prefill_compute_ms);
    fprintf(stderr, "      Data I/O:       %8.1f ms\n", t.t_prefill_data_ms);
    fprintf(stderr, "\n  Talker forward_step (total / per-frame):\n");
    fprintf(stderr, "    Total:            %8.1f ms   (%.1f ms/frame)\n", t.t_talker_forward_ms, nf > 0 ? t.t_talker_forward_ms / nf : 0.0);
    fprintf(stderr, "      Graph build:    %8.1f ms   (%.1f ms/frame)\n", t.t_talker_graph_build_ms, nf > 0 ? t.t_talker_graph_build_ms / nf : 0.0);
    fprintf(stderr, "      Graph alloc:    %8.1f ms   (%.1f ms/frame)\n", t.t_talker_graph_alloc_ms, nf > 0 ? t.t_talker_graph_alloc_ms / nf : 0.0);
    fprintf(stderr, "      Compute:        %8.1f ms   (%.1f ms/frame)\n", t.t_talker_compute_ms, nf > 0 ? t.t_talker_compute_ms / nf : 0.0);
    fprintf(stderr, "      Data I/O:       %8.1f ms   (%.1f ms/frame)\n", t.t_talker_data_ms, nf > 0 ? t.t_talker_data_ms / nf : 0.0);
    fprintf(stderr, "\n  Code predictor (total / per-frame):\n");
    fprintf(stderr, "    Backend:          %s\n", use_coreml_code_predictor_ ? "CoreML (CPU+NE)" : "GGML");
    if (use_coreml_code_predictor_ && !coreml_code_predictor_path_.empty()) {
        fprintf(stderr, "    CoreML model:     %s\n", coreml_code_predictor_path_.c_str());
    }
    fprintf(stderr, "    Total:            %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_ms, nf > 0 ? t.t_code_pred_ms / nf : 0.0);
    fprintf(stderr, "      Init/KV/embed:  %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_init_ms, nf > 0 ? t.t_code_pred_init_ms / nf : 0.0);
    fprintf(stderr, "      Prefill (2tok): %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_prefill_ms, nf > 0 ? t.t_code_pred_prefill_ms / nf : 0.0);
    fprintf(stderr, "      Steps (14):     %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_steps_ms, nf > 0 ? t.t_code_pred_steps_ms / nf : 0.0);
    fprintf(stderr, "      Graph build:    %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_graph_build_ms, nf > 0 ? t.t_code_pred_graph_build_ms / nf : 0.0);
    fprintf(stderr, "      Graph alloc:    %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_graph_alloc_ms, nf > 0 ? t.t_code_pred_graph_alloc_ms / nf : 0.0);
    fprintf(stderr, "      Compute:        %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_compute_ms, nf > 0 ? t.t_code_pred_compute_ms / nf : 0.0);
    fprintf(stderr, "      Data I/O:       %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_data_ms, nf > 0 ? t.t_code_pred_data_ms / nf : 0.0);
    fprintf(stderr, "      CoreML total:   %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_coreml_ms, nf > 0 ? t.t_code_pred_coreml_ms / nf : 0.0);
    fprintf(stderr, "\n  Embed lookups:      %8.1f ms   (%.1f ms/frame)\n", t.t_embed_lookup_ms, nf > 0 ? t.t_embed_lookup_ms / nf : 0.0);
    double accounted = t.t_prefill_build_ms + t.t_prefill_forward_ms + t.t_talker_forward_ms + t.t_code_pred_ms + t.t_embed_lookup_ms;
    fprintf(stderr, "  Other/overhead:     %8.1f ms\n", t.t_generate_total_ms - accounted);
    fprintf(stderr, "  ─────────────────────────────────────────\n");
    fprintf(stderr, "  Total generate:     %8.1f ms\n", t.t_generate_total_ms);
    if (nf > 0) {
        fprintf(stderr, "  Throughput:         %8.1f ms/frame (%.1f frames/s)\n",
                t.t_generate_total_ms / nf, 1000.0 * nf / t.t_generate_total_ms);
    }
#endif

    return true;
}

// ---------------------------------------------------------------------------
// ICL prefill builder
// Matches Python's generate_icl_prompt() (both streaming and non-streaming).
//
// Python reference (modeling_qwen3_tts.py:2188-2197):
//   text_id       = input_id[:, 3:-5]       -- body text, no role/trailing tokens
//   ref_id        = ref_ids[:, 3:-2]        -- ref text body, no role tokens
//   text_embed    = proj(cat([ref_id, text_id, eos]))
//   codec_embed   = [codec_bos | sum(16×cb_embed(frame))]
//
//   non_streaming_mode:
//     icl_block = [text_embed + codec_pad | codec_embed + tts_pad]
//     trailing  = tts_pad_embed
//   streaming (text_len > codec_len):
//     icl_block = text_embed[:codec_len] + codec_embed
//     trailing  = text_embed[codec_len:]
//   streaming (codec_len >= text_len):
//     icl_block = pad(text_embed, codec_len, tts_pad) + codec_embed
//     trailing  = tts_pad_embed
//
//   talker_input_embed = [base_framing | icl_block]
//
// The key difference from the base (non-ICL) path: ALL text (ref + body + eos)
// goes into the ICL overlay. The base prefill's first_text_overlaid position
// and base_trailing are dropped — the ICL block replaces them.
// ---------------------------------------------------------------------------
bool TTSTransformer::build_prefill_graph_icl(
        const int32_t * text_tokens, int32_t n_tokens,
        const float * speaker_embd, int32_t language_id,
        const int32_t * ref_text_tokens, int32_t n_ref_text_tokens,
        const int32_t * ref_codes, int32_t n_ref_frames,
        std::vector<float> & prefill_embd,
        std::vector<float> & trailing_text_hidden,
        std::vector<float> & tts_pad_embed,
        bool non_streaming_mode) {

    const auto & cfg = model_.config;
    const int32_t H = cfg.hidden_size;

    // Build the base prefill — we only keep the framing (role + codec overlay)
    // and strip the first_text_overlaid + base_trailing since all body text
    // goes into the ICL block.
    std::vector<float> full_base_prefill;
    std::vector<float> base_trailing;
    if (!build_prefill_graph(text_tokens, n_tokens, speaker_embd, language_id,
                              full_base_prefill, base_trailing, tts_pad_embed,
                              (n_ref_text_tokens > 0 && n_ref_frames > 0)
                                  ? false : non_streaming_mode)) {
        return false;
    }

    if (n_ref_text_tokens <= 0 || n_ref_frames <= 0) {
        prefill_embd         = std::move(full_base_prefill);
        trailing_text_hidden = std::move(base_trailing);
        return true;
    }

    // 1. Strip first_text_overlaid from base prefill.
    // build_prefill_graph (streaming) produces:
    //   full_base_prefill = [role(3) | codec_overlay | first_text_overlaid(1)]
    //   base_trailing     = [body_text[1:] | eos]
    //
    // We keep only the framing: [role(3) | codec_overlay]
    const int32_t full_base_len = (int32_t)(full_base_prefill.size() / H);
    const int32_t framing_len   = std::max(0, full_base_len - 1);
    std::vector<float> base_framing;
    if (framing_len > 0) {
        base_framing.assign(full_base_prefill.begin(),
                            full_base_prefill.begin() + (size_t)framing_len * H);
    }

    // 2. Project ref text tokens
    std::vector<float> ref_text_proj;
    if (!project_text_tokens(ref_text_tokens, n_ref_text_tokens, ref_text_proj)) {
        return false;
    }

    // 3. Project body text tokens (text_tokens[3:-5] — matches Python's input_id[:, 3:-5])
    // The trailing 5 tokens are always <|im_end|> \n <|im_start|> assistant \n,
    // which encode_for_tts() appends. body_end must match that exact layout.
    const int32_t body_start = 3;
    const int32_t body_end   = std::max(body_start, n_tokens - 5);
    const int32_t body_len   = body_end - body_start;
    std::vector<float> body_proj;
    if (body_len > 0) {
        if (!project_text_tokens(text_tokens + body_start, body_len, body_proj)) {
            return false;
        }
    }

    // 4. Project tts_eos
    int32_t eos_tok = cfg.tts_eos_token_id;
    std::vector<float> eos_proj;
    if (!project_text_tokens(&eos_tok, 1, eos_proj)) {
        return false;
    }

    // 5. Combined text embed: [ref_text | body_text | eos]  (matches Python)
    const int32_t text_len = n_ref_text_tokens + body_len + 1;
    std::vector<float> text_embed((size_t)text_len * H);
    {
        size_t off = 0;
        if (n_ref_text_tokens > 0) {
            memcpy(text_embed.data(), ref_text_proj.data(),
                   (size_t)n_ref_text_tokens * H * sizeof(float));
            off += (size_t)n_ref_text_tokens;
        }
        if (body_len > 0) {
            memcpy(text_embed.data() + off * H, body_proj.data(),
                   (size_t)body_len * H * sizeof(float));
        }
        memcpy(text_embed.data() + (size_t)(n_ref_text_tokens + body_len) * H,
               eos_proj.data(), H * sizeof(float));
    }

    // 6. Build reference codec embed: [codec_bos | sum_of_all_cb_embeds per frame]
    const int32_t codec_len = 1 + n_ref_frames;
    std::vector<float> codec_embed((size_t)codec_len * H, 0.0f);
    if (!lookup_single_embedding_row(model_.codec_embd, cfg.codec_bos_id, codec_embed.data())) {
        return false;
    }
    for (int32_t f = 0; f < n_ref_frames; ++f) {
        float * dst = codec_embed.data() + (size_t)(f + 1) * H;
        if (!lookup_single_embedding_row(model_.codec_embd,
                                          ref_codes[f * cfg.n_codebooks + 0], dst)) {
            return false;
        }
        std::vector<float> row(H);
        for (int cb = 1; cb < cfg.n_codebooks; ++cb) {
            if (!lookup_single_embedding_row(model_.code_pred_embd[cb - 1],
                                              ref_codes[f * cfg.n_codebooks + cb], row.data())) {
                return false;
            }
            for (int h = 0; h < H; ++h) dst[h] += row[h];
        }
    }

    // 7. Create ICL overlay
    // Python (modeling_qwen3_tts.py:2014-2019):
    //   non_streaming_mode:
    //     icl_block = [text_embed + codec_pad | codec_embed + tts_pad]
    //     trailing = tts_pad_embed
    //   streaming (text_lens > codec_lens):
    //     icl_block = text_embed[:, :codec_lens] + codec_embed
    //     trailing  = text_embed[:, codec_lens:]
    //   streaming (codec_lens >= text_lens):
    //     text_embed = pad(text_embed, codec_lens, tts_pad)
    //     icl_block = text_embed + codec_embed
    //     trailing  = tts_pad_embed
    std::vector<float> icl_block;
    std::vector<float> icl_trailing;

    if (non_streaming_mode) {
        std::vector<float> codec_pad_row(H);
        if (!lookup_single_embedding_row(model_.codec_embd, cfg.codec_pad_id, codec_pad_row.data())) {
            return false;
        }
        const int32_t total_len = text_len + codec_len;
        icl_block.resize((size_t)total_len * H);
        for (int32_t t = 0; t < text_len; ++t) {
            const float * te = text_embed.data() + (size_t)t * H;
            float * out = icl_block.data() + (size_t)t * H;
            for (int h = 0; h < H; ++h) out[h] = te[h] + codec_pad_row[h];
        }
        for (int32_t t = 0; t < codec_len; ++t) {
            const float * ce = codec_embed.data() + (size_t)t * H;
            float * out = icl_block.data() + (size_t)(text_len + t) * H;
            for (int h = 0; h < H; ++h) out[h] = ce[h] + tts_pad_embed.data()[h];
        }
        icl_trailing = tts_pad_embed;
    } else if (text_len > codec_len) {
        icl_block.resize((size_t)codec_len * H);
        for (int32_t t = 0; t < codec_len; ++t) {
            const float * te = text_embed.data()  + (size_t)t * H;
            const float * ce = codec_embed.data() + (size_t)t * H;
            float       * out = icl_block.data()  + (size_t)t * H;
            for (int h = 0; h < H; ++h) out[h] = te[h] + ce[h];
        }
        int32_t trail = text_len - codec_len;
        icl_trailing.resize((size_t)trail * H);
        memcpy(icl_trailing.data(),
               text_embed.data() + (size_t)codec_len * H,
               (size_t)trail * H * sizeof(float));
    } else {
        icl_block.resize((size_t)codec_len * H);
        memcpy(icl_block.data(), text_embed.data(), (size_t)text_len * H * sizeof(float));
        for (int32_t t = text_len; t < codec_len; ++t) {
            memcpy(icl_block.data() + (size_t)t * H, tts_pad_embed.data(), H * sizeof(float));
        }
        for (int32_t t = 0; t < codec_len; ++t) {
            float * out = icl_block.data() + (size_t)t * H;
            const float * ce = codec_embed.data() + (size_t)t * H;
            for (int h = 0; h < H; ++h) out[h] += ce[h];
        }
        icl_trailing = tts_pad_embed;
    }

    // 8. Final prefill = [base_framing | icl_block]
    const int32_t icl_block_len = (int32_t)(icl_block.size() / H);
    prefill_embd.resize((size_t)(framing_len + icl_block_len) * H);
    if (!base_framing.empty()) {
        memcpy(prefill_embd.data(),
               base_framing.data(), base_framing.size() * sizeof(float));
    }
    memcpy(prefill_embd.data() + base_framing.size(),
           icl_block.data(), icl_block.size() * sizeof(float));

    // Trailing = icl_trailing only (body text is in the ICL block, not base_trailing)
    trailing_text_hidden = std::move(icl_trailing);

    return true;
}

// ---------------------------------------------------------------------------
// Instruct prefill builder (VoiceDesign / CustomVoice)
// Prepends text_projection(instruct_tokens) before the base prefill
// ---------------------------------------------------------------------------
bool TTSTransformer::build_prefill_graph_instruct(
        const int32_t * text_tokens, int32_t n_tokens,
        const float * speaker_embd, int32_t language_id,
        const int32_t * instruct_tokens, int32_t n_instruct_tokens,
        std::vector<float> & prefill_embd,
        std::vector<float> & trailing_text_hidden,
        std::vector<float> & tts_pad_embed,
        bool non_streaming_mode) {

    const auto & cfg = model_.config;
    const int32_t H  = cfg.hidden_size;

    std::vector<float> base_prefill;
    std::vector<float> base_trailing;
    if (!build_prefill_graph(text_tokens, n_tokens, speaker_embd, language_id,
                              base_prefill, base_trailing, tts_pad_embed,
                              non_streaming_mode)) {
        return false;
    }

    if (n_instruct_tokens <= 0 || instruct_tokens == nullptr) {
        prefill_embd        = std::move(base_prefill);
        trailing_text_hidden = std::move(base_trailing);
        return true;
    }

    std::vector<float> instruct_proj;
    if (!project_text_tokens(instruct_tokens, n_instruct_tokens, instruct_proj)) {
        return false;
    }

    const int32_t base_len = (int32_t)(base_prefill.size() / H);
    prefill_embd.resize((size_t)(n_instruct_tokens + base_len) * H);
    memcpy(prefill_embd.data(),
           instruct_proj.data(), instruct_proj.size() * sizeof(float));
    memcpy(prefill_embd.data() + instruct_proj.size(),
           base_prefill.data(),  base_prefill.size()  * sizeof(float));

    trailing_text_hidden = std::move(base_trailing);
    return true;
}

// ---------------------------------------------------------------------------
// generate_icl: voice clone with full ICL (reference audio codes + transcript)
// ---------------------------------------------------------------------------
bool TTSTransformer::generate_icl(
        const int32_t * text_tokens, int32_t n_tokens,
        const float * speaker_embd,
        const int32_t * ref_text_tokens, int32_t n_ref_text_tokens,
        const int32_t * ref_codes,       int32_t n_ref_frames,
        int32_t max_len,
        std::vector<int32_t> & output,
        int32_t language_id,
        float   repetition_penalty,
        float   temperature,
        int32_t top_k,
        float   top_p,
        float   subtalker_temperature,
        int32_t subtalker_top_k,
        bool    non_streaming_mode,
        float   subtalker_top_p) {

    if (!model_.ctx) { error_msg_ = "Model not loaded"; return false; }
    if (!text_tokens || n_tokens < 4) {
        error_msg_ = "Need at least 4 text tokens";
        return false;
    }
    if (max_len <= 0) { output.clear(); return true; }

    const auto & cfg     = model_.config;
    const float  sub_temp  = (subtalker_temperature < 0.0f) ? temperature  : subtalker_temperature;
    const int32_t sub_topk = (subtalker_top_k       < 0)    ? top_k        : subtalker_top_k;
    const float  sub_topp  = (subtalker_top_p        < 0.0f) ? top_p       : subtalker_top_p;

    std::vector<float> prefill_embd, trailing_text_hidden, tts_pad_embed;

    if (!build_prefill_graph_icl(text_tokens, n_tokens, speaker_embd, language_id,
                                  ref_text_tokens, n_ref_text_tokens,
                                  ref_codes, n_ref_frames,
                                  prefill_embd, trailing_text_hidden, tts_pad_embed,
                                  non_streaming_mode)) {
        return false;
    }

    const int32_t prefill_len  = (int32_t)(prefill_embd.size()        / cfg.hidden_size);
    const int32_t trailing_len = (int32_t)(trailing_text_hidden.size() / cfg.hidden_size);

    const int32_t required_ctx = prefill_len + max_len + 8;
    if (state_.cache.n_ctx < required_ctx ||
        state_.cache.n_ctx > std::max<int32_t>(required_ctx * 2, 512)) {
        if (!init_kv_cache(required_ctx)) return false;
    }
    clear_kv_cache();

    std::vector<float> hidden_out, logits;
    if (!forward_prefill(prefill_embd.data(), prefill_len, 0, hidden_out, &logits)) {
        return false;
    }

    output.clear();
    output.reserve(max_len * cfg.n_codebooks);

    int32_t n_past = prefill_len;
    std::vector<int32_t> frame_codes(cfg.n_codebooks);
    std::unordered_set<int32_t> generated_cb0_tokens;
    const int32_t suppress_start = cfg.codec_vocab_size - 1024;
    std::vector<float> probs(cfg.codec_vocab_size);
    std::vector<float> step_embd(cfg.hidden_size, 0.0f);
    std::vector<float> embd_row(cfg.hidden_size);

    // Token history and counts for extended sampling (freq/presence/DRY penalties)
    std::vector<int32_t>            cb0_token_history;
    std::unordered_map<int32_t,int32_t> cb0_token_counts;

    // Build the sampling params struct once — reused every frame
    sample_token_params stp;
    stp.vocab_size         = cfg.codec_vocab_size;
    stp.eos_id             = cfg.codec_eos_id;
    stp.suppress_start     = suppress_start;
    stp.repetition_penalty = repetition_penalty;
    stp.frequency_penalty  = ext_frequency_penalty;
    stp.presence_penalty   = ext_presence_penalty;
    stp.temperature        = temperature;
    stp.dyntemp_range      = ext_dyntemp_range;
    stp.dyntemp_exponent   = ext_dyntemp_exponent;
    stp.top_k              = top_k;
    stp.top_p              = top_p;
    stp.min_p              = ext_min_p;
    stp.dry_multiplier     = ext_dry_multiplier;
    stp.dry_base           = ext_dry_base;
    stp.dry_allowed_length = ext_dry_allowed_length;
    stp.dry_penalty_last_n = ext_dry_penalty_last_n;
    stp.min_new_tokens     = 2;

    for (int frame = 0; frame < max_len; ++frame) {
        stp.n_generated = frame;
        int32_t next_token = sample_token(
            logits, generated_cb0_tokens,
            (ext_dry_multiplier != 0.0f) ? &cb0_token_history : nullptr,
            (ext_frequency_penalty != 0.0f || ext_presence_penalty != 0.0f) ? &cb0_token_counts : nullptr,
            probs, rng_, stp);

        if (next_token == cfg.codec_eos_id) break;
        frame_codes[0] = next_token;
        generated_cb0_tokens.insert(next_token);
        cb0_token_history.push_back(next_token);
        cb0_token_counts[next_token]++;

        // Fire per-frame logits callback
        if (logits_cb_) {
            int stop = logits_cb_((int32_t)frame, logits.data(),
                                  (int32_t)logits.size(), next_token);
            if (stop) break;
        }

        std::vector<int32_t> codes_1_15;
        if (!predict_codes_autoregressive(last_hidden_.data(), frame_codes[0],
                                            codes_1_15, sub_temp, sub_topk, sub_topp)) {
            return false;
        }
        for (int cb = 1; cb < cfg.n_codebooks; ++cb) frame_codes[cb] = codes_1_15[cb - 1];
        for (int cb = 0; cb < cfg.n_codebooks; ++cb) output.push_back(frame_codes[cb]);
        if (frame_cb_ && !frame_cb_(frame_codes.data(), cfg.n_codebooks)) {
            error_msg_ = "Complete-frame callback failed";
#ifdef QWEN3_TTS_TIMING
            timing_ = nullptr;
#endif
            return false;
        }

        if (frame + 1 >= max_len) break;

        std::fill(step_embd.begin(), step_embd.end(), 0.0f);
        if (!lookup_single_embedding_row(model_.codec_embd, frame_codes[0], embd_row.data())) return false;
        for (int32_t h = 0; h < cfg.hidden_size; ++h) step_embd[h] = embd_row[h];
        for (int cb = 1; cb < cfg.n_codebooks; ++cb) {
            if (!lookup_single_embedding_row(model_.code_pred_embd[cb-1], frame_codes[cb], embd_row.data())) return false;
            for (int32_t h = 0; h < cfg.hidden_size; ++h) step_embd[h] += embd_row[h];
        }

        const float * trailing_row = (frame < trailing_len)
            ? trailing_text_hidden.data() + (size_t)frame * cfg.hidden_size
            : tts_pad_embed.data();
        for (int32_t h = 0; h < cfg.hidden_size; ++h) step_embd[h] += trailing_row[h];

        if (!forward_step(step_embd.data(), n_past, logits)) return false;
        n_past++;
    }

    return true;
}

// ---------------------------------------------------------------------------
// generate_from_prefill: shared inner generation loop
// Used by generate_icl() and the instruct path in qwen3_tts.cpp.
// ---------------------------------------------------------------------------
bool TTSTransformer::generate_from_prefill(
        const std::vector<float> & prefill_embd,
        const std::vector<float> & trailing_text_hidden,
        const std::vector<float> & tts_pad_embed,
        int32_t max_len,
        std::vector<int32_t> & output,
        float repetition_penalty,
        float temperature,
        int32_t top_k,
        float top_p,
        float subtalker_temperature,
        int32_t subtalker_top_k,
        float subtalker_top_p) {

    if (!model_.ctx) { error_msg_ = "Model not loaded"; return false; }
    if (max_len <= 0) { output.clear(); return true; }

    const auto & cfg      = model_.config;
    const int32_t H       = cfg.hidden_size;
    const float  sub_temp  = (subtalker_temperature < 0.0f) ? temperature : subtalker_temperature;
    const int32_t sub_topk = (subtalker_top_k < 0) ? top_k : subtalker_top_k;
    const float  sub_topp  = (subtalker_top_p < 0.0f) ? top_p : subtalker_top_p;

    const int32_t prefill_len  = (int32_t)(prefill_embd.size()        / H);
    const int32_t trailing_len = (int32_t)(trailing_text_hidden.size() / H);

    if (prefill_len <= 0) { error_msg_ = "Empty prefill embedding"; return false; }

    const int32_t required_ctx = prefill_len + max_len + 8;
    if (state_.cache.n_ctx < required_ctx ||
        state_.cache.n_ctx > std::max<int32_t>(required_ctx * 2, 512)) {
        if (!init_kv_cache(required_ctx)) return false;
    }
    clear_kv_cache();

    std::vector<float> hidden_out, logits;
    if (!forward_prefill(prefill_embd.data(), prefill_len, 0, hidden_out, &logits)) {
        return false;
    }

    output.clear();
    output.reserve(max_len * cfg.n_codebooks);

    int32_t n_past = prefill_len;
    std::vector<int32_t> frame_codes(cfg.n_codebooks);
    std::unordered_set<int32_t> generated_cb0_tokens;
    const int32_t suppress_start = cfg.codec_vocab_size - 1024;
    std::vector<float> probs(cfg.codec_vocab_size);
    std::vector<float> step_embd(H, 0.0f);
    std::vector<float> embd_row(H);

    // Token history and counts for extended sampling
    std::vector<int32_t>            cb0_token_history;
    std::unordered_map<int32_t,int32_t> cb0_token_counts;

    sample_token_params stp;
    stp.vocab_size         = cfg.codec_vocab_size;
    stp.eos_id             = cfg.codec_eos_id;
    stp.suppress_start     = suppress_start;
    stp.repetition_penalty = repetition_penalty;
    stp.frequency_penalty  = ext_frequency_penalty;
    stp.presence_penalty   = ext_presence_penalty;
    stp.temperature        = temperature;
    stp.dyntemp_range      = ext_dyntemp_range;
    stp.dyntemp_exponent   = ext_dyntemp_exponent;
    stp.top_k              = top_k;
    stp.top_p              = top_p;
    stp.min_p              = ext_min_p;
    stp.dry_multiplier     = ext_dry_multiplier;
    stp.dry_base           = ext_dry_base;
    stp.dry_allowed_length = ext_dry_allowed_length;
    stp.dry_penalty_last_n = ext_dry_penalty_last_n;
    stp.min_new_tokens     = 2;

    for (int frame = 0; frame < max_len; ++frame) {
        stp.n_generated = frame;
        int32_t next_token = sample_token(
            logits, generated_cb0_tokens,
            (ext_dry_multiplier != 0.0f) ? &cb0_token_history : nullptr,
            (ext_frequency_penalty != 0.0f || ext_presence_penalty != 0.0f) ? &cb0_token_counts : nullptr,
            probs, rng_, stp);

        if (next_token == cfg.codec_eos_id) break;
        frame_codes[0] = next_token;
        generated_cb0_tokens.insert(next_token);
        cb0_token_history.push_back(next_token);
        cb0_token_counts[next_token]++;

        // Fire per-frame logits callback
        if (logits_cb_) {
            int stop = logits_cb_((int32_t)frame, logits.data(),
                                  (int32_t)logits.size(), next_token);
            if (stop) break;
        }

        // Predict CB1-15
        std::vector<int32_t> codes_1_15;
        if (!predict_codes_autoregressive(last_hidden_.data(), frame_codes[0],
                                           codes_1_15, sub_temp, sub_topk, sub_topp)) {
            return false;
        }
        for (int cb = 1; cb < cfg.n_codebooks; ++cb) frame_codes[cb] = codes_1_15[cb-1];
        for (int cb = 0; cb < cfg.n_codebooks; ++cb) output.push_back(frame_codes[cb]);
        if (frame_cb_ && !frame_cb_(frame_codes.data(), cfg.n_codebooks)) {
            error_msg_ = "Complete-frame callback failed";
#ifdef QWEN3_TTS_TIMING
            timing_ = nullptr;
#endif
            return false;
        }

        if (frame + 1 >= max_len) break;

        // Build next step embedding
        std::fill(step_embd.begin(), step_embd.end(), 0.0f);
        if (!lookup_single_embedding_row(model_.codec_embd, frame_codes[0], embd_row.data())) return false;
        for (int32_t h = 0; h < H; ++h) step_embd[h] = embd_row[h];
        for (int cb = 1; cb < cfg.n_codebooks; ++cb) {
            if (!lookup_single_embedding_row(model_.code_pred_embd[cb-1], frame_codes[cb], embd_row.data())) return false;
            for (int32_t h = 0; h < H; ++h) step_embd[h] += embd_row[h];
        }
        const float * trailing_row = (frame < trailing_len)
            ? trailing_text_hidden.data() + (size_t)frame * H
            : tts_pad_embed.data();
        for (int32_t h = 0; h < H; ++h) step_embd[h] += trailing_row[h];

        if (!forward_step(step_embd.data(), n_past, logits)) return false;
        n_past++;
    }
    return true;
}

bool TTSTransformer::generate_batch(
    const int32_t * const * text_tokens,
    const int32_t * n_tokens_per_batch,
    const float * const * speaker_embds,
    int32_t n_batch,
    int32_t max_len,
    std::vector<std::vector<int32_t>> & outputs,
    const int32_t * language_ids,
    const int32_t * const * instruct_tokens,
    const int32_t * n_instruct_tokens,
    float repetition_penalty,
    float temperature,
    int32_t top_k,
    float top_p,
    float subtalker_temperature,
    int32_t subtalker_top_k,
    float subtalker_top_p,
    bool non_streaming_mode) {

    if (!model_.ctx) { error_msg_ = "Model not loaded"; return false; }
    if (n_batch <= 0) { return true; }
    if (max_len <= 0) {
        outputs.assign(n_batch, std::vector<int32_t>());
        return true;
    }

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    tts_timing timing = {};
    auto t_gen_start = clk::now();
    auto t0 = t_gen_start, t1 = t_gen_start;
    timing_ = &timing;
#endif

    const auto & cfg = model_.config;
    const int32_t H = cfg.hidden_size;

    // ---- 1. Build prefill embeddings for each batch entry ----
    struct BatchEntry {
        std::vector<float> prefill_embd;
        std::vector<float> trailing_text_hidden;
        std::vector<float> tts_pad_embed;
        int32_t prefill_len = 0;
        int32_t trailing_len = 0;
        int32_t language_id = 2050;
    };
    std::vector<BatchEntry> entries(n_batch);

    int32_t max_prefill_len = 0;
    int32_t max_trailing_len = 0;

    for (int32_t b = 0; b < n_batch; ++b) {
#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        const int32_t lang_id = (language_ids) ? language_ids[b] : 2050;
        entries[b].language_id = lang_id;

        // Use instruct prefill builder when instruct tokens are provided
        const int32_t * inst_tok = (instruct_tokens) ? instruct_tokens[b] : nullptr;
        const int32_t   inst_n  = (instruct_tokens && n_instruct_tokens) ? n_instruct_tokens[b] : 0;

        if (inst_tok && inst_n > 0) {
            if (!build_prefill_graph_instruct(text_tokens[b], n_tokens_per_batch[b],
                                               speaker_embds[b], lang_id,
                                               inst_tok, inst_n,
                                               entries[b].prefill_embd,
                                               entries[b].trailing_text_hidden,
                                               entries[b].tts_pad_embed,
                                               non_streaming_mode)) {
                error_msg_ = "Batch entry " + std::to_string(b) + " instruct prefill failed: " + error_msg_;
#ifdef QWEN3_TTS_TIMING
                timing_ = nullptr;
#endif
                return false;
            }
        } else {
            if (!build_prefill_graph(text_tokens[b], n_tokens_per_batch[b],
                                     speaker_embds[b], lang_id,
                                     entries[b].prefill_embd,
                                     entries[b].trailing_text_hidden,
                                     entries[b].tts_pad_embed,
                                     non_streaming_mode)) {
                error_msg_ = "Batch entry " + std::to_string(b) + " prefill build failed: " + error_msg_;
#ifdef QWEN3_TTS_TIMING
                timing_ = nullptr;
#endif
                return false;
            }
        }

        entries[b].prefill_len  = (int32_t)(entries[b].prefill_embd.size() / H);
        entries[b].trailing_len = (int32_t)(entries[b].trailing_text_hidden.size() / H);
        if (entries[b].prefill_len > max_prefill_len)  max_prefill_len  = entries[b].prefill_len;
        if (entries[b].trailing_len > max_trailing_len) max_trailing_len = entries[b].trailing_len;
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_prefill_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    }

    // ---- 2. Initialize KV cache with batch dimension ----
    const int32_t required_ctx = max_prefill_len + max_len + 8;
    if (state_.cache.n_ctx < required_ctx ||
        state_.cache.n_ctx > std::max<int32_t>(required_ctx * 2, 512) ||
        state_.cache.n_batch < n_batch) {
        if (!init_kv_cache(required_ctx, n_batch)) {
#ifdef QWEN3_TTS_TIMING
            timing_ = nullptr;
#endif
            return false;
        }
    }
    clear_kv_cache();

    // ---- 3. Run prefills for all batch entries ----
    std::vector<int32_t> n_past(n_batch, 0);
    std::vector<bool> eos_reached(n_batch, false);
    std::vector<std::vector<float>> cached_logits(n_batch);
    std::vector<std::vector<float>> batch_hidden(n_batch);   // per-batch hidden state
    std::vector<int32_t> frame_count(n_batch, 0);

    for (int32_t b = 0; b < n_batch; ++b) {
#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        std::vector<float> hidden_out;
        std::vector<float> logits;
        if (!forward_prefill(entries[b].prefill_embd.data(), entries[b].prefill_len,
                             0, hidden_out, &logits, b)) {
            error_msg_ = "Batch entry " + std::to_string(b) + " prefill failed: " + error_msg_;
#ifdef QWEN3_TTS_TIMING
            timing_ = nullptr;
#endif
            return false;
        }
        n_past[b] = entries[b].prefill_len;
        cached_logits[b] = std::move(logits);
        // Store last hidden from the last prefill token
        if (entries[b].prefill_len > 0 && hidden_out.size() >= (size_t)entries[b].prefill_len * H) {
            batch_hidden[b].resize(H);
            memcpy(batch_hidden[b].data(),
                   hidden_out.data() + (size_t)(entries[b].prefill_len - 1) * H,
                   H * sizeof(float));
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_prefill_forward_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    }

    // ---- 4. Resolve subtalker params ----
    const float sub_temp  = (subtalker_temperature < 0.0f) ? temperature  : subtalker_temperature;
    const int32_t sub_topk = (subtalker_top_k < 0) ? top_k : subtalker_top_k;
    const float  sub_topp  = (subtalker_top_p < 0.0f) ? top_p : subtalker_top_p;

    // ---- 5. Resize outputs ----
    outputs.resize(n_batch);
    for (int32_t b = 0; b < n_batch; ++b) {
        outputs[b].clear();
        outputs[b].reserve(max_len * cfg.n_codebooks);
    }

    // ---- 6. Sampling state ----
    const int32_t suppress_start = cfg.codec_vocab_size - 1024;
    std::vector<float> probs(cfg.codec_vocab_size);
    std::vector<float> step_embd(H, 0.0f);
    std::vector<float> embd_row(H);
    std::vector<int32_t> frame_codes(cfg.n_codebooks);

    // Extended sampling state (per batch entry)
    std::vector<std::unordered_set<int32_t>> batch_gen_tokens(n_batch);
    std::vector<std::vector<int32_t>> batch_token_history(n_batch);
    std::vector<std::unordered_map<int32_t,int32_t>> batch_token_counts(n_batch);

    // Build the sampling params struct once — reused every frame for all entries
    sample_token_params stp;
    stp.vocab_size         = cfg.codec_vocab_size;
    stp.eos_id             = cfg.codec_eos_id;
    stp.suppress_start     = suppress_start;
    stp.repetition_penalty = repetition_penalty;
    stp.frequency_penalty  = ext_frequency_penalty;
    stp.presence_penalty   = ext_presence_penalty;
    stp.temperature        = temperature;
    stp.dyntemp_range      = ext_dyntemp_range;
    stp.dyntemp_exponent   = ext_dyntemp_exponent;
    stp.top_k              = top_k;
    stp.top_p              = top_p;
    stp.min_p              = ext_min_p;
    stp.dry_multiplier     = ext_dry_multiplier;
    stp.dry_base           = ext_dry_base;
    stp.dry_allowed_length = ext_dry_allowed_length;
    stp.dry_penalty_last_n = ext_dry_penalty_last_n;
    stp.min_new_tokens     = 2;

    // ---- 7. Main frame loop ----
    for (int32_t frame = 0; frame < max_len; ++frame) {
        bool any_active = false;
        for (int32_t b = 0; b < n_batch; ++b) {
            if (eos_reached[b]) continue;
            any_active = true;

            std::vector<float> & logits = cached_logits[b];

            // Sample CB0 with extended sampling (same gen_tokens tracking as generate())
            stp.n_generated = frame_count[b];
            int32_t next_token = sample_token(
                logits, batch_gen_tokens[b],
                (ext_dry_multiplier != 0.0f) ? &batch_token_history[b] : nullptr,
                (ext_frequency_penalty != 0.0f || ext_presence_penalty != 0.0f) ? &batch_token_counts[b] : nullptr,
                probs, rng_, stp);

            if (next_token == cfg.codec_eos_id) {
                eos_reached[b] = true;
                continue;
            }

            frame_codes[0] = next_token;
            frame_count[b]++;

            // Track token history and counts for extended sampling
            batch_gen_tokens[b].insert(next_token);
            batch_token_history[b].push_back(next_token);
            batch_token_counts[b][next_token]++;

            // Fire per-frame logits callback
            if (logits_cb_) {
                int stop = logits_cb_((int32_t)(frame_count[b] - 1), logits.data(),
                                      (int32_t)logits.size(), next_token);
                if (stop) {
                    eos_reached[b] = true;
                    continue;
                }
            }

            // Run code predictor for CB1-15 using this batch's hidden state
#ifdef QWEN3_TTS_TIMING
            t0 = clk::now();
#endif
            std::vector<int32_t> codes_1_15;
            if (!predict_codes_autoregressive(batch_hidden[b].data(), frame_codes[0],
                                               codes_1_15, sub_temp, sub_topk, sub_topp, b)) {
                error_msg_ = "Batch entry " + std::to_string(b) + " code predictor failed: " + error_msg_;
#ifdef QWEN3_TTS_TIMING
                timing_ = nullptr;
#endif
                return false;
            }
#ifdef QWEN3_TTS_TIMING
            t1 = clk::now();
            if (timing_) timing_->t_code_pred_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

            for (int32_t cb = 1; cb < cfg.n_codebooks; ++cb) {
                frame_codes[cb] = codes_1_15[cb - 1];
            }
            for (int32_t cb = 0; cb < cfg.n_codebooks; ++cb) {
                outputs[b].push_back(frame_codes[cb]);
            }

            if (frame + 1 >= max_len) {
                eos_reached[b] = true;
                continue;
            }

            // Build step embedding
#ifdef QWEN3_TTS_TIMING
            t0 = clk::now();
#endif
            std::fill(step_embd.begin(), step_embd.end(), 0.0f);
            if (!lookup_single_embedding_row(model_.codec_embd, frame_codes[0], embd_row.data())) {
#ifdef QWEN3_TTS_TIMING
                timing_ = nullptr;
#endif
                return false;
            }
            for (int32_t h = 0; h < H; ++h) step_embd[h] = embd_row[h];
            for (int32_t cb = 1; cb < cfg.n_codebooks; ++cb) {
                if (!lookup_single_embedding_row(model_.code_pred_embd[cb - 1], frame_codes[cb], embd_row.data())) {
#ifdef QWEN3_TTS_TIMING
                    timing_ = nullptr;
#endif
                    return false;
                }
                for (int32_t h = 0; h < H; ++h) step_embd[h] += embd_row[h];
            }

            // Add trailing text hidden or tts_pad
            const int32_t trail_idx = frame_count[b] - 1;
            const float * trailing_row = (trail_idx < entries[b].trailing_len)
                ? entries[b].trailing_text_hidden.data() + (size_t)trail_idx * H
                : entries[b].tts_pad_embed.data();
            for (int32_t h = 0; h < H; ++h) step_embd[h] += trailing_row[h];

#ifdef QWEN3_TTS_TIMING
            t1 = clk::now();
            if (timing_) timing_->t_embed_lookup_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            t0 = clk::now();
#endif
            // Forward step — capture hidden state per batch
            if (!forward_step(step_embd.data(), n_past[b], logits, &batch_hidden[b], b)) {
                error_msg_ = "Batch entry " + std::to_string(b) + " step failed: " + error_msg_;
#ifdef QWEN3_TTS_TIMING
                timing_ = nullptr;
#endif
                return false;
            }
#ifdef QWEN3_TTS_TIMING
            t1 = clk::now();
            if (timing_) timing_->t_talker_forward_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
            n_past[b]++;
        }

        if (!any_active) break;
    }

#ifdef QWEN3_TTS_TIMING
    timing.t_generate_total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_gen_start).count();
    int32_t total_frames = 0;
    for (int32_t b = 0; b < n_batch; ++b) total_frames += frame_count[b];
    timing.n_frames = total_frames;
    timing_ = nullptr;
    const auto & t = timing;
    int nf = t.n_frames;
    fprintf(stderr, "\n=== Batch Generation Timing (%d entries, %d total frames) ===\n", n_batch, nf);
    fprintf(stderr, "  Prefill build:       %8.1f ms\n", t.t_prefill_build_ms);
    fprintf(stderr, "  Prefill forward:     %8.1f ms\n", t.t_prefill_forward_ms);
    fprintf(stderr, "  Code predictor:      %8.1f ms   (%.1f ms/total-frame)\n", t.t_code_pred_ms, nf > 0 ? t.t_code_pred_ms / nf : 0.0);
    fprintf(stderr, "  Talker forward_step: %8.1f ms   (%.1f ms/total-frame)\n", t.t_talker_forward_ms, nf > 0 ? t.t_talker_forward_ms / nf : 0.0);
    fprintf(stderr, "  Embed lookups:       %8.1f ms\n", t.t_embed_lookup_ms);
    double accounted = t.t_prefill_build_ms + t.t_prefill_forward_ms + t.t_code_pred_ms + t.t_talker_forward_ms + t.t_embed_lookup_ms;
    fprintf(stderr, "  Other/overhead:      %8.1f ms\n", t.t_generate_total_ms - accounted);
    fprintf(stderr, "  ─────────────────────────────────────────\n");
    fprintf(stderr, "  Total generate:      %8.1f ms\n", t.t_generate_total_ms);
    if (nf > 0) {
        fprintf(stderr, "  Throughput:          %8.1f ms/total-frame (%.1f total-frames/s)\n",
                t.t_generate_total_ms / nf, 1000.0 * nf / t.t_generate_total_ms);
    }
#endif

    return true;
}

void free_transformer_model(tts_transformer_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
    model.layers.clear();
    model.code_pred_layers.clear();
    model.code_pred_embd.clear();
    model.code_pred_head.clear();
    model.code_pred_proj = nullptr;
    model.code_pred_proj_bias = nullptr;
    model.host_embd_raw.clear();
}

void free_tts_kv_cache(tts_kv_cache & cache) {
    if (cache.buffer) {
        ggml_backend_buffer_free(cache.buffer);
        cache.buffer = nullptr;
    }
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
    cache.k_cache.clear();
    cache.v_cache.clear();
    cache.n_ctx = 0;
    cache.n_used = 0;
    cache.n_batch = 1;
}

} // namespace qwen3_tts

