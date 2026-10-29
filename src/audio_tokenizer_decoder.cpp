#include "audio_tokenizer_decoder.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

#define QWEN3_TTS_DEC_MAX_NODES 32768
#define QWEN3_TTS_MAX_DEC_MAX_NODES 131072

namespace qwen3_tts {

AudioTokenizerDecoder::AudioTokenizerDecoder() = default;

AudioTokenizerDecoder::~AudioTokenizerDecoder() {
    unload_model();
}

void AudioTokenizerDecoder::unload_model() {
    free_audio_decoder_model(model_);
    if (state_.max_buffer) ggml_backend_buffer_free(state_.max_buffer);
    if (state_.max_ctx) ggml_free(state_.max_ctx);
    state_.max_buffer = nullptr;
    state_.max_ctx = nullptr;
    state_.max_taps.clear();
    state_.max_snake.clear();
    state_.max_codebooks.clear();
    state_.max_graph = false;
    
    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend) {
        release_preferred_backend(state_.backend);
        state_.backend = nullptr;
    }
    if (state_.backend_cpu) {
        ggml_backend_free(state_.backend_cpu);
        state_.backend_cpu = nullptr;
    }
    if (state_.threadpool) {
        ggml_threadpool_free(state_.threadpool);
        state_.threadpool = nullptr;
        state_.threadpool_n = 0;
    }

    state_.compute_meta.clear();
}

void AudioTokenizerDecoder::set_n_threads(int32_t n_threads) {
    if (n_threads <= 0) return;
    ggml_backend_t cpu_backend = state_.backend_cpu ? state_.backend_cpu : state_.backend;
    if (cpu_backend && ggml_backend_is_cpu(cpu_backend)) {
        ggml_backend_cpu_set_n_threads(cpu_backend, n_threads);
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

void AudioTokenizerDecoder::normalize_codebooks() {
    const float epsilon = 1e-5f;
    
    // normalize_codebooks() is called before the GPU backend is initialized.
    // At this point tensors are in CPU-accessible memory (loaded by
    // load_tensor_data_from_file before backend init), so direct ->data
    // access is always valid here. After normalization the caller is
    // responsible for syncing the modified host data back to the GPU backend
    // via ggml_backend_tensor_set (see upload_if_present in load_model).
    auto normalize_codebook = [epsilon](struct ggml_tensor * codebook, struct ggml_tensor * usage, const char *) {
        if (!codebook || !usage) return;
        if (!codebook->buffer && !codebook->data) return;
        if (!usage->buffer && !usage->data) return;
        
        int64_t codebook_dim = codebook->ne[0];
        int64_t codebook_size = codebook->ne[1];
        
        std::vector<float> usage_data(codebook_size);
        ggml_backend_tensor_get(usage, usage_data.data(), 0, codebook_size * sizeof(float));
        
        if (codebook->type == GGML_TYPE_F32) {
            std::vector<float> cb_data((size_t)codebook_dim * codebook_size);
            ggml_backend_tensor_get(codebook, cb_data.data(), 0, cb_data.size() * sizeof(float));
            for (int64_t emb_idx = 0; emb_idx < codebook_size; ++emb_idx) {
                float u = usage_data[emb_idx];
                if (u < epsilon) u = epsilon;
                float inv_u = 1.0f / u;
                for (int64_t dim_idx = 0; dim_idx < codebook_dim; ++dim_idx) {
                    int64_t mem_idx = dim_idx + emb_idx * codebook_dim;
                    cb_data[mem_idx] *= inv_u;
                }
            }
            ggml_backend_tensor_set(codebook, cb_data.data(), 0, cb_data.size() * sizeof(float));
        } else {
            // F16 (default) and any other type treated as F16
            std::vector<ggml_fp16_t> cb_data((size_t)codebook_dim * codebook_size);
            ggml_backend_tensor_get(codebook, cb_data.data(), 0, cb_data.size() * sizeof(ggml_fp16_t));
            for (int64_t emb_idx = 0; emb_idx < codebook_size; ++emb_idx) {
                float u = usage_data[emb_idx];
                if (u < epsilon) u = epsilon;
                float inv_u = 1.0f / u;
                for (int64_t dim_idx = 0; dim_idx < codebook_dim; ++dim_idx) {
                    int64_t mem_idx = dim_idx + emb_idx * codebook_dim;
                    float val = ggml_fp16_to_fp32(cb_data[mem_idx]);
                    cb_data[mem_idx] = ggml_fp32_to_fp16(val * inv_u);
                }
            }
            ggml_backend_tensor_set(codebook, cb_data.data(), 0, cb_data.size() * sizeof(ggml_fp16_t));
        }
    };
    
    normalize_codebook(model_.vq_first_codebook, model_.vq_first_usage, "first");
    
    for (int i = 0; i < 15; ++i) {
        char name[16];
        snprintf(name, sizeof(name), "rest%d", i);
        normalize_codebook(model_.vq_rest_codebook[i], model_.vq_rest_usage[i], name);
    }
}

bool AudioTokenizerDecoder::load_model(const std::string & model_path) {
    unload_model();

    GGUFLoader loader;
    if (!loader.open(model_path)) {
        error_msg_ = loader.get_error();
        return false;
    }
    
    model_.config.sample_rate = loader.get_u32("qwen3-tts-tokenizer.sample_rate",
                                loader.get_u32("qwen3-tts.tokenizer.sample_rate", 24000));
    model_.config.n_codebooks = loader.get_u32("qwen3-tts-tokenizer.num_codebooks",
                                loader.get_u32("qwen3-tts.tokenizer.num_codebooks", 16));
    model_.config.codebook_size = loader.get_u32("qwen3-tts-tokenizer.codebook_size",
                                  loader.get_u32("qwen3-tts.tokenizer.codebook_size", 2048));
    
    int64_t n_tensors = loader.get_n_tensors();
    int dec_tensor_count = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (name && strncmp(name, "tok_dec.", 8) == 0) {
            dec_tensor_count++;
        }
    }
    
    if (dec_tensor_count == 0) {
        error_msg_ = "No decoder tensors found in model";
        return false;
    }
    
    size_t ctx_size = ggml_tensor_overhead() * dec_tensor_count;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_msg_ = "Failed to initialize GGML context";
        return false;
    }
    
    struct gguf_context * gguf_ctx = loader.get_ctx();
    struct ggml_context * meta_ctx = loader.get_meta_ctx();
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name || strncmp(name, "tok_dec.", 8) != 0) {
            continue;
        }
        
        struct ggml_tensor * meta_tensor = ggml_get_tensor(meta_ctx, name);
        if (!meta_tensor) {
            continue;
        }
        
        struct ggml_tensor * tensor = ggml_dup_tensor(model_.ctx, meta_tensor);
        ggml_set_name(tensor, name);
        
        model_.tensors[name] = tensor;
        
        std::string sname(name);
        
        if (sname == "tok_dec.vq_first.input_proj.weight") model_.vq_first_input_proj = tensor;
        else if (sname == "tok_dec.vq_first.output_proj.weight") model_.vq_first_output_proj = tensor;
        else if (sname == "tok_dec.vq_first.0.codebook") model_.vq_first_codebook = tensor;
        else if (sname == "tok_dec.vq_first.0.usage") model_.vq_first_usage = tensor;
        else if (sname == "tok_dec.vq_rest.input_proj.weight") model_.vq_rest_input_proj = tensor;
        else if (sname == "tok_dec.vq_rest.output_proj.weight") model_.vq_rest_output_proj = tensor;
        else if (sname == "tok_dec.pre_conv.weight") model_.pre_conv_w = tensor;
        else if (sname == "tok_dec.pre_conv.bias") model_.pre_conv_b = tensor;
        else if (sname == "tok_dec.pre_tfm.input_proj.weight") model_.pre_tfm_input_proj_w = tensor;
        else if (sname == "tok_dec.pre_tfm.input_proj.bias") model_.pre_tfm_input_proj_b = tensor;
        else if (sname == "tok_dec.pre_tfm.norm.weight") model_.pre_tfm_norm_w = tensor;
        else if (sname == "tok_dec.pre_tfm.output_proj.weight") model_.pre_tfm_output_proj_w = tensor;
        else if (sname == "tok_dec.pre_tfm.output_proj.bias") model_.pre_tfm_output_proj_b = tensor;
        else if (sname == "tok_dec.dec.0.conv.weight") model_.dec0_conv_w = tensor;
        else if (sname == "tok_dec.dec.0.conv.bias") model_.dec0_conv_b = tensor;
        else if (sname == "tok_dec.dec.5.snake.alpha") model_.dec5_snake_alpha = tensor;
        else if (sname == "tok_dec.dec.5.snake.beta") model_.dec5_snake_beta = tensor;
        else if (sname == "tok_dec.dec.6.conv.weight") model_.dec6_conv_w = tensor;
        else if (sname == "tok_dec.dec.6.conv.bias") model_.dec6_conv_b = tensor;
        else if (sname.find("pre_tfm.blk.") != std::string::npos) {
            int blk_idx;
            if (sscanf(name, "tok_dec.pre_tfm.blk.%d.", &blk_idx) == 1 && blk_idx >= 0 && blk_idx < 8) {
                if (sname.find(".attn_v.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_v_w = tensor;
                else if (sname.find(".ffn_gate.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_gate_w = tensor;
                else if (sname.find(".attn_norm.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_norm_w = tensor;
                else if (sname.find(".attn_q.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_q_w = tensor;
                else if (sname.find(".attn_k.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_k_w = tensor;
                else if (sname.find(".attn_output.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_output_w = tensor;
                else if (sname.find(".attn_scale") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_scale = tensor;
                else if (sname.find(".ffn_norm.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_norm_w = tensor;
                else if (sname.find(".ffn_up.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_up_w = tensor;
                else if (sname.find(".ffn_down.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_down_w = tensor;
                else if (sname.find(".ffn_scale") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_scale = tensor;
            }
        }
        else {
            int blk_idx, res_idx, cb_idx, n = 0;
            char suffix[64];
            size_t name_len = strlen(name);
            
            #define MATCH1(fmt, var) (sscanf(name, fmt "%n", &var, &n) == 1 && (size_t)n == name_len)
            #define MATCH2(fmt, v1, v2) (sscanf(name, fmt "%n", &v1, &v2, &n) == 2 && (size_t)n == name_len)
            #define MATCH1S(fmt, var, suf) (sscanf(name, fmt, &var, suf) == 2)
            
            if (MATCH1("tok_dec.vq_rest.%d.codebook", cb_idx)) {
                if (cb_idx >= 0 && cb_idx < 15) {
                    model_.vq_rest_codebook[cb_idx] = tensor;
                }
            }
            else if (MATCH1("tok_dec.vq_rest.%d.usage", cb_idx)) {
                if (cb_idx >= 0 && cb_idx < 15) {
                    model_.vq_rest_usage[cb_idx] = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.conv.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].conv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].conv_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.dwconv.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].dwconv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].dwconv_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.norm.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].norm_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].norm_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.pwconv1.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].pwconv1_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].pwconv1_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.pwconv2.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].pwconv2_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].pwconv2_b = tensor;
                }
            }
            else if (MATCH1("tok_dec.upsample.%d.gamma", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 2) model_.upsample[blk_idx].gamma = tensor;
            }
            // pre_tfm.blk.* tensors are routed above via the sname.find("pre_tfm.blk.") branch
            // and never reach this else-block — no duplicate assignments needed here.
            else if (MATCH1("tok_dec.dec.%d.snake.alpha", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].snake_alpha = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.snake.beta", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].snake_beta = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.conv_t.weight", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].conv_t_w = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.conv_t.bias", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].conv_t_b = tensor;
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act1.alpha", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act1_alpha = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act1.beta", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act1_beta = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv1.weight", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv1_w = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv1.bias", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv1_b = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act2.alpha", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act2_alpha = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act2.beta", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act2_beta = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv2.weight", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv2_w = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv2.bias", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv2_b = tensor;
                }
            }
            #undef MATCH1
            #undef MATCH2
            #undef MATCH1S
        }
    }
    
    state_.backend = init_preferred_backend("AudioTokenizerDecoder", &error_msg_);
    if (!state_.backend) {
        return false;
    }

    if (!load_tensor_data_from_file(model_path, gguf_ctx, model_.ctx,
                                     model_.tensors, model_.buffer, error_msg_,
                                     GGML_BACKEND_DEVICE_TYPE_IGPU,
                                     state_.backend)) {
        return false;
    }
    
    for (int i = 0; i < 4; ++i) {
        model_.dec_blocks[i].res[0].dilation = 1;
        model_.dec_blocks[i].res[1].dilation = 3;
        model_.dec_blocks[i].res[2].dilation = 9;
    }
    
    normalize_codebooks();
    const char * max_graph = getenv("QWEN3_TTS_MAX_VOCODER");
    state_.max_graph = max_graph && strcmp(max_graph, "1") == 0;
    if (state_.max_graph && !prepare_max_weights()) return false;

    ggml_backend_dev_t device = ggml_backend_get_device(state_.backend);
    const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
    fprintf(stderr, "  AudioTokenizerDecoder backend: %s\n", device_name);
    
    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        state_.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!state_.backend_cpu) {
            error_msg_ = "Failed to initialize CPU fallback backend for AudioTokenizerDecoder";
            return false;
        }
    }

    std::vector<ggml_backend_t> backends;
    backends.push_back(state_.backend);
    if (state_.backend_cpu) {
        backends.push_back(state_.backend_cpu);
    }
    const size_t max_nodes = state_.max_graph ? QWEN3_TTS_MAX_DEC_MAX_NODES : QWEN3_TTS_DEC_MAX_NODES;
    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), max_nodes, false, true);
    if (!state_.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }
    
    state_.compute_meta.resize(ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false));
    
    return true;
}

// NPU-Max preserves GGUF weights and prepares only additional resident layouts.
bool AudioTokenizerDecoder::prepare_max_weights() {
    ggml_init_params params = {ggml_tensor_overhead() * 1024, nullptr, true};
    state_.max_ctx = ggml_init(params);
    if (!state_.max_ctx) { error_msg_ = "Cannot allocate NPU-Max constant metadata"; return false; }
    struct upload { ggml_tensor * tensor; std::vector<uint8_t> data; };
    std::vector<upload> uploads;
    auto read = [&](ggml_tensor * source) {
        std::vector<float> values(ggml_nelements(source));
        if (source->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(source, values.data(), 0, values.size() * sizeof(float));
        } else if (source->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> half(values.size());
            ggml_backend_tensor_get(source, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < values.size(); ++i) values[i] = ggml_fp16_to_fp32(half[i]);
        } else values.clear();
        return values;
    };
    auto add = [&](ggml_tensor * target, const std::vector<float> & data) {
        upload item; item.tensor = target; item.data.resize(ggml_nbytes(target));
        if (target->type == GGML_TYPE_F32) memcpy(item.data.data(), data.data(), item.data.size());
        else for (size_t i = 0; i < data.size(); ++i) {
            const ggml_fp16_t value = ggml_fp32_to_fp16(data[i]);
            memcpy(item.data.data() + i * sizeof(value), &value, sizeof(value));
        }
        uploads.push_back(std::move(item));
    };
    auto conv1d = [&](ggml_tensor * source) {
        if (!source) return false;
        std::vector<float> values = read(source);
        if (values.empty()) return false;
        const int64_t kernel = source->ne[0], in = source->ne[1], out = source->ne[2];
        for (int64_t tap = 0; tap < kernel; ++tap) {
            ggml_tensor * target = ggml_new_tensor_2d(state_.max_ctx, source->type, in, out);
            std::vector<float> data(in * out);
            for (int64_t o = 0; o < out; ++o)
                for (int64_t i = 0; i < in; ++i)
                    data[o * in + i] = values[(o * in + i) * kernel + tap];
            state_.max_taps[source].push_back(target); add(target, data);
        }
        return true;
    };
    auto transposed = [&](ggml_tensor * source) {
        std::vector<float> values = read(source);
        if (values.empty()) return false;
        const int64_t kernel = source->ne[0], out = source->ne[1], in = source->ne[2];
        for (int64_t tap = 0; tap < kernel; ++tap) {
            ggml_tensor * target = ggml_new_tensor_2d(state_.max_ctx, source->type, in, out);
            std::vector<float> data(in * out);
            for (int64_t o = 0; o < out; ++o)
                for (int64_t i = 0; i < in; ++i)
                    data[o * in + i] = values[(i * out + o) * kernel + tap];
            state_.max_taps[source].push_back(target); add(target, data);
        }
        return true;
    };
    auto depthwise = [&](ggml_tensor * source) {
        std::vector<float> values = read(source);
        if (values.empty()) return false;
        const int64_t kernel = source->ne[0], channels = source->ne[2];
        for (int64_t tap = 0; tap < kernel; ++tap) {
            ggml_tensor * target = ggml_new_tensor_3d(state_.max_ctx, GGML_TYPE_F32, 1, channels, 1);
            std::vector<float> data(channels);
            for (int64_t c = 0; c < channels; ++c) data[c] = values[c * kernel + tap];
            state_.max_taps[source].push_back(target); add(target, data);
        }
        return true;
    };
    auto snake = [&](ggml_tensor * source, float sign) {
        if (!source || state_.max_snake.count(source)) return true;
        std::vector<float> values = read(source);
        if (values.empty()) return false;
        for (float & value : values) value = expf(sign * value);
        ggml_tensor * target = ggml_new_tensor_1d(state_.max_ctx, GGML_TYPE_F32, values.size());
        state_.max_snake[source] = target; add(target, values); return true;
    };
    auto codebook = [&](ggml_tensor * source) {
        if (!source || state_.max_codebooks.count(source)) return true;
        std::vector<float> values = read(source);
        if (values.empty()) return false;
        ggml_tensor * target = ggml_new_tensor_2d(state_.max_ctx, GGML_TYPE_F32, source->ne[0], source->ne[1]);
        state_.max_codebooks[source] = target; add(target, values); return true;
    };
    bool ok = true;
    ok = codebook(model_.vq_first_codebook) && ok;
    for (int i = 0; i < 15; ++i) ok = codebook(model_.vq_rest_codebook[i]) && ok;
    ok = conv1d(model_.pre_conv_w) && conv1d(model_.dec0_conv_w) && conv1d(model_.dec6_conv_w) && ok;
    for (const auto & block : model_.upsample) {
        if (block.conv_w) ok = transposed(block.conv_w) && ok;
        if (block.dwconv_w) ok = depthwise(block.dwconv_w) && ok;
    }
    for (const auto & block : model_.dec_blocks) {
        ok = transposed(block.conv_t_w) && ok;
        ok = snake(block.snake_alpha, 1) && snake(block.snake_beta, -1) && ok;
        for (const auto & res : block.res) {
            ok = conv1d(res.conv1_w) && conv1d(res.conv2_w) && ok;
            ok = snake(res.act1_alpha, 1) && snake(res.act1_beta, -1) &&
                 snake(res.act2_alpha, 1) && snake(res.act2_beta, -1) && ok;
        }
    }
    ok = snake(model_.dec5_snake_alpha, 1) && snake(model_.dec5_snake_beta, -1) && ok;
    if (!ok) { error_msg_ = "NPU-Max requires F16/F32 convolution and Snake weights"; return false; }
    state_.max_buffer = ggml_backend_alloc_ctx_tensors(state_.max_ctx, state_.backend);
    if (!state_.max_buffer) { error_msg_ = "Cannot allocate NPU-Max resident constants"; return false; }
    ggml_backend_buffer_set_usage(state_.max_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const auto & item : uploads)
        ggml_backend_tensor_set(item.tensor, item.data.data(), 0, item.data.size());
    fprintf(stderr, "  NPU-Max vocoder resident transformed constants: %zu bytes\n",
            ggml_backend_buffer_get_size(state_.max_buffer));
    return true;
}

// A balanced tree avoids quadratic copies when assembling many time tiles.
static ggml_tensor * max_concat_tiles(ggml_context * ctx, std::vector<ggml_tensor *> tiles, int dim) {
    GGML_ASSERT(!tiles.empty());
    while (tiles.size() > 1) {
        std::vector<ggml_tensor *> next;
        next.reserve((tiles.size() + 1) / 2);
        for (size_t i = 0; i < tiles.size(); i += 2)
            next.push_back(i + 1 < tiles.size() ? ggml_concat(ctx, tiles[i], tiles[i + 1], dim) : tiles[i]);
        tiles = std::move(next);
    }
    return tiles[0];
}

// Exact im2col / per-tap lowering, using supported views/cont/concat; batches never exceed
// the real HTP backend's 1024-row matmul support bound.
ggml_tensor * AudioTokenizerDecoder::max_conv(ggml_context * ctx, ggml_tensor * weight,
                                              ggml_tensor * x, int dilation, bool channels_first) {
    const int64_t kernel = weight->ne[0], in = weight->ne[1], out = weight->ne[2];
    const int64_t x_len = channels_first ? x->ne[1] : x->ne[0];
    const int64_t length = x_len - (kernel - 1) * dilation;
    GGML_ASSERT(length > 0 && dilation > 0 &&
                (channels_first ? x->ne[0] == in : x->ne[1] == in) &&
                x->ne[2] == 1 && x->ne[3] == 1 && x->nb[0] == sizeof(float));
    std::vector<ggml_tensor *> tiles;
    const auto tap_it = state_.max_taps.find(weight);
    const bool has_taps = (tap_it != state_.max_taps.end() && (int64_t)tap_it->second.size() == kernel);
    if (has_taps) {
        ggml_tensor * x_ct = channels_first ? ggml_reshape_2d(ctx, x, in, x_len) :
                             ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, x, x_len, in)));
        const auto & taps = tap_it->second;
        for (int64_t base = 0; base < length; base += 1024) {
            const int64_t count = std::min<int64_t>(1024, length - base);
            ggml_tensor * sum = nullptr;
            for (int64_t tap = 0; tap < kernel; ++tap) {
                ggml_tensor * slice = ggml_view_2d(ctx, x_ct, in, count, x_ct->nb[1],
                                                   (base + tap * dilation) * x_ct->nb[1]);
                ggml_tensor * product = ggml_mul_mat(ctx, taps[tap], slice);
                if (out == 1) {
                    product = ggml_reshape_1d(ctx, product, count);
                }
                sum = sum ? ggml_add(ctx, sum, product) : product;
            }
            tiles.push_back(sum);
        }
        if (out == 1) {
            ggml_tensor * flat = max_concat_tiles(ctx, std::move(tiles), 0);
            return channels_first ? ggml_reshape_3d(ctx, flat, 1, length, 1) :
                                    ggml_reshape_3d(ctx, flat, length, 1, 1);
        }
        ggml_tensor * out_ct = max_concat_tiles(ctx, std::move(tiles), 1);
        if (channels_first) {
            return ggml_reshape_3d(ctx, out_ct, out, length, 1);
        }
        return ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, out_ct)), length, out, 1);
    }

    ggml_tensor * matrix = ggml_reshape_2d(ctx, weight, kernel * in, out);
    for (int64_t base = 0; base < length; base += 1024) {
        const int64_t count = std::min<int64_t>(1024, length - base);
        ggml_tensor * patch = nullptr;
        for (int64_t tap = 0; tap < kernel; ++tap) {
            ggml_tensor * slice = ggml_view_2d(ctx, x, count, in, x->nb[1],
                                              (base + tap * dilation) * x->nb[0]);
            slice = ggml_cont(ctx, ggml_transpose(ctx, slice));
            slice = ggml_reshape_3d(ctx, slice, 1, in, count);
            patch = patch ? ggml_concat(ctx, patch, slice, 0) : slice;
        }
        patch = ggml_reshape_2d(ctx, patch, kernel * in, count);
        // Activation must remain F32 for HTP HMX matmul kernels.
        ggml_tensor * tile = ggml_mul_mat(ctx, matrix, patch);
        tile = ggml_cont(ctx, ggml_transpose(ctx, tile));
        tiles.push_back(tile);
    }
    return ggml_reshape_3d(ctx, max_concat_tiles(ctx, std::move(tiles), 0), length, out, 1);
}

ggml_tensor * AudioTokenizerDecoder::max_depthwise_conv(ggml_context * ctx, ggml_tensor * weight,
                                                        ggml_tensor * x, bool channels_first) {
    const int64_t kernel = weight->ne[0], channels = weight->ne[2];
    const int64_t x_len = channels_first ? x->ne[1] : x->ne[0];
    const int64_t length = x_len - kernel + 1;
    GGML_ASSERT(length > 0 && (channels_first ? x->ne[0] == channels : x->ne[1] == channels) &&
                x->ne[2] == 1 && x->ne[3] == 1 && x->nb[0] == sizeof(float));
    ggml_tensor * result = nullptr;
    for (int64_t tap = 0; tap < kernel; ++tap) {
        if (channels_first) {
            ggml_tensor * slice = ggml_view_2d(ctx, x, channels, length, x->nb[1], tap * x->nb[1]);
            slice = ggml_reshape_3d(ctx, slice, channels, length, 1);
            ggml_tensor * tap_w = ggml_reshape_3d(ctx, state_.max_taps.at(weight)[tap], channels, 1, 1);
            ggml_tensor * product = ggml_mul(ctx, slice, tap_w);
            result = result ? ggml_add(ctx, result, product) : product;
        } else {
            ggml_tensor * slice = ggml_view_3d(ctx, x, length, channels, 1,
                                             x->nb[1], x->nb[2], tap * x->nb[0]);
            slice = ggml_cast(ctx, ggml_cast(ctx, slice, GGML_TYPE_F16), GGML_TYPE_F32);
            ggml_tensor * product = ggml_mul(ctx, slice, state_.max_taps.at(weight)[tap]);
            result = result ? ggml_add(ctx, result, product) : product;
        }
    }
    return result;
}

// Polyphase transposed convolution, with constant tap matrices prepared once.
ggml_tensor * AudioTokenizerDecoder::max_transpose_conv(ggml_context * ctx, ggml_tensor * weight,
                                                        ggml_tensor * x, int stride, bool channels_first) {
    const int64_t kernel = weight->ne[0], out = weight->ne[1], in = weight->ne[2];
    const int64_t x_len = channels_first ? x->ne[1] : x->ne[0];
    GGML_ASSERT(stride > 0 && kernel >= stride && x_len > 0 &&
                (channels_first ? x->ne[0] == in : x->ne[1] == in) &&
                x->ne[2] == 1 && x->ne[3] == 1 && x->nb[0] == sizeof(float));
    const int64_t length = (x_len - 1) * stride + kernel;
    const int64_t phase_length = (length + stride - 1) / stride;
    const int64_t halo = (kernel - 1) / stride;
    ggml_tensor * padded_ct = nullptr;
    if (channels_first) {
        padded_ct = ggml_pad_ext(ctx, x, 0, 0, halo, halo + stride, 0, 0, 0, 0);
    } else {
        ggml_tensor * padded = ggml_pad_ext(ctx, x, halo, halo + stride, 0, 0, 0, 0, 0, 0);
        padded_ct = ggml_cont(ctx, ggml_transpose(ctx, padded));
    }
    ggml_set_name(padded_ct, "tconv_padded");
    std::vector<ggml_tensor *> tiles;
    const auto & taps = state_.max_taps.at(weight);
    for (int64_t base = 0; base < phase_length; base += 1024) {
        const int64_t count = std::min<int64_t>(1024, phase_length - base);
        ggml_tensor * phases = nullptr;
        for (int phase = 0; phase < stride; ++phase) {
            ggml_tensor * sum = nullptr;
            for (int64_t tap = phase; tap < kernel; tap += stride) {
                const int64_t shift = tap / stride;
                ggml_tensor * slice = ggml_view_2d(ctx, padded_ct, in, count, padded_ct->nb[1],
                                                  (base + halo - shift) * padded_ct->nb[1]);
                // Activation must remain F32 for HTP HMX matmul kernels.
                ggml_tensor * product = ggml_mul_mat(ctx, taps[tap], slice);
                sum = sum ? ggml_add(ctx, sum, product) : product;
            }
            GGML_ASSERT(sum);
            sum = ggml_reshape_3d(ctx, sum, out, 1, count);
            char pname[32]; snprintf(pname, sizeof(pname), "tconv_phase_%d", phase);
            ggml_set_name(sum, pname);
            phases = phases ? ggml_concat(ctx, phases, sum, 1) : sum;
            char cname[32]; snprintf(cname, sizeof(cname), "tconv_concat_%d", phase);
            ggml_set_name(phases, cname);
        }
        ggml_tensor * tile = ggml_reshape_2d(ctx, phases, out, stride * count);
        tiles.push_back(tile);
    }
    ggml_tensor * result = max_concat_tiles(ctx, std::move(tiles), 1);
    result = ggml_view_2d(ctx, result, out, length, result->nb[1], 0);
    return channels_first ? result : ggml_cont(ctx, ggml_transpose(ctx, result));
}

struct ggml_tensor * AudioTokenizerDecoder::apply_snake(struct ggml_context * ctx,
                                                          struct ggml_tensor * x,
                                                          struct ggml_tensor * alpha,
                                                          struct ggml_tensor * beta) {
    if (state_.max_graph) {
        const int64_t channels = x->ne[0];
        struct ggml_tensor * alpha_3d = ggml_reshape_3d(ctx, state_.max_snake.at(alpha), channels, 1, 1);
        struct ggml_tensor * ax = ggml_mul(ctx, x, alpha_3d);
        struct ggml_tensor * sin_ax = ggml_sin(ctx, ax);
        struct ggml_tensor * sin_sq = ggml_sqr(ctx, sin_ax);
        struct ggml_tensor * inv_beta_3d = ggml_reshape_3d(ctx, state_.max_snake.at(beta), channels, 1, 1);
        struct ggml_tensor * scaled_sin = ggml_mul(ctx, sin_sq, inv_beta_3d);
        return ggml_add(ctx, x, scaled_sin);
    }

    int64_t seq_len = x->ne[0];
    int64_t channels = x->ne[1];
    int64_t batch = x->ne[2];
    
    struct ggml_tensor * alpha_exp = ggml_exp(ctx, alpha);
    struct ggml_tensor * alpha_3d = ggml_reshape_3d(ctx, alpha_exp, 1, channels, 1);
    struct ggml_tensor * alpha_broad = ggml_repeat(ctx, alpha_3d,
                                                   ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, channels, batch));
    
    struct ggml_tensor * ax = ggml_mul(ctx, x, alpha_broad);
    struct ggml_tensor * sin_ax = ggml_sin(ctx, ax);
    struct ggml_tensor * sin_sq = ggml_sqr(ctx, sin_ax);
    
    struct ggml_tensor * neg_beta = ggml_scale(ctx, beta, -1.0f);
    struct ggml_tensor * inv_beta_exp = ggml_exp(ctx, neg_beta);
    struct ggml_tensor * inv_beta_3d = ggml_reshape_3d(ctx, inv_beta_exp, 1, channels, 1);
    struct ggml_tensor * inv_beta = ggml_repeat(ctx, inv_beta_3d,
                                                ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, channels, batch));
    
    struct ggml_tensor * scaled_sin = ggml_mul(ctx, sin_sq, inv_beta);
    
    return ggml_add(ctx, x, scaled_sin);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_rms_norm(struct ggml_context * ctx,
                                                            struct ggml_tensor * x,
                                                            struct ggml_tensor * w,
                                                            float eps) {
    struct ggml_tensor * normed = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, normed, w);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_pre_tfm_layer(struct ggml_context * ctx,
                                                                 struct ggml_tensor * x,
                                                                 const pre_tfm_layer & layer,
                                                                 int32_t n_frames,
                                                                 struct ggml_tensor * positions,
                                                                 struct ggml_tensor * causal_mask) {
    const auto & cfg = model_.config;
    const int n_heads = cfg.n_heads;
    const int qkv_dim = cfg.latent_dim;
    const int head_dim = qkv_dim / n_heads;
    
    if (!layer.attn_norm_w || !layer.attn_q_w || !layer.attn_k_w || !layer.attn_v_w ||
        !layer.attn_output_w || !layer.ffn_norm_w || !layer.ffn_gate_w || 
        !layer.ffn_up_w || !layer.ffn_down_w) {
        return x;
    }
    
    struct ggml_tensor * residual = x;
    
    struct ggml_tensor * normed = apply_rms_norm(ctx, x, layer.attn_norm_w, cfg.rms_norm_eps);
    
    struct ggml_tensor * Qcur = ggml_mul_mat(ctx, layer.attn_q_w, normed);
    struct ggml_tensor * Kcur = ggml_mul_mat(ctx, layer.attn_k_w, normed);
    struct ggml_tensor * Vcur = ggml_mul_mat(ctx, layer.attn_v_w, normed);
    
    Qcur = ggml_reshape_3d(ctx, Qcur, head_dim, n_heads, n_frames);
    Kcur = ggml_reshape_3d(ctx, Kcur, head_dim, n_heads, n_frames);
    Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_heads, n_frames);
    
    Qcur = ggml_rope_ext(ctx, Qcur, positions, nullptr,
                         head_dim, GGML_ROPE_TYPE_NEOX, 0,
                         cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    
    Kcur = ggml_rope_ext(ctx, Kcur, positions, nullptr,
                         head_dim, GGML_ROPE_TYPE_NEOX, 0,
                         cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    
    struct ggml_tensor * Q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
    struct ggml_tensor * K = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
    struct ggml_tensor * V = ggml_permute(ctx, Vcur, 0, 2, 1, 3);
    
    struct ggml_tensor * attn_out = nullptr;
    if (state_.max_graph) {
        // Preserve causal attention while fusing matmul, masking and softmax.
        K = ggml_cast(ctx, K, GGML_TYPE_F16);
        V = ggml_cast(ctx, V, GGML_TYPE_F16);
        ggml_tensor * attended = ggml_flash_attn_ext(ctx, Q, K, V, causal_mask,
                                                    1.0f / sqrtf((float) head_dim), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
        attn_out = ggml_reshape_2d(ctx, attended, n_heads * head_dim, n_frames);
    } else {
        struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);
        KQ = ggml_scale(ctx, KQ, 1.0f / sqrtf((float)head_dim));
        KQ = ggml_diag_mask_inf(ctx, KQ, 0);
        KQ = ggml_soft_max(ctx, KQ);
        V = ggml_cont(ctx, ggml_transpose(ctx, V));
        struct ggml_tensor * KQV = ggml_mul_mat(ctx, V, KQ);
        KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);
        attn_out = ggml_cont_2d(ctx, KQV, n_heads * head_dim, n_frames);
    }

    attn_out = ggml_mul_mat(ctx, layer.attn_output_w, attn_out);
    
    if (layer.attn_scale) {
        attn_out = ggml_mul(ctx, attn_out, layer.attn_scale);
    }
    
    x = ggml_add(ctx, residual, attn_out);
    residual = x;
    
    normed = apply_rms_norm(ctx, x, layer.ffn_norm_w, cfg.rms_norm_eps);
    
    struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate_w, normed);
    struct ggml_tensor * up = ggml_mul_mat(ctx, layer.ffn_up_w, normed);
    
    gate = ggml_silu(ctx, gate);
    struct ggml_tensor * ffn_out = ggml_mul(ctx, gate, up);
    
    ffn_out = ggml_mul_mat(ctx, layer.ffn_down_w, ffn_out);
    
    if (layer.ffn_scale) {
        ffn_out = ggml_mul(ctx, ffn_out, layer.ffn_scale);
    }
    
    return ggml_add(ctx, residual, ffn_out);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_upsample_block(struct ggml_context * ctx,
                                                                  struct ggml_tensor * x,
                                                                  const upsample_block & block,
                                                                  int block_idx) {
    char prefix[16];
    snprintf(prefix, sizeof(prefix), "up%d_", block_idx);

    if (state_.max_graph) {
        const int64_t channels = x->ne[0];
        const int64_t seq_len = x->ne[1];
        struct ggml_tensor * x_2d = ggml_reshape_2d(ctx, x, channels, seq_len);
        x_2d = max_transpose_conv(ctx, block.conv_w, x_2d, 2, true);
        char name_conv_t[32]; snprintf(name_conv_t, sizeof(name_conv_t), "%sconv_t", prefix);
        ggml_set_name(x_2d, name_conv_t);

        const int64_t new_seq_len = x_2d->ne[1];
        x = ggml_reshape_3d(ctx, x_2d, channels, new_seq_len, 1);
        if (block.conv_b) {
            x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv_b, channels, 1, 1));
            char name_conv_b[32]; snprintf(name_conv_b, sizeof(name_conv_b), "%sconv_b", prefix);
            ggml_set_name(x, name_conv_b);
        }

        struct ggml_tensor * residual = x;
        if (block.dwconv_w) {
            x = ggml_pad_ext(ctx, x, 0, 0, 6, 0, 0, 0, 0, 0);
            char name_pad[32]; snprintf(name_pad, sizeof(name_pad), "%spad", prefix);
            ggml_set_name(x, name_pad);

            x = max_depthwise_conv(ctx, block.dwconv_w, x, true);
            char name_dwconv[32]; snprintf(name_dwconv, sizeof(name_dwconv), "%sdwconv", prefix);
            ggml_set_name(x, name_dwconv);

            if (block.dwconv_b) {
                x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.dwconv_b, channels, 1, 1));
                char name_dwconv_b[32]; snprintf(name_dwconv_b, sizeof(name_dwconv_b), "%sdwconv_b", prefix);
                ggml_set_name(x, name_dwconv_b);
            }
        }

        if (block.norm_w && block.norm_b) {
            x = ggml_norm(ctx, x, 1e-6f);
            x = ggml_mul(ctx, x, block.norm_w);
            x = ggml_add(ctx, x, block.norm_b);
            char name_norm[32]; snprintf(name_norm, sizeof(name_norm), "%snorm", prefix);
            ggml_set_name(x, name_norm);
        }

        x = ggml_mul_mat(ctx, block.pwconv1_w, x);
        if (block.pwconv1_b) {
            x = ggml_add(ctx, x, block.pwconv1_b);
        }
        char name_pw1[32]; snprintf(name_pw1, sizeof(name_pw1), "%spw1", prefix);
        ggml_set_name(x, name_pw1);

        x = ggml_gelu(ctx, x);
        char name_gelu[32]; snprintf(name_gelu, sizeof(name_gelu), "%sgelu", prefix);
        ggml_set_name(x, name_gelu);

        x = ggml_mul_mat(ctx, block.pwconv2_w, x);
        if (block.pwconv2_b) {
            x = ggml_add(ctx, x, block.pwconv2_b);
        }
        char name_pw2[32]; snprintf(name_pw2, sizeof(name_pw2), "%spw2", prefix);
        ggml_set_name(x, name_pw2);

        if (block.gamma) {
            x = ggml_mul(ctx, x, ggml_reshape_3d(ctx, block.gamma, channels, 1, 1));
            char name_gamma[32]; snprintf(name_gamma, sizeof(name_gamma), "%sgamma", prefix);
            ggml_set_name(x, name_gamma);
        }

        struct ggml_tensor * res_out = ggml_add(ctx, residual, x);
        char name_out[32]; snprintf(name_out, sizeof(name_out), "%sout", prefix);
        ggml_set_name(res_out, name_out);
        return res_out;
    }

    int64_t seq_len = x->ne[0];
    int64_t channels = x->ne[1];
    
    struct ggml_tensor * x_2d = ggml_reshape_2d(ctx, x, seq_len, channels);
    x_2d = ggml_conv_transpose_1d(ctx, block.conv_w, x_2d, 2, 0, 1);
    char name_conv_t[32]; snprintf(name_conv_t, sizeof(name_conv_t), "%sconv_t", prefix);
    ggml_set_name(x_2d, name_conv_t);
     
    int64_t new_seq_len = x_2d->ne[0];
    x = ggml_reshape_3d(ctx, x_2d, new_seq_len, channels, 1);
     
    if (block.conv_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv_b, 1, channels, 1));
        char name_conv_b[32]; snprintf(name_conv_b, sizeof(name_conv_b), "%sconv_b", prefix);
        ggml_set_name(x, name_conv_b);
    }
    
    struct ggml_tensor * residual = x;
     
    if (block.dwconv_w) {
        // Causal padding: pad left with 6 zeros (kernel_size - 1 = 7 - 1 = 6)
        x = ggml_pad_ext(ctx, x, 6, 0, 0, 0, 0, 0, 0, 0);  // left pad only
        char name_pad[32]; snprintf(name_pad, sizeof(name_pad), "%spad", prefix);
        ggml_set_name(x, name_pad);

        x = ggml_conv_1d_dw(ctx, block.dwconv_w, x, 1, 0, 1);  // no padding in conv
        char name_dwconv[32]; snprintf(name_dwconv, sizeof(name_dwconv), "%sdwconv", prefix);
        ggml_set_name(x, name_dwconv);

        if (block.dwconv_b) {
            x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.dwconv_b, 1, channels, 1));
            char name_dwconv_b[32]; snprintf(name_dwconv_b, sizeof(name_dwconv_b), "%sdwconv_b", prefix);
            ggml_set_name(x, name_dwconv_b);
        }
    }
    
    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);
    
    if (block.norm_w && block.norm_b) {
        x = ggml_norm(ctx, x, 1e-6f);
        x = ggml_mul(ctx, x, block.norm_w);
        x = ggml_add(ctx, x, block.norm_b);
        char name_norm[32]; snprintf(name_norm, sizeof(name_norm), "%snorm", prefix);
        ggml_set_name(x, name_norm);
    }
    
    x = ggml_mul_mat(ctx, block.pwconv1_w, x);
    if (block.pwconv1_b) {
        x = ggml_add(ctx, x, block.pwconv1_b);
    }
    char name_pw1[32]; snprintf(name_pw1, sizeof(name_pw1), "%spw1", prefix);
    ggml_set_name(x, name_pw1);
    
    x = ggml_gelu(ctx, x);
    char name_gelu[32]; snprintf(name_gelu, sizeof(name_gelu), "%sgelu", prefix);
    ggml_set_name(x, name_gelu);
    
    x = ggml_mul_mat(ctx, block.pwconv2_w, x);
    if (block.pwconv2_b) {
        x = ggml_add(ctx, x, block.pwconv2_b);
    }
    char name_pw2[32]; snprintf(name_pw2, sizeof(name_pw2), "%spw2", prefix);
    ggml_set_name(x, name_pw2);
    
    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);
    
    if (block.gamma) {
        struct ggml_tensor * gamma_3d = ggml_reshape_3d(ctx, block.gamma, 1, channels, 1);
        x = ggml_mul(ctx, x, ggml_repeat(ctx, gamma_3d, 
                                          ggml_new_tensor_3d(ctx, GGML_TYPE_F32, new_seq_len, channels, 1)));
        char name_gamma[32]; snprintf(name_gamma, sizeof(name_gamma), "%sgamma", prefix);
        ggml_set_name(x, name_gamma);
    }
    
    struct ggml_tensor * res_out = ggml_add(ctx, residual, x);
    char name_out[32]; snprintf(name_out, sizeof(name_out), "%sout", prefix);
    ggml_set_name(res_out, name_out);
    return res_out;
}

struct ggml_tensor * AudioTokenizerDecoder::apply_residual_block(struct ggml_context * ctx,
                                                                  struct ggml_tensor * x,
                                                                  const residual_block & block) {
    struct ggml_tensor * residual = x;
    
    if (block.act1_alpha) {
        x = apply_snake(ctx, x, block.act1_alpha, block.act1_beta);
        ggml_set_name(x, "r_act1");
    }
    
    int64_t out_channels = block.conv1_w->ne[2];
    int padding = 6 * block.dilation;
    if (state_.max_graph) {
        x = ggml_pad_ext(ctx, x, 0, 0, padding, 0, 0, 0, 0, 0);
        ggml_set_name(x, "r_pad");
        x = max_conv(ctx, block.conv1_w, x, block.dilation, true);
        if (block.conv1_b) {
            x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv1_b, out_channels, 1, 1));
        }
        ggml_set_name(x, "r_conv1");

        if (block.act2_alpha) {
            x = apply_snake(ctx, x, block.act2_alpha, block.act2_beta);
            ggml_set_name(x, "r_act2");
        }

        out_channels = block.conv2_w->ne[2];
        x = max_conv(ctx, block.conv2_w, x, 1, true);
        if (block.conv2_b) {
            x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv2_b, out_channels, 1, 1));
        }
        ggml_set_name(x, "r_conv2");
        return ggml_add(ctx, residual, x);
    }

    x = ggml_pad_ext(ctx, x, padding, 0, 0, 0, 0, 0, 0, 0);
    ggml_set_name(x, "r_pad");
    x = ggml_conv_1d(ctx, block.conv1_w, x, 1, 0, block.dilation);
    if (block.conv1_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv1_b, 1, out_channels, 1));
    }
    ggml_set_name(x, "r_conv1");
    
    if (block.act2_alpha) {
        x = apply_snake(ctx, x, block.act2_alpha, block.act2_beta);
        ggml_set_name(x, "r_act2");
    }
    
    out_channels = block.conv2_w->ne[2];
    x = ggml_conv_1d(ctx, block.conv2_w, x, 1, 0, 1);
    if (block.conv2_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv2_b, 1, out_channels, 1));
    }
    ggml_set_name(x, "r_conv2");
    
    return ggml_add(ctx, residual, x);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_decoder_block(struct ggml_context * ctx,
                                                                  struct ggml_tensor * x,
                                                                  const decoder_block & block,
                                                                  int upsample_rate) {
    if (block.snake_alpha && block.snake_beta) {
        x = apply_snake(ctx, x, block.snake_alpha, block.snake_beta);
        ggml_set_name(x, "dec_blk_snake");
    }
    
    int64_t out_channels = block.conv_t_w->ne[1];
    int kernel_size = block.conv_t_w->ne[0];
    int right_pad = kernel_size - upsample_rate;

    if (state_.max_graph) {
        int64_t in_channels = x->ne[0];
        int64_t seq_len = x->ne[1];
        struct ggml_tensor * x_2d = ggml_reshape_2d(ctx, x, in_channels, seq_len);
        x_2d = max_transpose_conv(ctx, block.conv_t_w, x_2d, upsample_rate, true);
        ggml_set_name(x_2d, "dec_blk_tconv");

        int64_t out_seq_len = x_2d->ne[1] - right_pad;
        x_2d = ggml_view_2d(ctx, x_2d, out_channels, out_seq_len, x_2d->nb[1], 0);
        x = ggml_reshape_3d(ctx, x_2d, out_channels, out_seq_len, 1);
        if (block.conv_t_b) {
            x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv_t_b, out_channels, 1, 1));
        }
        ggml_set_name(x, "dec_blk_tconv_out");
    } else {
        int64_t seq_len = x->ne[0];
        int64_t in_channels = x->ne[1];
        struct ggml_tensor * x_2d = ggml_reshape_2d(ctx, x, seq_len, in_channels);
        x_2d = ggml_conv_transpose_1d(ctx, block.conv_t_w, x_2d, upsample_rate, 0, 1);
        ggml_set_name(x_2d, "dec_blk_tconv");
        
        int64_t new_seq_len = x_2d->ne[0];
        x = ggml_reshape_3d(ctx, x_2d, new_seq_len, out_channels, 1);
        
        int64_t out_seq_len = new_seq_len - right_pad;
        x = ggml_view_3d(ctx, x, out_seq_len, out_channels, 1,
                         x->nb[1], x->nb[2], 0);
        x = ggml_cont(ctx, x);
        
        if (block.conv_t_b) {
            x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv_t_b, 1, out_channels, 1));
        }
        ggml_set_name(x, "dec_blk_tconv_out");
    }
    
    for (int i = 0; i < 3; ++i) {
        x = apply_residual_block(ctx, x, block.res[i]);
        char rname[32]; snprintf(rname, sizeof(rname), "dec_res_%d", i);
        ggml_set_name(x, rname);
    }
    
    return x;
}

struct ggml_cgraph * AudioTokenizerDecoder::build_graph(int32_t n_frames) {
    const auto & cfg = model_.config;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    const size_t max_nodes = state_.max_graph ? QWEN3_TTS_MAX_DEC_MAX_NODES : QWEN3_TTS_DEC_MAX_NODES;
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, max_nodes, false);
    
    static const char * cb_names[16] = {
        "codes_cb0", "codes_cb1", "codes_cb2", "codes_cb3",
        "codes_cb4", "codes_cb5", "codes_cb6", "codes_cb7",
        "codes_cb8", "codes_cb9", "codes_cb10", "codes_cb11",
        "codes_cb12", "codes_cb13", "codes_cb14", "codes_cb15"
    };
    
    struct ggml_tensor * cb_codes_tensors[16];
    for (int cb = 0; cb < 16; ++cb) {
        cb_codes_tensors[cb] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
        ggml_set_name(cb_codes_tensors[cb], cb_names[cb]);
        ggml_set_input(cb_codes_tensors[cb]);
    }
    
    struct ggml_tensor * first_codes = cb_codes_tensors[0];
    
     struct ggml_tensor * first_cb_w = state_.max_graph ? state_.max_codebooks.at(model_.vq_first_codebook) : model_.vq_first_codebook;
     struct ggml_tensor * first_emb = ggml_get_rows(ctx0, first_cb_w, first_codes);
     ggml_set_name(first_emb, "first_emb_raw");
     
     struct ggml_tensor * rest_emb[15];
     for (int cb = 0; cb < 15; ++cb) {
         struct ggml_tensor * cb_codes = cb_codes_tensors[cb + 1];
         struct ggml_tensor * rest_cb_w = state_.max_graph ? state_.max_codebooks.at(model_.vq_rest_codebook[cb]) : model_.vq_rest_codebook[cb];
         rest_emb[cb] = ggml_get_rows(ctx0, rest_cb_w, cb_codes);
         
         if (cb == 0) {
             ggml_set_name(rest_emb[cb], "rest_cb0_emb_raw");
         }
     }
    
     struct ggml_tensor * first_emb_2d = ggml_reshape_2d(ctx0, first_emb, cfg.codebook_dim, n_frames);
     ggml_set_name(first_emb_2d, "first_emb_2d");
     
     struct ggml_tensor * first_proj_weight_2d = ggml_reshape_2d(ctx0, model_.vq_first_output_proj, 
                                                                   cfg.codebook_dim, cfg.hidden_dim);
     struct ggml_tensor * first_proj_2d = ggml_mul_mat(ctx0, first_proj_weight_2d, first_emb_2d);
     ggml_set_name(first_proj_2d, "first_proj_2d");
    
    struct ggml_tensor * rest_proj_weight_2d = ggml_reshape_2d(ctx0, model_.vq_rest_output_proj,
                                                                 cfg.codebook_dim, cfg.hidden_dim);
    
     struct ggml_tensor * sum_cb_emb_2d = ggml_reshape_2d(ctx0, rest_emb[0], cfg.codebook_dim, n_frames);
     ggml_set_name(sum_cb_emb_2d, "rest_cb0_emb_2d");
     for (int cb = 1; cb < 15; ++cb) {
         struct ggml_tensor * cb_emb_2d = ggml_reshape_2d(ctx0, rest_emb[cb], cfg.codebook_dim, n_frames);
         sum_cb_emb_2d = ggml_add(ctx0, sum_cb_emb_2d, cb_emb_2d);
     }
     struct ggml_tensor * rest_proj_2d = ggml_mul_mat(ctx0, rest_proj_weight_2d, sum_cb_emb_2d);
     ggml_set_name(rest_proj_2d, "rest_proj_2d");
    
     struct ggml_tensor * latent_2d = ggml_add(ctx0, first_proj_2d, rest_proj_2d);
     ggml_set_name(latent_2d, "latent_2d");
     
     struct ggml_tensor * cur = nullptr;
     if (state_.max_graph) {
         struct ggml_tensor * latent = ggml_reshape_3d(ctx0, latent_2d, cfg.hidden_dim, n_frames, 1);
         ggml_set_name(latent, "vq_output");
         struct ggml_tensor * latent_padded = ggml_pad_ext(ctx0, latent, 0, 0, 2, 0, 0, 0, 0, 0);
         cur = max_conv(ctx0, model_.pre_conv_w, latent_padded, 1, true);
         if (model_.pre_conv_b) {
             cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.pre_conv_b, cfg.latent_dim, 1, 1));
         }
         ggml_set_name(cur, "pre_conv_output");
         cur = ggml_reshape_2d(ctx0, cur, cfg.latent_dim, n_frames);
         ggml_set_name(cur, "pre_conv_reshaped");
     } else {
         struct ggml_tensor * latent_t = ggml_transpose(ctx0, latent_2d);
         ggml_set_name(latent_t, "latent_t");
         
         struct ggml_tensor * latent_cont = ggml_cont(ctx0, latent_t);
         ggml_set_name(latent_cont, "latent_cont");
         
         struct ggml_tensor * latent = ggml_reshape_3d(ctx0, latent_cont, n_frames, cfg.hidden_dim, 1);
         ggml_set_name(latent, "vq_output");
        
         struct ggml_tensor * latent_for_conv = ggml_cont(ctx0, latent);
         struct ggml_tensor * latent_padded = ggml_pad_ext(ctx0, latent_for_conv, 2, 0, 0, 0, 0, 0, 0, 0);
         cur = ggml_conv_1d(ctx0, model_.pre_conv_w, latent_padded, 1, 0, 1);
         if (model_.pre_conv_b) {
             cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.pre_conv_b, 1, cfg.latent_dim, 1));
         }
         ggml_set_name(cur, "pre_conv_output");
         
         struct ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur, n_frames, cfg.latent_dim);
         struct ggml_tensor * cur_t = ggml_transpose(ctx0, cur_2d);
         cur = ggml_cont(ctx0, cur_t);
         ggml_set_name(cur, "pre_conv_reshaped");
     }
      
      cur = ggml_mul_mat(ctx0, model_.pre_tfm_input_proj_w, cur);
      if (model_.pre_tfm_input_proj_b) {
          cur = ggml_add(ctx0, cur, model_.pre_tfm_input_proj_b);
      }
      
      ggml_set_name(cur, "pre_tfm_input");
     
     struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
     ggml_set_name(positions, "positions");
     ggml_set_input(positions);
     
      ggml_tensor * causal_mask = nullptr;
      if (state_.max_graph) {
          const int mask_rows = ((n_frames + 31) / 32) * 32;
          causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_frames, mask_rows);
          ggml_set_name(causal_mask, "vocoder_causal_mask");
          ggml_set_input(causal_mask);
      }
      for (int i = 0; i < cfg.n_pre_tfm_layers; ++i) {
          cur = apply_pre_tfm_layer(ctx0, cur, model_.pre_tfm_layers[i], n_frames, positions, causal_mask);
      }
      
      if (model_.pre_tfm_norm_w) {
          cur = apply_rms_norm(ctx0, cur, model_.pre_tfm_norm_w, cfg.rms_norm_eps);
      }
      
      cur = ggml_mul_mat(ctx0, model_.pre_tfm_output_proj_w, cur);
      if (model_.pre_tfm_output_proj_b) {
          cur = ggml_add(ctx0, cur, model_.pre_tfm_output_proj_b);
      }
      
      ggml_set_name(cur, "pre_tfm_output");
     
      if (state_.max_graph) {
          cur = ggml_reshape_3d(ctx0, cur, cfg.latent_dim, n_frames, 1);
      } else {
          cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
          cur = ggml_cont(ctx0, cur);
          cur = ggml_reshape_3d(ctx0, cur, n_frames, cfg.latent_dim, 1);
      }
      
      ggml_set_name(cur, "pre_tfm_reshaped");
     
      for (int i = 0; i < 2; ++i) {
          cur = apply_upsample_block(ctx0, cur, model_.upsample[i], i);
      }
      
      ggml_set_name(cur, "upsample_output");
      
      // Causal padding: left pad with 6 (kernel_size - 1 = 7 - 1 = 6)
      if (state_.max_graph) {
          cur = ggml_pad_ext(ctx0, cur, 0, 0, 6, 0, 0, 0, 0, 0);
          cur = max_conv(ctx0, model_.dec0_conv_w, cur, 1, true);
          if (model_.dec0_conv_b) {
              cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.dec0_conv_b, cfg.decoder_dim, 1, 1));
          }
      } else {
          cur = ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
          cur = ggml_conv_1d(ctx0, model_.dec0_conv_w, cur, 1, 0, 1);
          if (model_.dec0_conv_b) {
              cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.dec0_conv_b, 1, cfg.decoder_dim, 1));
          }
      }
      
      ggml_set_name(cur, "dec0_output");
      
      int upsample_rates[4] = {8, 5, 4, 3};
      for (int i = 0; i < 4; ++i) {
          cur = apply_decoder_block(ctx0, cur, model_.dec_blocks[i], upsample_rates[i]);
          char name[32];
          snprintf(name, sizeof(name), "dec%d_output", i + 1);
          ggml_set_name(cur, name);
      }
      
      if (model_.dec5_snake_alpha) {
          cur = apply_snake(ctx0, cur, model_.dec5_snake_alpha, model_.dec5_snake_beta);
      }
      
      ggml_set_name(cur, "dec5_output");
      
      // Causal padding: left pad with 6 (kernel_size - 1 = 7 - 1 = 6)
      if (state_.max_graph) {
          cur = ggml_pad_ext(ctx0, cur, 0, 0, 6, 0, 0, 0, 0, 0);
          cur = max_conv(ctx0, model_.dec6_conv_w, cur, 1, true);
          const int64_t total_samples = cur->ne[1];
          cur = ggml_reshape_2d(ctx0, cur, 1920, n_frames);
          if (model_.dec6_conv_b) {
              cur = ggml_add(ctx0, cur, ggml_reshape_2d(ctx0, model_.dec6_conv_b, 1, 1));
          }
          ggml_set_name(cur, "dec6_output");
          cur = ggml_tanh(ctx0, cur);
          cur = ggml_reshape_1d(ctx0, cur, total_samples);
      } else {
          cur = ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
          cur = ggml_conv_1d(ctx0, model_.dec6_conv_w, cur, 1, 0, 1);
          if (model_.dec6_conv_b) {
              cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.dec6_conv_b, 1, 1, 1));
          }
          ggml_set_name(cur, "dec6_output");
          cur = ggml_tanh(ctx0, cur);
          cur = ggml_reshape_1d(ctx0, cur, cur->ne[0]);
      }
    
    ggml_set_name(cur, "audio");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

bool AudioTokenizerDecoder::decode(const int32_t * codes, int32_t n_frames,
                                    std::vector<float> & samples) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    const auto & cfg = model_.config;
    if (!codes || n_frames <= 0) {
        error_msg_ = "Invalid vocoder input";
        return false;
    }

    // In [channels, time] NPU-Max layout, ne[0] <= 1920 regardless of frame count,
    // so up to 100 frames (8.0s) runs in a single un-chunked pass. For >100 frames,
    // prepend 16 frames of causal overlap context so chunk boundaries have zero pitch/phase dip.
    const int32_t chunk_limit = 100;
    const int32_t left_context = 16;
    const int32_t chunk_step = chunk_limit - left_context;
    if (n_frames > chunk_limit) {
        samples.clear();
        samples.reserve((size_t)n_frames * 1920);
        for (int32_t base = 0; base < n_frames; base += chunk_step) {
            int32_t ctx_frames = std::min(left_context, base);
            int32_t this_chunk = std::min(chunk_step, n_frames - base);
            std::vector<float> chunk_samples;
            if (!decode(codes + (size_t)(base - ctx_frames) * 16, ctx_frames + this_chunk, chunk_samples)) {
                return false;
            }
            const size_t skip_samples = (size_t)ctx_frames * 1920;
            if (chunk_samples.size() > skip_samples) {
                samples.insert(samples.end(), chunk_samples.begin() + skip_samples, chunk_samples.end());
            }
        }
        return true;
    }
    int64_t t_start = ggml_time_us();
    struct ggml_cgraph * gf = build_graph(n_frames);
    int64_t t_graph = ggml_time_us();
    if (state_.max_graph && state_.backend) {
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            struct ggml_tensor * node = ggml_graph_node(gf, i);
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (node->src[s] && (node->src[s]->flags & GGML_TENSOR_FLAG_INPUT)) {
                    ggml_backend_sched_set_tensor_backend(state_.sched, node->src[s], state_.backend);
                }
            }
        }
    }
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
    int64_t t_alloc = ggml_time_us();
    static bool printed_splits = false;
    if (!printed_splits) {
        fprintf(stderr, "  [Vocoder] Graph splits: %d\n",
                ggml_backend_sched_get_n_splits(state_.sched));
        std::map<std::string, int> cpu_ops;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            struct ggml_tensor * node = ggml_graph_node(gf, i);
            if (ggml_backend_sched_get_tensor_backend(state_.sched, node) == state_.backend_cpu) {
                cpu_ops[ggml_op_name(node->op)]++;
            }
        }
        fprintf(stderr, "  [Vocoder] Ops placed on CPU:\n");
        for (auto & kv : cpu_ops) {
            fprintf(stderr, "    - %s: %d\n", kv.first.c_str(), kv.second);
        }
        printed_splits = true;
    }
    
    std::vector<int32_t> cb_codes(n_frames);
    for (int cb = 0; cb < 16; ++cb) {
        char name[32];
        snprintf(name, sizeof(name), "codes_cb%d", cb);
        struct ggml_tensor * cb_tensor = ggml_graph_get_tensor(gf, name);
        if (!cb_tensor) {
            error_msg_ = "Failed to find codes tensor for codebook " + std::to_string(cb);
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
        
        for (int f = 0; f < n_frames; ++f) {
            cb_codes[f] = codes[f * cfg.n_codebooks + cb];
        }
        
        ggml_backend_tensor_set(cb_tensor, cb_codes.data(), 0, n_frames * sizeof(int32_t));
    }
    
    struct ggml_tensor * positions_tensor = ggml_graph_get_tensor(gf, "positions");
    if (positions_tensor) {
        std::vector<int32_t> positions(n_frames);
        for (int i = 0; i < n_frames; ++i) {
            positions[i] = i;
        }
        ggml_backend_tensor_set(positions_tensor, positions.data(), 0, 
                                n_frames * sizeof(int32_t));
    }
    if (state_.max_graph) {
        ggml_tensor * mask = ggml_graph_get_tensor(gf, "vocoder_causal_mask");
        if (mask) {
            std::vector<ggml_fp16_t> data(ggml_nelements(mask));
            for (int64_t row = 0; row < mask->ne[1]; ++row)
                for (int64_t col = 0; col < mask->ne[0]; ++col)
                    data[row * mask->ne[0] + col] = ggml_fp32_to_fp16(col > row ? -INFINITY : 0.0f);
            ggml_backend_tensor_set(mask, data.data(), 0, data.size() * sizeof(ggml_fp16_t));
        }
    }
    int64_t t_inputs = ggml_time_us();
    
    bool is_cpu_backend = ggml_backend_is_cpu(state_.backend);
    bool debug_vocoder = (getenv("QWEN3_TTS_DEBUG_VOCODER") != nullptr);
    if (debug_vocoder) {
        auto debug_cb = [](struct ggml_tensor * t, bool ask, void * user_data) -> bool {
            if (ask) return true;
            const char * name = ggml_get_name(t);
            if (!name || strlen(name) == 0) return true;
            static const char * check_names[] = {
                "codes_cb0", "first_emb_raw", "first_proj_2d", "rest_proj_2d", "latent_2d", "pre_conv_output",
                "pre_tfm_input", "pre_tfm_output", "upsample_output",
                "dec0_output", "dec_blk_snake", "dec_blk_tconv", "dec_blk_tconv_out",
                "r_act1", "r_pad", "r_conv1", "r_act2", "r_conv2",
                "dec_res_0", "dec_res_1", "dec_res_2",
                "dec1_output", "dec2_output", "dec3_output",
                "dec4_output", "dec5_output", "dec6_output", "audio"
            };
            bool match = (strncmp(name, "up0_", 4) == 0 || strncmp(name, "up1_", 4) == 0);
            if (!match) {
                for (const char * cn : check_names) {
                    if (strcmp(name, cn) == 0) { match = true; break; }
                }
            }
            if (getenv("QWEN3_TTS_DEBUG_ALL_NODES")) match = true;
            if (!match) return true;
            bool is_cpu = *(bool *)user_data;
            size_t n = ggml_nelements(t);
            float v0 = 0, v1 = 0;
            double sum = 0, sq = 0;
            size_t step = std::max((size_t)1, n / 100);
            size_t count = 0;
            for (size_t i = 0; i < n; i += step) {
                float v = 0;
                if (t->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_get(t, &v, i * sizeof(float), sizeof(float));
                } else if (t->type == GGML_TYPE_F16) {
                    ggml_fp16_t hv = 0;
                    ggml_backend_tensor_get(t, &hv, i * sizeof(ggml_fp16_t), sizeof(ggml_fp16_t));
                    v = ggml_fp16_to_fp32(hv);
                } else if (t->type == GGML_TYPE_I32) {
                    int32_t iv = 0;
                    ggml_backend_tensor_get(t, &iv, i * sizeof(int32_t), sizeof(int32_t));
                    v = (float)iv;
                }
                if (i == 0) v0 = v;
                sum += v;
                sq += (double)v * v;
                count++;
            }
            if (n > 1) {
                if (t->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_get(t, &v1, sizeof(float), sizeof(float));
                } else if (t->type == GGML_TYPE_F16) {
                    ggml_fp16_t hv = 0;
                    ggml_backend_tensor_get(t, &hv, sizeof(ggml_fp16_t), sizeof(ggml_fp16_t));
                    v1 = ggml_fp16_to_fp32(hv);
                }
            }
            fprintf(stderr, "  [DEBUG %s] %-16s shape=[%5lld,%5lld,%2lld] mean=%10.5f rms=%10.5f v0=%10.5f v1=%10.5f\n",
                    is_cpu ? "CPU" : "GPU",
                    name, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
                    (float)(sum / count), (float)sqrt(sq / count), v0, v1);
            return true;
        };
        ggml_backend_sched_set_eval_callback(state_.sched, debug_cb, &is_cpu_backend);
    }
    
    if (audited_graph_compute("vocoder", state_.sched, gf, &error_msg_) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    int64_t t_compute = ggml_time_us();

    if (debug_vocoder) {
        ggml_backend_sched_set_eval_callback(state_.sched, nullptr, nullptr);
    }
    
    struct ggml_tensor * audio_tensor = ggml_graph_get_tensor(gf, "audio");
    if (!audio_tensor) {
        error_msg_ = "Failed to find audio tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    int64_t n_samples = audio_tensor->ne[0];
    samples.resize(n_samples);
    ggml_backend_tensor_get(audio_tensor, samples.data(), 0, n_samples * sizeof(float));
    int64_t t_end = ggml_time_us();
    
    fprintf(stderr, "  [Vocoder Breakdown] build_graph: %.1f ms | alloc: %.1f ms | inputs: %.1f ms | compute: %.1f ms | tensor_get: %.1f ms | total: %.1f ms\n",
            (t_graph - t_start) / 1000.0,
            (t_alloc - t_graph) / 1000.0,
            (t_inputs - t_alloc) / 1000.0,
            (t_compute - t_inputs) / 1000.0,
            (t_end - t_compute) / 1000.0,
            (t_end - t_start) / 1000.0);
    
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

void free_audio_decoder_model(audio_decoder_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
}

} // namespace qwen3_tts
