#include "mimi_encoder.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstdio>
#include <sstream>

namespace qwen3_tts {

// ============================================================
// Utility helpers (CPU-only reference conditioning for ICL)
// ============================================================

// Helper: get tensor float data into a host vector
static std::vector<float> tensor_to_float(struct ggml_tensor * t) {
    if (!t) return {};
    size_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        // For quantized types: use ggml_get_type_traits()->to_float
        size_t raw_bytes = ggml_nbytes(t);
        std::vector<uint8_t> raw(raw_bytes);
        ggml_backend_tensor_get(t, raw.data(), 0, raw_bytes);
        const struct ggml_type_traits * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(raw.data(), out.data(), (int64_t)n);
        }
        // else out stays zero — type unsupported, caller should notice missing values
    }
    return out;
}

// ELU activation in-place
void MimiEncoder::apply_elu(std::vector<float> & x) {
    for (float & v : x) {
        if (v < 0.0f) v = expf(v) - 1.0f;
    }
}

// Layer norm: normalize over last `dim` elements for each of `len/dim` vectors
void MimiEncoder::apply_layer_norm(float * data, int32_t len, int32_t dim,
                                    const float * w, const float * b, float eps) {
    int32_t n_vec = len / dim;
    for (int32_t i = 0; i < n_vec; ++i) {
        float * row = data + i * dim;
        // mean
        double sum = 0.0;
        for (int32_t d = 0; d < dim; ++d) sum += row[d];
        float mean = (float)(sum / dim);
        // variance
        double var = 0.0;
        for (int32_t d = 0; d < dim; ++d) {
            float diff = row[d] - mean;
            var += (double)(diff * diff);
        }
        float inv_std = 1.0f / sqrtf((float)(var / dim) + eps);
        // normalize + scale + shift
        for (int32_t d = 0; d < dim; ++d) {
            row[d] = (row[d] - mean) * inv_std;
            if (w) row[d] *= w[d];
            if (b) row[d] += b[d];
        }
    }
}

// ============================================================
// Causal Conv1d (CPU)
// input:  [in_len, in_ch]  row-major (time-major)
// weight: stored in GGUF as PyTorch's [OC, IC, K] row-major.
//         gguf-py reverses the dimensions in metadata (ne=[K,IC,OC])
//         but the actual bytes are the PyTorch C-contiguous layout.
//         So element w[oc, ic, k] is at raw offset: oc*IC*K + ic*K + k
// output: [out_len, out_ch]
// ============================================================
void MimiEncoder::apply_causal_conv1d(
        const float * input, int32_t in_len, int32_t in_ch, int32_t out_ch,
        const float * w, const float * b,
        int32_t kernel, int32_t stride, int32_t dilation,
        int32_t padding_left,
        std::vector<float> & output, int32_t & out_len) {

    int32_t eff_k   = (kernel - 1) * dilation + 1;
    int32_t padded  = in_len + padding_left;
    out_len = (padded - eff_k) / stride + 1;
    if (out_len <= 0) { out_len = 0; output.clear(); return; }

    output.assign((size_t)out_len * out_ch, 0.0f);

    // Weight layout: PyTorch [OC, IC, K] stored C-contiguous in GGUF bytes.
    // w[oc, ic, k] = w[oc * in_ch * kernel + ic * kernel + k]
    int32_t scalar_start = 0;
#ifdef QWEN3_TTS_MIMI_CONV_SIMD
    // Pack independent output channels together so the compiler can vectorize
    // without reassociating a reduction. Each lane still accumulates k then ic
    // in exactly the scalar order, including bias and causal padding. Keep this
    // temporary local: no model lifetime/cache invalidation or extra threads.
    constexpr int32_t lanes = 8;
    // On the validated Release build, pointwise convolutions round SIMD
    // products before their ordered sum, while strided weights use FMA.
    // Match both below. Keep the original loop for irregular input channels
    // (including the one-channel input convolution and reduction tails).
    scalar_start = (in_ch >= 16 && in_ch % 16 == 0) ? out_ch / lanes * lanes : 0;
    if (scalar_start > 0) {
        const size_t block_size = (size_t)kernel * in_ch * lanes;
        std::vector<float> packed((size_t)scalar_start * in_ch * kernel);
        for (int32_t oc = 0; oc < scalar_start; oc += lanes) {
            float * block = packed.data() + (size_t)(oc / lanes) * block_size;
            for (int32_t k = 0; k < kernel; ++k) {
                for (int32_t ic = 0; ic < in_ch; ++ic) {
                    for (int32_t lane = 0; lane < lanes; ++lane) {
                        block[((size_t)k * in_ch + ic) * lanes + lane] =
                            w[(size_t)(oc + lane) * in_ch * kernel + (size_t)ic * kernel + k];
                    }
                }
            }
        }
        for (int32_t t = 0; t < out_len; ++t) {
            for (int32_t oc = 0; oc < scalar_start; oc += lanes) {
                float sums[lanes];
                for (int32_t lane = 0; lane < lanes; ++lane)
                    sums[lane] = b ? b[oc + lane] : 0.0f;
                const float * block = packed.data() + (size_t)(oc / lanes) * block_size;
                for (int32_t k = 0; k < kernel; ++k) {
                    const int32_t src_t = t * stride + k * dilation - padding_left;
                    if (src_t < 0 || src_t >= in_len) continue;
                    const float * row = input + (size_t)src_t * in_ch;
                    for (int32_t ic = 0; ic < in_ch; ++ic) {
                        const float * weights = block + ((size_t)k * in_ch + ic) * lanes;
                        for (int32_t lane = 0; lane < lanes; ++lane) {
                            if (kernel == 1) {
                                // Contiguous scalar weights use rounded SIMD
                                // products followed by an ordered scalar sum.
                                const float product = row[ic] * weights[lane];
                                sums[lane] += product;
                            } else {
                                sums[lane] += row[ic] * weights[lane];
                            }
                        }
                    }
                }
                for (int32_t lane = 0; lane < lanes; ++lane)
                    output[(size_t)t * out_ch + oc + lane] = sums[lane];
            }
        }
    }
#endif
    for (int32_t t = 0; t < out_len; ++t) {
        for (int32_t oc = scalar_start; oc < out_ch; ++oc) {
            float sum = b ? b[oc] : 0.0f;
            for (int32_t k = 0; k < kernel; ++k) {
                int32_t src_t = t * stride + k * dilation - padding_left;
                if (src_t < 0 || src_t >= in_len) continue;
                for (int32_t ic = 0; ic < in_ch; ++ic) {
                    sum += input[(size_t)src_t * in_ch + ic]
                         * w[(size_t)oc * in_ch * kernel
                             + (size_t)ic * kernel
                             + k];
                }
            }
            output[(size_t)t * out_ch + oc] = sum;
        }
    }
}

// ============================================================
// Load model
// ============================================================
MimiEncoder::MimiEncoder()  = default;
MimiEncoder::~MimiEncoder() { unload_model(); }

void MimiEncoder::unload_model() {
    free_mimi_encoder_model(model_);
}

void free_mimi_encoder_model(mimi_encoder_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
    model.enc_resnet.clear();
    model.enc_down.clear();
    model.tfm_layers.clear();
    model.semantic_cbs.clear();
    model.acoustic_cbs.clear();
}

bool MimiEncoder::load_model(const std::string & model_path) {
    unload_model();

    GGUFLoader loader;
    if (!loader.open(model_path)) {
        error_msg_ = loader.get_error();
        return false;
    }

    // Read config from GGUF
    auto & cfg = model_.config;
    cfg.sample_rate       = loader.get_u32("qwen3-tts.tokenizer.sample_rate", 24000);
    cfg.num_valid_quantizers = loader.get_u32("qwen3-tts-tokenizer.num_codebooks",
                               loader.get_u32("qwen3-tts.tokenizer.num_codebooks", 16));
    cfg.num_quantizers    = loader.get_u32("mimi.num_quantizers", 32);
    cfg.codebook_size     = loader.get_u32("mimi.codebook_size", 2048);
    cfg.codebook_dim      = loader.get_u32("mimi.codebook_dim", 256);
    cfg.vq_hidden_dim     = loader.get_u32("mimi.vq_hidden_dim", 256);
    cfg.hidden_size       = loader.get_u32("mimi.hidden_size", 512);
    cfg.num_filters       = loader.get_u32("mimi.num_filters", 64);
    cfg.num_hidden_layers = loader.get_u32("mimi.num_hidden_layers", 8);
    cfg.num_attention_heads = loader.get_u32("mimi.num_attention_heads", 8);
    cfg.head_dim          = loader.get_u32("mimi.head_dim", 64);
    cfg.intermediate_size = loader.get_u32("mimi.intermediate_size", 2048);
    cfg.sliding_window    = loader.get_u32("mimi.sliding_window", 250);
    cfg.rope_theta        = loader.get_f32("mimi.rope_theta", 10000.0f);
    cfg.norm_eps          = loader.get_f32("mimi.norm_eps", 1e-5f);
    cfg.num_semantic_quantizers = loader.get_u32("mimi.num_semantic_quantizers", 1);

    // Count mimi_enc.* tensors
    int64_t n_tensors = loader.get_n_tensors();
    int enc_count = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (name && strncmp(name, "mimi_enc.", 9) == 0) ++enc_count;
    }

    if (enc_count == 0) {
        error_msg_ = "No Mimi encoder tensors (mimi_enc.*) found in tokenizer GGUF. "
                     "Please re-convert the tokenizer with an updated convert_tokenizer_to_gguf.py.";
        return false;
    }

    // Allocate GGML context
    size_t ctx_size = (size_t)enc_count * ggml_tensor_overhead() + 512;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    model_.ctx = ggml_init(params);
    if (!model_.ctx) { error_msg_ = "Failed to init GGML context"; return false; }

    struct gguf_context * gguf_ctx  = loader.get_ctx();
    struct ggml_context * meta_ctx  = loader.get_meta_ctx();

    // Read upsampling_ratios array from GGUF (needs gguf_ctx)
    {
        int64_t idx = gguf_find_key(gguf_ctx, "mimi.upsampling_ratios");
        if (idx >= 0) {
            size_t n = gguf_get_arr_n(gguf_ctx, idx);
            if (n >= 4) {
                const void * data = gguf_get_arr_data(gguf_ctx, idx);
                enum gguf_type elem_t = gguf_get_arr_type(gguf_ctx, idx);
                for (int i = 0; i < 4 && i < (int)n; ++i) {
                    int32_t v = 8;
                    if      (elem_t == GGUF_TYPE_INT32)  v = ((const int32_t  *)data)[i];
                    else if (elem_t == GGUF_TYPE_UINT32) v = (int32_t)(((const uint32_t*)data)[i]);
                    else if (elem_t == GGUF_TYPE_INT64)  v = (int32_t)(((const int64_t *)data)[i]);
                    cfg.upsampling_ratios[i] = v;
                }
            }
        }
    }

    // Duplicate all mimi_enc.* tensors into our context
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name || strncmp(name, "mimi_enc.", 9) != 0) continue;
        struct ggml_tensor * mt = ggml_get_tensor(meta_ctx, name);
        if (!mt) continue;
        struct ggml_tensor * t = ggml_dup_tensor(model_.ctx, mt);
        ggml_set_name(t, name);
        model_.tensors[name] = t;
    }

    // Load data from file
    if (!load_tensor_data_from_file(model_path, gguf_ctx, model_.ctx,
                                     model_.tensors, model_.buffer,
                                     error_msg_, GGML_BACKEND_DEVICE_TYPE_CPU)) {
        return false;
    }

    // Helper lambda to find a tensor by name
    auto get = [&](const std::string & n) -> struct ggml_tensor * {
        auto it = model_.tensors.find(n);
        return (it != model_.tensors.end()) ? it->second : nullptr;
    };

    // ---- Wire up conv encoder layers ----
    model_.enc_conv0.w = get("mimi_enc.enc.0.conv.weight");
    model_.enc_conv0.b = get("mimi_enc.enc.0.conv.bias");
    model_.enc_conv0.stride   = 1;
    model_.enc_conv0.dilation = 1;
    model_.enc_conv0.padding_left = cfg.kernel_size - 1;  // causal: full left pad

    // Encoder stages: 4 stages in reverse upsampling order
    // upsampling_ratios=[8,6,5,4] -> encoder strides=[4,5,6,8] (reversed)
    int32_t ratios[4];
    for (int i = 0; i < 4; ++i) ratios[i] = cfg.upsampling_ratios[3 - i];

    model_.enc_resnet.resize(4);
    model_.enc_down.resize(4);

    // Layer indices in the SEANet encoder (from tensor names):
    // layer 0:  initial conv
    // layer 1:  resnet block (layers.1.block.1, layers.1.block.3)
    // layer 2:  ELU (no weights)
    // layer 3:  downsample conv (stride=4, k=8)
    // layer 4:  resnet block
    // layer 5:  ELU
    // layer 6:  downsample conv (stride=5, k=10)
    // layer 7:  resnet block
    // layer 8:  ELU
    // layer 9:  downsample conv (stride=6, k=12)
    // layer 10: resnet block
    // layer 11: ELU
    // layer 12: downsample conv (stride=8, k=16)
    // layer 13: ELU (no weights)
    // layer 14: final conv (hidden_size*8 -> hidden_size, k=3)

    // Resnet block layer indices (in model): 1, 4, 7, 10
    int32_t res_idx[4] = {1, 4, 7, 10};
    // Downsample conv layer indices: 3, 6, 9, 12
    int32_t down_idx[4] = {3, 6, 9, 12};

    for (int s = 0; s < 4; ++s) {
        // Resnet block: block.1 = compressed conv, block.3 = expand conv
        char buf[128];
        snprintf(buf, sizeof(buf), "mimi_enc.enc.%d.block.1.conv.weight", res_idx[s]);
        model_.enc_resnet[s].conv1.w = get(buf);
        snprintf(buf, sizeof(buf), "mimi_enc.enc.%d.block.1.conv.bias", res_idx[s]);
        model_.enc_resnet[s].conv1.b = get(buf);
        model_.enc_resnet[s].conv1.stride   = 1;
        model_.enc_resnet[s].conv1.dilation = 1;  // dilation_growth_rate^0 = 1 (num_residual_layers=1)
        model_.enc_resnet[s].conv1.padding_left = cfg.residual_kernel_size - 1;

        snprintf(buf, sizeof(buf), "mimi_enc.enc.%d.block.3.conv.weight", res_idx[s]);
        model_.enc_resnet[s].conv2.w = get(buf);
        snprintf(buf, sizeof(buf), "mimi_enc.enc.%d.block.3.conv.bias", res_idx[s]);
        model_.enc_resnet[s].conv2.b = get(buf);
        model_.enc_resnet[s].conv2.stride   = 1;
        model_.enc_resnet[s].conv2.dilation = 1;
        model_.enc_resnet[s].conv2.padding_left = 0;  // kernel=1, no padding

        // Downsample conv: stride = ratios[s], kernel = 2*ratios[s]
        snprintf(buf, sizeof(buf), "mimi_enc.enc.%d.conv.weight", down_idx[s]);
        model_.enc_down[s].w = get(buf);
        snprintf(buf, sizeof(buf), "mimi_enc.enc.%d.conv.bias", down_idx[s]);
        model_.enc_down[s].b = get(buf);
        model_.enc_down[s].stride   = ratios[s];
        model_.enc_down[s].dilation = 1;
        // Causal padding for downsample: padding_total = kernel - stride = ratios[s]
        model_.enc_down[s].padding_left = ratios[s];
    }

    // Final conv
    model_.enc_final.w = get("mimi_enc.enc.14.conv.weight");
    model_.enc_final.b = get("mimi_enc.enc.14.conv.bias");
    model_.enc_final.stride   = 1;
    model_.enc_final.dilation = 1;
    model_.enc_final.padding_left = cfg.last_kernel_size - 1;

    // ---- Wire up transformer layers ----
    model_.tfm_layers.resize(cfg.num_hidden_layers);
    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        char buf[128];
        auto & ly = model_.tfm_layers[l];
        snprintf(buf, sizeof(buf), "mimi_enc.tfm.%d.attn_in_norm.weight", l);
        ly.attn_in_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "mimi_enc.tfm.%d.attn_in_norm.bias", l);
        ly.attn_in_norm_b = get(buf);
        snprintf(buf, sizeof(buf), "mimi_enc.tfm.%d.attn_q.weight", l);
        ly.attn_q_w = get(buf);
        snprintf(buf, sizeof(buf), "mimi_enc.tfm.%d.attn_k.weight", l);
        ly.attn_k_w = get(buf);
        snprintf(buf, sizeof(buf), "mimi_enc.tfm.%d.attn_v.weight", l);
        ly.attn_v_w = get(buf);
        snprintf(buf, sizeof(buf), "mimi_enc.tfm.%d.attn_o.weight", l);
        ly.attn_o_w = get(buf);
        snprintf(buf, sizeof(buf), "mimi_enc.tfm.%d.attn_scale", l);
        ly.attn_layer_scale = get(buf);
        snprintf(buf, sizeof(buf), "mimi_enc.tfm.%d.ffn_in_norm.weight", l);
        ly.ffn_in_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "mimi_enc.tfm.%d.ffn_in_norm.bias", l);
        ly.ffn_in_norm_b = get(buf);
        snprintf(buf, sizeof(buf), "mimi_enc.tfm.%d.ffn_fc1.weight", l);
        ly.ffn_fc1_w = get(buf);
        snprintf(buf, sizeof(buf), "mimi_enc.tfm.%d.ffn_fc2.weight", l);
        ly.ffn_fc2_w = get(buf);
        snprintf(buf, sizeof(buf), "mimi_enc.tfm.%d.ffn_scale", l);
        ly.ffn_layer_scale = get(buf);
    }

    // ---- Wire up downsample conv ----
    model_.downsample.w = get("mimi_enc.downsample.conv.weight");
    model_.downsample.b = nullptr;  // no bias per model inspection
    // k=4, stride=2, causal: padding_left = k - stride = 2
    model_.downsample.stride   = 2;
    model_.downsample.dilation = 1;
    model_.downsample.padding_left = 2;  // (4-1)*1 + 1 - 2 = 2

    // ---- Wire up RVQ codebooks ----
    // Semantic: 1 codebook
    model_.semantic_input_proj = get("mimi_enc.rvq_sem.input_proj.weight");
    model_.semantic_cbs.resize(1);
    model_.semantic_cbs[0].embed_sum     = get("mimi_enc.rvq_sem.cb.0.embed_sum");
    model_.semantic_cbs[0].cluster_usage = get("mimi_enc.rvq_sem.cb.0.cluster_usage");

    // Acoustic: num_valid_quantizers - 1 codebooks
    int32_t n_acoustic = cfg.num_valid_quantizers - 1;
    model_.acoustic_input_proj = get("mimi_enc.rvq_acou.input_proj.weight");
    model_.acoustic_cbs.resize(n_acoustic);
    for (int32_t i = 0; i < n_acoustic; ++i) {
        char buf[128];
        snprintf(buf, sizeof(buf), "mimi_enc.rvq_acou.cb.%d.embed_sum", i);
        model_.acoustic_cbs[i].embed_sum     = get(buf);
        snprintf(buf, sizeof(buf), "mimi_enc.rvq_acou.cb.%d.cluster_usage", i);
        model_.acoustic_cbs[i].cluster_usage = get(buf);
    }

    // Load pre-normalized codebook embeddings into host memory.
    // The converter (convert_tokenizer_to_gguf.py) already divides embed_sum
    // by cluster_usage before saving — so the GGUF tensor IS the normalized
    // embedding.  We just copy it into cb.embed for fast host-side lookup.
    // cluster_usage is NOT written to GGUF (skipped by the converter).
    auto normalize_codebook = [&](mimi_codebook & cb) {
        if (!cb.embed_sum) return;
        // GGML tensor shape: ne[0] = codebook_dim (256), ne[1] = codebook_size (2048)
        const int32_t cb_size = (int32_t)cb.embed_sum->ne[1];  // 2048
        const int32_t cb_dim  = (int32_t)cb.embed_sum->ne[0];  // 256

        // tensor_to_float returns data in GGML row-major order:
        // element [cb_size_idx, dim_idx] is at offset cb_size_idx * cb_dim + dim_idx
        std::vector<float> raw = tensor_to_float(cb.embed_sum);

        cb.embed.resize((size_t)cb_size * cb_dim);
        memcpy(cb.embed.data(), raw.data(), (size_t)cb_size * cb_dim * sizeof(float));
    };

    for (auto & cb : model_.semantic_cbs)  normalize_codebook(cb);
    for (auto & cb : model_.acoustic_cbs)  normalize_codebook(cb);

    // Pre-convert transformer weight tensors to float for fast inference
    for (auto & ly : model_.tfm_layers) {
        auto cache = [](struct ggml_tensor * t, std::vector<float> & out) {
            if (t) out = tensor_to_float(t);
        };
        cache(ly.attn_in_norm_w,   ly.attn_in_norm_w_f);
        cache(ly.attn_in_norm_b,   ly.attn_in_norm_b_f);
        cache(ly.attn_q_w,         ly.attn_q_w_f);
        cache(ly.attn_k_w,         ly.attn_k_w_f);
        cache(ly.attn_v_w,         ly.attn_v_w_f);
        cache(ly.attn_o_w,         ly.attn_o_w_f);
        if (ly.attn_layer_scale)
            ly.attn_scale_f = tensor_to_float(ly.attn_layer_scale);
        else
            ly.attn_scale_f.assign(cfg.hidden_size, 1.0f);
        cache(ly.ffn_in_norm_w,    ly.ffn_in_norm_w_f);
        cache(ly.ffn_in_norm_b,    ly.ffn_in_norm_b_f);
        cache(ly.ffn_fc1_w,        ly.ffn_fc1_w_f);
        cache(ly.ffn_fc2_w,        ly.ffn_fc2_w_f);
        if (ly.ffn_layer_scale)
            ly.ffn_scale_f = tensor_to_float(ly.ffn_layer_scale);
        else
            ly.ffn_scale_f.assign(cfg.hidden_size, 1.0f);
    }

    fprintf(stderr, "  MimiEncoder loaded: enc_tensors=%d\n", enc_count);
    return true;
}

// ============================================================
// Step 1: SEANet Conv Encoder
// Input:  samples [n_samples]
// Output: hidden [t_out, hidden_size]  (time-major)
// ============================================================
bool MimiEncoder::run_conv_encoder(const float * samples, int32_t n_samples,
                                    std::vector<float> & hidden_out, int32_t & t_out) {
    const auto & cfg = model_.config;

    // Reshape input to [n_samples, 1] (time-major, 1 channel)
    std::vector<float> cur(samples, samples + n_samples);
    int32_t cur_len = n_samples;
    int32_t cur_ch  = 1;

    // Helper to read a weight tensor to host float [OC, IC, K]
    auto get_w = [](struct ggml_tensor * t) -> std::vector<float> {
        if (!t) return {};
        return tensor_to_float(t);
    };

    // Layer 0: initial conv (1 -> num_filters, k=7, causal)
    // No activation after this layer — Python layers[0] is MimiConv1d with no activation
    {
        auto & c = model_.enc_conv0;
        if (!c.w) { error_msg_ = "enc_conv0 weight missing"; return false; }
        std::vector<float> w = get_w(c.w);
        std::vector<float> b = c.b ? get_w(c.b) : std::vector<float>();
        std::vector<float> out;
        int32_t out_len;
        apply_causal_conv1d(cur.data(), cur_len, cur_ch, cfg.num_filters,
                            w.data(), b.empty() ? nullptr : b.data(),
                            7, 1, 1, 6,  // k=7, stride=1, dil=1, pad_left=6
                            out, out_len);
        // No ELU here — Python layers[0] is just Conv1d
        cur     = std::move(out);
        cur_len = out_len;
        cur_ch  = cfg.num_filters;
    }

    // 4 stages: ResnetBlock + ELU + Downsample
    int32_t ratios[4];
    for (int i = 0; i < 4; ++i) ratios[i] = cfg.upsampling_ratios[3 - i];

    for (int s = 0; s < 4; ++s) {
        int32_t ch_in  = cur_ch;
        int32_t ch_hid = ch_in / cfg.compress;  // compressed channels in resnet

        // ResnetBlock: ELU -> conv1(k=3,dil) -> ELU -> conv2(k=1)
        {
            auto & rb = model_.enc_resnet[s];
            if (!rb.conv1.w) {
                error_msg_ = "resnet conv1 weight missing for stage " + std::to_string(s);
                return false;
            }

            // Save residual
            std::vector<float> residual = cur;

            // ELU
            apply_elu(cur);

            // conv1 (ch_in -> ch_hid, k=3, dilation)
            std::vector<float> wc1 = get_w(rb.conv1.w);
            std::vector<float> bc1 = rb.conv1.b ? get_w(rb.conv1.b) : std::vector<float>();
            // dilation_growth_rate^j where j=0 (only 1 residual layer per stage) = 1
            const int32_t dil = 1;
            int32_t pad1 = (cfg.residual_kernel_size - 1) * dil;  // causal
            std::vector<float> out1;
            int32_t out1_len;
            apply_causal_conv1d(cur.data(), cur_len, ch_in, ch_hid,
                                wc1.data(), bc1.empty() ? nullptr : bc1.data(),
                                cfg.residual_kernel_size, 1, dil, pad1,
                                out1, out1_len);
            apply_elu(out1);

            // conv2 (ch_hid -> ch_in, k=1)
            std::vector<float> wc2 = get_w(rb.conv2.w);
            std::vector<float> bc2 = rb.conv2.b ? get_w(rb.conv2.b) : std::vector<float>();
            std::vector<float> out2;
            int32_t out2_len;
            apply_causal_conv1d(out1.data(), out1_len, ch_hid, ch_in,
                                wc2.data(), bc2.empty() ? nullptr : bc2.data(),
                                1, 1, 1, 0,
                                out2, out2_len);

            // Residual add (lengths must match)
            if (out2_len != cur_len) {
                // Trim/pad to match — should be same since k=1 no padding change
                out2.resize((size_t)cur_len * ch_in, 0.0f);
                out2_len = cur_len;
            }
            for (size_t i = 0; i < (size_t)cur_len * ch_in; ++i) {
                out2[i] += residual[i];
            }
            cur = std::move(out2);
        }

        // ELU
        apply_elu(cur);

        // Downsample conv
        {
            auto & dc = model_.enc_down[s];
            if (!dc.w) {
                error_msg_ = "downsample weight missing for stage " + std::to_string(s);
                return false;
            }
            int32_t ratio = ratios[s];
            int32_t k = ratio * 2;
            int32_t ch_out = ch_in * 2;
            // Causal padding left = k - ratio (= padding_total)
            int32_t pad_left = k - ratio;
            // Python MimiConv1d also adds right padding ("extra_padding") so that
            // n_output_frames = ceil((in_len - k + pad_total) / stride + 1)
            // extra_padding = ceil_frames*stride + k - pad_total - in_len
            {
                int32_t eff_k = k;  // dilation=1
                int32_t padded = cur_len + pad_left;
                // exact frame count with ceil
                int64_t n_frames_ceil = ((int64_t)(padded - eff_k) + ratio - 1) / ratio + 1;
                // floor frame count
                int32_t n_frames_floor = (padded - eff_k) / ratio + 1;
                if ((int32_t)n_frames_ceil > n_frames_floor) {
                    // Need right padding
                    int32_t ideal_len = (int32_t)((n_frames_ceil - 1) * ratio + eff_k - pad_left);
                    int32_t extra = ideal_len - cur_len;
                    if (extra > 0) {
                        cur.resize((size_t)(cur_len + extra) * ch_in, 0.0f);
                        cur_len += extra;
                    }
                }
            }
            std::vector<float> wd = get_w(dc.w);
            std::vector<float> bd = dc.b ? get_w(dc.b) : std::vector<float>();
            std::vector<float> out;
            int32_t out_len;
            apply_causal_conv1d(cur.data(), cur_len, ch_in, ch_out,
                                wd.data(), bd.empty() ? nullptr : bd.data(),
                                k, ratio, 1, pad_left,
                                out, out_len);
            cur     = std::move(out);
            cur_len = out_len;
            cur_ch  = ch_out;
        }
    }

    // ELU
    apply_elu(cur);

    // Final conv (ch_in -> hidden_size, k=last_kernel_size=3)
    {
        auto & fc = model_.enc_final;
        if (!fc.w) { error_msg_ = "enc_final weight missing"; return false; }
        int32_t pad_f = cfg.last_kernel_size - 1;
        std::vector<float> wf = get_w(fc.w);
        std::vector<float> bf = fc.b ? get_w(fc.b) : std::vector<float>();
        std::vector<float> out;
        int32_t out_len;
        apply_causal_conv1d(cur.data(), cur_len, cur_ch, cfg.hidden_size,
                            wf.data(), bf.empty() ? nullptr : bf.data(),
                            cfg.last_kernel_size, 1, 1, pad_f,
                            out, out_len);
        hidden_out = std::move(out);
        t_out      = out_len;
    }

    return true;
}

// ============================================================
// Step 2: Transformer (8 layers, sliding-window attention)
// Input/output: [n_frames, hidden_size]
// ============================================================
// Step 2: Transformer — GGML-accelerated matmuls for numerical accuracy
// Input/output: [n_frames, hidden_size]
// ============================================================
bool MimiEncoder::run_transformer(const std::vector<float> & hidden_in, int32_t n_frames,
                                   std::vector<float> & hidden_out) {
    const auto & cfg   = model_.config;
    const int32_t H    = cfg.hidden_size;
    const int32_t n_heads = cfg.num_attention_heads;
    const int32_t hd   = cfg.head_dim;
    const float   eps  = cfg.norm_eps;
    const int32_t I    = cfg.intermediate_size;

    hidden_out = hidden_in;  // [n_frames, H]

    // ---- GGML context for temporary graph tensors ----
    // Per matmul we need: weight [OC*IC*4B] + input [T*IC*4B] + result [T*OC*4B]
    // Per layer: 5 matmuls (Q,K,V,O,fc1 with fc2) + per-head QK^T [T*T]
    // Worst case: fc1 needs 469*2048*4 + 512*2048*4 ≈ 8MB compute
    // QK^T per head: 469*469*4 = 879KB, × we realloc fresh context each call
    const size_t layer_buf = (size_t)n_frames * I * 3 * sizeof(float)   // fc1 worst case
                           + (size_t)H * I * sizeof(float)               // weight
                           + (size_t)n_frames * n_frames * sizeof(float) // QK^T per head
                           + (size_t)n_frames * hd * sizeof(float)       // head slice
                           + 128 * ggml_tensor_overhead()
                           + 8192;
    const size_t ctx_buf_size = layer_buf + 4 * 1024 * 1024;            // 4MB extra

    struct ggml_init_params gparams = {
        /*.mem_size   =*/ ctx_buf_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    // Create one CPU backend once
    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    if (!backend_cpu) {
        error_msg_ = "failed to init CPU backend for Mimi transformer";
        return false;
    }

    // Helper: matrix multiply [n_frames, A] x [A, B] -> [n_frames, B]
    // GGML mul_mat: A=[B, A] (column-major B), B=[A, n_frames] => result [B, n_frames]
    // We want: result[t, b] = sum_a input[t,a] * w[b,a]  (w stored [B, A] in PyTorch)
    // ggml_mul_mat(ctx, w_tensor, input_tensor) where w_tensor is [A, B] in GGML dims
    // and input_tensor is [A, n_frames] -> result is [B, n_frames]
    // So we feed input as [H, n_frames] (column-major for GGML) and w as [H, H] ggml dims.

    // Build RoPE frequencies
    std::vector<float> inv_freq(hd / 2);
    for (int32_t i = 0; i < hd / 2; ++i) {
        inv_freq[i] = 1.0f / powf(cfg.rope_theta, (float)(2 * i) / (float)hd);
    }

    // Helper lambdas working with raw float vectors (no GGML graph for non-matmul ops)
    auto layer_norm = [&](std::vector<float> & x, int32_t dim,
                          const std::vector<float> & w, const std::vector<float> & b) {
        apply_layer_norm(x.data(), (int32_t)x.size(), dim, w.data(), b.data(), eps);
    };

    // GGML matmul helper: weight [OC, IC] (PyTorch), input [n_frames, IC]
    // => output [n_frames, OC] using GGML's optimized ggml_mul_mat
    auto matmul_ggml = [&](const std::vector<float> & weight, int32_t OC, int32_t IC,
                            const std::vector<float> & input, int32_t T,
                            std::vector<float> & output) -> bool {
        struct ggml_context * ctx = ggml_init(gparams);
        if (!ctx) { error_msg_ = "ggml_init failed in matmul_ggml"; return false; }

        struct ggml_tensor * wt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IC, OC);
        memcpy(wt->data, weight.data(), (size_t)OC * IC * sizeof(float));

        struct ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IC, T);
        memcpy(xt->data, input.data(), (size_t)T * IC * sizeof(float));

        struct ggml_tensor * rt = ggml_mul_mat(ctx, wt, xt);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, rt);
        ggml_backend_graph_compute(backend_cpu, gf);

        output.resize((size_t)T * OC);
        memcpy(output.data(), rt->data, (size_t)T * OC * sizeof(float));

        ggml_free(ctx);
        return true;
    };

    for (int32_t l = 0; l < cfg.num_hidden_layers; ++l) {
        auto & ly = model_.tfm_layers[l];
        if (ly.attn_q_w_f.empty()) continue;

        // === Self-attention ===
        // 1. LayerNorm
        std::vector<float> normed(hidden_out);
        layer_norm(normed, H, ly.attn_in_norm_w_f, ly.attn_in_norm_b_f);

        // 2. Q, K, V projections via GGML matmul
        std::vector<float> Q, K, V;
        if (!matmul_ggml(ly.attn_q_w_f, H, H, normed, n_frames, Q)) { ggml_backend_free(backend_cpu); return false; }
        if (!matmul_ggml(ly.attn_k_w_f, H, H, normed, n_frames, K)) { ggml_backend_free(backend_cpu); return false; }
        if (!matmul_ggml(ly.attn_v_w_f, H, H, normed, n_frames, V)) { ggml_backend_free(backend_cpu); return false; }

        // 3. Apply RoPE to Q and K
        for (int32_t t = 0; t < n_frames; ++t) {
            for (int32_t h = 0; h < n_heads; ++h) {
                float * q_row = Q.data() + (size_t)t * H + (size_t)h * hd;
                float * k_row = K.data() + (size_t)t * H + (size_t)h * hd;
                for (int32_t d = 0; d < hd / 2; ++d) {
                    float freq  = (float)t * inv_freq[d];
                    float cos_f = cosf(freq);
                    float sin_f = sinf(freq);
                    float q0 = q_row[d], q1 = q_row[d + hd/2];
                    float k0 = k_row[d], k1 = k_row[d + hd/2];
                    q_row[d]       = q0 * cos_f - q1 * sin_f;
                    q_row[d+hd/2]  = q1 * cos_f + q0 * sin_f;
                    k_row[d]       = k0 * cos_f - k1 * sin_f;
                    k_row[d+hd/2]  = k1 * cos_f + k0 * sin_f;
                }
            }
        }

        // 4. Scaled dot-product attention with sliding window
        // Use GGML matmul for QK^T and AV to match PyTorch numerics
        const float scale = 1.0f / sqrtf((float)hd);
        std::vector<float> attn_out(n_frames * H, 0.0f);

        // Process head by head; each head: Q_h=[T,hd], K_h=[T,hd], V_h=[T,hd]
        // QK^T: [T, hd] x [hd, T] -> [T, T] via ggml_mul_mat([hd,T], [hd,T]) = [T,T]
        for (int32_t h = 0; h < n_heads; ++h) {
            // Extract head slices: [T, hd] — need contiguous buffers
            std::vector<float> Qh(n_frames * hd), Kh(n_frames * hd), Vh(n_frames * hd);
            for (int32_t t = 0; t < n_frames; ++t) {
                memcpy(Qh.data() + (size_t)t * hd, Q.data() + (size_t)t * H + (size_t)h * hd, hd * sizeof(float));
                memcpy(Kh.data() + (size_t)t * hd, K.data() + (size_t)t * H + (size_t)h * hd, hd * sizeof(float));
                memcpy(Vh.data() + (size_t)t * hd, V.data() + (size_t)t * H + (size_t)h * hd, hd * sizeof(float));
            }

            // QK^T: [T, hd] x [T, hd]^T -> [T, T]
            // ggml_mul_mat(A=[hd,T], B=[hd,T]) -> [T, T]  where result[i,j] = sum_d A[d,i]*B[d,j]
            // = K^T * Q  (i=key_pos, j=query_pos) → need transposed
            // Actually: ggml_mul_mat(K_h_t, Q_h_t) where K_h_t=[hd,T], Q_h_t=[hd,T]
            // result[k,q] = sum_d K[q_t=k,d]*Q[q_t=q... no
            // Let's just use the matmul: scores[q,k] = dot(Q[q], K[k])
            // = ggml_mul_mat(K_h as [hd,T], Q_h as [hd,T]) -> result[T_k, T_q]
            // result[k,q] = sum_d K_h[d,k] * Q_h[d,q]  (GGML col-major)
            // Since data is [T,hd] row-major: K_h data at offset k*hd+d = K_h[k,d]
            // GGML [hd, T] interprets element (d, k) at d + k*hd — matches!
            std::vector<float> scores_full;
            if (!matmul_ggml(Kh, n_frames, hd, Qh, n_frames, scores_full)) { ggml_backend_free(backend_cpu); return false; }
            // scores_full[q*T + k] = dot(K[k], Q[q]) — scores[q,k] ✓

            // Per query: apply full-causal mask, softmax, weighted-V sum
            for (int32_t t = 0; t < n_frames; ++t) {
                float * sc_row = scores_full.data() + (size_t)t * n_frames;

                // Full causal mask — Python uses is_causal=True with no sliding window mask.
                // The sliding_window config field (250) is NOT enforced in the encoder transformer
                // (confirmed empirically: position 0 still affects position 260 in Python).
                // Use full causal attention for all inputs to match Python exactly.
                for (int32_t k = 0; k < n_frames; ++k) {
                    if (k > t) sc_row[k] = -1e30f;
                    else sc_row[k] *= scale;
                }
                // Softmax over [0..t]
                float mx = -1e30f;
                for (int32_t k = 0; k <= t; ++k) if (sc_row[k] > mx) mx = sc_row[k];
                float sm = 0.0f;
                for (int32_t k = 0; k <= t; ++k) { sc_row[k] = expf(sc_row[k] - mx); sm += sc_row[k]; }
                for (int32_t k = 0; k <= t; ++k) sc_row[k] /= sm;

                // Weighted V
                float * out_h = attn_out.data() + (size_t)t * H + (size_t)h * hd;
                for (int32_t k = 0; k <= t; ++k) {
                    const float * vk = Vh.data() + (size_t)k * hd;
                    float w = sc_row[k];
                    for (int32_t d = 0; d < hd; ++d) out_h[d] += w * vk[d];
                }
            }
        }

        // 5. Output projection via GGML matmul
        std::vector<float> attn_proj;
        if (!matmul_ggml(ly.attn_o_w_f, H, H, attn_out, n_frames, attn_proj)) { ggml_backend_free(backend_cpu); return false; }

        // 6. Layer scale + residual
        for (int32_t t = 0; t < n_frames; ++t) {
            for (int32_t d = 0; d < H; ++d) {
                size_t idx = (size_t)t * H + d;
                hidden_out[idx] += attn_proj[idx] * ly.attn_scale_f[d];
            }
        }

        // === FFN ===
        // 7. LayerNorm
        std::vector<float> normed2(hidden_out);
        layer_norm(normed2, H, ly.ffn_in_norm_w_f, ly.ffn_in_norm_b_f);

        // 8. fc1 via GGML matmul + exact GELU
        std::vector<float> ffn_mid;
        if (!matmul_ggml(ly.ffn_fc1_w_f, I, H, normed2, n_frames, ffn_mid)) { ggml_backend_free(backend_cpu); return false; }
        for (float & v : ffn_mid) {
            v = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
        }

        // 9. fc2 via GGML matmul
        std::vector<float> ffn_out;
        if (!matmul_ggml(ly.ffn_fc2_w_f, H, I, ffn_mid, n_frames, ffn_out)) { ggml_backend_free(backend_cpu); return false; }

        // 10. Layer scale + residual
        for (int32_t t = 0; t < n_frames; ++t) {
            for (int32_t d = 0; d < H; ++d) {
                size_t idx = (size_t)t * H + d;
                hidden_out[idx] += ffn_out[idx] * ly.ffn_scale_f[d];
            }
        }
    }

    ggml_backend_free(backend_cpu);
    return true;
}

// ============================================================
// Step 3: Downsample conv (25 Hz -> 12.5 Hz)
// ============================================================
bool MimiEncoder::run_downsample(const std::vector<float> & hidden_in, int32_t n_frames,
                                  std::vector<float> & hidden_out, int32_t & n_frames_out) {
    const auto & cfg = model_.config;
    const int32_t H  = cfg.hidden_size;

    auto & dc = model_.downsample;
    if (!dc.w) {
        error_msg_ = "downsample weight missing";
        return false;
    }

    std::vector<float> w = tensor_to_float(dc.w);
    // k=4, stride=2, hidden -> hidden
    // IMPORTANT: encoder.downsample uses 'replicate' padding (not zero).
    // Left pad = 2 frames replicated from frame 0.
    // Extra right-padding uses 'replicate' (copy of last frame) for ceil output frames.
    int32_t ds_k = 4, ds_stride = 2, ds_pad_left = 2;
    int32_t padded = n_frames + ds_pad_left;
    int32_t n_frames_floor = (padded - ds_k) / ds_stride + 1;
    int64_t n_frames_ceil  = ((int64_t)(padded - ds_k) + ds_stride - 1) / ds_stride + 1;

    // Build padded input: replicate first frame on left, last frame on right
    int32_t extra_right = 0;
    if ((int32_t)n_frames_ceil > n_frames_floor) {
        int32_t ideal_len = (int32_t)((n_frames_ceil - 1) * ds_stride + ds_k - ds_pad_left);
        extra_right = ideal_len - n_frames;
        if (extra_right < 0) extra_right = 0;
    }
    int32_t ds_in_len = n_frames + ds_pad_left + extra_right;
    std::vector<float> ds_in((size_t)ds_in_len * H);
    // Left replicate: copy first frame ds_pad_left times
    for (int32_t p = 0; p < ds_pad_left; ++p) {
        memcpy(ds_in.data() + (size_t)p * H, hidden_in.data(), H * sizeof(float));
    }
    // Copy actual frames
    memcpy(ds_in.data() + (size_t)ds_pad_left * H, hidden_in.data(), (size_t)n_frames * H * sizeof(float));
    // Right replicate: copy last frame extra_right times
    for (int32_t p = 0; p < extra_right; ++p) {
        memcpy(ds_in.data() + (size_t)(ds_pad_left + n_frames + p) * H,
               hidden_in.data() + (size_t)(n_frames - 1) * H, H * sizeof(float));
    }

    // Now convolve with no padding (we've already padded the input)
    apply_causal_conv1d(ds_in.data(), ds_in_len, H, H,
                        w.data(), nullptr,
                        ds_k, ds_stride, 1, 0,  // pad_left=0, already in ds_in
                        hidden_out, n_frames_out);
    return true;
}

// ============================================================
// Step 4: RVQ quantize
// Input: latent [n_frames, hidden_size]
// Output: codes [n_frames * n_valid_quantizers]
// ============================================================
bool MimiEncoder::run_rvq(const std::vector<float> & latent, int32_t n_frames,
                           std::vector<int32_t> & codes_out) {
    const auto & cfg = model_.config;
    const int32_t H   = cfg.hidden_size;
    const int32_t VH  = cfg.vq_hidden_dim;
    const int32_t NQ  = cfg.num_valid_quantizers;
    const int32_t CS  = cfg.codebook_size;
    const int32_t CD  = cfg.codebook_dim;

    codes_out.resize((size_t)n_frames * NQ, 0);

    // Helper: project [n_frames, H] -> [n_frames, VH] using Conv1d k=1
    // proj_w PyTorch [VH, H, 1] stored as GGUF bytes [OC=VH, IC=H, K=1]
    // element (oc, ic) at offset oc * H + ic
    auto project = [&](const std::vector<float> & in, struct ggml_tensor * proj_w,
                        std::vector<float> & out) {
        if (!proj_w) { out = in; return; }
        std::vector<float> pw = tensor_to_float(proj_w);
        out.assign((size_t)n_frames * VH, 0.0f);
        for (int32_t t = 0; t < n_frames; ++t) {
            const float * in_row  = in.data() + (size_t)t * H;
            float * out_row = out.data() + (size_t)t * VH;
            for (int32_t oc = 0; oc < VH; ++oc) {
                float s = 0.0f;
                for (int32_t ic = 0; ic < H; ++ic) {
                    s += in_row[ic] * pw[(size_t)oc * H + ic];
                }
                out_row[oc] = s;
            }
        }
    };

    // Quantize using a single RVQ layer
    auto quantize_rvq = [&](const std::vector<float> & embeddings, int32_t n_frames_q,
                             int32_t dim, std::vector<mimi_codebook> & cbs,
                             int32_t cb_offset) {
        // dim should equal CD (codebook_dim) for correct distance computation
        (void)dim;
        std::vector<float> residual = embeddings;

        for (int32_t q = 0; q < (int32_t)cbs.size(); ++q) {
            auto & cb = cbs[q];
            if (cb.embed.empty()) continue;

            for (int32_t t = 0; t < n_frames_q; ++t) {
                const float * r = residual.data() + (size_t)t * CD;
                float best_dist = 1e30f;
                int32_t best_idx = 0;
                for (int32_t c = 0; c < CS; ++c) {
                    const float * e = cb.embed.data() + (size_t)c * CD;
                    float dist = 0.0f;
                    for (int32_t d = 0; d < CD; ++d) {
                        float diff = r[d] - e[d];
                        dist += diff * diff;
                    }
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_idx  = c;
                    }
                }
                codes_out[(size_t)t * NQ + (cb_offset + q)] = best_idx;

                const float * e = cb.embed.data() + (size_t)best_idx * CD;
                float * r_mut = residual.data() + (size_t)t * CD;
                for (int32_t d = 0; d < CD; ++d) r_mut[d] -= e[d];
            }
        }
    };

    // Semantic quantizer (1 codebook)
    std::vector<float> sem_proj;
    project(latent, model_.semantic_input_proj, sem_proj);
    quantize_rvq(sem_proj, n_frames, VH, model_.semantic_cbs, 0);

    // Acoustic quantizer (15 codebooks)
    std::vector<float> acou_proj;
    project(latent, model_.acoustic_input_proj, acou_proj);
    quantize_rvq(acou_proj, n_frames, VH, model_.acoustic_cbs, 1);

    return true;
}

// ============================================================
// Main encode() entry point
// ============================================================
bool MimiEncoder::encode(const float * samples, int32_t n_samples,
                          std::vector<int32_t> & codes_out, int32_t & n_frames_out) {
    if (!model_.ctx) {
        error_msg_ = "Mimi encoder not loaded";
        return false;
    }

    // Step 1: SEANet conv encoder
    std::vector<float> enc_hidden;
    int32_t enc_frames = 0;
    if (!run_conv_encoder(samples, n_samples, enc_hidden, enc_frames)) return false;

    // Step 2: Transformer
    std::vector<float> tfm_hidden;
    if (!run_transformer(enc_hidden, enc_frames, tfm_hidden)) return false;

    // Step 3: Downsample
    std::vector<float> ds_hidden;
    int32_t ds_frames = 0;
    if (!run_downsample(tfm_hidden, enc_frames, ds_hidden, ds_frames)) return false;

    // Step 4: RVQ
    if (!run_rvq(ds_hidden, ds_frames, codes_out)) return false;

    n_frames_out = ds_frames;
    return true;
}

} // namespace qwen3_tts
