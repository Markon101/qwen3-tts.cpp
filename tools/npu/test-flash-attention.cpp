// Isolate one-token GQA flash attention from TTS, sampling, and KV updates.
// Public deterministic inputs; the oracle uses double arithmetic and actual
// F16-rounded K/V values. No CPU GGML backend or scheduler fallback is used.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

namespace {
constexpr int D = 128;
constexpr int HQ = 16;
constexpr int HKV = 8;
constexpr int NQ = 1;
const char *const RESULT_DIR = ".cache/npu/flash-attention";

// Integer mixing makes the fixture independent of a standard-library RNG.
float public_value(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return float(x >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

std::string setting_tag(const char *name) {
    const char *value = std::getenv(name);
    if (!value || !*value) return "default";
    std::string tag;
    for (; *value; ++value) {
        const char c = *value;
        tag += ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                (c >= 'A' && c <= 'Z') || c == '-') ? c : '_';
    }
    return tag;
}

bool save_raw(const std::string &path, const std::vector<float> &values) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool written = std::fwrite(values.data(), sizeof(float), values.size(), f) == values.size();
    return std::fclose(f) == 0 && written;
}

struct CaseMemory {
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~CaseMemory() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
};

bool failure(int kv, const char *layout, const char *reason) {
    std::printf("RESULT {\"kind\":\"flash_attention\",\"KV\":%d,\"layout\":\"%s\","
                "\"Q\":1,\"Hq\":16,\"Hkv\":8,\"Dk\":128,\"Dv\":128,"
                "\"error\":\"%s\",\"pass\":false}\n", kv, layout, reason);
    std::fflush(stdout);
    return false;
}

bool run_case(ggml_backend_t backend, int kv, bool strided, const std::string &tag) {
    const char *layout = strided ? "strided" : "contiguous";
    const float scale = 1.0f / std::sqrt(float(D));
    std::vector<float> q(D * HQ);
    std::vector<ggml_fp16_t> stored_k(size_t(D) * kv * HKV), stored_v(stored_k.size());
    // Canonical logical order [D, KV, HKV], separate from physical storage.
    std::vector<float> k(stored_k.size()), v(stored_v.size());
    for (int h = 0; h < HQ; ++h) {
        for (int d = 0; d < D; ++d) {
            q[d + D * h] = 1.5f * public_value(0x1234u + uint32_t(d + D * h));
        }
    }
    for (int h = 0; h < HKV; ++h) {
        for (int t = 0; t < kv; ++t) {
            for (int d = 0; d < D; ++d) {
                // Coordinates, rather than KV length or layout, define values:
                // every case shares the same public prefix.
                const uint32_t seed = uint32_t(d) + 131u * uint32_t(t) + 65537u * uint32_t(h);
                const float key = 1.5f * public_value(seed ^ 0x71b35u);
                const float value = .7f * public_value(seed ^ 0x92ca1u) + .04f * h +
                                    .25f * std::sin(.037f * t) + .05f * std::cos(.11f * d);
                const size_t logical = size_t(d) + D * (size_t(t) + size_t(kv) * h);
                const size_t physical = strided ? size_t(d) + D * (size_t(h) + size_t(HKV) * t) : logical;
                stored_k[physical] = ggml_fp32_to_fp16(key);
                stored_v[physical] = ggml_fp32_to_fp16(value);
                k[logical] = ggml_fp16_to_fp32(stored_k[physical]);
                v[logical] = ggml_fp16_to_fp32(stored_v[physical]);
            }
        }
    }

    // Independent stable softmax oracle: Q is F32, K/V are the exact rounded
    // F16 values sent to HTP. GQA repeats each KV head for two query heads.
    std::vector<float> reference(D * HQ);
    std::vector<double> scores(kv);
    for (int h = 0; h < HQ; ++h) {
        const int kh = h / (HQ / HKV);
        double maximum = -std::numeric_limits<double>::infinity();
        for (int t = 0; t < kv; ++t) {
            double dot = 0;
            for (int d = 0; d < D; ++d) {
                dot += double(q[d + D * h]) * k[size_t(d) + D * (size_t(t) + size_t(kv) * kh)];
            }
            scores[t] = dot * double(scale);
            maximum = std::max(maximum, scores[t]);
        }
        double denominator = 0;
        for (double &score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (int d = 0; d < D; ++d) {
            double sum = 0;
            for (int t = 0; t < kv; ++t) {
                sum += scores[t] * v[size_t(d) + D * (size_t(t) + size_t(kv) * kh)];
            }
            reference[d + D * h] = float(sum / denominator);
        }
    }
    const std::string stem = std::string(RESULT_DIR) + "/" + tag + "-" + layout + "-kv" + std::to_string(kv);
    if (!save_raw(stem + "-reference.f32", reference)) return failure(kv, layout, "save_reference");

    CaseMemory memory;
    ggml_init_params params = {2 * 1024 * 1024, nullptr, true};
    memory.ctx = ggml_init(params);
    if (!memory.ctx) return failure(kv, layout, "context");
    ggml_tensor *qt = ggml_new_tensor_3d(memory.ctx, GGML_TYPE_F32, D, NQ, HQ);
    ggml_tensor *kb = ggml_new_tensor_3d(memory.ctx, GGML_TYPE_F16, D, strided ? HKV : kv, strided ? kv : HKV);
    ggml_tensor *vb = ggml_new_tensor_3d(memory.ctx, GGML_TYPE_F16, D, strided ? HKV : kv, strided ? kv : HKV);
    ggml_tensor *kt = strided ? ggml_permute(memory.ctx, kb, 0, 2, 1, 3) : kb;
    ggml_tensor *vt = strided ? ggml_permute(memory.ctx, vb, 0, 2, 1, 3) : vb;
    ggml_set_name(qt, "probe_q");
    ggml_set_name(kb, "probe_k_storage");
    ggml_set_name(vb, "probe_v_storage");
    ggml_tensor *out = ggml_flash_attn_ext(memory.ctx, qt, kt, vt, nullptr, scale, 0.0f, 0.0f);
    ggml_set_name(out, "probe_flash_output");
    if (out->ne[0] != D || out->ne[1] != HQ || out->ne[2] != NQ || out->ne[3] != 1) {
        return failure(kv, layout, "output_shape");
    }
    if (!ggml_backend_dev_supports_op(ggml_backend_get_device(backend), out)) {
        return failure(kv, layout, "unsupported");
    }
    ggml_cgraph *graph = ggml_new_graph(memory.ctx);
    ggml_build_forward_expand(graph, out);
    memory.buffer = ggml_backend_alloc_ctx_tensors(memory.ctx, backend);
    if (!memory.buffer) return failure(kv, layout, "allocation");
    ggml_backend_tensor_set(qt, q.data(), 0, q.size() * sizeof(float));
    ggml_backend_tensor_set(kb, stored_k.data(), 0, stored_k.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(vb, stored_v.data(), 0, stored_v.size() * sizeof(ggml_fp16_t));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) return failure(kv, layout, "compute");
    std::vector<float> output(reference.size());
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));
    const bool saved = save_raw(stem + "-output.f32", output);
    double dot = 0, aa = 0, bb = 0, error = 0, maximum_error = 0;
    int nonfinite = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i]) || !std::isfinite(reference[i])) { ++nonfinite; continue; }
        const double a = output[i], b = reference[i], delta = a - b;
        dot += a * b;
        aa += a * a;
        bb += b * b;
        error += delta * delta;
        maximum_error = std::max(maximum_error, std::fabs(delta));
    }
    const double cosine = dot / std::sqrt(std::max(aa * bb, 1.e-30));
    const double nrmse = std::sqrt(error / std::max(bb, 1.e-30));
    const bool pass = saved && nonfinite == 0 && cosine > .999 && nrmse < .03 && maximum_error < .02;
    std::printf("RESULT {\"kind\":\"flash_attention\",\"KV\":%d,\"layout\":\"%s\","
                "\"Q\":1,\"Hq\":16,\"Hkv\":8,\"Dk\":128,\"Dv\":128,"
                "\"count\":%zu,\"k_nb1\":%zu,\"k_nb2\":%zu,\"v_nb1\":%zu,\"v_nb2\":%zu,"
                "\"nonfinite\":%d,\"finite\":%s,", kv, layout, output.size(),
                kt->nb[1], kt->nb[2], vt->nb[1], vt->nb[2], nonfinite, nonfinite == 0 ? "true" : "false");
    if (nonfinite) std::printf("\"max_abs\":null,\"cosine\":null,\"nrmse\":null,");
    else std::printf("\"max_abs\":%.9g,\"cosine\":%.9g,\"nrmse\":%.9g,", maximum_error, cosine, nrmse);
    std::printf("\"raw_saved\":%s,\"pass\":%s}\n", saved ? "true" : "false", pass ? "true" : "false");
    std::fflush(stdout);
    return pass;
}
} // namespace

int main(int argc, char **argv) {
    bool do_strided = true, do_contiguous = true;
    if (argc == 2 && std::strcmp(argv[1], "--strided-only") == 0) do_contiguous = false;
    else if (argc == 2 && std::strcmp(argv[1], "--contiguous-only") == 0) do_strided = false;
    else if (argc != 1) {
        std::fprintf(stderr, "Usage: %s [--strided-only|--contiguous-only]\n", argv[0]);
        return 2;
    }
    std::error_code ec;
    std::filesystem::create_directories(RESULT_DIR, ec);
    if (ec) { std::fprintf(stderr, "Cannot create flash artifact directory: %s\n", ec.message().c_str()); return 2; }
    const std::string hmx = setting_tag("GGML_HEXAGON_USE_HMX");
    const std::string nhvx = setting_tag("GGML_HEXAGON_NHVX");
    const std::string tag = "hmx-" + hmx + "-nhvx-" + nhvx + "-run-" + setting_tag("QWEN_NPU_RESULT_TAG");
    ggml_backend_t backend = ggml_backend_init_by_name("HTP0", nullptr);
    if (!backend) { std::fprintf(stderr, "HTP0 initialization failed; no CPU fallback permitted\n"); return 2; }
    const char *name = ggml_backend_dev_name(ggml_backend_get_device(backend));
    if (std::strcmp(name, "HTP0") != 0) {
        std::fprintf(stderr, "Unexpected backend %s\n", name);
        ggml_backend_free(backend);
        return 2;
    }
    std::printf("CONFIG {\"backend\":\"HTP0\",\"hmx\":\"%s\",\"nhvx\":\"%s\","
                "\"oracle\":\"cpu_double_rounded_f16_kv\",\"output_layout\":\"D_Hq_Nq\","
                "\"artifact_dir\":\"%s\"}\n", hmx.c_str(), nhvx.c_str(), RESULT_DIR);
    const int lengths[] = {16, 64, 127, 128, 191, 192, 193, 216, 256, 297, 434};
    int cases = 0, failures = 0;
    for (bool strided : {true, false}) {
        if ((strided && !do_strided) || (!strided && !do_contiguous)) continue;
        for (int kv : lengths) {
            ++cases;
            if (!run_case(backend, kv, strided, tag)) ++failures;
        }
    }
    std::printf("FLASH_SUMMARY {\"cases\":%d,\"failures\":%d,\"pass\":%s}\n",
                cases, failures, failures == 0 ? "true" : "false");
    ggml_backend_free(backend);
    return failures == 0 ? 0 : 4;
}
