// Numerical diagnostic only; tracing serializes graph observations and changes timing.
#include "gguf_loader.h"
#include "text_tokenizer.h"
#include "tts_transformer.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool trace_node(ggml_tensor *t, bool ask, void *) {
    static bool captured = false;
    if (captured) return ask ? false : true;
    if (ask) return t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16;
    std::vector<unsigned char> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    std::vector<float> flat(ggml_nelements(t));
    size_t nans = 0;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        int64_t j = i;
        size_t offset = 0;
        for (int dim = 0; dim < 4; ++dim) {
            offset += (j % t->ne[dim]) * t->nb[dim]; j /= t->ne[dim];
        }
        if (t->type == GGML_TYPE_F32) memcpy(&flat[i], raw.data() + offset, 4);
        else { ggml_fp16_t x; memcpy(&x, raw.data() + offset, 2); flat[i] = ggml_fp16_to_fp32(x); }
        nans += std::isnan(flat[i]);
    }
    if (!nans) return true;
    captured = true;
    fprintf(stderr, "FIRST_NAN op=%s tensor=%s shape=%lld,%lld,%lld,%lld count=%zu/%zu\n",
            ggml_op_name(t->op), t->name, (long long)t->ne[0], (long long)t->ne[1],
            (long long)t->ne[2], (long long)t->ne[3], nans, flat.size());
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; ++i)
        fprintf(stderr, "  src%d %s %s %lld,%lld,%lld,%lld\n", i, t->src[i]->name,
                ggml_type_name(t->src[i]->type), (long long)t->src[i]->ne[0],
                (long long)t->src[i]->ne[1], (long long)t->src[i]->ne[2], (long long)t->src[i]->ne[3]);
    FILE *f = fopen(".cache/npu/trace-first-nan.f32", "wb");
    if (f) { fwrite(flat.data(), 4, flat.size(), f); fclose(f); }
    for (int idx = 0; idx < 2 && t->src[idx]; ++idx) {
        ggml_tensor *src = t->src[idx];
        std::vector<unsigned char> bytes(ggml_nbytes(src));
        ggml_backend_tensor_get(src, bytes.data(), 0, bytes.size());
        char path[100]; snprintf(path, sizeof(path), ".cache/npu/trace-src-%d.raw", idx);
        f = fopen(path, "wb"); if (f) { fwrite(bytes.data(), 1, bytes.size(), f); fclose(f); }
        if (src->type == GGML_TYPE_F32 && ggml_is_contiguous(src)) {
            size_t invalid = 0; float mx = 0;
            for (size_t off = 0; off < bytes.size(); off += 4) {
                float x; memcpy(&x, bytes.data() + off, 4);
                invalid += !std::isfinite(x); mx = std::max(mx, std::fabs(x));
            }
            fprintf(stderr, "SOURCE_RANGE src=%d nonfinite=%zu max_abs=%g\n", idx, invalid, mx);
        }
    }
    return false;
}
int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) { fprintf(stderr, "Usage: %s MODEL SPEAKER.bin [REPEATS]\n", argv[0]); return 2; }
    int repeats = argc == 4 ? atoi(argv[3]) : 1;
    if (repeats < 1 || repeats > 8) return 2;
    if (!getenv("QWEN3_TTS_TRANSFORMER_BACKEND")) setenv("QWEN3_TTS_TRANSFORMER_BACKEND", "hexagon", 1);
    setenv("QWEN3_TTS_VALIDATE_LOGITS", "1", 1);
    qwen3_tts::GGUFLoader loader;
    qwen3_tts::TextTokenizer tokenizer;
    if (!loader.open(argv[1]) || !tokenizer.load_from_gguf(loader.get_ctx())) return 3;
    auto tokens = tokenizer.encode_for_tts("The little fox wandered through the quiet forest, following the silver river toward the distant mountains. Along the way, it listened to the birds and watched sunlight dancing through the leaves.");
    FILE *f = fopen(argv[2], "rb");
    uint32_t n = 0;
    if (!f) return 3;
    bool ok = fread(&n, 4, 1, f) == 1 && n > 0 && n <= 16384;
    std::vector<float> speaker(ok ? n : 0);
    ok = ok && fread(speaker.data(), 4, n, f) == n;
    fclose(f);
    if (!ok) return 3;
    qwen3_tts::TTSTransformer transformer;
    if (!transformer.load_model(argv[1])) return 4;
    if (n != uint32_t(transformer.get_config().hidden_size)) {
        fprintf(stderr, "Speaker embedding dimension mismatch\n"); return 4;
    }
    transformer.set_n_threads(2); transformer.set_seed(20261026);
    if (getenv("QWEN_NPU_TRACE_NODES")) transformer.set_eval_callback(trace_node, nullptr);
    bool all_passed = true;
    for (int request = 0; request < repeats; ++request) {
    transformer.set_seed(20261026);
    std::vector<int32_t> codes;
    ok = transformer.generate(tokens.data(), tokens.size(), speaker.data(), 64, codes);
    fprintf(stderr, "GENERATION success=%d frames=%zu error=%s\n", ok, codes.size()/16, transformer.get_error().c_str());
    size_t zero = 0; for (int32_t x : codes) zero += x == 0;
    printf("RESULT {\"success\":%s,\"frames\":%zu,\"zero_codes\":%zu,\"codes\":%zu}\n", ok?"true":"false",codes.size()/16,zero,codes.size());
    all_passed = ok && all_passed;
    }
    return all_passed ? 0 : 5;
}
