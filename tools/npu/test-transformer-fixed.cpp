// Compare identical prefill and step embeddings, independently of sampling.
// Fixtures contain private speaker-derived activations; keep them outside Git.
#include "gguf_loader.h"
#include "text_tokenizer.h"
#include "tts_transformer.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool save(const std::string &path, const std::vector<float> &v) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(v.data(), sizeof(float), v.size(), f) == v.size();
    return fclose(f) == 0 && ok;
}
static bool load(const std::string &path, std::vector<float> &v) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return false; }
    long n = ftell(f);
    if (n <= 0 || n % sizeof(float) || n > 64 * 1024 * 1024) {
        fclose(f); return false;
    }
    rewind(f);
    v.resize(n / sizeof(float));
    bool ok = fread(v.data(), sizeof(float), v.size(), f) == v.size();
    fclose(f);
    return ok;
}
static bool check(const std::vector<float> &a, const std::vector<float> &b,
                  const char *name) {
    if (a.empty() || a.size() != b.size()) return false;
    double dot = 0, aa = 0, bb = 0, err = 0, mx = 0;
    int nonfinite = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { ++nonfinite; continue; }
        double d = double(a[i]) - b[i];
        dot += double(a[i]) * b[i]; aa += double(a[i]) * a[i];
        bb += double(b[i]) * b[i]; err += d * d; mx = std::max(mx, std::fabs(d));
    }
    double cosine = dot / std::sqrt(std::max(aa * bb, 1.e-30));
    double nrmse = std::sqrt(err / std::max(bb, 1.e-30));
    // This is a coarse numerical gate, not a speech-quality criterion.
    bool pass = nonfinite == 0 && cosine > .99 && nrmse < .15;
    printf("RESULT {\"tensor\":\"%s\",\"count\":%zu,\"nonfinite\":%d,"
           "\"cosine\":%.9g,\"nrmse\":%.9g,\"max_abs\":%.9g,\"pass\":%s}\n",
           name, a.size(), nonfinite, cosine, nrmse, mx, pass ? "true" : "false");
    return pass;
}
struct PredictorCapture { std::vector<std::vector<float>> logits; };
static bool capture_predictor(ggml_tensor *t, bool ask, void *userdata) {
    if (ask) return strcmp(t->name, "logits") == 0 && t->type == GGML_TYPE_F32;
    auto *capture = static_cast<PredictorCapture *>(userdata);
    std::vector<float> values(ggml_nelements(t));
    ggml_backend_tensor_get(t, values.data(), 0, values.size() * sizeof(float));
    capture->logits.push_back(std::move(values));
    return true;
}
int main(int argc, char **argv) {
    if (argc < 4 || (strcmp(argv[1], "record") && strcmp(argv[1], "compare"))) {
        fprintf(stderr, "Usage: %s record MODEL FIXTURE_DIR SPEAKER.bin\n"
                        "       %s compare MODEL FIXTURE_DIR\n", argv[0], argv[0]);
        return 2;
    }
    bool record = !strcmp(argv[1], "record");
    if (record && argc != 5) return 2;
    unsetenv("QWEN3_TTS_FORCE_CPU");
    setenv("QWEN3_TTS_TRANSFORMER_BACKEND", record ? "cpu" : "hexagon", 1);
    std::string dir = argv[3];
    qwen3_tts::TTSTransformer transformer;
    if (!transformer.load_model(argv[2])) {
        fprintf(stderr, "%s\n", transformer.get_error().c_str()); return 3;
    }
    transformer.set_n_threads(2);
    const int h = transformer.get_config().hidden_size;
    std::vector<float> prefill, steps;
    if (record) {
        qwen3_tts::GGUFLoader loader;
        qwen3_tts::TextTokenizer tokenizer;
        if (!loader.open(argv[2]) || !tokenizer.load_from_gguf(loader.get_ctx())) return 4;
        auto tokens = tokenizer.encode_for_tts("Hello there, this is a numerical validation test.");
        FILE *f = fopen(argv[4], "rb");
        uint32_t n = 0;
        if (!f) return 4;
        bool ok = fread(&n, 4, 1, f) == 1 && n == uint32_t(h);
        std::vector<float> speaker(ok ? n : 0);
        ok = ok && fread(speaker.data(), 4, n, f) == n;
        fclose(f);
        if (!ok) return 4;
        std::vector<float> trailing, pad;
        if (!transformer.build_prefill_graph(tokens.data(), tokens.size(), speaker.data(),
                    2050, prefill, trailing, pad)) return 5;
        // Fixed continuous inputs isolate forward math and KV updates from
        // autoregressive code choices. They are not a natural speech trajectory.
        steps.resize(8 * h);
        for (int j = 0; j < 8; ++j)
            for (int i = 0; i < h; ++i)
                steps[j * h + i] = prefill[prefill.size() - h + i] +
                                  .01f * std::sin(float(i * .013 + j * .17));
        if (!save(dir + "/prefill.f32", prefill) || !save(dir + "/steps.f32", steps)) return 6;
    } else if (!load(dir + "/prefill.f32", prefill) || !load(dir + "/steps.f32", steps)) return 6;
    if (prefill.empty() || prefill.size() % h || steps.size() != size_t(8 * h)) return 6;
    int n_tokens = prefill.size() / h;
    if (n_tokens > 192 || !transformer.init_kv_cache(256)) return 7;
    transformer.clear_kv_cache();
    bool passed = true;
    for (int step = -1; step < 8; ++step) {
        std::vector<float> hidden, logits;
        bool ok = step < 0 ? transformer.forward_prefill(prefill.data(), n_tokens, 0, hidden, &logits) :
                  transformer.forward_step(steps.data() + step * h, n_tokens + step, logits, &hidden);
        if (!ok) { fprintf(stderr, "%s\n", transformer.get_error().c_str()); return 8; }
        for (int which = 0; which < 2; ++which) {
            auto &v = which ? logits : hidden;
            std::string name = std::to_string(step + 1) + (which ? "-logits" : "-hidden");
            std::string path = dir + "/" + name + ".f32";
            for (float x : v) if (!std::isfinite(x)) return 9;
            if (record) { if (!save(path, v)) return 10; }
            else {
                std::vector<float> reference;
                if (!load(path, reference) || !save(dir + "/htp-" + name + ".f32", v)) return 10;
                passed = check(v, reference, name.c_str()) && passed;
            }
        }
    }
    // Isolate the predictor's first head using exactly the CPU prefill hidden
    // state and a fixed CB0. Later heads are checked for finiteness only because
    // their greedy feedback tokens can diverge between backends.
    std::vector<float> cpu_hidden;
    if (!load(dir + "/0-hidden.f32", cpu_hidden) || cpu_hidden.size() < size_t(h)) return 12;
    PredictorCapture capture;
    transformer.set_eval_callback(capture_predictor, &capture);
    std::vector<int32_t> codes;
    if (!transformer.predict_codes_autoregressive(cpu_hidden.data() + cpu_hidden.size() - h,
                                                  1995, codes, 0.0f)) return 13;
    transformer.clear_eval_callback();
    if (capture.logits.size() != 15) return 14;
    for (size_t i = 0; i < capture.logits.size(); ++i) {
        for (float x : capture.logits[i]) if (!std::isfinite(x)) return 15;
        std::string name = "predictor-" + std::to_string(i);
        std::string path = dir + "/" + name + ".f32";
        if (record) { if (!save(path, capture.logits[i])) return 16; }
        else {
            if (!save(dir + "/htp-" + name + ".f32", capture.logits[i])) return 16;
            if (i == 0) {
                std::vector<float> reference;
                if (!load(path, reference)) return 16;
                passed = check(capture.logits[i], reference, "predictor-first-head-logits") && passed;
            }
        }
    }
    printf("PREDICTOR_LOGITS {\"heads\":15,\"nonfinite\":0,\"later_heads_parity_checked\":false}\n");
    return passed ? 0 : 11;
}
