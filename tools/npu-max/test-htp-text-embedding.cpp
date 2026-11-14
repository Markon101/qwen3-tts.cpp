// Real-model placement/lifecycle and bit-exact projection controls.
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <functional>
#include <random>
#include <unordered_set>
#include <chrono>
#define private public
#include "tts_transformer.h"
#undef private
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
using namespace qwen3_tts;

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s MODEL hexagon|cpu|opencl\n", argv[0]);
        return 2;
    }
    setenv("QWEN3_TTS_TRANSFORMER_BACKEND", argv[2], 1);
    TTSTransformer t;
    std::vector<std::vector<float>> expected;
    const std::vector<std::vector<int32_t>> ids = {
        {0}, {151671, 151672, 151673}, {42, 42, 151935, 1},
        {101, 92, 17, 5, 203, 91, 113, 777, 8, 94, 16, 23, 301, 3, 19, 5, 120}
    };
    for (int pass = 0; pass < 3; ++pass) {
        setenv("QWEN3_TTS_HTP_HOST_TEXT_EMBD", pass == 1 ? "1" : "0", 1);
        if (!t.load_model(argv[1])) { fprintf(stderr, "%s\n", t.get_error().c_str()); return 3; }
        t.set_n_threads(2);
        const bool host = t.model_.text_embd_buffer != nullptr;
        const bool htp = !strcmp(argv[2], "hexagon");
        if (host != (pass == 1 && htp && t.model_.text_embd->type == GGML_TYPE_F16)) return 4;
        if (host && t.model_.text_embd->buffer != t.model_.text_embd_buffer) return 4;
        if (host) {
            auto original = t.model_.tensors.find(t.model_.text_embd->name);
            if (original == t.model_.tensors.end() || original->second->data != t.model_.text_embd->data ||
                original->second->buffer == t.model_.text_embd_buffer) return 4;
        }
        fprintf(stderr, "pass=%d host=%d type=%s shape=%lldx%lld main_bytes=%zu host_bytes=%zu\n",
            pass, host, ggml_type_name(t.model_.text_embd->type),
            (long long)t.model_.text_embd->ne[0], (long long)t.model_.text_embd->ne[1],
            ggml_backend_buffer_get_size(t.model_.buffer),
            host ? ggml_backend_buffer_get_size(t.model_.text_embd_buffer) : 0);
        for (size_t i = 0; i < ids.size(); ++i) {
            std::vector<float> actual;
            if (!t.project_text_tokens(ids[i].data(), (int)ids[i].size(), actual)) return 5;
            if (pass == 0) expected.push_back(actual);
            else if (actual.size() != expected[i].size() ||
                     memcmp(actual.data(), expected[i].data(), actual.size() * sizeof(float))) {
                double error = 0, norm = 0, peak = 0;
                for (size_t j = 0; j < actual.size(); ++j) {
                    const double d = actual[j] - expected[i][j];
                    error += d*d; norm += double(expected[i][j])*expected[i][j];
                    peak = std::max(peak, std::fabs(d));
                }
                fprintf(stderr, "projection mismatch pass=%d case=%zu nrmse=%g max_abs=%g first=%g expected=%g\n",
                    pass, i, std::sqrt(error / std::max(norm, 1e-30)), peak, actual[0], expected[i][0]);
                return 6;
            }
        }
        std::vector<float> empty(1, 1);
        if (!t.project_text_tokens(nullptr, 0, empty) || !empty.empty()) return 7;
        t.unload_model();
        if (t.model_.text_embd_buffer || t.model_.buffer || t.model_.ctx) return 8;
    }
    // Failed subsequent load and repeated unload must leave no owned buffers.
    if (t.load_model("/nonexistent/qwen3tts-placement-test.gguf")) return 9;
    t.unload_model();
    if (t.model_.text_embd_buffer || t.model_.buffer || t.model_.ctx) return 10;
    printf("PASS: original/host/original projections bit-exact; reload/unload and failed-load controls\n");
    return 0;
}
