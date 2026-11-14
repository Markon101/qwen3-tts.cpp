// Exact scalar-order oracle for Mimi's independent-output-channel SIMD path.
#include <string>
#include <map>
#include <vector>
#include <memory>
#define private public
#include "mimi_encoder.h"
#undef private
#include <cstdio>
#include <cstring>
#include <random>

static void reference(const float * input, int len, int in, int out,
                      const float * w, const float * b, int kernel,
                      int stride, int dilation, int left,
                      std::vector<float> & result, int & count) {
    count = (len + left - ((kernel - 1) * dilation + 1)) / stride + 1;
    if (count <= 0) { count = 0; result.clear(); return; }
    result.assign((size_t)count * out, 0);
    for (int t = 0; t < count; ++t) {
        for (int oc = 0; oc < out; ++oc) {
            float sum = b ? b[oc] : 0;
            for (int k = 0; k < kernel; ++k) {
                const int src = t * stride + k * dilation - left;
                if (src < 0 || src >= len) continue;
                for (int ic = 0; ic < in; ++ic)
                    sum += input[(size_t)src * in + ic] *
                           w[(size_t)oc * in * kernel + (size_t)ic * kernel + k];
            }
            result[(size_t)t * out + oc] = sum;
        }
    }
}

int main() {
    std::mt19937 rng(1337);
    // Non-dyadic values and mixed magnitudes expose reduction reassociation.
    auto fill = [&](std::vector<float> & v) {
        for (float & x : v) x = ((int)(rng() % 2001) - 1000) / 317.0f;
    };
    int cases = 0;
    for (int out : {1, 7, 8, 9, 15, 16, 17, 32, 65})
    for (int in : {1, 3, 16, 33, 64})
    for (int kernel : {1, 3, 7, 8, 10, 12, 16})
    for (int stride : {1, 2, 5})
    for (int dilation : {1, 2})
    for (bool bias : {false, true})
    for (bool padded : {false, true}) {
        const int len = 23;
        const int left = padded ? (kernel - 1) * dilation : 0;
        std::vector<float> input((size_t)len * in), w((size_t)out * in * kernel), b(out);
        fill(input); fill(w); fill(b);
        std::vector<float> expected, actual;
        int n_expected, n_actual;
        reference(input.data(), len, in, out, w.data(), bias ? b.data() : nullptr,
                  kernel, stride, dilation, left, expected, n_expected);
        qwen3_tts::MimiEncoder::apply_causal_conv1d(
            input.data(), len, in, out, w.data(), bias ? b.data() : nullptr,
            kernel, stride, dilation, left, actual, n_actual);
        if (n_actual != n_expected || actual.size() != expected.size() ||
            (!actual.empty() && std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)))) {
            fprintf(stderr, "FAIL case=%d in=%d out=%d k=%d stride=%d dilation=%d bias=%d\n",
                    cases, in, out, kernel, stride, dilation, bias);
            return 1;
        }
        ++cases;
    }
    // Empty output must clear a previous result as well.
    std::vector<float> output(8, 1);
    float zero = 0;
    int count = -1;
    qwen3_tts::MimiEncoder::apply_causal_conv1d(&zero, 1, 1, 8, &zero, nullptr,
                                               7, 1, 1, 0, output, count);
    if (count != 0 || !output.empty()) return 1;
    printf("PASS: %d bit-exact convolution cases and empty-output check\n", cases);
    return 0;
}
