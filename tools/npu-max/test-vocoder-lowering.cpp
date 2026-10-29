// Independent scalar oracle for NPU-Max convolution graph lowering.
// No model, audio recording, tokenizer, GPU or SDK is needed for --backend cpu.
#include <string>
#include <map>
#include <vector>
#include <memory>
#define private public
#include "audio_tokenizer_decoder.h"
#undef private
#include "gguf_loader.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace qwen3_tts;

static bool run_case(const char * kind, int kernel, int stride, int dilation,
                     int length, int in, int out, const char * backend, bool f32 = false, bool channels_first = false) {
    setenv("QWEN3_TTS_DECODER_BACKEND", backend, 1);
    AudioTokenizerDecoder decoder;
    decoder.state_.backend = init_preferred_backend("AudioTokenizerDecoder", nullptr);
    if (!decoder.state_.backend) return false;
    const bool cpu = ggml_backend_is_cpu(decoder.state_.backend);
    if (cpu) ggml_backend_cpu_set_n_threads(decoder.state_.backend, 2);
    if (!cpu) decoder.state_.backend_cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[] = {decoder.state_.backend, decoder.state_.backend_cpu};
    decoder.state_.sched = ggml_backend_sched_new(backends, nullptr, cpu ? 1 : 2, 32768, false, true);
    ggml_init_params params = {ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom(32768, false), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    const bool trans = !strcmp(kind, "transpose"), dw = !strcmp(kind, "depthwise");
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, channels_first ? in : length, channels_first ? length : in, 1);
    ggml_tensor * w = ggml_new_tensor_3d(ctx, f32 ? GGML_TYPE_F32 : GGML_TYPE_F16, kernel, trans ? out : (dw ? 1 : in), trans ? in : out);
    std::vector<ggml_tensor *> taps;
    if (trans || (!dw && channels_first)) for (int k = 0; k < kernel; ++k) taps.push_back(ggml_new_tensor_2d(ctx, w->type, in, out));
    if (dw) for (int k = 0; k < kernel; ++k) taps.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, in, 1));
    ggml_backend_buffer_t source_buffer = ggml_backend_alloc_ctx_tensors(ctx, decoder.state_.backend);
    if (!source_buffer) { ggml_free(ctx); return false; }
    std::vector<float> input(length * in);
    std::vector<ggml_fp16_t> weights(ggml_nelements(w));
    for (size_t i = 0; i < input.size(); ++i) input[i] = 0.35f * sinf(i * 0.713f) + 0.05f * cosf(i * 0.113f);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = ggml_fp32_to_fp16(0.13f * cosf(i * 0.537f));
    if (channels_first) {
        std::vector<float> x_ct(length * in);
        for (int t = 0; t < length; ++t) for (int c = 0; c < in; ++c) x_ct[t * in + c] = input[c * length + t];
        ggml_backend_tensor_set(x, x_ct.data(), 0, ggml_nbytes(x));
    } else {
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
    }
    std::vector<float> f32_weights(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) f32_weights[i] = ggml_fp16_to_fp32(weights[i]);
    ggml_backend_tensor_set(w, f32 ? (void *) f32_weights.data() : (void *) weights.data(), 0, ggml_nbytes(w));
    if (trans) for (int tap = 0; tap < kernel; ++tap) {
        std::vector<ggml_fp16_t> data(in * out);
        for (int o = 0; o < out; ++o) for (int i = 0; i < in; ++i)
            data[o * in + i] = weights[(i * out + o) * kernel + tap];
        std::vector<float> floats(data.size());
        for (size_t i = 0; i < data.size(); ++i) floats[i] = ggml_fp16_to_fp32(data[i]);
        ggml_backend_tensor_set(taps[tap], f32 ? (void *) floats.data() : (void *) data.data(), 0, ggml_nbytes(taps[tap]));
    }
    if (!trans && !dw && channels_first) for (int tap = 0; tap < kernel; ++tap) {
        std::vector<ggml_fp16_t> data(in * out);
        for (int o = 0; o < out; ++o) for (int i = 0; i < in; ++i)
            data[o * in + i] = weights[(o * in + i) * kernel + tap];
        std::vector<float> floats(data.size());
        for (size_t i = 0; i < data.size(); ++i) floats[i] = ggml_fp16_to_fp32(data[i]);
        ggml_backend_tensor_set(taps[tap], f32 ? (void *) floats.data() : (void *) data.data(), 0, ggml_nbytes(taps[tap]));
    }
    if (dw) for (int tap = 0; tap < kernel; ++tap) {
        std::vector<float> data(in);
        for (int c = 0; c < in; ++c) data[c] = ggml_fp16_to_fp32(weights[c * kernel + tap]);
        ggml_backend_tensor_set(taps[tap], data.data(), 0, ggml_nbytes(taps[tap]));
    }
    decoder.state_.max_taps[w] = taps;
    ggml_tensor * result = trans ? decoder.max_transpose_conv(ctx, w, x, stride, channels_first) :
                           dw ? decoder.max_depthwise_conv(ctx, w, x, channels_first) : decoder.max_conv(ctx, w, x, dilation, channels_first);
    if (channels_first) {
        const int output_length = trans ? (length - 1) * stride + kernel : length - (kernel - 1) * dilation;
        result = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, result, out, output_length))), output_length, out, 1);
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32768, false);
    ggml_build_forward_expand(graph, result);
    ggml_tensor * native = nullptr;
    if (cpu && !f32 && !channels_first) {
        native = trans ? ggml_conv_transpose_1d(ctx, w, ggml_reshape_2d(ctx, x, length, in), stride, 0, 1) :
                 dw ? ggml_conv_1d_dw(ctx, w, x, 1, 0, 1) : ggml_conv_1d(ctx, w, x, 1, 0, dilation);
        ggml_build_forward_expand(graph, native);
    }
    bool ok = ggml_backend_sched_alloc_graph(decoder.state_.sched, graph);
    int fallback = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        if (t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
            t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_PERMUTE) continue;
        if (!cpu && ggml_backend_sched_get_tensor_backend(decoder.state_.sched, t) != decoder.state_.backend) ++fallback;
    }
    if (ok) ok = ggml_backend_sched_graph_compute(decoder.state_.sched, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> actual(ggml_nelements(result));
    if (ok) ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
    std::vector<float> original(actual.size());
    if (ok && native) ggml_backend_tensor_get(native, original.data(), 0, ggml_nbytes(native));
    const int output_length = trans ? (length - 1) * stride + kernel : length - (kernel - 1) * dilation;
    std::vector<double> expected(output_length * out, 0.0);
    auto rounded = [&](int t, int c) { return (dw && !channels_first) ?
        (double) ggml_fp16_to_fp32(ggml_fp32_to_fp16(input[c * length + t])) : (double) input[c * length + t]; };
    for (int o = 0; o < out; ++o) for (int t = 0; t < output_length; ++t) {
        double sum = 0;
        for (int k = 0; k < kernel; ++k) {
            if (trans) {
                if (t < k || (t - k) % stride) continue;
                int at = (t - k) / stride;
                if (at >= length) continue;
                for (int c = 0; c < in; ++c)
                    sum += rounded(at, c) * ggml_fp16_to_fp32(weights[(c * out + o) * kernel + k]);
            } else if (dw) sum += rounded(t + k, o) * ggml_fp16_to_fp32(weights[o * kernel + k]);
            else for (int c = 0; c < in; ++c)
                sum += rounded(t + k * dilation, c) * ggml_fp16_to_fp32(weights[(o * in + c) * kernel + k]);
        }
        expected[o * output_length + t] = sum;
    }
    double error = 0, norm = 0, dot = 0, act_norm = 0, max_abs = 0;
    double native_error = 0, lowering_delta = 0;
    int nonfinite = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        nonfinite += !std::isfinite(actual[i]);
        if (native) {
            native_error += (original[i] - expected[i]) * (original[i] - expected[i]);
            lowering_delta = std::max(lowering_delta, fabs((double)actual[i] - original[i]));
        }
        double delta = actual[i] - expected[i]; error += delta * delta; norm += expected[i] * expected[i];
        dot += actual[i] * expected[i]; act_norm += (double)actual[i] * actual[i]; max_abs = std::max(max_abs, fabs(delta));
    }
    double nrmse = sqrt(error / std::max(1e-30, norm));
    double cosine = dot / sqrt(std::max(1e-30, norm * act_norm));
    double native_nrmse = sqrt(native_error / std::max(1e-30, norm));
    // CPU F16 dot products can use half-precision vector accumulators. Require
    // parity with the native convolution in addition to the double-oracle gate.
    ok = ok && !nonfinite && !fallback && nrmse < (cpu && f32 ? 2e-6 : 0.015) && cosine > 0.999 &&
         (!native || lowering_delta < 1e-5);
    if (!ok && !strcmp(kind, "transpose")) {
        printf("DEBUG TRANSPOSE MISMATCH (stride=%d, kernel=%d):\n", stride, kernel);
        for (size_t i = 0; i < std::min((size_t)15, actual.size()); ++i) {
            printf("  [%zu] actual=%.6f expected=%.6f diff=%.6f\n", i, actual[i], (float)expected[i], (float)(actual[i] - expected[i]));
        }
    }
    printf("{\"kind\":\"%s\",\"length\":%d,\"kernel\":%d,\"stride\":%d,\"dilation\":%d,\"in\":%d,\"out\":%d,\"nrmse\":%.9g,\"cosine\":%.9g,\"max_abs\":%.9g,\"native_nrmse\":%.9g,\"native_max_delta\":%.9g,\"weight_f32\":%s,\"nonfinite\":%d,\"fallback_compute\":%d,\"pass\":%s}\n",
           kind, length, kernel, stride, dilation, in, out, nrmse, cosine, max_abs, native_nrmse, lowering_delta, f32 ? "true" : "false", nonfinite, fallback, ok ? "true" : "false");
    ggml_backend_sched_reset(decoder.state_.sched);
    ggml_backend_buffer_free(source_buffer); ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc == 4 && !strcmp(argv[1], "--shape")) {
        const int frames = atoi(argv[3]);
        if (frames < 1 || frames > 100) return 2;
        setenv("QWEN3_TTS_DECODER_BACKEND", "cpu", 1);
        setenv("QWEN3_TTS_MAX_VOCODER", "1", 1);
        AudioTokenizerDecoder decoder;
        if (!decoder.load_model(argv[2])) {
            fprintf(stderr, "%s\n", decoder.get_error().c_str()); return 3;
        }
        ggml_cgraph * graph = decoder.build_graph(frames);
        printf("{\"shape_frames\":%d,\"nodes\":%d,\"metadata_bytes\":%zu,\"resident_derived_bytes\":%zu}\n",
               frames, ggml_graph_n_nodes(graph), decoder.state_.compute_meta.size(),
               ggml_backend_buffer_get_size(decoder.state_.max_buffer));
        return 0;
    }
    const char * backend = argc == 3 && !strcmp(argv[1], "--backend") ? argv[2] : "cpu";
    int pass = 0, total = 0;
    for (int n : {19, 1031}) {
        for (bool cf : {false, true}) {
            for (int k : {1, 3, 7}) { ++total; pass += run_case("conv", k, 1, k == 7 ? 3 : 1, n + 21, 8, 5, backend, false, cf); }
            for (int stride : {2, 3, 4, 5, 8}) { ++total; pass += run_case("transpose", stride * 2, stride, 1, n, 8, 5, backend, false, cf); }
            ++total; pass += run_case("depthwise", 7, 1, 1, n + 6, 9, 9, backend, false, cf);
            ++total; pass += run_case("conv", 3, 1, 1, n + 2, 8, 5, backend, true, cf);
            ++total; pass += run_case("transpose", 6, 3, 1, n, 8, 5, backend, true, cf);
            ++total; pass += run_case("depthwise", 7, 1, 1, n + 6, 9, 9, backend, true, cf);
        }
    }
    printf("SUMMARY %d/%d passed\n", pass, total);
    return pass == total ? 0 : 1;
}
