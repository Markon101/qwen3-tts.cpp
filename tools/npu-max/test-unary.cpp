// Public independent numeric controls for the NPU-Max F32 pointwise kernels.
// Direct HTP0 dispatch only: no scheduler or CPU execution fallback.
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
ggml_tensor *make_op(ggml_context *ctx, ggml_tensor *input, int op) {
    if (op == 0) return ggml_sin(ctx, input);
    if (op == 1) return ggml_relu(ctx, input);
    return ggml_clamp(ctx, input, -.75f, .625f);
}
float fixture(size_t i) {
    const float edge[] = {0.f, -0.f, 1.e-35f, -1.e-35f, 1.e-6f, -1.e-6f,
        -.75f, .625f, 1.57079632679f, -1.57079632679f, 3.14159265359f,
        -3.14159265359f, 127.999f, -127.999f, 128.f, -128.f,
        128.001f, -128.001f, 10000.f, -10000.f, 1.e10f, -1.e10f,
        1.e30f, -1.e30f};
    if (i % 3 == 0) return edge[(i / 3) % (sizeof(edge) / sizeof(edge[0]))];
    return 128.f * std::sin(float(i) * .173f);
}
bool run(ggml_backend_t backend, int width, bool strided, int op) {
    const int rows = 5, padded_width = width + (strided ? 7 : 0);
    ggml_init_params params{1024 * 1024, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    ggml_tensor *base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, padded_width, rows);
    ggml_tensor *input = strided ? ggml_view_2d(ctx, base, width, rows, padded_width * sizeof(float), 0) : base;
    ggml_tensor *out = strided && op == 0 ? ggml_sin_inplace(ctx, input) :
        (strided && op == 1 ? ggml_relu_inplace(ctx, input) : make_op(ctx, input, op));
    ggml_tensor *guard = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_set_input(base); ggml_set_output(out);
    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    bool pass = buffer && ggml_backend_dev_supports_op(ggml_backend_get_device(backend), out);
    double maximum = 0;
    bool printed_error = false;
    if (pass) {
        std::vector<float> values(size_t(padded_width) * rows, 9876.f), result(size_t(width) * rows);
        std::vector<float> canary(64, 3.25f);
        for (int row = 0; row < rows; ++row) for (int x = 0; x < width; ++x)
            values[size_t(row) * padded_width + x] = fixture(size_t(row) * width + x);
        ggml_backend_tensor_set(base, values.data(), 0, values.size() * sizeof(float));
        ggml_backend_tensor_set(guard, canary.data(), 0, canary.size() * sizeof(float));
        pass = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        for (int row = 0; row < rows; ++row)
            ggml_backend_tensor_get(out, result.data() + size_t(row) * width, row * out->nb[1], width * sizeof(float));
        ggml_backend_tensor_get(guard, canary.data(), 0, canary.size() * sizeof(float));
        for (size_t i = 0; i < result.size(); ++i) {
            const float v = fixture(i);
            const double expected = op == 0 ? std::sin(double(v)) :
                (op == 1 ? (v > 0.f ? v : 0.f) : std::min(.625f, std::max(-.75f, v)));
            const double error = std::fabs(double(result[i]) - expected);
            maximum = std::max(maximum, error);
            if (!std::isfinite(result[i]) || error > (op == 0 ? 2.e-5 : 0.)) {
                pass = false;
                if (!printed_error) {
                    std::fprintf(stderr, "FIRST_ERROR op=%d width=%d strided=%d index=%zu input=%.9g actual=%.9g expected=%.17g\n", op, width, int(strided), i, v, result[i], expected);
                    printed_error = true;
                }
            }
        }
        for (float v : canary) if (v != 3.25f) pass = false;
    }
    std::printf("RESULT {\"op\":\"%s\",\"width\":%d,\"rows\":5,\"strided\":%s,\"max_abs\":%.9g,\"pass\":%s}\n",
        op == 0 ? "sin" : (op == 1 ? "relu" : "clamp"), width, strided ? "true" : "false", maximum, pass ? "true" : "false");
    std::fflush(stdout);
    if (buffer) ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return pass;
}
bool support_controls(ggml_backend_t backend) {
    ggml_init_params params{1024 * 1024, nullptr, true};
    ggml_context *ctx = ggml_init(params);
    auto dev = ggml_backend_get_device(backend);
    bool pass = true;
    for (int op = 0; op < 3; ++op) {
        auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16777217, 2);
        auto out = make_op(ctx, input, op);
        if (ggml_backend_dev_supports_op(dev, out)) pass = false;
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 2);
        out = make_op(ctx, input, op);
        input->nb[0] = 8; // metadata-only hostile element stride
        if (ggml_backend_dev_supports_op(dev, out)) pass = false;
    }
    auto half = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 2);
    auto clamp = ggml_clamp(ctx, half, -1.f, 1.f);
    if (ggml_backend_dev_supports_op(dev, clamp)) pass = false;
    auto input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    for (const auto &bounds : std::vector<std::vector<float>>{{NAN, 1.f}, {-1.f, INFINITY}, {1.f, -1.f}}) {
        auto out = ggml_clamp(ctx, input, bounds[0], bounds[1]);
        if (ggml_backend_dev_supports_op(dev, out)) pass = false;
    }
    std::printf("SUPPORT {\"checks\":10,\"pass\":%s}\n", pass ? "true" : "false");
    ggml_free(ctx);
    return pass;
}
bool run_diag_mask_and_softmax(ggml_backend_t backend) {
    bool pass = true;
    for (int width : {7, 33, 63, 636}) {
        ggml_init_params params{16 * 1024 * 1024, nullptr, true};
        ggml_context *ctx = ggml_init(params);
        ggml_tensor *in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 5);
        ggml_tensor *masked = ggml_diag_mask_inf(ctx, in, 2);
        ggml_tensor *sm = ggml_soft_max(ctx, in);
        ggml_set_input(in); ggml_set_output(masked); ggml_set_output(sm);
        ggml_cgraph *graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, masked);
        ggml_build_forward_expand(graph, sm);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        auto dev = ggml_backend_get_device(backend);
        bool ok = buffer && ggml_backend_dev_supports_op(dev, masked) && ggml_backend_dev_supports_op(dev, sm);
        if (ok) {
            std::vector<float> vals(size_t(width) * 5), out_mask(vals.size()), out_sm(vals.size());
            for (size_t i = 0; i < vals.size(); ++i) vals[i] = std::sin(float(i) * 0.37f) * 3.0f;
            ggml_backend_tensor_set(in, vals.data(), 0, vals.size() * sizeof(float));
            ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(masked, out_mask.data(), 0, out_mask.size() * sizeof(float));
            ggml_backend_tensor_get(sm, out_sm.data(), 0, out_sm.size() * sizeof(float));
            for (int r = 0; r < 5; ++r) {
                double max_v = -1e30, sum_v = 0;
                for (int c = 0; c < width; ++c) {
                    float v = vals[r * width + c];
                    float m = out_mask[r * width + c];
                    if (c > 2 + r) { if (!std::isinf(m) || m > 0) ok = false; }
                    else { if (m != v) ok = false; }
                    max_v = std::max(max_v, double(v));
                }
                for (int c = 0; c < width; ++c) sum_v += std::exp(double(vals[r * width + c]) - max_v);
                for (int c = 0; c < width; ++c) {
                    double exp_sm = std::exp(double(vals[r * width + c]) - max_v) / sum_v;
                    if (std::fabs(double(out_sm[r * width + c]) - exp_sm) > 5e-5) ok = false;
                }
            }
        }
        std::printf("RESULT {\"op\":\"diag_mask_inf+softmax\",\"width\":%d,\"rows\":5,\"pass\":%s}\n", width, ok ? "true" : "false");
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        if (!ok) pass = false;
    }
    return pass;
}
}
int main(int argc, char **argv) {
    const bool invalid = argc == 2 && std::strcmp(argv[1], "--invalid-f16") == 0;
    if (argc != 1 && !invalid) return 2;
    ggml_backend_t backend = ggml_backend_init_by_name("HTP0", nullptr);
    if (!backend || std::strcmp(ggml_backend_dev_name(ggml_backend_get_device(backend)), "HTP0") != 0) {
        std::fprintf(stderr, "HTP0 required; no CPU fallback\n"); return 2;
    }
    if (invalid) {
        // Deliberately bypass supports_op: unsupported DSP arithmetic must fail
        // closed rather than return success with uninitialized output tensors.
        ggml_init_params params{1024 * 1024, nullptr, true};
        auto ctx = ggml_init(params);
        auto input = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 32);
        auto out = ggml_clamp(ctx, input, -1.f, 1.f);
        ggml_set_input(input); ggml_set_output(out);
        auto graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, out);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_buffer_clear(buffer, 0);
        ggml_backend_graph_compute(backend, graph);
        std::fprintf(stderr, "ERROR: invalid F16 clamp returned successfully\n");
        return 9;
    }
    int cases = 0, failures = (support_controls(backend) && run_diag_mask_and_softmax(backend)) ? 0 : 1;
    for (int width : {1, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 4097, 65537, 192000})
        for (bool strided : {false, true}) for (int op = 0; op < 3; ++op) {
            ++cases;
            if (!run(backend, width, strided, op)) ++failures;
        }
    std::printf("SUMMARY {\"cases\":%d,\"support_checks\":10,\"failures\":%d,\"pass\":%s}\n",
        cases, failures, failures == 0 ? "true" : "false");
    ggml_backend_free(backend);
    return failures == 0 ? 0 : 4;
}
