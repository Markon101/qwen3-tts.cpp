#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

static bool run_case(ggml_backend_t backend, int k, int m, int n, float activation_scale = 1.0f,
                     const char *weight_path = nullptr, const char *input_path = nullptr) {
  ggml_init_params params = {4 * 1024 * 1024, nullptr, true};
  ggml_context *weights_ctx = ggml_init(params);
  ggml_context *compute_ctx = ggml_init(params);
  if (!weights_ctx || !compute_ctx)
    return false;
  ggml_tensor *weights = ggml_new_tensor_2d(weights_ctx, GGML_TYPE_Q8_0, k, m);
  ggml_set_name(weights, "npu_probe_weights");
  ggml_tensor *input = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, k, n);
  ggml_set_name(input, "npu_probe_input");
  ggml_tensor *output = ggml_mul_mat(compute_ctx, weights, input);
  ggml_set_name(output, "npu_probe_output");
  if (!ggml_backend_dev_supports_op(ggml_backend_get_device(backend), output)) {
    fprintf(stderr, "HTP declined Q8 matrix shape k=%d m=%d n=%d\n", k, m, n);
    return false;
  }
  ggml_cgraph *graph = ggml_new_graph(compute_ctx);
  ggml_build_forward_expand(graph, output);
  ggml_backend_buffer_t weight_buffer =
      ggml_backend_alloc_ctx_tensors(weights_ctx, backend);
  ggml_backend_buffer_t compute_buffer =
      ggml_backend_alloc_ctx_tensors(compute_ctx, backend);
  if (!weight_buffer || !compute_buffer)
    return false;
  ggml_backend_buffer_set_usage(weight_buffer,
                                GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  std::vector<float> values(k * m), restored(k * m), x(k * n), y(m * n),
      reference(m * n);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = .12f * std::sin(i * .017f) + .03f * std::cos(i * .037f);
  }
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = activation_scale * (.3f * std::cos(i * .031f) + .07f * std::sin(i * .019f));
  }
  std::vector<unsigned char> canonical(ggml_nbytes(weights)),
      roundtrip(canonical.size());
  if (weight_path && input_path) {
    FILE *f = fopen(weight_path, "rb");
    if (!f) return false;
    bool loaded = fread(canonical.data(), 1, canonical.size(), f) == canonical.size() && fgetc(f) == EOF;
    fclose(f);
    f = fopen(input_path, "rb");
    if (!f) return false;
    loaded = loaded && fread(x.data(), 4, x.size(), f) == x.size() && fgetc(f) == EOF;
    fclose(f);
    if (!loaded) return false;
    for (float v : x) {
      if (!std::isfinite(v)) return false;
      activation_scale = std::max(activation_scale, std::fabs(v));
    }
  } else {
  size_t quantized = ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(),
                                         canonical.data(), 0, m, k, nullptr);
  if (quantized != canonical.size())
    return false;
  }
  ggml_get_type_traits(GGML_TYPE_Q8_0)
      ->to_float(canonical.data(), restored.data(), k * m);
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < m; ++row) {
      double sum = 0;
      for (int i = 0; i < k; ++i)
        sum += double(restored[row * k + i]) * x[col * k + i];
      reference[col * m + row] = float(sum);
    }
  }
  ggml_backend_tensor_set(weights, canonical.data(), 0, canonical.size());
  ggml_backend_tensor_get(weights, roundtrip.data(), 0, roundtrip.size());
  bool exact_roundtrip = roundtrip == canonical;
  if (!ggml_backend_dev_supports_op(ggml_backend_get_device(backend), output)) {
    fprintf(stderr, "HTP declined Q8 matrix after buffer allocation\n");
    return false;
  }
  ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
  auto start = std::chrono::steady_clock::now();
  ggml_status status = ggml_backend_graph_compute(backend, graph);
  ggml_backend_synchronize(backend);
  double wall_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - start)
                       .count();
  if (status != GGML_STATUS_SUCCESS)
    return false;
  ggml_backend_tensor_get(output, y.data(), 0, y.size() * sizeof(float));
  double max_error = 0, squared_error = 0, energy = 0, dot = 0,
         output_energy = 0;
  int nonfinite = 0;
  for (size_t i = 0; i < y.size(); ++i) {
    if (!std::isfinite(y[i])) {
      ++nonfinite;
      continue;
    }
    double error = double(y[i]) - reference[i];
    max_error = std::max(max_error, std::fabs(error));
    squared_error += error * error;
    energy += double(reference[i]) * reference[i];
    output_energy += double(y[i]) * y[i];
    dot += double(reference[i]) * y[i];
  }
  double nrmse = std::sqrt(squared_error / std::max(energy, 1.e-30));
  double cosine = energy == 0 ? (output_energy == 0 ? 1 : 0) :
                  dot / std::sqrt(std::max(energy * output_energy, 1.e-30));
  bool pass = exact_roundtrip && !nonfinite && nrmse < .02 && max_error < .05 * std::max(1.0f, activation_scale) &&
              cosine > .999;
  printf("RESULT {\"K\":%d,\"M\":%d,\"N\":%d,\"activation_scale\":%.9g,"
         "\"wall_ms\":%.6f,\"q8_roundtrip_exact\":%s,\"nonfinite\":%d,",
         k, m, n, activation_scale, wall_ms, exact_roundtrip ? "true" : "false", nonfinite);
  if (nonfinite) printf("\"max_abs\":null,\"nrmse\":null,\"cosine\":null,");
  else printf("\"max_abs\":%.9g,\"nrmse\":%.9g,\"cosine\":%.9g,", max_error, nrmse, cosine);
  printf("\"pass\":%s}\n", pass ? "true" : "false");
  fflush(stdout);
  char path[512];
  const char *result_dir = getenv("QWEN_NPU_RESULT_DIR");
  if (!result_dir) result_dir = ".cache/npu";
  snprintf(path, sizeof(path), "%s/result-%d-%d-%d-%.3g.f32", result_dir, k, m, n, activation_scale);
  FILE *raw = fopen(path, "wb");
  if (raw) {
    fwrite(y.data(), sizeof(float), y.size(), raw);
    fclose(raw);
  }
  snprintf(path, sizeof(path), "%s/reference-%d-%d-%d-%.3g.f32", result_dir, k, m, n, activation_scale);
  raw = fopen(path, "wb");
  if (raw) {
    fwrite(reference.data(), sizeof(float), reference.size(), raw);
    fclose(raw);
  }
  ggml_backend_buffer_free(compute_buffer);
  ggml_backend_buffer_free(weight_buffer);
  ggml_free(compute_ctx);
  ggml_free(weights_ctx);
  return pass;
}

int main(int argc, char **argv) {
  if (argc != 1 && argc != 6) {
    fprintf(stderr, "Usage: %s [K M N canonical-weights.raw input-f32.raw]\n", argv[0]);
    return 2;
  }
  ggml_backend_t backend = ggml_backend_init_by_name("HTP0", nullptr);
  if (!backend) {
    fprintf(stderr,
            "HTP0 initialization failed; no CPU fallback is permitted\n");
    return 2;
  }
  const char *name = ggml_backend_dev_name(ggml_backend_get_device(backend));
  printf("BACKEND %s\n", name);
  if (strncmp(name, "HTP", 3))
    return 3;
  bool passed;
  if (argc == 6) {
    int k = atoi(argv[1]), m = atoi(argv[2]), n = atoi(argv[3]);
    if (k < 32 || k > 16384 || k % 32 || m < 1 || m > 16384 || n < 1 || n > 128) return 2;
    passed = run_case(backend, k, m, n, 1, argv[4], argv[5]);
  } else passed = run_case(backend, 128, 128, 1) &&
                run_case(backend, 512, 256, 1) &&
                run_case(backend, 128, 128, 8) &&
                run_case(backend, 1024, 1024, 1) &&
                run_case(backend, 1024, 1024, 2) &&
                run_case(backend, 1024, 3072, 1) &&
                run_case(backend, 3072, 1024, 1) &&
                run_case(backend, 1024, 256, 1, 1000000.0f) &&
                run_case(backend, 3072, 1024, 2, 1000000.0f) &&
                run_case(backend, 1024, 256, 1, 0.0f);
  ggml_backend_free(backend);
  return passed ? 0 : 4;
}
