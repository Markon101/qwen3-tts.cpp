// CPU controls for the shared speaker convolution and optional FFT frontend.
#include "audio_tokenizer_encoder.h"
#include "qwen3_tts.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace qwen3_tts;

// Independent double-precision DFT oracle for the adversarial pure-tone
// input. Unlike the legacy frontend it forms phases and sums in double.
static std::vector<float> double_dft_mel(const std::vector<float> & samples) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    const int fft = 1024, hop = 256, bins = 513, mels = 128, pad = 384;
    const int frames = (samples.size() + 2 * pad - fft) / hop + 1;
    std::vector<double> basis(mels * bins), hz(mels + 2);
    auto hz_to_mel = [](double value) {
        return value < 1000 ? value / (200.0 / 3) :
               15 + std::log(value / 1000) / (std::log(6.4) / 27);
    };
    auto mel_to_hz = [](double value) {
        return value < 15 ? value * (200.0 / 3) :
               1000 * std::exp((value - 15) * (std::log(6.4) / 27));
    };
    for (int i = 0; i < mels + 2; ++i) hz[i] = mel_to_hz(hz_to_mel(12000) * i / (mels + 1));
    for (int m = 0; m < mels; ++m) for (int k = 0; k < bins; ++k) {
        const double frequency = 24000.0 * k / fft;
        basis[m * bins + k] = std::max(0.0, std::min((frequency - hz[m]) / (hz[m+1] - hz[m]),
                     (hz[m+2] - frequency) / (hz[m+2] - hz[m+1]))) * 2 / (hz[m+2] - hz[m]);
    }
    std::vector<float> result(mels * frames);
    std::vector<double> frame(fft), magnitude(bins);
    for (int f = 0; f < frames; ++f) {
        for (int i = 0; i < fft; ++i) {
            int index = f * hop + i;
            if (index < pad) index = pad - 1 - index;
            else if (index >= pad + (int)samples.size()) index = 2 * samples.size() - (index - pad) - 2;
            else index -= pad;
            index = std::max(0, std::min((int)samples.size() - 1, index));
            // Keep the production F32 Hann window identical so this oracle
            // isolates Fourier phase/summation accuracy, not window changes.
            const float window = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979323846f * i / fft));
            frame[i] = samples[index] * window;
        }
        for (int k = 0; k < bins; ++k) {
            double real = 0, imag = 0;
            for (int i = 0; i < fft; ++i) {
                const double angle = -2 * pi * k * i / fft;
                real += frame[i] * std::cos(angle);
                imag += frame[i] * std::sin(angle);
            }
            magnitude[k] = std::sqrt(real * real + imag * imag + 1e-9);
        }
        for (int m = 0; m < mels; ++m) {
            double sum = 0;
            for (int k = 0; k < bins; ++k) sum += basis[m * bins + k] * magnitude[k];
            result[m * frames + f] = std::log(std::max(sum, 1e-5));
        }
    }
    return result;
}

static bool compare_mel(const char * name, const std::vector<float> & samples) {
    speaker_encoder_config config;
    std::vector<float> old_mel, fft_mel;
    int old_frames = 0, fft_frames = 0;
    std::string error;
    if (!speaker_encoder_detail::compute_mel_spectrogram(config, samples.data(), samples.size(),
                 old_mel, old_frames, error, false) ||
        !speaker_encoder_detail::compute_mel_spectrogram(config, samples.data(), samples.size(),
                 fft_mel, fft_frames, error, true)) {
        fprintf(stderr, "%s\n", error.c_str());
        return false;
    }
    if (old_frames != fft_frames || old_mel.size() != fft_mel.size()) return false;
    double square_error = 0, energy = 0, max_error = 0;
    for (size_t i = 0; i < old_mel.size(); ++i) {
        const double delta = double(old_mel[i]) - fft_mel[i];
        square_error += delta * delta;
        energy += double(old_mel[i]) * old_mel[i];
        max_error = std::max(max_error, std::fabs(delta));
    }
    const double nrmse = std::sqrt(square_error / std::max(energy, 1e-30));
    // The original F32-phase DFT leaks energy into near-floor pure-tone
    // bins. Permit its measured log-mel discrepancy, but additionally require
    // FFT to agree tightly with an independent double DFT on that input.
    double fft_oracle_error = 0, legacy_oracle_error = 0;
    if (!strcmp(name, "sine")) {
        const auto oracle = double_dft_mel(samples);
        for (size_t i = 0; i < oracle.size(); ++i) {
            fft_oracle_error = std::max(fft_oracle_error, std::fabs(double(fft_mel[i]) - oracle[i]));
            legacy_oracle_error = std::max(legacy_oracle_error, std::fabs(double(old_mel[i]) - oracle[i]));
        }
    }
    const bool pass = std::isfinite(nrmse) && nrmse < 0.002 && max_error < 0.15 && fft_oracle_error < 0.001;
    if (!strcmp(name, "sine"))
        printf("{\"test\":\"mel_double_dft_oracle\",\"fft_max_log_error\":%.9g,\"legacy_max_log_error\":%.9g}\n",
               fft_oracle_error, legacy_oracle_error);
    printf("{\"test\":\"mel_fft\",\"input\":\"%s\",\"frames\":%d,\"nrmse\":%.9g,\"max_log_mel_error\":%.9g,\"pass\":%s}\n",
           name, old_frames, nrmse, max_error, pass ? "true" : "false");
    return pass;
}

int main(int argc, char ** argv) {
    if (argc != 1 && (argc != 3 || strcmp(argv[1], "--wav"))) {
        fprintf(stderr, "Usage: %s [--wav SOURCE.wav]\n", argv[0]);
        return 2;
    }
    int failed = 0;
    for (int time : {17, 37, 1023, 1025}) {
        for (int kernel : {1, 3, 5}) {
            for (int dilation : {1, 2, 4}) {
                ggml_context * ctx = ggml_init({128 * 1024 * 1024, nullptr, false});
                if (!ctx) return 2;
                ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, time, 8, 1);
                ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kernel, 8, 16);
                for (int i = 0; i < time * 8; ++i) ((float *)input->data)[i] = sinf(i * 0.043f);
                for (int i = 0; i < kernel * 8 * 16; ++i)
                    ((ggml_fp16_t *)weights->data)[i] = ggml_fp32_to_fp16(cosf(i * 0.031f) * 0.1f);
                ggml_tensor * original = ggml_conv_1d(ctx, weights, input, 1, 0, dilation);
                ggml_tensor * lowered = speaker_encoder_detail::lowered_conv1d(ctx, weights, input, 1, dilation);
                ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);
                ggml_build_forward_expand(graph, original);
                ggml_build_forward_expand(graph, lowered);
                ggml_backend_t cpu = ggml_backend_cpu_init();
                if (!cpu) { ggml_free(ctx); return 2; }
                ggml_backend_cpu_set_n_threads(cpu, 2);
                if (ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS) return 2;
                const bool pass = ggml_nbytes(original) == ggml_nbytes(lowered) &&
                    !memcmp(original->data, lowered->data, ggml_nbytes(original));
                printf("{\"test\":\"conv_lowering\",\"time\":%d,\"kernel\":%d,\"dilation\":%d,\"bit_equal\":%s}\n",
                       time, kernel, dilation, pass ? "true" : "false");
                failed += !pass;
                ggml_backend_free(cpu);
                ggml_free(ctx);
            }
        }
    }
    std::vector<float> samples(4096, 0);
    failed += !compare_mel("silence", samples);
    samples[1000] = 0.8f;
    failed += !compare_mel("impulse", samples);
    for (size_t i = 0; i < samples.size(); ++i)
        samples[i] = 0.5f * std::sin(2 * 3.14159265358979323846 * 437 * i / 24000);
    failed += !compare_mel("sine", samples);
    uint32_t random = 123456789;
    for (float & value : samples) {
        random = random * 1664525u + 1013904223u;
        value = (float(int32_t(random)) / 2147483648.0f) * 0.3f;
    }
    failed += !compare_mel("noise", samples);
    if (argc == 3) {
        std::vector<float> input;
        int rate = 0;
        if (!load_audio_file(argv[2], input, rate)) return 2;
        if (rate == 24000) samples = std::move(input);
        else resample_audio(input.data(), input.size(), rate, 24000, samples);
        // A bounded first second keeps the numerical control quick and never
        // interprets its run time as a benchmark.
        if (samples.size() > 24000) samples.resize(24000);
        failed += !compare_mel("source_wav_first_second", samples);
    }
    printf("{\"test\":\"summary\",\"failures\":%d}\n", failed);
    return failed ? 1 : 0;
}
