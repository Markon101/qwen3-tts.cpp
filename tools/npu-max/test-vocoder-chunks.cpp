// Same-code waveform and callback checks; deliberately no speed assertions.
// Access orchestration internals to load only the vocoder, not a multi-GiB talker.
#include "text_tokenizer.h"
#include "tts_transformer.h"
#include "audio_tokenizer_encoder.h"
#include "audio_tokenizer_decoder.h"
#include "mimi_encoder.h"
#include <functional>
#include <thread>
#define private public
#include "qwen3_tts.h"
#undef private
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
using namespace qwen3_tts;

static bool save(const std::string & path, const std::vector<float> & audio) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(audio.data(), sizeof(float), audio.size(), f) == audio.size();
    return fclose(f) == 0 && ok;
}

static bool valid_audio(const std::vector<float> & audio, int frames) {
    if (audio.size() != size_t(frames) * 1920) return false;
    for (float x : audio) if (!std::isfinite(x)) return false;
    return true;
}

static void difference(const char * label, const std::vector<float> & reference,
                       const std::vector<float> & candidate, size_t start) {
    double error = 0, norm = 0, peak = 0;
    for (size_t i = start; i < reference.size(); ++i) {
        const double d = double(candidate[i]) - reference[i];
        error += d * d; norm += double(reference[i]) * reference[i];
        peak = std::max(peak, std::fabs(d));
    }
    printf("{\"comparison\":\"%s\",\"start_sample\":%zu,\"nrmse\":%.9g,\"max_abs\":%.9g}\n",
           label, start, std::sqrt(error / std::max(norm, 1e-30)), peak);
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s TOKENIZER CODE_ROWS_I32 EXISTING_OUTPUT_DIR cpu|hexagon|opencl\n", argv[0]);
        return 2;
    }
    const bool htp = !strcmp(argv[4], "hexagon");
    if (!htp && strcmp(argv[4], "cpu") && strcmp(argv[4], "opencl")) return 2;
    setenv("QWEN3_TTS_DECODER_BACKEND", argv[4], 1);
    setenv("QWEN3_TTS_MAX_VOCODER", htp ? "1" : "0", 1);
    setenv("QWEN3_TTS_MAX_STRICT", htp ? "1" : "0", 1);
    setenv("QWEN3_TTS_VALIDATE_AUDIO", "1", 1);
    unsetenv("QWEN3_TTS_FORCE_CPU");
    FILE * f = fopen(argv[2], "rb");
    if (!f) return 2;
    std::vector<int32_t> seed;
    int32_t row[16];
    while (true) {
        const size_t n = fread(row, sizeof(int32_t), 16, f);
        if (!n) break;
        if (n != 16 || seed.size() >= 16 * 4096) { fclose(f); return 2; }
        for (int32_t code : row) if (code < 0 || code >= 2048) { fclose(f); return 2; }
        seed.insert(seed.end(), row, row + 16);
    }
    const bool read_ok = !ferror(f) && !seed.empty();
    fclose(f);
    if (!read_ok) return 2;
    std::vector<int32_t> codes(257 * 16);
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = seed[i % seed.size()];
    Qwen3TTS tts;
    auto & decoder = tts.audio_decoder_;
    std::vector<float> rejected;
    for (int n : {0, -1, 16, 257, std::numeric_limits<int32_t>::max()}) {
        if (decoder.decode(codes.data(), 8, rejected, n) ||
            decoder.get_error().find("17 to 256") == std::string::npos) return 3;
    }
    if (!decoder.load_model(argv[1])) { fprintf(stderr, "%s\n", decoder.get_error().c_str()); return 3; }
    tts.decoder_loaded_ = true;
    decoder.set_n_threads(2);
    const std::string out = argv[3];
    std::vector<float> implicit, explicit_default;
    if (!decoder.decode(codes.data(), 8, implicit) ||
        !decoder.decode(codes.data(), 8, explicit_default, 100) || implicit != explicit_default ||
        !valid_audio(implicit, 8) || !save(out + "/default-8.f32", implicit)) return 4;
    std::vector<int32_t> full_codes(codes.begin(), codes.begin() + 256 * 16);
    std::vector<float> full, window100, window200;
    // Drive the real orchestration path, checking tts_params propagation too.
    auto decode = [&](int window, std::vector<float> & audio) {
        tts_params params;
        params.vocoder_chunk_frames = window;
        tts_result result;
        if (!tts.decode_codes_streaming(full_codes, result, params) || !valid_audio(result.audio, 256)) {
            fprintf(stderr, "%s\n", result.error_msg.c_str()); return false;
        }
        audio = std::move(result.audio);
        return save(out + "/window-" + std::to_string(window) + ".f32", audio);
    };
    if (!decode(256, full) || !decode(100, window100) || !decode(200, window200)) return 5;
    difference("window100_vs_full256", full, window100, 0);
    difference("window100_after_boundary", full, window100, 84 * 1920);
    difference("window200_vs_full256", full, window200, 0);
    difference("window200_after_boundary", full, window200, 184 * 1920);
    std::vector<float> longer;
    if (!decoder.decode(codes.data(), 257, longer, 256) || !valid_audio(longer, 257) ||
        !save(out + "/window-256-frames-257.f32", longer)) return 6;
    int calls = 0, last_count = 0;
    bool callback_ok = true;
    std::vector<float> emitted;
    tts.set_audio_chunk_callback([&](const float * data, int n, int sr, int last) {
        ++calls; last_count += last;
        callback_ok &= sr == 24000 && n > 0 && n <= 184 * 1920;
        emitted.insert(emitted.end(), data, data + n);
        return 0;
    }, 1024);
    std::vector<float> streamed;
    if (!decode(200, streamed) || !callback_ok || calls != 2 || last_count != 1 ||
        streamed != window200 || emitted != window200) return 7;
    tts.set_audio_chunk_callback([](const float *, int, int, int) { return 1; }, 1024);
    tts_params params; params.vocoder_chunk_frames = 200;
    tts_result aborted;
    if (!tts.decode_codes_streaming(full_codes, aborted, params) ||
        !valid_audio(aborted.audio, 184) ||
        !std::equal(aborted.audio.begin(), aborted.audio.end(), window200.begin())) return 8;
    params.vocoder_chunk_frames = 16;
    tts_result invalid;
    if (tts.decode_codes_streaming(full_codes, invalid, params) || !invalid.audio.empty()) return 9;
    printf("{\"pass\":true,\"backend\":\"%s\",\"seed_frames\":%zu,\"test_frames\":256,"
           "\"longer_frames\":257,\"callback_calls\":%d,\"default_bit_exact\":true,"
           "\"callback_bit_exact\":true,\"scope\":\"repeated saved codes; numerical checks, not listening or speed\"}\n",
           argv[4], seed.size() / 16, calls);
    return 0;
}
