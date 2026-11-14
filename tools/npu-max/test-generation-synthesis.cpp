// Full cold ICL synthesis with retained codec IDs and pre-WAV float samples.
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
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iterator>
using namespace qwen3_tts;
using Clock = std::chrono::steady_clock;
static std::string read(const char * path) {
    std::ifstream f(path);
    return std::string(std::istreambuf_iterator<char>(f), {});
}
static bool save(const std::string & path, const void * data, size_t bytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(data, 1, bytes, f) == bytes;
    return fclose(f) == 0 && ok;
}
int main(int argc, char ** argv) {
    if (argc != 10) {
        fprintf(stderr, "Usage: %s MODEL TOKENIZER REFERENCE REF_TEXT_FILE TEXT_FILE OUT_PREFIX MAX_FRAMES SEED TEMPERATURE\n", argv[0]);
        return 2;
    }
    const auto start = Clock::now();
    Qwen3TTS tts;
    if (!tts.load_models(argv[1], argv[2])) { fprintf(stderr, "%s\n", tts.get_error().c_str()); return 3; }
    const double load_ms = std::chrono::duration<double, std::milli>(Clock::now()-start).count();
    tts_params p;
    p.n_threads = 2;
    p.print_progress = false;
    p.print_timing = true;
    p.language = "english";
    p.max_audio_tokens = atoi(argv[7]);
    p.ref_text = read(argv[4]);
    p.icl_mode = true;
    p.temperature = p.subtalker_temperature = strtof(argv[9], nullptr);
    p.top_k = p.subtalker_top_k = 50;
    p.top_p = p.subtalker_top_p = 1;
    tts.transformer_.set_seed((uint32_t)strtoul(argv[8], nullptr, 10));
    std::vector<int32_t> codes;
    tts.transformer_.set_frame_callback([&](const int32_t * row, int32_t n) {
        codes.insert(codes.end(), row, row+n); return true;
    });
    const auto synth_start = Clock::now();
    tts_result r = tts.synthesize_with_voice(read(argv[5]), argv[3], p);
    const double synth_wall_ms = std::chrono::duration<double, std::milli>(Clock::now()-synth_start).count();
    if (!r.success) { fprintf(stderr, "%s\n", r.error_msg.c_str()); return 4; }
    if (r.audio.empty() || codes.empty() || codes.size()%16 || r.audio.size() != codes.size()/16*1920) return 5;
    double norm = 0;
    for (float x : r.audio) { if (!std::isfinite(x)) return 6; norm += double(x)*x; }
    if (norm/r.audio.size() < 1e-12) return 7;
    const std::string prefix = argv[6];
    if (!save(prefix+"-codes.i32", codes.data(), codes.size()*sizeof(int32_t)) ||
        !save(prefix+"-audio.f32", r.audio.data(), r.audio.size()*sizeof(float)) ||
        !save(prefix+"-reference-codes.i32", tts.voice_ref_cache_.ref_codes.data(), tts.voice_ref_cache_.ref_codes.size()*sizeof(int32_t)) ||
        !save_audio_file(prefix+".wav", r.audio, r.sample_rate)) return 8;
    printf("{\"frames\":%zu,\"samples\":%zu,\"duration_s\":%.6f,\"load_ms\":%.3f,\"speaker_ms\":%lld,\"generate_ms\":%lld,\"decode_ms\":%lld,\"total_ms\":%lld,\"synth_wall_ms\":%.3f,\"rms\":%.9g,\"rss_peak_bytes\":%llu}\n",
        codes.size()/16, r.audio.size(), double(r.audio.size())/r.sample_rate, load_ms,
        (long long)r.t_encode_ms, (long long)r.t_generate_ms, (long long)r.t_decode_ms,
        (long long)r.t_total_ms, synth_wall_ms, sqrt(norm/r.audio.size()),
        (unsigned long long)r.mem_rss_peak_bytes);
    tts.transformer_.clear_frame_callback();
    return 0;
}
