/* qwen3tts_c_api.cpp — C API wrapper for Go/Nim/Python FFI. */
#include "qwen3_tts.h"
#include "qwen3tts_c_api.h"
#include "audio_player.h"

#ifdef __APPLE__
#include <objc/objc.h>
#include <objc/message.h>
static void * new_autorelease_pool() {
    return (void *)((id(*)(id,SEL))objc_msgSend)(
        (id)objc_getClass("NSAutoreleasePool"), sel_registerName("new"));
}
static void drain_autorelease_pool(void * p) {
    ((void(*)(id,SEL))objc_msgSend)((id)p, sel_registerName("drain"));
}
#define ARP_BEGIN void * _pool = new_autorelease_pool();
#define ARP_END   drain_autorelease_pool(_pool);
#else
#define ARP_BEGIN
#define ARP_END
#endif

#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>

// ============================================================
// Internal struct
// ============================================================
struct Qwen3Tts {
    qwen3_tts::Qwen3TTS engine;
    std::string last_error;
    std::string model_type_buf;
    std::string model_size_buf;
    // Default thread count stored at create time; applied as param default
    int32_t default_n_threads = 0;
    // Progress callback
    Qwen3TtsProgressFn progress_fn   = nullptr;
    void *             progress_data = nullptr;
    // Per-frame logits callback
    Qwen3TtsLogitsFn   logits_fn     = nullptr;
    void *             logits_data   = nullptr;
    // Streaming audio chunk callback
    Qwen3TtsAudioChunkFn chunk_fn    = nullptr;
    void *               chunk_data  = nullptr;
};

// ============================================================
// Helpers
// ============================================================
static qwen3_tts::tts_params to_cpp_params(const Qwen3TtsParams * p,
                                            int32_t default_n_threads = 0) {
    qwen3_tts::tts_params out;
    if (!p) return out;
    out.max_audio_tokens      = p->max_audio_tokens;
    out.temperature           = p->temperature;
    out.top_p                 = p->top_p;
    out.top_k                 = p->top_k;
    // Thread resolution order: params field > handle default > hardware auto-detect
    if (p->n_threads > 0)
        out.n_threads = p->n_threads;
    else if (default_n_threads > 0)
        out.n_threads = default_n_threads;
    else
        out.n_threads = (int32_t)std::min(8u, std::thread::hardware_concurrency());
    out.repetition_penalty    = p->repetition_penalty;
    out.subtalker_temperature = p->subtalker_temperature;
    out.subtalker_top_k       = p->subtalker_top_k;
    out.subtalker_top_p       = p->subtalker_top_p;
    out.icl_mode              = (p->icl_mode != 0);
    out.non_streaming_mode    = (p->non_streaming_mode != 0);
    out.print_timing          = (p->print_timing != 0);
    out.print_progress        = (p->print_progress != 0);
    // Extended sampling params
    out.min_p                 = p->min_p;
    out.frequency_penalty     = p->frequency_penalty;
    out.presence_penalty      = p->presence_penalty;
    out.dry_multiplier        = p->dry_multiplier;
    out.dry_base              = p->dry_base;
    out.dry_allowed_length    = p->dry_allowed_length;
    out.dry_penalty_last_n    = p->dry_penalty_last_n;
    out.dyntemp_range         = p->dyntemp_range;
    out.dyntemp_exponent      = p->dyntemp_exponent;
    if (p->language_name[0] != '\0') out.language = p->language_name;
    else if (p->language_id > 0)     out.language = std::to_string(p->language_id);
    if (p->speaker[0]  != '\0') out.speaker  = p->speaker;
    if (p->instruct[0] != '\0') out.instruct = p->instruct;
    if (p->ref_text[0] != '\0') out.ref_text = p->ref_text;
    return out;
}

// Copy float audio from tts_result into a heap-allocated Qwen3TtsAudio
static Qwen3TtsAudio make_audio_value(const qwen3_tts::tts_result & r) {
    Qwen3TtsAudio a = {};
    if (r.success && !r.audio.empty()) {
        auto * buf = new float[r.audio.size()];
        std::memcpy(buf, r.audio.data(), r.audio.size() * sizeof(float));
        a.samples     = buf;
        a.n_samples   = (int32_t)r.audio.size();
        a.sample_rate = r.sample_rate;
    }
    return a;
}

static Qwen3TtsResult * make_result(const qwen3_tts::tts_result & r) {
    auto * out       = new Qwen3TtsResult{};
    out->success     = r.success ? 1 : 0;
    out->audio       = make_audio_value(r);
    out->timing.t_tokenize_ms = r.t_tokenize_ms;
    out->timing.t_encode_ms   = r.t_encode_ms;
    out->timing.t_generate_ms = r.t_generate_ms;
    out->timing.t_decode_ms   = r.t_decode_ms;
    out->timing.t_total_ms    = r.t_total_ms;
    out->memory.rss_start     = r.mem_rss_start_bytes;
    out->memory.rss_end       = r.mem_rss_end_bytes;
    out->memory.rss_peak      = r.mem_rss_peak_bytes;
    out->memory.phys_start    = r.mem_phys_start_bytes;
    out->memory.phys_end      = r.mem_phys_end_bytes;
    out->memory.phys_peak     = r.mem_phys_peak_bytes;
    if (!r.success) {
        std::strncpy(out->error_msg, r.error_msg.c_str(), sizeof(out->error_msg) - 1);
    }
    return out;
}

// ============================================================
extern "C" {
// ============================================================

// ---- params -----------------------------------------------------------------

void qwen3_tts_default_params(Qwen3TtsParams * p) {
    if (!p) return;
    std::memset(p, 0, sizeof(*p));
    p->max_audio_tokens      = 4096;
    p->temperature           = 0.9f;
    p->top_p                 = 1.0f;
    p->top_k                 = 50;
    p->n_threads             = 0;   // 0 = auto
    p->repetition_penalty    = 1.05f;
    p->language_id           = 0;   // deprecated; language_name="auto" takes precedence
    p->subtalker_temperature = -1.0f;
    p->subtalker_top_k       = -1;
    p->subtalker_top_p       = -1.0f;
    p->icl_mode              = 0;
    p->non_streaming_mode    = 0;
    p->print_timing          = 0;
    p->print_progress        = 0;
    // Extended sampling — all disabled by default
    p->min_p                 = 0.0f;
    p->frequency_penalty     = 0.0f;
    p->presence_penalty      = 0.0f;
    p->dry_multiplier        = 0.0f;
    p->dry_base              = 1.75f;
    p->dry_allowed_length    = 2;
    p->dry_penalty_last_n    = -1;
    p->dyntemp_range         = 0.0f;
    p->dyntemp_exponent      = 1.0f;
    std::strncpy(p->language_name, "auto", sizeof(p->language_name) - 1);
}

// Thread-local storage for the last qwen3_tts_create() error message.
// Since the handle is deleted on failure, callers cannot retrieve the error
// via qwen3_tts_get_error(). Use qwen3_tts_get_last_create_error() instead.
static thread_local std::string s_last_create_error;

// ---- lifecycle --------------------------------------------------------------

Qwen3Tts * qwen3_tts_create(const char * model_dir, int32_t n_threads) {
    if (!model_dir) {
        s_last_create_error = "model_dir is NULL";
        return nullptr;
    }
    auto * tts = new Qwen3Tts;
    // Store the requested thread count so all subsequent synthesize calls use it
    // unless the caller overrides via params.n_threads.
    tts->default_n_threads = (n_threads > 0) ? n_threads : 0;
    if (!tts->engine.load_models(model_dir)) {
        s_last_create_error = tts->engine.get_error();
        delete tts;
        return nullptr;
    }
    s_last_create_error.clear();
    return tts;
}

Qwen3Tts * qwen3_tts_create_with_config(const char * model_dir,
                                          const char * gen_config_path,
                                          int32_t n_threads) {
    Qwen3Tts * tts = qwen3_tts_create(model_dir, n_threads);
    if (!tts) return nullptr;
    if (gen_config_path && gen_config_path[0] != '\0') {
        tts->engine.load_generation_config(gen_config_path);
    }
    return tts;
}

int qwen3_tts_is_loaded(const Qwen3Tts * tts) {
    return (tts && tts->engine.is_loaded()) ? 1 : 0;
}

void qwen3_tts_unload(Qwen3Tts * tts) {
    if (!tts) return;
    tts->engine.unload_models();
    tts->last_error.clear();
}

void qwen3_tts_destroy(Qwen3Tts * tts) { delete tts; }

const char * qwen3_tts_get_error(const Qwen3Tts * tts) {
    return tts ? tts->last_error.c_str() : "";
}

const char * qwen3_tts_get_last_create_error(void) {
    return s_last_create_error.c_str();
}

int32_t qwen3_tts_sample_rate(const Qwen3Tts * /*tts*/) { return 24000; }

int32_t qwen3_tts_embedding_size(const Qwen3Tts * tts) {
    if (!tts || !tts->engine.is_loaded()) return 0;
    // Query the loaded transformer's required speaker-row width; the encoder
    // is lazy-loaded, and a different model size may require a different width.
    int32_t dim = tts->engine.get_embedding_dim();
    return (dim > 0) ? dim : 1024;
}

// ---- progress callback ------------------------------------------------------

void qwen3_tts_set_progress_callback(Qwen3Tts * tts,
                                      Qwen3TtsProgressFn fn,
                                      void * userdata) {
    if (!tts) return;
    tts->progress_fn   = fn;
    tts->progress_data = userdata;
    if (fn) {
        // Wire through to C++ progress callback.
        // Return value from fn: non-zero = request early stop.
        // We honour this by calling set_abort_callback on the engine if stop is requested.
        tts->engine.set_progress_callback([tts](int done, int total) -> int {
            if (tts->progress_fn) {
                return tts->progress_fn((int32_t)done, (int32_t)total,
                                         tts->progress_data);
            }
            return 0;
        });
    } else {
        tts->engine.set_progress_callback(nullptr);
    }
}

void qwen3_tts_clear_progress_callback(Qwen3Tts * tts) {
    qwen3_tts_set_progress_callback(tts, nullptr, nullptr);
}

// ---- abort callback ---------------------------------------------------------

void qwen3_tts_set_abort_callback(Qwen3Tts * tts,
                                   ggml_abort_callback fn,
                                   void * userdata) {
    if (!tts) return;
    tts->engine.set_abort_callback(fn, userdata);
}

void qwen3_tts_clear_abort_callback(Qwen3Tts * tts) {
    if (!tts) return;
    tts->engine.clear_abort_callback();
}

// ---- introspection ----------------------------------------------------------

const char * qwen3_tts_model_type(const Qwen3Tts * tts) {
    if (!tts) return "";
    const_cast<Qwen3Tts *>(tts)->model_type_buf = tts->engine.get_model_type();
    return tts->model_type_buf.c_str();
}

const char * qwen3_tts_model_size(const Qwen3Tts * tts) {
    if (!tts) return "";
    const_cast<Qwen3Tts *>(tts)->model_size_buf = tts->engine.get_model_size();
    return tts->model_size_buf.c_str();
}

int qwen3_tts_list_speakers(const Qwen3Tts * tts, char * buf, int32_t buf_size) {
    if (!tts || !buf || buf_size <= 0) return -1;
    auto spk = tts->engine.get_supported_speakers();
    std::string out;
    for (const auto & s : spk) { out += s; out += '\n'; }
    std::strncpy(buf, out.c_str(), (size_t)(buf_size - 1));
    buf[buf_size - 1] = '\0';
    return (int)spk.size();
}

int qwen3_tts_list_languages(const Qwen3Tts * tts, char * buf, int32_t buf_size) {
    if (!tts || !buf || buf_size <= 0) return -1;
    auto langs = tts->engine.get_supported_languages();
    std::string out;
    for (const auto & l : langs) { out += l; out += '\n'; }
    std::strncpy(buf, out.c_str(), (size_t)(buf_size - 1));
    buf[buf_size - 1] = '\0';
    return (int)langs.size();
}

int32_t qwen3_tts_resolve_language(const Qwen3Tts * tts, const char * name) {
    if (!tts || !name) return -1;
    return tts->engine.resolve_language_id(name);
}

int32_t qwen3_tts_resolve_speaker(const Qwen3Tts * tts, const char * speaker_name) {
    if (!tts || !speaker_name) return -1;
    return tts->engine.resolve_speaker_id(speaker_name);
}

// ---- batch synthesis --------------------------------------------------------

// Forward declaration — defined after the simple/extended synthesis functions.
static Qwen3TtsResult ** batch_impl(
        Qwen3Tts * tts,
        const char ** texts, int32_t n_texts,
        const float * embedding, int32_t embedding_size,
        const Qwen3TtsParams * params,
        const char ** instruct_texts);

Qwen3TtsResult ** qwen3_tts_synthesize_batch(
        Qwen3Tts * tts, const char ** texts, int32_t n_texts,
        const float * embedding, int32_t embedding_size,
        const Qwen3TtsParams * params) {
    return batch_impl(tts, texts, n_texts, embedding, embedding_size, params, nullptr);
}

// ---- simple synthesis -------------------------------------------------------

Qwen3TtsAudio * qwen3_tts_synthesize(
        Qwen3Tts * tts, const char * text, const Qwen3TtsParams * params) {
    if (!tts || !text) return nullptr;
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.synthesize(text, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    Qwen3TtsAudio * out = nullptr;
    if (res.success && !res.audio.empty()) {
        out = new Qwen3TtsAudio;
        *out = make_audio_value(res);
    }
    ARP_END
    return out;
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_file(
        Qwen3Tts * tts, const char * text, const char * ref_path,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !ref_path) return nullptr;
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.synthesize_with_voice(text, ref_path, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    Qwen3TtsAudio * out = nullptr;
    if (res.success && !res.audio.empty()) { out = new Qwen3TtsAudio; *out = make_audio_value(res); }
    ARP_END
    return out;
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_samples(
        Qwen3Tts * tts, const char * text,
        const float * ref, int32_t n_ref,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !ref || n_ref <= 0) return nullptr;
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.synthesize_with_voice(text, ref, n_ref, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    Qwen3TtsAudio * out = nullptr;
    if (res.success && !res.audio.empty()) { out = new Qwen3TtsAudio; *out = make_audio_value(res); }
    ARP_END
    return out;
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_embedding(
        Qwen3Tts * tts, const char * text,
        const float * emb, int32_t emb_size,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !emb || emb_size <= 0) return nullptr;
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.synthesize_with_embedding(text, emb, emb_size, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    Qwen3TtsAudio * out = nullptr;
    if (res.success && !res.audio.empty()) { out = new Qwen3TtsAudio; *out = make_audio_value(res); }
    ARP_END
    return out;
}

void qwen3_tts_free_audio(Qwen3TtsAudio * a) {
    if (!a) return;
    delete[] a->samples;
    delete a;
}

// ---- extended synthesis (with timing + memory) ------------------------------

Qwen3TtsResult * qwen3_tts_synthesize_ex(
        Qwen3Tts * tts, const char * text, const Qwen3TtsParams * params) {
    if (!tts || !text) {
        auto * r = new Qwen3TtsResult{};
        std::strncpy(r->error_msg, "null handle or text", sizeof(r->error_msg)-1);
        return r;
    }
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.synthesize(text, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    auto * out = make_result(res);
    ARP_END
    return out;
}

Qwen3TtsResult * qwen3_tts_synthesize_with_voice_file_ex(
        Qwen3Tts * tts, const char * text, const char * ref_path,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !ref_path) {
        auto * r = new Qwen3TtsResult{}; std::strncpy(r->error_msg,"null arg",sizeof(r->error_msg)-1); return r;
    }
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.synthesize_with_voice(text, ref_path, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    auto * out = make_result(res);
    ARP_END
    return out;
}

Qwen3TtsResult * qwen3_tts_synthesize_with_voice_samples_ex(
        Qwen3Tts * tts, const char * text,
        const float * ref, int32_t n_ref,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !ref || n_ref <= 0) {
        auto * r = new Qwen3TtsResult{}; std::strncpy(r->error_msg,"null arg",sizeof(r->error_msg)-1); return r;
    }
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.synthesize_with_voice(text, ref, n_ref, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    auto * out = make_result(res);
    ARP_END
    return out;
}

Qwen3TtsResult * qwen3_tts_synthesize_with_embedding_ex(
        Qwen3Tts * tts, const char * text,
        const float * emb, int32_t emb_size,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !emb || emb_size <= 0) {
        auto * r = new Qwen3TtsResult{}; std::strncpy(r->error_msg,"null arg",sizeof(r->error_msg)-1); return r;
    }
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.synthesize_with_embedding(text, emb, emb_size, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    auto * out = make_result(res);
    ARP_END
    return out;
}

void qwen3_tts_free_result(Qwen3TtsResult * r) {
    if (!r) return;
    delete[] r->audio.samples;
    delete r;
}

// ---- speaker embedding ------------------------------------------------------

int32_t qwen3_tts_extract_embedding_file(
        Qwen3Tts * tts, const char * ref_path,
        float * out, int32_t max_size) {
    if (!tts || !ref_path || !out || max_size <= 0) return -1;
    ARP_BEGIN
    std::vector<float> emb;
    bool ok = tts->engine.extract_speaker_embedding(ref_path, emb);
    if (!ok) { tts->last_error = tts->engine.get_error(); ARP_END return -1; }
    int32_t n = (int32_t)std::min((size_t)max_size, emb.size());
    std::memcpy(out, emb.data(), n * sizeof(float));
    ARP_END
    return n;
}

int32_t qwen3_tts_extract_embedding_samples(
        Qwen3Tts * tts, const float * ref, int32_t n_ref,
        float * out, int32_t max_size) {
    if (!tts || !ref || n_ref <= 0 || !out || max_size <= 0) return -1;
    ARP_BEGIN
    std::vector<float> emb;
    qwen3_tts::tts_params p;
    bool ok = tts->engine.extract_speaker_embedding(ref, n_ref, emb, p);
    if (!ok) { tts->last_error = tts->engine.get_error(); ARP_END return -1; }
    int32_t n = (int32_t)std::min((size_t)max_size, emb.size());
    std::memcpy(out, emb.data(), n * sizeof(float));
    ARP_END
    return n;
}

int qwen3_tts_save_embedding(const char * path, const float * emb, int32_t size) {
    if (!path || !emb || size <= 0) return 0;
    std::vector<float> v(emb, emb + size);
    return qwen3_tts::save_speaker_embedding(path, v) ? 1 : 0;
}

int qwen3_tts_load_embedding(const char * path, float * out, int32_t max_size,
                              int32_t * size_out) {
    if (!path || !out || max_size <= 0) return 0;
    std::vector<float> v;
    if (!qwen3_tts::load_speaker_embedding(path, v)) return 0;
    int32_t n = (int32_t)std::min((size_t)max_size, v.size());
    std::memcpy(out, v.data(), n * sizeof(float));
    if (size_out) *size_out = n;
    return 1;
}

// ---- audio I/O --------------------------------------------------------------

int qwen3_tts_save_wav(const char * path,
                       const float * samples, int32_t n_samples,
                       int32_t sample_rate) {
    if (!path || !samples || n_samples <= 0 || sample_rate <= 0) return 0;
    std::vector<float> v(samples, samples + n_samples);
    return qwen3_tts::save_audio_file(path, v, sample_rate) ? 1 : 0;
}

int32_t qwen3_tts_load_wav(const char * path,
                            float * samples_out, int32_t max_samples,
                            int32_t * sample_rate_out) {
    if (!path) return -1;
    std::vector<float> samples;
    int sr = 0;
    if (!qwen3_tts::load_audio_file(path, samples, sr)) return -1;
    if (sample_rate_out) *sample_rate_out = sr;
    if (!samples_out) return (int32_t)samples.size();  // query size only
    int32_t n = (int32_t)std::min((size_t)max_samples, samples.size());
    std::memcpy(samples_out, samples.data(), n * sizeof(float));
    return n;
}

// ---- config -----------------------------------------------------------------

int qwen3_tts_load_generation_config(Qwen3Tts * tts, const char * json_path) {
    if (!tts || !json_path) return 0;
    return tts->engine.load_generation_config(json_path) ? 1 : 0;
}

// ---- resampling utility -------------------------------------------------------

int32_t qwen3_tts_resample(const float * input, int32_t n_input,
                             int32_t input_rate, int32_t output_rate,
                             float * output_out, int32_t max_output) {
    if (!input || n_input <= 0 || input_rate <= 0 || output_rate <= 0) return -1;
    if (input_rate == output_rate) {
        // No-op: same rate
        if (!output_out) return n_input;
        int32_t n = (n_input < max_output) ? n_input : max_output;
        std::memcpy(output_out, input, n * sizeof(float));
        return n;
    }
    std::vector<float> resampled;
    qwen3_tts::resample_audio(input, n_input, input_rate, output_rate, resampled);
    if (!output_out) return (int32_t)resampled.size();  // size-query mode
    int32_t n = (int32_t)std::min((size_t)max_output, resampled.size());
    std::memcpy(output_out, resampled.data(), n * sizeof(float));
    return n;
}

// ---- playback utility ---------------------------------------------------------

int32_t qwen3_tts_play(const float * input, int32_t n_samples,
                       int32_t sample_rate) {
    if (!input || n_samples <= 0 || sample_rate <= 0) return 0;
    return qwen3_tts::play_audio_samples(input, n_samples, sample_rate) ? 1 : 0;
}

// ---- logits callback --------------------------------------------------------

void qwen3_tts_set_logits_callback(Qwen3Tts * tts, Qwen3TtsLogitsFn fn, void * userdata) {
    if (!tts) return;
    tts->logits_fn   = fn;
    tts->logits_data = userdata;
    if (fn) {
        tts->engine.set_logits_callback(
            [tts](int32_t frame, const float * lg, int32_t sz, int32_t tok) -> int {
                return tts->logits_fn
                    ? tts->logits_fn(frame, lg, sz, tok, tts->logits_data)
                    : 0;
            });
    } else {
        tts->engine.set_logits_callback(nullptr);
    }
}

void qwen3_tts_clear_logits_callback(Qwen3Tts * tts) {
    qwen3_tts_set_logits_callback(tts, nullptr, nullptr);
}

// ---- streaming audio chunk callback -----------------------------------------

void qwen3_tts_set_audio_chunk_callback(Qwen3Tts * tts,
                                         Qwen3TtsAudioChunkFn fn,
                                         void * userdata,
                                         int32_t chunk_frames) {
    if (!tts) return;
    tts->chunk_fn   = fn;
    tts->chunk_data = userdata;
    if (fn) {
        tts->engine.set_audio_chunk_callback(
            [tts](const float * s, int32_t n, int32_t sr, int last) -> int {
                return tts->chunk_fn
                    ? tts->chunk_fn(s, n, sr, last, tts->chunk_data)
                    : 0;
            }, chunk_frames);
    } else {
        tts->engine.set_audio_chunk_callback(nullptr);
    }
}

void qwen3_tts_clear_audio_chunk_callback(Qwen3Tts * tts) {
    qwen3_tts_set_audio_chunk_callback(tts, nullptr, nullptr, 0);
}

// ---- speech codes access ----------------------------------------------------

// Internal helper: run synthesize_codes and optionally copy into caller buffer
static int32_t run_synthesize_codes(Qwen3Tts * /*tts*/,
                                     std::vector<int32_t> & codes,
                                     int32_t & n_cb,
                                     int32_t * codes_out, int32_t max_frames,
                                     int32_t * n_codebooks_out) {
    int32_t n_frames = (int32_t)(codes.size() / (size_t)n_cb);
    if (n_codebooks_out) *n_codebooks_out = n_cb;
    if (!codes_out) return n_frames;   // size-query mode
    int32_t copy_frames = std::min(n_frames, max_frames);
    std::memcpy(codes_out, codes.data(),
                (size_t)copy_frames * (size_t)n_cb * sizeof(int32_t));
    return copy_frames;
}

int32_t qwen3_tts_synthesize_codes(
        Qwen3Tts * tts, const char * text, const Qwen3TtsParams * params,
        int32_t * codes_out, int32_t max_frames, int32_t * n_codebooks_out) {
    if (!tts || !text) return -1;
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    std::vector<int32_t> codes;
    int32_t n_cb = 0;
    int32_t n_frames = tts->engine.synthesize_codes(text, codes, n_cb, cpp);
    if (n_frames < 0) { tts->last_error = tts->engine.get_error(); ARP_END return -1; }
    int32_t ret = run_synthesize_codes(tts, codes, n_cb, codes_out, max_frames, n_codebooks_out);
    ARP_END
    return ret;
}

int32_t qwen3_tts_synthesize_codes_with_voice_file(
        Qwen3Tts * tts, const char * text, const char * ref_path,
        const Qwen3TtsParams * params,
        int32_t * codes_out, int32_t max_frames, int32_t * n_codebooks_out) {
    if (!tts || !text || !ref_path) return -1;
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    std::vector<int32_t> codes;
    int32_t n_cb = 0;
    int32_t n_frames = tts->engine.synthesize_codes_with_voice(text, ref_path, codes, n_cb, cpp);
    if (n_frames < 0) { tts->last_error = tts->engine.get_error(); ARP_END return -1; }
    int32_t ret = run_synthesize_codes(tts, codes, n_cb, codes_out, max_frames, n_codebooks_out);
    ARP_END
    return ret;
}

int32_t qwen3_tts_synthesize_codes_with_voice_samples(
        Qwen3Tts * tts, const char * text,
        const float * ref, int32_t n_ref,
        const Qwen3TtsParams * params,
        int32_t * codes_out, int32_t max_frames, int32_t * n_codebooks_out) {
    if (!tts || !text || !ref || n_ref <= 0) return -1;
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    // Extract speaker embedding from raw samples, then synthesize codes
    std::vector<float> emb;
    qwen3_tts::tts_params ep;
    ep.n_threads = cpp.n_threads;
    bool ok = tts->engine.extract_speaker_embedding(ref, n_ref, emb, ep);
    if (!ok) { tts->last_error = tts->engine.get_error(); ARP_END return -1; }
    std::vector<int32_t> codes;
    int32_t n_cb = 0;
    int32_t n_frames = tts->engine.synthesize_codes_with_embedding(
                            text, emb.data(), (int32_t)emb.size(), codes, n_cb, cpp);
    if (n_frames < 0) { tts->last_error = tts->engine.get_error(); ARP_END return -1; }
    int32_t ret = run_synthesize_codes(tts, codes, n_cb, codes_out, max_frames, n_codebooks_out);
    ARP_END
    return ret;
}

int32_t qwen3_tts_synthesize_codes_with_embedding(
        Qwen3Tts * tts, const char * text,
        const float * emb, int32_t emb_size,
        const Qwen3TtsParams * params,
        int32_t * codes_out, int32_t max_frames, int32_t * n_codebooks_out) {
    if (!tts || !text || !emb || emb_size <= 0) return -1;
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    std::vector<int32_t> codes;
    int32_t n_cb = 0;
    int32_t n_frames = tts->engine.synthesize_codes_with_embedding(
                            text, emb, emb_size, codes, n_cb, cpp);
    if (n_frames < 0) { tts->last_error = tts->engine.get_error(); ARP_END return -1; }
    int32_t ret = run_synthesize_codes(tts, codes, n_cb, codes_out, max_frames, n_codebooks_out);
    ARP_END
    return ret;
}

Qwen3TtsResult * qwen3_tts_decode_codes(
        Qwen3Tts * tts,
        const int32_t * codes, int32_t n_frames, int32_t n_codebooks,
        const Qwen3TtsParams * params) {
    if (!tts || !codes || n_frames <= 0 || n_codebooks <= 0) {
        auto * r = new Qwen3TtsResult{};
        std::strncpy(r->error_msg, "null or invalid args", sizeof(r->error_msg)-1);
        return r;
    }
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.decode_speech_codes(codes, n_frames, n_codebooks, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    auto * out = make_result(res);
    ARP_END
    return out;
}

// ---- mimi encoder availability ----------------------------------------------

int qwen3_tts_has_mimi_encoder(const Qwen3Tts * tts) {
    if (!tts || !tts->engine.is_loaded()) return 0;
    return tts->engine.has_mimi_encoder() ? 1 : 0;
}

// ---- batch synthesis (with per-entry instruct) ------------------------------

// Internal shared implementation for both batch variants
static Qwen3TtsResult ** batch_impl(
        Qwen3Tts * tts,
        const char ** texts, int32_t n_texts,
        const float * embedding, int32_t embedding_size,
        const Qwen3TtsParams * params,
        const char ** instruct_texts) {
    if (!tts || !texts || n_texts <= 0) return nullptr;
    ARP_BEGIN

    auto cpp = to_cpp_params(params, tts->default_n_threads);

    std::vector<std::string> text_vec;
    text_vec.reserve((size_t)n_texts);
    for (int32_t i = 0; i < n_texts; ++i)
        text_vec.push_back(texts[i] ? texts[i] : "");

    // Build per-entry instruct vector if provided
    std::vector<std::string> instruct_vec;
    const std::vector<std::string> * instruct_ptr = nullptr;
    if (instruct_texts) {
        instruct_vec.reserve((size_t)n_texts);
        for (int32_t i = 0; i < n_texts; ++i)
            instruct_vec.push_back(instruct_texts[i] ? instruct_texts[i] : "");
        instruct_ptr = &instruct_vec;
    }

    if (embedding && embedding_size != tts->engine.get_embedding_dim()) {
        tts->last_error = "Speaker embedding dimension mismatch";
        ARP_END
        return nullptr;
    }
    const float * emb_ptr = (embedding && embedding_size > 0) ? embedding : nullptr;
    auto results = tts->engine.synthesize_batch(text_vec, emb_ptr, cpp, instruct_ptr);

    if (results.empty()) {
        tts->last_error = tts->engine.get_error();
        ARP_END
        return nullptr;
    }

    Qwen3TtsResult ** out = (Qwen3TtsResult **)calloc((size_t)n_texts, sizeof(Qwen3TtsResult *));
    if (!out) { ARP_END return nullptr; }

    for (int32_t i = 0; i < n_texts; ++i) {
        if (i < (int32_t)results.size()) {
            out[i] = make_result(results[i]);
            if (!results[i].success)
                tts->last_error = results[i].error_msg;
        } else {
            out[i] = new Qwen3TtsResult{};
            std::strncpy(out[i]->error_msg, "batch index out of range",
                         sizeof(out[i]->error_msg) - 1);
        }
    }
    ARP_END
    return out;
}

Qwen3TtsResult ** qwen3_tts_synthesize_batch_ex(
        Qwen3Tts * tts,
        const char ** texts, int32_t n_texts,
        const float * embedding, int32_t embedding_size,
        const Qwen3TtsParams * params,
        const char ** instruct_texts) {
    return batch_impl(tts, texts, n_texts, embedding, embedding_size,
                      params, instruct_texts);
}

void qwen3_tts_free_batch_results(Qwen3TtsResult ** results, int32_t n_results) {
    if (!results) return;
    for (int32_t i = 0; i < n_results; ++i)
        qwen3_tts_free_result(results[i]);
    free(results);
}

} // extern "C"
