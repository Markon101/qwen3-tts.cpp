#include "qwen3_tts.h"
#include "gguf_loader.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cctype>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <sstream>

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

// Minimal JSON value extraction without pulling in a library
// Only used for generation_config.json parsing (flat key-value).
static bool json_get_float(const std::string & json, const std::string & key, float & out) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    char * end = nullptr;
    float v = strtof(json.c_str() + pos, &end);
    if (end == json.c_str() + pos) return false;
    out = v;
    return true;
}
static bool json_get_int(const std::string & json, const std::string & key, int32_t & out) {
    float v;
    if (!json_get_float(json, key, v)) return false;
    out = (int32_t)v;
    return true;
}
static bool json_get_bool(const std::string & json, const std::string & key, bool & out) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    if (json.compare(pos, 4, "true") == 0)  { out = true;  return true; }
    if (json.compare(pos, 5, "false") == 0) { out = false; return true; }
    return false;
}

namespace qwen3_tts {

// ============================================================
// Time / memory helpers
// ============================================================
static int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct process_memory_snapshot {
    uint64_t rss_bytes           = 0;
    uint64_t phys_footprint_bytes = 0;
};

static bool get_process_memory_snapshot(process_memory_snapshot & out) {
#ifdef __APPLE__
    mach_task_basic_info_data_t basic_info = {};
    mach_msg_type_number_t basic_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&basic_info), &basic_count) != KERN_SUCCESS)
        return false;
    out.rss_bytes = (uint64_t)basic_info.resident_size;
    task_vm_info_data_t vm_info = {};
    mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&vm_info), &vm_count) == KERN_SUCCESS)
        out.phys_footprint_bytes = (uint64_t)vm_info.phys_footprint;
    else
        out.phys_footprint_bytes = out.rss_bytes;
    return true;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc = {};
    pmc.cb = sizeof(pmc);
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
                              reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                              sizeof(pmc)))
        return false;
    out.rss_bytes            = (uint64_t)pmc.WorkingSetSize;
    out.phys_footprint_bytes = (uint64_t)pmc.PrivateUsage;
    return true;
#else
    uint64_t vm_rss_kb = 0, vm_hwm_kb = 0;
    FILE * fp = fopen("/proc/self/status", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, "VmRSS:", 6) == 0) {
                sscanf(line + 6, "%llu", (unsigned long long *)&vm_rss_kb);
            } else if (strncmp(line, "VmHWM:", 6) == 0) {
                sscanf(line + 6, "%llu", (unsigned long long *)&vm_hwm_kb);
            }
        }
        fclose(fp);
    }
    if (vm_rss_kb > 0 || vm_hwm_kb > 0) {
        out.rss_bytes            = vm_rss_kb * 1024ULL;
        out.phys_footprint_bytes = (vm_hwm_kb > 0 ? vm_hwm_kb : vm_rss_kb) * 1024ULL;
        return true;
    }
    struct rusage usage = {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return false;
    out.rss_bytes            = (uint64_t)usage.ru_maxrss * 1024ULL;
    out.phys_footprint_bytes = out.rss_bytes;
    return true;
#endif
}

static std::string format_bytes(uint64_t bytes) {
    static const char * units[] = { "B", "KB", "MB", "GB", "TB" };
    double val = (double)bytes;
    int unit = 0;
    while (val >= 1024.0 && unit < 4) { val /= 1024.0; ++unit; }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f %s", val, units[unit]);
    return std::string(buf);
}

static void resample_linear(const float * input, int input_len, int input_rate,
                             std::vector<float> & output, int output_rate) {
    if (input_rate == output_rate) {
        output.assign(input, input + input_len);
        return;
    }
    // Kaiser-windowed sinc resampler (quality comparable to librosa's default).
    // Window half-length M controls quality/speed trade-off; 32 is a good default.
    const int M = 32;
    const double cutoff = (input_rate < output_rate)
                          ? 0.5 * input_rate                  // upsample: keep all input freqs
                          : 0.5 * output_rate;                 // downsample: anti-alias at output Nyquist
    const double fc = cutoff / input_rate;                     // normalised cutoff (0..0.5)

    // Kaiser window parameter beta ≈ 8.0 (good stopband attenuation ~80 dB)
    const double beta = 8.0;
    // Compute I0(beta) for normalisation
    auto bessel_i0 = [](double x) {
        double sum = 1.0, term = 1.0;
        for (int k = 1; k <= 30; ++k) {
            double r = (x / 2.0) / k;
            term *= r * r;
            sum += term;
        }
        return sum;
    };
    double inv_i0beta = 1.0 / bessel_i0(beta);

    // Precompute sinc filter kernel at integer sample offsets -M..M
    // h[n] = w[n] * sinc(2*fc*(n))    n in [-M, M]
    const int kernel_len = 2 * M + 1;
    std::vector<double> h(kernel_len);
    for (int n = -M; n <= M; ++n) {
        int idx = n + M;
        double r = (double)n / M;
        // Kaiser window
        double w_val = bessel_i0(beta * std::sqrt(1.0 - r * r)) * inv_i0beta;
        // Sinc
        double arg = 2.0 * fc * n;
        double s_val = (n == 0) ? 2.0 * fc : std::sin(3.14159265358979323846 * arg) / (3.14159265358979323846 * n);
        h[idx] = w_val * s_val;
    }

    double ratio = (double)input_rate / output_rate;
    int output_len = (int)std::ceil((double)input_len / ratio);
    output.resize(output_len);

    for (int i = 0; i < output_len; ++i) {
        double src = i * ratio;
        int i_center = (int)std::round(src);
        double frac = src - i_center;

        double acc = 0.0;
        double wt  = 0.0;
        for (int n = -M; n <= M; ++n) {
            int src_idx = i_center + n;
            if (src_idx < 0 || src_idx >= input_len) continue;
            // Sub-sample offset: kernel evaluated at (n - frac) in units of input samples
            double t_val = (n - frac);
            // Interpolate h at fractional position using linear interp between adjacent integers
            double h_val;
            if (std::fabs(t_val) >= M) {
                h_val = 0.0;
            } else {
                int t0 = (int)std::floor(t_val) + M;
                int t1 = t0 + 1;
                double f = t_val - std::floor(t_val);
                h_val = (t0 >= 0 && t1 < kernel_len)
                        ? h[t0] * (1.0 - f) + h[t1] * f
                        : (t0 >= 0 && t0 < kernel_len ? h[t0] : 0.0);
            }
            acc += input[src_idx] * h_val;
            wt  += h_val;
        }
        // Normalise (avoids edge taper artefacts)
        output[i] = (std::fabs(wt) > 1e-10) ? (float)(acc / wt) : 0.0f;
    }
}

// ============================================================
// Qwen3TTS
// ============================================================
Qwen3TTS::Qwen3TTS()  = default;
Qwen3TTS::~Qwen3TTS() = default;

// --- load_models (with explicit tokenizer path) ----------------------------
bool Qwen3TTS::load_models(const std::string & model_dir,
                            const std::string & tokenizer_path) {
    // Pre-populate decoder_model_path_ so the main load_models() discovery
    // skips its auto-search and uses the provided path directly.
    decoder_model_path_ = tokenizer_path;
    bool ok = load_models(model_dir);
    if (!ok) decoder_model_path_.clear();  // reset on failure
    return ok;
}

// --- load_models -----------------------------------------------------------
bool Qwen3TTS::load_models(const std::string & model_dir) {
    int64_t t_start = get_time_ms();

    transformer_.unload_model();
    audio_decoder_.unload_model();
    transformer_loaded_ = false;
    decoder_loaded_     = false;
    encoder_loaded_     = false;

    // Reset paths — will be re-populated by discovery below.
    // (decoder_model_path_ may have been pre-set by the two-arg overload;
    //  in that case tokenizer_override will be true and discovery is skipped.)
    tts_model_path_.clear();
    // Note: do NOT clear decoder_model_path_ here — the two-arg overload pre-sets it.

    // ---- Resolve TTS and tokenizer paths --------------------------------
    // Accepts either a directory (auto-selects best GGUF) or a direct .gguf file path.
    // If decoder_model_path_ was pre-populated by the two-argument overload,
    // skip tokenizer discovery entirely and use it as-is.
    const bool tokenizer_override = !decoder_model_path_.empty();
    {
        auto ends_with_gguf = [](const std::string & s) {
            return s.size() >= 5 && s.compare(s.size() - 5, 5, ".gguf") == 0;
        };

        if (ends_with_gguf(model_dir) && [&]{ FILE*f=fopen(model_dir.c_str(),"rb"); if(f){fclose(f);return true;} return false; }()) {
            // Direct .gguf file path — use it directly, find tokenizer alongside it
            tts_model_path_ = model_dir;
            if (!tokenizer_override) {
            size_t slash = model_dir.find_last_of("/\\");
            const std::string dir = (slash == std::string::npos) ? "." : model_dir.substr(0, slash);
            const char * tok_cands[] = {
                // Preferred: f16 vocoder (source is BF16, f32 adds nothing)
                // Mimi priority: f32 > f16 > q8_0 (f32 = bit-exact ICL)
                "qwen3-tts-tokenizer-f16-f32.gguf",
                "qwen3-tts-tokenizer-f16-f16.gguf",
                "qwen3-tts-tokenizer-f16-q8_0.gguf",
                "qwen3-tts-tokenizer-f32-f32.gguf",
                "qwen3-tts-tokenizer-f32-f16.gguf",
                "qwen3-tts-tokenizer-f32-q8_0.gguf",
                // Legacy single-tag names (backward compat)
                "qwen3-tts-tokenizer-f16.gguf",
                "qwen3-tts-tokenizer-f32.gguf",
                "qwen3-tts-tokenizer-q8_0.gguf",
                "qwen3-tts-tokenizer.gguf",
                nullptr
            };
            for (int i = 0; tok_cands[i]; ++i) {
                std::string p = dir + "/" + tok_cands[i];
                FILE * ft = fopen(p.c_str(), "rb");
                if (ft) { fclose(ft); decoder_model_path_ = p; break; }
            }
            if (decoder_model_path_.empty()) {
                error_msg_ = "No tokenizer GGUF found alongside: " + model_dir;
                return false;
            }
            } // !tokenizer_override
        } else {
            // Directory — auto-select best available GGUF
            // Priority: q8_0 > f16, 1.7b > 0.6b
            const char * candidates[] = {
                "qwen3-tts-1.7b-q8_0.gguf",
                "qwen3-tts-1.7b-q6_k.gguf",
                "qwen3-tts-1.7b-q5_k.gguf",
                "qwen3-tts-1.7b-q4_k.gguf",
                "qwen3-tts-1.7b-q3_k.gguf",
                "qwen3-tts-1.7b-q2_k.gguf",
                "qwen3-tts-1.7b-f16.gguf",
                "qwen3-tts-0.6b-q8_0.gguf",
                "qwen3-tts-0.6b-q6_k.gguf",
                "qwen3-tts-0.6b-q5_k.gguf",
                "qwen3-tts-0.6b-q4_k.gguf",
                "qwen3-tts-0.6b-q3_k.gguf",
                "qwen3-tts-0.6b-q2_k.gguf",
                "qwen3-tts-0.6b-f16.gguf",
                "qwen3-tts-q8_0.gguf",
                "qwen3-tts-q6_k.gguf",
                "qwen3-tts-q5_k.gguf",
                "qwen3-tts-q4_k.gguf",
                "qwen3-tts-q3_k.gguf",
                "qwen3-tts-q2_k.gguf",
                "qwen3-tts-f16.gguf",
                nullptr
            };
            for (int i = 0; candidates[i]; ++i) {
                std::string p = model_dir + "/" + candidates[i];
                FILE * f = fopen(p.c_str(), "rb");
                if (f) { fclose(f); tts_model_path_ = p; break; }
            }
            if (tts_model_path_.empty()) {
                error_msg_ = "No TTS model GGUF found in: " + model_dir;
                return false;
            }

            if (!tokenizer_override) {
            const char * tok_candidates[] = {
                // Preferred: f16 vocoder, Mimi priority: f32 > f16 > q8_0
                "qwen3-tts-tokenizer-f16-f32.gguf",
                "qwen3-tts-tokenizer-f16-f16.gguf",
                "qwen3-tts-tokenizer-f16-q8_0.gguf",
                "qwen3-tts-tokenizer-f32-f32.gguf",
                "qwen3-tts-tokenizer-f32-f16.gguf",
                "qwen3-tts-tokenizer-f32-q8_0.gguf",
                // Legacy single-tag names (backward compat)
                "qwen3-tts-tokenizer-f16.gguf",
                "qwen3-tts-tokenizer-f32.gguf",
                "qwen3-tts-tokenizer-q8_0.gguf",
                "qwen3-tts-tokenizer.gguf",
                nullptr
            };
            for (int i = 0; tok_candidates[i]; ++i) {
                std::string p = model_dir + "/" + tok_candidates[i];
                FILE * f = fopen(p.c_str(), "rb");
                if (f) { fclose(f); decoder_model_path_ = p; break; }
            }
            if (decoder_model_path_.empty()) {
                error_msg_ = "No tokenizer/vocoder GGUF found in: " + model_dir;
                return false;
            }
            } // !tokenizer_override
        }
    }

    const char * low_mem_env = std::getenv("QWEN3_TTS_LOW_MEM");
    low_mem_mode_ = low_mem_env && low_mem_env[0] != '\0' && low_mem_env[0] != '0';
    if (low_mem_mode_)
        fprintf(stderr, "  Low-memory mode enabled\n");

    // ---- Load text tokenizer -------------------------------------------
    fprintf(stderr, "Loading TTS model: %s\n", tts_model_path_.c_str());
    {
        GGUFLoader loader;
        if (!loader.open(tts_model_path_)) {
            error_msg_ = "Failed to open TTS model: " + loader.get_error();
            return false;
        }
        if (!tokenizer_.load_from_gguf(loader.get_ctx())) {
            error_msg_ = "Failed to load text tokenizer: " + tokenizer_.get_error();
            return false;
        }
        speaker_embedding_dim_ = loader.get_u32("qwen3-tts.speaker_encoder.embedding_length", 1024);
        encoder_sample_rate_   = loader.get_u32("qwen3-tts.speaker_encoder.sample_rate", 24000);
        fprintf(stderr, "  Text tokenizer loaded: vocab_size=%d\n",
                tokenizer_.get_config().vocab_size);
    }

    // ---- Load TTS transformer ------------------------------------------
    if (!transformer_.load_model(tts_model_path_)) {
        error_msg_ = "Failed to load TTS transformer: " + transformer_.get_error();
        return false;
    }
    transformer_loaded_ = true;
    if (abort_cb_) {
        transformer_.set_abort_callback(abort_cb_, abort_cb_data_);
    }
    {
        const auto & cfg = transformer_.get_config();
        fprintf(stderr, "  Transformer loaded: hidden=%d layers=%d model_type='%s' model_size='%s'\n",
                cfg.hidden_size, cfg.n_layers,
                cfg.model_type.empty() ? "base" : cfg.model_type.c_str(),
                cfg.model_size.empty() ? "unknown" : cfg.model_size.c_str());
        if (!cfg.spk_id.empty()) {
            fprintf(stderr, "  Named speakers: %zu\n", cfg.spk_id.size());
        }
    }

    // ---- Load vocoder (unless low-mem) ---------------------------------
    if (!low_mem_mode_) {
        fprintf(stderr, "Loading vocoder: %s\n", decoder_model_path_.c_str());
        if (!audio_decoder_.load_model(decoder_model_path_)) {
            error_msg_ = "Failed to load vocoder: " + audio_decoder_.get_error();
            return false;
        }
        decoder_loaded_ = true;
        fprintf(stderr, "  Vocoder loaded: sample_rate=%d\n",
                audio_decoder_.get_config().sample_rate);
    }

    // ---- Try to load Mimi encoder from tokenizer GGUF (for ICL) --------
    // This is optional — if it fails we just log and continue.
    {
        fprintf(stderr, "Loading Mimi encoder from %s...\n", decoder_model_path_.c_str());
        if (mimi_encoder_.load_model(decoder_model_path_)) {
            mimi_encoder_loaded_ = true;
            fprintf(stderr, "  Mimi encoder loaded (ICL voice cloning enabled)\n");
        } else {
            fprintf(stderr, "  Mimi encoder not found in tokenizer GGUF: %s\n",
                    mimi_encoder_.get_error().c_str());
            fprintf(stderr, "  ICL voice cloning unavailable — re-convert tokenizer "
                    "with updated convert_tokenizer_to_gguf.py to enable it.\n");
        }
    }

    // ---- Attempt to load generation_config.json ------------------------
    {
        // For a direct GGUF path, look for the JSON alongside the file.
        // For a directory path, look inside the directory as before.
        size_t slash = tts_model_path_.find_last_of("/\\");
        const std::string search_dir = (slash == std::string::npos)
                                       ? "." : tts_model_path_.substr(0, slash);
        std::string gen_cfg_path = search_dir + "/generation_config.json";
        load_generation_config(gen_cfg_path); // silent failure is OK
    }

    models_loaded_ = true;
    fprintf(stderr, "Models loaded in %lld ms\n", (long long)(get_time_ms() - t_start));
    return true;
}

// --- load_generation_config ------------------------------------------------
bool Qwen3TTS::load_generation_config(const std::string & json_path) {
    FILE * f = fopen(json_path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return false; }
    if (sz > 1024 * 1024) {
        fclose(f);
        fprintf(stderr, "ERROR: generation_config.json too large (%ld bytes)\n", sz);
        return false;
    }
    std::string json(sz, '\0');
    if (fread(&json[0], 1, sz, f) != (size_t)sz) { fclose(f); return false; }
    fclose(f);

    float fv; int32_t iv; bool bv;
    if (json_get_bool(json, "do_sample",           bv)) gen_defaults_.do_sample          = bv;
    if (json_get_float(json, "temperature",         fv)) gen_defaults_.temperature        = fv;
    if (json_get_int(json,   "top_k",               iv)) gen_defaults_.top_k              = iv;
    if (json_get_float(json, "top_p",               fv)) gen_defaults_.top_p              = fv;
    if (json_get_float(json, "repetition_penalty",  fv)) gen_defaults_.repetition_penalty  = fv;
    if (json_get_float(json, "subtalker_temperature", fv)) gen_defaults_.subtalker_temp    = fv;
    if (json_get_int(json,   "subtalker_top_k",     iv)) gen_defaults_.subtalker_top_k    = iv;
    if (json_get_float(json, "subtalker_top_p",     fv)) gen_defaults_.subtalker_top_p    = fv;
    if (json_get_int(json,   "max_new_tokens",       iv)) gen_defaults_.max_new_tokens    = iv;
    gen_defaults_.loaded = true;
    fprintf(stderr, "  generation_config.json loaded from %s\n", json_path.c_str());
    return true;
}

// --- unload_models ---------------------------------------------------------
void Qwen3TTS::unload_models() {
    if (transformer_loaded_) {
        transformer_.unload_model();
        transformer_loaded_ = false;
    }
    if (decoder_loaded_) {
        audio_decoder_.unload_model();
        decoder_loaded_ = false;
    }
    if (encoder_loaded_) {
        audio_encoder_.unload_model();
        encoder_loaded_ = false;
    }
    if (mimi_encoder_loaded_) {
        mimi_encoder_.unload_model();
        mimi_encoder_loaded_ = false;
    }
    models_loaded_ = false;
}

// --- Model introspection ---------------------------------------------------
std::string Qwen3TTS::get_model_type() const {
    if (!transformer_loaded_) return "";
    const auto & t = transformer_.get_config().model_type;
    return t.empty() ? std::string(MODEL_TYPE_BASE) : t;
}

std::string Qwen3TTS::get_model_size() const {
    if (!transformer_loaded_) return "";
    return transformer_.get_config().model_size;
}

std::vector<std::string> Qwen3TTS::get_supported_speakers() const {
    std::vector<std::string> out;
    if (!transformer_loaded_) return out;
    for (const auto & kv : transformer_.get_config().spk_id)
        out.push_back(kv.first);
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<std::string> Qwen3TTS::get_supported_languages() const {
    std::vector<std::string> out;
    if (!transformer_loaded_) return out;
    out.push_back("auto");
    for (const auto & kv : transformer_.get_config().codec_language_id) {
        if (kv.first != "auto") out.push_back(kv.first);
    }
    std::sort(out.begin(), out.end());
    return out;
}

int32_t Qwen3TTS::resolve_language_id(const std::string & language_name) const {
    if (!transformer_loaded_) return -1;
    const auto & cfg = transformer_.get_config();

    std::string lower = language_name;
    for (char & c : lower) c = (char)tolower((unsigned char)c);

    // Exact match in table
    auto it = cfg.codec_language_id.find(lower);
    if (it != cfg.codec_language_id.end()) return it->second;

    // "auto" → -1 which triggers the nothink (no language token) path in build_prefill_graph
    if (lower == "auto") return -1;

    // Try raw integer string
    char * end = nullptr;
    long v = strtol(lower.c_str(), &end, 10);
    if (end && *end == '\0') return (int32_t)v;

    // Alias mapping (covers common variants)
    static const struct { const char * alias; const char * canonical; } aliases[] = {
        {"english","english"}, {"en","english"},
        {"chinese","chinese"}, {"zh","chinese"}, {"mandarin","chinese"},
        {"japanese","japanese"}, {"ja","japanese"},
        {"korean","korean"}, {"ko","korean"},
        {"russian","russian"}, {"ru","russian"},
        {"german","german"}, {"de","german"},
        {"french","french"}, {"fr","french"},
        {"spanish","spanish"}, {"es","spanish"},
        {"italian","italian"}, {"it","italian"},
        {"portuguese","portuguese"}, {"pt","portuguese"},
        {nullptr, nullptr}
    };
    for (int i = 0; aliases[i].alias; ++i) {
        if (lower == aliases[i].alias) {
            auto it2 = cfg.codec_language_id.find(aliases[i].canonical);
            if (it2 != cfg.codec_language_id.end()) return it2->second;
        }
    }

    // Hard-coded fallback table (matches Python reference)
    static const struct { const char * name; int32_t id; } fallback[] = {
        {"english",    2050}, {"chinese",    2055}, {"japanese",   2058},
        {"korean",     2064}, {"russian",    2069}, {"german",     2053},
        {"french",     2061}, {"spanish",    2054}, {"italian",    2070},
        {"portuguese", 2071}, {nullptr, 0}
    };
    for (int i = 0; fallback[i].name; ++i) {
        if (lower == fallback[i].name) return fallback[i].id;
    }

    return cfg.english_language_id; // named language not found, default to english
}

int32_t Qwen3TTS::resolve_speaker_id(const std::string & speaker_name) const {
    if (!transformer_loaded_) return -1;
    std::string lower = speaker_name;
    for (char & c : lower) c = (char)tolower((unsigned char)c);
    auto it = transformer_.get_config().spk_id.find(lower);
    if (it != transformer_.get_config().spk_id.end()) return it->second;
    return -1;
}

// Resolve dialect override: if the speaker has a dialect mapping AND
// the requested language is Chinese or Auto, return the dialect language id.
// Returns -2 to signal "no override" (caller keeps their language_id).
static int32_t resolve_dialect_language(const tts_transformer_config & cfg,
                                         const std::string & speaker_lower,
                                         int32_t requested_language_id) {
    if (speaker_lower.empty()) return -2;
    auto dit = cfg.spk_is_dialect.find(speaker_lower);
    if (dit == cfg.spk_is_dialect.end()) return -2;

    // Python only overrides when language is Chinese or Auto (None)
    bool is_chinese_or_auto = false;
    auto chinese_it = cfg.codec_language_id.find("chinese");
    if (requested_language_id < 0) {
        is_chinese_or_auto = true; // "auto" → -1
    } else if (chinese_it != cfg.codec_language_id.end() &&
               requested_language_id == chinese_it->second) {
        is_chinese_or_auto = true;
    }

    if (!is_chinese_or_auto) return -2;

    const std::string & dialect_lang = dit->second;
    if (dialect_lang == "false" || dialect_lang.empty()) return -2;

    auto lit = cfg.codec_language_id.find(dialect_lang);
    if (lit != cfg.codec_language_id.end()) return lit->second;
    return -2;
}

// Returns true if model_size matches the given pattern case-insensitively.
// Pattern "1.7b" matches "1.7b", "1.7B", "1B7", etc.
static bool is_model_size_match(const std::string & sz, const char * pattern) {
    std::string a = sz, b = pattern;
    for (char & c : a) c = (char)tolower((unsigned char)c);
    for (char & c : b) c = (char)tolower((unsigned char)c);
    // Remove dots from both for flexible matching
    a.erase(std::remove(a.begin(), a.end(), '.'), a.end());
    b.erase(std::remove(b.begin(), b.end(), '.'), b.end());
    return a == b;
}

int32_t Qwen3TTS::params_language_id(const tts_params & params) const {
    std::string lang = params.language.empty() ? "auto" : params.language;

    // The 1.7B Base model requires a language token to reliably sample EOS.
    // Without it, generation free-runs to max_tokens for short inputs.
    // When the user hasn't specified a language ("auto"), default to english
    // for 1.7B so it always gets a language token.
    if (lang == "auto") {
        const std::string & sz = transformer_.get_config().model_size;
        if (is_model_size_match(sz, "1.7b")) {
            lang = "english";
        }
    }

    return resolve_language_id(lang);
}

// --- ensure_encoder_loaded -------------------------------------------------
bool Qwen3TTS::ensure_encoder_loaded(const tts_params & params, tts_result * result) {
    if (encoder_loaded_) return true;
    int64_t t0 = get_time_ms();
    if (!audio_encoder_.load_model(tts_model_path_)) {
        std::string msg = "Failed to load speaker encoder: " + audio_encoder_.get_error();
        if (result) result->error_msg = msg; else error_msg_ = msg;
        return false;
    }
    encoder_loaded_ = true;
    if (params.print_timing)
        fprintf(stderr, "  Speaker encoder loaded in %lld ms\n",
                (long long)(get_time_ms() - t0));
    return true;
}

// --- decode_codes ----------------------------------------------------------
bool Qwen3TTS::decode_codes(const std::vector<int32_t> & codes,
                             tts_result & result,
                             const tts_params & params) {
    if (!decoder_loaded_) {
        int64_t t0 = get_time_ms();
        if (!audio_decoder_.load_model(decoder_model_path_)) {
            result.error_msg = "Failed to load vocoder: " + audio_decoder_.get_error();
            return false;
        }
        decoder_loaded_ = true;
        if (params.print_timing)
            fprintf(stderr, "  Vocoder loaded in %lld ms\n",
                    (long long)(get_time_ms() - t0));
    }

    const int n_cb = transformer_.get_config().n_codebooks;
    const int n_frames = (int)codes.size() / n_cb;
    if (n_frames == 0) {
        result.error_msg = "No speech codes generated";
        return false;
    }

    int64_t t0 = get_time_ms();
    const int32_t chunk_limit = 100;
    if (n_frames <= chunk_limit) {
        if (!audio_decoder_.decode(codes.data(), n_frames, result.audio)) {
            result.error_msg = "Vocoder decode failed: " + audio_decoder_.get_error();
            return false;
        }
    } else {
        result.audio.clear();
        for (int32_t base = 0; base < n_frames; base += chunk_limit) {
            int32_t this_chunk = std::min(chunk_limit, n_frames - base);
            std::vector<float> chunk_audio;
            if (!audio_decoder_.decode(codes.data() + (size_t)base * n_cb,
                                       this_chunk, chunk_audio)) {
                result.error_msg = "Vocoder decode failed: " + audio_decoder_.get_error();
                return false;
            }
            result.audio.insert(result.audio.end(), chunk_audio.begin(), chunk_audio.end());
        }
    }
    result.t_decode_ms = get_time_ms() - t0;

    if (low_mem_mode_) {
        audio_decoder_.unload_model();
        decoder_loaded_ = false;
    }
    return true;
}

// --- decode_codes_streaming ------------------------------------------------
// Decodes in fixed-size chunks, firing audio_chunk_callback_ for each.
// Falls back to single-shot decode when no callback is registered.
bool Qwen3TTS::decode_codes_streaming(const std::vector<int32_t> & codes,
                                       tts_result & result,
                                       const tts_params & params) {
    // No chunk callback → delegate to normal single-shot path
    if (!audio_chunk_callback_) {
        return decode_codes(codes, result, params);
    }

    // Ensure decoder is loaded
    if (!decoder_loaded_) {
        int64_t t0 = get_time_ms();
        if (!audio_decoder_.load_model(decoder_model_path_)) {
            result.error_msg = "Failed to load vocoder: " + audio_decoder_.get_error();
            return false;
        }
        decoder_loaded_ = true;
        if (params.print_timing)
            fprintf(stderr, "  Vocoder loaded in %lld ms\n",
                    (long long)(get_time_ms() - t0));
    }

    const int32_t n_cb     = transformer_.get_config().n_codebooks;
    const int32_t n_frames = (int32_t)(codes.size() / (size_t)n_cb);
    if (n_frames == 0) {
        result.error_msg = "No speech codes generated";
        return false;
    }

    const int32_t chunk = audio_chunk_frames_;  // frames per decode call
    const int32_t sr    = audio_decoder_.get_config().sample_rate;

    int64_t t_decode_total = 0;
    bool aborted = false;

    for (int32_t base = 0; base < n_frames && !aborted; base += chunk) {
        int32_t remaining   = n_frames - base;
        int32_t this_chunk  = (chunk < remaining) ? chunk : remaining;
        int     is_last     = (base + this_chunk >= n_frames);

        std::vector<float> chunk_audio;
        int64_t t0 = get_time_ms();
        if (!audio_decoder_.decode(codes.data() + (size_t)base * n_cb,
                                    this_chunk, chunk_audio)) {
            result.error_msg = "Vocoder decode failed: " + audio_decoder_.get_error();
            return false;
        }
        t_decode_total += get_time_ms() - t0;

        // Append to result.audio so the caller gets the full waveform
        // even when using streaming (result is still fully populated at the end)
        result.audio.insert(result.audio.end(), chunk_audio.begin(), chunk_audio.end());

        // Fire chunk callback
        int stop = audio_chunk_callback_(chunk_audio.data(), (int32_t)chunk_audio.size(),
                                          sr, is_last);
        if (stop) aborted = true;
    }

    result.t_decode_ms = t_decode_total;

    if (low_mem_mode_) {
        audio_decoder_.unload_model();
        decoder_loaded_ = false;
    }

    if (aborted && result.audio.empty()) {
        result.error_msg = "Streaming synthesis aborted by chunk callback";
        return false;
    }
    return true;
}

// --- synthesize_codes -------------------------------------------------------
int32_t Qwen3TTS::synthesize_codes(const std::string & text,
                                    std::vector<int32_t> & codes_out,
                                    int32_t & n_codebooks_out,
                                    const tts_params & params) {
    tts_result result;
    if (!models_loaded_) { error_msg_ = "Models not loaded"; return -1; }
    std::vector<int32_t> codes;
    // Use speaker_embedding_dim_, NOT hidden_size — they differ on 1.7B
    // (hidden_size=2048, embedding_dim=1024). Passing hidden_size would cause
    // build_prefill_graph to read 8KB past the end of the vector.
    const int32_t spk_dim = speaker_embedding_dim_;
    std::vector<float> zero_emb(spk_dim, 0.0f);
    synthesize_internal(text, zero_emb.data(), params, result, &codes);
    if (!result.success) { error_msg_ = result.error_msg; return -1; }
    n_codebooks_out = transformer_.get_config().n_codebooks;
    codes_out = std::move(codes);
    return (int32_t)(codes_out.size() / (size_t)n_codebooks_out);
}

int32_t Qwen3TTS::synthesize_codes_with_voice(const std::string & text,
                                               const std::string & reference_audio,
                                               std::vector<int32_t> & codes_out,
                                               int32_t & n_codebooks_out,
                                               const tts_params & params) {
    // Load + resample reference audio then extract speaker embedding
    std::vector<float> ref_samples;
    int ref_sr;
    if (!load_audio_file(reference_audio, ref_samples, ref_sr)) {
        error_msg_ = "Failed to load reference audio: " + reference_audio;
        return -1;
    }
    if (ref_sr != encoder_sample_rate_) {
        std::vector<float> resampled;
        resample_linear(ref_samples.data(), (int)ref_samples.size(), ref_sr, resampled, encoder_sample_rate_);
        ref_samples = std::move(resampled);
    }
    if (!ensure_encoder_loaded(params)) return -1;
    std::vector<float> emb;
    if ((int32_t)ref_samples.size() > 30 * encoder_sample_rate_)
        fprintf(stderr, "  [warn] Reference audio %.1fs > 30s -- encoder will be slow. 3-30s recommended.\n",
                ref_samples.size() / (float)encoder_sample_rate_);
    if (!audio_encoder_.encode(ref_samples.data(), (int32_t)ref_samples.size(), emb)) {
        error_msg_ = "Speaker encoding failed: " + audio_encoder_.get_error();
        return -1;
    }
    return synthesize_codes_with_embedding(text, emb.data(), (int32_t)emb.size(),
                                            codes_out, n_codebooks_out, params);
}

int32_t Qwen3TTS::synthesize_codes_with_embedding(const std::string & text,
                                                    const float * embedding,
                                                    int32_t embedding_size,
                                                    std::vector<int32_t> & codes_out,
                                                    int32_t & n_codebooks_out,
                                                    const tts_params & params) {
    tts_result result;
    if (!models_loaded_) { error_msg_ = "Models not loaded"; return -1; }
    if (!embedding || embedding_size <= 0) { error_msg_ = "Invalid embedding"; return -1; }
    std::vector<int32_t> codes;
    synthesize_internal(text, embedding, params, result, &codes);
    if (!result.success) { error_msg_ = result.error_msg; return -1; }
    n_codebooks_out = transformer_.get_config().n_codebooks;
    codes_out = std::move(codes);
    return (int32_t)(codes_out.size() / (size_t)n_codebooks_out);
}

// --- decode_speech_codes ---------------------------------------------------
tts_result Qwen3TTS::decode_speech_codes(const int32_t * codes,
                                          int32_t n_frames,
                                          int32_t n_codebooks,
                                          const tts_params & params) {
    tts_result result;
    if (!models_loaded_) { result.error_msg = "Models not loaded"; return result; }
    if (!codes || n_frames <= 0 || n_codebooks <= 0) {
        result.error_msg = "Invalid codes input";
        return result;
    }
    const int32_t expected_n_cb = transformer_.get_config().n_codebooks;
    if (n_codebooks != expected_n_cb) {
        result.error_msg = "n_codebooks mismatch: got " + std::to_string(n_codebooks)
                         + ", expected " + std::to_string(expected_n_cb);
        return result;
    }
    const std::vector<int32_t> codes_vec(codes, codes + (size_t)n_frames * n_codebooks);
    if (!decode_codes_streaming(codes_vec, result, params)) return result;
    result.sample_rate = audio_decoder_.get_config().sample_rate;
    result.success     = true;
    return result;
}
std::vector<tts_result> Qwen3TTS::synthesize_batch(
    const std::vector<std::string> & texts,
    const float * speaker_embedding,
    const tts_params & raw_params,
    const std::vector<std::string> * instruct_per_entry) {

    std::vector<tts_result> results;
    if (!models_loaded_) {
        tts_result r; r.error_msg = "Models not loaded"; results.push_back(r);
        return results;
    }
    if (texts.empty()) return results;

    // Merge generation_config defaults
    tts_params params = raw_params;
    if (gen_defaults_.loaded) {
        if (params.temperature        == tts_params{}.temperature)
            params.temperature        = gen_defaults_.temperature;
        if (params.top_k              == tts_params{}.top_k)
            params.top_k              = gen_defaults_.top_k;
        if (params.top_p              == tts_params{}.top_p)
            params.top_p              = gen_defaults_.top_p;
        if (params.repetition_penalty == tts_params{}.repetition_penalty)
            params.repetition_penalty = gen_defaults_.repetition_penalty;
        if (params.max_audio_tokens   == tts_params{}.max_audio_tokens
                && gen_defaults_.max_new_tokens != tts_params{}.max_audio_tokens)
            params.max_audio_tokens   = gen_defaults_.max_new_tokens;
        if (!gen_defaults_.do_sample && params.temperature == gen_defaults_.temperature)
            params.temperature = 0.0f;
    }

    if (params.n_threads > 0) {
        transformer_.set_n_threads(params.n_threads);
        audio_decoder_.set_n_threads(params.n_threads);
    }

    // Wire extended sampling params
    transformer_.set_extended_sampling(
        params.min_p, params.frequency_penalty, params.presence_penalty,
        params.dry_multiplier, params.dry_base, params.dry_allowed_length,
        params.dry_penalty_last_n, params.dyntemp_range, params.dyntemp_exponent);

    // Wire logits + progress callbacks (chained)
    if (progress_callback_) {
        tts_logits_callback_t prog_cb =
            [this, &params](int32_t frame, const float * lg, int32_t sz, int32_t tok) -> int {
                int stop = progress_callback_(frame + 1, params.max_audio_tokens);
                if (stop) return stop;
                if (logits_callback_) return logits_callback_(frame, lg, sz, tok);
                return 0;
            };
        transformer_.set_logits_callback(std::move(prog_cb));
    } else if (logits_callback_) {
        transformer_.set_logits_callback(
            [this](int32_t frame, const float * lg, int32_t sz, int32_t tok) -> int {
                return logits_callback_ ? logits_callback_(frame, lg, sz, tok) : 0;
            });
    } else {
        transformer_.clear_logits_callback();
    }

    const int32_t N = (int32_t)texts.size();

    // Tokenize all texts
    std::vector<std::vector<int32_t>> all_tokens(N);
    for (int32_t i = 0; i < N; ++i) {
        all_tokens[i] = tokenizer_.encode_for_tts(texts[i]);
        if (all_tokens[i].empty()) {
            tts_result r; r.error_msg = "Failed to tokenize: " + texts[i];
            results.push_back(r);
            return results;
        }
    }

    // Tokenize instruct texts (per-entry, optional)
    std::vector<std::vector<int32_t>> all_instruct(N);
    bool has_instruct = (instruct_per_entry != nullptr);
    if (has_instruct && instruct_per_entry->size() != (size_t)N) {
        tts_result r; r.error_msg = "instruct_per_entry size mismatch";
        results.push_back(r);
        return results;
    }
    for (int32_t i = 0; i < N && has_instruct; ++i) {
        if (!(*instruct_per_entry)[i].empty()) {
            all_instruct[i] = tokenizer_.encode_instruct((*instruct_per_entry)[i]);
        }
    }

    // Build per-batch arrays for transformer::generate_batch
    std::vector<const int32_t *> token_ptrs(N);
    std::vector<int32_t> token_counts(N);
    std::vector<const float *> spk_ptrs(N, speaker_embedding);
    std::vector<int32_t> lang_ids(N, 2050);
    std::vector<const int32_t *> instruct_ptrs(N, nullptr);
    std::vector<int32_t> instruct_counts(N, 0);
    int32_t default_lang = resolve_language_id(params.language.empty() ? "auto" : params.language);

    for (int32_t i = 0; i < N; ++i) {
        token_ptrs[i] = all_tokens[i].data();
        token_counts[i] = (int32_t)all_tokens[i].size();
        lang_ids[i] = default_lang;
        if (has_instruct && !all_instruct[i].empty()) {
            instruct_ptrs[i] = all_instruct[i].data();
            instruct_counts[i] = (int32_t)all_instruct[i].size();
        }
    }

    // Run batch generation
    std::vector<std::vector<int32_t>> batch_codes(N);
    if (!transformer_.generate_batch(
            token_ptrs.data(), token_counts.data(),
            spk_ptrs.data(), N,
            params.max_audio_tokens, batch_codes,
            lang_ids.data(),
            has_instruct ? instruct_ptrs.data() : nullptr,
            has_instruct ? instruct_counts.data() : nullptr,
            params.repetition_penalty,
            params.temperature, params.top_k, params.top_p,
            params.subtalker_temperature, params.subtalker_top_k,
            params.subtalker_top_p,
            params.non_streaming_mode)) {
        tts_result r; r.error_msg = "Batch generation failed: " + transformer_.get_error();
        results.push_back(r);
        return results;
    }

    // Decode each result
    results.resize(N);
    for (int32_t i = 0; i < N; ++i) {
        if (batch_codes[i].empty()) {
            results[i].error_msg = "No speech codes generated for entry " + std::to_string(i);
            results[i].success = false;
            continue;
        }
        if (!decode_codes_streaming(batch_codes[i], results[i], params)) {
            // error_msg already set inside decode_codes_streaming
            results[i].success = false;
            continue;
        }
        results[i].sample_rate = audio_decoder_.get_config().sample_rate;
        results[i].success = true;
    }

    return results;
}

tts_result Qwen3TTS::synthesize(const std::string & text, const tts_params & params) {
    tts_result result;
    if (!models_loaded_) { result.error_msg = "Models not loaded"; return result; }

    std::string model_type = get_model_type();

    // CustomVoice: named speaker (+ optional instruct)
    if (model_type == MODEL_TYPE_CUSTOM_VOICE) {
        if (!params.speaker.empty()) {
            int32_t spk_id = resolve_speaker_id(params.speaker);
            if (spk_id < 0) {
                result.error_msg = "Unknown speaker: " + params.speaker;
                return result;
            }
            // For CustomVoice the speaker ID is injected as the language_id slot
            // (matches Python: speakers dict maps name -> codec token)
            tts_params p2 = params;
            p2.language = std::to_string(spk_id);
            return synthesize_internal(text, nullptr, p2, result);
        }
    }

    // VoiceDesign or CustomVoice with instruct text
    if ((model_type == MODEL_TYPE_VOICE_DESIGN || model_type == MODEL_TYPE_CUSTOM_VOICE)
            && !params.instruct.empty()) {
        // 0.6b CustomVoice does not support instruct — matches Python's check
        // `if self.model.tts_model_size in "0b6": instruct = None`
        std::string sz = get_model_size();
        bool instruct_supported = !(model_type == MODEL_TYPE_CUSTOM_VOICE &&
                                    is_model_size_match(sz, "0.6b"));
        if (instruct_supported) {
            return synthesize_internal(text, nullptr, params, result);
        }
        // Fall through to normal synthesis without instruct
    }

    // Base model: voice clone paths
    if (!params.reference_audio.empty()) {
        return synthesize_with_voice(text, params.reference_audio, params);
    }

    // Default: no voice reference — use a zero speaker embedding.
    // Size must match the speaker encoder output dimension (always 1024),
    // NOT the transformer hidden size (which is 2048 on 1.7B).
    // Using hidden_size here would cause build_prefill_graph to read 8KB
    // past the end of the vector on 1.7B models.
    const int32_t spk_dim = speaker_embedding_dim_;
    std::vector<float> zero_emb(spk_dim, 0.0f);
    return synthesize_internal(text, zero_emb.data(), params, result);
}

// --- synthesize_with_voice (from file) ------------------------------------
tts_result Qwen3TTS::synthesize_with_voice(const std::string & text,
                                             const std::string & ref_audio_path,
                                             const tts_params  & params) {
    tts_result result;
    std::vector<float> ref_samples;
    int ref_sr;
    if (!load_audio_file(ref_audio_path, ref_samples, ref_sr)) {
        result.error_msg = "Failed to load reference audio: " + ref_audio_path;
        return result;
    }
    if (ref_sr != encoder_sample_rate_) {
        std::vector<float> resampled;
        resample_linear(ref_samples.data(), (int)ref_samples.size(), ref_sr, resampled, encoder_sample_rate_);
        ref_samples = std::move(resampled);
    }
    return synthesize_with_voice(text, ref_samples.data(), (int32_t)ref_samples.size(), params);
}

// --- synthesize_with_voice (from samples) ---------------------------------
tts_result Qwen3TTS::synthesize_with_voice(const std::string & text,
                                             const float * ref_samples, int32_t n_ref_samples,
                                             const tts_params & params) {
    tts_result result;
    if (!models_loaded_) { result.error_msg = "Models not loaded"; return result; }

    int64_t t_total_start = get_time_ms();

    if (!ensure_encoder_loaded(params, &result)) return result;

    // Extract speaker embedding
    int64_t t0 = get_time_ms();
    std::vector<float> speaker_embedding;
    if (n_ref_samples > 30 * encoder_sample_rate_)
        fprintf(stderr, "  [warn] Reference audio %.1fs > 30s -- encoder will be slow. 3-30s recommended.\n",
                n_ref_samples / (float)encoder_sample_rate_);
    if (!audio_encoder_.encode(ref_samples, n_ref_samples, speaker_embedding)) {
        result.error_msg = "Speaker encoding failed: " + audio_encoder_.get_error();
        return result;
    }
    result.t_encode_ms = get_time_ms() - t0;

    // ICL mode: encode reference audio to discrete codes using Mimi encoder
    if (params.icl_mode && !params.ref_text.empty()) {
        if (mimi_encoder_loaded_) {
            // Memory tracking (same as synthesize_internal)
            auto snap_mem = [&](const char * label) {
                process_memory_snapshot mem;
                if (!get_process_memory_snapshot(mem)) return;
                if (result.mem_rss_start_bytes == 0) {
                    result.mem_rss_start_bytes   = mem.rss_bytes;
                    result.mem_phys_start_bytes  = mem.phys_footprint_bytes;
                }
                result.mem_rss_end_bytes   = mem.rss_bytes;
                result.mem_phys_end_bytes  = mem.phys_footprint_bytes;
                if (mem.rss_bytes            > result.mem_rss_peak_bytes)
                    result.mem_rss_peak_bytes  = mem.rss_bytes;
                if (mem.phys_footprint_bytes > result.mem_phys_peak_bytes)
                    result.mem_phys_peak_bytes = mem.phys_footprint_bytes;
                if (params.print_timing)
                    fprintf(stderr, "  [mem] %-24s rss=%s phys=%s\n",
                            label, format_bytes(mem.rss_bytes).c_str(),
                            format_bytes(mem.phys_footprint_bytes).c_str());
            };

            snap_mem("synth/start");

            // Encode reference audio to codec codes
            std::vector<int32_t> ref_codes;
            int32_t n_ref_frames = 0;
            if (mimi_encoder_.encode(ref_samples, n_ref_samples, ref_codes, n_ref_frames) &&
                n_ref_frames > 0) {

                // Tokenize reference text — body only (strip role tokens)
                // Matches Python's ref_ids[3:-2] passed to generate_icl_prompt()
                std::vector<int32_t> ref_text_toks = tokenizer_.encode_ref_text_body(params.ref_text);

                std::vector<int32_t> text_tokens = tokenizer_.encode_for_tts(text);
                if (text_tokens.empty()) {
                    result.error_msg = "Failed to tokenize text";
                    return result;
                }

                snap_mem("synth/after-tokenize");

                int32_t language_id = params_language_id(params);

                // Dialect override
                if (!params.speaker.empty()) {
                    std::string spk_lower = params.speaker;
                    for (char & c : spk_lower) c = (char)tolower((unsigned char)c);
                    int32_t did = resolve_dialect_language(
                        transformer_.get_config(), spk_lower, language_id);
                    if (did != -2) language_id = did;
                }

                // Wire extended sampling params and logits callback
                // (same setup as synthesize_internal — the ICL path bypasses it)
                transformer_.set_extended_sampling(
                    params.min_p, params.frequency_penalty, params.presence_penalty,
                    params.dry_multiplier, params.dry_base, params.dry_allowed_length,
                    params.dry_penalty_last_n, params.dyntemp_range, params.dyntemp_exponent);
                if (progress_callback_) {
                    transformer_.set_logits_callback(
                        [this, &params](int32_t frame, const float * lg, int32_t sz, int32_t tok) -> int {
                            int stop = progress_callback_(frame + 1, params.max_audio_tokens);
                            if (stop) return stop;
                            if (logits_callback_) return logits_callback_(frame, lg, sz, tok);
                            return 0;
                        });
                } else if (logits_callback_) {
                    transformer_.set_logits_callback(
                        [this](int32_t frame, const float * lg, int32_t sz, int32_t tok) -> int {
                            return logits_callback_ ? logits_callback_(frame, lg, sz, tok) : 0;
                        });
                } else {
                    transformer_.clear_logits_callback();
                }

                transformer_.clear_kv_cache();
                std::vector<int32_t> speech_codes;

                int64_t t_gen = get_time_ms();
                bool gen_ok = transformer_.generate_icl(
                    text_tokens.data(), (int32_t)text_tokens.size(),
                    speaker_embedding.data(),
                    ref_text_toks.data(), (int32_t)ref_text_toks.size(),
                    ref_codes.data(), n_ref_frames,
                    params.max_audio_tokens, speech_codes,
                    language_id,
                    params.repetition_penalty,
                    params.temperature, params.top_k, params.top_p,
                    params.subtalker_temperature, params.subtalker_top_k,
                    params.non_streaming_mode,
                    params.subtalker_top_p);

                result.t_generate_ms = get_time_ms() - t_gen;
                snap_mem("synth/after-generate");

                if (!gen_ok) {
                    result.error_msg = "ICL generation failed: " + transformer_.get_error();
                    return result;
                }

                if (speech_codes.empty()) {
                    result.error_msg = "ICL: no speech codes generated";
                    return result;
                }

                // For ICL: prepend ref_codes to speech_codes, decode together,
                // then trim the reference portion from the output waveform
                const int32_t n_cb = transformer_.get_config().n_codebooks;
                std::vector<int32_t> full_codes;
                full_codes.reserve(ref_codes.size() + speech_codes.size());
                full_codes.insert(full_codes.end(), ref_codes.begin(), ref_codes.end());
                full_codes.insert(full_codes.end(), speech_codes.begin(), speech_codes.end());

                if (!decode_codes_streaming(full_codes, result, params)) return result;
                snap_mem("synth/after-decode");

                // Trim the reference portion using float ratio to avoid overflow
                if (n_ref_frames > 0 && !result.audio.empty()) {
                    int32_t total_frames = (int32_t)(full_codes.size() / n_cb);
                    if (total_frames > 0) {
                        double ratio = (double)n_ref_frames / (double)total_frames;
                        size_t cut = (size_t)(ratio * (double)result.audio.size());
                        if (cut > result.audio.size()) cut = result.audio.size();
                        // Use move-assign from subrange instead of erase(begin,begin+n)
                        // to avoid a large O(n) element shift in-place.
                        std::vector<float> trimmed(
                            std::make_move_iterator(result.audio.begin() + (ptrdiff_t)cut),
                            std::make_move_iterator(result.audio.end()));
                        result.audio = std::move(trimmed);
                    }
                }

                result.sample_rate = audio_decoder_.get_config().sample_rate;
                result.success     = true;
                result.t_total_ms  = get_time_ms() - t_total_start;
                snap_mem("synth/end");
                return result;
            }
            fprintf(stderr, "  [ICL] Mimi encode failed: %s — falling back to x-vector\n",
                    mimi_encoder_.get_error().c_str());
        } else {
            fprintf(stderr, "  [ICL] Mimi encoder not loaded — falling back to x-vector.\n"
                    "  Re-convert the tokenizer GGUF to enable full ICL.\n");
        }
    }

    return synthesize_internal(text, speaker_embedding.data(), params, result);
}

// --- synthesize_with_embedding ---------------------------------------------
tts_result Qwen3TTS::synthesize_with_embedding(const std::string & text,
                                                 const std::vector<float> & embedding,
                                                 const tts_params & params) {
    tts_result result;
    if (embedding.empty()) { result.error_msg = "Empty embedding"; return result; }
    return synthesize_with_embedding(text, embedding.data(), (int32_t)embedding.size(), params);
}

tts_result Qwen3TTS::synthesize_with_embedding(const std::string & text,
                                                 const float * embedding, int32_t embedding_size,
                                                 const tts_params & params) {
    tts_result result;
    if (!models_loaded_) { result.error_msg = "Models not loaded"; return result; }
    if (!embedding || embedding_size <= 0) { result.error_msg = "Invalid embedding"; return result; }
    return synthesize_internal(text, embedding, params, result);
}

// --- extract_speaker_embedding (file) -------------------------------------
bool Qwen3TTS::extract_speaker_embedding(const std::string & ref_audio_path,
                                          std::vector<float> & embedding) {
    std::vector<float> samples;
    int sr;
    if (!load_audio_file(ref_audio_path, samples, sr)) {
        error_msg_ = "Failed to load audio: " + ref_audio_path;
        return false;
    }
    if (sr != encoder_sample_rate_) {
        std::vector<float> resampled;
        resample_linear(samples.data(), (int)samples.size(), sr, resampled, encoder_sample_rate_);
        samples = std::move(resampled);
    }
    return extract_speaker_embedding(samples.data(), (int32_t)samples.size(), embedding);
}

// --- extract_speaker_embedding (samples) ----------------------------------
bool Qwen3TTS::extract_speaker_embedding(const float * ref_samples, int32_t n_ref_samples,
                                          std::vector<float> & embedding,
                                          const tts_params & params) {
    if (!models_loaded_) { error_msg_ = "Models not loaded"; return false; }
    if (!ensure_encoder_loaded(params)) return false;
    if (n_ref_samples > 30 * encoder_sample_rate_)
        fprintf(stderr, "  [warn] Reference audio %.1fs > 30s -- encoder will be slow. 3-30s recommended.\n",
                n_ref_samples / (float)encoder_sample_rate_);
    if (!audio_encoder_.encode(ref_samples, n_ref_samples, embedding)) {
        error_msg_ = "Speaker encoding failed: " + audio_encoder_.get_error();
        return false;
    }
    return true;
}

// --- save / load speaker embedding ----------------------------------------
bool Qwen3TTS::save_speaker_embedding(const std::string & path,
                                       const std::vector<float> & embedding) {
    return qwen3_tts::save_speaker_embedding(path, embedding);
}

bool Qwen3TTS::load_speaker_embedding(const std::string & path,
                                       std::vector<float> & embedding) {
    return qwen3_tts::load_speaker_embedding(path, embedding);
}

// --- set_progress_callback ------------------------------------------------
void Qwen3TTS::set_progress_callback(tts_progress_callback_t cb) {
    progress_callback_ = cb;
}

// --- set_abort_callback / clear_abort_callback ----------------------------
void Qwen3TTS::set_abort_callback(ggml_abort_callback fn, void * userdata) {
    abort_cb_      = fn;
    abort_cb_data_ = userdata;
    if (transformer_loaded_) transformer_.set_abort_callback(fn, userdata);
}
void Qwen3TTS::clear_abort_callback() {
    abort_cb_      = nullptr;
    abort_cb_data_ = nullptr;
    if (transformer_loaded_) transformer_.clear_abort_callback();
}

// --- set_logits_callback --------------------------------------------------
void Qwen3TTS::set_logits_callback(tts_logits_callback_t cb) {
    logits_callback_ = cb;
    // Wire through to transformer immediately if already loaded
    if (transformer_loaded_) {
        if (cb) {
            transformer_.set_logits_callback(
                [this](int32_t frame, const float * lg, int32_t sz, int32_t tok) -> int {
                    return logits_callback_ ? logits_callback_(frame, lg, sz, tok) : 0;
                });
        } else {
            transformer_.clear_logits_callback();
        }
    }
}

// --- set_audio_chunk_callback ---------------------------------------------
void Qwen3TTS::set_audio_chunk_callback(tts_audio_chunk_callback_t cb, int32_t chunk_frames) {
    audio_chunk_callback_ = cb;
    audio_chunk_frames_   = (chunk_frames > 0) ? chunk_frames : 12;
}

// --- get_embedding_dim -----------------------------------------------------
int32_t Qwen3TTS::get_embedding_dim() const {
    if (encoder_loaded_) {
        int32_t d = audio_encoder_.get_config().embedding_dim;
        return (d > 0) ? d : speaker_embedding_dim_;
    }
    return speaker_embedding_dim_;
}

// --- synthesize_internal ---------------------------------------------------
tts_result Qwen3TTS::synthesize_internal(const std::string & text,
                                          const float * speaker_embedding,
                                          const tts_params & raw_params,
                                          tts_result & result,
                                          std::vector<int32_t> * codes_out) {
    int64_t t_total = get_time_ms();

    // Merge generation_config defaults — only apply when user left params at
    // their default values (can't distinguish "set to default" vs "not set",
    // but this matches the Python wrapper's pick() behaviour: config.json wins
    // over hard defaults, user explicit call wins over config.json).
    // We use the gen_defaults_.loaded flag: if JSON was loaded, use ALL its
    // values as the baseline (i.e. replace defaults, keep user non-defaults).
    tts_params params = raw_params;
    if (gen_defaults_.loaded) {
        // Only override when the param still equals the struct default value,
        // meaning the caller didn't touch it.
        if (params.temperature        == tts_params{}.temperature)
            params.temperature        = gen_defaults_.temperature;
        if (params.top_k              == tts_params{}.top_k)
            params.top_k              = gen_defaults_.top_k;
        if (params.top_p              == tts_params{}.top_p)
            params.top_p              = gen_defaults_.top_p;
        if (params.repetition_penalty == tts_params{}.repetition_penalty)
            params.repetition_penalty = gen_defaults_.repetition_penalty;
        if (params.subtalker_temperature == tts_params{}.subtalker_temperature
                && gen_defaults_.subtalker_temp > 0.0f)
            params.subtalker_temperature = gen_defaults_.subtalker_temp;
        if (params.subtalker_top_k    == tts_params{}.subtalker_top_k
                && gen_defaults_.subtalker_top_k >= 0)
            params.subtalker_top_k    = gen_defaults_.subtalker_top_k;
        if (params.subtalker_top_p    == tts_params{}.subtalker_top_p
                && gen_defaults_.subtalker_top_p >= 0.0f)
            params.subtalker_top_p    = gen_defaults_.subtalker_top_p;
        if (params.max_audio_tokens   == tts_params{}.max_audio_tokens
                && gen_defaults_.max_new_tokens != tts_params{}.max_audio_tokens)
            params.max_audio_tokens   = gen_defaults_.max_new_tokens;
        // do_sample=false forces greedy decoding (temperature=0)
        if (!gen_defaults_.do_sample && params.temperature == gen_defaults_.temperature)
            params.temperature = 0.0f;
    }

    // ---- Memory tracking ------------------------------------------------
    auto snap_mem = [&](const char * label) {
        process_memory_snapshot mem;
        if (!get_process_memory_snapshot(mem)) return;
        if (result.mem_rss_start_bytes == 0) {
            result.mem_rss_start_bytes   = mem.rss_bytes;
            result.mem_phys_start_bytes  = mem.phys_footprint_bytes;
        }
        result.mem_rss_end_bytes   = mem.rss_bytes;
        result.mem_phys_end_bytes  = mem.phys_footprint_bytes;
        if (mem.rss_bytes            > result.mem_rss_peak_bytes)
            result.mem_rss_peak_bytes  = mem.rss_bytes;
        if (mem.phys_footprint_bytes > result.mem_phys_peak_bytes)
            result.mem_phys_peak_bytes = mem.phys_footprint_bytes;
        if (params.print_timing)
            fprintf(stderr, "  [mem] %-24s rss=%s phys=%s\n",
                    label, format_bytes(mem.rss_bytes).c_str(),
                    format_bytes(mem.phys_footprint_bytes).c_str());
    };
    snap_mem("synth/start");

    // ---- Tokenize -------------------------------------------------------
    int64_t t0 = get_time_ms();

    // Apply thread count (default 4 → use all available cores for speed)
    if (params.n_threads > 0) {
        transformer_.set_n_threads(params.n_threads);
        audio_decoder_.set_n_threads(params.n_threads);
    }

    // Wire extended sampling parameters into the transformer generate loop
    transformer_.set_extended_sampling(
        params.min_p,
        params.frequency_penalty,
        params.presence_penalty,
        params.dry_multiplier,
        params.dry_base,
        params.dry_allowed_length,
        params.dry_penalty_last_n,
        params.dyntemp_range,
        params.dyntemp_exponent);

    // Wire logits callback into the transformer for this synthesis call.
    // Chain: progress_callback fires first, then logits_callback.
    if (progress_callback_) {
        transformer_.set_logits_callback(
            [this, &params](int32_t frame, const float * lg, int32_t sz, int32_t tok) -> int {
                int stop = progress_callback_(frame + 1, params.max_audio_tokens);
                if (stop) return stop;
                if (logits_callback_) return logits_callback_(frame, lg, sz, tok);
                return 0;
            });
    } else if (logits_callback_) {
        transformer_.set_logits_callback(
            [this](int32_t frame, const float * lg, int32_t sz, int32_t tok) -> int {
                return logits_callback_ ? logits_callback_(frame, lg, sz, tok) : 0;
            });
    } else {
        transformer_.clear_logits_callback();
    }

    std::string model_type = get_model_type();
    int32_t     language_id = params_language_id(params);

    // Dialect override: if the speaker has a dialect mapping, override language_id
    if (!params.speaker.empty()) {
        std::string spk_lower = params.speaker;
        for (char & c : spk_lower) c = (char)tolower((unsigned char)c);
        int32_t dialect_id = resolve_dialect_language(
            transformer_.get_config(), spk_lower, language_id);
        if (dialect_id != -2) {
            language_id = dialect_id;
        }
    }

    // Build token sequences
    std::vector<int32_t> text_tokens    = tokenizer_.encode_for_tts(text);
    std::vector<int32_t> instruct_tokens;
    std::vector<int32_t> ref_text_tokens;

    if (!params.instruct.empty()) {
        instruct_tokens = tokenizer_.encode_instruct(params.instruct);
    }
    if (!params.ref_text.empty()) {
        ref_text_tokens = tokenizer_.encode_ref_text(params.ref_text);
    }

    result.t_tokenize_ms = get_time_ms() - t0;
    snap_mem("synth/after-tokenize");

    if (text_tokens.empty()) {
        result.error_msg = "Failed to tokenize text";
        return result;
    }
    if (params.print_progress)
        fprintf(stderr, "Text tokenized: %zu tokens\n", text_tokens.size());

    // ---- Reload transformer if unloaded (low-mem mode) ------------------
    if (!transformer_loaded_) {
        if (!transformer_.load_model(tts_model_path_)) {
            result.error_msg = "Failed to reload transformer: " + transformer_.get_error();
            return result;
        }
        transformer_loaded_ = true;
        if (abort_cb_) {
            transformer_.set_abort_callback(abort_cb_, abort_cb_data_);
        }
    }
    transformer_.clear_kv_cache();

    // ---- Handle named speaker injection ---------------------------------
    // For CustomVoice: inject speaker token as the language_id override.
    // The Python model encodes speaker_id as codec_language_id value.
    if (model_type == MODEL_TYPE_CUSTOM_VOICE && !params.speaker.empty()) {
        int32_t spk_id = resolve_speaker_id(params.speaker);
        if (spk_id >= 0) {
            language_id = spk_id;
        } else {
            fprintf(stderr, "  Warning: unknown speaker '%s', using language_id\n",
                    params.speaker.c_str());
        }
    }

    // ---- Generate speech codes ------------------------------------------
    t0 = get_time_ms();
    std::vector<int32_t> speech_codes;
    bool gen_ok = false;

    if (!instruct_tokens.empty()) {
        // VoiceDesign / CustomVoice with instruction:
        // Python's generate_voice_design() defaults non_streaming_mode=True.
        // Apply that default here if the caller left it at its struct default (false).
        bool use_non_streaming = params.non_streaming_mode;
        if (model_type == MODEL_TYPE_VOICE_DESIGN && !params.non_streaming_mode) {
            use_non_streaming = true;
        }

        transformer_.clear_kv_cache();
        std::vector<float> prefill_embd, trailing, tts_pad;
        if (transformer_.build_prefill_graph_instruct(
                    text_tokens.data(), (int32_t)text_tokens.size(),
                    speaker_embedding, language_id,
                    instruct_tokens.data(), (int32_t)instruct_tokens.size(),
                    prefill_embd, trailing, tts_pad,
                    use_non_streaming)) {
            gen_ok = transformer_.generate_from_prefill(
                prefill_embd, trailing, tts_pad,
                params.max_audio_tokens, speech_codes,
                params.repetition_penalty,
                params.temperature, params.top_k, params.top_p,
                params.subtalker_temperature, params.subtalker_top_k,
                params.subtalker_top_p);
        }
        if (!gen_ok) {
            result.error_msg = "Instruct code generation failed: " + transformer_.get_error();
            return result;
        }
    } else {
        // Standard base / voice-clone path
        gen_ok = transformer_.generate(
            text_tokens.data(), (int32_t)text_tokens.size(),
            speaker_embedding,
            params.max_audio_tokens, speech_codes,
            language_id,
            params.repetition_penalty,
            params.temperature, params.top_k, params.top_p,
            params.subtalker_temperature, params.subtalker_top_k,
            params.non_streaming_mode,
            params.subtalker_top_p);
    }

    if (!gen_ok) {
        result.error_msg = "Code generation failed: " + transformer_.get_error();
        return result;
    }
    result.t_generate_ms = get_time_ms() - t0;
    snap_mem("synth/after-generate");

    if (speech_codes.empty()) {
        result.error_msg = "No speech codes generated";
        return result;
    }

    if (low_mem_mode_) {
        transformer_.unload_model();
        transformer_loaded_ = false;
    }

    // ---- If caller only wants codes (no audio decode) -------------------
    if (codes_out) {
        *codes_out = speech_codes;
        result.sample_rate = audio_decoder_.get_config().sample_rate;
        result.success     = true;
        result.t_total_ms  = get_time_ms() - t_total;
        return result;
    }

    // ---- Decode to waveform (streaming or single-shot) ------------------
    if (!decode_codes_streaming(speech_codes, result, params)) return result;
    snap_mem("synth/after-decode");

    result.sample_rate = audio_decoder_.get_config().sample_rate;
    result.success     = true;
    result.t_total_ms  = get_time_ms() - t_total;
    snap_mem("synth/end");

    if (params.print_timing) {
        const double audio_sec = result.sample_rate > 0
            ? (double)result.audio.size() / (double)result.sample_rate : 0.0;
        const double wall_sec  = result.t_total_ms / 1000.0;
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Tokenize:  %6lld ms\n", (long long)result.t_tokenize_ms);
        fprintf(stderr, "  Encode:    %6lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Generate:  %6lld ms\n", (long long)result.t_generate_ms);
        fprintf(stderr, "  Decode:    %6lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:     %6lld ms\n", (long long)result.t_total_ms);
        if (audio_sec > 0)
            fprintf(stderr, "  RTF:       %.3f  (%.2fx realtime)\n",
                    wall_sec / audio_sec, audio_sec / wall_sec);
    }

    return result;
}

// ============================================================
// Resampling (public wrapper around the internal sinc resampler)
// ============================================================
void resample_audio(const float * input, int32_t n_input,
                    int32_t input_rate, int32_t output_rate,
                    std::vector<float> & output) {
    resample_linear(input, (int)n_input, (int)input_rate, output, (int)output_rate);
}

// ============================================================
// WAV I/O
// ============================================================
bool load_audio_file(const std::string & path, std::vector<float> & samples, int & sample_rate) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open WAV: %s\n", path.c_str());
        return false;
    }

    char riff[4];
    uint32_t file_size;
    char wave[4];
    if (fread(riff, 1, 4, f) != 4 || strncmp(riff, "RIFF", 4) != 0) { fclose(f); return false; }
    if (fread(&file_size, 4, 1, f) != 1)                              { fclose(f); return false; }
    if (fread(wave, 1, 4, f) != 4 || strncmp(wave, "WAVE", 4) != 0)  { fclose(f); return false; }

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sr = 0;

    while (!feof(f)) {
        char   chunk_id[4];
        uint32_t chunk_size;
        if (fread(chunk_id,   1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            if (fread(&audio_format,    2, 1, f) != 1) break;
            if (fread(&num_channels,    2, 1, f) != 1) break;
            if (fread(&sr,              4, 1, f) != 1) break;
            fseek(f, 6, SEEK_CUR); // skip byte_rate + block_align
            if (fread(&bits_per_sample, 2, 1, f) != 1) break;
            if (chunk_size > 16) fseek(f, chunk_size - 16, SEEK_CUR);
            // Pad to even boundary (RIFF spec: all chunks are word-aligned)
            if (chunk_size & 1) fseek(f, 1, SEEK_CUR);
        } else if (strncmp(chunk_id, "data", 4) == 0) {
            sample_rate = (int)sr;
            if (num_channels == 0) {
                fprintf(stderr, "ERROR: WAV fmt chunk missing or corrupt: %s\n", path.c_str());
                fclose(f); return false;
            }
            // Reject absurdly large data chunks (DoS guard — ~10 min mono 16-bit @ 48 kHz).
            constexpr uint32_t k_max_data_bytes = 64u * 1024u * 1024u;
            if (chunk_size > k_max_data_bytes) {
                fprintf(stderr, "ERROR: WAV data chunk too large (%u bytes): %s\n",
                        chunk_size, path.c_str());
                fclose(f);
                return false;
            }
            if (audio_format == 1 && bits_per_sample == 16) {
                int n = (int)(chunk_size / (2u * num_channels));
                std::vector<int16_t> raw((size_t)n * num_channels);
                samples.resize((size_t)n);
                size_t got = fread(raw.data(), 2, (size_t)n * num_channels, f);
                int n_read = (int)(got / num_channels);
                for (int i = 0; i < n_read; ++i) {
                    float sum = 0;
                    for (int c = 0; c < num_channels; ++c) sum += raw[i*num_channels+c] / 32768.0f;
                    samples[i] = sum / num_channels;
                }
                samples.resize((size_t)n_read);
            } else if (audio_format == 1 && bits_per_sample == 32) {
                int n = (int)(chunk_size / (4u * num_channels));
                std::vector<int32_t> raw((size_t)n * num_channels);
                samples.resize((size_t)n);
                size_t got = fread(raw.data(), 4, (size_t)n * num_channels, f);
                int n_read = (int)(got / num_channels);
                for (int i = 0; i < n_read; ++i) {
                    float sum = 0;
                    for (int c = 0; c < num_channels; ++c) sum += raw[i*num_channels+c] / 2147483648.0f;
                    samples[i] = sum / num_channels;
                }
                samples.resize((size_t)n_read);
            } else if (audio_format == 3) {
                int n = (int)(chunk_size / (4u * num_channels));
                std::vector<float> raw((size_t)n * num_channels);
                samples.resize((size_t)n);
                size_t got = fread(raw.data(), 4, (size_t)n * num_channels, f);
                int n_read = (int)(got / num_channels);
                for (int i = 0; i < n_read; ++i) {
                    float sum = 0;
                    for (int c = 0; c < num_channels; ++c) sum += raw[i*num_channels+c];
                    samples[i] = sum / num_channels;
                }
                samples.resize((size_t)n_read);
            } else {
                fprintf(stderr, "ERROR: Unsupported WAV format %d / %d bps\n",
                        audio_format, bits_per_sample);
                fclose(f);
                return false;
            }
            fclose(f);
            return !samples.empty();
        } else {
            // Skip unknown chunk — pad to even boundary per RIFF spec
            uint32_t skip = chunk_size + (chunk_size & 1u);
#ifdef _WIN32
            _fseeki64(f, (int64_t)skip, SEEK_CUR);
#else
            fseeko(f, (off_t)skip, SEEK_CUR);
#endif
        }
    }
    fclose(f);
    return false;
}

bool save_audio_file(const std::string & path, const std::vector<float> & samples,
                     int sample_rate) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "ERROR: Cannot create WAV: %s\n", path.c_str()); return false; }

    uint16_t num_channels = 1, bits = 16;
    uint32_t byte_rate   = (uint32_t)(sample_rate * num_channels * bits / 8);
    uint16_t block_align = (uint16_t)(num_channels * bits / 8);
    uint32_t data_size   = (uint32_t)(samples.size() * block_align);
    uint32_t file_size   = 36 + data_size;
    uint32_t fmt_size    = 16;
    uint16_t pcm         = 1;
    uint32_t sr          = (uint32_t)sample_rate;

    // Write WAV header — check every write to detect full-disk early
#define W(ptr, sz, n) do { if (fwrite((ptr),(sz),(n),f)!=(size_t)(n)) { fclose(f); return false; } } while(0)
    W("RIFF",       1, 4);  W(&file_size,   4, 1);
    W("WAVE",       1, 4);  W("fmt ",       1, 4);
    W(&fmt_size,    4, 1);  W(&pcm,         2, 1);
    W(&num_channels,2, 1);  W(&sr,          4, 1);
    W(&byte_rate,   4, 1);  W(&block_align, 2, 1);
    W(&bits,        2, 1);  W("data",       1, 4);
    W(&data_size,   4, 1);
#undef W

    // Convert float32 → int16 in one batch and write once for performance
    std::vector<int16_t> pcm_buf(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        float s = samples[i];
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
        pcm_buf[i] = (int16_t)(s * 32767.0f);
    }
    if (!pcm_buf.empty() && fwrite(pcm_buf.data(), 2, pcm_buf.size(), f) != pcm_buf.size()) {
        fclose(f); return false;
    }
    fclose(f);
    return true;
}

bool save_speaker_embedding(const std::string & path, const std::vector<float> & embedding) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    uint32_t n = (uint32_t)embedding.size();
    fwrite(&n, 4, 1, f);
    fwrite(embedding.data(), 4, n, f);
    fclose(f);
    return true;
}

bool load_speaker_embedding(const std::string & path, std::vector<float> & embedding) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t n = 0;
    if (fread(&n, 4, 1, f) != 1 || n == 0) { fclose(f); return false; }
    embedding.resize(n);
    if (fread(embedding.data(), 4, n, f) != n) { fclose(f); embedding.clear(); return false; }
    fclose(f);
    return true;
}

} // namespace qwen3_tts


