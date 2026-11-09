#include "qwen3_tts.h"
#include "audio_player.h"
#include "ggml.h"

// ---------------------------------------------------------------------------
// GGML log filter — GGML's Metal backend logs every shader-pipeline
// compilation at DEBUG level, which floods the console on first use.
// Keep warnings and errors visible; drop everything below that.
// ---------------------------------------------------------------------------
static void ggml_log_filter(enum ggml_log_level level, const char * text, void * /*user*/) {
    static bool show_cont = false;
    if (level == GGML_LOG_LEVEL_CONT) {   // continuation of a previous message
        if (show_cont) fputs(text, stderr);
        return;
    }
    show_cont = (level >= GGML_LOG_LEVEL_WARN);
    if (show_cont) fputs(text, stderr);
}

#include <cstdio>
#include <cstring>
#include <string>
#include <cstdlib>
#include <iostream>
#include <cerrno>
#include <limits>
#include <chrono>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

static bool terminal_stderr() {
#ifdef _WIN32
    return _isatty(_fileno(stderr)) != 0;
#else
    return isatty(fileno(stderr)) != 0;
#endif
}

static auto terminal_progress() {
    using Clock = std::chrono::steady_clock;
    return [last = Clock::time_point{}](int tok, int max) mutable -> int {
        const auto now = Clock::now();
        if (tok == 1 || tok >= max || now - last >= std::chrono::milliseconds(250)) {
            fprintf(stderr, "\rGenerating: %d/%d tokens", tok, max);
            last = now;
        }
        return 0;
    };
}

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s -m <model_dir> -t <text> [options]\n\n", prog);
    fprintf(stderr, "Core:\n");
    fprintf(stderr, "  -m, --model <dir|file>        Model directory containing GGUF files (required)\n");
    fprintf(stderr, "                                Or a direct path to a .gguf TTS model file.\n");
    fprintf(stderr, "                                Direct file: tokenizer is found alongside it.\n");
    fprintf(stderr, "  --tokenizer <file>            Explicit path to tokenizer GGUF (overrides auto-discovery).\n");
    fprintf(stderr, "                                Useful for comparing f16-f16 vs f16-f32 Mimi quality.\n");
    fprintf(stderr, "  -t, --text <text>              Text to synthesize (required for synthesis)\n");
    fprintf(stderr, "  -o, --output <file>            Output WAV file (default: output.wav)\n");
    fprintf(stderr, "  -s, --speak                    Play audio through speakers instead of saving\n");
    fprintf(stderr, "                                 (pass -o as well to do both)\n");
    fprintf(stderr, "  --vocoder-chunk-frames <n>      Vocoder window including overlap (17..256, default: 100)\n");
    fprintf(stderr, "  --output-rate <hz>             Resample output to this rate (e.g. 48000).\n");
    fprintf(stderr, "                                 Default is native 24000 Hz. 48000 is 2x upsample.\n");
    fprintf(stderr, "                                 NOTE: upsampling does not improve audio quality.\n");
    fprintf(stderr, "\nVoice cloning (Base model):\n");
    fprintf(stderr, "  -r, --reference <file>         Reference audio WAV for voice cloning\n");
    fprintf(stderr, "  --ref-text <text>              Reference transcript (enables full ICL mode)\n");
    fprintf(stderr, "  --embedding-in <file>          Load pre-computed speaker embedding (.bin)\n");
    fprintf(stderr, "  --embedding-out <file>         Save speaker embedding and exit (no synthesis)\n");
    fprintf(stderr, "\nModel variants:\n");
    fprintf(stderr, "  --speaker <name>               Named speaker for CustomVoice models\n");
    fprintf(stderr, "  --list-speakers                List available speakers and exit\n");
    fprintf(stderr, "  --instruct <text>              Style/emotion instruction (VoiceDesign/CustomVoice)\n");
    fprintf(stderr, "\nLanguage:\n");
    fprintf(stderr, "  -l, --language <lang>          Output language (default: auto)\n");
    fprintf(stderr, "                                 auto, english, chinese, japanese, korean,\n");
    fprintf(stderr, "                                 russian, german, french, spanish, italian,\n");
    fprintf(stderr, "                                 portuguese, or raw codec ID (e.g. 2050)\n");
    fprintf(stderr, "  --list-languages               List supported languages and exit\n");
    fprintf(stderr, "\nSampling (main talker):\n");
    fprintf(stderr, "  --temperature <val>            Temperature (default: 0.9, 0=greedy)\n");
    fprintf(stderr, "  --top-k <n>                    Top-k (default: 50, 0=disabled)\n");
    fprintf(stderr, "  --top-p <val>                  Top-p nucleus sampling (default: 1.0)\n");
    fprintf(stderr, "  --min-p <val>                  Min-p: keep tokens where prob >= val*max (0=off)\n");
    fprintf(stderr, "  --repetition-penalty <val>     Repetition penalty (default: 1.05)\n");
    fprintf(stderr, "  --frequency-penalty <val>      Frequency penalty: subtract val*count (0=off)\n");
    fprintf(stderr, "  --presence-penalty <val>       Presence penalty: flat subtract if seen (0=off)\n");
    fprintf(stderr, "  --dry-multiplier <val>         DRY n-gram penalty scale (0=off, try 0.8)\n");
    fprintf(stderr, "  --dyntemp-range <val>          Dynamic temperature half-range (0=off)\n");
    fprintf(stderr, "\nSampling (sub-talker / code predictor):\n");
    fprintf(stderr, "  --sub-temperature <val>        Sub-talker temperature (-1=inherit main)\n");
    fprintf(stderr, "  --sub-top-k <n>                Sub-talker top-k (-1=inherit main)\n");
    fprintf(stderr, "  --sub-top-p <val>              Sub-talker top-p (-1=inherit main)\n");
    fprintf(stderr, "\nGeneration:\n");
    fprintf(stderr, "  --max-tokens <n>               Max audio frames to generate (default: 4096)\n");
    fprintf(stderr, "  --non-streaming                Non-streaming prefill layout\n");
    fprintf(stderr, "\nServer mode:\n");
    fprintf(stderr, "  --server                       Load model once, read JSON from stdin,\n");
    fprintf(stderr, "                                 write JSON to stdout. One request per line.\n");
    fprintf(stderr, "                                 Request:  {\"text\":\"...\",\"output\":\"out.wav\",...}\n");
    fprintf(stderr, "                                 Response: {\"success\":true,\"duration_s\":2.5,...}\n");
    fprintf(stderr, "                                 Send EOF to exit.\n");
    fprintf(stderr, "\nMisc:\n");
    fprintf(stderr, "  -j, --threads <n>              CPU threads (default: auto-detect)\n");
    fprintf(stderr, "  --gen-config <file>            Load generation_config.json\n");
    fprintf(stderr, "  --version                      Show version and exit\n");
    fprintf(stderr, "  -h, --help                     Show this help\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  # Basic synthesis\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello, world!\" -o hello.wav\n", prog);
    fprintf(stderr, "  # Force a specific model file (e.g. F16 for better quality)\n");
    fprintf(stderr, "  %s -m ./models/qwen3-tts-0.6b-f16.gguf -t \"Hello!\" -o hello.wav\n", prog);
    fprintf(stderr, "  # Voice clone\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" -r ref.wav -o clone.wav\n", prog);
    fprintf(stderr, "  # Voice clone with ICL (better quality)\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" -r ref.wav --ref-text \"The reference.\" -o icl.wav\n", prog);
    fprintf(stderr, "  # CustomVoice named speaker\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" --speaker Vivian -o vivian.wav\n", prog);
    fprintf(stderr, "  # VoiceDesign (describe the voice)\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" --instruct \"Calm, warm female voice\" -o vd.wav\n", prog);
    fprintf(stderr, "  # Chinese synthesis\n");
    fprintf(stderr, "  %s -m ./models -t \"你好世界\" -l chinese -o chinese.wav\n", prog);
    fprintf(stderr, "  # Save speaker embedding for fast reuse\n");
    fprintf(stderr, "  %s -m ./models -r ref.wav --embedding-out voice.bin\n", prog);
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" --embedding-in voice.bin -o out.wav\n", prog);
    fprintf(stderr, "  # Server mode\n");
    fprintf(stderr, "  %s -m ./models --server\n", prog);
    fprintf(stderr, "  # 48 kHz output (for DAW/video compatibility)\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" --output-rate 48000 -o hello_48k.wav\n", prog);
}

// ---------------------------------------------------------------------------
// RFC 8259 JSON object helpers — no external deps, UTF-8 + surrogate decoding
// ---------------------------------------------------------------------------

static void json_skip_ws(const std::string & s, size_t & pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\r' || s[pos] == '\n')) {
        ++pos;
    }
}

static bool json_parse_hex4(const std::string & s, size_t pos, uint32_t & out) {
    if (pos + 4 > s.size()) return false;
    uint32_t v = 0;
    for (size_t i = 0; i < 4; ++i) {
        char c = s[pos + i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
        else return false;
    }
    out = v;
    return true;
}

static void json_append_utf8(std::string & out, uint32_t cp) {
    if (cp <= 0x7Fu) {
        out.push_back((char)cp);
    } else if (cp <= 0x7FFu) {
        out.push_back((char)(0xC0u | (cp >> 6)));
        out.push_back((char)(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out.push_back((char)(0xE0u | (cp >> 12)));
        out.push_back((char)(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back((char)(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0x10FFFFu) {
        out.push_back((char)(0xF0u | (cp >> 18)));
        out.push_back((char)(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back((char)(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back((char)(0x80u | (cp & 0x3Fu)));
    }
}

static bool json_parse_string(const std::string & s, size_t & pos, std::string & out) {
    if (pos >= s.size() || s[pos] != '"') return false;
    ++pos;
    out.clear();
    while (pos < s.size()) {
        unsigned char c = (unsigned char)s[pos++];
        if (c == '"') return true;
        if (c < 0x20u) return false; // unescaped control character
        if (c != '\\') {
            out.push_back((char)c);
            continue;
        }
        if (pos >= s.size()) return false;
        char esc = s[pos++];
        switch (esc) {
            case '"':  out.push_back('"');  break;
            case '\\': out.push_back('\\'); break;
            case '/':  out.push_back('/');  break;
            case 'b':  out.push_back('\b'); break;
            case 'f':  out.push_back('\f'); break;
            case 'n':  out.push_back('\n'); break;
            case 'r':  out.push_back('\r'); break;
            case 't':  out.push_back('\t'); break;
            case 'u': {
                uint32_t cp = 0;
                if (!json_parse_hex4(s, pos, cp)) return false;
                pos += 4;
                if (cp >= 0xD800u && cp <= 0xDBFFu) {
                    if (pos + 6 > s.size() || s[pos] != '\\' || s[pos + 1] != 'u') return false;
                    uint32_t low = 0;
                    if (!json_parse_hex4(s, pos + 2, low)) return false;
                    if (low < 0xDC00u || low > 0xDFFFu) return false;
                    pos += 6;
                    cp = 0x10000u + (((cp - 0xD800u) << 10) | (low - 0xDC00u));
                } else if (cp >= 0xDC00u && cp <= 0xDFFFu) {
                    return false; // unpaired low surrogate
                }
                json_append_utf8(out, cp);
                break;
            }
            default:
                return false;
        }
    }
    return false;
}

static bool json_parse_number_token(const std::string & s, size_t & pos, std::string & raw) {
    size_t start = pos;
    if (pos < s.size() && s[pos] == '-') ++pos;
    if (pos >= s.size()) return false;
    if (s[pos] == '0') {
        ++pos;
        if (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') return false; // leading zero
    } else if (s[pos] >= '1' && s[pos] <= '9') {
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
    } else {
        return false;
    }
    if (pos < s.size() && s[pos] == '.') {
        ++pos;
        if (pos >= s.size() || s[pos] < '0' || s[pos] > '9') return false;
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
    }
    if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
        ++pos;
        if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) ++pos;
        if (pos >= s.size() || s[pos] < '0' || s[pos] > '9') return false;
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
    }
    raw = s.substr(start, pos - start);
    return true;
}

enum class JsonFieldKind { STRING, NUMBER, BOOL, NULL_VAL, COMPOSITE };

struct JsonFieldValue {
    JsonFieldKind kind = JsonFieldKind::NULL_VAL;
    std::string text;
};

static bool json_skip_value(const std::string & s, size_t & pos, JsonFieldValue * out_val, int depth = 0) {
    if (depth > 64) return false;
    json_skip_ws(s, pos);
    if (pos >= s.size()) return false;
    char c = s[pos];
    if (c == '"') {
        std::string decoded;
        if (!json_parse_string(s, pos, decoded)) return false;
        if (out_val) { out_val->kind = JsonFieldKind::STRING; out_val->text = std::move(decoded); }
        return true;
    }
    if (c == '{') {
        size_t start = pos++;
        json_skip_ws(s, pos);
        if (pos < s.size() && s[pos] == '}') {
            ++pos;
            if (out_val) { out_val->kind = JsonFieldKind::COMPOSITE; out_val->text = s.substr(start, pos - start); }
            return true;
        }
        while (pos < s.size()) {
            json_skip_ws(s, pos);
            std::string k;
            if (!json_parse_string(s, pos, k)) return false;
            json_skip_ws(s, pos);
            if (pos >= s.size() || s[pos] != ':') return false;
            ++pos;
            if (!json_skip_value(s, pos, nullptr, depth + 1)) return false;
            json_skip_ws(s, pos);
            if (pos < s.size() && s[pos] == ',') { ++pos; continue; }
            if (pos < s.size() && s[pos] == '}') {
                ++pos;
                if (out_val) { out_val->kind = JsonFieldKind::COMPOSITE; out_val->text = s.substr(start, pos - start); }
                return true;
            }
            return false;
        }
        return false;
    }
    if (c == '[') {
        size_t start = pos++;
        json_skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ']') {
            ++pos;
            if (out_val) { out_val->kind = JsonFieldKind::COMPOSITE; out_val->text = s.substr(start, pos - start); }
            return true;
        }
        while (pos < s.size()) {
            if (!json_skip_value(s, pos, nullptr, depth + 1)) return false;
            json_skip_ws(s, pos);
            if (pos < s.size() && s[pos] == ',') { ++pos; continue; }
            if (pos < s.size() && s[pos] == ']') {
                ++pos;
                if (out_val) { out_val->kind = JsonFieldKind::COMPOSITE; out_val->text = s.substr(start, pos - start); }
                return true;
            }
            return false;
        }
        return false;
    }
    if (s.compare(pos, 4, "true") == 0) {
        pos += 4;
        if (out_val) { out_val->kind = JsonFieldKind::BOOL; out_val->text = "true"; }
        return true;
    }
    if (s.compare(pos, 5, "false") == 0) {
        pos += 5;
        if (out_val) { out_val->kind = JsonFieldKind::BOOL; out_val->text = "false"; }
        return true;
    }
    if (s.compare(pos, 4, "null") == 0) {
        pos += 4;
        if (out_val) { out_val->kind = JsonFieldKind::NULL_VAL; out_val->text = "null"; }
        return true;
    }
    std::string num;
    if (json_parse_number_token(s, pos, num)) {
        if (out_val) { out_val->kind = JsonFieldKind::NUMBER; out_val->text = std::move(num); }
        return true;
    }
    return false;
}

static bool json_find_top_level_field(const std::string & json, const std::string & target_key,
                                      JsonFieldValue & found_val, bool & found, bool * valid_json = nullptr) {
    found = false;
    if (valid_json) *valid_json = false;
    size_t pos = 0;
    json_skip_ws(json, pos);
    if (pos >= json.size() || json[pos] != '{') return false;
    ++pos;
    json_skip_ws(json, pos);
    if (pos < json.size() && json[pos] == '}') {
        ++pos;
        json_skip_ws(json, pos);
        if (pos != json.size()) return false;
        if (valid_json) *valid_json = true;
        return true;
    }
    while (pos < json.size()) {
        json_skip_ws(json, pos);
        std::string key;
        if (!json_parse_string(json, pos, key)) return false;
        json_skip_ws(json, pos);
        if (pos >= json.size() || json[pos] != ':') return false;
        ++pos;
        JsonFieldValue val;
        if (!json_skip_value(json, pos, &val, 0)) return false;
        if (key == target_key) {
            found_val = std::move(val);
            found = true;
        }
        json_skip_ws(json, pos);
        if (pos < json.size() && json[pos] == ',') {
            ++pos;
            continue;
        }
        if (pos < json.size() && json[pos] == '}') {
            ++pos;
            json_skip_ws(json, pos);
            if (pos != json.size()) return false;
            if (valid_json) *valid_json = true;
            return true;
        }
        return false;
    }
    return false;
}

// Extract a string value from a top-level JSON object. Returns empty string if not found or invalid.
static std::string json_get_str(const std::string & json, const std::string & key) {
    JsonFieldValue val;
    bool found = false;
    if (!json_find_top_level_field(json, key, val, found) || !found || val.kind != JsonFieldKind::STRING) {
        return "";
    }
    return val.text;
}

static float json_get_float(const std::string & json, const std::string & key, float def) {
    JsonFieldValue val;
    bool found = false;
    if (!json_find_top_level_field(json, key, val, found) || !found || val.kind != JsonFieldKind::NUMBER) {
        return def;
    }
    errno = 0;
    char * end = nullptr;
    float v = strtof(val.text.c_str(), &end);
    if (end == val.text.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(v)) return def;
    return v;
}

static bool json_get_strict_int(const std::string & json, const std::string & key,
                                int32_t & out, bool & present) {
    JsonFieldValue val;
    present = false;
    if (!json_find_top_level_field(json, key, val, present)) return false;
    if (!present) return true;
    if (val.kind != JsonFieldKind::NUMBER) return false;
    errno = 0;
    char * end = nullptr;
    long v = strtol(val.text.c_str(), &end, 10);
    if (end == val.text.c_str() || *end != '\0' || errno == ERANGE ||
        v < std::numeric_limits<int32_t>::min() || v > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    out = (int32_t)v;
    return true;
}

static int json_get_int(const std::string & json, const std::string & key, int def) {
    int32_t v = def;
    bool present = false;
    if (!json_get_strict_int(json, key, v, present) || !present) return def;
    return (int)v;
}

// Escape a string for JSON output
static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        if      (c == '"')  { out += "\\\""; }
        else if (c == '\\') { out += "\\\\"; }
        else if (c == '\b') { out += "\\b";  }
        else if (c == '\f') { out += "\\f";  }
        else if (c == '\n') { out += "\\n";  }
        else if (c == '\r') { out += "\\r";  }
        else if (c == '\t') { out += "\\t";  }
        else if (c < 0x20u) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
            out += buf;
        } else {
            out.push_back((char)c);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Single synthesis helper — shared by one-shot and server modes
// ---------------------------------------------------------------------------

static int do_synthesize(qwen3_tts::Qwen3TTS & tts,
                          const std::string & text,
                          const std::string & output_file,
                          const std::string & reference_audio,
                          const std::string & ref_text,
                          const std::string & embedding_in,
                          const std::string & embedding_out_path,
                          qwen3_tts::tts_params params,
                          bool quiet,
                          bool speak,
                          bool output_explicit,
                          int32_t output_rate = 0) {

    // Propagate ICL mode
    if (!ref_text.empty()) {
        params.ref_text = ref_text;
        params.icl_mode = true;
    }

    if (!quiet && terminal_stderr()) {
        tts.set_progress_callback(terminal_progress());
    } else {
        tts.set_progress_callback(nullptr);
    }

    qwen3_tts::tts_result result;

    if (!embedding_in.empty()) {
        std::vector<float> embedding;
        if (!tts.load_speaker_embedding(embedding_in, embedding)) {
            fprintf(stderr, "Error: failed to load embedding from %s\n", embedding_in.c_str());
            return 1;
        }
        if (!quiet)
            fprintf(stderr, "Synthesizing with loaded embedding (%zu floats)\n", embedding.size());
        result = tts.synthesize_with_embedding(text, embedding, params);
    } else if (!reference_audio.empty()) {
        if (!quiet) {
            fprintf(stderr, "Synthesizing with voice clone: \"%s\"\n", text.c_str());
            fprintf(stderr, "Reference: %s%s\n", reference_audio.c_str(),
                    params.icl_mode ? " (ICL mode)" : " (x-vector only)");
        }
        params.reference_audio = reference_audio;
        result = tts.synthesize_with_voice(text, reference_audio, params);
        if (!embedding_out_path.empty()) {
            std::vector<float> embedding;
            if (tts.extract_speaker_embedding(reference_audio, embedding)) {
                if (!tts.save_speaker_embedding(embedding_out_path, embedding)) {
                    fprintf(stderr, "Warning: failed to save embedding to %s\n",
                            embedding_out_path.c_str());
                }
            }
        }
    } else {
        if (!quiet) {
            fprintf(stderr, "Synthesizing: \"%s\"\n", text.c_str());
            if (!params.speaker.empty())
                fprintf(stderr, "  Speaker: %s\n", params.speaker.c_str());
            if (!params.instruct.empty())
                fprintf(stderr, "  Instruct: %s\n", params.instruct.c_str());
            if (!params.language.empty())
                fprintf(stderr, "  Language: %s\n", params.language.c_str());
        }
        result = tts.synthesize(text, params);
    }

    if (!quiet) fprintf(stderr, "\n");

    if (!result.success) {
        fprintf(stderr, "Error: %s\n", result.error_msg.c_str());
        return 1;
    }

    // Optionally resample output to a different rate (e.g. 48000 for DAW compatibility)
    std::vector<float> final_audio = result.audio;
    int32_t final_rate = result.sample_rate;
    if (output_rate > 0 && output_rate != result.sample_rate) {
        std::vector<float> resampled;
        qwen3_tts::resample_audio(result.audio.data(), (int32_t)result.audio.size(),
                                   result.sample_rate, output_rate, resampled);
        final_audio = std::move(resampled);
        final_rate  = output_rate;
        if (!quiet)
            fprintf(stderr, "Resampled: %d Hz -> %d Hz\n", result.sample_rate, output_rate);
    }

    if (speak) {
        if (!quiet)
            fprintf(stderr, "Playing: %.2f s @ %d Hz...\n",
                    (float)final_audio.size() / (float)final_rate, final_rate);
        if (!qwen3_tts::play_audio_samples(final_audio.data(),
                                           (int32_t)final_audio.size(), final_rate)) {
            fprintf(stderr, "Error: audio playback failed\n");
            return 1;
        }
    }

    if (!speak || output_explicit) {
        if (!qwen3_tts::save_audio_file(output_file, final_audio, final_rate)) {
            fprintf(stderr, "Error: failed to save %s\n", output_file.c_str());
            return 1;
        }

        if (!quiet) {
            fprintf(stderr, "Saved: %s (%.2f s @ %d Hz)\n",
                    output_file.c_str(),
                    (float)final_audio.size() / (float)final_rate,
                    final_rate);
        }
    }

    if (!quiet && params.print_timing) {
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Tokenize:  %6lld ms\n", (long long)result.t_tokenize_ms);
        fprintf(stderr, "  Encode:    %6lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Generate:  %6lld ms\n", (long long)result.t_generate_ms);
        fprintf(stderr, "  Decode:    %6lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:     %6lld ms\n", (long long)result.t_total_ms);
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Server mode
// ---------------------------------------------------------------------------

// Run the server loop: read JSON lines from stdin, write JSON lines to stdout.
// The model is already loaded in `tts`. Each request is processed serially.
// This keeps the model in memory across all requests — no per-request reload.
//
// Protocol:
//   Input  (stdin):  one JSON object per line
//   Output (stdout): one JSON object per line (flushed immediately)
//   Logging:         stderr (does not interfere with stdout JSON protocol)
//
// Request fields:
//   text            (required) text to synthesize
//   output          (required) output WAV file path
//   reference       (optional) reference WAV for voice cloning
//   ref_text        (optional) reference transcript (enables ICL)
//   embedding_in    (optional) path to pre-computed speaker embedding
//   speaker         (optional) named speaker (CustomVoice)
//   instruct        (optional) style instruction (VoiceDesign/CustomVoice)
//   language        (optional) language name or codec ID
//   temperature     (optional) float, default 0.9
//   top_k           (optional) int, default 50
//   top_p           (optional) float, default 1.0
//   repetition_penalty (optional) float, default 1.05
//   max_tokens      (optional) int, default 4096
//
// Response fields:
//   success         true / false
//   output          (on success) output WAV file path
//   duration_s      (on success) audio duration in seconds
//   t_total_ms      (on success) total synthesis time in ms
//   error           (on failure) error message

static int run_server(qwen3_tts::Qwen3TTS & tts, const qwen3_tts::tts_params & default_params) {
    fprintf(stderr, "Server mode: ready. Reading JSON requests from stdin.\n");
    fprintf(stderr, "Send EOF (Ctrl+D on Unix, Ctrl+Z on Windows) to exit.\n\n");

    std::string line;
    int request_num = 0;

    while (std::getline(std::cin, line)) {
        // Skip blank lines and comment-like lines
        if (line.empty() || line[0] == '#') continue;

        ++request_num;
        fprintf(stderr, "[server] Request %d: processing...\n", request_num);

        JsonFieldValue probe_val;
        bool probe_found = false;
        bool valid_json = false;
        if (!json_find_top_level_field(line, "text", probe_val, probe_found, &valid_json) || !valid_json) {
            fprintf(stderr, "[server] Request %d: malformed JSON request\n", request_num);
            printf("{\"success\":false,\"error\":\"malformed JSON request\"}\n");
            fflush(stdout);
            continue;
        }

        // Parse request fields
        std::string text          = json_get_str(line, "text");
        std::string output_file   = json_get_str(line, "output");
        std::string reference     = json_get_str(line, "reference");
        std::string ref_text      = json_get_str(line, "ref_text");
        std::string embedding_in  = json_get_str(line, "embedding_in");

        if (text.empty()) {
            fprintf(stderr, "[server] Request %d: missing 'text' field\n", request_num);
            printf("{\"success\":false,\"error\":\"missing required field: text\"}\n");
            fflush(stdout);
            continue;
        }
        if (output_file.empty()) {
            // Default output name based on request number
            char buf[64];
            snprintf(buf, sizeof(buf), "server_output_%04d.wav", request_num);
            output_file = buf;
        }

        // Build params from defaults + per-request overrides
        qwen3_tts::tts_params params = default_params;
        params.print_timing   = false;
        params.print_progress = false;

        std::string speaker  = json_get_str(line, "speaker");
        std::string instruct = json_get_str(line, "instruct");
        std::string language = json_get_str(line, "language");
        if (!speaker.empty())  params.speaker  = speaker;
        if (!instruct.empty()) params.instruct = instruct;
        if (!language.empty()) params.language = language;

        float temp = json_get_float(line, "temperature", -999.0f);
        if (temp > -998.0f) params.temperature = temp;
        int top_k = json_get_int(line, "top_k", -999);
        if (top_k > -998) params.top_k = top_k;
        float top_p = json_get_float(line, "top_p", -999.0f);
        if (top_p > -998.0f) params.top_p = top_p;
        float rep_pen = json_get_float(line, "repetition_penalty", -999.0f);
        if (rep_pen > -998.0f) params.repetition_penalty = rep_pen;
        int max_tok = json_get_int(line, "max_tokens", -999);
        if (max_tok > -998) params.max_audio_tokens = max_tok;

        int32_t req_vocoder_window = params.vocoder_chunk_frames;
        bool has_vocoder_window = false;
        if (!json_get_strict_int(line, "vocoder_chunk_frames", req_vocoder_window, has_vocoder_window) ||
            (has_vocoder_window && !qwen3_tts::valid_vocoder_chunk_frames(req_vocoder_window))) {
            fprintf(stderr, "[server] Request %d: invalid vocoder_chunk_frames\n", request_num);
            printf("{\"success\":false,\"error\":\"vocoder_chunk_frames must be an integer from 17 to 256\"}\n");
            fflush(stdout);
            continue;
        }
        if (has_vocoder_window) {
            params.vocoder_chunk_frames = req_vocoder_window;
        }

        int req_threads = json_get_int(line, "threads", -999);
        if (req_threads > 0) params.n_threads = req_threads;

        // Synthesize
        if (!ref_text.empty()) {
            params.ref_text = ref_text;
            params.icl_mode = true;
        }

        qwen3_tts::tts_result result;
        if (!embedding_in.empty()) {
            std::vector<float> emb;
            if (!tts.load_speaker_embedding(embedding_in, emb)) {
                std::string err = "Failed to load embedding: " + tts.get_error();
                fprintf(stderr, "[server] Request %d: %s\n", request_num, err.c_str());
                printf("{\"success\":false,\"error\":\"%s\"}\n", json_escape(err).c_str());
                fflush(stdout);
                continue;
            }
            result = tts.synthesize_with_embedding(text, emb, params);
        } else if (!reference.empty()) {
            params.reference_audio = reference;
            result = tts.synthesize_with_voice(text, reference, params);
        } else {
            result = tts.synthesize(text, params);
        }

        if (!result.success) {
            fprintf(stderr, "[server] Request %d: synthesis failed: %s\n",
                    request_num, result.error_msg.c_str());
            printf("{\"success\":false,\"error\":\"%s\"}\n",
                   json_escape(result.error_msg).c_str());
            fflush(stdout);
            continue;
        }

        // Save WAV
        if (!qwen3_tts::save_audio_file(output_file, result.audio, result.sample_rate)) {
            std::string err = "Failed to save WAV: " + output_file;
            fprintf(stderr, "[server] Request %d: %s\n", request_num, err.c_str());
            printf("{\"success\":false,\"error\":\"%s\"}\n", json_escape(err).c_str());
            fflush(stdout);
            continue;
        }

        double duration_s = result.sample_rate > 0
            ? (double)result.audio.size() / (double)result.sample_rate : 0.0;

        fprintf(stderr, "[server] Request %d: done — %.2fs audio in %lldms → %s\n",
                request_num, duration_s, (long long)result.t_total_ms, output_file.c_str());

        printf("{\"success\":true,\"output\":\"%s\",\"duration_s\":%.3f,\"t_tokenize_ms\":%lld,\"t_encode_ms\":%lld,\"t_generate_ms\":%lld,\"t_decode_ms\":%lld,\"t_total_ms\":%lld,\"rss_bytes\":%llu,\"peak_rss_bytes\":%llu}\n",
               json_escape(output_file).c_str(),
               duration_s,
               (long long)result.t_tokenize_ms,
               (long long)result.t_encode_ms,
               (long long)result.t_generate_ms,
               (long long)result.t_decode_ms,
               (long long)result.t_total_ms,
               (unsigned long long)result.mem_rss_end_bytes,
               (unsigned long long)result.mem_phys_peak_bytes);
        fflush(stdout);
    }

    fprintf(stderr, "[server] EOF received — processed %d request(s). Exiting.\n", request_num);
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    ggml_log_set(ggml_log_filter, nullptr);

    std::string model_dir;
    std::string tokenizer_path;  // optional override for tokenizer GGUF
    std::string text;
    std::string output_file = "output.wav";
    std::string reference_audio;
    std::string ref_text;
    std::string embedding_in;
    std::string embedding_out;
    std::string gen_config_path;
    bool list_speakers  = false;
    bool list_languages = false;
    bool server_mode    = false;
    bool speak          = false;
    bool output_explicit = false;
    int32_t output_rate = 0;  // 0 = native 24000 Hz, no resampling

    qwen3_tts::tts_params params;
    params.print_timing = true;  // CLI always shows timing; library callers get false by default

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

#define NEXT_ARG(dest)   do { if (++i >= argc) { fprintf(stderr, "Missing value for %s\n", arg.c_str()); return 1; } dest = argv[i]; } while(0)
#define NEXT_FLOAT(dest) do { \
    if (++i >= argc) { fprintf(stderr, "Missing value for %s\n", arg.c_str()); return 1; } \
    char * _end = nullptr; float _v = strtof(argv[i], &_end); \
    if (_end == argv[i] || *_end != '\0') { fprintf(stderr, "Invalid float for %s: %s\n", arg.c_str(), argv[i]); return 1; } \
    dest = _v; } while(0)
#define NEXT_INT(dest)   do { \
    if (++i >= argc) { fprintf(stderr, "Missing value for %s\n", arg.c_str()); return 1; } \
    errno = 0; char * _end = nullptr; long _v = strtol(argv[i], &_end, 10); \
    if (_end == argv[i] || *_end != '\0' || errno == ERANGE || \
        _v < std::numeric_limits<int32_t>::min() || _v > std::numeric_limits<int32_t>::max()) { fprintf(stderr, "Invalid integer for %s: %s\n", arg.c_str(), argv[i]); return 1; } \
    dest = (int32_t)_v; } while(0)

        if      (arg == "-h" || arg == "--help")            { print_usage(argv[0]); return 0; }
        else if (arg == "--version")                        { fprintf(stdout, "qwen3-tts-cli 0.1.0 (C++17/GGML)\n"); return 0; }
        else if (arg == "-m" || arg == "--model")           { NEXT_ARG(model_dir); }
        else if (arg == "--tokenizer")                      { NEXT_ARG(tokenizer_path); }
        else if (arg == "-t" || arg == "--text")            { NEXT_ARG(text); }
        else if (arg == "-o" || arg == "--output")          { NEXT_ARG(output_file); output_explicit = true; }
        else if (arg == "-s" || arg == "--speak")           { speak = true; }
        else if (arg == "-r" || arg == "--reference")       { NEXT_ARG(reference_audio); }
        else if (arg == "--ref-text")                       { NEXT_ARG(ref_text); }
        else if (arg == "--embedding-in")                   { NEXT_ARG(embedding_in); }
        else if (arg == "--embedding-out")                  { NEXT_ARG(embedding_out); }
        else if (arg == "--speaker")                        { NEXT_ARG(params.speaker); }
        else if (arg == "--instruct")                       { NEXT_ARG(params.instruct); }
        else if (arg == "-l" || arg == "--language")        { NEXT_ARG(params.language); }
        else if (arg == "--temperature")                    { NEXT_FLOAT(params.temperature); }
        else if (arg == "--top-k")                          { NEXT_INT(params.top_k); }
        else if (arg == "--top-p")                          { NEXT_FLOAT(params.top_p); }
        else if (arg == "--min-p")                          { NEXT_FLOAT(params.min_p); }
        else if (arg == "--repetition-penalty")             { NEXT_FLOAT(params.repetition_penalty); }
        else if (arg == "--frequency-penalty")              { NEXT_FLOAT(params.frequency_penalty); }
        else if (arg == "--presence-penalty")               { NEXT_FLOAT(params.presence_penalty); }
        else if (arg == "--dry-multiplier")                 { NEXT_FLOAT(params.dry_multiplier); }
        else if (arg == "--dyntemp-range")                  { NEXT_FLOAT(params.dyntemp_range); }
        else if (arg == "--sub-temperature")                { NEXT_FLOAT(params.subtalker_temperature); }
        else if (arg == "--sub-top-k")                      { NEXT_INT(params.subtalker_top_k); }
        else if (arg == "--sub-top-p")                      { NEXT_FLOAT(params.subtalker_top_p); }
        else if (arg == "--max-tokens")                     { NEXT_INT(params.max_audio_tokens); }
        else if (arg == "--vocoder-chunk-frames")            { NEXT_INT(params.vocoder_chunk_frames); }
        else if (arg == "--non-streaming")                  { params.non_streaming_mode = true; }
        else if (arg == "--output-rate")                    { NEXT_INT(output_rate); }
        else if (arg == "-j" || arg == "--threads")         { NEXT_INT(params.n_threads); }
        else if (arg == "--gen-config")                     { NEXT_ARG(gen_config_path); }
        else if (arg == "--list-speakers")                  { list_speakers  = true; }
        else if (arg == "--list-languages")                 { list_languages = true; }
        else if (arg == "--server")                         { server_mode = true; }
        else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }

#undef NEXT_ARG
#undef NEXT_FLOAT
#undef NEXT_INT
    }

    if (!qwen3_tts::valid_vocoder_chunk_frames(params.vocoder_chunk_frames)) {
        fprintf(stderr, "Error: --vocoder-chunk-frames must be an integer from 17 to 256\n");
        return 1;
    }

    if (model_dir.empty()) {
        fprintf(stderr, "Error: model directory required (-m)\n");
        print_usage(argv[0]);
        return 1;
    }

    // ---- Load models (once) --------------------------------------------
    qwen3_tts::Qwen3TTS tts;
    fprintf(stderr, "Loading models from: %s\n", model_dir.c_str());
    bool loaded = tokenizer_path.empty()
        ? tts.load_models(model_dir)
        : tts.load_models(model_dir, tokenizer_path);
    if (!loaded) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }

    if (!gen_config_path.empty()) {
        if (!tts.load_generation_config(gen_config_path)) {
            fprintf(stderr, "Warning: failed to load generation config from %s\n",
                    gen_config_path.c_str());
        }
    }

    // ---- Info queries --------------------------------------------------
    if (list_speakers) {
        auto spk = tts.get_supported_speakers();
        if (spk.empty()) {
            fprintf(stdout, "No named speakers available (model type: %s)\n",
                    tts.get_model_type().c_str());
        } else {
            fprintf(stdout, "Supported speakers (%zu):\n", spk.size());
            for (const auto & s : spk) fprintf(stdout, "  %s\n", s.c_str());
        }
        return 0;
    }
    if (list_languages) {
        auto langs = tts.get_supported_languages();
        fprintf(stdout, "Supported languages (%zu):\n", langs.size());
        for (const auto & l : langs) fprintf(stdout, "  %s\n", l.c_str());
        return 0;
    }

    // ---- Server mode ---------------------------------------------------
    if (server_mode) {
        return run_server(tts, params);
    }

    // ---- Embedding save-only mode (no synthesis) -----------------------
    if (!embedding_out.empty() && !reference_audio.empty() && text.empty()) {
        fprintf(stderr, "Extracting speaker embedding from: %s\n", reference_audio.c_str());
        std::vector<float> embedding;
        if (!tts.extract_speaker_embedding(reference_audio, embedding)) {
            fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
            return 1;
        }
        if (!tts.save_speaker_embedding(embedding_out, embedding)) {
            fprintf(stderr, "Error: failed to save embedding to %s\n", embedding_out.c_str());
            return 1;
        }
        fprintf(stderr, "Embedding saved to: %s (%zu floats)\n",
                embedding_out.c_str(), embedding.size());
        return 0;
    }

    if (text.empty()) {
        fprintf(stderr, "Error: text required (-t)\n");
        print_usage(argv[0]);
        return 1;
    }

    // ---- One-shot synthesis --------------------------------------------
    return do_synthesize(tts, text, output_file, reference_audio, ref_text,
                         embedding_in, embedding_out, params, /*quiet=*/false,
                         speak, output_explicit, output_rate);
}
