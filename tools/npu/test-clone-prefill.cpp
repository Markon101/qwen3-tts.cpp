// Synthetic and optional explicitly supplied real ICL controls; no raw audio input.
// Float fixtures remain in a caller-selected private directory outside Git.
#include "gguf_loader.h"
#include "text_tokenizer.h"
#include "tts_transformer.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <unordered_set>

static bool save(const std::string &path, const std::vector<float> &v) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(v.data(), sizeof(float), v.size(), f) == v.size();
    return fclose(f) == 0 && ok;
}
static bool load(const std::string &path, std::vector<float> &v) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return false; }
    long n = ftell(f);
    if (n <= 0 || n % sizeof(float) || n > 64 * 1024 * 1024) {
        fclose(f); return false;
    }
    rewind(f);
    v.resize(n / sizeof(float));
    bool ok = fread(v.data(), sizeof(float), v.size(), f) == v.size();
    return fclose(f) == 0 && ok;
}
static size_t nonfinite(const std::vector<float> &v) {
    return std::count_if(v.begin(), v.end(), [](float x) { return !std::isfinite(x); });
}
static bool tensor_result(const std::string &dir, const std::string &name,
                          const std::vector<float> &v, bool record) {
    size_t invalid = nonfinite(v);
    bool saved = save(dir + "/" + (record ? "" : "htp-") + name + ".f32", v);
    if (record) {
        printf("TENSOR {\"name\":\"%s\",\"mode\":\"record\",\"count\":%zu,"
               "\"nonfinite\":%zu,\"cosine\":null,\"nrmse\":null,\"saved\":%s,\"pass\":%s}\n",
               name.c_str(), v.size(), invalid, saved ? "true" : "false",
               saved && !invalid && !v.empty() ? "true" : "false");
        return saved && !invalid && !v.empty();
    }
    std::vector<float> ref;
    bool loaded = load(dir + "/" + name + ".f32", ref);
    size_t ref_invalid = nonfinite(ref);
    if (!loaded || v.empty() || v.size() != ref.size() || invalid || ref_invalid) {
        printf("TENSOR {\"name\":\"%s\",\"mode\":\"compare\",\"count\":%zu,"
               "\"reference_count\":%zu,\"nonfinite\":%zu,\"reference_nonfinite\":%zu,"
               "\"cosine\":null,\"nrmse\":null,\"saved\":%s,\"reference_loaded\":%s,\"pass\":false}\n",
               name.c_str(), v.size(), ref.size(), invalid, ref_invalid,
               saved ? "true" : "false", loaded ? "true" : "false");
        return false;
    }
    double dot = 0, aa = 0, bb = 0, error = 0, max_abs = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        double a = v[i], b = ref[i], d = a - b;
        dot += a * b; aa += a * a; bb += b * b; error += d * d;
        max_abs = std::max(max_abs, std::fabs(d));
    }
    double cosine = aa == 0 && bb == 0 ? 1 : dot / std::sqrt(std::max(aa * bb, 1.e-30));
    double nrmse = std::sqrt(error / std::max(bb, 1.e-30));
    // Coarse numerical gate only; synthetic codes cannot establish speech quality.
    bool pass = saved && cosine > .99 && nrmse < .15;
    printf("TENSOR {\"name\":\"%s\",\"mode\":\"compare\",\"count\":%zu,"
           "\"reference_count\":%zu,\"nonfinite\":0,\"reference_nonfinite\":0,"
           "\"cosine\":%.9g,\"nrmse\":%.9g,\"max_abs\":%.9g,\"saved\":%s,\"pass\":%s}\n",
           name.c_str(), v.size(), ref.size(), cosine, nrmse, max_abs,
           saved ? "true" : "false", pass ? "true" : "false");
    return pass;
}
static bool shape_result(const std::string &name, const std::vector<float> &prefill,
                         const std::vector<float> &trailing, const std::vector<float> &pad,
                         int hidden, size_t expected_prefill, size_t expected_trailing) {
    bool pass = prefill.size() == expected_prefill * hidden &&
                trailing.size() == expected_trailing * hidden && pad.size() == size_t(hidden);
    printf("SHAPE {\"name\":\"%s\",\"hidden_size\":%d,\"prefill_floats\":%zu,"
           "\"expected_prefill_positions\":%zu,\"trailing_floats\":%zu,"
           "\"expected_trailing_positions\":%zu,\"pad_floats\":%zu,\"pass\":%s}\n",
           name.c_str(), hidden, prefill.size(), expected_prefill, trailing.size(),
           expected_trailing, pad.size(), pass ? "true" : "false");
    return pass;
}
// Read only explicitly supplied real-input paths; never print text or input values.
static bool read_bytes(const char *path, std::vector<uint8_t> &bytes, size_t limit) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return false; }
    long n = ftell(f);
    if (n <= 0 || size_t(n) > limit) { fclose(f); return false; }
    rewind(f);
    bytes.resize(n);
    bool ok = fread(bytes.data(), 1, bytes.size(), f) == bytes.size();
    return fclose(f) == 0 && ok;
}
static int real_controls(qwen3_tts::TTSTransformer &transformer,
                         qwen3_tts::TextTokenizer &tokenizer, bool record,
                         const std::string &dir, char **argv) {
    const auto &cfg = transformer.get_config();
    const int hidden = cfg.hidden_size;
    if (hidden <= 0 || cfg.n_codebooks <= 0 || cfg.codec_vocab_size <= 0 ||
        cfg.code_pred_vocab_size <= 0) return 5;
    std::vector<uint8_t> bytes;
    if (!read_bytes(argv[4], bytes, 1024 * 1024) || bytes.size() < sizeof(uint32_t)) {
        fprintf(stderr, "Invalid real speaker file.\n"); return 5;
    }
    uint32_t speaker_count = 0;
    memcpy(&speaker_count, bytes.data(), sizeof(speaker_count));
    if (speaker_count != uint32_t(hidden) ||
        bytes.size() != sizeof(uint32_t) + size_t(hidden) * sizeof(float)) {
        fprintf(stderr, "Real speaker header must match model hidden_size.\n"); return 5;
    }
    std::vector<float> speaker(hidden);
    memcpy(speaker.data(), bytes.data() + sizeof(uint32_t), hidden * sizeof(float));
    if (nonfinite(speaker)) {
        fprintf(stderr, "Real speaker contains non-finite values.\n"); return 5;
    }
    if (!read_bytes(argv[5], bytes, 4096 * size_t(cfg.n_codebooks) * sizeof(int32_t)) ||
        bytes.size() % (size_t(cfg.n_codebooks) * sizeof(int32_t))) {
        fprintf(stderr, "Real reference codes must be nonempty raw frame-major i32.\n"); return 5;
    }
    std::vector<int32_t> codes(bytes.size() / sizeof(int32_t));
    memcpy(codes.data(), bytes.data(), bytes.size());
    const int frames = codes.size() / cfg.n_codebooks;
    for (size_t i = 0; i < codes.size(); ++i) {
        int vocab = i % cfg.n_codebooks ? cfg.code_pred_vocab_size : cfg.codec_vocab_size;
        if (codes[i] < 0 || codes[i] >= vocab) {
            fprintf(stderr, "Real reference codec ID outside codebook vocabulary.\n"); return 5;
        }
    }
    if (!read_bytes(argv[6], bytes, 1024 * 1024) ||
        std::find(bytes.begin(), bytes.end(), uint8_t(0)) != bytes.end()) {
        fprintf(stderr, "Invalid real reference text file.\n"); return 5;
    }
    std::string reference_text(bytes.begin(), bytes.end());
    if (!read_bytes(argv[7], bytes, 1024 * 1024) ||
        std::find(bytes.begin(), bytes.end(), uint8_t(0)) != bytes.end()) {
        fprintf(stderr, "Invalid real target text file.\n"); return 5;
    }
    std::string target_text(bytes.begin(), bytes.end());
    auto target = tokenizer.encode_for_tts(target_text);
    auto reference = tokenizer.encode_ref_text_body(reference_text);
    if (target.size() < 9 || reference.empty()) return 5;
    std::vector<float> base, base_trailing, base_pad;
    if (!transformer.build_prefill_graph(target.data(), target.size(), speaker.data(), 2050,
                                          base, base_trailing, base_pad, false) ||
        base.empty() || base.size() % hidden) return 6;
    const size_t framing = base.size() / hidden - 1;
    const size_t text_len = reference.size() + target.size() - 8 + 1;
    printf("REAL_INPUT {\"mode\":\"%s\",\"speaker_count\":%zu,\"reference_frames\":%d,"
           "\"target_tokens\":%zu,\"reference_body_tokens\":%zu,\"framing_positions\":%zu}\n",
           record ? "record" : "compare", speaker.size(), frames, target.size(),
           reference.size(), framing);
    std::vector<float> prefill, trailing, pad;
    if (!transformer.build_prefill_graph_icl(target.data(), target.size(), speaker.data(), 2050,
            reference.data(), reference.size(), codes.data(), frames,
            prefill, trailing, pad, false)) {
        fprintf(stderr, "Real ICL assembly: %s\n", transformer.get_error().c_str()); return 6;
    }
    bool passed = shape_result("real-stream", prefill, trailing, pad, hidden,
                                framing + frames + 1,
                                text_len > size_t(frames + 1) ? text_len - frames - 1 : 1);
    passed = tensor_result(dir, "real-prefill", prefill, record) && passed;
    passed = tensor_result(dir, "real-trailing", trailing, record) && passed;
    passed = tensor_result(dir, "real-pad", pad, record) && passed;
    std::vector<float> frozen;
    if (!load(dir + "/real-prefill.f32", frozen) || frozen.empty() || frozen.size() % hidden) {
        fprintf(stderr, "Missing or invalid frozen CPU real prefill.\n"); return 7;
    }
    const size_t length = frozen.size() / hidden;
    if (length > 4096 || !transformer.init_kv_cache(length + 72)) return 7;
    transformer.clear_kv_cache();
    std::vector<float> output, logits;
    if (!transformer.forward_prefill(frozen.data(), length, 0, output, &logits)) {
        fprintf(stderr, "Real frozen forward: %s\n", transformer.get_error().c_str());
        printf("REAL_FORWARD {\"pass\":false}\n");
        passed = false;
    } else {
        bool shape = output.size() == length * hidden && logits.size() == size_t(cfg.codec_vocab_size);
        printf("REAL_FORWARD {\"positions\":%zu,\"hidden_floats\":%zu,\"logit_count\":%zu,"
               "\"shape_pass\":%s}\n", length, output.size(), logits.size(), shape ? "true" : "false");
        passed = shape && passed;
        passed = tensor_result(dir, "real-forward-hidden", output, record) && passed;
        passed = tensor_result(dir, "real-forward-logits", logits, record) && passed;
    }
    const char *generate = getenv("QWEN3_TTS_CLONE_GENERATE");
    if (generate && !strcmp(generate, "1")) {
        transformer.set_seed(20261026);
        std::vector<int32_t> generated;
        bool ok = transformer.generate_icl(target.data(), target.size(), speaker.data(),
            reference.data(), reference.size(), codes.data(), frames, 64, generated,
            2050, 1.05f, .9f, 50, 1.0f, .9f, 50, false, 1.0f);
        std::string path = dir + (record ? "/real-codes.i32" : "/htp-real-codes.i32");
        FILE *f = fopen(path.c_str(), "wb");
        bool saved = false;
        if (f) {
            saved = fwrite(generated.data(), sizeof(int32_t), generated.size(), f) == generated.size();
            saved = fclose(f) == 0 && saved;
        }
        bool valid = !generated.empty() && generated.size() % cfg.n_codebooks == 0 &&
                     generated.size() <= 64 * size_t(cfg.n_codebooks);
        std::unordered_set<int32_t> all_unique, cb0_unique;
        for (size_t i = 0; i < generated.size(); ++i) {
            int vocab = i % cfg.n_codebooks ? cfg.code_pred_vocab_size : cfg.codec_vocab_size;
            valid = generated[i] >= 0 && generated[i] < vocab && valid;
            all_unique.insert(generated[i]);
            if (i % cfg.n_codebooks == 0) cb0_unique.insert(generated[i]);
        }
        if (!ok) fprintf(stderr, "Real sampled generation: %s\n", transformer.get_error().c_str());
        printf("REAL_GENERATION {\"seed\":20261026,\"temperature\":0.9,\"max_frames\":64,"
               "\"frames\":%zu,\"code_count\":%zu,\"distinct_codes\":%zu,\"distinct_cb0\":%zu,"
               "\"saved\":%s,\"valid_codes\":%s,\"parity_checked\":false,\"pass\":%s}\n",
               generated.size() / cfg.n_codebooks, generated.size(), all_unique.size(), cb0_unique.size(),
               saved ? "true" : "false", valid ? "true" : "false",
               ok && saved && valid ? "true" : "false");
        passed = ok && saved && valid && passed;
    }
    printf("SUMMARY {\"mode\":\"%s\",\"inputs\":\"real\",\"pass\":%s,\"speech_quality_tested\":false}\n",
           record ? "record" : "compare", passed ? "true" : "false");
    return passed ? 0 : 11;
}
int main(int argc, char **argv) {
    if ((argc != 4 && argc != 8) || (strcmp(argv[1], "record") && strcmp(argv[1], "compare"))) {
        fprintf(stderr, "Usage: %s {record|compare} MODEL FIXTURE_DIR [SPEAKER.bin REF_CODES.i32 REF_TEXT.txt TARGET.txt]\n"
                        "Set QWEN3_TTS_CLONE_GENERATE=1 for optional real-input 64-frame generation.\n", argv[0]);
        return 2;
    }
    bool record = !strcmp(argv[1], "record");
    const std::string dir = argv[3];
    if (record && mkdir(dir.c_str(), 0700) && errno != EEXIST) return 3;
    struct stat directory = {};
    if (stat(dir.c_str(), &directory) || !S_ISDIR(directory.st_mode) || (directory.st_mode & 0077)) {
        fprintf(stderr, "FIXTURE_DIR must be a private directory (mode 0700).\n");
        return 3;
    }
    unsetenv("QWEN3_TTS_FORCE_CPU");
    setenv("QWEN3_TTS_TRANSFORMER_BACKEND", record ? "cpu" : "hexagon", 1);
    setenv("QWEN3_TTS_VALIDATE_LOGITS", "1", 1);
    qwen3_tts::TTSTransformer transformer;
    if (!transformer.load_model(argv[2])) {
        fprintf(stderr, "%s\n", transformer.get_error().c_str()); return 4;
    }
    transformer.set_n_threads(2);
    qwen3_tts::GGUFLoader loader;
    qwen3_tts::TextTokenizer tokenizer;
    if (!loader.open(argv[2]) || !tokenizer.load_from_gguf(loader.get_ctx())) return 5;
    if (argc == 8) return real_controls(transformer, tokenizer, record, dir, argv);
    auto target = tokenizer.encode_for_tts("This public target sentence checks deterministic prompt assembly.");
    auto reference = tokenizer.encode_ref_text_body("A public reference sentence for arithmetic validation.");
    if (target.size() < 9 || reference.empty()) return 5;
    const auto &cfg = transformer.get_config();
    const int hidden = cfg.hidden_size;
    const int code_range = std::min(2048, std::min(cfg.code_pred_vocab_size, cfg.codec_vocab_size));
    if (hidden <= 0 || cfg.n_codebooks <= 0 || code_range <= 0) return 5;
    std::vector<float> speaker(hidden, 0.0f), base, base_trailing, base_pad;
    if (!transformer.build_prefill_graph(target.data(), target.size(), speaker.data(), 2050,
                                          base, base_trailing, base_pad, false)) return 6;
    if (base.empty() || base.size() % hidden) return 6;
    const size_t framing = base.size() / hidden - 1;
    const size_t target_body = target.size() - 8;
    const size_t text_len = reference.size() + target_body + 1;
    printf("INPUT {\"mode\":\"%s\",\"target_tokens\":%zu,\"target_body_tokens\":%zu,"
           "\"reference_body_tokens\":%zu,\"framing_positions\":%zu,\"speaker\":\"zero\","
           "\"codebooks\":%d,\"synthetic_codes\":true}\n",
           record ? "record" : "compare", target.size(), target_body,
           reference.size(), framing, cfg.n_codebooks);
    struct Case { const char *name; int frames; bool non_streaming; };
    const Case cases[] = {{"stream2", 2, false}, {"stream16", 16, false},
                          {"stream32", 32, false}, {"stream48", 48, false},
                          {"stream64", 64, false}, {"stream206", 206, false},
                          {"nonstream32", 32, true}, {"nonstream-fallback", 0, true}};
    bool passed = true;
    for (const auto &test : cases) {
        std::vector<int32_t> codes(size_t(test.frames) * cfg.n_codebooks);
        for (int f = 0; f < test.frames; ++f)
            for (int cb = 0; cb < cfg.n_codebooks; ++cb)
                codes[size_t(f) * cfg.n_codebooks + cb] = (17 + f * 37 + cb * 101) % code_range;
        std::vector<float> prefill, trailing, pad;
        bool built = transformer.build_prefill_graph_icl(
            target.data(), target.size(), speaker.data(), 2050,
            test.frames ? reference.data() : nullptr, test.frames ? reference.size() : 0,
            codes.empty() ? nullptr : codes.data(), test.frames, prefill, trailing, pad,
            test.non_streaming);
        if (!built) {
            fprintf(stderr, "%s: %s\n", test.name, transformer.get_error().c_str());
            printf("BUILD {\"name\":\"%s\",\"pass\":false}\n", test.name);
            passed = false; continue;
        }
        size_t codec_len = 1 + test.frames;
        size_t expected_prefill = framing + (test.non_streaming ? text_len + codec_len : codec_len);
        size_t expected_trailing = !test.non_streaming && text_len > codec_len ? text_len - codec_len : 1;
        if (!test.frames) expected_prefill = framing + target_body + 1 + 1; // body + EOS + BOS
        passed = shape_result(test.name, prefill, trailing, pad, hidden,
                              expected_prefill, expected_trailing) && passed;
        // Retain all independent tensors even when expected-position checks fail.
        passed = tensor_result(dir, std::string(test.name) + "-prefill", prefill, record) && passed;
        passed = tensor_result(dir, std::string(test.name) + "-trailing", trailing, record) && passed;
        passed = tensor_result(dir, std::string(test.name) + "-pad", pad, record) && passed;
    }
    // Both backends consume the frozen CPU-built long streaming fixture exactly.
    std::vector<float> frozen;
    if (!load(dir + "/stream206-prefill.f32", frozen) || frozen.empty() || frozen.size() % hidden) {
        fprintf(stderr, "Missing or invalid frozen CPU stream206 prefill.\n"); return 7;
    }
    const size_t full_length = frozen.size() / hidden;
    if (full_length > 480 || full_length < 49 || !transformer.init_kv_cache(512)) return 7;
    const size_t lengths[] = {16, 32, 48, 49, full_length};
    for (size_t length : lengths) {
        transformer.clear_kv_cache();
        std::vector<float> output, logits;
        std::string name = "forward-" + std::to_string(length);
        if (!transformer.forward_prefill(frozen.data(), length, 0, output, &logits)) {
            fprintf(stderr, "%s: %s\n", name.c_str(), transformer.get_error().c_str());
            printf("FORWARD {\"name\":\"%s\",\"pass\":false}\n", name.c_str());
            passed = false; continue;
        }
        bool shape = output.size() == length * hidden && logits.size() == size_t(cfg.codec_vocab_size);
        printf("FORWARD {\"name\":\"%s\",\"positions\":%zu,\"hidden_floats\":%zu,"
               "\"logit_count\":%zu,\"shape_pass\":%s}\n", name.c_str(), length,
               output.size(), logits.size(), shape ? "true" : "false");
        passed = shape && passed;
        passed = tensor_result(dir, name + "-hidden", output, record) && passed;
        passed = tensor_result(dir, name + "-logits", logits, record) && passed;
    }
    printf("SUMMARY {\"mode\":\"%s\",\"pass\":%s,\"speech_quality_tested\":false}\n",
           record ? "record" : "compare", passed ? "true" : "false");
    return passed ? 0 : 11;
}
