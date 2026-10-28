// Isolated processes own separate HTP sessions. Never share a mutable session
// concurrently between generator and decoder threads.
#include "qwen3_tts.h"
#include "tts_transformer.h"
#include "text_tokenizer.h"
#include "audio_tokenizer_decoder.h"
#include "gguf_loader.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace qwen3_tts;
int main(int argc, char **argv) {
    if (argc == 7 && !strcmp(argv[1], "generate")) {
        const int frames = atoi(argv[6]);
        if (frames < 1 || frames > 2048) return 2;
        GGUFLoader loader; TextTokenizer tokenizer; TTSTransformer transformer;
        if (!loader.open(argv[2]) || !tokenizer.load_from_gguf(loader.get_ctx()) || !transformer.load_model(argv[2])) return 3;
        std::vector<float> speaker;
        if (!load_speaker_embedding(argv[3], speaker) || speaker.size() != size_t(transformer.get_config().hidden_size)) return 4;
        for (float x : speaker) if (!std::isfinite(x)) return 4;
        auto tokens = tokenizer.encode_for_tts(argv[4]);
        transformer.set_seed(20261026); transformer.set_n_threads(2);
        transformer.set_frame_callback([](const int32_t *row, int32_t n) {
            return n == 16 && fwrite(row, sizeof(int32_t), n, stdout) == size_t(n) && fflush(stdout) == 0;
        });
        std::vector<int32_t> codes;
        if (!transformer.generate(tokens.data(), tokens.size(), speaker.data(), frames, codes)) {
            fprintf(stderr, "%s\n", transformer.get_error().c_str()); return 5;
        }
        FILE *f = fopen(argv[5], "wb");
        if (!f) return 6;
        bool ok = fwrite(codes.data(), 4, codes.size(), f) == codes.size(); fclose(f);
        return ok ? 0 : 6;
    }
    if (argc == 4 && !strcmp(argv[1], "decode")) {
        int chunk = atoi(argv[3]);
        if (chunk < 1 || chunk > 100) return 2;
        AudioTokenizerDecoder decoder;
        if (!decoder.load_model(argv[2])) { fprintf(stderr, "%s\n",decoder.get_error().c_str()); return 3; }
        decoder.set_n_threads(2);
        std::vector<int32_t> codes;
        int emitted = 0, total = 0;
        auto flush = [&]() {
            int frames = codes.size()/16;
            if (frames == emitted) return true;
            std::vector<float> audio;
            if (!decoder.decode(codes.data(), frames, audio) || audio.size() != size_t(frames)*1920) return false;
            for (float x : audio) if (!std::isfinite(x)) return false;
            size_t skip = size_t(emitted)*1920;
            if (fwrite(audio.data()+skip,4,audio.size()-skip,stdout) != audio.size()-skip || fflush(stdout)) return false;
            fprintf(stderr,"PIPELINE_CHUNK frames=%d prefix_frames=%d\n",frames-emitted,frames);
            emitted=frames;
            if (frames == 100) { codes.clear(); emitted=0; }
            return true;
        };
        int32_t row[16];
        while (true) {
            size_t n = fread(row,4,16,stdin);
            if (!n) break;
            if (n != 16) return 4;
            for (int32_t code : row) if (code < 0 || code > 2047) return 4;
            codes.insert(codes.end(),row,row+16);++total;
            int frames = codes.size()/16;
            if ((frames-emitted >= chunk || frames==100) && !flush()) return 5;
        }
        if (ferror(stdin) || !total || !flush()) return 5;
        return 0;
    }
    fprintf(stderr,"Usage: pipeline-stage generate MODEL EMBEDDING TEXT CODES_OUT MAX_FRAMES\n       pipeline-stage decode TOKENIZER CHUNK_FRAMES\n");
    return 2;
}
