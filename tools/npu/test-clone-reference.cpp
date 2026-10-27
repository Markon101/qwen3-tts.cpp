// Inspect reference conditioning independently of TTS generation/decoding.
#include "audio_tokenizer_encoder.h"
#include "mimi_encoder.h"
#include "audio_tokenizer_decoder.h"
#include "gguf_loader.h"
#include "qwen3_tts.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>
int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s speaker|codec|decode MODEL_OR_TOKENIZER AUDIO OUT_PREFIX\n", argv[0]);
        return 2;
    }
    if (!strcmp(argv[1], "decode")) {
        FILE *f=fopen(argv[3],"rb"); if(!f) return 3;
        fseek(f,0,SEEK_END);long bytes=ftell(f);rewind(f);
        if(bytes<=0 || bytes%64 || bytes>16*1024*1024) { fclose(f);return 3; }
        std::vector<int32_t> codes(bytes/4);
        bool ok=fread(codes.data(),4,codes.size(),f)==codes.size();fclose(f);if(!ok)return 3;
        qwen3_tts::AudioTokenizerDecoder decoder;
        if(!decoder.load_model(argv[2])) {fprintf(stderr,"%s\n",decoder.get_error().c_str());return 4;}
        qwen3_tts::GGUFLoader loader;
        if(loader.open(argv[2])) for(int64_t i=0;i<loader.get_n_tensors();++i) {
            const char *name=loader.get_tensor_name(i);
            if(strstr(name,"upsample") && strstr(name,"conv.weight")) {
                ggml_tensor *t=ggml_get_tensor(loader.get_meta_ctx(),name);
                fprintf(stderr,"KERNEL %s %lld,%lld,%lld\n",name,(long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2]);
            }
        }
        int frames=codes.size()/16;std::vector<float> audio;
        for(int base=0;base<frames;base+=100) {
            int n=std::min(100,frames-base);std::vector<float> chunk;
            if(!decoder.decode(codes.data()+base*16,n,chunk)) {fprintf(stderr,"%s\n",decoder.get_error().c_str());return 4;}
            for(float x:chunk)if(!std::isfinite(x))return 6;
            audio.insert(audio.end(),chunk.begin(),chunk.end());
        }
        f=fopen((std::string(argv[4])+".f32").c_str(),"wb");if(!f)return 5;
        ok=fwrite(audio.data(),4,audio.size(),f)==audio.size();fclose(f);if(!ok)return 5;
        if(!qwen3_tts::save_audio_file(std::string(argv[4])+".wav",audio,24000))return 5;
        printf("RESULT {\"kind\":\"decode\",\"frames\":%d,\"samples\":%zu,\"expected_samples\":%d,\"nonfinite\":0}\n",frames,audio.size(),frames*1920);
        return 0;
    }
    std::vector<float> input, samples;
    int rate = 0;
    if (!qwen3_tts::load_audio_file(argv[3], input, rate)) return 3;
    if (rate == 24000) samples = std::move(input);
    else qwen3_tts::resample_audio(input.data(), input.size(), rate, 24000, samples);
    for (float x : samples) if (!std::isfinite(x)) return 3;
    fprintf(stderr, "REFERENCE samples=%zu rate=24000\n", samples.size());
    if (!strcmp(argv[1], "speaker")) {
        qwen3_tts::AudioTokenizerEncoder encoder;
        std::vector<float> embedding;
        if (!encoder.load_model(argv[2]) || !encoder.encode(samples.data(), samples.size(), embedding)) {
            fprintf(stderr, "ERROR %s\n", encoder.get_error().c_str()); return 4;
        }
        double energy=0; float mx=0; int bad=0;
        for (float x : embedding) { bad+=!std::isfinite(x); energy+=double(x)*x; mx=std::max(mx,std::fabs(x)); }
        if (!qwen3_tts::save_speaker_embedding(std::string(argv[4])+".bin", embedding)) return 5;
        printf("RESULT {\"kind\":\"speaker\",\"count\":%zu,\"nonfinite\":%d,\"l2_norm\":%.9g,\"max_abs\":%.9g}\n", embedding.size(),bad,std::sqrt(energy),mx);
        return bad || energy==0 ? 6 : 0;
    }
    if (!strcmp(argv[1], "codec")) {
        qwen3_tts::MimiEncoder encoder;
        std::vector<int32_t> codes; int frames=0;
        if (!encoder.load_model(argv[2]) || !encoder.encode(samples.data(),samples.size(),codes,frames)) {
            fprintf(stderr, "ERROR %s\n",encoder.get_error().c_str()); return 4;
        }
        FILE *f=fopen((std::string(argv[4])+".i32").c_str(),"wb");
        if (!f) return 5;
        bool ok=fwrite(codes.data(),4,codes.size(),f)==codes.size();fclose(f);
        if (!ok) return 5;
        std::set<int32_t> unique(codes.begin(),codes.end());
        printf("RESULT {\"kind\":\"codec\",\"frames\":%d,\"count\":%zu,\"unique\":%zu,\"min\":%d,\"max\":%d}\n",frames,codes.size(),unique.size(),codes.empty()?0:*unique.begin(),codes.empty()?0:*unique.rbegin());
        return frames>0 && codes.size()==size_t(frames)*16 ? 0:6;
    }
    return 2;
}
