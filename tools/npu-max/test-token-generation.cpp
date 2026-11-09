// Fixed public inputs isolate predictor arithmetic from talker/speaker variation.
// Capture runs are correctness controls; only uninstrumented runs are timed.
#include "tts_transformer.h"
#include "text_tokenizer.h"
#include "gguf_loader.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
using namespace qwen3_tts;
namespace {
using Clock = std::chrono::steady_clock;
struct Capture {
    std::vector<std::vector<float>> logits;
    const std::vector<int32_t> * feedback = nullptr;
    bool valid = true;
};
bool capture(ggml_tensor *t, bool ask, void *data) {
    if (ask) return t->type == GGML_TYPE_F32 && !strcmp(t->name, "logits");
    auto &c = *static_cast<Capture *>(data);
    std::vector<float> values(ggml_nelements(t));
    ggml_backend_tensor_get(t, values.data(), 0, values.size() * sizeof(float));
    for (float x : values) c.valid = c.valid && std::isfinite(x);
    const size_t step = c.logits.size();
    c.logits.push_back(values);
    // Save raw logits first, then force the baseline feedback. This checks all
    // heads on identical histories even if small arithmetic differences sample
    // a different token. This callback is never installed in timing runs.
    if (c.feedback) {
        if (step >= c.feedback->size() || (*c.feedback)[step] < 0 ||
            size_t((*c.feedback)[step]) >= values.size()) return false;
        std::fill(values.begin(), values.end(), -1000.f);
        values[(*c.feedback)[step]] = 1000.f;
        ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(float));
    }
    return true;
}
void fusion(bool enabled) {
    const char * variant = getenv("QWEN_TOKEN_GEN_CANDIDATE");
    const bool swiglu = !variant || !strcmp(variant, "swiglu") || !strcmp(variant, "both");
    if (variant && (!strcmp(variant, "trace-summary") || !strcmp(variant, "trace-off"))) {
        const char * path = getenv("QWEN_TOKEN_GEN_TRACE_PATH");
        if (enabled && !strcmp(variant, "trace-off")) unsetenv("QWEN3_TTS_EXECUTION_TRACE");
        else if (path) setenv("QWEN3_TTS_EXECUTION_TRACE", path, 1);
        setenv("QWEN3_TTS_EXECUTION_TRACE_DETAIL", enabled ? "summary" : "full", 1);
    }
    const bool strided = !variant || !strcmp(variant, "strided-kv") || !strcmp(variant, "both");
    setenv("QWEN3_TTS_CODE_PRED_SWIGLU", enabled && swiglu ? "1" : "0", 1);
    setenv("QWEN3_TTS_CODE_PRED_STRIDED_KV", enabled && strided ? "1" : "0", 1);
}
bool foreground() {
    std::ifstream f("/proc/self/cgroup");
    std::string line;
    while (std::getline(f, line)) {
        auto a = line.find(':');
        auto b = line.find(':', a == std::string::npos ? 0 : a + 1);
        if (a != std::string::npos && b != std::string::npos &&
            line.substr(a + 1, b - a - 1) == "cpuset")
            return line.substr(b + 1) == "/top-app";
    }
    return false;
}
bool save(const std::string &path, const std::vector<float> &v) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(v.data(), sizeof(float), v.size(), f) == v.size();
    return fclose(f) == 0 && ok;
}
bool compare(const Capture &a, const Capture &b, const char *label, int test) {
    if (!a.valid || !b.valid || a.logits.size() != 15 || b.logits.size() != 15) return false;
    bool all = true;
    for (size_t head = 0; head < 15; ++head) {
        const auto &x = a.logits[head]; const auto &y = b.logits[head];
        if (x.empty() || x.size() != y.size()) return false;
        double aa=0, bb=0, ab=0, err=0, mx=0;
        for (size_t i=0; i<x.size(); ++i) {
            double d=double(x[i])-y[i];
            aa+=double(x[i])*x[i]; bb+=double(y[i])*y[i];
            ab+=double(x[i])*y[i]; err+=d*d; mx=std::max(mx,std::fabs(d));
        }
        const double cosine=ab/std::sqrt(std::max(aa*bb,1e-30));
        const double nrmse=std::sqrt(err/std::max(aa,1e-30));
        const bool exact=!memcmp(x.data(),y.data(),x.size()*sizeof(float));
        // Predeclared same-backend gate; much tighter than cross-backend gates.
        const bool pass=cosine>.99999 && nrmse<.001 && mx<.05;
        printf("{\"event\":\"head\",\"test\":%d,\"comparison\":\"%s\",\"head\":%zu,\"cosine\":%.12g,\"nrmse\":%.12g,\"max_abs\":%.12g,\"bit_exact\":%s,\"pass\":%s}\n",
               test,label,head,cosine,nrmse,mx,exact?"true":"false",pass?"true":"false");
        all=all&&pass;
    }
    return all;
}
}
int main(int argc, char **argv) {
    if (argc != 6 || (strcmp(argv[1],"check") && strcmp(argv[1],"bench") && strcmp(argv[1],"generate")) ||
        (strcmp(argv[3],"cpu") && strcmp(argv[3],"opencl") && strcmp(argv[3],"hexagon"))) {
        fprintf(stderr,"Usage: %s check|bench|generate MODEL cpu|opencl|hexagon OUTPUT_DIR CALLS_PER_BLOCK\n",argv[0]); return 2;
    }
    const bool integrated=!strcmp(argv[1],"generate");
    const bool bench=strcmp(argv[1],"check") != 0;
    char *end=nullptr; long calls=strtol(argv[5],&end,10);
    if (!*argv[5] || *end || calls<1 || calls>128) return 2;
    if (bench && !foreground()) { fprintf(stderr,"Requires cpuset /top-app\n"); return 3; }
    unsetenv("QWEN3_TTS_FORCE_CPU");
    setenv("QWEN3_TTS_TRANSFORMER_BACKEND",argv[3],1);
    setenv("QWEN3_TTS_VALIDATE_LOGITS","1",1);
    TTSTransformer t;
    if (!t.load_model(argv[2])) { fprintf(stderr,"%s\n",t.get_error().c_str()); return 4; }
    t.set_n_threads(2);
    const auto &cfg=t.get_config();
    printf("{\"event\":\"config\",\"hidden_size\":%d,\"predictor_hidden_size\":%d,\"layers\":%d,\"backend\":\"%s\"}\n",
           cfg.hidden_size,cfg.code_pred_hidden_size,cfg.code_pred_layers,argv[3]);
    // Include a talker graph before predictor reuse and use two distinct slots.
    if (!t.init_kv_cache(32,2)) return 5;
    std::vector<float> hidden(cfg.hidden_size), input(2*cfg.hidden_size), out;
    for (size_t i=0;i<input.size();++i) input[i]=.2f*std::sin(float(i)*.017f);
    if (!t.forward_prefill(input.data(),2,0,out)) return 6;
    std::copy(out.end()-cfg.hidden_size,out.end(),hidden.begin());
    if (!save(std::string(argv[4])+"/hidden.f32",hidden)) return 7;
    GGUFLoader loader; TextTokenizer tokenizer;
    std::vector<int32_t> tokens;
    std::vector<float> speaker(cfg.hidden_size, 0.f);
    if (integrated) {
        if (!loader.open(argv[2]) || !tokenizer.load_from_gguf(loader.get_ctx())) return 4;
        tokens=tokenizer.encode_for_tts("The little fox wandered through the quiet forest, following the silver river toward the distant mountains. Along the way it listened to the birds and watched sunlight dancing through the leaves.");
    }
    auto predict=[&](bool fused, int cb0, float temp, float p, int slot, Capture *c,
                     std::vector<int32_t> &codes) {
        fusion(fused); t.set_seed(20261109);
        if (integrated) {
            const bool ok=t.generate(tokens.data(),tokens.size(),speaker.data(),32,codes);
            return ok && codes.size()==32*16;
        }
        if (c) t.set_eval_callback(capture,c);
        bool ok=t.predict_codes_autoregressive(hidden.data(),cb0,codes,temp,50,p,slot);
        t.clear_eval_callback();
        if (!ok) fprintf(stderr,"%s\n",t.get_error().c_str());
        if (codes.size()!=15) return false;
        for (auto code:codes) if (code<0 || code>=cfg.code_pred_vocab_size) return false;
        return ok;
    };
    bool all=true;
    if (!bench) {
        for (int test=0;test<6;++test) {
            const float temp=test%3==0?0.f:.9f, p=test%3==2?.9f:1.f;
            const int cb0=test%3==0?0:test%3==1?1995:2047, slot=test/3;
            Capture a,b,forced,replay; std::vector<int32_t> ca,cb,cf,cr;
            if (!predict(false,cb0,temp,p,slot,&a,ca) || !predict(true,cb0,temp,p,slot,&b,cb)) return 8;
            forced.feedback=&ca;
            if (!predict(true,cb0,0.f,1.f,slot,&forced,cf) || !predict(false,cb0,temp,p,slot,&replay,cr)) return 8;
            all=compare(a,forced,"teacher_forced_candidate",test)&&all;
            all=compare(a,replay,"baseline_replay",test)&&all;
            const bool same=ca==cb, repeat=ca==cr, forced_ok=ca==cf;
            for (size_t head=0;head<a.logits.size();++head) {
                const std::string name=std::string(argv[4])+"/case-"+std::to_string(test)+"-head-"+std::to_string(head);
                if (!save(name+"-baseline.f32",a.logits[head]) || !save(name+"-candidate.f32",forced.logits[head])) return 7;
            }
            printf("{\"event\":\"case\",\"test\":%d,\"slot\":%d,\"temperature\":%.2f,\"top_p\":%.2f,\"same_sampled_codes\":%s,\"baseline_repeat\":%s,\"feedback_forced\":%s}\n",
                   test,slot,temp,p,same?"true":"false",repeat?"true":"false",forced_ok?"true":"false");
            all=all&&same&&repeat&&forced_ok;
        }
    } else {
        // Fixed input and CB0 per call; seeded sampling reset every time. Three
        // excluded warmups per variant, then serial A/B/A on resident weights.
        for (int block=0;block<3;++block) {
            bool fused=block==1; std::vector<int32_t> codes;
            for (int j=0;j<(integrated?1:3);++j) {
                if (!foreground() || !predict(fused,1995,.9f,.9f,0,nullptr,codes) || !foreground()) return 9;
            }
            for (int j=0;j<calls;++j) {
                if (!foreground()) return 9;
                auto start=Clock::now();
                if (!predict(fused,1995,.9f,.9f,0,nullptr,codes)) return 8;
                double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
                if (!foreground()) return 9;
                printf("{\"event\":\"timing\",\"block\":%d,\"candidate_enabled\":%s,\"call\":%d,\"elapsed_ms\":%.6f,\"codes\":[",block,fused?"true":"false",j,ms);
                for (size_t k=0;k<codes.size();++k) printf("%s%d",k?",":"",codes[k]);
                printf("]}\n"); fflush(stdout);
            }
        }
    }
    printf("{\"event\":\"complete\",\"pass\":%s,\"mode\":\"%s\"}\n",all?"true":"false",argv[1]);
    return all?0:10;
}
