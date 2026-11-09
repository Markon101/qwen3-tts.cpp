// A cached compact placement must never bypass a later strict placement check.
#include "gguf_loader.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace qwen3_tts;
int main(int argc, char **argv) {
    if (argc != 3 || (strcmp(argv[1],"summary") && strcmp(argv[1],"full"))) return 2;
    setenv("QWEN3_TTS_EXECUTION_TRACE_DETAIL",argv[1],1);
    setenv("QWEN3_TTS_EXECUTION_TRACE",argv[2],1);
    ggml_backend_t backend=ggml_backend_cpu_init();
    if (!backend) return 3;
    ggml_backend_t backends[]={backend};
    ggml_backend_sched_t sched=ggml_backend_sched_new(backends,nullptr,1,64,false,true);
    if (!sched) return 3;
    bool passed=true;
    int rejections=0;
    for (int i=0;i<4;++i) {
        const int width=i%2?7:5;
        ggml_init_params params={ggml_tensor_overhead()*8+ggml_graph_overhead_custom(64,false),nullptr,true};
        ggml_context *ctx=ggml_init(params);
        auto *a=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,width);
        auto *b=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,width);
        ggml_set_input(a); ggml_set_input(b);
        auto *sum=ggml_add(ctx,a,b); ggml_set_output(sum);
        auto *graph=ggml_new_graph_custom(ctx,64,false);
        ggml_build_forward_expand(graph,sum);
        if (!ggml_backend_sched_alloc_graph(sched,graph)) return 4;
        std::vector<float> input(width,1.f), output(width);
        ggml_backend_tensor_set(a,input.data(),0,width*sizeof(float));
        ggml_backend_tensor_set(b,input.data(),0,width*sizeof(float));
        setenv("QWEN3_TTS_MAX_STRICT","0",1);
        std::string error;
        passed=audited_graph_compute("trace_control",sched,graph,&error)==GGML_STATUS_SUCCESS&&passed;
        ggml_backend_tensor_get(sum,output.data(),0,width*sizeof(float));
        for (float v:output) passed=passed&&v==2.f;
        // Same graph and placement signature, but the policy changed: reject.
        setenv("QWEN3_TTS_MAX_STRICT","1",1);
        const bool rejected=audited_graph_compute("trace_control",sched,graph,&error)!=GGML_STATUS_SUCCESS;
        rejections+=rejected;
        passed=passed&&rejected&&!error.empty();
        ggml_backend_sched_reset(sched); ggml_free(ctx);
    }
    ggml_backend_sched_free(sched); ggml_backend_free(backend);
    printf("{\"mode\":\"%s\",\"strict_rejections\":%d,\"pass\":%s}\n",argv[1],rejections,passed?"true":"false");
    return passed?0:5;
}
