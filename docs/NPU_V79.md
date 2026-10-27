# Native Hexagon v79 / Termux experiment

The S25 Ultra now runs source-built GGML Hexagon v79 arithmetic and Qwen3-TTS generation through the Android vendor FastRPC broker, without root. The transformer uses HTP0 and the vocoder uses Adreno OpenCL. Initial finite WAVs concealed non-finite predictor logits and collapsed speech codes. Those timings were rejected. This branch therefore validates raw logits as well as audio and retains the failed controls. Read the dated report for the current gate outcome; listening quality and energy efficiency are unvalidated.

The stable `snapdragon-adreno830` branch remains the default. This experiment is on `npu-v79-bringup`, with its own pinned GGML branch. No model, voice, SDK, vendor library or compiled DSP binary is published.

## What made it work

Direct opens of the CDSP device nodes failed with EACCES, and NNAPI exposed only its CPU reference device. Those observations did not establish that NPU access was blocked. The working path uses Android's exported `sphal` namespace to load the phone's `/vendor/lib64/libcdsprpc.so`, then native FastRPC shared memory and DSP queues.

The capability probe reported architecture `0x8c79`, unsigned PD support, six HVX128 units and one 8 MiB VTCM page. These are vendor capability values; the HMX-related zero values are not interpreted as absence of HMX. Actual source-built matrix calculations, rather than enumeration, establish execution.

The host and DSP toolchain follow the [S24 Ultra Termux example](https://github.com/Ishabdullah/OpenCL-S24-Ultra/blob/main/docs/NPU_INSTALL.md), with MIT attribution included in both forks. That author's tested target is v75; this branch builds v79. x86 QEMU runs only SDK compiler/QAIC/linker tools. Host inference is native AArch64 and the DSP library is native Hexagon ELF32 with architecture flag `0x79`.

This legacy GGML revision defaults to `GGML_HEXAGON_HOSTBUF=1`. With the application's ordinary weight allocator, that left Q8 matrix operations on CPU: code-predictor and talker step graphs had 71 and 393 splits. Explicit `GGML_HEXAGON_HOSTBUF=0` enables backend-managed repacking and removes those matrix fallbacks: the reported step graphs each have one split (one metadata RESHAPE was listed as CPU). Weight set/get roundtrips and numerical controls were checked after the change. The hostbuf environment setting is local to this experiment; the backend's global default is preserved.

## Prepare and build

This workflow was tested in aarch64 Termux. It needs the normal native Clang/CMake/Ninja/Python/curl/dpkg tools and OpenCL development headers/loader. Native QEMU also needs Termux's glib, libandroid-shmem, libgnutls, libpixman, libbz2 and liblzma dependencies available. The preparation scripts extract the pinned QEMU/libdw/libelf/argp packages and the x86 Debian compiler runtimes into project-private `.cache/npu/`; they do not globally install packages. They fail if required pinned package bytes are unavailable or checksums differ.

```sh
git clone -b npu-v79-bringup --recurse-submodules https://github.com/Markon101/qwen3-tts.cpp.git
cd qwen3-tts.cpp
python tools/npu/prepare-runtime.py .
python tools/npu/prepare-toolchain.py .
sh tools/npu/build-host-dsp.sh
```

The SDK archive is about 694 MB and its extracted toolchain is several GiB; the exact SDK/runtime sizes and SHA-256 values are in `tools/npu/toolchain-pins.json`. Review the SDK source and applicable license before downloading it. SDK contents are local build dependencies and are not included in the Git forks. Building the DSP first and host second avoids the concurrent QAIC stall observed in an early attempt; its root cause was not established.

The build produces combined CPU/OpenCL/Hexagon host libraries, one v79 DSP library, the TTS CLI, and optional native numerical tools. Building does not launch a hardware benchmark.

## Run and verify

Use existing external model assets with the same names as the [OpenCL profile](ANDROID_ADRENO830.md).

```sh
QWEN_MODELS_DIR=/path/to/models sh tools/npu/run-tts.sh 0.6b \
  --text 'Hello from the Hexagon backend.' --embedding-in /path/to/speaker.bin \
  --language english --output speech.wav
```

The launcher explicitly selects HTP0 for generation and GPUOpenCL for decoding, rejects non-finite raw logits before sampling and non-finite float audio, and logs CPU/GPU, NSP HVX/HMX and DDR temperatures plus available clocks/cgroup/process counters at about 100 ms intervals. It adds no temperature limit, cooldown or sysfs clock lock. The existing DSP backend requests maximum core/bus/HMX performance corners through Qualcomm HAP; it was not changed to implement thermal throttling. Android's own thermal controls continue to operate. Default experimental settings are four HVX threads, profiling off, and host busy polling on. `GGML_HEXAGON_NHVX`, `GGML_HEXAGON_OPPOLL` and `GGML_HEXAGON_PROFILE` can override those settings. Busy polling trades host activity for latency; no energy advantage is established.

```sh
sh tools/npu/native-env.sh .cache/npu/app-build/tools/npu/npu-probe-vendor
GGML_HEXAGON_PROFILE=1 sh tools/npu/native-env.sh .cache/npu/app-build/tools/npu/npu-test-matmul
mkdir -p .cache/npu/new-fixed-inputs
sh tools/npu/native-env.sh .cache/npu/app-build/tools/npu/npu-test-transformer \
  record /path/to/models/qwen3-tts-0.6b-q8_0.gguf .cache/npu/new-fixed-inputs /path/to/speaker.bin
sh tools/npu/native-env.sh .cache/npu/app-build/tools/npu/npu-test-transformer \
  compare /path/to/models/qwen3-tts-0.6b-q8_0.gguf .cache/npu/new-fixed-inputs
```

`record` explicitly uses CPU; `compare` explicitly requests HTP. CPU-generated prefill embeddings and eight continuous step embeddings are reused exactly, so sampling cannot mask forward-math errors. Steps are a numerical KV-cache control, not a natural speech trajectory. Fixtures contain private speaker-derived activations and remain ignored under `.cache/`.

Only run performance comparisons while Termux is confirmed in the foreground:

```sh
python tools/npu/benchmark.py --models-dir /path/to/models \
  --embedding /path/to/speaker.bin --frames 64 --requests 2
# Also supported: --model-size 1.7b --cases gpu htp4-poll gpu
```

The comparison uses a long prompt, seed 20261026, normal 0.9 talker/subtalker sampling, an excluded warmup, then fixed maximum frame counts. It records actual frame counts and flags early EOS. Each variant runs serially with resident weights and KV allocations; separate variants reload their backend. The CPU profile pins two threads to S25 Ultra cores 6-7. Audio, raw speech codes and raw telemetry stay private in `.cache/`. Timings exclude model loading. The script never imposes temperature gates.

## Discovered activation overflow

Full speech-code inspection found a failure that finite WAV checks missed: valid decoder output can be produced from collapsed zero codes. New `QWEN3_TTS_VALIDATE_LOGITS=1` checks raw talker/predictor logits before sampler masking intentionally adds negative infinity. Failed graphs now return an error rather than appearing as completed synthesis.

Node tracing isolated a predictor Q8 FFN-down multiply receiving finite input with maximum magnitude 157,183.609. The old HVX activation quantizer converted input to FP16 before dividing by its quantization scale; finite values above 65,504 became infinity. The branch adds an FP32 fallback for those blocks whose amax/127 scale remains representable in FP16, normalizing before integer conversion while retaining FP16 stored scales and the existing ordinary-value path. The regression includes large synthetic activations and replay of the captured real FFN weights/input. Merely disabling HMX did not fix the failure.

## Numerical evidence and measured performance

See [the dated results and limitations](performance/npu-v79/REPORT.md), including reviewed request timings and temperature/clock tables. The seven initial ordinary-range Q8 matrix shapes passed an independent dequantized CPU-double oracle with exact canonical Q8 roundtrips, no non-finite outputs, cosine at least 0.9999903 and NRMSE at most 0.004738. The fixed-input talker gate passed all 18 output comparisons: logits cosine 0.997936-0.999409, hidden cosine 0.996730-0.999308. The predeclared coarse full-model gate was cosine >0.99 and NRMSE <0.15; this is not bit parity or a listening test. A separate fixed-input predictor first-head comparison passed at cosine 0.9996614; later heads are checked for finiteness, with their feedback allowed to diverge.

A fixed-input predictor first-head logit comparison also feeds the CPU hidden state and a fixed CB0 to both backends; later heads are checked for finiteness, with their greedy feedback allowed to diverge. Remaining work includes teacher forcing across all predictor heads, wider prompts/languages and listening controls; bounded queue/graph/weight-stream profiling; and evaluating a fused or fixed-shape predictor only if its synchronization costs justify the integration. DDR is shared by CPU, GPU and HTP. DSP VTCM adds local scratch, not extra model RAM. Vocoder operations still run on OpenCL; this branch does not claim full-model NPU offload or overlap the two stages.

Independent review added a conservative K%128 support restriction for dense/indexed quantized matrices (70 metadata checks, plus supported K256/384 numeric cases), a fail-closed code-dump association check after late request failures, and recovery from partially removed private compiler runtimes. See [the review record](performance/npu-v79/REVIEW.md).

Reference conditioning has additional [correctness checks and two application fixes](performance/voice-cloning-20261027/REPORT.md). Use an embedding extracted for the loaded model: the supplied 0.6B speaker row has 1024 floats and the 1.7B row has 2048. A wrong-width saved embedding now returns an explicit error. The non-streaming ICL framing fix preserves the ordinary streaming fast path. Latest reference/ICL listening is pending; this is not a blanket voice-quality claim.
