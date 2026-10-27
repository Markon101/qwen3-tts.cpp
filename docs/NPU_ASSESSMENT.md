# Hexagon NPU feasibility and Termux examples

The native v79 prototype is now implemented on the experimental `npu-v79-bringup` branch. Source-built DSP matrix arithmetic and fixed-input talker controls passed on this phone. Full TTS testing exposed a separate predictor activation overflow, showing why finite WAVs alone were insufficient. Read [the implementation and validation guide](NPU_V79.md) and [dated evidence](performance/npu-v79/REPORT.md) for the current result. The original feasibility observations below explain the chosen route.

## The strongest Termux example

[Ishabdullah/OpenCL-S24-Ultra](https://github.com/Ishabdullah/OpenCL-S24-Ultra) documents native, non-root Termux inference on a Samsung S24 Ultra, Android 16, Hexagon **v75**. Its [validation record](https://github.com/Ishabdullah/OpenCL-S24-Ultra/blob/main/docs/NPU_VALIDATION.md) reports eight source-built DSP matrix tests against CPU and model generation. Those are the author's published results, not a reproduction on this phone.

Its [installation guide](https://github.com/Ishabdullah/OpenCL-S24-Ultra/blob/main/docs/NPU_INSTALL.md) explains loading device-provided CDSP/FastRPC through Android's exported `sphal` namespace. Compiler tools run under QEMU, while ARM64 host inference and DSP kernels run natively. It requires significant SDK/build storage and matching runtime components. The relevant adaptation here is **v79**, not blindly using its v75 binaries. Direct file permissions are not a substitute for testing this vendor-broker path. The guide is more useful than copying a generic `LD_LIBRARY_PATH` recipe.

## Other references, with their limits

- [haozixu/llama.cpp-npu](https://github.com/haozixu/llama.cpp-npu/blob/htp-backend/README.md) is the authors' research implementation for *Scaling LLM Test-Time Compute with Mobile NPU on Smartphones*. It explicitly describes Termux as an alternative to ADB, with possible permission constraints. It requires host cross-compilation and a separate HTP operator library; this is a technical reference, not equivalent evidence of a clean non-root S25 Ultra installation.
- [Model-To-NPU](https://github.com/VitalikDen0/Model-To-NPU/blob/main/README_EN.md) is an adjacent v79/OnePlus 13 QNN example. The currently read guide uses a native Android server/APK/ADB route and explicitly says Termux is unnecessary. Search snippets still exposed older Termux instructions; those are not presented here as current verified instructions. Persistent contexts and memory reuse are relevant design ideas, but its diffusion performance is not a TTS forecast.
- [mini-htp](https://github.com/xwhzz/mini-htp) is a small matrix demonstration scaffold. Its README explicitly says it was not compiled or hardware-tested in that environment, so it is excluded as proof of a working installation.

## This phone's measured access state

- `/vendor/lib64/libcdsprpc.so` and `libSnpeHtpV79Stub.so` are present.
- At the initial feasibility audit, SDK environment variables were unset and `GGML_HEXAGON` was OFF in the OpenCL baseline. The experimental branch now uses a separate private SDK and combined backend build.
- Opening `/dev/fastrpc-cdsp` and `/dev/fastrpc-cdsp-secure` directly failed with EACCES in both read-only and read/write modes. The local probe record showed EACCES for these modes.
- A native probe loaded `/apex/com.android.neuralnetworks/lib64/libneuralnetworks.so` and successfully enumerated **one** device: `nnapi-reference`, type 2 (CPU). No accelerator was exposed to this process through that API. The local probe output is summarized here. NNAPI is [deprecated in Android 15](https://developer.android.com/ndk/guides/neuralnetworks); it is not assumed to be the preferred new integration route.

These observations do not prove that the vendor FastRPC broker is unavailable. The later `sphal` probe and native v79 DSP calculations succeeded here. They also do not justify claiming root is required.

## Integration route and original validation plan

[Qualcomm's supported-device table](https://docs.qualcomm.com/nav/home/QNN_general_overview.html?product=924033590759186372) maps SM8750/Snapdragon 8 Elite to **Hexagon v79**. [Upstream GGML's developer guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/developer.md) describes separate host and DSP libraries, tiled weight layouts, shared DDR and DSP-local VTCM. The [build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md) documents the corresponding SDK/toolchain workflow.

Start with the existing native GGML backend rather than immediately converting the entire TTS model to ONNX/QNN:

1. Reuse only the relevant, reviewed vendor-namespace loading approach to test library symbols and a real v79 session. Follow with one source-built matrix calculation checked against CPU. Enumeration and successful library loading alone do not count as NPU execution.
2. Audit the exact predictor graph. This checkout maps matrix multiply, RMS normalization, RoPE, softmax, flash attention and several other operations. It does not map `DIAG_MASK_INF`, `IM2COL`, or general transposed convolution in its Hexagon operation switch. Mask rewriting or fallback is needed for prefill; the vocoder is a larger coverage problem.
3. Add explicit per-component HTP selection. The application currently recognizes a CPU override; its ordinary backend preference otherwise favors the GPU. Merely building another backend does not select it.
4. Benchmark a **whole** code-predictor graph with resident weights and KV state. The 15 sequential predictions per frame create synchronization costs; fast individual NPU matrix operations are insufficient evidence of a speedup. Preserve logits, codes, finite audio, fallback/split counts, temperatures and clocks.
5. Consider fixed-shape QNN contexts or a custom fused predictor only after that gate. The predictor's short KV range is attractive for fixed shapes, but sampling, embedding lookup and recurrence still have to preserve model behavior.

All engines share the phone's physical DDR. HTP adds compute and local scratch memory, not extra model RAM or free transfers between backend allocators. In the validated fast 0.6B profile, decoder time is about half of synthesis time: even zero generation time would yield only about **2.08×** total speedup from that profile. This is an arithmetic ceiling for generation-only acceleration, not an NPU prediction. Decoder optimization or correct streaming overlap would still matter.
