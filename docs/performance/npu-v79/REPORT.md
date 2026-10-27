# Hexagon v79 experiment — 2026-10-27 UTC

Native, non-root Termux v79 execution is working on this S25 Ultra. The source-built backend passes independent matrix and fixed-input model controls after an activation-overflow fix. The final NPU TTS path completed fixed 64-frame requests with raw-logit and decoded-audio validation. **No end-to-end speedup is established:** 0.6B remains faster on the tuned CPU generator, and failed 1.7B CPU/OpenCL control requests invalidate a clean speed comparison.

The stable OpenCL default branch is preserved. Source, build helpers and reviewed numeric evidence are published on `npu-v79-bringup`; models, voices, generated audio, raw activations/codes, SDK contents and compiled libraries remain private. [Build/run instructions](../../NPU_V79.md) describe the exact route and its attribution.

## Correctness gates and the failure they caught

The Android `sphal` broker exposed the phone's FastRPC implementation even though direct CDSP file opens failed. A source-built v79 DSP then executed native Q8 matrix operations, with an independent CPU-double oracle over canonically dequantized Q8 weights. Canonical weight set/get roundtrips were exact.

Selecting HTP alone was insufficient: the old `HOSTBUF=1` default left Q8 matrix multiplies on CPU. `GGML_HEXAGON_HOSTBUF=0` removed the matrix fallbacks from the reported step graphs. Talker and predictor steps each then had one split; prefill still has CPU causal-mask work. This is mixed-backend execution, not an assertion that every transformer operation is on DSP.

Finite WAVs were also insufficient. Some early runs collapsed to zero speech codes while the decoder still emitted a finite non-silent waveform. Those runs are retained as rejected diagnostics in [requests.csv](requests.csv). For example, one 1.7B HTP request had all 1,024 codes equal to zero; one 0.6B HTP request had 869 zero codes. Their apparent speed gains are withdrawn.

Raw-logit checks and per-node tracing isolated an FFN-down Q8 multiply receiving **finite input with maximum magnitude 157,183.609**. The HVX activation quantizer converted it to FP16 before normalization, exceeding FP16's 65,504 finite range. Disabling HMX did not solve this. The fix preserves the normal vector path. For large blocks whose amax/127 remains representable in FP16, it normalizes activations in FP32 before writing integer quants and FP16 scales. Input/scale scratch aliasing is permitted explicitly.

| Numerical control | Before | Final result |
|---|---:|---:|
| Large synthetic Q8 multiply, K=1024/M=256/N=1 | 256 NaNs | finite; NRMSE 0.001196 |
| Captured real FFN, K=3072/M=1024/N=2 | 1,024 NaNs in isolated replay | finite; NRMSE 0.0002759; cosine 0.999999999 |
| Full synthetic suite | ordinary-range cases passed; large case failed | 12/12, including K256/384, large activations and zero input |
| Talker: identical CPU prefill and eight fixed step embeddings | separate initial control passed | 18/18 hidden/logit comparisons pass |
| Predictor first head: identical CPU hidden state and fixed CB0 | not covered by initial talker check | cosine 0.9996614; NRMSE 0.02779 |
| Predictor later heads | autoregressive feedback differs | all 15 heads finite; later-head parity not claimed |

Ordinary-range matrix max error was at most 0.004916, NRMSE at most 0.004738 and cosine at least 0.9999903. Large-input absolute errors scale with output magnitude, so their normalized errors are reported separately. The matrix gate is cosine >0.999, NRMSE <0.02, finite outputs, exact Q8 roundtrip, and an activation-scale-adjusted absolute bound. The full-model gate was declared as cosine >0.99 and NRMSE <0.15; it is deliberately coarse and does not establish bit parity or listening quality. The configured activation quantization group size was 128; 32/64 share the fallback but were not separately built in this audit.

Independent review also found an existing quantized short-K tail overwrite risk. The host now conservatively declines K lengths not divisible by 128 for dense and indexed quantized multiplies; 70 metadata support checks passed across five types and seven K lengths. Unsafe shapes were never executed. See [review dispositions](REVIEW.md).

[The numerical records](numerical.json) retain failed and successful controls. Metrics derived from non-finite outputs are marked null rather than summarized over only their finite subset.

## Final 0.6B comparison

Each process loads once, performs an excluded eight-frame warmup, then two seeded 64-frame requests (5.12 seconds audio). Same long English prompt, saved speaker embedding, model/tokenizer, seed 20261026, and normal 0.9 talker/subtalker sampling. The CPU generator uses two threads on cores 6-7. HTP uses four HVX threads, `HOSTBUF=0`, busy polling, profiling off. All use the OpenCL vocoder and validate raw logits before sampling plus float audio before PCM saving. Times exclude model loading.

| Sequential block | Generate ms, median | Decode ms, median | Synthesis ms, median | Individual synthesis ms |
|---|---:|---:|---:|---|
| CPU A | 2655.5 | 2816.0 | 5472.5 | 5438, 5507 |
| HTP | 3400.0 | 3814.5 | 7216.0 | 6995, 7437 |
| CPU B | 2777.0 | 3962.5 | 6740.5 | 6864, 6617 |

HTP generation and total synthesis were slower than both CPU blocks. Changing device state also increased decoder time; these are exploratory device-state measurements, not a clock-controlled causal estimate. There is no reason to replace the fast 0.6B CPU/OpenCL profile with this NPU profile on current evidence.

All six measured requests completed 64 frames, passed raw-logit/audio checks and emitted varied speech codes (one zero code out of 1,024 per request is not the rejected all-zero collapse). CPU and HTP seeded speech codes differ. No equivalent listening-quality claim is made.

## 1.7B: valid HTP requests, invalid performance controls

The final HTP block completed its eight-frame warmup and both 64-frame requests with validation enabled. Measured generation was **4447 and 4867 ms**, decoder **3603 and 3995 ms**, total **8052 and 8865 ms**. Both measured requests had 1,024 codes and no zero-code collapse.

However, the bracketed CPU control had three raw-logit failures across six requests, including warmups, in talker prefill. A separate OpenCL control had one failed warmup followed by two successful full requests. [request-failures.json](request-failures.json) preserves these errors; server process exit zero alone did not establish successful synthesis. The benchmark now records request failures and returns nonzero overall when any fail.

These failures preclude claiming a 1.7B NPU speed gain from this session. An [isolated transformer diagnostic](generation-controls.json) completed three F16-KV CPU requests and three F32-KV CPU requests; that did not reproduce or explain the full-pipeline failure. F32 KV is therefore not promoted as a fix. Component coexistence, allocation/state and prefill numerical behavior remain open investigations. The runtime logit guard prevents these failures from being silently saved as successful speech on this branch.

## Temperatures, clocks and memory observations

There were **no temperature aborts, cooldown waits or added thermal limits**. Sampling is measurement-only at roughly 100 ms; Android's controls remain active. The existing DSP backend requests maximum core/bus/HMX performance corners using HAP, while actual CPU/GPU clocks and thermal readings vary.

| Final block | CPU peak °C | GPU peak °C | NSP HVX peak °C | NSP HMX peak °C | DDR peak °C | Battery peak °C | Median reported GPU MHz |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.6B CPU A | 105.4 | 104.9 | 72.6 | 72.9 | 83.0 | 37.7 | 1200 |
| 0.6B HTP | 104.6 | 88.4 | 86.9 | 85.7 | 81.1 | 38.5 | 734 |
| 0.6B CPU B | 105.4 | 79.6 | 68.7 | 72.2 | 73.7 | 39.1 | 734 |
| 1.7B HTP | 104.6 | 86.5 | 91.1 | 86.9 | 80.7 | 38.7 | 734 |

Names and units follow the firmware's readable sensors. NSP HVX/HMX and DDR were added after discovery; earlier missing readings remain missing, not backfilled. CPU/GPU frequency readings are reported policy/current samples, not cycle-accurate residency. Most live samples were in `/top-app`; a few process-exit samples moved to the root group. No foreground gate was used to discard inconvenient samples.

[telemetry.csv](telemetry.csv) retains timestamped group maxima and reported CPU/GPU clocks for curves, and [runs.json](runs.json) retains peak/count summaries. Raw private JSONL additionally preserves individual sensor values, scheduling state and process counters. [dsp-profile.csv](dsp-profile.csv) contains operation cycle/usec estimates from a separate diagnostic profile, not the timed speech runs. The `cycles/usec` MHz estimate is coarse on very short operations; restrict analysis to sufficiently long operations and do not treat it as a direct DSP frequency-control readout or align its wrapping cycle counter to host time without calibration.

All engines share DDR; the capability probe reported one 8 MiB VTCM page. This work did not establish sustained energy advantage, extra memory bandwidth, a flash-bandwidth bottleneck, or causal thermal superiority. Earlier [storage/GPU-copy measurements](../adreno830/REPORT.md) remain separate synthetic observations. The DSP profile places most execution time in matrix multiplies, but it does not distinguish arithmetic, transfer and cache causes without further controls.

## Provenance and remaining work

[artifacts.json](artifacts.json) records source revisions, file/artifact hashes, toolchain pin, device identity and retained-log hashes. The implementation was rebuilt after committing it; final numerical checks and both public NPU launchers passed. Review then added the host-only shape restriction, and a clean reviewed build passed all 70 support and 12 numeric checks. Artifact hashes describe the final reviewed set plus the preserved pre-fix DSP; historical logs span intermediate builds and are not all attributed to the final binary hashes. The 64-frame timing blocks preceded the host-only tail restriction; their tested TTS dimensions remain supported. The original tested repositories remain unchanged; the public default branches retain their earlier heads.

Next investigations with clear value are the full-pipeline 1.7B prefill failure; fixed-input teacher forcing through every predictor head; listening comparisons; and separating talker/predictor placement with resident weights. Fixed-shape/fused contexts, queue/polling costs and DSP power-vote lifetime are candidates for later measurements. None is reported as an implemented speed gain here.
