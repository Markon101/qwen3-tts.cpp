# Qualcomm Snapdragon 8 Elite & Adreno 830 Optimizations for Qwen3-TTS.cpp

This document describes the native Android/Termux optimizations, bug fixes, and architectural adaptations implemented for Qwen3-TTS on Qualcomm Snapdragon 8 Elite (SM8750) with Adreno 830 GPU.

---

## 1. Summary of Optimizations

### 1.1 OpenCL GPU Acceleration for Adreno 830
- **Adreno OpenCL ICD Linkage**: Direct dynamic loading of proprietary Qualcomm OpenCL driver (`/vendor/lib64/libOpenCL.so`).
- **Coalesced Register Memory & Subgroup Optimization**: Enabled Qualcomm subgroup shuffle extensions (`cl_qcom_subgroup_shuffle`), vector subgroup broadcasts, and optimized workgroup dimensions (`64x1x1` / `128x1x1`).
- **Mixed-Precision Matrix Multiply (F16xF32)**: Dedicated Adreno subgroup kernels for quantized `Q8_0` and half-precision `F16` weights multiplied by `F32` activation buffers.
- **On-Demand FlashAttention Kernel Compilation**: Deferred FlashAttention compilation to eliminate cold-start compile latency on mobile.

### 1.2 Host Embedding Pre-Lookup & Direct Dispatch
- `tts_transformer.cpp`: Cached host memory representation of codebook and vocabulary embedding weights (`host_embd_raw`).
- Replaced device-side quantized row lookups (`ggml_get_rows`) for the 14-step autoregressive code predictor with instant CPU pointer copies into device tensors.
- Eliminated GPU graph splits and kernel enqueue overhead during autoregressive generation steps.

### 1.3 Vocoder Memory Management & Live Probe
- Added non-invasive evaluation callbacks (`ggml_backend_sched_set_eval_callback`) guarded by `QWEN3_TTS_DEBUG_VOCODER` to observe intermediate tensor values live during evaluation without memory-reuse corruption.
- Added public accessor `AudioTokenizerDecoder::get_model()` for inspection and debugging.

---

## 2. Root Cause Analysis & Fix: 12.5 Hz Audio Choppiness

### 2.1 Symptom
When synthesizing speech using the GPU OpenCL decoder (`AudioTokenizerDecoder` on Adreno 830), the generated speech exhibited audible periodic clicking and choppiness at ~12.5 Hz (every 80 ms, corresponding exactly to 1920 audio samples at 24 kHz). The CPU decoder produced smooth speech without clicks.

### 2.2 Numerical Diagnosis
Waveform boundary analysis at 1920-sample multiples revealed:
- **CPU Decoder**: Mean boundary discontinuity = `0.01062`, Discontinuity ratio = `0.94x` (smooth baseline).
- **GPU Decoder (Unpatched)**: Mean boundary discontinuity = `0.05114` (5x higher), Max discontinuity = `0.22635` (4x higher), Discontinuity ratio = `4.16x` (severe discontinuous step at every vocoder frame boundary).

Live node-by-node tracing through the computation graph isolated the divergence:
1. Embeddings, pre-conv, and pre-transformer matched CPU within FP16 arithmetic error (`< 1e-4`).
2. Upsample Block 0 `dwconv` (depthwise 1D convolution) collapsed on GPU:
   - CPU `up0_dwconv`: `mean = 0.04697, rms = 0.49323`
   - GPU `up0_dwconv`: `mean = -0.00073, rms = 0.00737` (signal energy collapsed by ~98.5%).

### 2.3 The Root Cause
In GGML's OpenCL `im2col` kernels (`src/ggml-opencl/kernels/im2col_f16.cl` and `im2col_f32.cl`), the 1D convolution fast path (`OH == 1 && KH == 1`) had a critical bug:

```c
// BUGGY ORIGINAL CODE:
if (OH == 1 && KH == 1) {
    int chw = (int)(gid % CHW);
    int ix  = (int)(gid / CHW);
    int ic  = chw / (int)KW;
    int kx  = chw % (int)KW;

    int iiw = ix * s0 + kx * d0 - p0;
    if (iiw < 0 || iiw >= (int)IW) {
        dst[gid] = 0.0f;
    } else {
        ulong offset_src = (ulong)ic * delta_offset;
        dst[gid] = src1[offset_src + iiw];
    }
    return;
}
```

1. **Missing Batch Offset**: `offset_src` only accounted for `(ulong)ic * delta_offset` and completely omitted `batch * batch_offset`.
2. **Batch Index Mixed into Spatial Index**: `gid / CHW` spans `[0, batch * OW - 1]`. For any batch index `batch_idx >= 1`, `ix` was treated as purely spatial, so `iiw = ix * s0 + ...` exceeded the input sequence length `IW`.
3. In depthwise 1D convolution (`ggml_conv_1d_dw`), channels (1024) are folded into the batch dimension of `new_b` (`src1->ne[2] = 1024`, `batch = 1024`).
4. As a result, **1023 out of 1024 channels were completely zeroed out** (`dst[gid] = 0.0f`). Only channel 0 was written. This hollowed out the vocoder activations, destroying boundary continuity across frames.

### 2.4 The Fix
Decompose `gid / CHW` into `spatial_ix = (gid / CHW) % OW` and `batch_idx = (gid / CHW) / OW`, and include `batch_idx * batch_offset` in `offset_src`:

```c
// FIXED CODE:
if (OH == 1 && KH == 1) {
    int chw = (int)(gid % CHW);
    int spatial = (int)(gid / CHW);
    int ix  = spatial % (int)OW;
    int batch_idx = spatial / (int)OW;
    int ic  = chw / (int)KW;
    int kx  = chw % (int)KW;

    int iiw = ix * s0 + kx * d0 - p0;
    if (iiw < 0 || iiw >= (int)IW) {
        dst[gid] = 0.0f;
    } else {
        ulong offset_src = (ulong)ic * delta_offset + (ulong)batch_idx * batch_offset;
        dst[gid] = (half)src1[offset_src + iiw];
    }
    return;
}
```

### 2.5 Verification Results
- **Full 30-Frame Divergence Test**:
  - Max sample diff CPU vs GPU: dropped from `0.9898` to `0.0103`.
  - Mean sample diff CPU vs GPU: dropped from `0.0804` to `0.00034` (236x improvement).
- **Speech Synthesis Discontinuity Analysis**:
  - Mean boundary step: `0.01025` (vs CPU `0.01062`).
  - Max boundary step: `0.09369` (vs CPU `0.05774`, unpatched GPU `0.22635`).
  - **Discontinuity Ratio**: `0.96x` (vs unpatched GPU `4.16x`). The audio is completely smooth and indistinguishable from CPU synthesis.

---

## 3. Running Speech Synthesis on Termux Android

### 3.1 Environment Setup
```bash
export OCL_ICD_VENDORS="/vendor/lib64/libOpenCL.so"
export LD_LIBRARY_PATH="/data/data/com.termux/files/home/qwen3-tts-workspace/repos/CHE3MZ-qwen3-tts.cpp/build-opencl:/data/data/com.termux/files/home/qwen3-tts-workspace/repos/CHE3MZ-qwen3-tts.cpp/ggml/build-opencl/src:/data/data/com.termux/files/home/qwen3-tts-workspace/repos/CHE3MZ-qwen3-tts.cpp/ggml/build-opencl/src/ggml-opencl:/system/lib64:/vendor/lib64"
```

### 3.2 Launchers
- **0.6B Model**: `./run-tts-0.6b.sh -t "Hello world!" -o output.wav`
- **1.7B Model**: `./run-tts-1.7b.sh -t "Hello world!" -o output.wav`
- **Voice Cloning with Reference**: `./run-tts-0.6b.sh -t "Cloned voice text" -r reference.wav -o cloned.wav`
- **Voice Cloning with Transcript (ICL Mode)**: `./run-tts-0.6b.sh -t "Cloned text" -r reference.wav --ref-text "Reference transcript." -o icl.wav`
- **Interactive Server Mode**: `./run-tts-0.6b.sh --server`
