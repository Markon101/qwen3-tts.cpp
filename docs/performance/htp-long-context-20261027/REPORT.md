# Long-context HMX attention correction — 2026-10-27

The remaining garbled ICL output was reproduced as a **numerical NPU attention failure**, independent of reference transcription, Mimi encoding, sampling, and vocoder trimming. A minimal producer-completion patch corrects it while retaining HMX, its pipeline, the existing thread settings and the ordinary short-context route.

## Isolation

CPU ICL-generated codes decoded to the requested sentence. NPU-generated codes decoded to unintelligible output both alone and with the reference prefix prepended/trimmed. Blind Gemini 3.8 Flash transcription requests received only audio and an intelligibility/transcription instruction, without the expected sentence. These are automated listening controls, not a substitute for the user's quality judgement.

Frozen CPU reference-conditioned input produced closely matching CPU/HTP **prefill** outputs. However, eight deterministic **decode steps after a 216-position prefill** failed: original HMX hidden-state cosine was about 0.21–0.30, logit cosine 0.766–0.823. Disabling HMX passed the same control. Earlier long-prefill checks had missed this because prefill uses explicit attention while decode uses `FLASH_ATTN_EXT`.

A standalone direct-HTP attention test removes the model entirely. It uses public deterministic F32 queries, F16-rounded keys/values, 16 query heads, eight KV heads, 128 channels and an independent CPU-double stable-softmax oracle. Both realistic strided KV storage and contiguous controls are tested.

| Original HMX KV length | Result, both layouts |
|---|---|
| 16, 64, 127, 128, 191 | Pass |
| 192, 193, 216, 256, 297, 434 | Fail; NRMSE about 24–49 |

The threshold matches HMX's multi-block pipeline activation with multiple workers. Vector-only execution passed all 22 cases. A fresh pristine-source DSP rebuild was **byte-identical to the preserved failing binary**, and two repeat runs each reproduced 10 passes / 12 failures. This excluded a stale-library explanation.

## Minimal correction

Key interleaving issues VTCM scatter writes on HVX producer workers. The pipelined QK calculation reads those tiles on a separate HMX queue thread. The correction adds a compiler memory barrier and volatile VMEM drains for every tile touched by each producer before worker completion is published. Drains occur only when the pipeline is active. The indexing also covers partial row groups and odd-row padding.

This matches completion patterns already present in the backend's diagonal-tile and matrix paths. The precise compiler/hardware timing is not asserted as a general erratum. Temporary diagnostic variants changed instruction scheduling enough to hide the failure, so they were discarded; the final patch was tested against the independently rebuilt original. No diagnostic modes, global HMX disable, CPU attention fallback, block-size change or thread-count reduction is shipped.

## Validation

- **22/22 attention cases pass with two, four and six HVX workers**; maximum NRMSE below 0.001.
- All ten originally passing short-context outputs remain **bit-identical** to the original binary.
- Eight real long-context decode steps pass; logit cosine is above 0.9999 against CPU.
- Existing 70 support checks and 12 matrix-oracle cases pass.
- The established reference-only output remains **byte-identical**.
- The corrected ICL continuation transcribes as the requested sentence; the failing NPU continuation was unintelligible under the same blinded protocol.
- The previously failing `smalldon.wav` end-to-end ICL trial now transcribes as the full requested test sentence. Its corrected WAV was exported locally, and the user confirmed clear speech.

[controls.json](controls.json), [summary.json](summary.json) and [blinded-transcription.json](blinded-transcription.json) preserve reviewed numbers, exact comparisons and artifact hashes. Input recordings, reference transcripts, model tensors and generated audio remain outside Git. The full sampled ICL trial is an intelligibility check, not a claim of bit-identical stochastic CPU/NPU generation.

No new speedup is claimed. The kernel's routing, HMX/pipeline use, polling and quantization settings remain in place; nonpipeline execution skips the added drains. CPU/GPU/DSP temperature and clock logging continues without added thermal limits. Background state during some controls invalidates maximum-performance timing claims.

## Reproduce

`tools/npu/build-host-dsp.sh` builds `npu-test-flash-attention` along with the existing controls. Run it through `tools/npu/native-env.sh`; `--strided-only` and `--contiguous-only` select layouts. `QWEN_NPU_RESULT_TAG` distinguishes retained raw artifacts across builds. Numerical gates are cosine >0.999, NRMSE <0.03, maximum error <0.02 and finite output; they were fixed before evaluating the patch.

The real-input mode of `npu-test-clone-prefill` also checks eight teacher-forced decode steps after the long prefill. Generated-code feedback is excluded from these comparisons so sampling cannot conceal a decode-kernel error. The user subsequently confirmed that the corrected end-to-end ICL sample contains clear speech. This listening result establishes intelligibility for that trial; broader voice-similarity evaluation remains separate.
