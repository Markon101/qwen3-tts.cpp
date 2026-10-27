# Reference-voice correctness checks — 2026-10-27

Two confirmed application defects are corrected, with the NPU kernels and execution settings preserved. These fixes are separate from an unresolved report of garbled reference-conditioned audio. Listening to the latest reference tests is pending; no claim that all voice-cloning modes are fixed is made.

## Confirmed fixes

**Embedding width:** the supplied 0.6B model requires a 1024-float speaker row and the supplied 1.7B model requires 2048. The saved-embedding APIs previously discarded the supplied size before the transformer copied its full hidden-width row. Reusing the 1024-float preset with 1.7B therefore read outside the embedding allocation. Audio/codes APIs now reject mismatched widths and non-finite embeddings; C batch input also checks width. Fresh-reference synthesis validates the extracted embedding. Default zero embeddings and the dimension query use the loaded transformer's required width. The local workspace demo supplies its existing preset only for 0.6B.

The wrong-width 1.7B test now reports `got 1024, expected 2048`, produces no WAV, and stops before generation. The valid 2048-float reference-conditioned NPU output remains byte-identical to the frozen previous build. This defect also means earlier 1.7B tests that supplied the 0.6B preset were not valid model-conditioning controls, even if logits/audio were finite. Previously rejected 1.7B speed comparisons remain rejected.

**Non-streaming ICL framing:** the builder previously asked ordinary non-streaming synthesis for its entire prompt, removed only its final position, then appended the complete ICL prompt. This duplicated target text and EOS. Active ICL now obtains mode-independent framing from the streaming base builder, while preserving the requested mode when no reference is supplied. The synthetic non-streaming fixture formerly had 70 positions where 60 were expected; corrected CPU and HTP fixtures both have 60. Streaming cases and no-reference fallback remain checked separately. This matches the framing separation in the [upstream Qwen implementation](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py).

## What the reference tests establish

The first reference recording was tested independently through speaker extraction, Mimi encoding, decoder reconstruction, prompt assembly and transformer forward math:

- CPU/default-GPU ECAPA extraction produced identical, finite 2048-float embeddings.
- Mimi produced 206 frames × 16 valid codec IDs. Its computation is CPU based in this checkout.
- CPU/OpenCL reconstruction returned the expected 395,520 samples; normalized CPU/GPU waveform difference was about 0.00328. This is numerical decoder agreement, not proof of voice quality.
- CPU/HTP-built synthetic ICL tensors and frozen CPU forward inputs were checked across chunk boundaries at 16/32/48/49/216 positions. The only expected baseline shape failure was non-streaming framing.
- Actual reference embeddings/codes/transcript were replayed separately. The real frozen-prefill final-logit cosine was 0.999935, NRMSE 0.01140. Sampling outputs were checked for valid/diverse IDs; equality of sampled CPU/HTP codes was not required.
- The user listened to the ordinary cached-reference CPU/NPU sample pair and reported that both sounded great. These samples did not exercise ICL.

For the second reference, the cached-reference CPU/NPU pair, direct-reference NPU route and transcript-conditioned ICL route were run. The full-pipeline speaker extraction exactly matched standalone extraction, and direct `-r` output exactly matched the cached-reference NPU WAV with the same sampling seed. Gemini 3.8 Flash through OpenRouter produced the draft transcript used for the ICL trial. The draft and audio remain private/local. Listening to this second set is pending because the user requested a pause.

[controls.json](controls.json) retains reviewed numeric outcomes and raw-log hashes. [summary.json](summary.json) records exact equality checks and the distinction between human-confirmed and pending listening. Failed CPU task-affinity launch attempts were setup failures, not numerical model failures, and were rerun without unavailable pinning. These are correctness controls; no new speed comparison is asserted.

The existing bounded decoder chunks discard context between chunks, and the ICL callback path may expose the reference portion before final trimming. Those source findings are not established causes of the reported garbling and are not silently changed by this patch. All-head teacher-forced predictor parity, broader voices, and listening remain further work after the requested pause.

## Reproduce the focused controls

With the private SDK/build already prepared, `tools/npu/build-host-dsp.sh` also builds:

- `npu-test-clone-reference`: separate speaker extraction, codec extraction, and bounded decoder reconstruction. Inputs/results are private local paths.
- `npu-test-clone-prefill`: CPU `record` / HTP `compare` controls over public deterministic synthetic prompts and codec IDs, including non-streaming and no-reference cases. An optional real-input mode accepts private embedding/code/text files without printing their contents.

Run either through `tools/npu/native-env.sh`. Private fixture directories must be mode 0700. All model/reference/generated assets remain excluded from Git, and existing temperature/clock logging imposes no thermal limits. No DSP kernel, quantization group size, HVX-thread setting or polling configuration was changed.
