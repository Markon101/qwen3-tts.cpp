#!/data/data/com.termux/files/usr/bin/sh
set -eu
REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
MODEL_SIZE=${1:-}
case "$MODEL_SIZE" in
  0.6b|1.7b) shift ;;
  *) printf '%s\n' 'Usage: tools/npu/run-tts.sh 0.6b|1.7b [TTS options]' >&2; exit 2 ;;
esac
MODELS_DIR=${QWEN_MODELS_DIR:-"$REPO_DIR/models"}
CLI="$REPO_DIR/.cache/npu/app-build/qwen3-tts-cli"
if [ ! -x "$CLI" ]; then
  printf '%s\n' 'Build first: sh tools/npu/build-host-dsp.sh' >&2
  exit 2
fi
unset QWEN3_TTS_FORCE_CPU GGML_PROFILE_OPS GGML_OPENCL_QCOM_RECORD GGML_OPENCL_Q8_SMALL2
export QWEN3_TTS_TRANSFORMER_BACKEND=hexagon
export QWEN3_TTS_DECODER_BACKEND=opencl
export QWEN3_TTS_VALIDATE_AUDIO=1
export QWEN3_TTS_VALIDATE_LOGITS=1
exec python "$REPO_DIR/scripts/android/profile_runner.py" \
  sh "$REPO_DIR/tools/npu/native-env.sh" "$CLI" \
  -m "$MODELS_DIR/qwen3-tts-$MODEL_SIZE-q8_0.gguf" \
  --tokenizer "$MODELS_DIR/qwen3-tts-tokenizer-f16-f32.gguf" --threads 2 "$@"
