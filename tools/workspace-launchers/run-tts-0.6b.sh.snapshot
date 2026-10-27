#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/repos/CHE3MZ-qwen3-tts.cpp"
MODEL_FILE="${SCRIPT_DIR}/models/qwen3-tts-0.6b-q8_0.gguf"
TOKENIZER_FILE="${SCRIPT_DIR}/models/qwen3-tts-tokenizer-f16-f32.gguf"
CLI_BIN="${REPO_DIR}/build-opencl/qwen3-tts-cli"

export OCL_ICD_VENDORS="/vendor/lib64/libOpenCL.so"
export LD_LIBRARY_PATH="${REPO_DIR}/build-opencl:${REPO_DIR}/ggml/build-opencl/src:${REPO_DIR}/ggml/build-opencl/src/ggml-opencl:/system/lib64:/vendor/lib64:${LD_LIBRARY_PATH:-}"

exec "${CLI_BIN}" -m "${MODEL_FILE}" --tokenizer "${TOKENIZER_FILE}" "$@"
