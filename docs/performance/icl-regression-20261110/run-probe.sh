#!/data/data/com.termux/files/usr/bin/sh
# ICL parity probe for one DSP library.
# Usage: sh run-probe.sh <label> [so-path]
# Installs the given libggml-htp-v79.so (if provided), runs the real-input ICL
# clone-prefill comparison against the frozen CPU fixtures, and stores output.
set -eu
R=/data/data/com.termux/files/home/qwen3-tts-workspace/repos/Markon101-qwen3-tts.cpp
D=$R/.cache/icl-debug/phase2
SO=$R/.cache/npu-max/build/src/ggml-hexagon/libggml-htp-v79.so
LABEL=${1:?label required}
[ $# -ge 2 ] && cp -p "$2" "$SO"
echo "lib: $(sha256sum "$SO")"
cd "$R"
sh tools/npu-max/native-env.sh .cache/npu-max/app-build/tools/npu/npu-test-clone-prefill \
  compare ../../models/qwen3-tts-0.6b-q8_0.gguf "$D/fixtures" \
  "$D/fixtures/spk-cpu.bin" "$D/fixtures/ref-codes.i32" \
  "$D/fixtures/ref.txt" "$D/fixtures/target.txt" > "$D/$LABEL.txt" 2>&1 || true
grep -E "^SUMMARY|real-step-[07]-hidden" "$D/$LABEL.txt"
