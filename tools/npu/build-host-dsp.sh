#!/data/data/com.termux/files/usr/bin/sh
set -eu
REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
SDK_DIR="$REPO_DIR/.cache/npu/toolchain/sdk/6.6.0.0"
EMULATION_DIR="$REPO_DIR/.cache/npu/toolchain/emulation"
BUILD_DIR="$REPO_DIR/.cache/npu/build"
if [ ! -f "$SDK_DIR/build/cmake/hexagon_fun.cmake" ]; then
  printf '%s\n' 'Prepare the private toolchain first: python tools/npu/prepare-toolchain.py .' >&2
  exit 2
fi
unset LD_LIBRARY_PATH LD_PRELOAD
cmake -S "$REPO_DIR/ggml" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=OFF \
  -DGGML_HEXAGON=ON -DGGML_OPENCL=ON -DGGML_BUILD_TESTS=ON -DGGML_BUILD_EXAMPLES=OFF \
  -DHEXAGON_SDK_ROOT="$SDK_DIR" -DHEXAGON_TOOLS_ROOT="$SDK_DIR/tools/HEXAGON_Tools/19.0.07" \
  -DGGML_HEXAGON_USE_SPHAL=ON -DGGML_HEXAGON_ARCHITECTURES=v79 \
  -DGGML_HEXAGON_QAIC_DIR="$EMULATION_DIR" -DGGML_HEXAGON_EMULATION_DIR="$EMULATION_DIR" \
  -DGGML_HEXAGON_DSP_BUILD_JOBS=2 -DPREBUILT_LIB_DIR=android_aarch64
# Do not run host and DSP QAIC generators concurrently in this SDK/QEMU path.
cmake --build "$BUILD_DIR" --target htp-v79 --parallel 1
cmake --build "$BUILD_DIR" --target ggml --parallel "${HOST_BUILD_JOBS:-2}"

cmake -S "$REPO_DIR" -B "$REPO_DIR/.cache/npu/app-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_BUILD_DIR="$BUILD_DIR" \
  -DQWEN3_TTS_TIMING=ON -DQWEN3_TTS_NPU_TESTS=ON
cmake --build "$REPO_DIR/.cache/npu/app-build" --target qwen3-tts-cli \
  npu-probe-vendor npu-test-matmul npu-test-transformer npu-test-generation --parallel "${HOST_BUILD_JOBS:-2}"
