#!/data/data/com.termux/files/usr/bin/sh
set -eu
REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
BUILD_JOBS=${BUILD_JOBS:-2}
if [ ! -f "$REPO_DIR/ggml/CMakeLists.txt" ]; then
  printf '%s\n' 'Initialize the pinned submodule: git submodule update --init ggml' >&2
  exit 2
fi
cmake -S "$REPO_DIR/ggml" -B "$REPO_DIR/ggml/build-opencl" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_OPENCL=ON -DGGML_NATIVE=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=ON -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
cmake --build "$REPO_DIR/ggml/build-opencl" --target ggml -j "$BUILD_JOBS"
cmake -S "$REPO_DIR" -B "$REPO_DIR/build-opencl" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_BUILD_DIR="$REPO_DIR/ggml/build-opencl" -DQWEN3_TTS_TIMING=ON
cmake --build "$REPO_DIR/build-opencl" --target qwen3-tts-cli -j "$BUILD_JOBS"
