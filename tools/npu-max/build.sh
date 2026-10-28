#!/data/data/com.termux/files/usr/bin/sh
# Independent build; never writes the working .cache/npu runtime.
set -eu
R=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
SDK="$R/.cache/npu/toolchain/sdk/6.6.0.0"
EMU="$R/.cache/npu/toolchain/emulation"
B="$R/.cache/npu-max/build"
python "$R/tools/npu-max/prepare-qaic.py"
unset LD_LIBRARY_PATH LD_PRELOAD
cmake -S "$R/ggml" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DGGML_HEXAGON=ON -DGGML_OPENCL=ON \
  -DGGML_BUILD_TESTS=ON -DGGML_BUILD_EXAMPLES=OFF \
  -DHEXAGON_SDK_ROOT="$SDK" -DHEXAGON_TOOLS_ROOT="$SDK/tools/HEXAGON_Tools/19.0.07" \
  -DGGML_HEXAGON_USE_SPHAL=ON -DGGML_HEXAGON_ARCHITECTURES=v79 \
  -DGGML_HEXAGON_QAIC_DIR="$R/.cache/npu-max/qaic-bin" -DGGML_HEXAGON_EMULATION_DIR="$EMU" \
  -DGGML_HEXAGON_DSP_BUILD_JOBS=2 -DPREBUILT_LIB_DIR=android_aarch64
cmake --build "$B" --target htp-v79 --parallel 1
cmake --build "$B" --target ggml --parallel "${HOST_BUILD_JOBS:-2}"
cmake -S "$R" -B "$R/.cache/npu-max/app-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_BUILD_DIR="$B" -DQWEN3_TTS_TIMING=ON -DQWEN3_TTS_NPU_TESTS=ON
cmake --build "$R/.cache/npu-max/app-build" --parallel "${HOST_BUILD_JOBS:-2}"
