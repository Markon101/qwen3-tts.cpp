#!/data/data/com.termux/files/usr/bin/sh
set -eu
REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
LIB_DIR="$REPO_DIR/.cache/npu-max/build/src"
export LD_LIBRARY_PATH="$LIB_DIR:$LIB_DIR/ggml-hexagon:$LIB_DIR/ggml-opencl:/system/lib64:/vendor/lib64"
unset LD_PRELOAD
export ADSP_LIBRARY_PATH="$LIB_DIR/ggml-hexagon"
export OCL_ICD_VENDORS=/vendor/lib64/libOpenCL.so
# Legacy GGML's hostbuf default would leave Q8 matrix products on CPU.
export GGML_HEXAGON_HOSTBUF=0
export GGML_HEXAGON_NHVX=${GGML_HEXAGON_NHVX:-4}
export GGML_HEXAGON_OPPOLL=${GGML_HEXAGON_OPPOLL:-1}
export GGML_HEXAGON_PROFILE=${GGML_HEXAGON_PROFILE:-0}
export GGML_OPENCL_BINARY_CACHE_DIR=${GGML_OPENCL_BINARY_CACHE_DIR:-"$REPO_DIR/.cache/adreno830-opencl"}
case "$GGML_HEXAGON_NHVX" in 1|2|3|4|5|6) ;; *) printf '%s\n' 'NPU-Max: NHVX must be 1..6 for this v79 device' >&2; exit 2 ;; esac
exec "$@"
