#!/data/data/com.termux/files/usr/bin/sh
# Portable NPU entry point; the shared launcher handles modes and validation.
set -eu
REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python "$REPO_DIR/tools/npu/launch.py" --demo "$@"
