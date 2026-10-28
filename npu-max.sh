#!/data/data/com.termux/files/usr/bin/sh
set -eu
R=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python "$R/tools/npu-max/launch.py" --demo "$@"
