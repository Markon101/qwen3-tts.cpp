#!/data/data/com.termux/files/usr/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$SCRIPT_DIR/run-tts.sh" 0.6b "$@"
