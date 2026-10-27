#!/data/data/com.termux/files/usr/bin/sh
set -eu
REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$REPO_DIR/scripts/android/run-tts.sh" 0.6b "$@"
