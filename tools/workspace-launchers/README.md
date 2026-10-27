# Workspace launcher snapshots

The outer phone workspace is not a Git checkout. Every top-level `.sh` launcher's contents are now versioned here or in the tracked NPU shortcut template; `manifest.json` maps the original names to exact content hashes.

The five `.snapshot` files preserve the existing older OpenCL launchers unchanged. Their relative paths expect the workspace root and may refer to retained private experiment binaries. They are restoration/provenance copies, not the portable entry points for a fresh clone. The source fork already provides portable OpenCL launchers at `run-tts-fast-0.6b.sh`, `run-tts-fast-1.7b.sh` and `scripts/android/`.

The active workspace `npu.sh` is a byte-identical copy of `tools/npu/workspace-npu.sh.in`. It delegates to the tracked repository-root `npu.sh` and the shared `tools/npu/launch.py`. The NPU mode logic lives in the repository rather than the outer workspace. Models, voices, SDK files and generated audio remain outside Git.
