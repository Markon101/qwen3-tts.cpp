# tools/

Developer and end-user tools for qwen3-tts.cpp.

## Android launchers

The repository-root `npu-max.sh` selects NPU-Max profiles on a prepared Termux installation; see the [fork audit](../docs/FORK_AUDIT.md). The separate `npu.sh` is documented in [NPU launchers](../docs/NPU_LAUNCHERS.md). Older workspace snapshots are described in [workspace-launchers](workspace-launchers/README.md).

## model-downloader/

**Easiest path.** Downloads pre-built GGUF files directly from HuggingFace.
No Python conversion dependencies — only `huggingface_hub` required.
See [model-downloader/README.md](model-downloader/README.md).

```
Windows: tools\model-downloader\download.bat
macOS:   tools/model-downloader/download.sh
```

## build-scripts/

Build the project from source. See [build-scripts/README.md](build-scripts/README.md).

```
Windows: tools\build-scripts\build.bat
macOS:   tools/build-scripts/build.sh
```

## model-converter/

Interactive wizard to download HuggingFace safetensors and convert to GGUF locally.
Requires `torch`, `safetensors`, `gguf`. See [model-converter/README.md](model-converter/README.md).

```
Windows: tools\model-converter\setup_models.bat
macOS:   tools/model-converter/setup_models.sh
```

## packages/

Package manager manifests for distributing pre-built binaries.
See [packages/README.md](packages/README.md) for the full release process.

- `scoop/qwen-tts.cpp.json` — Scoop manifest (Windows): `scoop install qwen-tts.cpp` → use as `qwen-tts`
- `homebrew/qwen-tts.cpp.rb` — Homebrew formula (macOS/Linux): `brew install qwen-tts.cpp` → use as `qwen-tts`

These files are reference copies — the live manifests live in separate bucket/tap repos.
