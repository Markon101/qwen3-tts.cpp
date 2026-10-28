#!/usr/bin/env python3
"""Private, hash-locked QAIC output reuse; never substitute a changed interface."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

# These identities are from the working pinned 6.6.0.0 SDK / unchanged interface.
# Generated SDK code remains private under .cache; only hashes are published.
INPUTS = {
    'ggml/src/ggml-hexagon/htp/htp_iface.idl': '2ab12174d817a6f42b8831b6256706815b9d07d3b738f852a9c867928ae1b935',
    '.cache/npu/toolchain/sdk/6.6.0.0/incs/stddef/AEEStdDef.idl': '0779a289a081b67dccaf48a9403bf33215017b3d6f4cc636318bd584072e1835',
    '.cache/npu/toolchain/sdk/6.6.0.0/incs/remote.idl': 'cd7cce7388d556a7ad07d3bc31f9891e72b938ec5af341831c19dff8acfd5caf',
    '.cache/npu/toolchain/sdk/6.6.0.0/ipc/fastrpc/qaic/bin/qaic': '29853bdf2e4ffad8d31e331b77ec5186eb1e9076c4ab2b3d035264fd81776dce',
}
OUTPUTS = {
    'htp_iface.h': 'c866b661e3cb36b98f9053be139c0973150b60244a6b40bf819726b754caf4fe',
    'htp_iface_stub.c': '8807abc3dfd89bdb66a4503c351dff12ed9d5e617fe499e5911b0ac6e92a25e5',
    'htp_iface_skel.c': '12bd4003731105e680425cd1d00c030ee44e64c901620df50eb50281d5af4228',
}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_inputs(root):
    for relative, expected in INPUTS.items():
        if digest(root / relative) != expected:
            raise ValueError('QAIC cache input identity changed: ' + relative)

def validate_cache(root):
    cache = root / '.cache/npu-max/qaic-cache'
    manifest = json.loads((cache / 'manifest.json').read_text())
    if manifest.get('inputs') != INPUTS or manifest.get('outputs') != OUTPUTS:
        raise ValueError('QAIC cache manifest does not match the published identities')
    validate_inputs(root)
    for name, expected in OUTPUTS.items():
        if digest(cache / name) != expected:
            raise ValueError('QAIC generated cache corrupted: ' + name)
    return cache

def parse(args):
    out = None
    includes = []
    idl = None
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == '-mdll':
            pass
        elif arg in ('-o', '-I'):
            idx += 1
            if idx >= len(args): return None
            if arg == '-o': out = Path(args[idx])
            else: includes.append(Path(args[idx]))
        elif arg.startswith('-I'):
            includes.append(Path(arg[2:]))
        elif arg.endswith('.idl') and not arg.startswith('-') and idl is None:
            idl = Path(arg)
        else:
            return None
        idx += 1
    if out is None or idl is None: return None
    return out.resolve(), idl.resolve(), [path.resolve() for path in includes]

def invoke(root, args):
    parsed = parse(args)
    cache_dir = root / '.cache/npu-max/qaic-cache'
    use_cache = os.environ.get('QWEN_NPU_MAX_QAIC_CACHE', '1') != '0'
    if use_cache and parsed and parsed[1] == (root / 'ggml/src/ggml-hexagon/htp/htp_iface.idl') and cache_dir.exists():
        out, idl, includes = parsed
        cache = validate_cache(root)
        # Include-search precedence must not shadow either pinned dependency.
        for basename, relative in [('AEEStdDef.idl', '.cache/npu/toolchain/sdk/6.6.0.0/incs/stddef/AEEStdDef.idl'),
                                   ('remote.idl', '.cache/npu/toolchain/sdk/6.6.0.0/incs/remote.idl')]:
            found = next((p / basename for p in [idl.parent, *includes] if (p / basename).is_file()), None)
            if found is None or found.resolve() != (root / relative).resolve():
                raise ValueError('QAIC cache refuses changed include resolution: ' + basename)
        out.mkdir(parents=True, exist_ok=True)
        for name in OUTPUTS:
            with tempfile.NamedTemporaryFile(dir=out, delete=False) as temp:
                temp.write((cache / name).read_bytes())
                temporary = Path(temp.name)
            os.replace(temporary, out / name)
        print('QAIC: reused verified private generated interface (unchanged IDL/SDK)', file=sys.stderr)
        return 0
    # A fresh installation has no generated cache. Use the genuine SDK QAIC;
    # the native preprocessor bypasses its unreliable emulated child process.
    genuine = root / '.cache/npu/toolchain/emulation/bin/qaic'
    cpp = shutil.which('cpp')
    if not cpp: raise ValueError('NPU-Max build requires native cpp')
    os.execv(str(genuine), [str(genuine), '-p', cpp, *args])

if __name__ == '__main__':
    try:
        root = Path(sys.argv[1]).resolve()
        raise SystemExit(invoke(root, sys.argv[2:]))
    except (OSError, ValueError, KeyError) as exc:
        print('QAIC cache error: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
