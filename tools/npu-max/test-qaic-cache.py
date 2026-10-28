#!/usr/bin/env python3
"""Private temp fixtures verify cached SDK generation identities fail closed."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import shutil
import tempfile

SOURCE = Path(__file__).with_name('qaic-wrapper.py')
ROOT = SOURCE.resolve().parents[2]
spec = spec_from_file_location('qaic_cache', SOURCE)
qaic = module_from_spec(spec)
spec.loader.exec_module(qaic)

with tempfile.TemporaryDirectory(prefix='qwen-qaic-cache-') as tmp:
    root = Path(tmp)
    for relative in qaic.INPUTS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, path)
    cache = root / '.cache/npu-max/qaic-cache'
    shutil.copytree(ROOT / '.cache/npu-max/qaic-cache', cache)
    idl = root / 'ggml/src/ggml-hexagon/htp/htp_iface.idl'
    sdk = root / '.cache/npu/toolchain/sdk/6.6.0.0'
    args = ['-mdll', '-o', str(root / 'generated'), '-I' + str(sdk / 'incs'),
            '-I' + str(sdk / 'incs/stddef'), str(idl)]
    assert qaic.invoke(root, args) == 0
    for name, expected in qaic.OUTPUTS.items():
        assert qaic.digest(root / 'generated' / name) == expected
    checks = 1
    def rejects_change(path, changed):
        global checks
        old = path.read_bytes()
        path.write_bytes(changed)
        try:
            try:
                qaic.invoke(root, args)
            except ValueError:
                checks += 1
            else:
                raise AssertionError('Changed cache identity accepted: ' + str(path))
        finally:
            path.write_bytes(old)
    rejects_change(idl, idl.read_bytes() + b'\n// changed interface\n')
    dep = sdk / 'incs/remote.idl'
    rejects_change(dep, dep.read_bytes() + b'\n// changed SDK dependency\n')
    binary = sdk / 'ipc/fastrpc/qaic/bin/qaic'
    rejects_change(binary, binary.read_bytes() + b'changed binary')
    generated = cache / 'htp_iface_stub.c'
    rejects_change(generated, generated.read_bytes() + b'corrupted')
    manifest = cache / 'manifest.json'
    altered = json.loads(manifest.read_text())
    altered['outputs']['htp_iface.h'] = '0' * 64
    rejects_change(manifest, json.dumps(altered).encode())
    shadow = idl.parent / 'remote.idl'
    shutil.copyfile(dep, shadow)
    try:
        qaic.invoke(root, args)
    except ValueError:
        checks += 1
    else:
        raise AssertionError('Shadow SDK dependency accepted')
    print(json.dumps({'kind': 'qaic_cache', 'checks': checks, 'pass': True}))
