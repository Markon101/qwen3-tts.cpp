#!/usr/bin/env python3
"""Extract pinned native Termux build-tool packages into a private directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main(root):
    pins = json.loads((root / 'tools/npu/toolchain-pins.json').read_text())
    cache = root / '.cache/npu/android-packages'
    runtime = root / '.cache/npu/android-runtime'
    cache.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    for package in pins['native_runtime']:
        path = cache / package['filename']
        if not path.exists():
            subprocess.run(['apt', 'download', package['name'] + '=' + package['version']],
                           cwd=cache, check=True)
        if (path.is_symlink() or path.stat().st_size != package['size'] or
                hashlib.sha256(path.read_bytes()).hexdigest() != package['sha256']):
            raise RuntimeError('Package differs from pinned bytes: ' + str(path))
        subprocess.run(['dpkg-deb', '-x', str(path), str(runtime)], check=True)
    prefix = runtime / 'data/data/com.termux/files/usr'
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(prefix / 'lib')
    env.pop('LD_PRELOAD', None)
    subprocess.run([str(prefix / 'bin/qemu-x86_64'), '--version'], env=env, check=True)
    print('Private native QEMU runtime ready; no global package install performed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path, nargs='?', default=Path('.'))
    args = parser.parse_args()
    main(args.project.resolve())
