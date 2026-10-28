#!/usr/bin/env python3
"""Prepare the isolated QAIC wrapper and, when available, verified private cache."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from importlib.util import module_from_spec, spec_from_file_location

R = Path(__file__).resolve().parents[2]
source = Path(__file__).with_name('qaic-wrapper.py')
spec = spec_from_file_location('npu_max_qaic', source)
qaic = module_from_spec(spec)
spec.loader.exec_module(qaic)
D = R / '.cache/npu-max/qaic-bin/bin'
D.mkdir(parents=True, exist_ok=True)
(D / 'qaic').write_text('#!' + sys.executable + '\nimport runpy,sys\nsys.argv=[sys.argv[0], ' + repr(str(R)) + ', *sys.argv[1:]]\nrunpy.run_path(' + repr(str(source)) + ',run_name="__main__")\n')
(D / 'qaic').chmod(0o700)
cache = R / '.cache/npu-max/qaic-cache'
try:
    if cache.exists():
        qaic.validate_cache(R)
        print('QAIC private cache: existing identities verified')
    else:
        baseline = R / '.cache/npu/build/src/ggml-hexagon'
        if not all((baseline / name).is_file() for name in qaic.OUTPUTS):
            print('QAIC private cache unavailable; wrapper will invoke genuine SDK')
        else:
            qaic.validate_inputs(R)
            for name, expected in qaic.OUTPUTS.items():
                if qaic.digest(baseline / name) != expected:
                    raise ValueError('Baseline generated interface is not the tested identity: ' + name)
            cache.mkdir(parents=True)
            for name in qaic.OUTPUTS:
                shutil.copyfile(baseline / name, cache / name)
            manifest = {'schema': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                        'source': str(baseline.relative_to(R)),
                        'sdk_archive_sha256': '4a916e42c1dab9efdf2e58773f901ea780fa43c907bf054ab76572a3b3d942f4',
                        'inputs': qaic.INPUTS, 'outputs': qaic.OUTPUTS,
                        'reason': 'Reuse working generated unchanged IDL; genuine QAIC under QEMU intermittently hangs with SDK includes.'}
            (cache / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            qaic.validate_cache(R)
            print('QAIC private cache imported and hash verified; no SDK code published')
except (OSError, ValueError, KeyError) as exc:
    raise SystemExit('Refusing unsafe QAIC cache reuse: ' + str(exc))
