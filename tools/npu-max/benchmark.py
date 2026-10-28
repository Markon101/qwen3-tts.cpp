#!/usr/bin/env python3
"""Bounded serial original/full-HTP controls with a fresh foreground gate.

Cold means process-cold, not Android filesystem-cache cold. Provenance hashing
itself reads the models before launch. Raw requests, audio, codes, telemetry and
paths stay private under ignored .cache. No temperature limits are imposed.
"""
import argparse
import array
import datetime
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import threading
import time
import wave

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/android'))
from adreno_telemetry import sample
from telemetry import battery_sample, energy_summary

LIBRARIES = ('libggml.so', 'libggml-base.so', 'libggml-cpu.so',
             'ggml-hexagon/libggml-hexagon.so', 'ggml-hexagon/libggml-htp-v79.so',
             'ggml-opencl/libggml-opencl.so')
TEXT = ('The little fox wandered through the quiet forest, following the silver river toward the distant mountains. '
        'Along the way it listened to the birds and watched sunlight dancing through the leaves.')


class RunStopped(RuntimeError):
    pass


def write_json(path, value):
    """Replace a report atomically; interruptions retain the previous complete JSON."""
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def foreground():
    if os.environ.get('QWEN_BENCHMARK_ALLOW_BACKGROUND') == '1':
        return True
    try:
        return '/top-app' in Path('/proc/self/cgroup').read_text()
    except OSError:
        return False


def check_running(state):
    if state['signal'] is not None:
        raise RunStopped('signal ' + signal.Signals(state['signal']).name)


def require_foreground(state):
    check_running(state)
    if not foreground():
        raise RunStopped('foreground cpuset lost or unreadable; comparison invalidated')


def file_identity(path, state):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    if not path.is_file():
        raise RuntimeError('not a regular file: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            check_running(state)
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    after = path.stat()
    fields = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if fields(before) != fields(after):
        raise RuntimeError('file changed while hashing: ' + str(path))
    return {'path': str(path), 'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns,
            'sha256': digest.hexdigest()}


def git_identity(path):
    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(path), *arguments], stderr=subprocess.DEVNULL)
    try:
        return {'head': git('rev-parse', 'HEAD').decode().strip(),
                'status': git('status', '--short').decode(),
                'tracked_diff_sha256': hashlib.sha256(git('diff', '--binary', 'HEAD')).hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        return {'available': False}


def native_root(variant):
    return ROOT / ('.cache/npu' if variant == 'original' else '.cache/npu-max')


def provenance(args, state):
    models = [args.models_dir / ('qwen3-tts-' + args.size + '-q8_0.gguf'),
              args.models_dir / 'qwen3-tts-tokenizer-f16-f32.gguf']
    config = args.models_dir / 'generation_config.json'
    if config.is_file():
        models.append(config)
    runtimes = {}
    for runtime in sorted({str(native_root(v)) for v in args.variants}):
        base = Path(runtime)
        paths = [base / 'app-build/qwen3-tts-cli', *[base / 'build/src' / name for name in LIBRARIES]]
        runtimes[runtime] = [file_identity(path, state) for path in paths]
    assets = [p for p in (args.reference, args.embedding, args.ref_text_file) if p]
    return {'app_source': git_identity(ROOT), 'ggml_source': git_identity(ROOT / 'ggml'),
            'benchmark_script': file_identity(__file__, state),
            'models': [file_identity(path, state) for path in models],
            'private_voice_inputs': [file_identity(path, state) for path in assets],
            'runtimes': runtimes,
            'baseline_definition': 'Existing .cache/npu executable and libraries, identified by their actual hashes; source correspondence is not assumed.',
            'cold_definition': 'New process; page cache is not dropped and provenance hashing reads model files first.'}


def environment(args, variant, directory, trace):
    base = native_root(variant) / 'build/src'
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('QWEN3_TTS_', 'GGML_HEXAGON_', 'GGML_PROFILE', 'GGML_OPENCL_')):
            env.pop(key)
    backend = {'original': 'hexagon', 'htp': 'hexagon', 'mixed': 'hexagon', 'cpu': 'cpu', 'opencl': 'opencl'}[variant]
    env.update(LD_LIBRARY_PATH=f'{base}:{base}/ggml-hexagon:{base}/ggml-opencl:/system/lib64:/vendor/lib64',
               LD_PRELOAD='', ADSP_LIBRARY_PATH=str(base / 'ggml-hexagon'), OCL_ICD_VENDORS='/vendor/lib64/libOpenCL.so',
               GGML_HEXAGON_HOSTBUF='0', GGML_HEXAGON_NHVX=str(args.nhvx), GGML_HEXAGON_OPPOLL=args.poll,
               GGML_HEXAGON_PROFILE='0', QWEN3_TTS_TRANSFORMER_BACKEND=backend, QWEN3_TTS_ENCODER_BACKEND=backend,
               QWEN3_TTS_DECODER_BACKEND='opencl' if variant in ('original', 'mixed') else backend,
               QWEN3_TTS_MAX_VOCODER='1' if variant == 'htp' else '0',
               QWEN3_TTS_ENCODER_FFT='0' if variant == 'original' else '1',
               QWEN3_TTS_MAX_STRICT='1' if variant == 'htp' else '0',
               QWEN3_TTS_VALIDATE_AUDIO='1', QWEN3_TTS_VALIDATE_LOGITS='1',
               QWEN3_TTS_SEED='20261026', QWEN3_TTS_CODES_DIR=str(directory),
               GGML_OPENCL_BINARY_CACHE_DIR=str(ROOT / '.cache/adreno830-opencl'))
    if trace:
        env['QWEN3_TTS_EXECUTION_TRACE'] = str(directory / 'component-trace.jsonl')
    return env


def read_response(child, pending, state, background, timeout=240):
    deadline = time.monotonic() + timeout
    while b'\n' not in pending:
        check_running(state)
        if background.is_set():
            raise RunStopped('active background telemetry observed; comparison invalidated')
        if time.monotonic() >= deadline:
            raise RuntimeError('response timeout')
        if select.select([child.stdout], [], [], min(.1, max(0, deadline - time.monotonic())))[0]:
            block = os.read(child.stdout.fileno(), 65536)
            if not block:
                raise RuntimeError('server stdout closed before a complete response')
            pending.extend(block)
            if len(pending) > 1024 * 1024:
                raise RuntimeError('server response exceeds 1 MiB')
    line, _, remaining = pending.partition(b'\n')
    pending[:] = remaining
    response = json.loads(line.decode('utf-8'))
    if not isinstance(response, dict):
        raise RuntimeError('server response is not an object')
    return response


def verify_outputs(directory, index, response, expected_frames, previous_codes, state):
    expected_wav = directory / (str(index) + '.wav')
    if Path(response.get('output', '')).resolve() != expected_wav.resolve():
        raise RuntimeError('response output does not match this request')
    expected_codes = directory / ('codes-' + str(index + 1) + '.i32')
    added_codes = set(directory.glob('codes-*.i32')) - previous_codes
    if added_codes != {expected_codes}:
        raise RuntimeError('missing, extra or mismatched per-request speech-code artifact')
    raw_codes = expected_codes.read_bytes()
    if not raw_codes or len(raw_codes) % 64 or len(raw_codes) > expected_frames * 64:
        raise RuntimeError('invalid speech-code frame shape or frame budget')
    codes = array.array('i')
    codes.frombytes(raw_codes)
    if sys.byteorder != 'little':
        codes.byteswap()
    frames = len(codes) // 16
    if min(codes) < 0 or max(codes) > 2047:
        raise RuntimeError('speech code outside valid 0..2047 range')
    unique = [len(set(codes[book::16])) for book in range(16)]
    if len(set(codes)) < 2 or (frames >= 16 and max(unique) < 2):
        raise RuntimeError('collapsed speech codes; finite WAV alone is not a quality gate')
    with wave.open(str(expected_wav), 'rb') as wav:
        rate, samples, channels, width = wav.getframerate(), wav.getnframes(), wav.getnchannels(), wav.getsampwidth()
        if rate != 24000 or channels != 1 or width != 2 or samples != frames * 1920:
            raise RuntimeError('WAV shape does not match the retained speech codes')
        pcm = array.array('h')
        pcm.frombytes(wav.readframes(samples))
    if sys.byteorder != 'little':
        pcm.byteswap()
    if len(pcm) != samples or not pcm or not any(pcm):
        raise RuntimeError('truncated or silent WAV')
    if abs(float(response['duration_s']) - samples / rate) > .001:
        raise RuntimeError('reported duration does not match WAV sample count')
    return {'speech_codes': file_identity(expected_codes, state), 'wav': file_identity(expected_wav, state),
            'frames': frames, 'unique_codes': len(set(codes)), 'unique_per_codebook': unique,
            'pcm_peak': max(abs(value) for value in pcm),
            'pcm_rms': (sum(value * value for value in pcm) / len(pcm)) ** .5,
            'fixed_frame_match': frames == expected_frames,
            'quality_scope': 'Shape, range, noncollapse and nonzero PCM controls; does not establish listening quality or numerical parity.'}


def run_variant(args, variant, directory, state, transcript, trace=False):
    require_foreground(state)
    base = native_root(variant)
    command = [str(base / 'app-build/qwen3-tts-cli'), '-m',
               str(args.models_dir / ('qwen3-tts-' + args.size + '-q8_0.gguf')),
               '--tokenizer', str(args.models_dir / 'qwen3-tts-tokenizer-f16-f32.gguf'), '--threads', '2', '--server']
    row = {'variant': variant, 'measurement_kind': 'component_trace_diagnostic' if trace else 'throughput',
           'command': command, 'responses': [], 'valid': False,
           'suitable_fixed_work_comparison': False, 'error': None, 'nhvx': args.nhvx, 'host_poll': args.poll,
           'thermal_policy': 'measurement_only', 'temperature_limits': None}
    write_json(directory / 'status.json', row)
    started = time.monotonic()
    stop, background = threading.Event(), threading.Event()
    counters = {'total_samples': 0, 'foreground_samples': 0, 'unknown_cgroup_samples': 0,
                'active_background_samples': 0, 'post_exit_samples': 0, 'monitor_wakeup_lag_max_ms': 0,
                'monitor_sample_latency_max_ms': 0}
    energy = []
    child, thread = None, None
    with (directory / 'stderr.log').open('x') as errors, (directory / 'requests.jsonl').open('x', encoding='utf-8') as requests, \
            (directory / 'responses.jsonl').open('x', encoding='utf-8') as responses, \
            (directory / 'telemetry.jsonl').open('x') as telemetry, (directory / 'battery.jsonl').open('x') as battery:
        try:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                     env=environment(args, variant, directory, trace), start_new_session=True)
            state['child'] = child

            def monitor():
                target = time.monotonic()
                try:
                    while not stop.is_set():
                        now = time.monotonic()
                        counters['monitor_wakeup_lag_max_ms'] = max(counters['monitor_wakeup_lag_max_ms'], 1000 * max(0, now - target))
                        observation = sample(child.pid)
                        counters['total_samples'] += 1
                        cgroup = observation.get('cgroup')
                        if (cgroup and '/top-app' in cgroup) or os.environ.get('QWEN_BENCHMARK_ALLOW_BACKGROUND') == '1':
                            counters['foreground_samples'] += 1
                        elif child.poll() is None:
                            # Android can move a naturally exiting process to
                            # the root cpuset. Give that exit a short grace,
                            # while invalidating genuinely active background work.
                            try:
                                child.wait(timeout=.03)
                                counters['post_exit_samples'] += 1
                            except subprocess.TimeoutExpired:
                                if cgroup:
                                    counters['active_background_samples'] += 1
                                else:
                                    counters['unknown_cgroup_samples'] += 1
                                background.set()
                        telemetry.write(json.dumps(observation) + '\n'); telemetry.flush()
                        reading = battery_sample()
                        energy.append(reading)
                        battery.write(json.dumps(reading) + '\n'); battery.flush()
                        counters['monitor_sample_latency_max_ms'] = max(counters['monitor_sample_latency_max_ms'], 1000 * (time.monotonic() - now))
                        target = time.monotonic() + .1
                        stop.wait(.1)
                except Exception as error:
                    row['monitor_error'] = repr(error)
                    background.set()

            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            pending = bytearray()
            index, sustained_start = 0, None
            count = 1 if trace else args.requests + 1
            while index < count or (not trace and args.sustained_seconds and
                    (sustained_start is None or time.monotonic() - sustained_start < args.sustained_seconds)):
                require_foreground(state)
                if background.is_set():
                    raise RunStopped('active background/unknown telemetry observed; comparison invalidated')
                request = {'text': TEXT, 'language': 'english', 'max_tokens': args.frames,
                           'output': str(directory / (str(index) + '.wav'))}
                if args.embedding:
                    request['embedding_in'] = str(args.embedding)
                if args.reference:
                    request['reference'] = str(args.reference)
                if transcript is not None:
                    request['ref_text'] = transcript
                # The legacy native parser does not implement JSON \\u escapes.
                wire = json.dumps(request, ensure_ascii=False)
                requests.write(wire + '\n'); requests.flush()
                existing_codes = set(directory.glob('codes-*.i32'))
                sent = time.monotonic()
                child.stdin.write((wire + '\n').encode('utf-8')); child.stdin.flush()
                response = read_response(child, pending, state, background, args.response_timeout)
                response.update(request=index, cold_process_request=index == 0,
                                wall_request_ms=1000 * (time.monotonic() - sent))
                responses.write(json.dumps(response, ensure_ascii=False) + '\n'); responses.flush()
                row['responses'].append(response)
                if not response.get('success'):
                    raise RuntimeError('native request failed: ' + str(response))
                response.update(verify_outputs(directory, index, response, args.frames, existing_codes, state))
                write_json(directory / 'responses.json', row['responses'])
                index += 1
                if sustained_start is None:
                    sustained_start = time.monotonic()
            child.stdin.close()
            if child.wait(timeout=30):
                raise RuntimeError('server exited nonzero')
            check_running(state)
            if background.is_set():
                raise RunStopped('background/unknown telemetry observed; comparison invalidated')
            require_foreground(state)
            row['valid'] = True
            row['suitable_fixed_work_comparison'] = not trace and all(r['fixed_frame_match'] for r in row['responses'])
            if not all(r['fixed_frame_match'] for r in row['responses']):
                row['fixed_work_limitation'] = 'Early EOS: actual work differs; do not compare fixed-budget latency as a speedup.'
        except Exception as error:
            row['error'] = ('RunStopped: signal ' + signal.Signals(state['signal']).name
                            if state['signal'] else type(error).__name__ + ': ' + str(error))
        finally:
            stop.set()
            if thread:
                thread.join(timeout=5)
                if thread.is_alive():
                    row['valid'] = False
                    row['error'] = 'monitor thread failed to stop within 5 seconds'
            if child is not None:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait()
                row['child_exit_code'] = child.returncode
                for stream in (child.stdin, child.stdout):
                    if stream and not stream.closed:
                        stream.close()
            state['child'] = None
            row['stop_signal'] = state['signal']
            row.update(counters)
            row['process_wall_s'] = time.monotonic() - started
            row['energy'] = energy_summary(energy)
            row['energy_scope'] = 'Whole device over process lifetime, including loading, telemetry and output verification; not isolated HTP energy.'
            row['responsiveness_scope'] = 'Monitor wakeup lag is a scheduling proxy, not Android UI latency.'
            if trace:
                trace_path = directory / 'component-trace.jsonl'
                row['component_trace_available'] = trace_path.is_file()
                row['component_compute'] = {}
                if trace_path.is_file():
                    try:
                        with trace_path.open(encoding='utf-8') as stream:
                            for line in stream:
                                event = json.loads(line)
                                if event.get('event') == 'compute':
                                    total = row['component_compute'].setdefault(event['component'], {'calls': 0, 'elapsed_ms': 0})
                                    total['calls'] += 1
                                    total['elapsed_ms'] += event['elapsed_ms']
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        row['valid'] = False
                        row['error'] = 'component trace cannot be parsed: ' + str(error)
                elif variant == 'original':
                    row['component_trace_limitation'] = 'Frozen baseline may predate node audit instrumentation; native component timing is retained in responses.'
                else:
                    row['valid'] = False
                    row['error'] = 'requested component audit trace is missing'
            write_json(directory / 'responses.json', row['responses'])
            write_json(directory / 'status.json', row)
    return row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foreground-confirmed', action='store_true')
    parser.add_argument('--models-dir', type=Path, default=ROOT.parent.parent / 'models')
    parser.add_argument('--size', choices=['0.6b', '1.7b'], default='0.6b')
    parser.add_argument('--variants', nargs='+', choices=['original', 'htp', 'mixed', 'cpu', 'opencl'], default=['original', 'htp', 'original'])
    parser.add_argument('--requests', type=int, default=3, help='Warm requests per variant, after one process-cold request.')
    parser.add_argument('--sustained-seconds', type=float, default=0, help='Additional minimum warm window after the cold request, at most 3600 seconds.')
    parser.add_argument('--frames', type=int, default=64)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--ref-text-file', type=Path)
    parser.add_argument('--embedding', type=Path)
    parser.add_argument('--nhvx', type=int, choices=range(1, 7), default=4)
    parser.add_argument('--poll', choices=['0', '1'], default='1')
    parser.add_argument('--component-trace', action='store_true', help='Separately reload each variant for one audited diagnostic request; excluded from throughput results.')
    parser.add_argument('--response-timeout', type=float, default=240)
    args = parser.parse_args(argv)
    if not args.foreground_confirmed:
        parser.error('confirm Termux foreground before this performance test')
    if not 1 <= args.requests <= 10000 or not 1 <= args.frames <= 2048 or not 0 <= args.sustained_seconds <= 3600:
        parser.error('invalid run bounds')
    if not 1 <= args.response_timeout <= 3600:
        parser.error('response timeout must be 1..3600 seconds')
    if args.reference and args.embedding:
        parser.error('choose reference or embedding')
    if args.ref_text_file and not args.reference:
        parser.error('transcript requires reference')
    args.models_dir = args.models_dir.resolve()
    for name in ('reference', 'embedding', 'ref_text_file'):
        value = getattr(args, name)
        if value:
            try:
                setattr(args, name, value.resolve(strict=True))
            except OSError as error:
                parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    state = {'signal': None, 'child': None}

    def on_signal(signum, _frame):
        if state['signal'] is None:
            state['signal'] = signum
        child = state['child']
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    previous = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    directory = None
    try:
        require_foreground(state)
        transcript = None
        if args.ref_text_file:
            with args.ref_text_file.open('rb') as stream:
                raw = stream.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise RuntimeError('reference transcript exceeds 1 MiB')
            transcript = raw.decode('utf-8-sig').strip()
            if not transcript or '\0' in transcript:
                raise RuntimeError('reference transcript must be nonempty UTF-8 text without NUL')
        tag = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + str(os.getpid())
        directory = ROOT / '.cache/npu-max/benchmarks' / tag
        directory.mkdir(parents=True, exist_ok=False)
        manifest = provenance(args, state)
        manifest.update(arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                        foreground_cgroup=Path('/proc/self/cgroup').read_text(), thermal_policy='measurement_only', temperature_limits=None)
        write_json(directory / 'manifest.json', manifest)
        results = []
        for index, variant in enumerate(args.variants):
            require_foreground(state)
            variant_directory = directory / (str(index) + '-' + variant)
            variant_directory.mkdir()
            result = run_variant(args, variant, variant_directory, state, transcript)
            results.append(result)
            write_json(directory / 'results.json', results)
            print(variant, 'valid=' + str(result['valid']), 'fixed_work=' + str(result['suitable_fixed_work_comparison']),
                  [(r.get('cold_process_request'), r.get('t_total_ms'), r.get('fixed_frame_match')) for r in result['responses']], flush=True)
            if not result['valid']:
                raise RunStopped(result['error'] or 'variant invalidated')
            if args.component_trace:
                require_foreground(state)
                diagnostic = directory / (str(index) + '-' + variant + '-component-trace')
                diagnostic.mkdir()
                trace_result = run_variant(args, variant, diagnostic, state, transcript, trace=True)
                results.append(trace_result)
                write_json(directory / 'results.json', results)
                if not trace_result['valid']:
                    raise RunStopped(trace_result['error'] or 'component trace invalidated')
        print('Private results:', directory)
        return 0
    except Exception as error:
        failure = {'error': type(error).__name__ + ': ' + str(error), 'signal': state['signal'],
                   'thermal_policy': 'measurement_only', 'temperature_limits': None}
        if directory:
            write_json(directory / 'failure.json', failure)
            print('Private partial results:', directory, file=sys.stderr)
        print(failure['error'], file=sys.stderr)
        return 128 + state['signal'] if state['signal'] else 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
