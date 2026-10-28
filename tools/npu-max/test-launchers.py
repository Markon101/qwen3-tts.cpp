#!/usr/bin/env python3
"""Inference-free NPU-Max launcher checks using an isolated fake installation.

Copies production scripts into a temporary directory under this repository's
.cache. All model/library files are readable placeholders; only the fake CLI
runs. No SDK, model loading, synthesis, or hardware execution is performed.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ('npu-max.sh', 'tools/npu-max/native-env.sh',
           'tools/npu-max/launch.py', 'scripts/android/profile_runner.py',
           'scripts/android/adreno_telemetry.py')
LIBRARIES = ('libggml.so', 'libggml-base.so', 'libggml-cpu.so',
             'ggml-hexagon/libggml-hexagon.so',
             'ggml-hexagon/libggml-htp-v79.so', 'ggml-opencl/libggml-opencl.so')
CAPTURE_ENV = ('QWEN_MODELS_DIR', 'QWEN_NPU_MAX_PROFILE', 'QWEN3_TTS_TRANSFORMER_BACKEND',
               'QWEN3_TTS_ENCODER_BACKEND', 'QWEN3_TTS_ENCODER_FFT',
               'QWEN3_TTS_MAX_STRICT', 'QWEN3_TTS_MAX_VOCODER',
               'QWEN3_TTS_EXECUTION_TRACE',
               'QWEN3_TTS_DECODER_BACKEND', 'QWEN3_TTS_VALIDATE_AUDIO',
               'QWEN3_TTS_VALIDATE_LOGITS', 'QWEN3_TTS_FORCE_CPU',
               'GGML_HEXAGON_HOSTBUF', 'GGML_HEXAGON_NHVX',
               'GGML_HEXAGON_OPPOLL', 'GGML_HEXAGON_PROFILE',
               'GGML_HEXAGON_OPSTAGE', 'GGML_PROFILE_OPS',
               'GGML_OPENCL_QCOM_RECORD', 'GGML_OPENCL_Q8_SMALL2',
               'GGML_OPENCL_BINARY_CACHE_DIR', 'LD_LIBRARY_PATH',
               'LD_PRELOAD', 'ADSP_LIBRARY_PATH', 'OCL_ICD_VENDORS')
FAKE_CLI = r'''
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
record = {'argv': args, 'cwd': os.getcwd(), 'pid': os.getpid(),
          'environment': {key: os.environ[key] for key in CAPTURE_ENV if key in os.environ}}
Path(os.environ['QWEN_NPU_TEST_RECORD']).write_text(json.dumps(record))
if os.environ.get('QWEN_NPU_TEST_SLEEP'):
    while True:
        time.sleep(.05)
if os.environ.get('QWEN_NPU_TEST_EXIT'):
    raise SystemExit(int(os.environ['QWEN_NPU_TEST_EXIT']))
if '--server' in args:
    for line in sys.stdin:
        if line.strip():
            print(json.dumps({'success': True, 'request': json.loads(line)}), flush=True)
    raise SystemExit(0)
values = {}
i = 0
while i < len(args):
    arg = args[i]
    if arg in VALUE_FLAGS and i + 1 < len(args):
        values[ALIASES.get(arg, arg)] = args[i + 1]
        i += 2
    else:
        i += 1
info = any(arg in args for arg in ('--help', '-h', '--version', '--list-speakers', '--list-languages'))
extract = values.get('--embedding-out') and values.get('--reference') and not values.get('--text')
if not info and not extract and not values.get('--text'):
    print('Fake CLI: text required for synthesis', file=sys.stderr)
    raise SystemExit(1)
print(json.dumps({'fake_cli': True, 'extraction_only': bool(extract)}))
'''
# Mirror the CLI's value-taking flags to distinguish flags from text values.
VALUE_FLAGS = set(('-m --model --tokenizer -t --text -o --output -r --reference '
                   '--ref-text --embedding-in --embedding-out --speaker --instruct '
                   '-l --language --temperature --top-k --top-p --min-p '
                   '--repetition-penalty --frequency-penalty --presence-penalty '
                   '--dry-multiplier --dyntemp-range --sub-temperature --sub-top-k '
                   '--sub-top-p --max-tokens --output-rate -j --threads --gen-config').split())
ALIASES = {'-r': '--reference', '-t': '--text', '-o': '--output', '-m': '--model'}


def put(path, text='placeholder\n'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def install(path, assets=True):
    for name in SCRIPTS:
        dest = path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    if assets:
        fake = ('#!' + sys.executable + '\n'
                + 'CAPTURE_ENV = ' + repr(CAPTURE_ENV) + '\n'
                + 'VALUE_FLAGS = ' + repr(VALUE_FLAGS) + '\n'
                + 'ALIASES = ' + repr(ALIASES) + '\n' + FAKE_CLI)
        cli = path / '.cache/npu-max/app-build/qwen3-tts-cli'
        put(cli, fake)
        cli.chmod(0o700)
        models(path / 'models')
        for name in LIBRARIES:
            put(path / '.cache/npu-max/build/src' / name)
    return path


def models(path):
    for size in ('0.6b', '1.7b'):
        put(path / ('qwen3-tts-' + size + '-q8_0.gguf'))
    put(path / 'qwen3-tts-tokenizer-f16-f32.gguf')


def parsed(arguments):
    values, flags = {}, []
    i = 0
    while i < len(arguments):
        arg = arguments[i]
        if arg in VALUE_FLAGS:
            assert i + 1 < len(arguments), ('missing fake CLI argument', arg)
            values[ALIASES.get(arg, arg)] = arguments[i + 1]
            i += 2
        else:
            flags.append(arg)
            i += 1
    return values, flags


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def cleanup(proc, record):
    """Reap the profile parent and stop only this invocation's fake CLI group."""
    if proc.poll() is None:
        # Give the production runner time to forward the signal and reap its
        # child itself before using the bounded emergency cleanup below.
        proc.terminate()
        try:
            proc.communicate(timeout=12)
        except subprocess.TimeoutExpired:
            pass
    child_pid = None
    try:
        child_pid = json.loads(record.read_text())['pid']
    except (OSError, ValueError, KeyError):
        pass
    if child_pid is not None and alive(child_pid):
        try:
            os.killpg(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.communicate(timeout=5)


class Checks:
    def __init__(self, base):
        self.repo = install(base / 'fake install with spaces')
        self.bare = install(base / 'help without assets', assets=False)
        self.partial = install(base / 'partial build help')
        (self.partial / '.cache/npu-max/build/src/libggml.so').unlink()
        self.cwd = base / 'caller with spaces'
        self.cwd.mkdir()
        self.reference = self.cwd / 'reference voice.wav'
        self.embedding = self.cwd / 'saved voice.bin'
        put(self.reference)
        # The launcher validates structure, while the real C++ model validates
        # model-specific width. The fake CLI deliberately does no width check.
        payload = struct.pack('<f', .25) * 1024
        self.embedding.write_bytes(struct.pack('<I', 1024) + payload)
        self.bad_embeddings = {}
        for kind, data in {
                'corrupted header': b'bad!',
                'truncated header': b'\x00\x04\x00',
                'truncated data': struct.pack('<I', 1024) + payload[:-4],
                'nonmatching header': struct.pack('<I', 1025) + payload,
                'empty': b'',
                'zero count': struct.pack('<I', 0),
        }.items():
            filename = self.cwd / (kind + ' embedding.bin')
            filename.write_bytes(data)
            self.bad_embeddings[kind] = filename
        oversized = self.cwd / 'oversized embedding.bin'
        with oversized.open('wb') as stream:
            count = 16 * 1024 * 1024 // 4
            stream.write(struct.pack('<I', count))
            stream.truncate(4 + 4 * count)
        self.bad_embeddings['oversized'] = oversized
        self.bad_embeddings['reference WAV as embedding'] = self.reference
        self.transcript = self.cwd / 'reference transcript UTF-8.txt'
        self.transcript_text = 'Café — 你好!\n第二行 with spaces.'
        put(self.transcript, ' \n\t' + self.transcript_text + '\n ')
        self.bom_transcript = self.cwd / 'reference transcript UTF-8 BOM.txt'
        self.bom_transcript.write_bytes(b'\xef\xbb\xbf' + self.transcript.read_bytes())
        self.bad_transcripts = {
            'empty': self.cwd / 'empty transcript.txt',
            'missing': self.cwd / 'missing transcript.txt',
            'non-UTF8': self.cwd / 'invalid UTF-8 transcript.txt',
            'NUL': self.cwd / 'NUL transcript.txt',
            'oversized': self.cwd / 'oversized transcript.txt',
        }
        put(self.bad_transcripts['empty'], ' \n\t')
        self.bad_transcripts['non-UTF8'].write_bytes(b'\xff\xfe invalid UTF-8')
        put(self.bad_transcripts['NUL'], 'before\0after')
        put(self.bad_transcripts['oversized'], 'x' * (1024 * 1024 + 1))
        self.results = []
        self.serial = 0
        self.env = os.environ.copy()
        for key in list(self.env):
            if key.startswith(('QWEN', 'GGML')) or key in (
                    'LD_LIBRARY_PATH', 'LD_PRELOAD', 'ADSP_LIBRARY_PATH', 'OCL_ICD_VENDORS'):
                self.env.pop(key)

    def invoke(self, entry='npu-max.sh', args=(), env=None, stdin='', repo=None, code=0):
        self.serial += 1
        record = self.cwd / ('capture-' + str(self.serial) + '.json')
        child_env = self.env.copy()
        child_env['QWEN_NPU_TEST_RECORD'] = str(record)
        child_env.update(env or {})
        command = [sys.executable if entry.endswith('.py') else 'sh',
                   str((repo or self.repo) / entry), *args]
        proc = subprocess.Popen(command, cwd=self.cwd, env=child_env,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(input=stdin, timeout=15)
            result = subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
            assert result.returncode == code, (result.returncode, code, result.stderr)
            capture = json.loads(record.read_text()) if record.exists() else None
            return capture, result
        finally:
            cleanup(proc, record)

    def check(self, name, callback):
        try:
            callback()
            self.results.append({'name': name, 'passed': True})
        except Exception as error:
            self.results.append({'name': name, 'passed': False,
                                 'error': type(error).__name__ + ': ' + str(error)})

    def basic(self, size, entry):
        capture, _ = self.invoke(entry, (size, '--text', 'Hello world'))
        values, _ = parsed(capture['argv'])
        assert values['--model'] == str(self.repo / 'models' / ('qwen3-tts-' + size + '-q8_0.gguf'))
        assert values['--text'] == 'Hello world'
        env = capture['environment']
        assert env['QWEN3_TTS_TRANSFORMER_BACKEND'] == 'hexagon'
        assert env['QWEN3_TTS_DECODER_BACKEND'] == 'hexagon'
        assert env['QWEN3_TTS_ENCODER_BACKEND'] == 'hexagon'
        assert env['QWEN3_TTS_MAX_VOCODER'] == env['QWEN3_TTS_MAX_STRICT'] == '1'
        assert env['QWEN3_TTS_ENCODER_FFT'] == '1'
        assert Path(env['QWEN3_TTS_EXECUTION_TRACE']).parent == self.repo / '.cache/npu-max/traces'
        assert env['QWEN3_TTS_VALIDATE_AUDIO'] == env['QWEN3_TTS_VALIDATE_LOGITS'] == '1'
        assert env['GGML_HEXAGON_HOSTBUF'] == '0'
        assert env['GGML_HEXAGON_NHVX'] == '4' and env['GGML_HEXAGON_OPPOLL'] == '1'
        assert capture['cwd'] == str(self.cwd)
        library_root = self.repo / '.cache/npu-max/build/src'
        assert env['LD_LIBRARY_PATH'].split(':')[0] == str(library_root)
        assert env['ADSP_LIBRARY_PATH'] == str(library_root / 'ggml-hexagon')
        assert env['OCL_ICD_VENDORS'] == '/vendor/lib64/libOpenCL.so'
        assert 'LD_PRELOAD' not in env

    def demo(self, size, preset=False):
        env = {'QWEN_NPU_DEMO_EMBEDDING_06B': str(self.embedding)} if preset else {}
        capture, _ = self.invoke(args=(size,), env=env)
        values, _ = parsed(capture['argv'])
        assert values['--text'] == 'Hello from the Hexagon NPU.'
        assert ('--embedding-in' in values) == (preset and size == '0.6b')
        if preset and size == '0.6b':
            assert values['--embedding-in'] == str(self.embedding)

    def mode(self, size, mode):
        options = {'reference': ['-r', str(self.reference), '--text', 'Clone'],
                   'icl': ['-r', str(self.reference), '--ref-text', 'Reference transcript', '--text', 'Clone'],
                   'embedding': ['--embedding-in', str(self.embedding), '--text', 'Clone'],
                   'extract': ['-r', str(self.reference), '--embedding-out', str(self.cwd / 'new embedding.bin')]}
        capture, result = self.invoke(args=(size, *options[mode]),
                                      env={'QWEN_NPU_DEMO_EMBEDDING_06B': str(self.embedding)})
        values, _ = parsed(capture['argv'])
        if mode in ('reference', 'icl', 'extract'):
            assert values['--reference'] == str(self.reference)
            assert '--embedding-in' not in values
        if mode == 'icl':
            assert values['--ref-text'] == 'Reference transcript'
        if mode == 'embedding':
            assert values['--embedding-in'] == str(self.embedding)
        if mode == 'extract':
            assert '--text' not in values and '--output' not in values
            assert json.loads(result.stdout)['extraction_only']

    def transcript_file(self, size, entry, bom=False):
        transcript = self.bom_transcript if bom else self.transcript
        capture, _ = self.invoke(entry, (size, '-r', str(self.reference),
                                        '--ref-text-file', transcript.name,
                                        '--text', 'Clone from transcript file'))
        values, flags = parsed(capture['argv'])
        assert values['--ref-text'] == self.transcript_text
        assert values['--reference'] == str(self.reference)
        assert '--ref-text-file' not in capture['argv']
        assert '--ref-text-file' not in flags
        assert values['--text'] == 'Clone from transcript file'

    def literal_transcript_flag(self):
        capture, _ = self.invoke(args=('0.6b', '--text', '--ref-text-file'))
        values, flags = parsed(capture['argv'])
        assert values['--text'] == '--ref-text-file'
        assert '--ref-text' not in values and '--ref-text-file' not in flags

    def empty_transcript_embedding(self, size):
        capture, _ = self.invoke(args=(size, '--embedding-in', str(self.embedding),
                                      '--ref-text', '', '--text', 'Embedding without ICL'))
        values, _ = parsed(capture['argv'])
        assert values['--ref-text'] == ''
        assert values['--embedding-in'] == str(self.embedding)
        assert '--reference' not in values

    def help_transcript(self, size, entry):
        capture, result = self.invoke(entry, (size, '--ref-text-file', 'missing transcript.txt',
                                             '--help'), repo=self.bare)
        assert capture is None and 'Usage:' in result.stdout
        assert not (self.bare / '.cache').exists()

    def playback(self, size, explicit_output=False):
        options = [size, '--text', 'Playback only', '-s']
        if explicit_output:
            options += ['--output', 'explicit playback output.wav']
        capture, _ = self.invoke(args=options)
        values, flags = parsed(capture['argv'])
        assert '-s' in flags
        if explicit_output:
            assert values['--output'] == 'explicit playback output.wav'
        else:
            assert '--output' not in values

    def generate_and_save_embedding(self, size):
        output = self.repo / '.cache/npu-max/demo output.wav'
        embedding = self.cwd / 'generated embedding.bin'
        capture, result = self.invoke(args=(size, '-r', str(self.reference),
                                            '--embedding-out', str(embedding),
                                            '--text', 'Generate and retain embedding'),
                                      env={'QWEN_NPU_OUTPUT': str(output)})
        values, _ = parsed(capture['argv'])
        assert values['--text'] == 'Generate and retain embedding'
        assert values['--embedding-out'] == str(embedding)
        assert values['--output'] == str(output)
        assert not json.loads(result.stdout)['extraction_only']

    def reject(self, arguments):
        capture, result = self.invoke(args=('0.6b', *arguments), code=2)
        assert capture is None, 'invalid invocation reached fake CLI'
        assert 'NPU-Max launcher:' in result.stderr

    def passthrough(self):
        capture, _ = self.invoke(args=('0.6b', '--text', '--reference', '--non-streaming'),
                                env={'QWEN_NPU_DEMO_EMBEDDING_06B': str(self.embedding)})
        values, flags = parsed(capture['argv'])
        assert values['--text'] == '--reference' and '--reference' not in values
        assert '--non-streaming' in flags

    def empty_text(self):
        capture, _ = self.invoke(args=('0.6b', '--text', ''), code=1)
        values, _ = parsed(capture['argv'])
        assert values['--text'] == ''

    def server(self, size):
        requests = [{'text': 'Hello --reference', 'reference': 'voice with spaces.wav'},
                    {'text': 'Second', 'embedding_in': 'voice.bin'}]
        stdin = ''.join(json.dumps(request) + '\n' for request in requests)
        capture, result = self.invoke(args=(size, '--server'), stdin=stdin,
                                      env={'QWEN_NPU_DEMO_EMBEDDING_06B': str(self.embedding)})
        assert [json.loads(line) for line in result.stdout.splitlines()] == [
            {'success': True, 'request': request} for request in requests]
        values, flags = parsed(capture['argv'])
        assert '--server' in flags
        assert not set(values).intersection(('--text', '--reference', '--ref-text', '--embedding-in', '--output'))
        assert 'Thermal/clock trace:' in result.stderr

    def help(self, entry):
        capture, result = self.invoke(entry, ('--help',), repo=self.bare)
        assert capture is None and 'Usage:' in result.stdout
        assert not (self.bare / '.cache').exists()

    def partial_build_help(self, entry):
        capture, result = self.invoke(entry, ('0.6b', '--help', '--embedding-in',
                                             str(self.bad_embeddings['empty'])), repo=self.partial)
        assert capture is None and 'Usage:' in result.stdout
        assert not (self.partial / '.cache/tts-runs').exists()

    def version_ignores_voice(self):
        capture, _ = self.invoke(args=('0.6b', '--version', '--embedding-in',
                                      str(self.bad_embeddings['empty'])))
        assert capture is not None
        _, flags = parsed(capture['argv'])
        assert '--version' in flags

    def invalid_demo_embedding(self):
        capture, result = self.invoke(args=('0.6b',), code=2,
                                      env={'QWEN_NPU_DEMO_EMBEDDING_06B':
                                           str(self.bad_embeddings['empty'])})
        assert capture is None and 'NPU-Max launcher:' in result.stderr

    def custom_models(self):
        directory = self.cwd / 'custom models'
        models(directory)
        capture, _ = self.invoke(args=('1.7b', '--text', 'External'),
                                env={'QWEN_MODELS_DIR': str(directory)})
        values, _ = parsed(capture['argv'])
        assert values['--model'] == str(directory / 'qwen3-tts-1.7b-q8_0.gguf')
        assert values['--tokenizer'] == str(directory / 'qwen3-tts-tokenizer-f16-f32.gguf')

    def relative_paths(self):
        capture, _ = self.invoke(args=('0.6b', '-r', self.reference.name, '--text', 'Caller paths',
                                      '-o', 'output with spaces.wav'))
        values, _ = parsed(capture['argv'])
        assert values['--reference'] == self.reference.name
        assert values['--output'] == 'output with spaces.wav'
        assert capture['cwd'] == str(self.cwd)

    def child_exit(self):
        self.invoke(args=('0.6b', '--text', 'Exit test'), env={'QWEN_NPU_TEST_EXIT': '23'}, code=23)

    def log_root(self):
        status_paths = list((self.repo / '.cache/tts-runs').glob('*.status.json'))
        assert status_paths, 'no status logs in isolated repository'
        for path in status_paths:
            status = json.loads(path.read_text())
            trace = Path(status['trace'])
            assert trace.parent == self.repo / '.cache/tts-runs' and trace.is_file()

    def signal_cleanup(self, signum):
        expected_status = 128 + int(signum)
        signal_name = signal.Signals(signum).name
        record = self.cwd / ('signal-child-' + signal_name + '.json')
        env = self.env.copy()
        env.update(QWEN_NPU_TEST_RECORD=str(record), QWEN_NPU_TEST_SLEEP='1')
        proc = subprocess.Popen(['sh', str(self.repo / 'npu-max.sh'), '0.6b', '--text', 'Signal test'],
                                cwd=self.cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        child_pid = None
        try:
            deadline = time.monotonic() + 10
            while not record.exists() and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(.025)
            assert record.exists(), 'fake child did not start'
            child_pid = json.loads(record.read_text())['pid']
            assert child_pid != proc.pid and alive(child_pid)
            proc.send_signal(signum)
            stdout, stderr = proc.communicate(timeout=15)
            assert proc.returncode == expected_status, (proc.returncode, stdout, stderr)
            assert not alive(child_pid), 'fake child remains alive or unreaped'
            statuses = [json.loads(path.read_text()) for path in
                        (self.repo / '.cache/tts-runs').glob('*.status.json')]
            assert any(status['stop_reason'] == 'signal ' + signal_name
                       and status['exit_status'] == expected_status for status in statuses)
        finally:
            cleanup(proc, record)

    def profile(self, size, profile, mode):
        options = {
            'none': ['--text', 'Ordinary speech'],
            'reference': ['-r', str(self.reference), '--text', 'Clone'],
            'icl': ['-r', str(self.reference), '--ref-text', 'Transcript', '--text', 'Clone'],
            'file_icl': ['-r', str(self.reference), '--ref-text-file', str(self.transcript), '--text', 'Clone'],
            'embedding': ['--embedding-in', str(self.embedding), '--text', 'Clone'],
            'extract': ['-r', str(self.reference), '--embedding-out', str(self.cwd / 'profile-out.bin')],
            'server': ['--server'],
            'named_voice': ['--speaker', 'Vivian', '--text', 'Named voice'],
            'voice_design': ['--instruct', 'A gentle voice', '--text', 'Designed voice'],
            'non_streaming_icl': ['--non-streaming', '-r', str(self.reference), '--ref-text', 'Transcript', '--text', 'Clone'],
        }
        capture, _ = self.invoke(args=(size, *options[mode]), env={'QWEN_NPU_MAX_PROFILE': profile})
        env = capture['environment']
        transformer = {'htp': 'hexagon', 'mixed': 'hexagon', 'cpu': 'cpu', 'opencl': 'opencl'}[profile]
        assert env['QWEN3_TTS_TRANSFORMER_BACKEND'] == transformer
        assert env['QWEN3_TTS_ENCODER_BACKEND'] == transformer
        assert env['QWEN3_TTS_DECODER_BACKEND'] == ('opencl' if profile == 'mixed' else transformer)
        assert env['QWEN3_TTS_MAX_STRICT'] == env['QWEN3_TTS_MAX_VOCODER'] == ('1' if profile == 'htp' else '0')
        assert env['QWEN3_TTS_ENCODER_FFT'] == '1'
        assert env['QWEN3_TTS_VALIDATE_AUDIO'] == env['QWEN3_TTS_VALIDATE_LOGITS'] == '1'

    def explicit_models(self, size):
        custom_model = self.cwd / ('custom ' + size + ' model.gguf')
        custom_tokenizer = self.cwd / 'custom tokenizer.gguf'
        put(custom_model)
        put(custom_tokenizer)
        capture, _ = self.invoke(args=(size, '-m', str(custom_model), '--tokenizer', str(custom_tokenizer),
                                      '--text', 'Custom models'),
                                env={'QWEN_NPU_DEMO_EMBEDDING_06B': str(self.embedding)})
        values, _ = parsed(capture['argv'])
        assert values['--model'] == str(custom_model)
        assert values['--tokenizer'] == str(custom_tokenizer)
        assert '--embedding-in' not in values, 'custom model inherited a model-specific preset'

    def invalid_environment(self, variable, value):
        capture, result = self.invoke(args=('0.6b', '--text', 'Rejected'), env={variable: value}, code=2)
        assert capture is None, 'invalid environment reached fake CLI'
        assert 'NPU-Max' in result.stderr

    def valid_threads(self, value):
        capture, _ = self.invoke(args=('1.7b', '--text', 'Threads'), env={'GGML_HEXAGON_NHVX': value})
        assert capture['environment']['GGML_HEXAGON_NHVX'] == value

    def dangerous_inherited_environment(self):
        inherited = dict(QWEN3_TTS_FORCE_CPU='1', GGML_PROFILE_OPS='1', GGML_OPENCL_QCOM_RECORD='1',
                         GGML_OPENCL_Q8_SMALL2='1', GGML_HEXAGON_OPSTAGE='1',
                         QWEN3_TTS_MAX_STRICT='0', QWEN3_TTS_MAX_VOCODER='0',
                         QWEN3_TTS_TRANSFORMER_BACKEND='cpu', QWEN3_TTS_ENCODER_BACKEND='cpu',
                         QWEN3_TTS_DECODER_BACKEND='cpu')
        capture, _ = self.invoke(args=('0.6b', '--text', 'Isolated settings'), env=inherited)
        env = capture['environment']
        assert env['QWEN3_TTS_TRANSFORMER_BACKEND'] == env['QWEN3_TTS_ENCODER_BACKEND'] == env['QWEN3_TTS_DECODER_BACKEND'] == 'hexagon'
        assert env['QWEN3_TTS_MAX_STRICT'] == env['QWEN3_TTS_MAX_VOCODER'] == '1'
        assert not set(env).intersection(('QWEN3_TTS_FORCE_CPU', 'GGML_PROFILE_OPS',
                                          'GGML_OPENCL_QCOM_RECORD', 'GGML_OPENCL_Q8_SMALL2', 'GGML_HEXAGON_OPSTAGE'))

    def run(self):
        for size in ('0.6b', '1.7b'):
            for entry in ('npu-max.sh', 'tools/npu-max/launch.py'):
                self.check(size + ' basic ' + entry, lambda s=size, e=entry: self.basic(s, e))
            for preset in (False, True):
                self.check(size + ' demo preset=' + str(preset), lambda s=size, p=preset: self.demo(s, p))
            for mode in ('reference', 'icl', 'embedding', 'extract'):
                self.check(size + ' ' + mode, lambda s=size, m=mode: self.mode(s, m))
            self.check(size + ' server stdin/stdout', lambda s=size: self.server(s))
            for entry in ('npu-max.sh', 'tools/npu-max/launch.py'):
                self.check(size + ' UTF-8 transcript file ' + entry,
                           lambda s=size, e=entry: self.transcript_file(s, e))
                self.check(size + ' help ignores missing transcript file ' + entry,
                           lambda s=size, e=entry: self.help_transcript(s, e))
            self.check(size + ' UTF-8 BOM transcript file',
                       lambda s=size: self.transcript_file(s, 'npu-max.sh', bom=True))
            self.check(size + ' empty transcript with saved embedding',
                       lambda s=size: self.empty_transcript_embedding(s))
            for explicit in (False, True):
                self.check(size + ' playback explicit_output=' + str(explicit),
                           lambda s=size, e=explicit: self.playback(s, e))
            self.check(size + ' synthesize and save embedding with cache output',
                       lambda s=size: self.generate_and_save_embedding(s))
        self.check('literal transcript-file flag text', self.literal_transcript_flag)
        self.check('flag-like text and non-streaming', self.passthrough)
        self.check('explicit empty text preserved', self.empty_text)
        rejected = {
            'extract without reference': ['--embedding-out', str(self.cwd / 'out.bin')],
            'voice conflict': ['-r', str(self.reference), '--embedding-in', str(self.embedding)],
            'transcript without reference': ['--ref-text', 'Transcript'],
            'missing reference file': ['-r', 'missing reference.wav'],
            'transcript text/file conflict': ['-r', str(self.reference), '--ref-text', 'Direct transcript',
                                              '--ref-text-file', str(self.transcript)],
            'missing transcript-file option value': ['-r', str(self.reference), '--ref-text-file'],
        }
        for kind, filename in self.bad_transcripts.items():
            rejected[kind + ' transcript file'] = ['-r', str(self.reference),
                                                  '--ref-text-file', str(filename)]
        for kind, filename in self.bad_embeddings.items():
            rejected[kind + ' saved embedding'] = ['--embedding-in', str(filename), '--text', 'Reject file']
        for name, arguments in rejected.items():
            self.check(name, lambda a=arguments: self.reject(a))
        for option, value in (('--reference', str(self.reference)), ('--embedding-in', str(self.embedding)),
                              ('--embedding-out', str(self.cwd / 'out.bin')), ('--ref-text', 'Transcript'),
                              ('--text', 'Ignored'), ('--output', 'ignored.wav'), ('--output-rate', '48000')):
            self.check('server rejects CLI ' + option,
                       lambda o=option, v=value: self.reject(['--server', o, v]))
        self.check('server rejects playback', lambda: self.reject(['--server', '--speak']))
        for entry in ('npu-max.sh', 'tools/npu-max/launch.py'):
            self.check('help without assets ' + entry, lambda e=entry: self.help(e))
            self.check('partial-build help ignores invalid embedding ' + entry,
                       lambda e=entry: self.partial_build_help(e))
        self.check('version ignores invalid saved embedding', self.version_ignores_voice)
        self.check('invalid demo embedding rejected', self.invalid_demo_embedding)
        self.check('custom model directory', self.custom_models)
        self.check('spaces and caller-relative paths', self.relative_paths)
        self.check('child exit 23 propagated', self.child_exit)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            self.check(signal.Signals(signum).name + ' returns ' + str(128 + int(signum))
                       + ' and reaps child', lambda sig=signum: self.signal_cleanup(sig))
        self.check('telemetry stays in isolated repository', self.log_root)
        for size in ('0.6b', '1.7b'):
            for profile in ('htp', 'mixed', 'cpu', 'opencl'):
                for mode in ('none', 'reference', 'icl', 'file_icl', 'embedding', 'extract', 'server',
                             'named_voice', 'voice_design', 'non_streaming_icl'):
                    self.check(size + ' profile=' + profile + ' mode=' + mode,
                               lambda s=size, p=profile, m=mode: self.profile(s, p, m))
        for size in ('0.6b', '1.7b'):
            self.check(size + ' explicit model/tokenizer override', lambda s=size: self.explicit_models(s))
        for value in ('invalid', '', 'HTP', '../htp'):
            self.check('reject profile ' + repr(value),
                       lambda v=value: self.invalid_environment('QWEN_NPU_MAX_PROFILE', v))
        for value in ('0', '7', '-1', 'auto', '1; echo invalid'):
            self.check('reject NHVX ' + repr(value),
                       lambda v=value: self.invalid_environment('GGML_HEXAGON_NHVX', v))
        for value in ('1', '2', '3', '4', '5', '6'):
            self.check('accept NHVX ' + value, lambda v=value: self.valid_threads(v))
        self.check('inherited diagnostic flags cleared and HTP selection enforced',
                   self.dangerous_inherited_environment)
        return self.results


def main():
    (ROOT / '.cache').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='launcher tests ', dir=ROOT / '.cache') as directory:
        results = Checks(Path(directory)).run()
    failed = sum(not result['passed'] for result in results)
    print(json.dumps({'suite': 'inference-free NPU-Max launchers', 'tests': len(results),
                      'passed': len(results) - failed, 'failed': failed,
                      'hardware_execution': False, 'results': results}))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
