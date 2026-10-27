#!/usr/bin/env python3
"""Launch native HTP generation/OpenCL decoding with measurement-only logging."""
import os
from pathlib import Path
import sys
import struct

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_LIBRARIES = ('libggml.so', 'libggml-base.so', 'libggml-cpu.so',
                     'ggml-hexagon/libggml-hexagon.so', 'ggml-hexagon/libggml-htp-v79.so',
                     'ggml-opencl/libggml-opencl.so')
VALUE_FLAGS = {
    '-m', '--model', '--tokenizer', '-t', '--text', '-o', '--output',
    '-r', '--reference', '--ref-text', '--ref-text-file', '--embedding-in', '--embedding-out',
    '--speaker', '--instruct', '-l', '--language', '--temperature', '--top-k',
    '--top-p', '--min-p', '--repetition-penalty', '--frequency-penalty',
    '--presence-penalty', '--dry-multiplier', '--dyntemp-range',
    '--sub-temperature', '--sub-top-k', '--sub-top-p', '--max-tokens',
    '--output-rate', '-j', '--threads', '--gen-config',
}
ALIASES = {'-r': '--reference', '-t': '--text', '-o': '--output', '-m': '--model'}


def fail(message):
    print('NPU launcher: ' + message, file=sys.stderr)
    raise SystemExit(2)


def usage():
    print('Usage: sh npu.sh [0.6b|1.7b] [TTS options]\n'
          '       sh tools/npu/run-tts.sh 0.6b|1.7b [TTS options]\n\n'
          'Voice modes:\n'
          '  --text TEXT                              no-reference speech\n'
          '  -r REF.wav --text TEXT                   speaker-only clone\n'
          '  -r REF.wav --ref-text TRANSCRIPT --text TEXT  ICL clone\n'
          '  -r REF.wav --ref-text-file REF.txt --text TEXT  ICL from transcript file\n'
          '  --embedding-in VOICE.bin --text TEXT     saved speaker embedding\n'
          '  -r REF.wav --embedding-out VOICE.bin     extract without synthesis\n'
          '  --server                                JSON lines on stdin/stdout\n\n'
          'Server voice, text and output paths belong in each JSON request.\n'
          'Root npu.sh supplies a short demo only when text was not provided.\n'
          'Use QWEN_MODELS_DIR for an external model directory.\n'
          'Temperatures/clocks are logged without added thermal limits.')


def scan(arguments):
    values = {}
    flags = set()
    i = 0
    while i < len(arguments):
        arg = arguments[i]
        if arg in VALUE_FLAGS:
            if i + 1 == len(arguments):
                fail('missing value for ' + arg)
            values[ALIASES.get(arg, arg)] = arguments[i + 1]
            i += 2
        else:
            flags.add(arg)
            i += 1
    return values, flags


def readable_file(value, label):
    if not value or not Path(value).is_file() or not os.access(value, os.R_OK):
        fail(label + ' is not a readable file: ' + repr(value))


def speaker_file(value):
    readable_file(value, 'Speaker embedding')
    try:
        size = Path(value).stat().st_size
        with open(value, 'rb') as stream:
            header = stream.read(4)
    except OSError as error:
        fail('cannot read speaker embedding: ' + str(error))
    count = struct.unpack('<I', header)[0] if len(header) == 4 else 0
    if not count or size != 4 + 4 * count or size > 16 * 1024 * 1024:
        fail('invalid saved speaker embedding; use a .bin created by --embedding-out')


def writable_parent(value, label):
    parent = Path(value).parent
    if not value or not parent.is_dir() or not os.access(parent, os.W_OK):
        fail(label + ' needs an existing writable parent directory: ' + repr(value))


def main(argv):
    demo = bool(argv and argv[0] == '--demo')
    if demo:
        argv = argv[1:]
    if argv and argv[0] in ('0.6b', '1.7b'):
        size, args = argv[0], argv[1:]
    elif demo and (not argv or argv[0].startswith('-')):
        size, args = '0.6b', argv
    elif argv and argv[0] in ('-h', '--help'):
        usage()
        return 0
    else:
        fail('select model size 0.6b or 1.7b; use --help for modes')
    values, flags = scan(args)
    if flags.intersection(('-h', '--help')):
        usage()
        cli = ROOT / '.cache/npu/app-build/qwen3-tts-cli'
        base = ROOT / '.cache/npu/build/src'
        runtime_ready = all((base / name).is_file() for name in RUNTIME_LIBRARIES)
        if cli.is_file() and os.access(cli, os.X_OK) and runtime_ready:
            sys.stdout.flush()
            os.execvp('sh',
                     ['sh', str(ROOT / 'tools/npu/native-env.sh'), str(cli), '--help'])
        return 0
    if '--ref-text-file' in values:
        if '--ref-text' in values:
            fail('choose --ref-text or --ref-text-file')
        filename = values['--ref-text-file']
        readable_file(filename, 'Reference transcript')
        try:
            with open(filename, 'rb') as stream:
                raw_transcript = stream.read(1024 * 1024 + 1)
            if len(raw_transcript) > 1024 * 1024:
                fail('reference transcript exceeds 1 MiB')
            transcript = raw_transcript.decode('utf-8-sig').strip()
        except (OSError, UnicodeError) as error:
            fail('cannot read UTF-8 reference transcript: ' + str(error))
        if not transcript or len(transcript) > 1024 * 1024 or '\0' in transcript:
            fail('reference transcript must be nonempty UTF-8 text, at most 1 MiB')
        expanded = []
        i = 0
        while i < len(args):
            option = args[i]
            if option in VALUE_FLAGS:
                value = args[i + 1]
                expanded += ['--ref-text', transcript] if option == '--ref-text-file' else [option, value]
                i += 2
            else:
                expanded.append(option)
                i += 1
        args = expanded
        values, flags = scan(args)
    server = '--server' in flags
    info = bool(flags.intersection(('--version', '--list-speakers', '--list-languages')))
    reference = values.get('--reference')
    embedding = values.get('--embedding-in')
    embedding_out = values.get('--embedding-out')
    if not info:
        if server and any(k in values for k in (
                '--reference', '--ref-text', '--embedding-in', '--embedding-out', '--output', '--text', '--output-rate')):
            fail('in server mode, put voice, text and output fields in each JSON request')
        if server and flags.intersection(('-s', '--speak')):
            fail('server mode writes JSON responses; playback is a one-shot CLI option')
        if reference is not None and embedding is not None:
            fail('choose --reference or --embedding-in; supplying both would ignore the reference')
        if values.get('--ref-text') and not reference:
            fail('--ref-text requires --reference; a saved embedding alone does not enable ICL')
        if embedding_out is not None and not reference:
            fail('--embedding-out requires --reference')
        for key, label in [('--reference', 'Reference audio'), ('--embedding-in', 'Speaker embedding')]:
            if key in values:
                readable_file(values[key], label)
        if embedding is not None:
            speaker_file(embedding)
        for key in ('--output', '--embedding-out'):
            if key in values:
                writable_parent(values[key], key)
    extraction_only = bool(embedding_out) and not values.get('--text')
    defaults = []
    if demo:
        defaults += ['--language', 'english', '--temperature', '.9', '--sub-temperature', '.9', '--max-tokens', '128']
        if not server and not info:
            if '--text' not in values and not embedding_out:
                defaults += ['--text', 'Hello from the Hexagon NPU.']
            voice_selected = any(k in values for k in (
                '--reference', '--embedding-in', '--speaker', '--instruct', '--model'))
            if not voice_selected:
                preset = os.environ.get('QWEN_NPU_DEMO_EMBEDDING_' + ('06B' if size == '0.6b' else '17B'))
                if preset:
                    speaker_file(preset)
                    defaults += ['--embedding-in', preset]
            if '--output' not in values and not extraction_only and not flags.intersection(('-s', '--speak')):
                output = (os.environ.get('QWEN_NPU_OUTPUT') or str(Path.cwd() / 'npu-test.wav'))
                writable_parent(output, 'Demo output')
                defaults += ['--output', output]
    cli = ROOT / '.cache/npu/app-build/qwen3-tts-cli'
    if not cli.is_file() or not os.access(cli, os.X_OK):
        fail('build first: sh tools/npu/build-host-dsp.sh')
    if '--version' not in flags:
        model_dir = Path((os.environ.get('QWEN_MODELS_DIR') or str(ROOT / 'models')))
        model = values.get('--model', str(model_dir / ('qwen3-tts-' + size + '-q8_0.gguf')))
        tokenizer = values.get('--tokenizer', str(model_dir / 'qwen3-tts-tokenizer-f16-f32.gguf'))
        if not Path(model).exists() or not os.access(model, os.R_OK):
            fail('model is not readable: ' + repr(model))
        readable_file(tokenizer, 'Tokenizer model')
    base = ROOT / '.cache/npu/build/src'
    for name in RUNTIME_LIBRARIES:
        readable_file(str(base / name), 'Runtime library')
    model_dir = Path((os.environ.get('QWEN_MODELS_DIR') or str(ROOT / 'models')))
    command = ['sh', str(ROOT / 'tools/npu/native-env.sh'), str(cli),
               '-m', str(model_dir / ('qwen3-tts-' + size + '-q8_0.gguf')),
               '--tokenizer', str(model_dir / 'qwen3-tts-tokenizer-f16-f32.gguf'),
               '--threads', '2', *defaults, *args]
    env = os.environ.copy()
    for key in ['QWEN3_TTS_FORCE_CPU', 'GGML_PROFILE_OPS', 'GGML_OPENCL_QCOM_RECORD',
                'GGML_OPENCL_Q8_SMALL2', 'GGML_HEXAGON_OPSTAGE']:
        env.pop(key, None)
    env.update(QWEN3_TTS_TRANSFORMER_BACKEND='hexagon', QWEN3_TTS_DECODER_BACKEND='opencl',
               QWEN3_TTS_VALIDATE_AUDIO='1', QWEN3_TTS_VALIDATE_LOGITS='1')
    # Reuse the measurement runner in this interpreter; no extra Python
    # process or model import is added to the synthesis startup path.
    sys.path.insert(0, str(ROOT / 'scripts/android'))
    from profile_runner import main as run_profile
    return run_profile(command, env=env)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
