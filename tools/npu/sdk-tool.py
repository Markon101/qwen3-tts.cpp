#!/data/data/com.termux/files/usr/bin/python
# Adapted from OpenCL-S24-Ultra (MIT); see LICENSE-OpenCL-S24-Ultra.txt.
import os,pathlib,sys,subprocess,shlex
R=pathlib.Path(__file__).resolve().parents[2]
name=pathlib.Path(sys.argv[0]).name
sdk=R/'sdk/6.6.0.0'
tools=sdk/'tools/HEXAGON_Tools/19.0.07/Tools/bin'
program=sdk/'ipc/fastrpc/qaic/bin/qaic' if name=='qaic' else tools/name
args=sys.argv[1:]
# Use absolute positional IDL paths consistently in the emulated QAIC job.
# The observed concurrent QAIC stall has no established root cause.
if name == 'qaic':
 args=[str(pathlib.Path(a).resolve()) if not a.startswith('-') and a.endswith('.idl') and pathlib.Path(a).is_file() else a for a in args]
env=os.environ.copy()
for k in ['LD_LIBRARY_PATH','LD_PRELOAD']:env.pop(k,None)
android=R.parent/'android-runtime/data/data/com.termux/files/usr'
qemu=os.environ.get('QWEN_NPU_QEMU',str(android/'bin/qemu-x86_64'))
env['LD_LIBRARY_PATH']=str(android/'lib')
prefix=[qemu,'-L',str(R/'x86-root'),'-U','LD_LIBRARY_PATH','-U','LD_PRELOAD']
if name in ['hexagon-clang','hexagon-clang++']:
 args=['-B'+str(R/'emulation/bin'),*args]
 if any(a in args for a in ['--version','-###','-print-resource-dir']) or not args:
  os.execve(qemu,[*prefix,str(program),*args],env)
 plan=subprocess.run([*prefix,str(program),'-###',*args],env=env,capture_output=True,text=True)
 if plan.returncode:
  sys.stderr.write(plan.stderr);sys.exit(plan.returncode)
 commands=[]
 for line in plan.stderr.splitlines():
  if line.lstrip().startswith('"'):
   commands.append(shlex.split(line))
 if not commands:
  sys.stdout.write(plan.stdout);sys.stderr.write(plan.stderr);sys.exit(0)
 for cmd in commands:
  executable=pathlib.Path(cmd[0])
  if str(executable).startswith(str(tools)):
   cmd=[*prefix,*cmd]
  result=subprocess.run(cmd,env=env)
  if result.returncode:sys.exit(result.returncode)
 sys.exit(0)
os.execve(qemu,[*prefix,str(program),*args],env)
