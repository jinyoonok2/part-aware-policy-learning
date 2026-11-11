#!/usr/bin/env bash
#SBATCH --job-name=part-val-expert-planner-build
#SBATCH --account=xna8aw_base
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${PART_VAL_EXPERT_RUN:?Set the prepared native expert pilot directory}"
: "${PART_VAL_CUROBO_SOURCE:?Set the pinned local CuRobo checkout}"
module load cuda/12.8.1 gcc/12.4.0
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export CC="$(command -v gcc)" CXX="$(command -v g++)"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export MAX_JOBS=1 TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0+PTX'
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR=/bigtemp/xna8aw/part-aware-policy-learning/cache/torch_extensions/curobo_417
/bigtemp/xna8aw/part-aware-policy-learning/envs/mlspaces/bin/python -u - <<'PY'
import hashlib,importlib.metadata as m,json,os,subprocess,sys
from pathlib import Path

run=Path(os.environ['PART_VAL_EXPERT_RUN']);source=Path(os.environ['PART_VAL_CUROBO_SOURCE'])
study=json.loads((run/'study.json').read_text())
def normal(n):return n.lower().replace('_','-').replace('.','-')
def versions():return {normal(d.metadata['Name']):d.version for d in m.distributions()}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
assert actual==study['curobo_revision'],'Incorrect native planner checkout'
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip(),'CuRobo source has tracked edits'
assert json.loads((run/'audit/prerequisites_receipt.json').read_text())['existing_versions_unchanged']
before=versions();assert 'nvidia-curobo' not in before,'Do not replace an existing native planner distribution'
assert before['torch']==study['torch_version'] and before['mujoco']==study['mujoco_version']
save(run/'audit/packages_before_curobo_build.json',before)
wheels=run/'audit/curobo_wheels';wheels.mkdir(exist_ok=True)
subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(wheels),str(source)],check=True)
found=list(wheels.glob('nvidia_curobo-*.whl'));assert len(found)==1,'Expected exactly one newly built planner wheel'
subprocess.run([sys.executable,'-m','pip','install','--no-deps',str(found[0])],check=True)
after=versions();assert all(after.get(n)==v for n,v in before.items()),'An existing package version changed'
assert set(after)-set(before)=={'nvidia-curobo'},'Unexpected additional package installation'
dist=m.distribution('nvidia-curobo')
receipt=dict(state='built_and_installed',curobo_revision=actual,
    source_directory=str(source),source_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=source,text=True).strip(),
    installed_record_sha256=hashlib.sha256((dist.read_text('RECORD') or '').encode()).hexdigest(),
    wheel=str(found[0]),wheel_sha256=digest(found[0]),existing_versions_unchanged=True,
    added_distribution='nvidia-curobo',installed_version=dist.version,
    cuda_arch_list=os.environ['TORCH_CUDA_ARCH_LIST'],
    torch_version=after['torch'],mujoco_version=after['mujoco'],numpy_version=after['numpy'],
    gpu_runtime_verified=False,training_started=False,model_inference_used=False)
save(run/'audit/curobo_build_receipt.json',receipt)
save(run/'audit/packages_after_curobo_build.json',after)
print(json.dumps(receipt,indent=2),flush=True)
PY
