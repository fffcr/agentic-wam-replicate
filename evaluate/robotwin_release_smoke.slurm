#!/bin/bash
#SBATCH --job-name=fcr_rw_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00

# Submit with absolute --output/--error paths after creating the log directory.
set -eo pipefail
repo=/public/home/terrypei/fcr_ws/AW-0/aw-0
runtime=/public/home/terrypei/fcr_ws/AW-0/envs-conda-test
source /public/home/terrypei/fcr_ws/env_manage/miniconda3/etc/profile.d/conda.sh
conda activate "$runtime"
set -u
cd "$repo"
export OMP_NUM_THREADS=4 AGENTICWAM_DEPLOY_DTYPE=bf16
export AGENTICWAM_FAST_JOINT=0 AGENTICWAM_TORCH_COMPILE=
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/public/home/terrypei/fcr_ws/AW-0/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TMPDIR="$repo/runs/robotwin_smoke/tmp-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

hostname
nvidia-smi
echo SITE_PREFLIGHT
"$runtime/bin/python" - <<'PY'
import os
from pathlib import Path
import sys
sys.path.insert(0, 'robotwin')
from hybrid_eval import config
s = config.site()
for key in ('python_bin', 'robotwin_root', 'vulkan_icd', 'hf_cache_policy'):
    print(f'{key}={s[key]}', flush=True)
for key in ('robotwin_root', 'vulkan_icd'):
    if not Path(s[key]).exists():
        raise SystemExit(f'Missing {key}: {s[key]}; fix site.yaml before rollout')
python = Path(s['python_bin']) / 'python'
if not python.is_file() or not os.access(python, os.X_OK):
    raise SystemExit(f'site.python_bin must be an environment bin directory: {python}')
PY

name="wam_smoke_${SLURM_JOB_ID}"
cmd=("$runtime/bin/python" scripts/eval_robotwin_release.py wam
     --tasks click_alarmclock --plan smoke --gpus 0 --name "$name")
echo AUTHOR_SMOKE_DRY_RUN
"${cmd[@]}" --dry-run --launch-args --extra paths=cluster
echo AUTHOR_SMOKE_START
srun "${cmd[@]}" --launch-args --extra paths=cluster
echo AUTHOR_SMOKE_PROCESS_OK
echo "Results: $repo/evaluate_results/robotwin_benchmark/$name"
# PROCESS_OK means the author's runner returned successfully, not task success.
# Inspect per-episode counts/scores and errors in the benchmark results/logs.
