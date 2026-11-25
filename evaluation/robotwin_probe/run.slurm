#!/bin/bash
#SBATCH --job-name=fcr_rw_compile
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00

set -eo pipefail
repo=/public/home/terrypei/fcr_ws/AW-0/aw-0
runtime=/public/home/terrypei/fcr_ws/AW-0/envs-conda-test
source /public/home/terrypei/fcr_ws/env_manage/miniconda3/etc/profile.d/conda.sh
conda activate "$runtime"
set -u
cd "$repo"
out="$repo/runs/robotwin_compile/${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4 AGENTICWAM_DEPLOY_DTYPE=bf16
export AGENTICWAM_FAST_JOINT=0 AGENTICWAM_TORCH_COMPILE=
# Capture uses the original eager policy; runner removes these for the compile worker.
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export HF_HOME=/public/home/terrypei/fcr_ws/AW-0/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$repo/runs/robotwin_compile/tmp-${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
export TORCHINDUCTOR_COMPILE_THREADS=2
export RW_PROBE_CAPTURE_DIFFUSION=0
mkdir -p "$TMPDIR"
hostname
nvidia-smi
srun "$runtime/bin/python" -u evaluation/robotwin_probe/runner.py \
  --output "$out" --task "${PROBE_TASK:-click_alarmclock}" \
  --protocols "${PROBE_PROTOCOLS:-clean}" \
  --modes compile --compile-mode "${PROBE_COMPILE_MODE:-reduce-overhead}" \
  --max-replans "${PROBE_MAX_REPLANS:-2}" \
  --warmup "${PROBE_WARMUP:-5}" --reps "${PROBE_REPS:-10}" \
  --stage-reps "${PROBE_STAGE_REPS:-10}"
