#!/bin/bash
#SBATCH --job-name=fcr_rw_probe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00

set -eo pipefail
repo=/public/home/terrypei/fcr_ws/AW-0/aw-0
runtime=/public/home/terrypei/fcr_ws/AW-0/envs-conda-test
source /public/home/terrypei/fcr_ws/env_manage/miniconda3/etc/profile.d/conda.sh
conda activate "$runtime"
set -u
cd "$repo"
out="$repo/runs/robotwin_probe/${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4 AGENTICWAM_DEPLOY_DTYPE=bf16
export AGENTICWAM_FAST_JOINT=0 AGENTICWAM_TORCH_COMPILE=
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export HF_HOME=/public/home/terrypei/fcr_ws/AW-0/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$repo/runs/robotwin_probe/tmp-${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
mkdir -p "$TMPDIR"
hostname
nvidia-smi
srun "$runtime/bin/python" -u evaluate/robotwin_probe/runner.py \
  --output "$out" --task "${PROBE_TASK:-click_alarmclock}" \
  --protocols "${PROBE_PROTOCOLS:-clean}" \
  --modes "${PROBE_MODES:-eager,compile}" \
  --warmup "${PROBE_WARMUP:-2}" --reps "${PROBE_REPS:-5}" --trace
