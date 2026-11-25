#!/bin/bash
#SBATCH --job-name=fcr_expert_bench
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

tag=${BENCH_TAG:-baseline}
args=()
# Set BENCH_DETAIL=0 for clean-only tests of compiled implementations.
if [[ ${BENCH_DETAIL:-1} == 1 ]]; then args+=(--detail); fi
if [[ -n ${BENCH_COMPARE:-} ]]; then args+=(--compare "$BENCH_COMPARE"); fi
if [[ -n ${BENCH_INPUT_NPZ:-} ]]; then args+=(--input-npz "$BENCH_INPUT_NPZ"); fi
mkdir -p "$repo/runs/bench_expert_latency/tmp"

hostname
nvidia-smi
srun env \
  OMP_NUM_THREADS=4 \
  HF_HOME=/public/home/terrypei/fcr_ws/AW-0/hf-cache \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  DIFFSYNTH_DOWNLOAD_SOURCE=huggingface \
  DIFFSYNTH_MODEL_BASE_PATH="$repo/checkpoints" \
  LD_LIBRARY_PATH="$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  TMPDIR="$repo/runs/bench_expert_latency/tmp" \
  AGENTICWAM_DEPLOY_DTYPE=bf16 \
  AGENTICWAM_FAST_JOINT="${BENCH_FAST_JOINT:-0}" \
  AGENTICWAM_TORCH_COMPILE="${BENCH_COMPILE_MODE:-}" \
  TORCH_COMPILE_DISABLE="${BENCH_DISABLE_COMPILE:-1}" \
  TORCHDYNAMO_DISABLE="${BENCH_DISABLE_COMPILE:-1}" \
  "$runtime/bin/python" -u evaluate/bench_expert_latency.py \
    --ckpt "$repo/checkpoints/robotwin_release/wam.pt" \
    --task robotwin_wam --config robotwin/sim_robotwin \
    --nis "${BENCH_NIS:-20}" --warmup "${BENCH_WARMUP:-2}" --reps "${BENCH_REPS:-10}" \
    --tag "$tag" --output "$repo/runs/bench_expert_latency/${tag}-${SLURM_JOB_ID}" \
    "${args[@]}" \
    --extra paths=cluster \
    data.train.pretrained_norm_stats="$repo/checkpoints/robotwin_release/dataset_stats.json"
