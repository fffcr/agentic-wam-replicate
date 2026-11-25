#!/bin/bash
#SBATCH --job-name=fcr_action_matrix
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
run_dir="$repo/runs/action_matrix/${SLURM_JOB_ID}"
mkdir -p "$run_dir/tmp"
export OMP_NUM_THREADS=4
export HF_HOME=/public/home/terrypei/fcr_ws/AW-0/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface DIFFSYNTH_MODEL_BASE_PATH="$repo/checkpoints"
export LD_LIBRARY_PATH="$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TMPDIR="$run_dir/tmp" AGENTICWAM_DEPLOY_DTYPE=bf16
export TORCHINDUCTOR_CACHE_DIR="$run_dir/inductor_cache"
# Keep optional experiment gates out of the controlled matrix.
export AGENTICWAM_CODE_PIN=0 AGENTICWAM_RTC=0 AGENTICWAM_FAST_JOINT_NO_COMPILE=0
export AGENTICWAM_FAST_JOINT=0 AGENTICWAM_TORCH_COMPILE=
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
common=(--ckpt "$repo/checkpoints/robotwin_release/wam.pt"
        --task robotwin_wam --config robotwin/sim_robotwin
        --nis "${BENCH_NIS:-20}" --warmup "${BENCH_WARMUP:-5}" --reps "${BENCH_REPS:-20}")
overrides=(--extra paths=cluster
           "data.train.pretrained_norm_stats=$repo/checkpoints/robotwin_release/dataset_stats.json")
if [[ -n ${BENCH_INPUT_NPZ:-} ]]; then common+=(--input-npz "$BENCH_INPUT_NPZ"); fi
hostname
nvidia-smi

# One Slurm allocation, one GPU, separate processes; clean timing precedes diagnostics.
srun "$runtime/bin/python" -u evaluate/bench_expert_latency.py \
  "${common[@]}" --output "$run_dir/eager" --tag eager --detail --trace \
  "${overrides[@]}" > "$run_dir/eager.out" 2> "$run_dir/eager.err"
"$runtime/bin/python" evaluate/analyze_expert_bench.py "$run_dir/eager" > "$run_dir/eager_analysis.out"

failed=0
if ! srun env AGENTICWAM_TORCH_COMPILE=reduce-overhead \
  TORCH_COMPILE_DISABLE=0 TORCHDYNAMO_DISABLE=0 \
  "$runtime/bin/python" -u evaluate/bench_expert_latency.py \
  "${common[@]}" --output "$run_dir/compile_mot" --tag compile_mot \
  --compare "$run_dir/eager" "${overrides[@]}" \
  > "$run_dir/compile_mot.out" 2> "$run_dir/compile_mot.err"; then
  echo 'COMPILE_MOT_FAILED: inspect compile_mot.err'
  failed=1
fi

srun "$runtime/bin/python" -m unittest discover -s evaluate -p test_expert_streams_cuda.py -v \
  > "$run_dir/streams_unit.out" 2> "$run_dir/streams_unit.err"
if ! srun "$runtime/bin/python" -u evaluate/bench_expert_latency.py \
  "${common[@]}" --output "$run_dir/multistream" --tag multistream \
  --multistream --trace --compare "$run_dir/eager" "${overrides[@]}" \
  > "$run_dir/multistream.out" 2> "$run_dir/multistream.err"; then
  echo 'MULTISTREAM_FAILED: inspect multistream.err/equivalence report'
  failed=1
fi
if [[ -f "$run_dir/multistream/trace.json" ]]; then
  "$runtime/bin/python" evaluate/analyze_expert_bench.py "$run_dir/multistream" \
    > "$run_dir/multistream_analysis.out"
fi

if [[ ${BENCH_RUN_FAST_JOINT:-0} == 1 ]]; then
  if ! srun env AGENTICWAM_FAST_JOINT=1 TORCH_COMPILE_DISABLE=0 TORCHDYNAMO_DISABLE=0 \
    "$runtime/bin/python" -u evaluate/bench_expert_latency.py \
    "${common[@]}" --output "$run_dir/fast_joint" --tag fast_joint \
    --compare "$run_dir/eager" "${overrides[@]}" \
    > "$run_dir/fast_joint.out" 2> "$run_dir/fast_joint.err"; then
    echo 'FAST_JOINT_FAILED: inspect fast_joint.err'
    failed=1
  fi
fi
echo "MATRIX_RESULTS=$run_dir"
if [[ $failed == 0 ]]; then echo ACTION_MATRIX_OK; fi
exit "$failed"
