# Expert latency and action inspection

This benchmark uses real model weights, without RoboTwin, Vulkan, skill execution,
or physical motion. Synthetic inputs test computation, not task success or motion safety.

## Server installation and first run

Copy `evaluate/bench_expert_latency.py` and `evaluate/bench_expert_latency.slurm`
to the server repository's `evaluate/` directory. Keep the previously applied local
Cosmos processor loading fix. No core model files need changing for this benchmark.

```bash
cd ~/fcr_ws/AW-0/aw-0
mkdir -p runs/bench_expert_latency
bash -n evaluate/bench_expert_latency.slurm
sbatch --output="$PWD/runs/bench_expert_latency/%j.out" \
       --error="$PWD/runs/bench_expert_latency/%j.err" \
       evaluate/bench_expert_latency.slurm
```

Defaults: BF16, 20 denoising steps, 2 warmup rounds, 10 measured rounds,
one additional action diagnostic round and one additional detailed round.
Every call resets the episode and RNG outside timing. Text cache remains warm.
The same three random images, zero state and instruction are reused. This differs
from `bench_aw0_stages.py`, which measures evolving history: do not treat their
absolute timings as identical protocols. Timing includes returning the action to CPU,
but excludes reset, saving reports, and later robot execution.

## Does accelerating video improve total latency?

1. Run the baseline above. Read `report.json` and `detail_calls.csv`.
2. Apply your actual video implementation optimization, retaining checkpoint,
   inputs, precision, denoising steps and all other settings.
3. Submit a candidate run referencing the baseline output directory:

```bash
BENCH_TAG=video_candidate \
BENCH_COMPARE="$PWD/runs/bench_expert_latency/baseline-BASE_JOB_ID" \
sbatch --export=ALL \
       --output="$PWD/runs/bench_expert_latency/%j.out" \
       --error="$PWD/runs/bench_expert_latency/%j.err" \
       evaluate/bench_expert_latency.slurm
```

`comparison.mean_speedup = baseline mean / candidate mean`. Above 1 means lower
observed total latency. Both values come from **uninstrumented** rounds. This
script does not implement a video optimization and does not remove or mock video
computation to manufacture a speedup. Alternate baseline/candidate runs on an idle
GPU for stronger evidence; 10 samples alone do not establish statistical significance.

Detailed timing covers every layer's expert `pre_attn` / `post_attn`, each
expert's available `pre_dit` / `post_dit`, video patchification/output, and shared
attention. `detail_calls.csv` includes zero-based step/layer indices. CFG paths
with extra MoT calls are rejected because their call-to-step mapping differs.
Expert token lengths are recorded. Shared attention cannot be attributed exclusively
to video: reducing video token count may accelerate shared attention too.

All detail intervals are CUDA current-stream timeline intervals, including CPU launch
gaps, not summed GPU kernel execution time. The many events add overhead, so detail
totals are explanatory only. Amdahl estimates hold other work fixed and are projections,
not measured speedups. The video group excludes shared attention and the VAE.
Schedulers, code selection, concatenation, and remaining orchestration are not fully
attributed. `--trace` optionally saves a separate CPU/CUDA trace and operator table.

`--detail` deliberately rejects fast_joint and ordinary MoT compilation. Compiled
optimization candidates should use clean-only mode (`BENCH_DETAIL=0`); compare their
uninstrumented latency, never detailed eager time against compiled clean time.
For example, fast_joint uses `BENCH_DETAIL=0 BENCH_FAST_JOINT=1
BENCH_DISABLE_COMPILE=0`; it optimizes multiple experts, so it is not evidence of
a video-only speedup. Check logs for compiler fallback. Raise `BENCH_WARMUP` and
the Slurm time limit if compilation needs longer.

## Action output and adjacent time steps

`PolicyRuntime.infer` returns absolute joint targets `[T,D]`. The saved batch
has shape `[round,T,D]`. For the current 14-D contract the column order is:
left joints 0..5, left gripper, right joints 0..5, right gripper. Do not interpret
all 14 values as Cartesian coordinates or use a shared unit for joints and grippers.
Names come from the runtime contract, not independent physical calibration.

- `action_rows.csv`: each action time step's absolute target.
- `action_adjacent_delta.csv`: row t is `action[t+1] - action[t]`, within one chunk.
- `actions.npz`: absolute targets, offsets from input state, adjacent-step deltas,
  and differences between repeated calls at aligned horizon indices.
- `action_intermediates.npz`: normalized and denormalized model outputs captured
  in a separate untimed call, plus runtime absolute targets and input state.
- `report.json`: shapes, finiteness, per-dimension ranges and adjacent differences;
  same-input repeat consistency and A/B action allclose checks (with explicit tolerances).
- `codes.json`: generated code for each clean round.

Model outputs before runtime conversion may describe offsets relative to the
observation state; these are **not** necessarily differences between adjacent action
rows. EEF checkpoints also undergo FK/IK conversion. Zero input state makes some
absolute/relative values numerically indistinguishable. Use a real nonzero state
and verify the processor transform to establish representation semantics.

`repeat_delta` compares repeated predictions from the **same** observation with
reset history and RNG. It is not a comparison of consecutive camera observations,
not a closed-loop rollout, and not a cross-chunk executed-trajectory discontinuity.
No task-quality or physical-safety conclusion follows from smooth numerical output.

For recorded input, create an NPZ with `head_camera`, `left_camera`, `right_camera`
(HxWx3 RGB uint8) and `state` (runtime raw joint vector). Set `BENCH_INPUT_NPZ`
to that file. The benchmark hashes actual input values for matching A/B runs.
It checks checkpoint path/size/mtime, workload and GPU model; it does not hash
multi-GB weights or guarantee the same GPU load/software. Reports include git HEAD,
selected environment flags, host and torch/CUDA versions. Keep a diff of source
changes alongside each report, including changes in a dirty checkout.
