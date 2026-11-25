# Action latency investigation and consecutive replan overlap

## What can be tested now

Run on the GPU cluster, not the login node. Copy the updated files in `evaluate/`
(`bench_expert_latency.py`, both Slurm templates, `expert_streams.py`,
`expert_trace.py`, `analyze_expert_bench.py`) to the same directory on the server.
The matrix does not require real observations; defaults remain synthetic inputs.
It does not change source model math, precision, weights or the number of denoising steps.

```bash
cd ~/fcr_ws/AW-0/aw-0
mkdir -p runs/action_matrix
bash -n evaluate/bench_action_matrix.slurm
sbatch --output="$PWD/runs/action_matrix/%j.out" \
       --error="$PWD/runs/action_matrix/%j.err" \
       evaluate/bench_action_matrix.slurm
```

One H200 allocation runs separate processes for:

1. Eager serial: clean 20 rounds, then independent detailed/trace diagnostics.
2. MoT compilation (`reduce-overhead`): clean timing and action comparison.
3. Eager multi-stream: one stream/expert for per-layer pre/post work. Joint attention
   stays on the caller stream, with explicit stream dependencies and allocator
   ownership. An eager-vs-stream action/code equality check runs before timing and
   aborts the experiment on a mismatch. It is an experimental inference-only patch,
   installed only on this process's MoT instance. No training/default serving behavior changes.

`BENCH_RUN_FAST_JOINT=1` optionally adds the broader compiled-step implementation.
Compilation can exhaust the 20-minute default limit. Request more time via `sbatch
--time=...` if needed. Compiler errors remain failures; the matrix continues to the
stream test after a MoT compiler error. fast_joint itself can fall back to eager:
inspect its stderr and the reported Dynamo counters before claiming compiled speed.
Do not combine multi-stream and torch.compile in this initial experiment.

Each mode reports clean wall mean/p50/p95, action differences, parameter counts and
dtype histograms. All rounds reset the episode/RNG; compilation and multi-stream do
not use different input data. Modes run on the same allocated GPU, but thermal/load
drift is still possible. Repeat the matrix and reverse order when investigating small
differences. A speedup without output agreement needs further correctness work.

## Read the action breakdown

For the existing baseline, no GPU is needed to analyze its CSV:

```bash
python3 evaluate/analyze_expert_bench.py \
  runs/bench_expert_latency/baseline-579248
```

`analysis.json` gives action `pre_attn`, `post_attn`, `pre_dit`, `post_dit` totals,
call counts, mean/call and shares within action. It does not invent a breakdown from
the earlier aggregate 605 ms. Action post-attention includes output projection,
text/state cross-attention, normalization, gating and FFN.

New eager traces include `expert.action.pre_attn/post_attn` and nested ranges for
each block's `cross_attn`, `ffn`, and `out_projection`. Kernel attribution uses
PyTorch external/correlation IDs, and reports unmatched kernels explicitly. It
also reports the union of GPU kernel intervals (avoiding double-counting concurrent
kernels), and time uncovered by kernels between first/last kernel. Uncovered time
can include copies, synchronization and host launch gaps; it is not automatically
CPU overhead. Kernel duration/occupancy evidence is needed to distinguish memory
bandwidth saturation from arithmetic bottlenecks. This parser does not claim to
measure SM utilization or memory bandwidth.

The key questions are whether action time is in repeated text K/V projections,
FFN, QKV/RoPE, or many small launches. Attention K/V caching would require checking
which context inputs are invariant, and is not implemented or assumed safe here.
Trace intervals are separate from clean speed measurements; multi-stream group
durations overlap and must not be added as serial costs.

Outputs: `runs/action_matrix/JOB/{eager,compile_mot,multistream}/report.json`,
`eager/analysis.json`, `eager/detail_calls.csv`, `eager/trace.json`,
`multistream/analysis.json`, `multistream/trace.json`; mode stderr/stdout live
beside those directories. Trace files can be large. Compiled clean timings do
not include detail hooks; the matrix does not insert annotation hooks into compiled graphs.

## Record actual consecutive replans (requires a rollout)

Existing benchmark repeats cannot answer overlap across real replans. Do not align
them as consecutive observations. Copy `deploy/common/replan_record.py` to the
server as well. Integrate this opt-in recorder at the real controller's runtime
call site (adapt camera acquisition and control to the existing application):

```python
from common.replan_record import ReplanRecorder

recorder = ReplanRecorder("runs/replan_capture/run_001")  # new directory
actual_tick = 0
rt.reset()  # once per episode, NOT once per replan
while episode_is_running:
    images, state, observation_stamp_ns = get_current_observation()
    chunk, code, infer_ms = rt.infer(images, state, instruction)
    recorder.record(images=images, state=state, chunk=chunk,
                    instruction=instruction, episode_id=episode_id,
                    start_step=actual_tick, code=code,
                    observation_stamp_ns=observation_stamp_ns)
    for target in chunk[:rt.replan_steps]:
        execute_and_confirm_one_action_tick(target)
        actual_tick += 1  # increment after each actual action tick
        if episode_finished():
            break
```

This is an integration example, not a runnable robot controller. The recorder saves
images, raw state, absolute joint targets, instruction/code, episode and actual
action-tick index. Do not increment by a planned interval when execution was
interrupted. It writes synchronously and changes rollout timing; use for diagnostics,
not clean latency. If control continues during inference, start_step must refer to
the actual first tick of the returned chunk, not observation capture time. If the
controller interpolates/resamples/discards targets, log the corresponding time grid
and resample predictions before comparing; this simple tick alignment then does
not apply automatically. A RoboTwin policy-only path without PolicyRuntime must
first convert its action representation to the same absolute qpos contract.

After collecting:

```bash
python3 evaluate/analyze_replan_overlap.py runs/replan_capture/run_001 --atol 0.001
```

With old horizon H and actual advance R, compare old[R:R+N] to new[:N],
N=max(0,min(H-R,new_horizon)). Episode boundaries are never compared. Outputs
include per-dimension MAE/max difference, tolerance fractions, exact equality,
and `overlap_rows.csv` with aligned action tick and old/new values. The scalar
tolerance is for numerical inspection; joints and grippers have different units.
Overlapping predictions can still differ because of new observations, history
and stochastic sampling. Similar outputs do not establish reusable hidden states
or permit blindly keeping old commands.
