# Compiled RoboTwin policy: latency and coarse stage table

This entry point lives under evaluation/ and is transferred with Git. It does not
edit author model/policy/evaluator files. Historical evaluate/ scripts are unchanged.

## Submit

After syncing this directory and evaluation/analyze_replan_overlap.py:

```bash
cd ~/fcr_ws/AW-0/aw-0
mkdir -p runs/robotwin_compile
bash -n evaluation/robotwin_probe/run.slurm && sbatch \
  --output="$PWD/runs/robotwin_compile/%j.out" \
  --error="$PWD/runs/robotwin_compile/%j.err" \
  evaluation/robotwin_probe/run.slurm
```

Resources: one H200, 8 CPU cores, 96 GB host RAM, one hour. Compiler worker
parallelism is limited to two. The initial rollout remains BF16 eager to collect
real inputs with the original author runner (one clean click_alarmclock episode).
It records up to two replans with images, state, instruction, qpos and pre-inference
RNG/history. Diffusion intermediate recording is disabled for this experiment.

The second captured replan is selected if available; otherwise the first. Only a
compile worker is launched. There is no eager replay baseline or automatic speedup
comparison. This tests compiled inference on a real observation, not compiled
closed-loop task success. The initial capture task's success is not evidence of
compiled policy task success.

## What is compiled and measured

Only the original bound model.mot.forward is wrapped with torch.compile,
fullgraph=True, mode=reduce-overhead. The surrounding policy and simulation are
not compiled. Failure to compile is an error; no eager fallback is accepted as a
compiled speed result. PROBE_COMPILE_MODE=default selects the other compile mode
without enabling CUDA graphs through reduce-overhead.

1. First full policy call includes lazy compilation, inference and any initial
   graph capture. first_mode_call_ms reports that combined startup interval, not
   compiler time alone. It is excluded from steady samples.
2. Five warmup calls, restoring the captured RNG/history before every call.
3. Ten uninstrumented compiled policy calls, timed with perf_counter and GPU
   synchronization at round boundaries. Context restore and file I/O are excluded.
4. Coarse CUDA-event wrappers are then installed, followed by one instrumented
   warmup and ten instrumented calls. They wrap preprocessing/text/VAE/Qwen and use
   the model's existing denoising callback. They never wrap expert methods or the
   compiled MoT internals. No profiler, trace export or trace analyzer is invoked.

The stage table matches the earlier layout:

- Obs preprocess + H2D
- Text encode (warm cache)
- Vision latent (VAE)
- Vision tokens (Qwen)
- Memory build, only if the model has that module
- Denoising
- Other device timeline (ROUND minus nonoverlapping stages; derived, zero direct calls)
- ROUND device timeline
- ROUND wall

Each row reports mean/P50/P95 ms, percentage of mean ROUND device timeline, and
calls per round. P95 is the 95th percentile, not the maximum. CUDA-event spans
include launch gaps and are not summed GPU kernel busy time. Instrumented table
ROUND wall and clean_wall_ms are separate measurements. Internal experts and
nested action components are not measured by this entry point.

Compiled first-call and measured outputs are compared against the action saved
from the capture rollout, with atol=1e-4/rtol=1e-3. The stage outputs are also
compared against the first uninstrumented compiled output. Numeric/code agreement
is reported, not assumed. Compilation counters must show a graph. New graphs
compiled during measured calls invalidate a steady-state interpretation; the
samples remain available and are flagged in the report.

## Outputs and logs

All outputs are under runs/robotwin_compile/JOBID/:

- compile.out and compile.err: model load, compile startup, each warmup/sample and stage table.
- compile/report.json: clean_wall_ms, startup, graph counters, output checks and stage summary.
- compile/stage_summary.csv and compile/stage_table.md: table of coarse stage statistics.
- compile/stage_rounds.csv: individual instrumented round totals.
- compile/stage_report.json: stage statistics, output checks and compile stability.
- compile/actions.npz: measured compiled predictions and captured reference.
- capture/: observations, actions and replay contexts from the real rollout.
- summary.json: process status and before/after source hashes.

The report containing clean samples is written before coarse timing starts. A
later stage failure preserves the clean result and adds stage_error when possible.

```bash
tail -n 100 runs/robotwin_compile/JOBID.out
tail -n 100 runs/robotwin_compile/JOBID.err
tail -n 100 runs/robotwin_compile/JOBID/compile.out
tail -n 100 runs/robotwin_compile/JOBID/compile.err
cat runs/robotwin_compile/JOBID/compile/stage_table.md
```

Optional parameters: PROBE_WARMUP=5, PROBE_REPS=10, PROBE_STAGE_REPS=10,
PROBE_MAX_REPLANS=2, PROBE_TASK=click_alarmclock, PROBE_PROTOCOLS=clean.
Keep the default task/protocol/precision/steps to compare workload with the prior run.
Comparing to old job 581676 is historical context, not a fresh paired A/B experiment.

## Job 582914: compilation was disabled by an environment string

In PyTorch 2.7.1, Dynamo config reads TORCH_COMPILE_DISABLE directly from the
environment. The nonempty string "0" is truthy and still disables graph capture.
The compile worker now unsets both disable variables before importing PyTorch,
checks Dynamo's actual config before loading the model, and checks for a graph
immediately after its first call. compiler_environment.json and first_call.json
retain those checks. No zero-graph samples are accepted as compiled performance.

After syncing the fix, submit the normal Slurm command above. It runs a new
RoboTwin capture episode before compiled inference and the coarse stage table.
