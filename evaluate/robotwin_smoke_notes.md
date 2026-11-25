# First validate the author's RoboTwin release smoke test

Reference: docs/ROBOTWIN.md, configs/robotwin/benchmark.yaml.
The rollout smoke is `scripts/eval_robotwin_release.py wam --tasks
click_alarmclock --plan smoke --gpus 0`: one clean and one randomized episode.
`robotwin/dataprep/smoke.py` is a different dataset-loading check.

The server must already have a working `configs/robotwin/site.yaml` (or an
`AGENTICWAM_HYBRID_SITE` override), RoboTwin assets and simulation dependencies.
The local checkout has no site.yaml, so this template deliberately does not guess
simulator or NVIDIA Vulkan ICD paths. The configuration must name:

- `python_bin`: bin directory of the joint model/simulator environment.
- `robotwin_root`: installed RoboTwin checkout with assets.
- `vulkan_icd`: actual NVIDIA Vulkan ICD JSON on the compute node.
- `hf_cache_policy`: policy cache (usually /public/home/terrypei/fcr_ws/AW-0/hf-cache here).
- release weights, dataset_stats.json, task_skill_prompt.json and local SDAR path.

Use the working local-processor Cosmos patch from the earlier benchmark. The
Slurm template forwards `paths=cluster` through the author's launcher, so model
components use your existing local checkpoint paths instead of default placeholders.
It uses BF16 eager without changing the author's 20-step/24-replan configuration.

After uploading evaluate/robotwin_release_smoke.slurm:

```bash
cd ~/fcr_ws/AW-0/aw-0
cat configs/robotwin/site.yaml
mkdir -p runs/robotwin_smoke
bash -n evaluate/robotwin_release_smoke.slurm
sbatch --output="$PWD/runs/robotwin_smoke/%j.out" \
       --error="$PWD/runs/robotwin_smoke/%j.err" \
       evaluate/robotwin_release_smoke.slurm
```

Preflight checks the resolved machine paths, followed by the author's dry-run
and actual smoke command. Disabling saved videos still requires camera rendering;
it does not bypass Vulkan. No hybrid perception servers are started.

If the job fails, inspect Slurm stderr plus the benchmark shard log in
evaluate_results/robotwin_benchmark/wam_smoke_JOB/logs/ and runs/eval_logs/.
Keep each job's unique name to avoid reusing old completed results.
Passing the pipeline means both episodes finish with valid observations and
actions and no runtime errors; it does not require winning both tasks. The
author's results contain per-episode completion and success counts.

After this passes, connect recording to robotwin/deploy/deploy_policy.py's
`step`/`_fill_action_queue` path. Count successful task_env.take_action calls,
record the exact queued targets and corresponding fresh observations at replans,
and exclude hybrid code execution. Do not connect only PolicyRuntime's recorder:
this launcher does not call PolicyRuntime.infer. The simulation is then useful
for the 32-horizon/24-executed-step overlap experiment, while clean inference
benchmarks remain separate from rendering and recording overhead.
