# Evaluation fixes and offline analysis

New analysis code lives here and can be transferred with Git. The new compiled probe is in evaluation/robotwin_probe; see its README.md.
Historical evaluate/robotwin_probe entry points remain for old jobs.

## Recover job 581676 without another GPU job

```bash
cd ~/fcr_ws/AW-0/aw-0
python evaluation/analyze_expert_bench.py runs/robotwin_probe/581676/eager
```

The original error only established that the count of complete BENCH_ROUND
events was not one. It did not distinguish missing events from duplicate events.
Kineto may export both CPU and GPU user annotations with the same name; the
analyzer selects exactly one CPU annotation. It rejects multiple CPU rounds
rather than merging different measurements. Missing or ambiguous markers retain
the CUDA-event detail analysis, write marker diagnostics to analysis.json, and
set analysis_status=partial and trace_status=failed. This does not assert a valid
GPU trace. Use --strict-trace to also return a failing exit code.

`--details-only` skips trace analysis entirely. Neither mode rewrites raw traces,
action captures, report.json, or the old run's summary.json. The old Slurm job
remains FAILED even if offline reanalysis succeeds.

Tests (no CUDA or PyTorch required):

```bash
python -m unittest discover -s evaluation -p 'test_*.py' -v
```
