"""Coarse CUDA-event stage statistics outside the compiled MoT graph. No profiler."""
from collections import defaultdict
import csv
import json
import time


ROUND = 'ROUND device timeline'
DISJOINT = ('Obs preprocess + H2D', 'Text encode (warm cache)', 'Vision latent (VAE)',
            'Vision tokens (Qwen)', 'Memory build (history -> slots)', 'Denoising')


class StageTimer:
    def __init__(self, torch):
        self.torch = torch
        self.events = defaultdict(list)
        self.pending = {}
        self.restores = []

    def mark(self, name, start):
        name = 'Denoising' if name == 'denoising' else name
        event = self.torch.cuda.Event(enable_timing=True)
        event.record()
        if start:
            if name in self.pending:
                raise RuntimeError(f'Overlapping stage: {name}')
            self.pending[name] = event
        else:
            self.events[name].append((self.pending.pop(name), event))

    def replace(self, obj, attr, value):
        self.restores.append((obj, attr, attr in vars(obj), vars(obj).get(attr)))
        setattr(obj, attr, value)

    def wrap(self, obj, attr, name):
        original = getattr(obj, attr)

        def timed(*args, **kwargs):
            self.mark(name, True)
            try:
                return original(*args, **kwargs)
            finally:
                self.mark(name, False)

        self.replace(obj, attr, timed)

    def install(self, policy):
        model = policy.model
        self.wrap(policy, '_build_obs_video', DISJOINT[0])
        self.wrap(policy.text, 'encode', DISJOINT[1])
        self.wrap(model, '_encode_video_latents', DISJOINT[2])
        self.wrap(model, '_vit_prefix', DISJOINT[3])
        if getattr(model, 'code_memory', None) is not None:
            self.wrap(model, '_build_code_memory_state', DISJOINT[4])
        # The model's existing callback bounds the loop; never wrap compiled experts.
        self.replace(model, '_bench_stage_timer', self.mark)

    def clear(self):
        if self.pending:
            raise RuntimeError(f'Incomplete stages: {list(self.pending)}')
        self.events.clear()

    def restore(self):
        for obj, attr, had_local, local in reversed(self.restores):
            if had_local:
                setattr(obj, attr, local)
            else:
                delattr(obj, attr)
        self.restores.clear()

    def totals(self):
        if self.pending:
            raise RuntimeError(f'Incomplete stages: {list(self.pending)}')
        if len(self.events.get('Denoising', [])) != 1:
            raise RuntimeError('Expected one joint denoising callback; check model version/path')
        values = {name: sum(s.elapsed_time(e) for s, e in pairs)
                  for name, pairs in self.events.items()}
        residual = values[ROUND] - sum(values.get(name, 0) for name in DISJOINT)
        if residual < -0.1:
            raise RuntimeError('Coarse stage intervals overlap; residual would be negative')
        values['Other device timeline'] = residual
        return values


def summarize(rounds, counts):
    import numpy as np
    denominator = float(np.mean([r[ROUND] for r in rounds]))
    names = [name for name in DISJOINT if any(name in r for r in rounds)]
    names += ['Other device timeline', ROUND, 'ROUND wall']
    result = []
    for name in names:
        if not all(name in r for r in rounds):
            raise RuntimeError(f'Stage missing from some rounds: {name}')
        v = np.asarray([r[name] for r in rounds])
        mean = float(v.mean())
        result.append(dict(stage=name, mean_ms=mean, p50_ms=float(np.percentile(v, 50)),
                           p95_ms=float(np.percentile(v, 95)),
                           percent=None if name == 'ROUND wall' else 100*mean/denominator,
                           calls_per_round=None if name == 'ROUND wall' else
                           float(np.mean([c.get(name, 0) for c in counts]))))
    return result


def measure_stages(a, policy, prepare, infer, compiled_reference, torch):
    from runner import action_diff
    from torch._dynamo.utils import counters
    timer = StageTimer(torch)
    rounds, counts, parity = [], [], []
    try:
        timer.install(policy)
        # Allow any changed Python guards to settle before instrumented samples.
        prepare()
        infer()
        torch.cuda.synchronize()
        timer.clear()
        graphs_before = counters.get('stats', {}).get('unique_graphs', 0)
        for index in range(a.stage_reps):
            prepare()
            timer.clear()
            start = time.perf_counter()
            timer.mark(ROUND, True)
            action = infer()
            timer.mark(ROUND, False)
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter()-start)*1000
            values = timer.totals()
            values['ROUND wall'] = wall_ms
            rounds.append(values)
            counts.append({k: len(v) for k, v in timer.events.items()})
            parity.append(action_diff(compiled_reference, action))
            print(f'STAGE_SAMPLE {index+1}/{a.stage_reps} wall_ms={wall_ms:.3f} '
                  f'denoising_ms={values["Denoising"]:.3f}', flush=True)
        measured_new_graphs = counters.get('stats', {}).get('unique_graphs', 0)-graphs_before
    finally:
        timer.restore()
    statistics = summarize(rounds, counts)
    fields = ['stage', 'mean_ms', 'p50_ms', 'p95_ms', 'percent', 'calls_per_round']
    with (a.output/'stage_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(statistics)
    with (a.output/'stage_rounds.csv').open('w', newline='') as f:
        names = [r['stage'] for r in statistics]
        writer = csv.writer(f)
        writer.writerow(['round', *names])
        writer.writerows([[i, *[r[name] for name in names]] for i, r in enumerate(rounds)])
    lines = ['| Stage | Mean (ms) | P50 (ms) | P95 (ms) | Percent | Calls/Round |',
             '|---|---:|---:|---:|---:|---:|']
    for row in statistics:
        percent = '-' if row['percent'] is None else f'{row["percent"]:.1f}%'
        calls = '-' if row['calls_per_round'] is None else f'{row["calls_per_round"]:.1f}'
        lines.append(f'| {row["stage"]} | {row["mean_ms"]:.3f} | {row["p50_ms"]:.3f} | '
                     f'{row["p95_ms"]:.3f} | {percent} | {calls} |')
    (a.output/'stage_table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)
    result = dict(rounds=a.stage_reps, statistics=statistics,
                  new_graphs_during_measurement=measured_new_graphs,
                  valid_steady_measurement=measured_new_graphs == 0,
                  output_parity_to_uninstrumented_compiled=parity,
                  output_check_passed=all(p.get('allclose', False) for p in parity),
                  scope='Separate instrumented compiled policy calls; CUDA current-stream spans '
                        'include launch gaps. Percent denominator is mean ROUND device timeline. '
                        'Other is a residual (zero direct calls). P95 is a percentile, not maximum. '
                        'No expert hooks, profiler or eager replay baseline.')
    (a.output/'stage_report.json').write_text(json.dumps(result, indent=2))
    return result
