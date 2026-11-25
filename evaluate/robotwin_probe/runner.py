#!/usr/bin/env python3
"""Author RoboTwin smoke -> captured real observation -> fair eager/compile replay.

All instrumentation is external and process-local. No author source is rewritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'evaluate'))


def source_hashes():
    paths = ['robotwin/deploy/deploy_policy.py', 'robotwin/eval/robotwin_entry.py',
             'src/agenticwam/models/cosmos_mot/agenticwam_model.py',
             'src/agenticwam/models/cosmos_mot/generic_mot.py']
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def action_diff(x, y, atol=1e-4, rtol=1e-3):
    import numpy as np
    if x.shape != y.shape:
        return dict(same_shape=False, shape_a=list(x.shape), shape_b=list(y.shape))
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return dict(same_shape=True, allclose=False, finite=False)
    return dict(same_shape=True, allclose=bool(np.allclose(x, y, atol=atol, rtol=rtol)),
                mean_abs=float(np.abs(x-y).mean()), max_abs=float(np.abs(x-y).max()),
                atol=atol, rtol=rtol)


def worker(a):
    import numpy as np
    import torch
    from hook import restore
    from timing import Timer, install, write_csv

    directory = a.record.parent
    info = json.loads((directory / 'run_info.json').read_text())
    records = [json.loads(s) for s in (directory / 'index.jsonl').read_text().splitlines()]
    meta = next(r for r in records if r['file'] == a.record.name)
    os.environ.update(info['env'])
    os.environ.pop('RW_PROBE_CAPTURE', None)
    os.environ['AGENTICWAM_FAST_JOINT'] = '0'
    os.environ['AGENTICWAM_TORCH_COMPILE'] = ''
    sys.path.insert(0, str(ROOT / 'src'))
    sys.path.insert(0, str(ROOT / 'robotwin/deploy'))
    from deploy_policy import get_model

    args = dict(info['usr_args'])
    args['eval_output_dir'] = None
    policy = get_model(args)
    if policy.hybrid is not None:
        raise ValueError('Only WAM replay is supported')
    with np.load(a.record, allow_pickle=False) as data:
        obs = dict(observation={c: dict(rgb=data[c].copy()) for c in
                               ('head_camera', 'left_camera', 'right_camera')},
                   joint_action=dict(vector=data['state'].tolist()))
        captured_action = data['absolute_qpos'].copy()
    # Only load contexts produced by this tool from your own rollout.
    saved = torch.load(a.record.with_suffix('.context.pt'), map_location='cpu', weights_only=False)
    model, original_mot = policy.model, policy.model.mot
    layers = original_mot.num_layers

    def prepare():
        restore(policy, saved)
        torch.cuda.synchronize()

    def infer():
        return np.asarray(policy._infer_action_chunk(obs, meta['instruction']))

    # Prove the observation/context replay matches the action actually recorded in simulation.
    prepare()
    reference = infer()
    torch.cuda.synchronize()
    capture_match = action_diff(captured_action, reference)
    if not capture_match.get('allclose'):
        raise RuntimeError(f'Replay does not match captured action: {capture_match}')

    if a.mode == 'compile':
        # Patch this instance's bound forward, not the read-only model.mot accessor.
        original_mot.forward = torch.compile(original_mot.forward, mode='reduce-overhead', fullgraph=True)
    elif a.mode == 'multistream':
        from expert_streams import install as install_streams
        install_streams(original_mot)
    prepare()
    actual = infer()
    torch.cuda.synchronize()
    parity = action_diff(reference, actual)
    same_code = policy._last_code == meta.get('code')

    for _ in range(a.warmup):
        prepare()
        infer()
    samples, actions = [], []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(a.reps):
        prepare()
        start = time.perf_counter()
        x = infer()
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000)
        actions.append(x.copy())
    a.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(a.output / 'actions.npz', predictions=np.stack(actions),
                        captured=captured_action, replay_reference=reference)
    from torch._dynamo.utils import counters
    report = dict(mode=a.mode, record=str(a.record), task=info['usr_args'].get('task_name'),
                  protocol=info['usr_args'].get('task_config'), gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, cuda=torch.version.cuda, nis=policy.num_inference_steps,
                  layers=layers, action_shape=list(actual.shape),
                  warmup=a.warmup, reps=a.reps, samples_ms=samples,
                  clean_wall_ms=dict(mean=float(np.mean(samples)), p50=float(np.percentile(samples, 50)),
                                     p95=float(np.percentile(samples, 95))),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  capture_match=capture_match, output_parity=parity, same_code=same_code,
                  dynamo_counters={str(k): dict(v) for k, v in counters.items()},
                  source_hashes=source_hashes(),
                  timing_scope='warm-text-cache policy inference on captured real observation; '
                               'excludes simulation/rendering, context reset, file I/O and predicted-video export')
    report['experts'] = {n: dict(parameters=sum(p.numel() for p in e.parameters()),
                                 dtype_counts={dtype: sum(p.numel() for p in e.parameters() if str(p.dtype) == dtype)
                                               for dtype in {str(p.dtype) for p in e.parameters()}})
                         for n, e in original_mot.experts.items()}
    (a.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

    # The clean result is saved BEFORE any diagnostic overhead or potential profiler failure.
    if a.mode != 'multistream':
        timer = Timer(torch)
        prepare()
        try:
            install(timer, policy, eager=a.mode == 'eager')
            infer()
            torch.cuda.synchronize()
        finally:
            timer.restore()
        totals = timer.export(a.output, layers, policy.num_inference_steps)
        report['diagnostic_ms'] = totals
        # Nested action components are already included in action.post_attn.
        report['action_nested_ms'] = {k: v for k, v in totals.items() if k.startswith('nested.action.')}
        report['expert_ms'] = ({n: sum(v for k, v in totals.items() if k.startswith(n+'.'))
                               for n in ('video', 'code', 'action', 'shared')}
                              if a.mode == 'eager' else None)
        report['internal_attribution'] = ('available in separate eager diagnostic call' if a.mode == 'eager'
                                          else 'compiled internal experts are opaque; not reported as zero')
        print('DIAGNOSTIC expert ms:', report['expert_ms'], flush=True)
        report['diffusion_expert_ms'] = timer.diffusion_expert_ms if a.mode == 'eager' else None
        print('DIFFUSION expert ms:', report['diffusion_expert_ms'], flush=True)
        print('ACTION post_attn nested ms:', report['action_nested_ms'], flush=True)
        print('DENOISING total ms:', totals.get('denoising'), flush=True)
        for name, ms in sorted(totals.items()):
            print(f'SECTION_MS {name}={ms:.3f} (inclusive CUDA event span)', flush=True)

    if a.trace and a.mode == 'eager':
        from expert_trace import TraceRanges
        from torch.profiler import profile, ProfilerActivity, record_function
        prepare()
        ranges = TraceRanges()
        try:
            ranges.install(model)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function('BENCH_ROUND'):
                    infer()
                    torch.cuda.synchronize()
        finally:
            ranges.restore()
        prof.export_chrome_trace(str(a.output / 'trace.json'))
        (a.output / 'operators.txt').write_text(
            prof.key_averages().table(sort_by='self_device_time_total', row_limit=40))
    if a.mode == 'compile' and counters.get('stats', {}).get('unique_graphs', 0) < 1:
        raise RuntimeError('No Dynamo compiled graph recorded; cannot claim compiled speed')
    (a.output / 'report.json').write_text(json.dumps(report, indent=2))
    print('ROBOTWIN_REPLAY_OK', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=['all', 'capture', 'replay'], default='all')
    p.add_argument('--capture-dir', type=Path)
    p.add_argument('--task', default='click_alarmclock')
    p.add_argument('--protocols', default='clean', choices=['clean', 'randomized', 'clean,randomized'])
    p.add_argument('--modes', default='eager', help='eager; optionally add compile or multistream')
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--max-replans', type=int, default=8)
    p.add_argument('--trace', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--mode', choices=['eager', 'compile', 'multistream'], help=argparse.SUPPRESS)
    p.add_argument('--record', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    a.output = a.output.resolve()
    if min(a.warmup, a.reps, a.max_replans) < 1:
        p.error('warmup/reps/max-replans must be positive')
    if a.worker:
        worker(a)
        return
    modes = a.modes.split(',')
    if not modes or len(set(modes)) != len(modes) or any(m not in ('eager', 'compile', 'multistream') for m in modes):
        p.error('modes must be distinct eager/compile/multistream entries')
    if a.phase == 'replay' and not a.capture_dir:
        p.error('--phase replay requires --capture-dir')
    a.output.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    status = dict(source_before=before, phases={})
    try:
        capture_dir = (a.capture_dir or a.output / 'capture').resolve()
        if a.phase in ('all', 'capture'):
            bootstrap = a.output / 'bootstrap'
            bootstrap.mkdir()
            (bootstrap / 'sitecustomize.py').write_text(
                f'import sys\nsys.path.insert(0, {str(HERE)!r})\nfrom hook import enable\nenable()\n')
            env = dict(os.environ, RW_PROBE_CAPTURE=str(capture_dir),
                       RW_PROBE_MAX_REPLANS=str(a.max_replans))
            env['PYTHONPATH'] = str(bootstrap) + os.pathsep + env.get('PYTHONPATH', '')
            cmd = [sys.executable, str(ROOT / 'scripts/eval_robotwin_release.py'), 'wam',
                   '--tasks', a.task, '--plan', 'smoke', '--protocols', a.protocols,
                   '--gpus', '0', '--name', 'probe_'+a.output.name,
                   '--launch-args', '--extra', 'paths=cluster']
            print('AUTHOR SMOKE:', ' '.join(cmd), flush=True)
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            status['phases']['author_smoke'] = 'process_completed'
        shards = sorted(capture_dir.glob('*/index.jsonl'))
        if not shards:
            raise RuntimeError('No captured replans. Check smoke logs and whether the external hook loaded')
        from analyze_replan_overlap import analyze
        for index in shards:
            analyze(index.parent, 1e-3)
        status['capture_dir'] = str(capture_dir)
        if a.phase == 'capture':
            return
        # Prefer a second replan to avoid first-instruction and first-episode cold effects.
        candidates = [(index.parent, [json.loads(s) for s in index.read_text().splitlines()]) for index in shards]
        directory, records = max(candidates, key=lambda pair: len(pair[1]))
        rec = records[min(1, len(records)-1)]
        record = directory / rec['file']
        status['record'] = str(record)
        for mode in modes:
            env = dict(os.environ)
            env.pop('RW_PROBE_CAPTURE', None)
            disabled = '0' if mode == 'compile' else '1'
            env.update(TORCH_COMPILE_DISABLE=disabled, TORCHDYNAMO_DISABLE=disabled,
                       AGENTICWAM_FAST_JOINT='0', AGENTICWAM_TORCH_COMPILE='')
            cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--mode', mode,
                   '--record', str(record), '--output', str(a.output / mode),
                   '--warmup', str(a.warmup), '--reps', str(a.reps)]
            if a.trace:
                cmd.append('--trace')
            print(f'REPLAY_START={mode}', flush=True)
            with (a.output / f'{mode}.out').open('w') as stdout, (a.output / f'{mode}.err').open('w') as stderr:
                result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            status['phases'][mode] = 'ok' if result.returncode == 0 else 'failed'
            if result.returncode == 0 and mode != 'multistream':
                subprocess.run([sys.executable, str(ROOT / 'evaluation/analyze_expert_bench.py'),
                                str(a.output / mode)], check=True)
                analysis = json.loads((a.output / mode / 'analysis.json').read_text())
                status.setdefault('analysis', {})[mode] = dict(
                    status=analysis['analysis_status'], trace_status=analysis['trace_status'],
                    trace_error=analysis.get('trace_error'))
        reports = {m: json.loads((a.output / m / 'report.json').read_text()) for m in modes
                   if status['phases'].get(m) == 'ok'}
        if 'eager' in reports:
            status['comparison'] = {m: dict(
                mean_speedup=reports['eager']['clean_wall_ms']['mean']/r['clean_wall_ms']['mean'],
                action_parity=r['output_parity'], same_code=r['same_code']) for m, r in reports.items()}
        print(json.dumps(status, indent=2), flush=True)
        if any(status['phases'].get(m) != 'ok' for m in modes):
            raise RuntimeError('Some replay modes failed; inspect mode stderr. Failed modes are not speed results')
        print('ROBOTWIN_PROBE_OK', flush=True)
    finally:
        status['source_after'] = source_hashes()
        status['author_source_unchanged'] = before == status['source_after']
        (a.output / 'summary.json').write_text(json.dumps(status, indent=2))
        if not status['author_source_unchanged']:
            raise RuntimeError('Author source files changed during the experiment')


if __name__ == '__main__':
    main()
