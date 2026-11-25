#!/usr/bin/env python3
"""Author RoboTwin capture -> compiled MoT latency and coarse stage statistics.

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
sys.path.insert(0, str(ROOT / 'evaluation'))


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


def compile_check(mode, unique_graphs, measured_new_graphs):
    if mode != 'compile':
        return None
    return dict(scope='MoT.forward only', fullgraph=True,
                unique_graphs=unique_graphs, new_graphs_during_measurement=measured_new_graphs,
                valid_steady_measurement=unique_graphs > 0 and measured_new_graphs == 0)


def worker(a):
    import numpy as np
    import torch
    from hook import restore

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

    reference = captured_action
    from torch._dynamo.utils import counters
    counters.clear()
    print(f'COMPILE_SETUP scope=MoT.forward mode={a.compile_mode} fullgraph=True', flush=True)
    original_mot.forward = torch.compile(original_mot.forward, mode=a.compile_mode, fullgraph=True)
    prepare()
    first_start = time.perf_counter()
    print(f'FIRST_MODE_CALL_START={a.mode}', flush=True)
    actual = infer()
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter()-first_start)*1000
    print(f'FIRST_MODE_CALL_DONE={a.mode} elapsed_ms={first_call_ms:.3f} '
          '(includes lazy compilation/capture when enabled; excluded from steady samples)', flush=True)
    parity = action_diff(reference, actual)
    capture_match = parity
    same_code = policy._last_code == meta.get('code')

    warmup_start = time.perf_counter()
    for index in range(a.warmup):
        prepare()
        infer()
        torch.cuda.synchronize()
        print(f'WARMUP {index+1}/{a.warmup} mode={a.mode}', flush=True)
    warmup_wall_ms = (time.perf_counter()-warmup_start)*1000
    samples, actions = [], []
    graphs_before_samples = counters.get('stats', {}).get('unique_graphs', 0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for index in range(a.reps):
        prepare()
        start = time.perf_counter()
        x = infer()
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000)
        actions.append(x.copy())
        print(f'SAMPLE {index+1}/{a.reps} mode={a.mode} elapsed_ms={samples[-1]:.3f}', flush=True)
    a.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(a.output / 'actions.npz', predictions=np.stack(actions),
                        captured=captured_action, replay_reference=reference)
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
    unique_graphs = counters.get('stats', {}).get('unique_graphs', 0)
    report['compile_check'] = compile_check(a.mode, unique_graphs, unique_graphs-graphs_before_samples)
    report['compile_mode'] = a.compile_mode
    report['comparison_reference'] = 'saved action from capture rollout; no eager replay baseline'
    report['first_mode_call_ms'] = first_call_ms
    report['first_mode_call_scope'] = 'first full policy call in selected mode; includes lazy compilation '
    report['first_mode_call_scope'] += 'and graph capture if applicable, not pure compiler time'
    report['warmup_wall_ms_including_context_restore'] = warmup_wall_ms
    report['sample_parity'] = [action_diff(reference, x) for x in actions]
    report['accuracy_check_passed'] = (parity.get('allclose', False) and same_code and
                                       all(x.get('allclose', False) for x in report['sample_parity']))
    report['experts'] = {n: dict(parameters=sum(p.numel() for p in e.parameters()),
                                 dtype_counts={dtype: sum(p.numel() for p in e.parameters() if str(p.dtype) == dtype)
                                               for dtype in {str(p.dtype) for p in e.parameters()}})
                         for n, e in original_mot.experts.items()}
    (a.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

    # Coarse timers are installed only AFTER all clean latency samples are saved.
    if unique_graphs < 1:
        raise RuntimeError('No Dynamo compiled graph recorded; cannot claim compiled speed')
    from stages import measure_stages
    try:
        report['stage_measurement'] = measure_stages(a, policy, prepare, infer, actual, torch)
    except Exception as exc:
        report['stage_error'] = f'{type(exc).__name__}: {exc}'
        (a.output / 'report.json').write_text(json.dumps(report, indent=2))
        raise
    (a.output / 'report.json').write_text(json.dumps(report, indent=2))
    print('ROBOTWIN_REPLAY_OK', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=['all', 'capture', 'replay'], default='all')
    p.add_argument('--capture-dir', type=Path)
    p.add_argument('--task', default='click_alarmclock')
    p.add_argument('--protocols', default='clean', choices=['clean', 'randomized', 'clean,randomized'])
    p.add_argument('--modes', choices=['compile'], default='compile', help='compile only; no eager baseline')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--reps', type=int, default=10)
    p.add_argument('--max-replans', type=int, default=2)
    p.add_argument('--stage-reps', type=int, default=10)
    p.add_argument('--compile-mode', choices=['default', 'reduce-overhead', 'max-autotune'],
                   default='reduce-overhead')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--mode', choices=['compile'], default='compile', help=argparse.SUPPRESS)
    p.add_argument('--record', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    a.output = a.output.resolve()
    if min(a.warmup, a.reps, a.max_replans, a.stage_reps) < 1:
        p.error('warmup/reps/max-replans/stage-reps must be positive')
    if a.worker:
        worker(a)
        return
    modes = a.modes.split(',')
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
                       RW_PROBE_MAX_REPLANS=str(a.max_replans), RW_PROBE_CAPTURE_DIFFUSION='0')
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
            disabled = '0'
            env.update(TORCH_COMPILE_DISABLE=disabled, TORCHDYNAMO_DISABLE=disabled,
                       AGENTICWAM_FAST_JOINT='0', AGENTICWAM_TORCH_COMPILE='')
            cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--mode', mode,
                   '--record', str(record), '--output', str(a.output / mode),
                   '--warmup', str(a.warmup), '--reps', str(a.reps),
                   '--compile-mode', a.compile_mode, '--stage-reps', str(a.stage_reps)]
            print(f'REPLAY_START={mode}', flush=True)
            with (a.output / f'{mode}.out').open('w') as stdout, (a.output / f'{mode}.err').open('w') as stderr:
                result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            status['phases'][mode] = 'ok' if result.returncode == 0 else 'failed'
        report_path = a.output / 'compile/report.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            status['compiled_result'] = {k: report.get(k) for k in
                ('clean_wall_ms', 'compile_check', 'accuracy_check_passed', 'first_mode_call_ms',
                 'stage_error')}
            status['compiled_result']['report'] = str(report_path)
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
