#!/usr/bin/env python3
"""Clean end-to-end A/B timing, optional eager attribution, and action inspection.

No simulator or skill executor is started. See bench_expert_latency.md.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy'))


def describe_action(chunk, expected_dim):
    x = np.asarray(chunk)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != expected_dim:
        raise ValueError(f'Expected nonempty [T,{expected_dim}] action, got {x.shape}')
    if not np.isfinite(x).all():
        raise ValueError('Action contains NaN or Inf')
    delta = np.diff(x, axis=0)
    return {
        'shape': list(x.shape), 'dtype': str(x.dtype), 'finite': True,
        'min_per_dim': x.min(axis=0).tolist(), 'max_per_dim': x.max(axis=0).tolist(),
        'adjacent_mean_abs_per_dim': (np.abs(delta).mean(axis=0).tolist()
                                      if len(delta) else None),
        'adjacent_max_abs_per_dim': (np.abs(delta).max(axis=0).tolist()
                                     if len(delta) else None),
    }


def compare_actions(x, y, atol, rtol):
    if x.shape != y.shape:
        return {'same_shape': False, 'shape_a': list(x.shape), 'shape_b': list(y.shape)}
    return {'same_shape': True, 'allclose': bool(np.allclose(x, y, atol=atol, rtol=rtol)),
            'mean_abs_difference': float(np.abs(x - y).mean()),
            'max_abs_difference': float(np.abs(x - y).max()),
            'atol': atol, 'rtol': rtol}


def summary(values):
    return {k: float(v) for k, v in zip(('mean', 'p50', 'p95'),
            (np.mean(values), np.percentile(values, 50), np.percentile(values, 95)))}


def write_csv(path, rows, header):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


class DetailTimer:
    """Leaf intervals only; no per-leaf synchronize. Current-stream timeline, not kernel busy time."""
    def __init__(self, torch):
        self.torch = torch
        self.restores = []
        self.events = []
        self.counts = {}
        self.token_lengths = {}
        self.input_shapes = {}

    def wrap(self, obj, attr, name):
        original = getattr(obj, attr)
        had_local, local = attr in vars(obj), vars(obj).get(attr)

        def timed(*args, **kwargs):
            index = self.counts.get(name, 0)
            self.counts[name] = index + 1
            if name.endswith('.pre_attn'):
                self.token_lengths[name.split('.')[0]] = int(args[1].shape[1])
            if name == 'action.pre_dit':
                self.input_shapes[name] = {k: list(v.shape) for k, v in kwargs.items()
                                          if hasattr(v, 'shape')}
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                return original(*args, **kwargs)
            finally:
                end.record()
                self.events.append((name, index, start, end))

        setattr(obj, attr, timed)
        self.restores.append((obj, attr, had_local, local))

    def restore(self):
        for obj, attr, had_local, local in reversed(self.restores):
            if had_local:
                setattr(obj, attr, local)
            else:
                delattr(obj, attr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--task', default='robotwin_wam')
    p.add_argument('--config', default='robotwin/sim_robotwin')
    p.add_argument('--dataset', default='')
    p.add_argument('--nis', type=int, default=20)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--reps', type=int, default=10)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--instruction', default='move the bottle to the left')
    p.add_argument('--input-npz', help='Optional head_camera/left_camera/right_camera uint8 images + state')
    p.add_argument('--detail', action='store_true', help='Additional eager-only attribution pass')
    p.add_argument('--trace', action='store_true', help='Additional PyTorch CPU/CUDA trace; not used for latency')
    p.add_argument('--multistream', action='store_true', help='Experimental parallel expert pre/post scheduling')
    p.add_argument('--output', required=True, help='New output directory (must not exist)')
    p.add_argument('--tag', default='baseline')
    p.add_argument('--compare', help='Other run directory; require matching workload and GPU')
    p.add_argument('--atol', type=float, default=1e-4)
    p.add_argument('--rtol', type=float, default=1e-3)
    p.add_argument('--extra', nargs='*', default=[])
    a = p.parse_args()
    if min(a.nis, a.warmup, a.reps) < 1:
        p.error('nis, warmup, reps must be positive')
    if a.atol < 0 or a.rtol < 0:
        p.error('comparison tolerances must be nonnegative')
    if not Path(a.ckpt).is_file():
        p.error('checkpoint does not exist')
    import torch
    from common.policy_runtime import PolicyRuntime
    from common.contract import CAMERAS

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    rt = PolicyRuntime(ckpt=str(Path(a.ckpt).resolve()), task=a.task, dataset=a.dataset,
                       num_inference_steps=a.nis, device='cuda', overrides=a.extra,
                       config_name=a.config)
    if a.detail and (rt.fast_joint or rt.compile_mode):
        p.error('--detail requires fast_joint and AGENTICWAM_TORCH_COMPILE off; clean timing supports both')
    if a.multistream and (a.detail or rt.fast_joint or rt.compile_mode):
        p.error('--multistream requires --detail off and both compilation paths off')
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(a.seed)
    if a.input_npz:
        with np.load(a.input_npz, allow_pickle=False) as data:
            imgs = {c: data[c].copy() for c in CAMERAS}
            state = np.asarray(data['state'], dtype=np.float64).reshape(-1)
    else:
        imgs = {c: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for c in CAMERAS}
        state = np.zeros(rt.action_dim, dtype=np.float64)
    for c, image in imgs.items():
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f'{c}: expected HxWx3 uint8')
    if state.shape != (rt.action_dim,) or not np.isfinite(state).all():
        raise ValueError('Wrong state shape or non-finite state')
    digest = hashlib.sha256()
    for x in [*(imgs[c] for c in CAMERAS), state]:
        digest.update(str((x.shape, str(x.dtype))).encode())
        digest.update(x.tobytes())

    def prepare():
        # Fixed observation, episode reset, and RNG reset: A/B does not confound growing history.
        rt.reset()
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        torch.cuda.synchronize()

    stream_check = None
    if a.multistream:
        from expert_streams import install
        prepare()
        reference, reference_code, _ = rt.infer(imgs, state, a.instruction)
        torch.cuda.synchronize()
        install(rt.policy.model.mot)
        prepare()
        candidate, candidate_code, _ = rt.infer(imgs, state, a.instruction)
        torch.cuda.synchronize()
        stream_check = compare_actions(np.asarray(reference), np.asarray(candidate), a.atol, a.rtol)
        stream_check['same_code'] = reference_code == candidate_code
        if not stream_check.get('allclose') or not stream_check['same_code']:
            (out / 'stream_equivalence_failure.json').write_text(json.dumps(stream_check, indent=2))
            raise RuntimeError('Multi-stream equivalence check failed; no performance claim made')

    for _ in range(a.warmup):
        prepare()
        rt.infer(imgs, state, a.instruction)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    walls, chunks, codes = [], [], []
    for _ in range(a.reps):
        prepare()
        start = time.perf_counter()
        chunk, code, _ = rt.infer(imgs, state, a.instruction)
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - start) * 1000)
        describe_action(chunk, rt.action_dim)
        chunks.append(np.asarray(chunk).copy())
        codes.append(code)
    actions = np.stack(chunks)
    peak = torch.cuda.max_memory_allocated()
    np.savez_compressed(out / 'actions.npz', absolute_qpos=actions, state=state,
                        relative_to_input=actions-state[None, None, :],
                        adjacent_delta=np.diff(actions, axis=1),
                        repeat_delta=np.diff(actions, axis=0))
    names = [f'dim_{i}' for i in range(rt.action_dim)]
    if rt.action_dim == 14:
        names = [*(f'left_joint_{i}' for i in range(6)), 'left_gripper',
                 *(f'right_joint_{i}' for i in range(6)), 'right_gripper']
    for filename, array in [('action_rows.csv', actions),
                            ('action_adjacent_delta.csv', np.diff(actions, axis=1))]:
        write_csv(out / filename,
                  ([r, t, *array[r, t].tolist()] for r in range(len(array))
                   for t in range(array.shape[1])), ['round', 'time_index', *names])
    sha = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True).stdout.strip()
    env = {k: v for k, v in os.environ.items()
           if k.startswith(('AGENTICWAM_', 'TORCH', 'CUDA_', 'OMP_'))
           and 'TOKEN' not in k and 'SECRET' not in k}
    workload = dict(ckpt=str(Path(a.ckpt).resolve()), ckpt_size=Path(a.ckpt).stat().st_size,
                    ckpt_mtime_ns=Path(a.ckpt).stat().st_mtime_ns, task=a.task,
                    config=a.config, dataset=a.dataset, extra=a.extra, nis=a.nis,
                    instruction=a.instruction, input_sha256=digest.hexdigest(), seed=a.seed,
                    action_dim=rt.action_dim, model_dtype=rt.model_dtype)
    report = dict(tag=a.tag, workload=workload, git_head=sha, env=env,
                  host=platform.node(), gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  fast_joint=rt.fast_joint, compile_mode=rt.compile_mode,
                  multistream=a.multistream, stream_equivalence=stream_check,
                  warmup=a.warmup, reps=a.reps, round_wall_ms=summary(walls),
                  round_samples_ms=walls, peak_allocated_bytes=peak,
                  input_kind='recorded observation' if a.input_npz else 'synthetic fixed images + zero state',
                  protocol='reset episode and RNG before every round; warm text cache',
                  action_format='absolute joint targets [round,time,dim]; NOT adjacent-frame deltas',
                  action=describe_action(actions[0], rt.action_dim),
                  repeat_comparison=([compare_actions(actions[0], actions[1], a.atol, a.rtol)]
                                     if a.reps > 1 else []))
    (out / 'codes.json').write_text(json.dumps(codes, indent=2))

    # Inspect intermediate action tensors in a separate, explicitly untimed call.
    captured = {}
    original_denorm = rt.policy._denormalize_action
    had_local = '_denormalize_action' in vars(rt.policy)
    old_local = vars(rt.policy).get('_denormalize_action')

    def capture_denorm(x):
        captured['normalized_model_action'] = x.detach().float().cpu().numpy().copy()
        value = original_denorm(x)
        captured['denormalized_model_action'] = np.asarray(value).copy()
        return value

    rt.policy._denormalize_action = capture_denorm
    try:
        prepare()
        diagnostic_chunk, _, _ = rt.infer(imgs, state, a.instruction)
        torch.cuda.synchronize()
    finally:
        if had_local:
            rt.policy._denormalize_action = old_local
        else:
            delattr(rt.policy, '_denormalize_action')
    if not captured:
        raise RuntimeError('Action diagnostic hook did not fire; unsupported output path')
    np.savez_compressed(out / 'action_intermediates.npz', **captured,
                        absolute_qpos=diagnostic_chunk, input_state=state)
    report['action_intermediate_shapes'] = {k: list(v.shape) for k, v in captured.items()}
    report['zero_state_note'] = ('With zero input state, values alone cannot distinguish absolute '
                                 'targets from offsets relative to input; inspect conversion code '
                                 'or provide a real nonzero state via --input-npz.')

    if a.detail:
        timer = DetailTimer(torch)
        model = rt.policy.model
        mot = model.mot
        n_layers = mot.num_layers
        try:
            for name in mot.expert_order:
                expert = mot.experts[name]
                for attr in ('pre_attn', 'post_attn'):
                    timer.wrap(expert, attr, f'{name}.{attr}')
                for attr in ('pre_dit', 'post_dit'):
                    if hasattr(expert, attr):
                        timer.wrap(expert, attr, f'{name}.{attr}')
            timer.wrap(mot, '_mixed_attention', 'shared.attention')
            for attr in ('_video_pre', '_video_post'):
                timer.wrap(model, attr, f'video.{attr}')
            # Do not wrap video_expert itself: it runs through pre_attn/post_attn, not forward().
            prepare()
            start = time.perf_counter()
            rt.infer(imgs, state, a.instruction)
            torch.cuda.synchronize()
            detail_wall = (time.perf_counter() - start) * 1000
        finally:
            timer.restore()
        rows, totals = [], {}
        for name, index, s, e in timer.events:
            ms = s.elapsed_time(e)
            is_layer = name.endswith(('pre_attn', 'post_attn')) or name == 'shared.attention'
            step, layer = ((index // n_layers, index % n_layers) if is_layer else
                           (index if timer.counts[name] == a.nis else '', ''))
            rows.append([name, index, step, layer, ms])
            totals[name] = totals.get(name, 0) + ms
        expected = a.nis * n_layers
        if timer.counts.get('shared.attention') != expected:
            raise RuntimeError(f'Expected {expected} shared calls, got {timer.counts}; '
                               'CFG or another path changes step mapping; no valid detail report')
        write_csv(out / 'detail_calls.csv', rows, ['stage', 'call', 'step', 'layer', 'ms'])
        groups = {g: sum(v for k, v in totals.items() if k.startswith(g + '.'))
                  for g in ('video', 'code', 'action', 'shared')}
        report['detail'] = dict(stage_ms=totals, group_ms=groups,
                                calls=timer.counts, token_lengths=timer.token_lengths,
                                input_shapes=timer.input_shapes,
                                instrumented_wall_ms=detail_wall,
                                warning='Instrumented current-stream intervals include launch gaps; '
                                        'NOT clean A/B latency or summed kernel busy time. '
                                        'video group excludes shared attention and VAE.')
        fraction = groups['video'] / detail_wall
        report['detail']['amdahl_estimates'] = {
            str(s): 1 / (1 - fraction + fraction / s) for s in (1.25, 1.5, 2, 10)}
        print('DETAIL attribution ms (additional instrumented round):', groups)
        print('Amdahl projections only:', report['detail']['amdahl_estimates'])

    if a.trace:
        from torch.profiler import profile, ProfilerActivity, record_function
        from expert_trace import TraceRanges
        prepare()
        ranges = TraceRanges()
        try:
            # Do not mutate methods inside an already compiled graph for attribution.
            if not (rt.fast_joint or rt.compile_mode):
                ranges.install(rt.policy.model)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True) as prof:
                with record_function('BENCH_ROUND'):
                    rt.infer(imgs, state, a.instruction)
                    torch.cuda.synchronize()
        finally:
            ranges.restore()
        prof.export_chrome_trace(str(out / 'trace.json'))
        (out / 'operators.txt').write_text(
            prof.key_averages().table(sort_by='self_device_time_total', row_limit=40))
        report['trace_expert_annotations'] = not (rt.fast_joint or rt.compile_mode)

    try:
        from torch._dynamo.utils import counters
        report['dynamo_counters'] = {str(k): dict(v) for k, v in counters.items()}
    except ImportError:
        report['dynamo_counters'] = None
    report['experts'] = {}
    for name, expert in rt.policy.model.mot.experts.items():
        dtypes = {}
        for param in expert.parameters():
            key = str(param.dtype)
            dtypes[key] = dtypes.get(key, 0) + param.numel()
        report['experts'][name] = dict(parameters=sum(dtypes.values()), parameters_by_dtype=dtypes,
                                       layers=len(expert.blocks),
                                       architecture={k: getattr(expert, k) for k in
                                                     ('hidden_dim', 'ffn_dim', 'num_heads', 'attn_head_dim')
                                                     if isinstance(getattr(expert, k, None), (int, float))})

    if a.compare:
        base_dir = Path(a.compare)
        base = json.loads((base_dir / 'report.json').read_text())
        if base['workload'] != workload or base['gpu'] != report['gpu']:
            raise ValueError('A/B workload or GPU mismatch; compare rejected')
        with np.load(base_dir / 'actions.npz', allow_pickle=False) as data:
            action_diff = compare_actions(data['absolute_qpos'][0], actions[0], a.atol, a.rtol)
        report['comparison'] = dict(
            baseline=str(base_dir.resolve()),
            mean_speedup=base['round_wall_ms']['mean'] / report['round_wall_ms']['mean'],
            action_first_round=action_diff,
            warning='Observed wall speedup, not proof of task quality; inspect env/software differences.')
        print('COMPARISON:', json.dumps(report['comparison'], indent=2))
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('CLEAN ROUND wall ms:', report['round_wall_ms'])
    print('ACTION:', json.dumps(report['action'], indent=2))
    print('Saved:', out)
    print('EXPERT_BENCH_OK')


if __name__ == '__main__':
    main()
