#!/usr/bin/env python3
"""Measure stages INSIDE real rt.infer rounds on the GPU box.

python evaluate/bench_aw0_stages.py --ckpt <wam.pt> --nis 20 --reps 20

CUDA event intervals are device timeline latency (including CPU launch gaps),
not summed kernel execution time. Synchronise only at round boundaries.
Memory reads are nested in denoising: do not add them to the denoising total.
Warmup populates the text cache; measured rounds retain evolving code history.
"""
import argparse
from collections import defaultdict
from functools import wraps
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy'))


class StageTimer:
    def __init__(self):
        self.events = defaultdict(list)
        self.pending = {}
        self.restores = []

    def mark(self, name, start):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        if start:
            self.pending[name] = event
        else:
            self.events[name].append((self.pending.pop(name), event))

    def wrap(self, obj, attr, name):
        original = getattr(obj, attr)
        had_local = attr in vars(obj)
        local = vars(obj).get(attr)

        @wraps(original)
        def timed(*args, **kwargs):
            self.mark(name, True)
            try:
                return original(*args, **kwargs)
            finally:
                self.mark(name, False)

        setattr(obj, attr, timed)
        self.restores.append((obj, attr, had_local, local))

    def restore(self):
        for obj, attr, had_local, local in reversed(self.restores):
            if had_local:
                setattr(obj, attr, local)
            else:
                delattr(obj, attr)

    def totals(self):
        return {name: sum(s.elapsed_time(e) for s, e in pairs)
                for name, pairs in self.events.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--task', default='robotwin_wam')
    ap.add_argument('--config', default='robotwin/sim_robotwin', help='Hydra root config')
    ap.add_argument('--dataset', default='')
    ap.add_argument('--nis', type=int, default=20)
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--extra', nargs='*', default=[], help='Hydra key=value overrides')
    a = ap.parse_args()
    if a.reps < 1 or a.warmup < 1 or a.nis < 1:
        ap.error('reps, warmup and nis must be positive')
    if not Path(a.ckpt).is_file():
        ap.error('checkpoint must exist; smoke mode is not a valid benchmark')

    from common.policy_runtime import PolicyRuntime
    from common.contract import CAMERAS
    rt = PolicyRuntime(ckpt=str(Path(a.ckpt).resolve()), task=a.task,
                       dataset=a.dataset, num_inference_steps=a.nis,
                       device='cuda', overrides=a.extra, config_name=a.config)
    model = rt.policy.model
    rng = np.random.default_rng(0)
    imgs = {c: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for c in CAMERAS}
    state = np.zeros(14, dtype=np.float32)
    instr = 'move the bottle to the left'
    for _ in range(a.warmup):
        rt.infer(imgs, state, instr)
    torch.cuda.synchronize()

    timer = StageTimer()
    timer.wrap(rt.policy, '_build_obs_video', 'obs preprocess + H2D')
    timer.wrap(rt.policy.text, 'encode', 'text encode (warm cache)')
    timer.wrap(model, '_encode_video_latents', 'vision latent (VAE)')
    timer.wrap(model, '_vit_prefix', 'vision tokens (Qwen)')
    if getattr(model, 'code_memory', None) is not None:
        timer.wrap(model, '_build_code_memory_state', 'memory build (history -> slots)')
        timer.wrap(model.code_memory, 'read', 'memory read (nested in denoising)')
    model._bench_stage_timer = timer.mark
    rounds = []
    counts = []
    walls = []
    try:
        for _ in range(a.reps):
            timer.events.clear()
            timer.pending.clear()
            torch.cuda.synchronize()
            start = time.perf_counter()
            timer.mark('ROUND device timeline', True)
            rt.infer(imgs, state, instr)
            timer.mark('ROUND device timeline', False)
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - start) * 1000)
            row = timer.totals()
            if 'denoising' not in row:
                raise RuntimeError('No joint denoising timer fired; this benchmark requires the joint inference path')
            # Disjoint stages only: memory read is already inside denoising.
            disjoint = ['obs preprocess + H2D', 'text encode (warm cache)',
                        'vision latent (VAE)', 'vision tokens (Qwen)',
                        'memory build (history -> slots)', 'denoising']
            row['other device timeline'] = row['ROUND device timeline'] - sum(row.get(k, 0) for k in disjoint)
            rounds.append(row)
            counts.append({k: len(v) for k, v in timer.events.items()})
    finally:
        timer.restore()
        del model._bench_stage_timer

    print(f'\nnis={rt.num_inference_steps}, fast_joint={rt.fast_joint}, '
          f'memory_enabled={getattr(model, "code_memory_enabled", False)}')
    print('Warm rounds, evolving history; memory read is INCLUDED in denoising.')
    print('CUDA intervals include launch gaps; percentages use ROUND device timeline.')
    print(f'{"stage":44s} {"mean ms":>10} {"p50 ms":>10} {"p95 ms":>10} {"share":>8} {"calls/round":>12}')
    total = np.mean([r['ROUND device timeline'] for r in rounds])
    names = list(dict.fromkeys(k for r in rounds for k in r))
    for name in names + ['ROUND wall']:
        values = np.array(walls if name == 'ROUND wall' else [r.get(name, 0) for r in rounds])
        calls = '-' if name == 'ROUND wall' else f'{np.mean([c.get(name, 0) for c in counts]):.1f}'
        share = '-' if name == 'ROUND wall' else f'{100 * values.mean() / total:.1f}%'
        print(f'{name:44s} {values.mean():10.2f} {np.percentile(values, 50):10.2f} '
              f'{np.percentile(values, 95):10.2f} {share:>8} {calls:>12}')
    if getattr(model, 'code_memory', None) is None:
        print('Memory module absent: memory stages are N/A (not measured as zero).')


if __name__ == '__main__':
    main()
