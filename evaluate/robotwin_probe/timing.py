"""CUDA-event wrappers, installed only during a separate diagnostic call."""
from collections import defaultdict
import csv


class Timer:
    def __init__(self, torch):
        self.torch = torch
        self.events = defaultdict(list)
        self.pending = {}
        self.restores = []
        self.step_starts = []

    def event(self):
        e = self.torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def mark(self, name, start):
        if start:
            self.pending[name] = self.event()
        else:
            self.events[name].append((self.pending.pop(name), self.event()))

    def wrap(self, obj, attr, name, step_boundary=False):
        original = getattr(obj, attr)
        had_local, local = attr in vars(obj), vars(obj).get(attr)

        def timed(*args, **kwargs):
            if step_boundary:
                self.step_starts.append(self.event())
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

    def export(self, directory, layers, nis):
        if self.pending:
            raise RuntimeError('Incomplete timing intervals')
        rows, totals = [], {}
        for name, pairs in self.events.items():
            totals[name] = sum(s.elapsed_time(e) for s, e in pairs)
            layer_stage = (name.endswith(('.pre_attn', '.post_attn')) or
                           name == 'shared.attention' or name.startswith('nested.action.'))
            if layer_stage and len(pairs) != layers * nis:
                raise RuntimeError('Unexpected expert call count; CFG/another path changes step mapping')
            for i, (s, e) in enumerate(pairs):
                step, layer = ((i // layers, i % layers) if layer_stage else
                               ((i, '') if len(pairs) == nis else ('', '')))
                rows.append([name, i, step, layer, s.elapsed_time(e)])
        write_csv(directory / 'detail_calls.csv',
                  ['stage', 'call', 'step', 'layer', 'ms'], rows)
        if len(self.step_starts) != nis or len(self.events.get('denoising', [])) != 1:
            raise RuntimeError('Expected one joint denoising loop with nis video_pre calls')
        start, end = self.events['denoising'][0]
        boundaries = [start, *self.step_starts[1:], end]
        write_csv(directory / 'denoise_intervals.csv', ['step', 'ms', 'boundary'],
                  [[i, boundaries[i].elapsed_time(boundaries[i+1]), 'video_pre entry; see README']
                   for i in range(nis)])
        stages = sorted({r[0] for r in rows if r[2] != ''})
        per_step = []
        for step in range(nis):
            per_step.append([step, *[sum(r[4] for r in rows if r[0] == stage and r[2] == step)
                                     for stage in stages]])
        write_csv(directory / 'per_step_parts.csv', ['step', *stages], per_step)
        return totals


def write_csv(path, header, rows):
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def install(timer, policy, eager):
    model = policy.model
    timer.wrap(policy, '_build_obs_video', 'obs.preprocess')
    timer.wrap(policy.text, 'encode', 'obs.text_encode_warm')
    timer.wrap(model, '_encode_video_latents', 'obs.vae')
    timer.wrap(model, '_vit_prefix', 'obs.qwen')
    timer.wrap(model, '_video_pre', 'video.pre_dit', step_boundary=True)
    timer.wrap(model, '_video_post', 'video.post_dit')
    timer.wrap(model.video_scheduler, 'step', 'video.scheduler')
    timer.wrap(model.action_scheduler, 'step', 'action.scheduler')
    if eager:
        for name in model.mot.expert_order:
            expert = model.mot.experts[name]
            for attr in ('pre_attn', 'post_attn', 'pre_dit', 'post_dit'):
                if hasattr(expert, attr):
                    timer.wrap(expert, attr, f'{name}.{attr}')
        timer.wrap(model.mot, '_mixed_attention', 'shared.attention')
        for block in model.action_expert.blocks:
            timer.wrap(block.cross_attn, 'forward', 'nested.action.cross_attn')
            timer.wrap(block.ffn, 'forward', 'nested.action.ffn')
            timer.wrap(block.self_attn.o, 'forward', 'nested.action.out_projection')
    # Existing author-provided callback; installed/restored in memory only.
    had_local, local = '_bench_stage_timer' in vars(model), vars(model).get('_bench_stage_timer')
    model._bench_stage_timer = timer.mark
    timer.restores.append((model, '_bench_stage_timer', had_local, local))
