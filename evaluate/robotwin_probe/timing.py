"""CUDA-event wrappers, installed only during a separate diagnostic call."""
from collections import defaultdict
import csv
import time


class Timer:
    def __init__(self, torch):
        self.torch = torch
        self.events = defaultdict(list)
        self.pending = {}
        self.restores = []
        self.step_starts = []
        self.wall_ms = defaultdict(list)
        self.current_step = None
        self.pending_steps = {}
        self.event_steps = defaultdict(list)
        self.diffusion_expert_ms = {}
        self.internal_experts_available = True

    def event(self):
        e = self.torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def mark(self, name, start):
        if start:
            if name == 'denoising':
                self.current_step = None
            self.pending[name] = self.event()
            self.pending_steps[name] = self.current_step
        else:
            self.events[name].append((self.pending.pop(name), self.event()))
            self.event_steps[name].append(self.pending_steps.pop(name))
            if name == 'denoising':
                self.current_step = None

    def wrap(self, obj, attr, name, step_boundary=False):
        original = getattr(obj, attr)
        had_local, local = attr in vars(obj), vars(obj).get(attr)

        def timed(*args, **kwargs):
            if step_boundary:
                self.current_step = len(self.step_starts)
                self.step_starts.append(self.event())
            self.mark(name, True)
            wall_start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.wall_ms[name].append((time.perf_counter()-wall_start)*1000)
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
                # Code heads may be skipped or called repeatedly: attribute by the
                # observed video_pre iteration, never infer their step from call count.
                if len(self.event_steps.get(name, [])) == len(pairs):
                    observed = self.event_steps[name][i]
                    step = observed if observed is not None else ''
                rows.append([name, i, step, layer, s.elapsed_time(e)])
        write_csv(directory / 'detail_calls.csv',
                  ['stage', 'call', 'step', 'layer', 'ms'], rows)
        write_csv(directory / 'sections.csv',
                  ['stage', 'calls', 'cuda_event_total_ms', 'cpu_wall_total_ms', 'scope'],
                  [[name, len(pairs), totals[name],
                    sum(self.wall_ms[name]) if name in self.wall_ms else '',
                    'inclusive; nested stages are not additive; CPU wall may be asynchronous']
                   for name, pairs in self.events.items()])
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
        # Each expert's pre/post blocks plus head/embedding and scheduler; no nested components.
        # The joint attention is one shared operation and must not be assigned to all experts.
        names = ('video', 'code', 'action', 'shared')
        if self.internal_experts_available:
            self.diffusion_expert_ms = {
                n: sum(r[4] for r in rows if r[2] != '' and r[0].startswith(n+'.'))
                for n in names}
            write_csv(directory / 'per_step_experts.csv', ['step', *[n+'_ms' for n in names]],
                      [[step, *[sum(r[4] for r in rows if r[2] == step and r[0].startswith(n+'.'))
                                 for n in names]] for step in range(nis)])
        return totals


def write_csv(path, header, rows):
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def install(timer, policy, eager):
    timer.internal_experts_available = eager
    model = policy.model
    timer.wrap(policy, '_infer_action_chunk', 'policy.total')
    timer.wrap(policy, '_build_obs_video', 'obs.preprocess')
    timer.wrap(policy, '_normalize_state', 'obs.state_normalize')
    timer.wrap(policy, '_denormalize_action', 'output.denormalize')
    timer.wrap(policy.text, 'encode', 'obs.text_encode_warm')
    timer.wrap(model, '_encode_video_latents', 'obs.vae')
    timer.wrap(model, '_vit_prefix', 'obs.qwen')
    timer.wrap(model.mot, 'forward', 'mot.total')
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
