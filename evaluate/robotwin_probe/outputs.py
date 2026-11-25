"""Action tensors captured in memory; disk export happens after inference.

This is deliberately separate from CUDA timing and profiler passes.
"""
import csv
import json


class ActionOutputs:
    def __init__(self, policy):
        self.policy = policy
        self.values = {}
        self.dtypes = {}
        self.restores = []

    def keep(self, name, tensor):
        self.dtypes[name] = str(tensor.dtype)
        # Clone on the original device: never synchronise each diffusion step.
        self.values.setdefault(name, []).append(tensor.detach().clone())

    def wrap(self, obj, attr, callback):
        original = getattr(obj, attr)
        had_local, local = attr in vars(obj), vars(obj).get(attr)

        def wrapped(*args, **kwargs):
            return callback(original, *args, **kwargs)

        setattr(obj, attr, wrapped)
        self.restores.append((obj, attr, had_local, local))

    def install(self):
        def expert(original, *args, **kwargs):
            result = original(*args, **kwargs)
            self.keep('expert_prediction', result)
            return result

        def scheduler(original, model_output, delta, sample):
            self.keep('scheduler_prediction', model_output)
            self.keep('latent_before', sample)
            self.keep('delta', delta)
            result = original(model_output, delta, sample)
            self.keep('latent_after', result)
            return result

        def denormalize(original, action):
            self.keep('normalized_final', action)
            return original(action)

        self.wrap(self.policy.model.action_expert, 'post_dit', expert)
        self.wrap(self.policy.model.action_scheduler, 'step', scheduler)
        self.wrap(self.policy, '_denormalize_action', denormalize)
        return self

    def restore(self):
        for obj, attr, had_local, local in reversed(self.restores):
            if had_local:
                setattr(obj, attr, local)
            else:
                delattr(obj, attr)
        self.restores.clear()

    def export(self, stem, chunk):
        import numpy as np
        import torch
        arrays = {k: torch.stack(v).float().cpu().numpy() for k, v in self.values.items()}
        required = {'expert_prediction', 'scheduler_prediction', 'latent_before',
                    'latent_after', 'delta', 'normalized_final'}
        if set(arrays) != required:
            raise RuntimeError(f'Incomplete action capture: {set(arrays)}')
        nis = self.policy.num_inference_steps
        if len(arrays['latent_after']) != nis:
            raise RuntimeError(f'Expected {nis} action scheduler calls')
        if len(arrays['expert_prediction']) != nis:
            raise RuntimeError('Expected one action expert output per diffusion step; check CFG')
        if not all(np.isfinite(x).all() for x in arrays.values()):
            raise RuntimeError('Non-finite diffusion action capture')
        np.savez_compressed(str(stem) + '.diffusion.npz', **arrays)
        with open(str(stem) + '.actions.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['chunk_step', 'dim', 'absolute_qpos', 'next_minus_current'])
            for t, row in enumerate(chunk):
                for d, value in enumerate(row):
                    w.writerow([t, d, float(value),
                                float(chunk[t+1, d]-value) if t+1 < len(chunk) else ''])
        schema = dict(
            arrays={k: dict(shape=list(v.shape), stored_dtype=str(v.dtype),
                            original_dtype=self.dtypes[k]) for k, v in arrays.items()},
            absolute_qpos_shape=list(chunk.shape),
            semantics={
                'expert_prediction': 'Flow-matching velocity head output; NOT executable qpos.',
                'scheduler_prediction': 'Velocity passed to scheduler (may include CFG).',
                'latent_before': 'Normalized noisy action before this diffusion step.',
                'latent_after': 'Normalized action after sample + prediction * delta.',
                'normalized_final': 'Input to original policy denormalization, one call axis.',
                'actions_csv': 'Original denormalized absolute qpos targets; adjacent rows in time '
                               'are action-command ticks, not camera frames or diffusion steps.'},
            capture_scope='Real rollout; tensor clones and disk I/O perturb latency. '
                          'Use separate replay timing for performance.')
        with open(str(stem) + '.schema.json', 'w') as f:
            json.dump(schema, f, indent=2)
        print(f'RW_ACTION_OUTPUT shape={list(chunk.shape)} diffusion_steps={nis} '
              f'first_target={chunk[0].tolist()} file={stem}.actions.csv', flush=True)
        self.values.clear()
