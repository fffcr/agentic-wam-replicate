"""External, process-local capture wrappers. Never writes any author source file."""
import copy
import csv
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def snapshot(policy):
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                policy={k: copy.deepcopy(getattr(policy, k)) for k in
                        ('code_cache', '_exec_slices', 'step_count', 'episode_count') if hasattr(policy, k)},
                model={k: copy.deepcopy(getattr(policy.model, k)) for k in
                       ('_mem_code_history', '_pinned_code_ids') if hasattr(policy.model, k)})


def restore(policy, saved):
    import numpy as np
    import torch
    policy.reset()
    for k, v in saved['policy'].items():
        setattr(policy, k, copy.deepcopy(v))
    for k, v in saved['model'].items():
        setattr(policy.model, k, copy.deepcopy(v))
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    torch.set_rng_state(saved['torch'].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda']])


def save_context(saved, path):
    import torch
    torch.save(saved, path)


class Capture:
    def __init__(self, policy, usr_args):
        from recording import ReplanRecorder
        if policy.hybrid is not None:
            raise ValueError('Capture supports pure WAM, not hybrid skill execution')
        if policy.model.action_expert.action_dim != 14:
            raise ValueError('Expected released RoboTwin 14-D qpos policy')
        self.policy = policy
        self.episode = None
        self.tick = self.episode_records = 0
        self.limit = int(os.environ.get('RW_PROBE_MAX_REPLANS', '8'))
        if self.limit < 1:
            raise ValueError('RW_PROBE_MAX_REPLANS must be positive')
        self.directory = Path(os.environ['RW_PROBE_CAPTURE']) / f'shard_{os.getpid()}_{time.time_ns()}'
        self.recorder = ReplanRecorder(self.directory)
        env = {k: v for k, v in os.environ.items()
               if (k.startswith('AGENTICWAM_') or k in ('TASK', 'CODE_VISION'))
               and not any(s in k for s in ('SECRET', 'TOKEN', 'KEY'))}
        (self.directory / 'run_info.json').write_text(json.dumps(
            dict(usr_args=usr_args, env=env,
                 tick_contract='completed take_action(qpos) command calls, not physics substeps'), indent=2))
        print(f'RW_PROBE_CAPTURE_READY={self.directory}', flush=True)

    def sync_episode(self):
        if self.episode != self.policy.episode_count:
            self.episode = self.policy.episode_count
            self.tick = self.episode_records = 0
            self.active_replan = None

    def before(self):
        self.sync_episode()
        self.active_replan = None
        return snapshot(self.policy) if self.episode_records < self.limit else None

    def after(self, observation, instruction, chunk, saved, outputs=None):
        if saved is None:
            return
        index = self.recorder.count
        self.recorder.record(
            images={c: observation['observation'][c]['rgb'] for c in
                    ('head_camera', 'left_camera', 'right_camera')},
            state=observation['joint_action']['vector'], chunk=chunk,
            instruction=instruction, episode_id=self.episode, start_step=self.tick,
            code=self.policy._last_code)
        save_context(saved, self.directory / f'replan_{index:06d}.context.pt')
        if outputs is not None:
            outputs.export(self.directory / f'replan_{index:06d}', chunk)
        self.active_replan = index
        self.episode_records += 1

    def executed(self, target):
        """Targets passed to completed take_action calls, including chunk association."""
        import numpy as np
        if getattr(self, 'active_replan', None) is None:
            return
        path = self.directory / 'executed_actions.csv'
        exists = path.exists()
        with path.open('a', newline='') as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(['episode', 'action_tick', 'replan', 'dim', 'target'])
            for d, value in enumerate(np.asarray(target).reshape(-1)):
                w.writerow([self.episode, self.tick, self.active_replan, d, float(value)])


def attach(policy, capture):
    original_infer, original_step = policy._infer_action_chunk, policy.step

    def infer(observation, instruction):
        saved = capture.before()
        outputs = None
        if saved is not None and os.environ.get('RW_PROBE_CAPTURE_DIFFUSION', '1') == '1':
            from outputs import ActionOutputs
            outputs = ActionOutputs(policy).install()
        try:
            chunk = original_infer(observation, instruction)
        finally:
            if outputs is not None:
                outputs.restore()
        capture.after(observation, instruction, chunk, saved, outputs)
        return chunk

    def step(task_env, observation):
        capture.sync_episode()
        original = task_env.take_action
        had_local, local = 'take_action' in vars(task_env), vars(task_env).get('take_action')

        def take_action(*args, **kwargs):
            target = copy.deepcopy(args[0] if args else kwargs.get('action'))
            result = original(*args, **kwargs)
            if hasattr(capture, 'executed'):
                capture.executed(target)
            capture.tick += 1
            return result

        task_env.take_action = take_action
        try:
            return original_step(task_env, observation)
        finally:
            if had_local:
                task_env.take_action = local
            else:
                delattr(task_env, 'take_action')

    policy._infer_action_chunk, policy.step = infer, step


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.rsplit('.', 1)[-1] != 'deploy_policy':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        expected = ROOT / 'robotwin/deploy/deploy_policy.py'
        if spec is None or not spec.origin or Path(spec.origin).resolve() != expected.resolve():
            return None
        original_loader = spec.loader

        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                create = getattr(original_loader, 'create_module', None)
                return create(spec) if create else None

            def exec_module(self, module):
                original_loader.exec_module(module)
                get_model = module.get_model

                def captured_get_model(usr_args):
                    policy = get_model(usr_args)
                    attach(policy, Capture(policy, usr_args))
                    return policy

                module.get_model = captured_get_model

        spec.loader = Loader()
        return spec


def enable():
    if os.environ.get('RW_PROBE_CAPTURE'):
        sys.meta_path.insert(0, Finder())
