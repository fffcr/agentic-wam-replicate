import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hook
from runner import action_diff, source_hashes
from timing import Timer
from outputs import ActionOutputs


class HookTests(unittest.TestCase):
    def test_output_wrappers_restore_when_inference_raises(self):
        class Policy:
            def _infer_action_chunk(self, obs, instruction):
                raise RuntimeError('inference failed')

            def step(self, env, obs):
                pass

        capture = types.SimpleNamespace(before=lambda: {}, after=lambda *a: self.fail('unexpected export'))
        policy = Policy()
        with patch('outputs.ActionOutputs') as outputs:
            hook.attach(policy, capture)
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                policy._infer_action_chunk({}, 'move')
            outputs.return_value.install.return_value.restore.assert_called_once()

    def test_actual_executed_ticks_and_method_restoration(self):
        class Env:
            def take_action(self, target, **kwargs):
                return True

        class Policy:
            episode_count = 0
            step_count = 0

            def _infer_action_chunk(self, observation, instruction):
                return np.ones((32, 14))

            def step(self, env, obs):
                if self.step_count % 24 == 0:
                    self._infer_action_chunk(obs, 'move')
                env.take_action(np.zeros(14), action_type='qpos')
                self.step_count += 1

        class Capture:
            tick = 0

            def __init__(self):
                self.replans = []

            def sync_episode(self):
                pass

            def before(self):
                self.replans.append(self.tick)

            def after(self, *args):
                pass

        policy, env, capture = Policy(), Env(), Capture()
        hook.attach(policy, capture)
        for _ in range(25):
            policy.step(env, {})
        self.assertEqual(capture.replans, [0, 24])
        self.assertEqual(capture.tick, 25)
        self.assertNotIn('take_action', vars(env))
        with patch.object(env, 'take_action', side_effect=ValueError('failed action')):
            with self.assertRaises(ValueError):
                policy.step(env, {})
            self.assertEqual(capture.tick, 25)

    def test_import_hook_wraps_module_in_memory_without_source_write(self):
        before = source_hashes()
        module = types.ModuleType('policy.agenticwam_policy.deploy_policy')
        sentinel = object()

        class Loader:
            def exec_module(self, module):
                module.get_model = lambda args: sentinel

        spec = importlib.util.spec_from_loader(module.__name__, Loader(),
                                              origin=str(hook.ROOT / 'robotwin/deploy/deploy_policy.py'))
        with patch.object(hook.importlib.machinery.PathFinder, 'find_spec', return_value=spec), \
                patch.object(hook, 'Capture') as capture, patch.object(hook, 'attach') as attach:
            result = hook.Finder().find_spec(module.__name__)
            result.loader.exec_module(module)
            self.assertIs(module.get_model({'seed': 0}), sentinel)
            attach.assert_called_once_with(sentinel, capture.return_value)
        self.assertEqual(before, source_hashes())


class TimingTests(unittest.TestCase):
    def test_skipped_code_heads_keep_observed_diffusion_step(self):
        class Event:
            def __init__(self, n):
                self.n = n

            def elapsed_time(self, other):
                return other.n-self.n

        t = Timer(None)
        t.events['denoising'] = [(Event(0), Event(100))]
        t.events['code.post_dit'] = [(Event(2), Event(5)), (Event(70), Event(75))]
        t.event_steps['code.post_dit'] = [0, 2]
        t.step_starts = [Event(1), Event(30), Event(60)]
        with tempfile.TemporaryDirectory() as d:
            t.export(Path(d), layers=1, nis=3)
            with (Path(d) / 'per_step_experts.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([float(r['code_ms']) for r in rows], [3, 0, 5])

    def test_nested_timings_and_interval_partition(self):
        class Event:
            def __init__(self, n):
                self.n = n

            def elapsed_time(self, other):
                return other.n - self.n

        t = Timer(None)
        t.events['denoising'] = [(Event(0), Event(100))]
        t.events['action.post_attn'] = [(Event(1), Event(20)), (Event(60), Event(80))]
        t.events['nested.action.ffn'] = [(Event(5), Event(10)), (Event(65), Event(70))]
        t.step_starts = [Event(2), Event(55)]
        with tempfile.TemporaryDirectory() as d:
            totals = t.export(Path(d), layers=1, nis=2)
            self.assertEqual(totals['action.post_attn'], 39)
            self.assertEqual(totals['nested.action.ffn'], 10)
            with (Path(d) / 'denoise_intervals.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(sum(float(r['ms']) for r in rows), 100)
            with (Path(d) / 'per_step_experts.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(float(rows[0]['action_ms']), 19)
            self.assertEqual(float(rows[1]['action_ms']), 20)

    def test_action_shape_and_nonfinite_checks(self):
        x = np.ones((32, 14))
        self.assertTrue(action_diff(x, x)['allclose'])
        self.assertFalse(action_diff(x, x[:8])['same_shape'])
        self.assertFalse(action_diff(x, x*np.nan)['allclose'])


class OutputTests(unittest.TestCase):
    def test_diffusion_capture_values_shapes_and_restoration(self):
        # CPU-only tensor stand-in: verifies cloning and recording without a CUDA dependency.
        class Tensor:
            def __init__(self, data):
                self.data = np.array(data, dtype=np.float32)
                self.dtype = 'float32'

            def detach(self):
                return self

            def clone(self):
                return Tensor(self.data.copy())

            def float(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.data

        class Expert:
            def post_dit(self, tokens, pre):
                return tokens

        class Scheduler:
            @staticmethod
            def step(model_output, delta, sample):
                return Tensor(sample.data + model_output.data * delta.data)

        class Policy:
            num_inference_steps = 2
            model = types.SimpleNamespace(action_expert=Expert(), action_scheduler=Scheduler())

            def _denormalize_action(self, action):
                return action.data * 10

        policy = Policy()
        capture = ActionOutputs(policy).install()
        sample = Tensor(np.zeros((1, 3, 2)))
        source = Tensor(np.ones((1, 3, 2)))
        try:
            for _ in range(2):
                pred = policy.model.action_expert.post_dit(source, {})
                sample = policy.model.action_scheduler.step(pred, Tensor(-0.5), sample)
            chunk = policy._denormalize_action(Tensor(sample.data[0]))
            source.data[:] = 99  # retained expert outputs must be independent clones
        finally:
            capture.restore()
        for obj, attr in ((policy, '_denormalize_action'),
                          (policy.model.action_expert, 'post_dit'),
                          (policy.model.action_scheduler, 'step')):
            self.assertNotIn(attr, vars(obj))
        fake_torch = types.SimpleNamespace(stack=lambda v: Tensor(np.stack([x.data for x in v])))
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules, {'torch': fake_torch}):
            stem = Path(d) / 'replan_000000'
            capture.export(stem, chunk)
            with np.load(str(stem)+'.diffusion.npz') as data:
                self.assertEqual(data['expert_prediction'].shape, (2, 1, 3, 2))
                np.testing.assert_array_equal(data['expert_prediction'], 1)
                np.testing.assert_array_equal(data['latent_after'][-1], -1)
                np.testing.assert_allclose(data['latent_after'], data['latent_before'] +
                                           data['scheduler_prediction'] * data['delta'][:, None, None, None])
            with open(str(stem)+'.actions.csv') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 6)
            self.assertEqual(float(rows[0]['absolute_qpos']), -10)
            self.assertEqual(rows[-1]['next_minus_current'], '')


if __name__ == '__main__':
    unittest.main()
