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


class HookTests(unittest.TestCase):
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

    def test_action_shape_and_nonfinite_checks(self):
        x = np.ones((32, 14))
        self.assertTrue(action_diff(x, x)['allclose'])
        self.assertFalse(action_diff(x, x[:8])['same_shape'])
        self.assertFalse(action_diff(x, x*np.nan)['allclose'])


if __name__ == '__main__':
    unittest.main()
