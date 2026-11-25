"""CPU checks for action interpretation and transparent timing wrappers."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    'bench_expert_latency', Path(__file__).with_name('bench_expert_latency.py'))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class ActionTests(unittest.TestCase):
    def test_adjacent_difference_is_not_offset_from_input(self):
        actions = np.array([[11., 22.], [13., 21.], [12., 25.]])
        stats = bench.describe_action(actions, 2)
        self.assertEqual(stats['shape'], [3, 2])
        np.testing.assert_allclose(stats['adjacent_mean_abs_per_dim'], [1.5, 2.5])
        np.testing.assert_allclose(stats['adjacent_max_abs_per_dim'], [2., 4.])
        self.assertFalse(np.array_equal(np.diff(actions, axis=0),
                                        (actions - [10., 20.])[:-1]))

    def test_bad_shape_and_nonfinite_are_rejected(self):
        for values in (np.zeros((2, 3)), np.zeros((0, 2)), np.zeros(2),
                       np.array([[np.nan, 0.]]), np.array([[np.inf, 0.]])):
            with self.assertRaises(ValueError):
                bench.describe_action(values, 2)

    def test_single_frame_has_no_adjacent_delta(self):
        self.assertIsNone(bench.describe_action(np.ones((1, 2)), 2)[
            'adjacent_max_abs_per_dim'])

    def test_comparison_tolerance_and_shape(self):
        x = np.array([[1., 2.]])
        self.assertTrue(bench.compare_actions(x, x + 1e-5, 1e-4, 0)['allclose'])
        self.assertFalse(bench.compare_actions(x, x + 1., 1e-4, 0)['allclose'])
        self.assertFalse(bench.compare_actions(x, np.zeros((2, 2)), 0, 0)['same_shape'])


class WrapperTests(unittest.TestCase):
    def test_wrapper_preserves_result_and_restores_inherited_method(self):
        class Event:
            def __init__(self, **kwargs):
                pass

            def record(self):
                pass

        class FakeTorch:
            class cuda:
                pass

        FakeTorch.cuda.Event = Event

        class Example:
            def call(self, x):
                return x + 1

        obj = Example()
        timer = bench.DetailTimer(FakeTorch)
        timer.wrap(obj, 'call', 'test')
        self.assertEqual(obj.call(3), 4)
        self.assertEqual(timer.counts['test'], 1)
        self.assertEqual(len(timer.events), 1)
        timer.restore()
        self.assertNotIn('call', vars(obj))
        self.assertEqual(obj.call(4), 5)


if __name__ == '__main__':
    unittest.main()
