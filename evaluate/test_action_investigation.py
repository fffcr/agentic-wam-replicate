import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy'))
from analyze_expert_bench import analyze_detail, analyze_trace, union_duration
from analyze_replan_overlap import aligned_overlap, analyze
from common.replan_record import ReplanRecorder


class AnalysisTests(unittest.TestCase):
    def test_union_does_not_double_count_concurrent_kernels(self):
        self.assertEqual(union_duration([(0, 10), (2, 4), (8, 12), (20, 21)]), 13)

    def test_detail_action_breakdown_excludes_other_experts(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'detail_calls.csv'
            with path.open('w') as f:
                w = csv.writer(f)
                w.writerow(['stage', 'call', 'step', 'layer', 'ms'])
                for stage, ms in [('action.pre_attn', 2), ('action.post_attn', 5),
                                  ('action.post_attn', 3), ('video.pre_attn', 20)]:
                    w.writerow([stage, 0, 0, 0, ms])
            result = analyze_detail(path)
            self.assertEqual(result['action_total_ms'], 10)
            self.assertEqual(result['action_stages']['action.post_attn']['action_share'], .8)

    def test_trace_correlation_nested_action_range_and_overlap(self):
        def event(name, cat, ts, dur, args=None, tid=1):
            return dict(name=name, cat=cat, ph='X', ts=ts, dur=dur,
                        args=args or {}, pid=1, tid=tid)
        events = [event('BENCH_ROUND', 'user_annotation', 0, 100),
                  event('expert.action.post_attn', 'user_annotation', 1, 15),
                  event('expert.action.block_0.ffn', 'user_annotation', 2, 10),
                  event('aten::linear', 'cpu_op', 3, 2, {'External id': 123}),
                  event('cudaLaunchKernel', 'cuda_runtime', 4, 1,
                        {'External id': 123, 'correlation': 7}),
                  event('gemm', 'kernel', 20, 10, {'correlation': 7}, tid=7),
                  event('other', 'kernel', 25, 10, {}, tid=8)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'trace.json'
            path.write_text(json.dumps({'traceEvents': events}))
            result = analyze_trace(path)
        self.assertAlmostEqual(result['kernel_union_ms'], .015)
        self.assertEqual(result['unattributed_kernel_count'], 1)
        self.assertEqual(result['attributed_groups']['expert.action.ffn']['kernel_count'], 1)


class ReplanTests(unittest.TestCase):
    def test_alignment_uses_executed_steps_not_same_indices(self):
        old = np.arange(32.)[:, None]
        new = np.arange(24., 56.)[:, None]
        x, y = aligned_overlap(old, new, 24)
        self.assertEqual(x.shape, (8, 1))
        np.testing.assert_array_equal(x, y)
        self.assertFalse(np.array_equal(old[:8], new[:8]))

    def test_different_horizons_no_overlap_and_invalid_advance(self):
        self.assertEqual(len(aligned_overlap(np.zeros((32, 2)), np.zeros((4, 2)), 24)[0]), 4)
        self.assertEqual(len(aligned_overlap(np.zeros((32, 2)), np.zeros((4, 2)), 32)[0]), 0)
        with self.assertRaises(ValueError):
            aligned_overlap(np.zeros((32, 2)), np.zeros((4, 2)), 0)

    def test_recording_roundtrip_separates_episodes(self):
        images = {c: np.zeros((2, 3, 3), dtype=np.uint8)
                  for c in ('head_camera', 'left_camera', 'right_camera')}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'recording'
            recorder = ReplanRecorder(path)
            for episode, tick in [('a', 0), ('a', 24), ('b', 0)]:
                recorder.record(images=images, state=np.zeros(2),
                                chunk=np.repeat(np.arange(tick, tick+32.)[:, None], 2, axis=1),
                                instruction='move', episode_id=episode, start_step=tick)
            result = analyze(path, 0)
            self.assertEqual(len(result['pairs']), 1)
            self.assertEqual(result['pairs'][0]['overlap_steps'], 8)
            self.assertEqual(result['pairs'][0]['exact_equal_fraction'], 1)
            with self.assertRaises(ValueError):
                recorder.record(images=images, state=np.zeros(2), chunk=np.ones((32, 2)),
                                instruction='move', episode_id='b', start_step=0)


if __name__ == '__main__':
    unittest.main()
