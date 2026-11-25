import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from analyze_expert_bench import analyze_trace, main, select_round


class TraceTests(unittest.TestCase):
    def test_cpu_round_selected_when_gpu_annotation_duplicates_name(self):
        events = [
            dict(name='BENCH_ROUND', cat='user_annotation', ph='X', ts=0, dur=100,
                 pid=1, tid=1),
            dict(name='BENCH_ROUND', cat='gpu_user_annotation', ph='X', ts=10, dur=80,
                 pid=0, tid=7),
            dict(name='kernel', cat='kernel', ph='X', ts=20, dur=50, args={}),
        ]
        self.assertEqual(select_round(events), (0, 100))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.json'
            path.write_text(json.dumps({'traceEvents': events}))
            report = analyze_trace(path)
            self.assertEqual(report['kernel_count'], 1)
            self.assertAlmostEqual(report['kernel_union_ms'], 0.05)

    def test_ambiguous_cpu_rounds_are_not_silently_merged(self):
        event = dict(name='BENCH_ROUND', cat='user_annotation', ph='X', ts=0, dur=100)
        with self.assertRaisesRegex(ValueError, 'cpu_ranges=2'):
            select_round([event, dict(event, ts=200)])

    def test_missing_marker_preserves_detail_results_and_reports_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'detail_calls.csv').write_text('stage,ms\naction.post_attn,12.5\n')
            (root/'trace.json').write_text(json.dumps({'traceEvents': []}))
            with patch.object(sys, 'argv', ['analyzer', str(root)]), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                main()
            result = json.loads((root/'analysis.json').read_text())
            self.assertEqual(result['detail']['action_total_ms'], 12.5)
            self.assertEqual(result['analysis_status'], 'partial')
            self.assertEqual(result['trace_status'], 'failed')
            self.assertIn('cpu_ranges=0', result['trace_error'])


if __name__ == '__main__':
    unittest.main()
