import types
import unittest

from runner import compile_check
from stages import DISJOINT, ROUND, StageTimer, summarize


class CoarseTests(unittest.TestCase):
    def test_percentile_uses_round_totals_and_residual_does_not_double_count(self):
        rounds = [{DISJOINT[0]: 2, 'Denoising': 90, 'Other device timeline': 8,
                   ROUND: 100, 'ROUND wall': 101},
                  {DISJOINT[0]: 4, 'Denoising': 180, 'Other device timeline': 16,
                   ROUND: 200, 'ROUND wall': 202}]
        counts = [{DISJOINT[0]: 1, 'Denoising': 1, ROUND: 1}]*2
        rows = {r['stage']: r for r in summarize(rounds, counts)}
        self.assertEqual(rows['Denoising']['mean_ms'], 135)
        self.assertEqual(rows['Denoising']['p50_ms'], 135)
        self.assertEqual(rows['Denoising']['p95_ms'], 175.5)
        self.assertEqual(rows['Denoising']['percent'], 90)
        self.assertEqual(rows['Other device timeline']['calls_per_round'], 0)
        self.assertIsNone(rows['ROUND wall']['percent'])
        self.assertEqual(sum(rows[n]['percent'] for n in
                             (DISJOINT[0], 'Denoising', 'Other device timeline')), 100)

    def test_instrumentation_leaves_mot_untouched_and_restores_callback(self):
        class Event:
            count = 0

            def __init__(self, **kwargs):
                pass

            def record(self):
                self.n = Event.count
                Event.count += 1

            def elapsed_time(self, other):
                return other.n-self.n

        sentinel = lambda: None
        model = types.SimpleNamespace(_encode_video_latents=sentinel, _vit_prefix=sentinel,
                                      mot=types.SimpleNamespace(forward=sentinel))
        policy = types.SimpleNamespace(_build_obs_video=sentinel,
                                       text=types.SimpleNamespace(encode=sentinel), model=model)
        timer = StageTimer(types.SimpleNamespace(cuda=types.SimpleNamespace(Event=Event)))
        try:
            timer.install(policy)
            self.assertIs(model.mot.forward, sentinel)
            timer.mark(ROUND, True)
            policy._build_obs_video()
            policy.text.encode()
            model._encode_video_latents()
            model._vit_prefix()
            model._bench_stage_timer('denoising', True)
            model._bench_stage_timer('denoising', False)
            timer.mark(ROUND, False)
            totals = timer.totals()
            self.assertEqual(sum(totals.get(n, 0) for n in DISJOINT) +
                             totals['Other device timeline'], totals[ROUND])
        finally:
            timer.restore()
        self.assertFalse(hasattr(model, '_bench_stage_timer'))
        self.assertIs(policy._build_obs_video, sentinel)
        self.assertIs(model.mot.forward, sentinel)

    def test_new_graphs_or_missing_compilation_are_not_steady_results(self):
        self.assertFalse(compile_check('compile', 0, 0)['valid_steady_measurement'])
        self.assertFalse(compile_check('compile', 2, 1)['valid_steady_measurement'])
        self.assertTrue(compile_check('compile', 2, 0)['valid_steady_measurement'])


if __name__ == '__main__':
    unittest.main()
