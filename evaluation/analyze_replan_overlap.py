#!/usr/bin/env python3
"""Compare overlapping absolute action targets at aligned executed-action ticks."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def aligned_overlap(old, new, advance):
    old, new = np.asarray(old), np.asarray(new)
    if old.ndim != 2 or new.ndim != 2 or old.shape[1] != new.shape[1]:
        raise ValueError('Chunks must be [T,D] with equal D')
    if not isinstance(advance, (int, np.integer)) or advance <= 0:
        raise ValueError('Expected positive actual executed-tick advance')
    n = max(0, min(len(old)-advance, len(new)))
    return old[advance:advance+n], new[:n]


def analyze(directory, atol):
    directory = Path(directory)
    records = [json.loads(line) for line in (directory / 'index.jsonl').read_text().splitlines() if line]
    last, reports, rows = {}, [], []
    for rec in records:
        if rec['action_type'] != 'qpos_abs' or rec['tick_source'] != 'actual_controller':
            raise ValueError('Require absolute targets and actual action-tick alignment')
        # Only load files within the recorder directory.
        path = (directory / rec['file']).resolve()
        if path.parent != directory.resolve():
            raise ValueError('Invalid recording filename')
        with np.load(path, allow_pickle=False) as data:
            current = data['absolute_qpos'].copy()
        if not np.isfinite(current).all():
            raise ValueError('Non-finite action targets')
        episode = rec['episode_id']
        if episode in last:
            prev, old = last[episode]
            advance = rec['start_step'] - prev['start_step']
            x, y = aligned_overlap(old, current, advance)
            n = len(x)
            item = dict(old_file=prev['file'], new_file=rec['file'], episode_id=episode,
                        executed_ticks=advance, overlap_steps=n,
                        same_instruction=prev['instruction'] == rec['instruction'])
            if n:
                diff = y - x
                within = np.abs(diff) <= atol
                item.update(mae_per_dim=np.abs(diff).mean(axis=0).tolist(),
                            max_abs_per_dim=np.abs(diff).max(axis=0).tolist(),
                            fraction_within_tolerance_per_dim=within.mean(axis=0).tolist(),
                            fraction_entire_step_within_tolerance=float(within.all(axis=1).mean()),
                            exact_equal_fraction=float((x == y).mean()))
                for t in range(n):
                    for d in range(x.shape[1]):
                        rows.append([prev['file'], rec['file'], episode,
                                     rec['start_step']+t, d, x[t, d], y[t, d], diff[t, d]])
            reports.append(item)
        last[episode] = rec, current
    result = dict(atol=atol, pairs=reports,
                  warning='Overlap measures consistency of predicted targets, not hidden-feature '
                          'reuse or safety. Scalar tolerance has different physical meaning for '
                          'joints and grippers. Requires the same action-tick grid without dropped ticks.')
    (directory / 'overlap_report.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    with (directory / 'overlap_rows.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['old_file', 'new_file', 'episode', 'action_tick', 'dim', 'old', 'new', 'new_minus_old'])
        w.writerows(rows)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording', type=Path)
    p.add_argument('--atol', type=float, default=1e-3)
    a = p.parse_args()
    if a.atol < 0:
        p.error('atol must be nonnegative')
    print(json.dumps(analyze(a.recording, a.atol), indent=2))


if __name__ == '__main__':
    main()
