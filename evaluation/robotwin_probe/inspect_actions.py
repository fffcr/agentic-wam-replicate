#!/usr/bin/env python3
"""Print actual action rows, diffusion shapes, and the first consecutive replan pair."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    a = p.parse_args()
    np.set_printoptions(precision=6, suppress=True, linewidth=180)
    indexes = sorted(a.run.glob('capture/*/index.jsonl'))
    if not indexes:
        raise SystemExit('No capture/*/index.jsonl found in this run')
    for index in indexes:
        print('\nSHARD:', index.parent)
        records = [json.loads(line) for line in index.read_text().splitlines() if line]
        previous = None
        for rec in records:
            path = index.parent / rec['file']
            with np.load(path, allow_pickle=False) as data:
                chunk = data['absolute_qpos'].copy()
                state = data['state'].copy()
            print('REPLAN:', rec['file'], 'episode:', rec['episode_id'], 'tick:', rec['start_step'])
            print('SHAPE:', chunk.shape, 'DTYPE:', chunk.dtype)
            print('OBSERVED_STATE:', state)
            print('TARGET[0]:', chunk[0])
            if len(chunk) > 1:
                print('TARGET[1]:', chunk[1])
                print('TARGET[1]-TARGET[0]:', chunk[1]-chunk[0])
            with np.load(path.with_suffix('.diffusion.npz'), allow_pickle=False) as data:
                print('DIFFUSION_SHAPES:', {k: list(data[k].shape) for k in data.files})
                print('EXPERT_VELOCITY[step0,batch0,action0]:', data['expert_prediction'][0, 0, 0])
            if previous is not None and previous[0]['episode_id'] == rec['episode_id']:
                old_rec, old, old_state = previous
                advance = rec['start_step']-old_rec['start_step']
                n = min(max(0, len(old)-advance), len(chunk))
                print('OBSERVED_STATE_CHANGE:', state-old_state)
                print('ACTUAL_EXECUTED_TICKS:', advance, 'OVERLAP_STEPS:', n)
                if n:
                    print(f'OLD_TARGET[{advance}:{advance+n}]:\n', old[advance:advance+n])
                    print(f'NEW_TARGET[:{n}]:\n', chunk[:n])
                    print('NEW_MINUS_OLD:\n', chunk[:n]-old[advance:advance+n])
            previous = rec, chunk, state


if __name__ == '__main__':
    main()
