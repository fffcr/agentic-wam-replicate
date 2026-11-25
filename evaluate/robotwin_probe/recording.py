"""Record actual consecutive replans at the caller's executed-action tick.

Use after rt.infer, with start_step counted by the real controller, not wall time
or a guessed replan interval. Disk recording is outside the model timing.
"""
import json
from pathlib import Path
import time

import numpy as np


class ReplanRecorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.count = 0
        self.last_steps = {}

    def record(self, *, images, state, chunk, instruction, episode_id,
               start_step, code=None, observation_stamp_ns=None):
        """chunk must be ABSOLUTE qpos [T,D], on the controller's action-tick grid.

        start_step is the tick of chunk[0]. Only count actual executed action ticks.
        Do not pass expected/planned ticks. Observation timestamps are optional.
        """
        CAMERAS = ('head_camera', 'left_camera', 'right_camera')
        x = np.asarray(chunk)
        state = np.asarray(state).reshape(-1)
        if x.ndim != 2 or len(x) < 1 or x.shape[1] != len(state):
            raise ValueError('Expected absolute qpos chunk [T,D] and state [D]')
        if not np.isfinite(x).all() or not np.isfinite(state).all():
            raise ValueError('Non-finite state/action')
        if not isinstance(start_step, (int, np.integer)) or start_step < 0:
            raise ValueError('start_step must be a nonnegative actual action-tick index')
        episode = str(episode_id)
        if episode in self.last_steps and start_step <= self.last_steps[episode]:
            raise ValueError('Replan start ticks must strictly increase within an episode')
        arrays = {'absolute_qpos': x, 'state': state}
        for c in CAMERAS:
            image = np.asarray(images[c])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f'{c}: expected HxWx3 RGB uint8')
            arrays[c] = image
        name = f'replan_{self.count:06d}.npz'
        meta = dict(file=name, episode_id=episode, start_step=int(start_step),
                    action_type='qpos_abs', horizon=len(x), action_dim=x.shape[1],
                    instruction=instruction, code=code, observation_stamp_ns=observation_stamp_ns,
                    saved_stamp_ns=time.time_ns(), tick_source='actual_controller')
        np.savez_compressed(self.directory / name, **arrays)
        with (self.directory / 'index.jsonl').open('a') as f:
            f.write(json.dumps(meta) + '\n')
        self.last_steps[episode] = int(start_step)
        self.count += 1
