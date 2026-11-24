"""Choose how much of an AW-0 action chunk to execute before replanning.

The score is the per-action-step RMS variation of clean-action estimates from
the last few flow denoising iterations. It is a ranking signal, not a calibrated
probability of success. Keep the threshold explicit so experiments can sweep it.
"""

import math
from typing import Sequence


def select_commit_steps(
    scores: Sequence[float] | None,
    *,
    max_steps: int,
    min_steps: int,
    threshold: float,
) -> int:
    """Return a bounded prefix; fall back to the fixed horizon without a score."""
    if max_steps < 1 or min_steps < 1 or min_steps > max_steps:
        raise ValueError("require 1 <= min_steps <= max_steps")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if scores is None or len(scores) < max_steps:
        return max_steps
    for i in range(min_steps, max_steps):
        score = float(scores[i])
        if not math.isfinite(score) or score > threshold:
            return i
    return max_steps
