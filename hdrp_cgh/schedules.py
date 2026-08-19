from __future__ import annotations


def linear_temperature(step: int, total_steps: int, start: float = 0.05, end: float = 0.01) -> float:
    """Linear soft-min temperature annealing for HDRP training."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if start <= 0 or end <= 0:
        raise ValueError("temperatures must be positive")
    t = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    return start + t * (end - start)
