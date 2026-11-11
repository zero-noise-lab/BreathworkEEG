"""Bounded change relative to a baseline, as used by the paper's MI plots."""
import numpy as np


def compute_modulation_index(values, baseline):
    """Return (values - baseline) / (abs(values) + abs(baseline)).
    """
    if np.iscomplexobj(values) or np.iscomplexobj(baseline):
        raise ValueError("Values and baseline must be real-valued")
    values, baseline = np.broadcast_arrays(
        np.asarray(values, dtype=np.float64), np.asarray(baseline, dtype=np.float64))
    if not values.size or np.isinf(values).any() or np.isinf(baseline).any():
        raise ValueError("Provide nonempty finite values/baselines, or NaN for missing entries")
    # Scaling preserves the ratio without overflowing large finite inputs.
    scale = np.maximum(np.abs(values), np.abs(baseline))
    with np.errstate(divide="ignore", invalid="ignore"):
        event, reference = values / scale, baseline / scale
        return (event - reference) / (np.abs(event) + np.abs(reference))
