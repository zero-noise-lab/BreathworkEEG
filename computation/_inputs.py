"""Shared array checks and optional EEG band defaults. No file loading."""
import numpy as np

DEFAULT_BANDS = {
    "Broadband": (1.0, 45.0), "Delta": (1.0, 4.0), "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0), "Beta": (13.0, 30.0), "Gamma": (30.0, 45.0),
}


def epochs_array(epochs, sfreq, n_jobs=1):
    if np.iscomplexobj(epochs):
        raise ValueError("Epochs must be real-valued")
    data = np.asarray(epochs, dtype=np.float64)
    if data.ndim != 3 or min(data.shape[:2]) < 1 or data.shape[-1] < 5:
        raise ValueError("Expected a nonempty epochs x channels x samples array")
    if not np.isfinite(data).all():
        raise ValueError("Epochs must contain only finite values")
    if not np.isscalar(sfreq) or not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be positive and finite")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    return data


def frequency_bands(bands, sfreq):
    selected = dict(DEFAULT_BANDS if bands is None else bands)
    if not selected:
        raise ValueError("Provide at least one frequency band")
    for name, limits in selected.items():
        bounds = np.asarray(limits, dtype=float)
        if (not isinstance(name, str) or bounds.shape != (2,)
                or not np.isfinite(bounds).all()
                or not 0 < bounds[0] < bounds[1] < sfreq / 2):
            raise ValueError(f"Band {name!r} must satisfy 0 < low < high < sfreq/2")
    return selected
