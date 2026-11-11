"""Median-threshold LZ76; no dataset, channel-count, or file-format assumptions.

Lempel & Ziv (1976), doi:10.1109/TIT.1976.1055501. The MIT-licensed
NeuroKit2-derived phrase counter is credited in README.md and LICENSE.
"""
import math
from concurrent.futures import ThreadPoolExecutor

import mne
import numpy as np

from ._inputs import epochs_array, frequency_bands
from .fast_lz76 import count


def compute_lzc(epochs, sfreq, bands=None, n_jobs=1):
    """Return {band: values}, with values shaped (epochs, channels).

    Apply a fourth-order zero-phase Butterworth bandpass, threshold each trace
    at its own median, then return c(n)*log2(n)/n. Input may be voltage or
    another real signal domain chosen by the caller; the EEG example uses
    voltage. Results are not shuffle-normalized and need not be bounded by 1.
    """
    data = epochs_array(epochs, sfreq, n_jobs)
    selected = frequency_bands(bands, sfreq)
    scale = math.log2(data.shape[-1]) / data.shape[-1]

    def epoch_values(binary):
        return np.asarray([count(trace) * scale for trace in binary])

    result = {}
    for name, (low, high) in selected.items():
        filtered = mne.filter.filter_data(
            data, sfreq, low, high, method="iir", phase="zero", copy=True,
            iir_params={"order": 4, "ftype": "butter", "output": "sos"},
            n_jobs=1, verbose=False,
        )
        binary = filtered > np.median(filtered, axis=-1, keepdims=True)
        with ThreadPoolExecutor(max_workers=min(int(n_jobs), len(data))) as pool:
            result[name] = np.vstack(list(pool.map(epoch_values, binary)))
    return result
