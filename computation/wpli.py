"""Multitaper wPLI using MNE-Connectivity.

Vinck et al. (2011), doi:10.1016/j.neuroimage.2011.01.055.
"""
import numpy as np
from mne_connectivity import spectral_connectivity_epochs

from ._inputs import epochs_array, frequency_bands


def compute_wpli(epochs, sfreq, bands=None, channel_names=None,
                 mt_bandwidth=1.0, n_jobs=1):
    """Return {band: symmetric channels x channels wPLI matrix}.

    Input: epochs x channels x samples. CSD, referencing, and artifact handling
    are caller choices; the supplied example uses CSD. Ordinary wPLI is
    aggregated over epochs and multitapers (not debiased squared wPLI).
    Band upper endpoints are exclusive except the largest upper endpoint.
    """
    data = epochs_array(epochs, sfreq, n_jobs)
    selected = frequency_bands(bands, sfreq)
    if min(data.shape[:2]) < 2:
        raise ValueError("wPLI requires at least two epochs and two channels")
    if channel_names is not None and (len(channel_names) != data.shape[1]
                                     or len(set(channel_names)) != len(channel_names)):
        raise ValueError("Provide one unique name per channel")
    if not np.isfinite(mt_bandwidth) or mt_bandwidth <= 0:
        raise ValueError("mt_bandwidth must be positive and finite")
    low = min(limits[0] for limits in selected.values())
    high = max(limits[1] for limits in selected.values())
    estimate = spectral_connectivity_epochs(
        data, names=channel_names, method="wpli", mode="multitaper", sfreq=sfreq,
        fmin=low, fmax=high, faverage=False, mt_bandwidth=mt_bandwidth,
        mt_adaptive=False, mt_low_bias=True, n_jobs=int(n_jobs), verbose=False,
    )
    dense = np.asarray(estimate.get_data(output="dense"), dtype=float)
    # MNE's default all-pairs output stores the lower triangle.
    lower = np.tril_indices(data.shape[1], -1)
    spectrum = np.zeros_like(dense)
    spectrum[lower] = dense[lower]
    spectrum[(lower[1], lower[0])] = dense[lower]
    if not np.isfinite(spectrum).all():
        raise ValueError("Undefined wPLI: check signal variance and epoch count")
    frequencies = np.asarray(estimate.freqs)
    result = {}
    for name, (start, stop) in selected.items():
        mask = (frequencies >= start) & (
            (frequencies <= stop) if stop == high else (frequencies < stop))
        if not mask.any():
            raise ValueError(f"No Fourier bins in band {name!r}")
        matrix = np.clip(spectrum[:, :, mask].mean(axis=-1), 0, 1)
        np.fill_diagonal(matrix, 0)
        result[name] = matrix
    return result
