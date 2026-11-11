"""Lagged scalar Gaussian-copula conditional information and signed NetDI.

Ince et al. (2017), doi:10.1002/hbm.23471; https://github.com/robince/gcmi.

"""
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm, rankdata

from ._inputs import epochs_array


def gaussianize(values):
    """Average ranks, transformed with norm.ppf((rank - 0.5) / n)."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 5 or not np.isfinite(values).all():
        raise ValueError("Provide a finite 1-D signal of length at least five")
    ranks = rankdata(values, method="average")
    return norm.ppf(np.clip((ranks - 0.5) / len(ranks), 1e-8, 1 - 1e-8))


def directed_bits(source, target, lag):
    """Low-level CMI for already Gaussianized signals; lag is in samples."""
    def correlation(a, b):
        a, b = a - a.mean(), b - b.mean()
        denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
        return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0

    y, z, x = target[lag:], target[:-lag], source[:-lag]
    yx, yz, xz = correlation(y, x), correlation(y, z), correlation(x, z)
    denominator = np.sqrt(max((1 - yz**2) * (1 - xz**2), np.finfo(float).tiny))
    partial = np.clip((yx - yz * xz) / denominator, -1 + 1e-12, 1 - 1e-12)
    return max(0.0, float(-0.5 * np.log2(1 - partial**2)))


def directed_information(source, target, lag_samples):
    """Return I(source_past; target_now | target_past) in bits for one lag.

    Inputs are equal-length 1-D signals. Rank-normalization precedes lagging.
    This is a one-past-sample lagged statistic, not time-summed directed
    information. Calling this with the arguments reversed gives reverse DI.
    """
    if np.iscomplexobj(source) or np.iscomplexobj(target):
        raise ValueError("Signals must be real-valued")
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if source.ndim != 1 or source.shape != target.shape:
        raise ValueError("source and target must be equal-length 1-D signals")
    if (isinstance(lag_samples, bool) or not isinstance(lag_samples, (int, np.integer))
            or not 1 <= lag_samples < len(source) - 3):
        raise ValueError("lag_samples must be an integer in [1, n_samples-4]")
    return directed_bits(gaussianize(source), gaussianize(target), int(lag_samples))


def compute_directed_information(epochs, sfreq, lags_ms, rois=None,
                                 channel_names=None, n_jobs=1):
    """Return DI/NetDI arrays shaped epochs x regions x regions x lags.

    rois optionally maps any region names to lists of zero-based channel
    indices; those channels are averaged before rank-normalization. Without
    rois, each channel is a region. Axis 1 is source, axis 2 target.
    Actual representable lags and region names are returned alongside arrays.
    """
    data = epochs_array(epochs, sfreq, n_jobs)
    requested = np.asarray(lags_ms, dtype=float)
    if requested.ndim != 1 or not len(requested) or not np.isfinite(requested).all():
        raise ValueError("lags_ms must be a nonempty finite 1-D sequence")
    lags = np.rint(requested * sfreq / 1000).astype(int)
    if (np.any(requested <= 0) or np.any(np.diff(requested) <= 0)
            or np.any(lags < 1) or np.any(lags >= data.shape[-1] - 3)
            or len(np.unique(lags)) != len(lags)):
        raise ValueError("Lags must be increasing, distinct, positive sample lags shorter than the signal")
    if rois is None:
        names = list(channel_names) if channel_names is not None else [f"Channel {i+1}" for i in range(data.shape[1])]
        if len(names) != data.shape[1] or len(set(names)) != len(names):
            raise ValueError("Provide one unique name per channel")
        regional = data
    else:
        if not rois or len(set(rois)) != len(rois):
            raise ValueError("Provide a nonempty mapping of unique ROI names")
        names, signals = [], []
        for name, members in rois.items():
            indices = np.asarray(members)
            if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
                    or len(np.unique(indices)) != len(indices)
                    or np.any(indices < 0) or np.any(indices >= data.shape[1])):
                raise ValueError(f"ROI {name!r} needs unique valid channel indices")
            names.append(str(name))
            signals.append(data[:, indices].mean(axis=1))
        regional = np.stack(signals, axis=1)
    if len(names) < 2:
        raise ValueError("At least two channels or ROIs are needed")
    normalized = np.empty_like(regional)
    for e in range(len(regional)):
        for r in range(len(names)):
            normalized[e, r] = gaussianize(regional[e, r])

    def one_epoch(e):
        result = np.zeros((len(names), len(names), len(lags)))
        for source in range(len(names)):
            for target in range(len(names)):
                if source != target:
                    result[source, target] = [directed_bits(
                        normalized[e, source], normalized[e, target], lag) for lag in lags]
        return result

    di = np.stack(Parallel(n_jobs=int(n_jobs), prefer="threads")(
        delayed(one_epoch)(e) for e in range(len(regional))))
    return {"di": di, "netdi": di - di.swapaxes(1, 2),
            "lags_ms": 1000 * lags / sfreq, "lag_samples": lags,
            "region_names": np.asarray(names, dtype=str)}
