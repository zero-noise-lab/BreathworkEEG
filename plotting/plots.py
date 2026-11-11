"""Topoplots, matrices and lag curves from arrays supplied by the caller.
"""
import numpy as np
import matplotlib.pyplot as plt
from mne.viz import plot_topomap as mne_topomap


def _axes(ax):
    return plt.subplots(figsize=(5, 4))[1] if ax is None else ax


def plot_topoplot(values, positions, *, ax=None, title="", label="", cmap="RdBu_r",
                   vlim=(None, None), significant=None, head_radius=0.5):
    """One value per electrode; positions is channels x 2 in head coordinates.

    Coordinates must use the same scale as head_radius. The example uses a
    head circle of radius 0.5. Optional significance markers are supplied by
    the caller; they are not inferred by the plotting function.
    """
    values, positions = np.asarray(values, float), np.asarray(positions, float)
    if (values.ndim != 1 or positions.shape != (len(values), 2) or len(values) < 3
            or not np.isfinite(values).all() or not np.isfinite(positions).all()
            or len(np.unique(positions, axis=0)) != len(values)
            or np.linalg.matrix_rank(positions - positions.mean(axis=0)) < 2):
        raise ValueError("Provide finite values and at least three distinct noncollinear 2-D positions")
    if not np.isfinite(head_radius) or head_radius <= 0:
        raise ValueError("head_radius must be positive")
    mask = None if significant is None else np.asarray(significant, bool)
    if mask is not None and mask.shape != values.shape:
        raise ValueError("Significance mask must match the channel vector")
    ax = _axes(ax)
    image, _ = mne_topomap(
        values, positions, axes=ax, show=False, contours=0, cmap=cmap, vlim=vlim,
        # Box support also handles symmetric/cocircular small sensor layouts;
        # MNE still clips the displayed interpolation to the head outline.
        sphere=(0, 0, 0, head_radius), mask=mask, extrapolate="box")
    ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, shrink=0.72, label=label)
    return ax


def plot_heatmap(values, row_labels=None, column_labels=None, *, ax=None, title="",
                 label="", cmap="RdBu_r", vlim=None, significant=None, annotate=False,
                 value_format=".2f"):
    """Plot an arbitrary 2-D matrix; NaNs are untested/missing grey cells."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.size:
        raise ValueError("Provide a nonempty 2-D matrix")
    if significant is not None and np.shape(significant) != values.shape:
        raise ValueError("Significance mask must match the matrix")
    if row_labels is not None and len(row_labels) != values.shape[0]:
        raise ValueError("row_labels must match the rows")
    if column_labels is not None and len(column_labels) != values.shape[1]:
        raise ValueError("column_labels must match the columns")
    ax = _axes(ax)
    palette = plt.get_cmap(cmap).copy()
    palette.set_bad("#e5e5e5")
    limits = (None, None) if vlim is None else vlim
    image = ax.imshow(np.ma.masked_invalid(values), cmap=palette,
                      vmin=limits[0], vmax=limits[1], aspect="auto")
    if row_labels is not None:
        ax.set_yticks(range(len(row_labels)), labels=row_labels)
    if column_labels is not None:
        ax.set_xticks(range(len(column_labels)), labels=column_labels, rotation=35, ha="right")
    for i, j in np.ndindex(values.shape):
        if not np.isfinite(values[i, j]):
            continue
        text = format(values[i,j], value_format) if annotate else ""
        if significant is not None and significant[i, j]:
            text += "*"
        if text:
            rgb = np.asarray(image.cmap(image.norm(values[i, j]))[:3])
            color = "white" if rgb @ [0.2126, 0.7152, 0.0722] < 0.45 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color)
    ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, shrink=0.85, label=label)
    return ax


def plot_lag_curves(values, lags_ms, labels=None, *, ax=None, title="", ylabel="NetDI (bits)"):
    """Plot mean +/- SEM across subjects, not across electrodes or epochs.

    Input: subjects x lags, or subjects x curves x lags.
    """
    values, lags = np.asarray(values, float), np.asarray(lags_ms, float)
    if values.ndim == 2:
        values = values[:, None, :]
    if (values.ndim != 3 or min(values.shape) < 1 or lags.shape != (values.shape[-1],)
            or not np.isfinite(values).all() or not np.isfinite(lags).all()
            or np.any(np.diff(lags) <= 0)):
        raise ValueError("Provide finite subject x curve x lag values and increasing lag coordinates")
    if labels is not None and len(labels) != values.shape[1]:
        raise ValueError("labels must match the curve axis")
    ax = _axes(ax)
    for index in range(values.shape[1]):
        traces = values[:, index]
        mean = traces.mean(axis=0)
        sem = traces.std(axis=0, ddof=1) / np.sqrt(len(traces)) if len(traces) > 1 else np.zeros_like(mean)
        line, = ax.plot(lags, mean, label=labels[index] if labels is not None else None)
        ax.fill_between(lags, mean - sem, mean + sem, color=line.get_color(), alpha=0.18)
    ax.axhline(0, color="0.5", lw=0.7, ls="--")
    ax.set(xlabel="Lag (ms)", ylabel=ylabel, title=title)
    if labels is not None:
        ax.legend(frameon=False)
    return ax


def plot_modulation_index(values, condition_labels=None, series_labels=None, *,
                          ax=None, title="", significant=None):
    """Plot precomputed subject-level MI as mean +/- SEM across subjects.
    """
    if np.iscomplexobj(values):
        raise ValueError("Modulation indices must be real-valued")
    values = np.asarray(values, dtype=float)
    if (values.ndim not in (2, 3) or min(values.shape) < 1
            or np.isinf(values).any() or np.any(np.abs(values) > 1)):
        raise ValueError("Provide subjects x conditions [x series] MI in [-1, 1], with NaNs for missing entries")
    mask = None if significant is None else np.asarray(significant, bool)
    if mask is not None and mask.shape != values.shape[1:]:
        raise ValueError("Significance mask must match the condition/series axes")
    if values.ndim == 2:
        values = values[..., None]
        if mask is not None:
            mask = mask[:, None]
    if condition_labels is not None and len(condition_labels) != values.shape[1]:
        raise ValueError("condition_labels must match the condition axis")
    if series_labels is not None and len(series_labels) != values.shape[2]:
        raise ValueError("series_labels must match the series axis")
    ax = _axes(ax)
    x = np.arange(values.shape[1])
    for index in range(values.shape[2]):
        data = values[:, :, index]
        finite = np.isfinite(data)
        n = finite.sum(axis=0)
        mean = np.divide(np.nansum(data, axis=0), n,
                         out=np.full(n.shape, np.nan), where=n > 0)
        squares = np.where(finite, data - mean, 0)**2
        variance = np.divide(squares.sum(axis=0), n - 1,
                             out=np.full(n.shape, np.nan), where=n > 1)
        sem = np.sqrt(np.divide(variance, n, out=np.full(n.shape, np.nan), where=n > 0))
        line, = ax.plot(x, mean, "-o", ms=4,
                        label=series_labels[index] if series_labels is not None else None)
        ax.fill_between(x, mean - sem, mean + sem, color=line.get_color(), alpha=0.18)
        if mask is not None:
            for condition in np.flatnonzero(mask[:, index] & np.isfinite(mean)):
                ax.annotate("*", (x[condition], mean[condition] + np.nan_to_num(sem[condition])),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", color=line.get_color())
    ax.axhline(0, color="0.5", lw=0.7, ls="--")
    ax.set_xticks(x, labels=condition_labels if condition_labels is not None else x,
                  rotation=30, ha="right")
    ax.set(xlabel="Condition", ylabel="Modulation index", title=title)
    ax.margins(x=0.05, y=0.2)
    if series_labels is not None:
        ax.legend(frameon=False)
    return ax
