"""Within-subject correlations, paired tests and BH-FDR for arbitrary arrays.

Bakdash & Marusich (2017), doi:10.3389/fpsyg.2017.00456;
Benjamini & Hochberg (1995), doi:10.1111/j.2517-6161.1995.tb02031.x.
"""
import numpy as np
from scipy import stats


def bh_fdr(pvalues):
    """BH-adjust all finite p values as one family; preserve shape and NaNs."""
    p = np.asarray(pvalues, dtype=float)
    if np.isinf(p).any() or np.any((p < 0) | (p > 1)):
        raise ValueError("p values must be in [0, 1], or NaN for untested entries")
    q = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    valid = p[finite]
    if len(valid):
        order = np.argsort(valid)
        ranked = valid[order] * len(valid) / np.arange(1, len(valid) + 1)
        corrected = np.minimum.accumulate(ranked[::-1])[::-1]
        restored = np.empty_like(corrected)
        restored[order] = np.minimum(corrected, 1)
        q[finite] = restored
    return q


def repeated_measures_correlation(x, y):
    """Common within-subject slope, r, t and two-sided parametric p.

    x: subjects x repeated observations. y: same first two axes, optionally
    followed by feature axes. Complete subjects with varying x are selected
    separately per feature; at least three are required. No event labels,
    condition counts or file formats are assumed. Degenerate response features
    return NaN statistics. The df is observations - subjects - 1.
    """
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        raise ValueError("Predictors and responses must be real-valued")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2 or y.shape[:2] != x.shape:
        raise ValueError("x must be subjects x observations; y must share those axes")
    shape = y.shape[2:]
    columns = y.reshape(*x.shape, -1)
    result = {key: np.full(columns.shape[-1], np.nan)
              for key in ("r", "beta", "t", "p", "df", "n_subjects", "n_observations")}
    finite_x = np.isfinite(x).all(axis=1)
    varying_x = np.zeros(x.shape[0], dtype=bool)
    varying_x[finite_x] = np.ptp(x[finite_x], axis=1) > 0
    for index in range(columns.shape[-1]):
        values = columns[:, :, index]
        valid = finite_x & varying_x & np.isfinite(values).all(axis=1)
        subjects = int(valid.sum())
        result["n_subjects"][index] = subjects
        result["n_observations"][index] = subjects * x.shape[1]
        if subjects < 3:
            continue
        xc = x[valid] - x[valid].mean(axis=1, keepdims=True)
        yc = values[valid] - values[valid].mean(axis=1, keepdims=True)
        sx, sy = np.square(xc).sum(), np.square(yc).sum()
        observations, df = xc.size, xc.size - subjects - 1
        result["df"][index] = df
        result["n_subjects"][index] = subjects
        result["n_observations"][index] = observations
        if sx <= 0 or sy <= 0 or df <= 0:
            continue
        cross = (xc * yc).sum()
        r = float(np.clip(cross / np.sqrt(sx * sy), -1, 1))
        t = np.copysign(np.inf, r) if abs(r) == 1 else r * np.sqrt(df / (1 - r*r))
        for key, value in {"r": r, "beta": cross / sx, "t": t,
                           "p": 2 * stats.t.sf(abs(t), df)}.items():
            result[key][index] = value
    return {key: value.reshape(shape) if shape else float(value[0])
            for key, value in result.items()}


def between_subject_correlation(x, y):
    """Between-subject Pearson r, OLS slope, t, two-sided p and BH-FDR q.

    x: one value per subject (1-D). y: subjects x optional feature axes, e.g.
    columns for DED, VR, OBN and MEQ total. Rows must refer to the same subjects
    in the same order. For peak ASC versus survey scores, supply each subject's
    precomputed peak ASC as x and their survey scores as y. Peak selection is
    not performed here. Call separately for each group/session.

    Each subject contributes one pair per feature; this is not repeated-measures
    correlation. Nonfinite pairs are removed separately per feature. At least
    three pairs and variation in both variables are required. Constant inputs
    return NaN statistics. df = n_subjects - 2; beta is the slope of y on x,
    with an intercept. q adjusts all finite feature p values as one family
    (e.g. the four survey dimensions supplied together).

    Example (complete ratings, subjects x repeated measurements):
        peak_depth = asc_ratings.max(axis=1)
        result = between_subject_correlation(peak_depth, survey_scores)
    """
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        raise ValueError("Predictors and responses must be real-valued")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 1 or not x.size or y.ndim < 1 or not y.size
            or y.shape[0] != x.size):
        raise ValueError("x must be one value per subject; y must share the subject axis")
    shape = y.shape[1:]
    columns = y.reshape(x.size, -1)
    result = {key: np.full(columns.shape[1], np.nan)
              for key in ("r", "beta", "t", "p", "df", "n_subjects")}
    for index, values in enumerate(columns.T):
        valid = np.isfinite(x) & np.isfinite(values)
        n = int(valid.sum())
        result["n_subjects"][index] = n
        if n < 3:
            continue
        result["df"][index] = n - 2
        xv, yv = x[valid], values[valid]
        if np.ptp(xv) == 0 or np.ptp(yv) == 0:
            continue
        r, p = stats.pearsonr(xv, yv)
        xc, yc = xv - xv.mean(), yv - yv.mean()
        t = np.copysign(np.inf, r) if abs(r) == 1 else r * np.sqrt((n - 2) / (1 - r*r))
        for key, value in {"r": r, "beta": (xc * yc).sum() / np.square(xc).sum(),
                           "t": t, "p": p}.items():
            result[key][index] = value
    result["q"] = bh_fdr(result["p"])
    return {key: value.reshape(shape) if shape else float(value[0])
            for key, value in result.items()}


def paired_test(before, after, method="wilcoxon"):
    """Paired tests along axis 0; remaining axes are arbitrary features.

    Positive mean_difference means after > before. Missing pairs are removed
    per feature. q adjusts all finite feature p values as one family; select
    a different correction family explicitly with bh_fdr when appropriate.
    """
    if np.iscomplexobj(before) or np.iscomplexobj(after):
        raise ValueError("Paired measurements must be real-valued")
    a, b = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if a.shape != b.shape or a.ndim < 1 or not a.size or method not in {"wilcoxon", "ttest"}:
        raise ValueError("Provide matched arrays and method='wilcoxon' or 'ttest'")
    shape = a.shape[1:]
    differences = (b - a).reshape(a.shape[0], -1)
    result = {key: np.full(differences.shape[-1], np.nan)
              for key in ("statistic", "p", "mean_difference", "n")}
    for index, values in enumerate(differences.T):
        values = values[np.isfinite(values)]
        result["n"][index] = len(values)
        if len(values) < 2:
            continue
        if not np.any(values):
            statistic, p = 0.0, 1.0
        else:
            test = stats.wilcoxon(values) if method == "wilcoxon" else stats.ttest_1samp(values, 0)
            statistic, p = test.statistic, test.pvalue
        result["statistic"][index], result["p"][index] = statistic, p
        result["mean_difference"][index] = values.mean()
    result["q"] = bh_fdr(result["p"])
    return {key: value.reshape(shape) if shape else float(value[0])
            for key, value in result.items()}
