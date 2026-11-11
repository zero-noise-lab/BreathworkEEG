"""Two-sided one-sample cluster permutation tests on lags or sensors.

Maris & Oostenveld (2007), doi:10.1016/j.jneumeth.2007.03.024.
"""
import warnings
import numpy as np
from scipy import sparse, stats
from mne.stats import permutation_cluster_1samp_test


def cluster_permutation_1samp(values, adjacency=None, permutations=5000, seed=42,
                             cluster_p=0.05, alpha=0.05):
    """Test subject-level differences (subjects x locations) against zero.
    """
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 1
            or not np.isfinite(values).all()):
        raise ValueError("Provide finite subject-level differences: at least 3 subjects x locations")
    if np.any(np.var(values, axis=0) == 0):
        raise ValueError("Each location must have nonzero between-subject variance")
    if (isinstance(permutations, bool) or not isinstance(permutations, (int, np.integer))
            or permutations < 2 or not 0 < cluster_p < 1 or not 0 < alpha < 1):
        raise ValueError("Need permutations >= 2 and probabilities strictly between 0 and 1")
    if adjacency is not None:
        adjacency = sparse.csr_matrix(adjacency, dtype=bool)
        if adjacency.shape != (values.shape[1], values.shape[1]) or (adjacency != adjacency.T).nnz:
            raise ValueError("adjacency must be square, symmetric, and match locations")
    threshold = stats.t.ppf(1 - cluster_p / 2, values.shape[0] - 1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No clusters found.*")
        t, found, p, _ = permutation_cluster_1samp_test(
            values, threshold=threshold, tail=0, adjacency=adjacency,
            n_permutations=int(permutations), seed=seed, out_type="indices", verbose=False)
    clusters, significant = [], np.zeros(values.shape[1], dtype=bool)
    for component, probability in zip(found, p):
        indices = np.asarray(component[0] if isinstance(component, tuple) else component)
        clusters.append({"indices": indices, "mass": float(t[indices].sum()),
                         "p": float(probability)})
        if probability < alpha:
            significant[indices] = True
    return {"t": t, "clusters": clusters, "significant": significant,
            "map_p": min((cluster["p"] for cluster in clusters), default=1.0)}
