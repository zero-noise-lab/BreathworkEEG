"""Demonstrate the general methods using the bundled synthetic dataset.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from importlib.metadata import version

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "eeg-methods-mpl"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from computation.lzc import compute_lzc
from computation.wpli import compute_wpli
from computation.directed_information import compute_directed_information
from computation.modulation_index import compute_modulation_index
from stats.correlations import repeated_measures_correlation, paired_test, bh_fdr
from stats.permutation import cluster_permutation_1samp
from plotting.plots import plot_topoplot, plot_heatmap, plot_lag_curves, plot_modulation_index


def run_example(data_path, output):
    with np.load(data_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    voltage, csd, sfreq = data["voltage"], data["csd"], float(data["sfreq"])
    subjects, conditions, epochs, channels, _ = voltage.shape
    bands = {"Broadband": (1,45), "Alpha": (8,13), "Beta": (13,30)}
    names = list(bands)
    positions, condition_names = data["positions"], list(data["condition_names"])
    rois = {str(name): np.flatnonzero(data["roi_membership"] == i)
            for i, name in enumerate(data["roi_names"])}
    lags_ms = np.arange(10, 161, 10)
    lzc = np.empty((subjects,conditions,channels,len(bands)))
    wpli = np.empty((subjects,conditions,channels,channels,len(bands)))
    netdi = np.empty((subjects,conditions,len(rois),len(rois),len(lags_ms)))
    for s in range(subjects):
        for c in range(conditions):
            complexity = compute_lzc(voltage[s,c], sfreq, bands=bands)
            connectivity = compute_wpli(csd[s,c], sfreq, bands=bands)
            information = compute_directed_information(csd[s,c], sfreq, lags_ms, rois=rois)
            for b, band in enumerate(names):
                lzc[s,c,:,b] = complexity[band].mean(axis=0)
                wpli[s,c,:,:,b] = connectivity[band]
            netdi[s,c] = information["netdi"].mean(axis=0)
    nodal = wpli.sum(axis=3)
    edges = np.triu_indices(channels, 1)
    wpli_global = wpli[:,:,edges[0],edges[1],:].mean(axis=2)
    features = np.concatenate([lzc.mean(axis=2), wpli_global,
                               netdi[:,:,0,-1,:].mean(axis=-1)[...,None]], axis=-1)
    feature_names = [f"LZC {band}" for band in names] + [f"wPLI {band}" for band in names] + ["NetDI lag mean"]
    mi_lzc = compute_modulation_index(lzc.mean(axis=2), lzc[:, [0]].mean(axis=2))
    mi_wpli = compute_modulation_index(wpli_global, wpli_global[:, [0]])
    # Wilcoxon against zero; one FDR family across nonbaseline conditions per
    # metric/band, matching the original modulation-index analysis.
    mi_tests = [[paired_test(np.zeros_like(mi[:, 1:, b]), mi[:, 1:, b])
                 for b in range(len(bands))] for mi in (mi_lzc, mi_wpli)]
    predictors = (data["rating_start"], data["rating_end"])
    correlations = [repeated_measures_correlation(x, features) for x in predictors]
    r = np.column_stack([result["r"] for result in correlations])
    p = np.column_stack([result["p"] for result in correlations])
    q = bh_fdr(p)  # One explicitly chosen family: every displayed heatmap test.
    channel_metrics = (lzc[:,:,:,names.index("Alpha")], nodal[:,:,:,names.index("Alpha")])
    sensor_tests = [[repeated_measures_correlation(x, metric) for x in predictors]
                    for metric in channel_metrics]
    sensor_r = np.array([[test["r"] for test in metric] for metric in sensor_tests])
    sensor_p = np.array([[test["p"] for test in metric] for metric in sensor_tests])
    sensor_q = bh_fdr(sensor_p)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"svg.fonttype":"none", "font.size":9})

    def save(fig, name):
        for extension in ("png", "svg"):
            fig.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, len(bands), figsize=(11,6), layout="constrained")
    for b, band in enumerate(names):
        lzc_values = lzc[:,-1,:,b].mean(axis=0)
        wpli_values = nodal[:,-1,:,b].mean(axis=0)
        plot_topoplot(lzc_values, positions, ax=axes[0,b],
                      title=f"LZC | {band}", label="Normalized LZ76", cmap="viridis",
                      vlim=(lzc_values.min(), lzc_values.max()))
        plot_topoplot(wpli_values, positions, ax=axes[1,b],
                      title=f"wPLI nodal strength | {band}", label="Sum of edge weights", cmap="viridis",
                      vlim=(wpli_values.min(), wpli_values.max()))
    fig.suptitle(f"Synthetic data | {condition_names[-1]}")
    save(fig, "lzc_wpli_topoplots")

    fig, axes = plt.subplots(1, 2, figsize=(11,4), layout="constrained")
    for index, (mi, metric) in enumerate(((mi_lzc, "LZC"), (mi_wpli, "wPLI"))):
        significant = np.zeros(mi.shape[1:], dtype=bool)
        significant[1:] = np.column_stack([test["q"] < .05 for test in mi_tests[index]])
        plot_modulation_index(mi, condition_names, names, ax=axes[index],
                              title=f"Synthetic {metric} modulation (mean +/- SEM)",
                              significant=significant)
    fig.supxlabel("* Wilcoxon vs zero; BH-FDR across nonbaseline conditions within each metric/band", fontsize=8)
    save(fig, "modulation_index")

    curves = netdi[:,1:,0,-1,:] - netdi[:,[0],0,-1,:]
    fig, ax = plt.subplots(figsize=(7,4), layout="constrained")
    plot_lag_curves(curves, lags_ms, [str(name)+" - baseline" for name in condition_names[1:]],
                    ax=ax, title="Synthetic frontal-posterior NetDI change (mean +/- SEM)")
    save(fig, "netdi_lag_curves")

    matrix = (netdi[:,-1] - netdi[:,0]).mean(axis=(0,3))
    np.fill_diagonal(matrix, np.nan)
    limit = max(float(np.nanmax(np.abs(matrix))), 1e-9)
    fig, ax = plt.subplots(figsize=(5,4), layout="constrained")
    plot_heatmap(matrix, list(rois), list(rois), ax=ax,
                 title="Synthetic NetDI change: last condition - baseline",
                 label="Lag-mean change (bits)", vlim=(-limit,limit), annotate=True,
                 value_format=".4f")
    ax.set(xlabel="Target ROI", ylabel="Source ROI")
    save(fig, "netdi_heatmap")

    fig, ax = plt.subplots(figsize=(6,5), layout="constrained")
    plot_heatmap(r, feature_names, ["Start rating", "End rating"], ax=ax,
                 title="Synthetic within-subject rating correlations",
                 label="Within-subject r", vlim=(-1,1), significant=q<.05, annotate=True)
    fig.supxlabel(f"* BH-FDR q < .05 across all {p.size} displayed tests; synthetic demonstration",fontsize=8)
    save(fig, "rating_correlation_heatmap")

    fig, axes = plt.subplots(2,2,figsize=(8,6),layout="constrained")
    for method, label in enumerate(("LZC Alpha", "wPLI Alpha nodal strength")):
        for predictor, rating in enumerate(("Start rating", "End rating")):
            plot_topoplot(sensor_tests[method][predictor]["r"], positions,
                          ax=axes[method,predictor], title=f"{label} | {rating}",
                          label="Within-subject r", vlim=(-1,1),
                          significant=sensor_q[method,predictor]<.05)
    fig.suptitle(f"Synthetic sensor correlations | BH-FDR across {sensor_q.size} sensor tests")
    fig.supxlabel("White circles: BH-FDR q < .05", fontsize=8)
    save(fig, "rating_correlation_topoplots")

    paired = paired_test(features[:,0], features[:,-1])
    cluster = cluster_permutation_1samp(curves[:,-1], permutations=256, seed=42)
    summary = {
        "description":"Entirely synthetic demonstration; no study estimates",
        "feature_names":feature_names,
        "rating_names":["Start rating", "End rating"],
        "paired_comparison":f"{condition_names[-1]} minus {condition_names[0]}",
        "paired_fdr_family":"All supplied feature tests",
        "heatmap_fdr_family":int(p.size), "sensor_fdr_family":int(sensor_q.size),
        "paired_wilcoxon_p":np.asarray(paired["p"]).tolist(),
        "paired_wilcoxon_q":np.asarray(paired["q"]).tolist(),
        "modulation_index":{
            "fdr_family":"Nonbaseline conditions separately within each metric/band",
            "bands":names,
            "conditions":[str(name) for name in condition_names[1:]],
            **{metric:{key:np.column_stack([test[key] for test in tests]).tolist()
                       for key in ("p", "q")}
               for metric, tests in zip(("lzc", "wpli"), mi_tests)},
        },
        "lag_cluster_map_p":cluster["map_p"],
        "lag_cluster_comparison":{
            "condition":str(condition_names[-1]), "baseline":str(condition_names[0]),
            "source_roi":str(data["roi_names"][0]), "target_roi":str(data["roi_names"][-1]),
            "lags_ms":information["lags_ms"].tolist(),
            "requested_permutations":256, "seed":42,
            "correction":"Maximum cluster within this lag map; no across-map correction",
        },
        "software_versions":{name:version(name) for name in
                             ("numpy", "scipy", "mne", "mne-connectivity", "matplotlib", "joblib")},
        "lag_clusters":[{"indices":entry["indices"].tolist(),"p":entry["p"]} for entry in cluster["clusters"]],
    }
    (output/"statistics.json").write_text(json.dumps(summary,indent=2)+"\n")
    np.savez_compressed(output/"example_results.npz", lzc=lzc, wpli=wpli, netdi=netdi,
                        r=r, p=p, q=q, lags_ms=information["lags_ms"],
                        lag_samples=information["lag_samples"],
                        sensor_r=sensor_r, sensor_p=sensor_p, sensor_q=sensor_q,
                        mi_lzc=mi_lzc, mi_wpli=mi_wpli,
                        subject_names=data["subject_names"], condition_names=data["condition_names"],
                        channel_names=data["channel_names"], roi_names=data["roi_names"],
                        roi_membership=data["roi_membership"], positions=positions,
                        positions_3d=data["positions_3d"], sfreq=sfreq,
                        band_names=np.asarray(names), band_limits_hz=np.asarray(list(bands.values())),
                        feature_names=np.asarray(feature_names), features=features,
                        rating_names=np.asarray(["Start rating", "End rating"]),
                        rating_start=data["rating_start"], rating_end=data["rating_end"],
                        sensor_metric_names=np.asarray(["LZC Alpha", "wPLI Alpha nodal strength"]))
    savemat(output/"topoplots.mat", {"positions":positions,
        "lzc_alpha":lzc[:,-1,:,names.index("Alpha")].mean(axis=0),
        "wpli_alpha":nodal[:,-1,:,names.index("Alpha")].mean(axis=0)})
    print(f"Saved six PNG/SVG figure pairs and example statistics to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path(__file__).parent / "data" / "synthetic_eeg.npz")
    parser.add_argument("--output",type=Path,default=Path(tempfile.gettempdir())/"eeg-methods-example")
    args = parser.parse_args()
    run_example(args.data,args.output)
