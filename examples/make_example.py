"""Generate entirely synthetic EEG, spherical-spline CSD, and repeated ratings.
"""
import argparse
import csv
from pathlib import Path

import mne
import numpy as np
from scipy.signal import lfilter


def make_example(output, seed=42, ratings_output=None):
    rng = np.random.default_rng(seed)
    n_subjects, n_conditions, n_epochs, n_channels, n_samples = 6, 5, 4, 8, 500
    sfreq, radius = 100.0, 0.095
    positions = np.array([[-.23,.31],[.23,.31],[-.34,0],[.34,0],
                          [-.25,-.25],[.25,-.25],[-.16,-.43],[.16,-.43]])
    xy = positions * (2 * radius)
    xyz = np.column_stack([xy, np.sqrt(radius**2 - np.square(xy).sum(axis=1))])
    channel_names = np.array([f"Sensor {i+1}" for i in range(n_channels)])
    subject_names = np.array([f"Synthetic {i+1}" for i in range(n_subjects)])
    condition_names = np.array(["Baseline", "Condition 1", "Condition 2", "Condition 3", "Condition 4"])
    start = np.clip(rng.normal(2, .6, (n_subjects, 1)) + np.array([0,2,5,3,1])
                    + rng.normal(0, .5, (n_subjects, n_conditions)), 0, 10)
    end = np.clip(start + rng.normal(0, .9, start.shape) + np.array([0,1,-1,.5,-.5]), 0, 10)
    voltage = np.empty((n_subjects, n_conditions, n_epochs, n_channels, n_samples))
    mixing = np.array([[1,.2,.05],[.9,.1,.05],[.3,1,.1],[.2,1,.1],
                       [.1,.3,1],[.1,.2,1],[.05,.1,1.1],[.05,.1,1]])
    time = np.arange(n_samples + 100) / sfreq
    for s in range(n_subjects):
        for c in range(n_conditions):
            for e in range(n_epochs):
                sources = lfilter([1], [1,-.65], rng.normal(size=(3, n_samples+100)), axis=-1)
                sources[0] += (2.4 - .16 * start[s,c]) * np.sin(2*np.pi*10*time + rng.uniform(0,2*np.pi))
                sources[2,2:] += (.2 + .07 * start[s,c]) * sources[0,:-2]
                voltage[s,c,e] = 8e-6 * (mixing @ sources[:,100:]
                                          + .6 * rng.normal(size=(n_channels,n_samples)))
    info = mne.create_info(list(channel_names), sfreq, "eeg")
    info.set_montage(mne.channels.make_dig_montage(
        ch_pos=dict(zip(channel_names, xyz)), coord_frame="head"))
    epochs = mne.EpochsArray(voltage.reshape(-1,n_channels,n_samples), info, verbose=False)
    # Manuscript settings. MNE returns the negative surface Laplacian;
    # multiplying by conductivity gives FieldTrip's -sigma * Laplacian convention.
    # Input volts and positions in meters yield conductivity-scaled CSD in A/m^3.
    csd_parameters = dict(lambda2=1e-5, stiffness=4, n_legendre_terms=14)
    conductivity = 0.33  # S/m
    transformed = mne.preprocessing.compute_current_source_density(
        epochs, sphere=(0,0,0,radius), **csd_parameters, verbose=False)
    csd = conductivity * transformed.get_data(copy=True).reshape(voltage.shape)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, voltage=voltage.astype(np.float32), csd=csd.astype(np.float32),
        sfreq=sfreq, positions=positions, positions_3d=xyz,
        channel_names=channel_names, subject_names=subject_names,
        condition_names=condition_names, rating_start=start, rating_end=end,
        roi_membership=np.array([0,0,1,1,2,2,2,2]),
        roi_names=np.array(["Frontal", "Central", "Posterior"]), seed=seed,
        description="Entirely synthetic signals and ratings; no study observations",
        voltage_unit="V", csd_unit="A/m^3",
        csd_transform="MNE spherical splines, conductivity-scaled negative surface Laplacian",
        csd_sphere=np.array([0, 0, 0, radius]), csd_lambda2=csd_parameters["lambda2"],
        csd_stiffness=csd_parameters["stiffness"],
        csd_n_legendre_terms=csd_parameters["n_legendre_terms"],
        csd_conductivity_s_per_m=conductivity,
        mne_version=mne.__version__)
    ratings_output = (Path(ratings_output) if ratings_output is not None
                      else output.with_name(output.stem.removesuffix("_eeg") + "_ratings.csv"))
    ratings_output.parent.mkdir(parents=True, exist_ok=True)
    with ratings_output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["subject", "condition", "rating_start", "rating_end"])
        for s in range(n_subjects):
            for c in range(n_conditions):
                writer.writerow([subject_names[s], condition_names[c], start[s,c], end[s,c]])
    print(f"Saved synthetic dataset: {output}")
    print(f"Saved synthetic ratings: {ratings_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data" / "synthetic_eeg.npz")
    parser.add_argument("--ratings-output", type=Path,
                        help="CSV destination; defaults to a sibling <name>_ratings.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make_example(args.output, args.seed, args.ratings_output)
