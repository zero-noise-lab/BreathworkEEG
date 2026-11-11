# BreathworkEEG

Code accompanying **Neural and Experiential Dynamics of High-Ventilation
Breathwork Mirror Psychedelic States**.

This repository contains EEG preprocessing, spectral analysis, signal complexity,
functional connectivity, directed-information analysis, and transcript word-cloud
code. A small synthetic EEG and ratings dataset demonstrates the Python methods
without participant recordings.

## Repository structure

```text
EEG/MATLAB/
  PREPROCESSING/       EEG and auxiliary-signal cleaning
  ANALYSIS/            Spectral analysis, statistics, and figures
Annotations/           Transcript processing and word-cloud analysis
computation/           LZC, wPLI, DI/NetDI, and modulation index
stats/                 Correlations, paired tests, FDR, and cluster permutations
plotting/              Python plots and a MATLAB topoplot helper
examples/
  make_example.py      Synthetic EEG and ratings generator
  run_example.py       Runnable analysis and plotting example
  data/
    synthetic_eeg.npz
    synthetic_ratings.csv
README.md
requirements.txt
LICENSE
```

## Analysis workflow

The manuscript analyses five eyes-closed recordings: baseline (B) and R1–R4.
Each four-minute recording is divided into five-second analysis windows.
Preprocessing uses 63 scalp channels at 500 Hz; retained voltage windows are
used for spectral power and LZC, and their CSD representation for wPLI and DI.

```text
Raw EEG -> filtering, artifact cleaning, channel repair, average reference
        -> recording segmentation and five-second window selection
           -> voltage -> spectral power and LZC
           -> CSD     -> wPLI and DI/NetDI
        -> participant-level summaries -> statistics -> figures
```

CSD is applied to cleaned time series before connectivity analysis. Retain the
same selected windows and channel identities in the voltage and CSD branches.
Participant/session labels, recording labels, sample boundaries, units, electrode
coordinates, and processing parameters are needed when preparing study inputs.

### MATLAB preprocessing and spectral analysis

Follow the [preprocessing instructions](EEG/MATLAB/PREPROCESSING/PP_README)
for the FieldTrip/EEGLAB pipeline. `PP_05_FIN_CLEAN_fun.m` saves cleaned,
channel-repaired, average-referenced recordings; `PP_06_MERGE_PASS_fun.m` merges
them with auxiliary and survey information.

The per-participant spectral entry point is
[`RUN_FREQ_ANALYSIS_fun.m`](EEG/MATLAB/ANALYSIS/RUN_FREQ_ANALYSIS_fun.m), followed
by [`FREQ_RESULTS.m`](EEG/MATLAB/ANALYSIS/FREQ_RESULTS.m) for results and figures.
The [analysis instructions](EEG/MATLAB/ANALYSIS/FREQ_README) refer to the
per-participant entry point as `run_freq_analysis_mean.m`; use the filename above.
Set the data paths and toolbox paths in the MATLAB scripts for your environment.

The spectral script performs window rejection and random subsampling internally.
Its saved frequency results contain counts but do not explicitly retain the
selected window indices or export the cleaned time series. The Python functions
accept prepared arrays; this repository does not include an automatic
MATLAB-to-Python study-data exporter.

### Transcript analysis

See the [word-cloud instructions](Annotations/code/annotations_word-clouds/README.md).
The supplied annotation data are synthetic. Transcript extraction requires the
participant source files and is separate from the runnable word-cloud example.
Install the annotation directory's own requirements for that workflow.

## Run the Python example

From the repository root, create a Python environment and install the dependencies.
Use Python 3.13 with the pinned requirements on Linux/macOS. GCC or Clang is needed
for the LZ76 C counter, which compiles into a temporary directory on first use.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/run_example.py --output outputs/synthetic_demo
```

`outputs/` is ignored by Git. Without `--output`, results go to
`eeg-methods-example` in the system temporary directory. Reusing an output path
overwrites the named result files. The root requirements cover the Python EEG
methods; MATLAB toolboxes and annotation dependencies are configured separately.

To regenerate the bundled synthetic inputs, or analyse another compatible archive:

```bash
python examples/make_example.py --seed 42
python examples/run_example.py --data /path/to/synthetic_eeg.npz --output /path/to/results
```

The generator's default command overwrites the two bundled inputs. Use
`--output /path/to/synthetic_eeg.npz` to write a separate copy; its ratings CSV
is saved alongside it as `synthetic_ratings.csv`. `--ratings-output` overrides
the CSV destination. The generator calculates CSD once; the runner loads it.

### Example inputs

Load `examples/data/synthetic_eeg.npz` with `np.load(path, allow_pickle=False)`.

| Fields | Dimensions or meaning |
|---|---|
| `voltage`, `csd` | 6 participants × 5 conditions × 4 windows × 8 channels × 500 samples |
| `sfreq` | 100 Hz; five seconds per window |
| `rating_start`, `rating_end` | 6 participants × 5 conditions |
| `channel_names`, `subject_names`, `condition_names` | Array labels |
| `positions`, `positions_3d` | 2-D plotting positions and 3-D electrode positions in meters |
| `roi_membership`, `roi_names` | Three illustrative sensor groups |

Voltage is in volts. The generator uses regularization `1e-5`, spline order 4,
and 14 Legendre terms, then scales MNE's negative surface Laplacian by conductivity
`0.33 S/m`. With voltage and geometry in SI units, the saved conductivity-scaled
CSD is in A/m³. These settings, units, and the MNE version are stored in the NPZ.
The CSV contains only `subject,condition,rating_start,rating_end`, not EEG samples.
The runner reads ratings from the NPZ, so changing only the CSV does not change
an analysis. The synthetic montage and ratings are not study observations.

### Example outputs

Six figure sets are saved in PNG and editable SVG:

- `lzc_wpli_topoplots`: final-condition LZC and wPLI nodal strength.
- `modulation_index`: baseline-relative LZC/wPLI changes, mean ± SEM.
- `netdi_lag_curves`: signed condition-minus-baseline NetDI, mean ± SEM.
- `netdi_heatmap`: signed NetDI change averaged across participants and lags.
- `rating_correlation_heatmap`: within-participant feature/rating correlations.
- `rating_correlation_topoplots`: alpha LZC/wPLI sensor/rating correlations.

`example_results.npz` saves the measures, correlation r/p/q arrays, ratings,
axis labels, bands, ROIs, sampling rate, geometry, and lags. `statistics.json`
saves paired tests, modulation-index tests, lag-cluster results, contrast labels,
correction families, and software versions. `topoplots.mat` contains selected
alpha values and coordinates for `plotting/plot_topoplot.m`.

In the results archive, LZC has axes participant × condition × channel × band;
wPLI adds a second channel axis; NetDI has participant × condition × source ROI
× target ROI × lag. Feature correlations have feature × rating axes, and sensor
correlations have metric × rating × channel axes. The `*_names` arrays identify
the corresponding dimensions.

### Relation to the manuscript

The example demonstrates the methods with reduced synthetic data. It does not
reproduce the manuscript's numerical results or all figure-specific analyses.

| Setting | Manuscript | Bundled example |
|---|---|---|
| Data | 25 participants; 63 scalp channels; 500 Hz | 6 simulated participants; 8 channels; 100 Hz |
| CSD | FieldTrip spherical splines, λ = 1e-5, order 4, degree 14, conductivity 0.33 S/m | MNE spherical splines with the same regularization, order, degree, and conductivity scaling; synthetic sphere geometry |
| ROIs and lags | 7 ROIs; 4–160 ms in 4-ms steps | 3 illustrative ROIs; 10–160 ms in 10-ms steps |
| NetDI summaries | Signed curves and peak baseline-relative changes in absolute NetDI | Signed differences and lag-mean summaries |
| Sensor inference | Spatial cluster permutations and analysis-specific FDR | Sensor correlation p-values with BH-FDR |

The example's modulation-index stars use Wilcoxon tests against zero, with
BH-FDR across four nonbaseline conditions within each metric/band. Feature
correlation stars use BH-FDR across 14 tests; sensor correlation circles use
BH-FDR across 32 tests. The paired-feature and lag-cluster results are saved
without significance markers on the descriptive topoplots or lag curves.
These are demonstration correction families, not substitutes for the manuscript's
ROI-edge, band, and contrast families. Refer to the manuscript Methods and
supplementary material for the study ROI definitions, peak selection, and full
statistical specification. The bundled runner demonstrates a reduced analysis;
it does not execute every manuscript figure. The synthetic CSD uses MNE rather
than invoking FieldTrip and is not a validation against the original study data.

## Python functions

For one participant/condition, computational inputs are
`windows × channels × samples`. Compute each participant/condition separately
when window counts differ, then combine comparable summaries for statistics.

| Function | Purpose and output |
|---|---|
| `compute_lzc` | Band-filtered median-threshold LZ76; windows × channels per band |
| `compute_wpli` | Multitaper wPLI across supplied windows; channel × channel per band |
| `compute_directed_information` | DI/NetDI in bits; windows × source ROI × target ROI × lag |
| `compute_modulation_index` | `(value - baseline) / (abs(value) + abs(baseline))` |
| `repeated_measures_correlation` | Common within-participant association across recordings |
| `between_subject_correlation` | Pearson association using one pair per participant |
| `paired_test` | Wilcoxon or paired-t test; paired differences, p and BH-adjusted p |
| `bh_fdr` | Adjust supplied finite p-values as one correction family |
| `cluster_permutation_1samp` | Sign-flip inference across lags or a supplied sensor adjacency |

LZC uses fourth-order zero-phase Butterworth filtering and `c(N)*log2(N)/N`
normalization; finite-sequence values can exceed 1. LZC and DI/NetDI are computed
per window and averaged in the demo. wPLI is estimated jointly across windows
and multitapers, then averaged within each frequency band. Python wPLI excludes
upper band endpoints except the largest requested endpoint; the MATLAB spectral
band selection includes both endpoints.

DI uses Gaussian-copula partial correlation to estimate
`I(source_past; target_now | target_past)`, with one past target sample and no
analytic finite-sample bias correction. ROI channels are averaged before
rank normalization. NetDI is forward minus reverse DI; actual sample/time lags
are returned. This is a lagged conditional-information statistic.

Modulation indices are calculated per participant before averaging. Zero/zero
pairs return NaN. Repeated-measures correlation selects complete participants
with varying predictors separately for each feature and requires at least three.
Between-participant correlations remove missing pairs per feature. Correction
families are determined by the p-values passed to `bh_fdr`.

Plotting functions return Matplotlib axes; statistics and significance masks are
supplied separately. Function docstrings describe parameters and array shapes.

## References and license

- **DI/Gaussian copula:** Ince et al. (2017), *A statistical framework for
  neuroimaging data analysis based on mutual information estimated via a Gaussian
  copula*, Human Brain Mapping, 38, 1541–1573.
  [doi:10.1002/hbm.23471](https://doi.org/10.1002/hbm.23471);
  [robince/gcmi](https://github.com/robince/gcmi). The scalar implementation here
  does not include upstream GCMI source.
- **LZC:** [Lempel & Ziv (1976)](https://doi.org/10.1109/TIT.1976.1055501),
  [Kaspar & Schuster (1987)](https://doi.org/10.1103/PhysRevA.36.842).
  The C parser translates
  [NeuroKit2 0.2.12](https://github.com/neuropsychology/NeuroKit/blob/0.2.12/neurokit2/complexity/complexity_lempelziv.py);
  cite [Makowski et al. (2021)](https://doi.org/10.3758/s13428-020-01516-y).
- **wPLI:** [Vinck et al. (2011)](https://doi.org/10.1016/j.neuroimage.2011.01.055);
  implementation: [MNE-Connectivity](https://github.com/mne-tools/mne-connectivity).
- **Software/statistics:** [MNE-Python, Gramfort et al. (2013)](https://doi.org/10.3389/fnins.2013.00267),
  [repeated-measures correlation, Bakdash & Marusich (2017)](https://doi.org/10.3389/fpsyg.2017.00456),
  [BH-FDR, Benjamini & Hochberg (1995)](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x),
  [cluster inference, Maris & Oostenveld (2007)](https://doi.org/10.1016/j.jneumeth.2007.03.024).
  The example's CSD uses [MNE's spherical-spline implementation](https://mne.tools/stable/generated/mne.preprocessing.compute_current_source_density.html),
  following [Perrin et al. (1989)](https://doi.org/10.1016/0013-4694(89)90180-6).

See [LICENSE](LICENSE) for the directory-specific terms and the retained MIT
notice for the NeuroKit2-derived LZ76 counter.
