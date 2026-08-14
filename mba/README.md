# Genuine Many-Body Approximation (MBA) MRI Experiment

HC-vs-MCI structural MRI experiment using the
official Many-Body Approximation (MBA) implementation.

## Method

FreeSurfer morphometric features are aggregated into 27 biologically
defined disease-related nodes. Training-fold reference statistics are
used to construct positive disease-evidence representations, which are
organised as 3×3×3 probability tensors.

The vendored official PyMBA implementation is then used to compute the
nested body-order projections

\[
P \rightarrow Q_1,\;Q_2,\;Q_3,
\]

via `mproject.MBA_LBFGS(P, body)`.

For a three-dimensional tensor, \(Q_3=P\) is the complete model.
\(Q_1\) and \(Q_2\) are lower-order information-geometric projections.
Higher-order structure beyond the two-body approximation is examined
using quantities such as

\[
\Delta\theta_3
=
\theta(P)-\theta(Q_2).
\]

The complete distribution \(P=Q_3\) contains first-, second-, and
third-order structure and should not be interpreted as containing
third-order interactions alone.

## Data

- Cohort: 100 participants (61 HC, 39 MCI)
- Biological representation: 27 nodes
- Tensor: 3×3×3
- Input:
  `analysis/freesurfer_all_roi_outputs/all_freesurfer_roi_features.csv`

All reference estimation, imputation, feature selection, model
selection, and threshold optimisation are restricted to training data.

## Official MBA backend

The experiment uses the vendored PyMBA implementation in:

`external/gkazunii_pymba/src`

The MBA optimisation is performed using:

`mproject.MBA_LBFGS(P, body)`

The MBA mathematics is not reimplemented in this package.

## Package

- `run_genuine_mba.py` — command-line entry point
- `experiment.py` — experiment orchestration
- `data.py` — cohort and feature loading
- `node_mapping.py` — biological node construction
- `disease_evidence.py` — disease-evidence embeddings
- `tensor.py` — 3×3×3 tensor construction
- `official_backend.py` — official PyMBA interface
- `representations.py` — Q/theta/residual representations
- `models.py` — classification and feature selection
- `results.py` — summaries and statistical comparisons
- `validation.py` — scientific and numerical validation

## Run

Run from the repository root:

```bash
python -u -m mba.run_genuine_mba \
  --outdir genuine_mba_results \
  --n-repeats 50 \
  --outer-splits 5 \
  --inner-splits 5 \
  --n-jobs 1 \
  --fail-on-nonconvergence