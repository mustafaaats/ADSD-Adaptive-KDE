# Adaptive KDE Bandwidth for ADSD Anomaly Detection

An empirical replication and extension of **Anomaly Detection with Score Distribution Discrimination (ADSD)**, presented at KDD 2023.

This project investigates whether adaptive kernel density estimation (KDE) bandwidth selection can improve anomaly detection performance compared with the original fixed-bandwidth approach.

Developed for COMP8240 at Macquarie University.

## Original Work

This project builds upon the official ADSD implementation:

https://github.com/Minqi824/Overlap

The original ADSD method uses score distribution discrimination to separate normal and anomalous observations. We retain its core training framework and investigate an adaptive KDE bandwidth extension.

The original code is distributed under the MIT License. The original copyright and licence notice are preserved in `LICENSE`.

## Proposed Extension

The original implementation uses a fixed KDE bandwidth of 1.0.

Our extension introduces a hybrid bandwidth that combines the original fixed value with a data-driven bandwidth estimate based on Silverman's rule of thumb.

The mixing parameter alpha controls the contribution of the adaptive estimate:

- alpha = 0.0: original fixed bandwidth
- alpha = 0.2: mostly fixed bandwidth
- alpha = 0.5: equal weighting
- alpha = 0.8: mostly adaptive bandwidth
- alpha = 1.0: fully adaptive bandwidth

The adaptive estimate is constrained to a predefined bandwidth range for numerical stability.

The extension is implemented in `baseline/ADSD/fit.py`.

## Datasets

| Dataset | Samples | Features | Anomalies | Anomaly Rate |
|---|---:|---:|---:|---:|
| Thyroid | 3,772 | 6 | 93 | 2.47% |
| Seismic Bumps | 2,584 | 24 | 170 | 6.58% |
| Synthetic Constructed | 10,000 | 10 | 500 | 5.00% |

**Thyroid:** Existing benchmark dataset included in the original ADSD repository.

**Seismic Bumps:** Additional real-world dataset sourced from the UCI Machine Learning Repository. Categorical variables were encoded to produce 24 features.

**Synthetic Constructed:** Reproducibly generated using `synthetic_generator.py` with seed 42. It contains four anomaly types: global, local, dependency and clustered, with 125 anomalies of each type.

The synthetic dataset and its anomaly-type metadata are saved in `datasets/`.

## Experimental Setup

- Model: ADSD with MLP architecture
- Training epochs: 20
- Labelled anomaly ratios: 5%, 10%, 20%
- Random seeds: 1, 2, 3, 4, 5
- Bandwidth mixing parameters: 0.0, 0.2, 0.5, 0.8, 1.0
- Evaluation metrics: ROC-AUC and Average Precision (AP)
- Total experiments: 225

Data splitting is stratified, and feature scaling is fitted using the training partition.

## Installation

The experiments were conducted using Python 3.10.20.

Create a Python environment and install the project dependencies:

```bash
python -m pip install -r requirements.txt

```

The dependency versions reflect the environment used for the experiments. Installation has not yet been verified in a fresh environment.

## Reproducing the Experiments

Run the fixed-bandwidth baseline and alpha=0.2 hybrid experiments:

```bash
python run_multiseed.py
```

Run the remaining alpha ablation experiments:

```bash
python run_alpha_ablation.py
```

Results are saved to `multiseed_results.csv` and `alpha_ablation_results.csv`.

Both scripts skip experiments already recorded in their output CSV files. To repeat the experiments, preserve the existing CSV files elsewhere before running the scripts without those output files.

To regenerate the synthetic dataset:

```bash
python synthetic_generator.py
```

Warning: this command overwrites the existing synthetic dataset and metadata files.


## Results

The original result tables and plots below show the 5% supervision setting. The additional 10% and 20% results are recorded in `label_ratio_results.csv`.

The following values are means across five random seeds.

| Dataset | Alpha | ROC-AUC | Average Precision |
|---|---:|---:|---:|
| Seismic Bumps | 0.0 | 0.5272 | 0.0951 |
| Seismic Bumps | 0.2 | 0.5153 | 0.0914 |
| Seismic Bumps | 0.5 | 0.4939 | 0.0861 |
| Seismic Bumps | 0.8 | 0.4889 | 0.0903 |
| Seismic Bumps | 1.0 | 0.4867 | 0.0874 |
| Synthetic | 0.0 | 0.7362 | 0.5219 |
| Synthetic | 0.2 | 0.7157 | 0.5217 |
| Synthetic | 0.5 | 0.7249 | 0.5158 |
| Synthetic | 0.8 | 0.7350 | 0.5508 |
| Synthetic | 1.0 | 0.7152 | 0.5337 |
| Thyroid | 0.0 | 0.9811 | 0.8138 |
| Thyroid | 0.2 | 0.9719 | 0.8091 |
| Thyroid | 0.5 | 0.9627 | 0.7872 |
| Thyroid | 0.8 | 0.9080 | 0.6833 |
| Thyroid | 1.0 | 0.8495 | 0.4784 |

Full per-seed results are available in the CSV files.

### Key Findings

At 5% supervision, the fixed-bandwidth method achieved the highest mean Average Precision on Thyroid and Seismic Bumps. This does not hold across all supervision levels.

On the synthetic dataset, alpha=0.8 achieved the highest mean AP at all three supervision levels. Paired comparisons with the fixed baseline gave p-values of approximately 0.267, 0.340, and 0.082 at 5%, 10%, and 20%, respectively. None was statistically significant at the 5% level.

Adaptive bandwidth selection did not consistently outperform the fixed-bandwidth baseline. The alpha comparisons are exploratory because the settings were evaluated using test-set outcomes rather than selected on an independent validation set.

## Visualisations

- `aucpr_alpha_comparison.png`: Average Precision across alpha values
- `aucroc_alpha_comparison.png`: ROC-AUC across alpha values
- `synthetic_anomaly_types_pca.png`: PCA visualisation of the synthetic dataset

The alpha comparison plots show mean performance with standard deviation error bars across five seeds.

## Limitations

The evaluation uses three datasets, five random seeds, one model architecture and 20 training epochs.

Performance may depend on dataset characteristics, model configuration and bandwidth selection. The proposed extension has not demonstrated a consistent improvement over the original fixed-bandwidth approach.

## Acknowledgements

This project builds upon the official implementation of *Anomaly Detection with Score Distribution Discrimination* (KDD 2023):

https://github.com/Minqi824/Overlap

The original authors retain copyright over their contributions.

## Licence

The original ADSD source code is distributed under the MIT License. The original copyright and licence notice are preserved in `LICENSE`.



## Additional Supervision Experiments

The complete evaluation contains 225 runs:
3 datasets, 3 labelled-anomaly ratios, 5 bandwidth weights,
and 5 random seeds.

The original scripts cover 75 runs at 5% supervision.
The following script covers 150 additional runs at 10% and 20%:

```bash
python run_label_ratios.py
```

Additional results are saved to `label_ratio_results.csv`.
The script skips configurations already recorded in that file.
For a complete rerun, move the existing result file to a backup
location before running the script.

Bandwidth effects varied across datasets and supervision levels.
The tested adaptive method did not consistently outperform the
fixed baseline. Comparisons are exploratory.


## Auxiliary LLM Evaluation

Two GPT-6 Instant responses were evaluated on 40 synthetic
test observations. Accuracy was 77.5% and 80.0%.
Binary decisions agreed on 87.5% of observations
(Cohen's kappa: 0.7475).

The sample was deliberately balanced, so its precision and AP
are not directly comparable with ADSD results on the full test set.
This auxiliary assessment does not establish bandwidth superiority.

Prompts, responses, scoring code, collection metadata,
and detailed results are available in [llm_judge](llm_judge/README.md).
