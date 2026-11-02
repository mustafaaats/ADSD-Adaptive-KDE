import csv
from pathlib import Path

import numpy as np

from data_generator import DataGenerator
from baseline.ADSD.run import adsd


DATASETS = [
    "thyroid",
    "seismic_bumps",
    "synthetic_constructed",
]
LABEL_RATIOS = [0.10, 0.20]
ALPHAS = [0.0, 0.2, 0.5, 0.8, 1.0]
SEEDS = [1, 2, 3, 4, 5]

OUTPUT = Path("label_ratio_results.csv")
FIELDS = [
    "dataset",
    "label_ratio",
    "alpha",
    "seed",
    "n_labelled",
    "aucroc",
    "aucpr",
]

completed = set()

if OUTPUT.exists() and OUTPUT.stat().st_size > 0:
    with OUTPUT.open(newline="") as handle:
        for row in csv.DictReader(handle):
            completed.add((
                row["dataset"],
                float(row["label_ratio"]),
                float(row["alpha"]),
                int(row["seed"]),
            ))

total = (
    len(DATASETS)
    * len(LABEL_RATIOS)
    * len(ALPHAS)
    * len(SEEDS)
)

for dataset in DATASETS:
    for seed in SEEDS:
        reference_test = None
        previous_labelled = None

        # Include 5% here only to check consistency.
        # No 5% model is trained by this script.
        for ratio in [0.05] + LABEL_RATIOS:
            generator = DataGenerator(
                generate_duplicates=True,
                n_samples_threshold=1000,
            )
            generator.seed = seed
            generator.dataset = dataset
            data = generator.generator(la=ratio)

            # Changing supervision must preserve the test set.
            current_test = (
                data["X_test"].copy(),
                data["y_test"].copy(),
            )

            if reference_test is None:
                reference_test = current_test
            else:
                np.testing.assert_array_equal(
                    reference_test[0], current_test[0]
                )
                np.testing.assert_array_equal(
                    reference_test[1], current_test[1]
                )

            # Verify that smaller labelled pools are nested
            # within larger pools under the original sampler.
            labelled = set(
                np.flatnonzero(data["y_train"] == 1).tolist()
            )

            if previous_labelled is not None:
                if not previous_labelled.issubset(labelled):
                    raise RuntimeError(
                        "Labelled pools are not nested: "
                        f"{dataset}, seed={seed}, ratio={ratio}. "
                        "Stop and inspect sampling before continuing."
                    )

            previous_labelled = labelled

            if ratio == 0.05:
                continue

            for alpha in ALPHAS:
                key = (dataset, ratio, alpha, seed)

                if key in completed:
                    print(f"SKIPPING: {key}", flush=True)
                    continue

                print(
                    f"\nRUNNING: dataset={dataset}, "
                    f"ratio={ratio}, alpha={alpha}, "
                    f"seed={seed}, labelled={len(labelled)}",
                    flush=True,
                )

                bandwidth = (
                    1.0 if alpha == 0.0
                    else f"hybrid:{alpha}"
                )

                model = adsd(
                    seed=seed,
                    architecture="MLP",
                    loss_name="ADSD",
                    epochs=20,
                    bw_u=bandwidth,
                    bw_a=bandwidth,
                )

                result = model.fit2test(data)

                row = {
                    "dataset": dataset,
                    "label_ratio": ratio,
                    "alpha": alpha,
                    "seed": seed,
                    "n_labelled": len(labelled),
                    "aucroc": result["aucroc"],
                    "aucpr": result["aucpr"],
                }

                write_header = (
                    not OUTPUT.exists()
                    or OUTPUT.stat().st_size == 0
                )

                with OUTPUT.open("a", newline="") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=FIELDS
                    )
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)

                completed.add(key)
                print(f"RESULT: {row}", flush=True)
                print(
                    f"Completed: {len(completed)}/{total}",
                    flush=True,
                )

print("\nAll additional label-ratio experiments completed.")
