import csv
from pathlib import Path

from data_generator import DataGenerator
from baseline.ADSD.run import adsd

datasets = ["thyroid", "seismic_bumps", "synthetic_constructed"]
alphas = [0.5, 0.8, 1.0]
seeds = [1, 2, 3, 4, 5]

output = Path("alpha_ablation_results.csv")
fields = ["dataset", "alpha", "seed", "aucroc", "aucpr"]

completed = set()

if output.exists():
    with output.open(newline="") as f:
        for row in csv.DictReader(f):
            completed.add((row["dataset"], float(row["alpha"]), int(row["seed"])))

for dataset in datasets:
    for seed in seeds:
        generator = DataGenerator(
            generate_duplicates=True,
            n_samples_threshold=1000
        )
        generator.seed = seed
        generator.dataset = dataset
        data = generator.generator(la=0.05)

        for alpha in alphas:
            if (dataset, alpha, seed) in completed:
                print(f"SKIPPING: {dataset}, alpha={alpha}, seed={seed}", flush=True)
                continue

            print(f"\nRUNNING: {dataset}, alpha={alpha}, seed={seed}", flush=True)

            bandwidth = f"hybrid:{alpha}"

            model = adsd(
                seed=seed,
                architecture="MLP",
                loss_name="ADSD",
                epochs=20,
                bw_u=bandwidth,
                bw_a=bandwidth
            )

            result = model.fit2test(data)

            row = {
                "dataset": dataset,
                "alpha": alpha,
                "seed": seed,
                "aucroc": result["aucroc"],
                "aucpr": result["aucpr"]
            }

            write_header = not output.exists() or output.stat().st_size == 0

            with output.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

            print(f"RESULT: {row}", flush=True)

print("\nAll alpha ablation experiments completed.")
