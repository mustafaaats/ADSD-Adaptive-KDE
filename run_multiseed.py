import csv
from pathlib import Path

from data_generator import DataGenerator
from baseline.ADSD.run import adsd

datasets = ["thyroid", "seismic_bumps", "synthetic_constructed"]
methods = {"fixed": 1.0, "hybrid": "hybrid"}
seeds = [1, 2, 3, 4, 5]

output = Path("multiseed_results.csv")
fields = ["dataset", "method", "seed", "aucroc", "aucpr"]

completed = set()

if output.exists():
    with output.open(newline="") as f:
        for row in csv.DictReader(f):
            completed.add((row["dataset"], row["method"], int(row["seed"])))

for dataset in datasets:
    for seed in seeds:
        generator = DataGenerator(
            generate_duplicates=True,
            n_samples_threshold=1000
        )
        generator.seed = seed
        generator.dataset = dataset
        data = generator.generator(la=0.05)

        for method, bandwidth in methods.items():
            if (dataset, method, seed) in completed:
                print(f"SKIPPING: {dataset}, {method}, seed={seed}", flush=True)
                continue

            print(f"\\nRUNNING: {dataset}, {method}, seed={seed}", flush=True)

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
                "method": method,
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

print("\\nAll experiments completed.")
