"""Read-only experiment checks; run from a disposable repository checkout."""
from pathlib import Path
import ast
import csv
import itertools
import json
import os
import platform
import runpy
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_generator import DataGenerator
from baseline.ADSD.run import adsd

root = Path.cwd()
print('ENVIRONMENT', sys.version, platform.platform(), platform.machine(), flush=True)
for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'tensorflow']:
    from importlib.metadata import version
    print(name, version(name), flush=True)

for p in root.rglob('*.py'):
    if '.git' not in p.parts:
        ast.parse(p.read_text(), filename=str(p))
print('PASS: Python source syntax and model imports', flush=True)

expected_shapes = {'thyroid': (3772, 6, 93), 'seismic_bumps': (2584, 24, 170), 'synthetic_constructed': (10000, 10, 500)}
for ds, (n, features, anomalies) in expected_shapes.items():
    d = np.load(root / 'datasets' / (ds + '.npz'))
    X, y = d['X'], d['y']
    assert X.shape == (n, features) and y.shape == (n,)
    assert np.isfinite(X).all() and set(np.unique(y)) == {0, 1}
    assert int(y.sum()) == anomalies
    if ds == 'synthetic_constructed':
        types = np.load(root / 'datasets/synthetic_constructed_metadata.npz')['anomaly_type']
        assert np.array_equal(types > 0, y == 1)
        assert [(types == k).sum() for k in range(5)] == [9500, 125, 125, 125, 125]
    for seed in range(1, 6):
        reference = None
        previous = set()
        for ratio in [.05, .10, .20]:
            data = DataGenerator(seed=seed, dataset=ds, generate_duplicates=True).generator(la=ratio)
            test = (data['X_test'], data['y_test'])
            if reference is not None:
                np.testing.assert_array_equal(reference[0], test[0])
                np.testing.assert_array_equal(reference[1], test[1])
            reference = test
            labelled = set(np.flatnonzero(data['y_train'] == 1))
            assert previous.issubset(labelled)
            _, _, truth_train, _ = train_test_split(X, y, test_size=.3, stratify=y, random_state=seed)
            assert len(labelled) == int(ratio * truth_train.sum())
            assert data['X_train'].min() >= -1e-10 and data['X_train'].max() <= 1 + 1e-10
            previous = labelled
print('PASS: all datasets, metadata, five seeds, supervision counts, nested pools and matching test sets', flush=True)

old = pd.read_csv(root / 'multiseed_results.csv')
old['alpha'] = old['method'].map({'fixed': 0., 'hybrid': .2})
old['label_ratio'] = .05
ablation = pd.read_csv(root / 'alpha_ablation_results.csv')
ablation['label_ratio'] = .05
new = pd.read_csv(root / 'label_ratio_results.csv')
all_runs = pd.concat([old, ablation, new], ignore_index=True)
keys = ['dataset', 'label_ratio', 'alpha', 'seed']
expected = set(itertools.product(expected_shapes, [.05, .1, .2], [0., .2, .5, .8, 1.], range(1, 6)))
assert len(all_runs) == 225 and not all_runs.duplicated(keys).any()
assert set(all_runs[keys].itertuples(index=False, name=None)) == expected
assert np.isfinite(all_runs[['aucroc', 'aucpr']]).all().all()
assert all_runs[['aucroc', 'aucpr']].ge(0).all().all() and all_runs[['aucroc', 'aucpr']].le(1).all().all()
print('PASS: complete 225-run grid, unique keys and valid metrics', flush=True)

# Execute the generator only in an isolated directory: original data remain untouched.
with tempfile.TemporaryDirectory(prefix='comp8240_synthetic_') as folder:
    temp = Path(folder)
    (temp / 'datasets').mkdir()
    os.chdir(temp)
    try:
        runpy.run_path(str(root / 'synthetic_generator.py'), run_name='__main__')
    finally:
        os.chdir(root)
    for name in ['synthetic_constructed.npz', 'synthetic_constructed_metadata.npz']:
        original = np.load(root / 'datasets' / name)
        regenerated = np.load(temp / 'datasets' / name)
        assert set(original.files) == set(regenerated.files)
        for key in original.files:
            np.testing.assert_allclose(original[key], regenerated[key], rtol=1e-10, atol=1e-10)
print('PASS: synthetic generator reproduces archived arrays', flush=True)

saved_summary = json.loads((root / 'llm_judge/evaluation_summary.json').read_text())
subprocess.run([sys.executable, '-B', 'llm_judge/score_judge.py', 'llm_judge/response_1.txt', 'llm_judge/response_2.txt'], check=True)
rescored = json.loads((root / 'llm_judge/evaluation_summary.json').read_text())
for run in ['run_1', 'run_2']:
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'ROC_AUC', 'AP']:
        np.testing.assert_allclose(rescored[run][metric], saved_summary[run][metric], rtol=0, atol=1e-12)
assert rescored['repeat_reliability'] == saved_summary['repeat_reliability']
with tempfile.TemporaryDirectory(prefix='comp8240_prompt_') as folder:
    subprocess.run([sys.executable, '-B', 'llm_judge/prepare_judge.py', '--output', folder], check=True)
    for name in ['judge_prompt_plain.txt', 'answer_key_private.json']:
        assert (root / 'llm_judge' / name).read_bytes() == (Path(folder) / name).read_bytes()
print('PASS: LLM metrics, prompt and answer-key reconstruction', flush=True)

# Representative end-to-end replays, including both fixed and hybrid losses.
for ds, ratio, alpha in [('thyroid', .05, 0.), ('synthetic_constructed', .20, 0.), ('synthetic_constructed', .20, .8)]:
    seed = 1
    print(f'REPLAY: {ds}, ratio={ratio}, alpha={alpha}, seed={seed}', flush=True)
    data = DataGenerator(seed=seed, dataset=ds, generate_duplicates=True).generator(la=ratio)
    bandwidth = 1.0 if alpha == 0 else f'hybrid:{alpha}'
    model = adsd(seed=seed, architecture='MLP', loss_name='ADSD', epochs=20, bw_u=bandwidth, bw_a=bandwidth)
    result = model.fit2test(data)
    row = all_runs[(all_runs.dataset == ds) & (all_runs.label_ratio == ratio) & (all_runs.alpha == alpha) & (all_runs.seed == seed)].iloc[0]
    for metric in ['aucroc', 'aucpr']:
        value = float(result[metric])
        assert np.isfinite(value) and 0 <= value <= 1
        print(f'{metric}: rerun={value:.10f}, archived={row[metric]:.10f}, difference={value-row[metric]:+.10f}', flush=True)
        np.testing.assert_allclose(value, row[metric], rtol=0, atol=1e-6,
                                   err_msg='Replay differs: inspect environment and code before claiming numerical replication.')
print('PASS: three representative 20-epoch experiments match archived scores within 1e-6', flush=True)
print('ALL FINAL CHECKS PASSED', flush=True)
