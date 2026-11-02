import numpy as np
from sklearn.datasets import make_spd_matrix

# ============================================================
# Reproducible Synthetic Anomaly Detection Dataset
# 10,000 samples | 10 features | 5% anomalies
#
# Anomaly types:
#   1 = global
#   2 = local
#   3 = dependency
#   4 = clustered
#
# ADSD labels:
#   0 = normal
#   1 = anomaly
# ============================================================

SEED = 42
N_SAMPLES = 10000
N_FEATURES = 10
ANOMALY_RATIO = 0.05

rng = np.random.default_rng(SEED)

n_anomalies = int(N_SAMPLES * ANOMALY_RATIO)
n_normal = N_SAMPLES - n_anomalies

# Divide anomalies equally among the four types.
base = n_anomalies // 4
counts = {
    "global": base,
    "local": base,
    "dependency": base,
    "clustered": n_anomalies - 3 * base
}

# ------------------------------------------------------------
# 1. NORMAL DATA
# ------------------------------------------------------------
# Correlated multivariate Gaussian distribution.
cov = make_spd_matrix(
    n_dim=N_FEATURES,
    random_state=SEED
)

# Convert covariance to a correlation-like matrix to avoid
# unnecessarily large differences in individual feature scales.
std = np.sqrt(np.diag(cov))
cov = cov / np.outer(std, std)

mean = np.zeros(N_FEATURES)

X_normal = rng.multivariate_normal(
    mean=mean,
    cov=cov,
    size=n_normal
)

# ------------------------------------------------------------
# 2. GLOBAL ANOMALIES
# ------------------------------------------------------------
# Samples clearly outside the normal data range.
normal_min = X_normal.min(axis=0)
normal_max = X_normal.max(axis=0)
normal_range = normal_max - normal_min

n_global = counts["global"]

X_global = rng.uniform(
    low=normal_max + 0.5 * normal_range,
    high=normal_max + 1.5 * normal_range,
    size=(n_global, N_FEATURES)
)

# ------------------------------------------------------------
# 3. LOCAL ANOMALIES
# ------------------------------------------------------------
# Located near the normal distribution but generated with
# increased covariance, making them harder to distinguish.
n_local = counts["local"]

X_local = rng.multivariate_normal(
    mean=mean,
    cov=cov * 3.0,
    size=n_local
)

# ------------------------------------------------------------
# 4. DEPENDENCY ANOMALIES
# ------------------------------------------------------------
# Preserve approximate marginal distributions while breaking
# dependencies between features by independently permuting
# feature values sampled from normal observations.
n_dependency = counts["dependency"]

idx = rng.choice(
    n_normal,
    size=n_dependency,
    replace=False
)

X_dependency = X_normal[idx].copy()

for j in range(N_FEATURES):
    X_dependency[:, j] = rng.permutation(X_dependency[:, j])

# ------------------------------------------------------------
# 5. CLUSTERED ANOMALIES
# ------------------------------------------------------------
# Compact anomalous cluster shifted away from the normal centre.
n_clustered = counts["clustered"]

cluster_mean = np.full(N_FEATURES, 3.0)

X_clustered = rng.multivariate_normal(
    mean=cluster_mean,
    cov=cov * 0.25,
    size=n_clustered
)

# ------------------------------------------------------------
# 6. COMBINE DATA
# ------------------------------------------------------------
X_anomaly = np.vstack([
    X_global,
    X_local,
    X_dependency,
    X_clustered
])

X = np.vstack([
    X_normal,
    X_anomaly
])

y = np.concatenate([
    np.zeros(n_normal, dtype=np.int64),
    np.ones(n_anomalies, dtype=np.int64)
])

# Extra metadata used only for analysis/visualisation.
# 0 = normal
# 1 = global
# 2 = local
# 3 = dependency
# 4 = clustered
anomaly_type = np.concatenate([
    np.zeros(n_normal, dtype=np.int64),
    np.full(n_global, 1, dtype=np.int64),
    np.full(n_local, 2, dtype=np.int64),
    np.full(n_dependency, 3, dtype=np.int64),
    np.full(n_clustered, 4, dtype=np.int64)
])

# Shuffle while keeping X, y and anomaly type aligned.
order = rng.permutation(N_SAMPLES)

X = X[order]
y = y[order]
anomaly_type = anomaly_type[order]

# ------------------------------------------------------------
# 7. SAVE
# ------------------------------------------------------------
# Main ADSD-compatible dataset.
np.savez(
    "datasets/synthetic_constructed.npz",
    X=X.astype(np.float64),
    y=y
)

# Separate metadata file for analysis and presentation.
np.savez(
    "datasets/synthetic_constructed_metadata.npz",
    anomaly_type=anomaly_type,
    seed=SEED
)

# ------------------------------------------------------------
# 8. SUMMARY
# ------------------------------------------------------------
print("=== SYNTHETIC DATASET CREATED ===")
print("Seed:", SEED)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Normal:", int((y == 0).sum()))
print("Anomalies:", int((y == 1).sum()))
print("Anomaly ratio: %.2f%%" % (100 * y.mean()))
print()
print("Global anomalies:", int((anomaly_type == 1).sum()))
print("Local anomalies:", int((anomaly_type == 2).sum()))
print("Dependency anomalies:", int((anomaly_type == 3).sum()))
print("Clustered anomalies:", int((anomaly_type == 4).sum()))
print()
print("NaN values:", int(np.isnan(X).sum()))
print("Saved: datasets/synthetic_constructed.npz")
print("Saved: datasets/synthetic_constructed_metadata.npz")
