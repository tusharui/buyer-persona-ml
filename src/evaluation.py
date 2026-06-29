"""Evaluation module — cluster validation metrics."""

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.metrics.pairwise import euclidean_distances


def validate_clusters(X, labels):
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return {"silhouette": sil, "davies_bouldin": db}


def silhouette_plot_data(X, labels):
    return silhouette_samples(X, labels)


def intra_inter_distances(X, labels):
    dist = euclidean_distances(X)
    intra, inter = [], []
    for i in sorted(set(labels)):
        mask = labels == i
        d = dist[np.ix_(mask, mask)]
        intra.append(d[d > 0].mean())
    for i in sorted(set(labels)):
        for j in sorted(set(labels)):
            if i < j:
                m_i, m_j = labels == i, labels == j
                inter.append(dist[np.ix_(m_i, m_j)].mean())
    return {"intra": intra, "inter_mean": np.mean(inter) if inter else 0}


def cluster_profiles(df, feature_cols, cluster_col="Cluster"):
    return df.groupby(cluster_col)[feature_cols].mean()


def cluster_stability_score(X, labels, n_splits=5, sample_frac=0.8, random_state=42):
    """Evaluate cluster stability by repeatedly subsampling and comparing labels."""
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.RandomState(random_state)
    n = len(X)
    scores = []
    for _ in range(n_splits):
        idx = rng.choice(n, int(n * sample_frac), replace=False)
        sub_labels = labels[idx]
        # Refit KMeans on subsample and predict on same subsample
        from sklearn.cluster import KMeans
        n_clusters = len(set(labels))
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        refit_labels = km.fit_predict(X[idx])
        scores.append(adjusted_rand_score(sub_labels, refit_labels))
    return {"mean_ari": np.mean(scores), "std_ari": np.std(scores), "scores": scores}
