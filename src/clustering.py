"""Clustering module — KMeans + DBSCAN with tuning."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import joblib


def kmeans_optimal_k(X, k_range=range(2, 11), random_state=42):
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))
    best_k = k_range[np.argmax(sil_scores)]
    return best_k, inertias, sil_scores


def kmeans_fit(X, k, random_state=42, save_path=None):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    if save_path:
        joblib.dump(km, save_path)
    return labels, km


def dbscan_k_distance_graph(X, n_neighbors=5):
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    return np.sort(dists[:, -1])


def dbscan_tune(X, eps_values, min_samples_values):
    results = []
    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            results.append({"eps": eps, "min_samples": ms,
                            "n_clusters": n_clusters, "n_noise": n_noise})
    return pd.DataFrame(results)


def dbscan_fit(X, eps=1.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    return labels, n_clusters, n_noise
