import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from typing import Optional

from src.config import RANDOM_STATE, OPTUNA_N_TRIALS


def tune_kmeans_optuna(X: np.ndarray, n_trials: int = None) -> tuple[int, float]:
    import optuna
    n_trials = n_trials or OPTUNA_N_TRIALS

    def objective(trial):
        k = trial.suggest_int("n_clusters", 2, 10)
        init = trial.suggest_categorical("init", ["k-means++", "random"])
        km = KMeans(n_clusters=k, init=init, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        return silhouette_score(X, labels)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_k = int(study.best_params["n_clusters"])
    best_sil = float(study.best_value)
    return best_k, best_sil, study.best_params


def tune_pca_optuna(X: np.ndarray, n_trials: int = None) -> tuple[PCA, float]:
    import optuna
    n_trials = max(n_trials or OPTUNA_N_TRIALS // 2, 10)

    def objective(trial):
        n_components = trial.suggest_int("n_components", 2, min(X.shape[1], 10))
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X)
        k = trial.suggest_int("k", 2, 8)
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_pca)
        return silhouette_score(X_pca, labels)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_n = int(study.best_params["n_components"])
    best_k = int(study.best_params["k"])
    pca = PCA(n_components=best_n, random_state=RANDOM_STATE)
    pca.fit(X)
    return pca, best_k, study.best_params


def tune_dbscan_optuna(X: np.ndarray, n_trials: int = None) -> tuple[float, int, dict]:
    import optuna
    n_trials = n_trials or OPTUNA_N_TRIALS // 2

    def objective(trial):
        eps = trial.suggest_float("eps", 0.1, 3.0)
        min_samples = trial.suggest_int("min_samples", 2, 20)
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        if n_clusters < 2 or n_noise > len(X) * 0.5:
            return -1.0
        return silhouette_score(X, labels)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params["eps"], int(study.best_params["min_samples"]), study.best_params
