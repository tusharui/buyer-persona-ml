import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def pca_scatter(ax: Axes, coords: np.ndarray, labels: np.ndarray) -> None:
    ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10",
               alpha=0.6, s=15, edgecolors="none")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Projection (2D)")


def feature_heatmap(ax: Axes, profile: pd.DataFrame) -> None:
    sns.heatmap(profile, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, ax=ax, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Heatmap per Cluster")
    ax.set_ylabel("Cluster")
