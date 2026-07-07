import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PROCESSED_FILES, PERSONA_MAP, PERSONA_DESCRIPTIONS, BUSINESS_RECOMMENDATIONS
from src.evaluation import validate_clusters
from src.visualization import pca_scatter, feature_heatmap

st.set_page_config(page_title="Buyer Persona ML", layout="wide")
st.title("Buyer Persona ML — Dashboard")


@st.cache_data
def load_data():
    csv_path = PROCESSED_FILES["personas"]
    if csv_path.exists():
        return pd.read_csv(csv_path)
    st.error("No data found. Run the pipeline first: python -m src.pipeline")
    return pd.DataFrame()


df = load_data()

if df.empty:
    st.stop()

feat_cols = [c for c in df.columns if c not in ("CustomerID", "Cluster", "Persona", "PC1", "PC2")]
X = df[feat_cols].values

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "Dataset Overview", "Feature Engineering", "PCA & UMAP",
    "Clustering Results", "Persona Explorer", "Business Recommendations",
])

if page == "Dataset Overview":
    st.header("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", df.shape[0])
    col2.metric("Features", len(feat_cols))
    col3.metric("Clusters", df["Cluster"].nunique())
    col4.metric("Personas", df["Persona"].nunique())

    st.subheader("Sample Data")
    display_cols = [c for c in df.columns if c != "CustomerID"]
    st.dataframe(df[display_cols].head(10), use_container_width=True)
    st.subheader("Descriptive Statistics")
    st.dataframe(df[feat_cols].describe(), use_container_width=True)
    st.subheader("Persona Distribution")
    st.bar_chart(df["Persona"].value_counts())

elif page == "Feature Engineering":
    st.header("Feature Engineering Summary")
    st.subheader("Feature Distributions")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, c in enumerate(feat_cols[:8]):
        axes.flatten()[i].hist(df[c], bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        axes.flatten()[i].set_title(c)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[feat_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, ax=ax, linewidths=0.5)
    st.pyplot(fig)

elif page == "PCA & UMAP":
    st.header("PCA & UMAP Visualization")
    st.subheader("PCA Scatter (2D)")
    fig, ax = plt.subplots(figsize=(10, 7))
    pca_scatter(ax, df[["PC1", "PC2"]].values, df["Cluster"].values)
    st.pyplot(fig)

    st.subheader("PCA Scatter (3D)")
    from sklearn.decomposition import PCA as SkPCA
    pca3 = SkPCA(n_components=3, random_state=42).fit_transform(X)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pca3[:, 0], pca3[:, 1], pca3[:, 2],
                    c=df["Cluster"], cmap="tab10", alpha=0.6, s=15)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.colorbar(sc, ax=ax, label="Cluster", shrink=0.6)
    st.pyplot(fig)

    st.subheader("UMAP Scatter")
    if st.button("Run UMAP (may take a moment)"):
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
        X_umap = reducer.fit_transform(X)
        fig, ax = plt.subplots(figsize=(10, 7))
        sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=df["Cluster"],
                        cmap="tab10", alpha=0.6, s=15)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        st.pyplot(fig)

elif page == "Clustering Results":
    st.header("Clustering Results")
    st.subheader("Validation Metrics")
    val = validate_clusters(X, df["Cluster"].values)
    col1, col2 = st.columns(2)
    col1.metric("Silhouette Score", f"{val['silhouette']:.4f}", help="Higher is better (-1 to 1)")
    col2.metric("Davies-Bouldin Index", f"{val['davies_bouldin']:.4f}", help="Lower is better")

    st.subheader("Feature Heatmap per Cluster")
    profile = df.groupby("Cluster")[feat_cols].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    feature_heatmap(ax, profile)
    st.pyplot(fig)

    st.subheader("Cluster Sizes")
    st.bar_chart(df["Persona"].value_counts())

elif page == "Persona Explorer":
    st.header("Persona Explorer")
    persona = st.selectbox("Select Persona", list(PERSONA_MAP.values()))
    mask = df["Persona"] == persona
    subset = df[mask]
    st.write(f"**{persona}** — {subset.shape[0]} customers ({subset.shape[0]/df.shape[0]*100:.1f}%)")
    st.write(PERSONA_DESCRIPTIONS[persona])

    st.subheader("Average Feature Profile")
    avg = subset[feat_cols].mean().round(3)
    st.dataframe(avg.to_frame().T, use_container_width=True)

    st.subheader("Radar Profile")
    profile = df.groupby("Persona")[feat_cols].mean()
    norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
    from math import pi
    categories = list(norm.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    vals = norm.loc[persona].values.flatten().tolist() + [norm.loc[persona].values[0]]
    ax.plot(angles, vals, linewidth=2, linestyle="solid", color="red")
    ax.fill(angles, vals, alpha=0.1, color="red")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title(persona, fontsize=14, pad=20)
    st.pyplot(fig)

    st.subheader("PCA View (highlighted)")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = np.where(df["Persona"] == persona, "red", "lightgray")
    ax.scatter(df["PC1"], df["PC2"], c=colors, alpha=0.5, s=15)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Customers Matching: {persona}")
    st.pyplot(fig)

else:
    st.header("Business Recommendations")
    st.markdown("Targeted recommendations for each persona based on behavioral analysis.")
    for persona, recs in BUSINESS_RECOMMENDATIONS.items():
        count = int((df["Persona"] == persona).sum())
        with st.expander(f"{persona} ({count} customers, {count/df.shape[0]*100:.1f}%)"):
            st.write(PERSONA_DESCRIPTIONS[persona])
            st.markdown("**Recommendations:**")
            for r in recs:
                st.write(f"- {r}")

    st.subheader("Download Reports")
    from src.config import REPORTS_DIR
    if REPORTS_DIR.is_dir():
        for fname in sorted(REPORTS_DIR.iterdir()):
            with open(fname, "rb") as f:
                st.download_button(
                    label=f"Download {fname.name}",
                    data=f, file_name=fname.name,
                    mime="application/pdf" if fname.suffix == ".pdf" else "text/csv",
                )
