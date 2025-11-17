"""Unsupervised learning helpers for the vibration dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from umap import UMAP


@dataclass
class UnsupervisedResult:
    """Stores the dimensionality reduction output and cluster assignments."""

    embedding: np.ndarray
    clusters: np.ndarray
    reducer: str
    cluster_method: str


class UnsupervisedModel:
    """Runs PCA/UMAP and clustering on engineered features."""

    def __init__(
        self,
        n_components: int = 2,
        clustering_k: int = 3,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.clustering_k = clustering_k
        self.random_state = random_state

    def run_pca(self, features: pd.DataFrame) -> np.ndarray:
        """Compute a PCA embedding."""
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        return pca.fit_transform(features)

    def run_umap(self, features: pd.DataFrame) -> np.ndarray:
        """Compute a UMAP embedding."""
        reducer = UMAP(
            n_components=self.n_components,
            random_state=self.random_state,
            n_neighbors=30,
            min_dist=0.1,
        )
        return reducer.fit_transform(features)

    def run_clustering(self, features: pd.DataFrame) -> np.ndarray:
        """Run KMeans clustering to obtain discrete assignments."""
        model = KMeans(
            n_clusters=self.clustering_k,
            n_init="auto",
            random_state=self.random_state,
        )
        return model.fit_predict(features)

    def analyze(
        self,
        features: pd.DataFrame,
        method: str = "umap",
        run_clusters: bool = True,
    ) -> UnsupervisedResult:
        """High-level helper to generate embeddings and clusters."""
        if method == "pca":
            embedding = self.run_pca(features)
            reducer_name = "pca"
        else:
            embedding = self.run_umap(features)
            reducer_name = "umap"
        clusters = self.run_clustering(features) if run_clusters else np.array([])
        return UnsupervisedResult(
            embedding=embedding,
            clusters=clusters,
            reducer=reducer_name,
            cluster_method="kmeans" if run_clusters else "none",
        )
