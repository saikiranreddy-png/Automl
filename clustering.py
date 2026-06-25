# clustering.py
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

from preprocess import build_preprocessor

def run_clustering(df: pd.DataFrame, algorithms: Optional[List[str]] = None, n_clusters: int = 3, random_state: int = 42) -> Dict[str, Any]:
    X = df.copy()
    preprocessor, num_cols, cat_cols = build_preprocessor(X)
    preprocess_pipe = Pipeline([("preproc", preprocessor)])
    X_trans = preprocess_pipe.fit_transform(X)
    # If sparse matrix returned, convert to dense
    try:
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
    except Exception:
        pass

    if algorithms is None:
        algorithms = ["kmeans", "agglo", "dbscan"]

    results = {}
    best_key = None
    best_score = None
    best_labels = None
    best_model = None

    for key in algorithms:
        try:
            if key == "kmeans":
                model = KMeans(n_clusters=n_clusters, random_state=random_state)
                labels = model.fit_predict(X_trans)
            elif key == "agglo":
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(X_trans)
            elif key == "dbscan":
                model = DBSCAN()
                labels = model.fit_predict(X_trans)
            else:
                continue

            unique_labels = np.unique(labels)
            if len(unique_labels) <= 1 or len(unique_labels) >= X_trans.shape[0]:
                score = -1.0
            else:
                score = float(silhouette_score(X_trans, labels))
            results[key] = {"silhouette": score, "labels": labels}
            if best_key is None or score > (best_score if best_score is not None else -999):
                best_key, best_score, best_labels, best_model = key, score, labels, model
        except Exception as e:
            results[key] = {"error": str(e)}
            continue

    if best_key is None:
        raise RuntimeError("No clustering algorithm succeeded.")

    return {
        "task": "clustering",
        "best_algo": best_key,
        "best_model": best_model,
        "clusters": best_labels,
        "metric_name": "Silhouette Score",
        "metric": best_score,
        "algo_results": results
    }
