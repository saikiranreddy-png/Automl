# supervised.py
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.inspection import permutation_importance
import warnings

from preprocess import build_preprocessor

def run_supervised(df: pd.DataFrame, target: str, task_hint: str = "auto",
                   algorithms: Optional[List[str]] = None, cv: int = 3, random_state: int = 42) -> Dict[str, Any]:
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Decide task
    if task_hint == "auto":
        if pd.api.types.is_float_dtype(y) and y.nunique() > 20:
            task = "regression"
        else:
            task = "classification"
    else:
        task = task_hint

    # Default algorithms
    if algorithms is None:
        if task == "classification":
            algorithms = ["rf", "logistic", "gb"]
        else:
            algorithms = ["rf_reg", "ridge", "gb_reg"]

    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # Build candidate pipelines
    pipelines = {}
    if task == "classification":
        pipelines = {
            "rf": Pipeline([("preprocessor", preprocessor),
                            ("model", RandomForestClassifier(n_estimators=100, random_state=random_state))]),
            "logistic": Pipeline([("preprocessor", preprocessor),
                                  ("model", LogisticRegression(max_iter=2000))]),
            "gb": Pipeline([("preprocessor", preprocessor),
                            ("model", GradientBoostingClassifier(random_state=random_state))])
        }
        scoring = "f1_macro"
        higher_is_better = True
    else:
        pipelines = {
            "rf_reg": Pipeline([("preprocessor", preprocessor),
                                ("model", RandomForestRegressor(n_estimators=100, random_state=random_state))]),
            "ridge": Pipeline([("preprocessor", preprocessor),
                               ("model", Ridge())]),
            "gb_reg": Pipeline([("preprocessor", preprocessor),
                                ("model", GradientBoostingRegressor(random_state=random_state))])
        }
        scoring = "neg_mean_squared_error"
        higher_is_better = False

    cv_results = {}
    for k in algorithms:
        if k not in pipelines:
            warnings.warn(f"Unknown algorithm key {k} - skipping")
            continue
        pipe = pipelines[k]
        try:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
            cv_results[k] = {"mean_cv": float(np.mean(scores)), "scores": scores}
        except Exception as e:
            cv_results[k] = {"error": str(e)}

    # Choose best
    best_key = None
    best_val = None
    for k, v in cv_results.items():
        if "mean_cv" not in v:
            continue
        val = v["mean_cv"]
        if task == "regression" and scoring == "neg_mean_squared_error":
            rmse = float(np.sqrt(-val))
            cmp_val = rmse
            if best_key is None or cmp_val < best_val:
                best_key, best_val = k, cmp_val
        else:
            cmp_val = val
            if best_key is None or cmp_val > best_val:
                best_key, best_val = k, cmp_val

    if best_key is None:
        raise RuntimeError("No algorithm succeeded in cross-validation. Check data and options.")

    # Fit best pipeline on full training split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    best_pipeline = pipelines[best_key]
    best_pipeline.fit(X_train, y_train)
    preds = best_pipeline.predict(X_test)

    if task == "classification":
        final_metric = float(f1_score(y_test, preds, average="macro"))
        metric_name = "F1_macro"
    else:
        final_metric = float(np.sqrt(mean_squared_error(y_test, preds, squared=True)))
        metric_name = "RMSE"

    # Feature importance extraction
    model = best_pipeline.named_steps["model"]
    importance_df = pd.DataFrame({"feature": [], "importance": []})
    try:
        # feature names
        try:
            feat_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
            feat_names = list(feat_names)
        except Exception:
            # fallback
            feat_names = X.columns.tolist()
        if hasattr(model, "feature_importances_"):
            imps = model.feature_importances_
            n = min(len(feat_names), len(imps))
            importance_df = pd.DataFrame({"feature": feat_names[:n], "importance": imps[:n]}).sort_values("importance", ascending=False)
        elif hasattr(model, "coef_"):
            coef = np.array(model.coef_)
            if coef.ndim > 1:
                coef_vals = np.mean(np.abs(coef), axis=0)
            else:
                coef_vals = np.abs(coef)
            n = min(len(feat_names), len(coef_vals))
            importance_df = pd.DataFrame({"feature": feat_names[:n], "importance": coef_vals[:n]}).sort_values("importance", ascending=False)
        else:
            # permutation importance fallback
            perm = permutation_importance(best_pipeline, X_train, y_train, n_repeats=10, random_state=random_state, n_jobs=1)
            imps = perm.importances_mean
            n = min(len(feat_names), len(imps))
            importance_df = pd.DataFrame({"feature": feat_names[:n], "importance": imps[:n]}).sort_values("importance", ascending=False)
    except Exception:
        importance_df = pd.DataFrame({"feature": [], "importance": []})

    result = {
        "task": task,
        "best_algo": best_key,
        "cv_results": cv_results,
        "best_pipeline": best_pipeline,
        "model": model,
        "metric_name": metric_name,
        "metric": final_metric,
        "importance": importance_df
    }
    return result
cd
