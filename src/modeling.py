"""Regression, classification and clustering: candidates, evaluation, selection."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .config import (
    BINARY_FEATURES,
    CV_FOLDS,
    K_RANGE,
    NUMERIC_FEATURES,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
)
from .preprocessing import build_preprocessor


def _pipe(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


# ------------------------------------------------------------------ regression
def regression_candidates(seed: int) -> dict[str, Any]:
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }


def regression_scores(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_regression(X_train, y_train, X_test, y_test, seed: int):
    """Fit baseline + candidates; select by 5-fold CV RMSE on the training set only."""
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    results: dict[str, Any] = {}
    fitted: dict[str, Pipeline] = {}
    for name, model in regression_candidates(seed).items():
        pipe = _pipe(model)
        cv_rmse = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
        pipe.fit(X_train, y_train)
        scores = regression_scores(y_test, pipe.predict(X_test))
        scores["cv_rmse_mean"] = float(cv_rmse.mean())
        scores["cv_rmse_std"] = float(cv_rmse.std())
        results[name] = scores
        fitted[name] = pipe

    candidates = [n for n in results if n != "mean_baseline"]
    selected = min(candidates, key=lambda n: results[n]["cv_rmse_mean"])
    sel, base = results[selected], results["mean_baseline"]
    reason = (
        f"{selected} has the lowest 5-fold cross-validated RMSE on the training set "
        f"({sel['cv_rmse_mean']:.4f} t) among the candidates; it was chosen before looking at the test set. "
        f"On the held-out test set it reaches MAE {sel['mae']:.4f}, RMSE {sel['rmse']:.4f}, R2 {sel['r2']:.4f} "
        f"versus the mean baseline's RMSE {base['rmse']:.4f} and R2 {base['r2']:.4f}."
    )
    return results, selected, reason, fitted


# -------------------------------------------------------------- classification
def classification_candidates(seed: int) -> dict[str, Any]:
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }


def classification_scores(y_true, y_pred, y_prob) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    both_classes = len(np.unique(y_true)) == 2
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else None,
        "confusion_matrix": {
            "labels": ["not_high_risk(0)", "high_risk(1)"],
            "matrix": cm.tolist(),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
    }


def train_classification(X_train, y_train, X_test, y_test, seed: int):
    """Fit baseline + candidates; select by 5-fold stratified CV F1 on the training set."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    results: dict[str, Any] = {}
    fitted: dict[str, Pipeline] = {}
    for name, model in classification_candidates(seed).items():
        pipe = _pipe(model)
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        scores = classification_scores(y_test, pred, prob)
        scores["cv_f1_mean"] = float(cv_f1.mean())
        scores["cv_f1_std"] = float(cv_f1.std())
        results[name] = scores
        fitted[name] = pipe

    candidates = [n for n in results if n != "majority_baseline"]
    selected = max(candidates, key=lambda n: results[n]["cv_f1_mean"])
    sel = results[selected]
    reason = (
        f"{selected} has the highest 5-fold stratified cross-validated F1 on the training set "
        f"({sel['cv_f1_mean']:.4f}); chosen before looking at the test set. Test result: precision "
        f"{sel['precision']:.4f}, recall {sel['recall']:.4f}, F1 {sel['f1']:.4f}, ROC-AUC {sel['roc_auc']:.4f}. "
        f"The majority baseline cannot find any high-risk farm (recall {results['majority_baseline']['recall']:.4f})."
    )
    return results, selected, reason, fitted


# ------------------------------------------------------------------ clustering
CLUSTER_PHRASES = {
    "rainfall_mm": ("higher rainfall", "lower rainfall"),
    "avg_temp_c": ("warmer temperature", "cooler temperature"),
    "soil_moisture_pct": ("wetter soil", "drier soil"),
    "fertilizer_kg": ("heavier fertilizer use", "lighter fertilizer use"),
    "field_size_ha": ("larger fields", "smaller fields"),
}


def silhouette_label(score: float) -> str:
    """Kaufman & Rousseeuw rule of thumb."""
    if score > 0.70:
        return "strong structure"
    if score > 0.50:
        return "reasonable structure"
    if score > 0.25:
        return "weak structure"
    return "no substantial structure"


def train_clustering(X_train: pd.DataFrame, seed: int):
    """Scale inputs, test K=2..6, pick K by silhouette. Targets are never used."""
    prep = build_preprocessor()
    Z = prep.fit_transform(X_train)
    scores = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(Z)
        scores.append({"k": int(k), "silhouette": float(silhouette_score(Z, labels)), "inertia": float(km.inertia_)})
    best = max(scores, key=lambda s: s["silhouette"])
    k = best["k"]
    pipe = Pipeline([("prep", build_preprocessor()), ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed))])
    pipe.fit(X_train)
    return scores, k, pipe


def describe_clusters(train_df: pd.DataFrame, labels: np.ndarray, test_labels: np.ndarray) -> list[dict[str, Any]]:
    """Plain-language profile per cluster (outcomes are descriptive only, not clustering inputs)."""
    overall_mean = train_df[NUMERIC_FEATURES].mean()
    overall_std = train_df[NUMERIC_FEATURES].std()
    profiles = []
    for c in sorted(np.unique(labels)):
        sub = train_df[labels == c]
        center = {f: float(sub[f].mean()) for f in NUMERIC_FEATURES}
        support_share = float(sub[BINARY_FEATURES[0]].mean())
        season_a_share = float((sub["season"] == "A").mean())
        phrases = []
        for f in NUMERIC_FEATURES:
            z = (center[f] - overall_mean[f]) / overall_std[f]
            if z >= 0.5:
                phrases.append(CLUSTER_PHRASES[f][0])
            elif z <= -0.5:
                phrases.append(CLUSTER_PHRASES[f][1])
        if support_share >= 0.7:
            phrases.append("mostly cooperative-supported")
        elif support_share <= 0.3:
            phrases.append("mostly unsupported")
        if season_a_share >= 0.7:
            phrases.append("mostly season A")
        elif season_a_share <= 0.3:
            phrases.append("mostly season B")
        summary = ", ".join(phrases) if phrases else "close to the cooperative average on every measured condition"
        profiles.append(
            {
                "cluster": int(c),
                "n_train": int(len(sub)),
                "n_test": int((test_labels == c).sum()),
                "center_original_units": {f: round(center[f], 4) for f in NUMERIC_FEATURES},
                "cooperative_support_share": round(support_share, 4),
                "season_A_share": round(season_a_share, 4),
                "mean_yield_tons_descriptive": round(float(sub[TARGET_REGRESSION].mean()), 4),
                "high_risk_rate_descriptive": round(float(sub[TARGET_CLASSIFICATION].mean()), 4),
                "profile": f"Cluster {int(c)} ({len(sub)} training farms): {summary}.",
            }
        )
    return profiles
