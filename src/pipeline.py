"""End-to-end orchestration used by run_all.py."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

from data.generate_data import generate_dataframe, sha256_of_file, write_csv, write_fingerprint

from . import plots
from .config import (
    ARTIFACTS_DIR, DATA_DIR, FEATURES, MODELS_DIR, PLOTS_DIR, ROOT, SYNTHETIC_NOTICE,
    TARGET_CLASSIFICATION, TARGET_REGRESSION,
)
from .modeling import (
    describe_clusters, silhouette_label, train_classification, train_clustering, train_regression,
)
from .preprocessing import clean_dataset, feature_frame, load_dataset, make_split
from .validation import check_leakage, validate_raw_dataset, validate_seed

DEFAULT_NAME = "IGIRANEZA Alain Providence"


def _dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def run(student_id: str, seed: int, student_name: str = DEFAULT_NAME, root: Path = ROOT, verbose: bool = True) -> dict[str, Any]:
    """Recreate data, validate, clean, train all three tasks, write every artifact."""
    say = print if verbose else (lambda *a, **k: None)
    ident = validate_seed(student_id, seed)
    code, regdigits = ident["verification_code"], ident["regdigits"]
    registration = f"{regdigits[:2]}/{regdigits[2:]}"
    artifacts, models_dir, plots_dir = root / "artifacts", root / "models", root / "artifacts" / "plots"
    for d in (artifacts, models_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    say(f"== Musanze Harvest Intelligence Service | {code} ==")
    say(f"Student: {student_name} | Reg: {registration} | seed (LLLL) = {seed}")

    # 1. data ------------------------------------------------------------------
    csv_path = root / "data" / f"AI_A2_{regdigits}.csv"
    raw_df = generate_dataframe(seed)
    sha = write_csv(raw_df, csv_path)
    write_fingerprint(artifacts / "data_fingerprint.txt", csv_path, sha, len(raw_df), seed)
    say(f"[1/6] Dataset written: {csv_path.name} rows={len(raw_df)}")
    say(f"      SHA-256 fingerprint: {sha}")

    # 2. validate + clean ---------------------------------------------------------
    raw = load_dataset(csv_path)
    check_leakage(FEATURES)
    validation = validate_raw_dataset(raw, seed)
    clean, cleaning = clean_dataset(raw)
    say(f"[2/6] Validation passed | missing={validation['missing_by_column']} duplicates={validation['n_duplicate_rows']} "
        f"-> clean rows={cleaning['rows_after']}")

    # 3. split ---------------------------------------------------------------------
    train, test = make_split(clean, seed)
    X_tr, X_te = feature_frame(train), feature_frame(test)
    say(f"[3/6] Split (random_state={seed}): train={len(train)} test={len(test)}")

    # 4. regression ----------------------------------------------------------------
    reg_res, reg_sel, reg_reason, reg_fit = train_regression(X_tr, train[TARGET_REGRESSION], X_te, test[TARGET_REGRESSION], seed)
    _dump({"verification_code": code, "target": TARGET_REGRESSION, "test_rows": len(test), "models": reg_res,
           "selected_model": reg_sel, "selection_reason": reg_reason}, artifacts / "regression_metrics.json")
    say(f"[4/6] Regression selected: {reg_sel} | test MAE={reg_res[reg_sel]['mae']:.4f} RMSE={reg_res[reg_sel]['rmse']:.4f} R2={reg_res[reg_sel]['r2']:.4f}")

    # 5. classification --------------------------------------------------------------
    clf_res, clf_sel, clf_reason, clf_fit = train_classification(X_tr, train[TARGET_CLASSIFICATION], X_te, test[TARGET_CLASSIFICATION], seed)
    _dump({"verification_code": code, "target": TARGET_CLASSIFICATION, "test_rows": len(test),
           "positive_class": "high_risk = 1 (yield per hectare below the median)", "models": clf_res,
           "selected_model": clf_sel, "selection_reason": clf_reason}, artifacts / "classification_metrics.json")
    s = clf_res[clf_sel]
    say(f"[5/6] Classification selected: {clf_sel} | P={s['precision']:.4f} R={s['recall']:.4f} F1={s['f1']:.4f} AUC={s['roc_auc']:.4f}")

    # 6. clustering --------------------------------------------------------------------
    k_scores, k, cl_pipe = train_clustering(X_tr, seed)
    lab_tr, lab_te = cl_pipe.predict(X_tr), cl_pipe.predict(X_te)
    profiles = describe_clusters(train, lab_tr, lab_te)
    best_sil = max(x["silhouette"] for x in k_scores)
    quality = silhouette_label(best_sil)
    interp = (
        f"K={k} has the highest silhouette ({best_sil:.4f}), which indicates {quality} (Kaufman-Rousseeuw scale). "
        "Clusters were built from the seven input features only; yield and risk are shown for description, not used."
    )
    _dump({"verification_code": code, "features_used": FEATURES, "scaling": "StandardScaler + one-hot season (fit on training rows)",
           "k_scores": k_scores, "selected_k": k, "selection_rule": "highest silhouette score among K=2..6",
           "cluster_profiles": profiles, "interpretation": interp}, artifacts / "clustering_report.json")
    say(f"[6/6] Clustering: selected K={k} (silhouette {best_sil:.4f}, {quality})")

    # artifacts ---------------------------------------------------------------------------
    reg_pred = reg_fit[reg_sel].predict(X_te)
    prob = clf_fit[clf_sel].predict_proba(X_te)[:, 1]
    cls_pred = (prob >= 0.5).astype(int)
    out = pd.DataFrame({
        "row_id": test.index.to_numpy(),
        "actual_yield_tons": test[TARGET_REGRESSION].to_numpy(),
        "predicted_yield_tons": reg_pred,
        "yield_error_tons": reg_pred - test[TARGET_REGRESSION].to_numpy(),
        "abs_yield_error_tons": np.abs(reg_pred - test[TARGET_REGRESSION].to_numpy()),
        "actual_high_risk": test[TARGET_CLASSIFICATION].to_numpy(),
        "predicted_high_risk": cls_pred,
        "prob_high_risk": prob,
        "classification_correct": (cls_pred == test[TARGET_CLASSIFICATION].to_numpy()).astype(int),
        "cluster": lab_te,
    }).sort_values("row_id")
    out.to_csv(artifacts / "test_predictions.csv", index=False, float_format="%.4f", lineterminator="\n")

    plots.plot_residuals(test[TARGET_REGRESSION], reg_pred, plots_dir / "residuals.png", code, reg_sel)
    plots.plot_confusion_matrix(s["confusion_matrix"]["matrix"], plots_dir / "confusion_matrix.png", code, clf_sel)
    prep = cl_pipe.named_steps["prep"]
    plots.plot_clusters(prep.transform(X_tr), lab_tr, prep.transform(X_te), lab_te,
                        cl_pipe.named_steps["kmeans"].cluster_centers_, plots_dir / "clusters.png", code, k)

    # models ----------------------------------------------------------------------------------
    joblib.dump(reg_fit[reg_sel], models_dir / "regressor.joblib")
    joblib.dump(clf_fit[clf_sel], models_dir / "classifier.joblib")
    joblib.dump(cl_pipe, models_dir / "clusterer.joblib")
    metadata = {
        "verification_code": code, "student_name": student_name, "student_id": ident["student_id"],
        "registration_number": registration, "seed": seed, "dataset_sha256": sha, "dataset_file": csv_path.name,
        "rows_raw": len(raw), "rows_clean": len(clean), "train_rows": len(train), "test_rows": len(test),
        "features": FEATURES,
        "selected_models": {"regression": reg_sel, "classification": clf_sel, "clustering": f"kmeans(K={k})"},
        "test_metrics": {"regression": {m: reg_res[reg_sel][m] for m in ("mae", "rmse", "r2")},
                         "classification": {m: s[m] for m in ("precision", "recall", "f1", "roc_auc")}},
        "n_clusters": k, "cluster_profiles": profiles,
        "library_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__, "joblib": joblib.__version__},
        "notice": SYNTHETIC_NOTICE,
    }
    _dump(metadata, models_dir / "metadata.json")
    _dump({"verification_code": code, "validation": validation, "cleaning": cleaning,
           "split": {"random_state": seed, "test_size": 0.2, "stratified_on": TARGET_CLASSIFICATION,
                     "train_rows": len(train), "test_rows": len(test)}, "sha256": sha}, artifacts / "run_summary.json")

    files = sorted(str(p.relative_to(root)).replace("\\", "/") for d in (artifacts, models_dir) for p in d.rglob("*") if p.is_file())
    say("\nGenerated artifacts:")
    for f in files:
        say(f"  - {f}")
    say(f"\nVerification code: {code}  |  Fingerprint: {sha}")
    return {"verification_code": code, "sha256": sha, "files": files}
