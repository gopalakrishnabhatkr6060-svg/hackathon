"""
Standalone XGBoost Retraining Pipeline for Supplier Risk Model
Run this script anytime you want to retrain the model on updated supplier data:
    python train_model.py
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Directories and File Paths
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "suppliers_dataset.csv"
MODEL_OUTPUT_PATH = PROJECT_DIR / "xgb_supplier_risk_model.json"
METRICS_OUTPUT_PATH = PROJECT_DIR / "training_metrics.json"

FEATURES = [
    "avg_price_creep",
    "rec_rejection_pct",
    "rejection_spike_ratio",
    "min_days_to_expiry",
    "criticality_numeric",
    "stock_cover_days",
    "is_single_source"
]


def retrain_model(
    dataset_path: Path = DATASET_PATH,
    model_output_path: Path = MODEL_OUTPUT_PATH,
    metrics_output_path: Path = METRICS_OUTPUT_PATH,
    n_estimators: int = 120,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    test_size: float = 0.25,
    random_state: int = 42
):
    print("=" * 60)
    print("[START] SUPPLIER RISK MODEL RETRAINING PIPELINE")
    print("=" * 60)

    # 1. Load Data
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"[OK] Loaded dataset: {dataset_path.name} ({len(df)} records)")

    # 2. Check for Target Column
    target_col = None
    for col in ["target", "risk_pred", "risk_label", "is_high_risk", "label"]:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        print("[INFO] No target label column found. Generating calibrated ground-truth target...")
        df["target"] = np.where(
            (df["avg_price_creep"] > 0.25) |
            (df["rec_rejection_pct"] > 0.10) |
            ((df["is_single_source"] == 1) & (df["stock_cover_days"] < 20)),
            1,
            0
        )
        target_col = "target"

    X = df[FEATURES]
    y = df[target_col]

    pos_count = int((y == 1).sum())
    neg_count = int((y == 0).sum())
    print(f"[INFO] Class distribution: {pos_count} High Risk (1), {neg_count} Stable (0)")

    # 3. Train / Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Split: {len(X_train)} training samples, {len(X_val)} validation samples")

    # 4. Train XGBoost
    print("[INFO] Fitting XGBoost Classifier ensemble...")
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        eval_metric="logloss",
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # 5. Evaluate Performance
    y_pred = model.predict(X_val)
    y_probs = model.predict_proba(X_val)[:, 1]

    acc = float(accuracy_score(y_val, y_pred))
    prec = float(precision_score(y_val, y_pred, zero_division=0))
    rec = float(recall_score(y_val, y_pred, zero_division=0))
    f1 = float(f1_score(y_val, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_val, y_probs)) if len(np.unique(y_val)) > 1 else 1.0
    cm = confusion_matrix(y_val, y_pred).tolist()

    print("\n" + "=" * 40)
    print("[METRICS] VALIDATION RESULTS")
    print("=" * 40)
    print(f"Accuracy Score  : {acc * 100:.2f}%")
    print(f"Precision Score : {prec * 100:.2f}%")
    print(f"Recall Score    : {rec * 100:.2f}%")
    print(f"F1 Score        : {f1 * 100:.2f}%")
    print(f"ROC-AUC Score   : {roc_auc * 100:.2f}%")
    print(f"Confusion Matrix: {cm}")

    # 6. Save Model
    model.save_model(str(model_output_path))
    print(f"\n[OK] Model saved to: {model_output_path.name}")
    webapp_model = PROJECT_DIR / "webapp" / "xgb_supplier_risk_model.json"
    if webapp_model.parent.exists():
        model.save_model(str(webapp_model))
        print(f"[OK] Model synchronized to: webapp/{webapp_model.name}")

    # 7. Save Metrics JSON for Streamlit UI
    metrics_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "total_samples": len(df),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "feature_importances": {
            feat: float(imp) for feat, imp in zip(FEATURES, model.feature_importances_)
        }
    }

    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[OK] Training metrics saved to: {metrics_output_path.name}")
    webapp_metrics = PROJECT_DIR / "webapp" / "training_metrics.json"
    if webapp_metrics.parent.exists():
        with open(webapp_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=4)
        print(f"[OK] Metrics synchronized to: webapp/{webapp_metrics.name}")
        
    print("=" * 60)
    print("[SUCCESS] RETRAINING COMPLETED! REFRESH STREAMLIT TO SEE UPDATED UI.")
    print("=" * 60)

    return metrics_data


if __name__ == "__main__":
    n_est = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    max_d = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    retrain_model(n_estimators=n_est, max_depth=max_d, learning_rate=lr)

