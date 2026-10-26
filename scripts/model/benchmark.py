"""Stage 6: Baseline & Candidate Model Benchmarking.

Evaluates 5 candidate classifiers on the 94D FeatureSchema v1.0 feature space:
  1. Logistic Regression (Baseline, with StandardScaler)
  2. Random Forest (Unscaled Tree Ensemble)
  3. Extra Trees (Extremely Randomized Trees)
  4. SVM with RBF Kernel (with StandardScaler)
  5. Linear SVM (with StandardScaler)

All models are trained strictly on X_train and evaluated on X_val.
Primary selection metric: Macro F1.
Secondary metrics: Accuracy, Macro Precision, Macro Recall, Weighted F1, Latency, Size.
"""

import json
from pathlib import Path
import sys
import time
from typing import Any
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.model.dataset_loader import DatasetSplits, load_and_validate_dataset


def get_candidate_models(random_state: int = 42) -> dict[str, Any]:
    """Instantiates candidate models and pipelines."""
    return {
        "Logistic Regression (Baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state, C=1.0)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_split=3,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=25,
            min_samples_split=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "SVM (RBF Kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=random_state)),
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=1.0, probability=True, random_state=random_state)),
        ]),
    }


def evaluate_model(
    name: str,
    model: Any,
    splits: DatasetSplits,
) -> dict[str, Any]:
    """Trains a candidate model on X_train and evaluates performance on X_val."""
    print(f"\nEvaluating: {name}...")
    t_train_start = time.perf_counter()
    model.fit(splits.X_train, splits.y_train)
    train_duration = time.perf_counter() - t_train_start

    # Validation predictions
    t_infer_start = time.perf_counter()
    val_preds = model.predict(splits.X_val)
    infer_duration = time.perf_counter() - t_infer_start
    latency_ms_per_sample = (infer_duration / max(1, len(splits.X_val))) * 1000.0

    # Serialization size estimation
    import io
    buf = io.BytesIO()
    joblib.dump(model, buf, compress=3)
    model_size_kb = len(buf.getvalue()) / 1024.0

    acc = float(accuracy_score(splits.y_val, val_preds))
    macro_p = float(precision_score(splits.y_val, val_preds, average="macro", zero_division=0))
    macro_r = float(recall_score(splits.y_val, val_preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(splits.y_val, val_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(splits.y_val, val_preds, average="weighted", zero_division=0))

    print(f"  Validation Accuracy:  {acc * 100:.2f}%")
    print(f"  Macro F1 Score:       {macro_f1:.4f}")
    print(f"  Weighted F1 Score:    {weighted_f1:.4f}")
    print(f"  Train Time:           {train_duration:.2f}s")
    print(f"  Latency:              {latency_ms_per_sample:.3f} ms/sample")
    print(f"  Model Size (joblib):  {model_size_kb:.1f} KB")

    return {
        "model_name": name,
        "validation_accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "train_time_seconds": round(train_duration, 2),
        "inference_latency_ms_per_sample": round(latency_ms_per_sample, 4),
        "model_size_kb": round(model_size_kb, 1),
    }


def run_benchmark(output_dir: Path | None = None) -> list[dict[str, Any]]:
    """Runs full benchmarking across all 5 candidate models."""
    target_reports = output_dir or (PROJECT_ROOT / "reports" / "stage6")
    target_reports.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" SIGNBRIDGE — STAGE 6: MODEL BENCHMARKING & EVALUATION")
    print("=" * 80)

    # 1. Load validated dataset with strict train/val/test partitioning
    splits = load_and_validate_dataset()
    print(f"Loaded {splits.total_samples:,} validated samples across {splits.num_classes} classes.")
    print(f"  Train split:      {len(splits.X_train):,} samples ({len(splits.X_train)/splits.total_samples*100:.1f}%)")
    print(f"  Validation split: {len(splits.X_val):,} samples ({len(splits.X_val)/splits.total_samples*100:.1f}%)")
    print(f"  Test split:       {len(splits.X_test):,} samples ({len(splits.X_test)/splits.total_samples*100:.1f}%) [UNTOUCHED]")

    candidates = get_candidate_models(random_state=42)
    results: list[dict[str, Any]] = []

    for name, model in candidates.items():
        metrics = evaluate_model(name, model, splits)
        results.append(metrics)

    # Sort candidates by primary metric: Macro F1 descending
    ranked = sorted(results, key=lambda r: r["macro_f1"], reverse=True)

    print("\n" + "=" * 80)
    print(" BENCHMARK RANKING (Ranked by Macro F1)")
    print("=" * 80)
    print(f"{'Rank':<5} {'Model':<32} {'Macro F1':<10} {'Accuracy':<10} {'Latency (ms)':<14} {'Size (KB)':<10}")
    print("-" * 80)
    for rank, r in enumerate(ranked, 1):
        print(
            f"{rank:<5} {r['model_name']:<32} {r['macro_f1']:<10.4f} "
            f"{r['validation_accuracy']*100:<9.2f}% {r['inference_latency_ms_per_sample']:<14.3f} "
            f"{r['model_size_kb']:<10.1f}"
        )

    # Write model_comparison.json
    out_file = target_reports / "model_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "ranking": ranked,
            "primary_metric": "macro_f1",
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train_samples": len(splits.X_train),
            "val_samples": len(splits.X_val),
            "test_samples": len(splits.X_test),
        }, f, indent=2)

    print(f"\nSaved benchmark comparison to: {out_file}")
    return ranked


if __name__ == "__main__":
    run_benchmark()
