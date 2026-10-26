"""Stage 6: Final Model Training, Evaluation & Artifact Generation.

Tuning, final training on Train+Val, one-time evaluation on untouched Test set,
generation of production model artifacts, metrics, and error analysis.
"""

import json
from pathlib import Path
import sys
import time
from typing import Any
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.features.schema import FeatureSchema
from signbridge.model.base import ModelMetadata
from signbridge.model.classifier import SignClassifier
from signbridge.model.dataset_loader import DatasetSplits, load_and_validate_dataset
from signbridge.model.persistence import save_sign_classifier


def build_estimator(model_type: str, params: dict[str, Any], random_state: int = 42) -> Any:
    """Factory creating an estimator or pipeline for a given model type and hyperparameters."""
    if model_type == "linear_svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=params.get("C", 1.0), max_iter=2500, random_state=random_state),
                ensemble=False,
            )),
        ])
    elif model_type == "rbf_svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                SVC(kernel="rbf", C=params.get("C", 10.0), gamma=params.get("gamma", "scale"), random_state=random_state),
                ensemble=False,
            )),
        ])
    elif model_type == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 25),
            min_samples_split=params.get("min_samples_split", 2),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 20),
            min_samples_split=params.get("min_samples_split", 3),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=params.get("C", 1.0), max_iter=1000, random_state=random_state)),
        ])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def tune_and_select_best_model(
    splits: DatasetSplits,
    random_state: int = 42,
) -> tuple[str, str, dict[str, Any], float, float]:
    """Evaluates fine-tuning candidates on X_train / X_val and selects the best by Macro F1."""
    print("\n[HYPERPARAMETER TUNING] Evaluating candidate configurations...")

    candidate_configs = [
        ("Linear SVM (C=1.0)", "linear_svm", {"C": 1.0}),
        ("Linear SVM (C=0.5)", "linear_svm", {"C": 0.5}),
        ("Linear SVM (C=2.0)", "linear_svm", {"C": 2.0}),
        ("SVM (RBF Kernel, C=10.0)", "rbf_svm", {"C": 10.0, "gamma": "scale"}),
        ("SVM (RBF Kernel, C=20.0)", "rbf_svm", {"C": 20.0, "gamma": "scale"}),
        ("Extra Trees (n=200, max_depth=25)", "extra_trees", {"n_estimators": 200, "max_depth": 25, "min_samples_split": 2}),
        ("Random Forest (n=200, max_depth=20)", "random_forest", {"n_estimators": 200, "max_depth": 20, "min_samples_split": 3}),
    ]

    scored = []
    for display_name, m_type, params in candidate_configs:
        t0 = time.perf_counter()
        model = build_estimator(m_type, params, random_state=random_state)
        model.fit(splits.X_train, splits.y_train)
        val_preds = model.predict(splits.X_val)
        val_f1 = float(f1_score(splits.y_val, val_preds, average="macro", zero_division=0))
        val_acc = float(accuracy_score(splits.y_val, val_preds))
        elapsed = time.perf_counter() - t0
        print(f"  {display_name:<40} -> Macro F1: {val_f1:.4f} | Acc: {val_acc*100:.2f}% ({elapsed:.2f}s)")
        scored.append((val_f1, val_acc, display_name, m_type, params))

    # Primary selection metric: Macro F1
    scored.sort(key=lambda item: item[0], reverse=True)
    best_f1, best_acc, best_name, best_type, best_params = scored[0]
    print(f"\n[WINNING MODEL]: {best_name} with Macro F1: {best_f1:.4f} (Accuracy: {best_acc*100:.2f}%)")
    return best_name, best_type, best_params, best_f1, best_acc


def generate_visual_reports(
    conf_matrix: np.ndarray,
    class_names: list[str],
    per_class_f1: dict[str, float],
    test_confidences: list[float],
    reports_dir: Path,
) -> None:
    """Generates visual plots without requiring a GUI display."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("SignBridge Stage 6: Test Confusion Matrix (26 Classes)", fontsize=14, pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=0, fontsize=9)
    plt.yticks(tick_marks, class_names, fontsize=9)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
    plt.ylabel("True Class", fontsize=11, labelpad=8)
    plt.tight_layout()
    cm_path = reports_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # 2. Per-Class F1 Bar Chart
    plt.figure(figsize=(13, 5))
    classes_sorted = sorted(per_class_f1.keys())
    f1_vals = [per_class_f1[c] for c in classes_sorted]
    bars = plt.bar(classes_sorted, f1_vals, color="#2b5c8f", edgecolor="#183658", alpha=0.85)
    plt.axhline(y=np.mean(f1_vals), color="red", linestyle="--", label=f"Mean F1: {np.mean(f1_vals):.3f}")
    plt.title("SignBridge Stage 6: Per-Class F1 Scores on Unseen Test Partition", fontsize=13, pad=12)
    plt.xlabel("ASL Alphabet Sign Class", fontsize=11)
    plt.ylabel("F1 Score", fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    f1_path = reports_dir / "per_class_f1.png"
    plt.savefig(f1_path, dpi=150)
    plt.close()

    # 3. Confidence Distribution Histogram
    plt.figure(figsize=(9, 5))
    plt.hist(test_confidences, bins=25, range=(0.0, 1.0), color="#107c41", edgecolor="#094a27", alpha=0.8)
    plt.axvline(x=np.mean(test_confidences), color="red", linestyle="--", label=f"Mean: {np.mean(test_confidences):.3f}")
    plt.title("SignBridge Stage 6: Test Set Prediction Confidence Distribution", fontsize=13, pad=12)
    plt.xlabel("Top-1 Prediction Confidence", fontsize=11)
    plt.ylabel("Sample Frequency", fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    conf_path = reports_dir / "confidence_distribution.png"
    plt.savefig(conf_path, dpi=150)
    plt.close()

    # 4. Model Comparison Chart (from model_comparison.json if available)
    comp_json = reports_dir / "model_comparison.json"
    if comp_json.exists():
        try:
            with open(comp_json, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            ranking = c_data.get("ranking", [])
            if ranking:
                plt.figure(figsize=(11, 5))
                m_names = [r["model_name"] for r in ranking][::-1]
                m_f1 = [r["macro_f1"] for r in ranking][::-1]
                m_acc = [r["validation_accuracy"] for r in ranking][::-1]

                y_pos = np.arange(len(m_names))
                plt.barh(y_pos - 0.2, m_f1, 0.4, label="Macro F1", color="#1f77b4")
                plt.barh(y_pos + 0.2, m_acc, 0.4, label="Accuracy", color="#2ca02c")
                plt.yticks(y_pos, m_names, fontsize=10)
                plt.xlim(0.85, 1.0)
                plt.xlabel("Score", fontsize=11)
                plt.title("Candidate Model Benchmarking Comparison (Validation Set)", fontsize=13, pad=12)
                plt.legend(loc="lower right")
                plt.grid(axis="x", linestyle=":", alpha=0.6)
                plt.tight_layout()
                comp_path = reports_dir / "model_comparison.png"
                plt.savefig(comp_path, dpi=150)
                plt.close()
        except Exception as e:
            print(f"Warning: Could not plot model comparison chart: {e}")

    print(f"  Visual reports saved to: {reports_dir}")


def train_and_evaluate_pipeline(
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
    random_state: int = 42,
) -> tuple[SignClassifier, dict[str, Any]]:
    """Runs the complete end-to-end Stage 6 training, testing, artifacting, and error analysis."""
    target_models = models_dir or (PROJECT_ROOT / "models" / "sign_classifier")
    target_reports = reports_dir or (PROJECT_ROOT / "reports" / "stage6")

    print("=" * 80)
    print(" SIGNBRIDGE — STAGE 6: FINAL MODEL TRAINING & EVALUATION")
    print("=" * 80)

    # 1. Ingest and validate dataset
    splits = load_and_validate_dataset(random_state=random_state)
    print(f"[DATASET VALIDATED] Total: {splits.total_samples:,} samples across {splits.num_classes} classes.")
    print(f"  Train:      {len(splits.X_train):,} samples")
    print(f"  Validation: {len(splits.X_val):,} samples")
    print(f"  Test:       {len(splits.X_test):,} samples [STRICTLY ISOLATED]")

    # 2. Hyperparameter tuning & model selection on Train / Val
    best_name, best_type, best_params, val_f1, val_acc = tune_and_select_best_model(
        splits, random_state=random_state
    )

    # 3. Final Training: Fit selected model on TRAIN + VALIDATION
    print(f"\n[FINAL TRAINING] Refitting {best_name} on Train + Validation partitions...")
    X_train_val = np.vstack([splits.X_train, splits.X_val])
    y_train_val = np.concatenate([splits.y_train, splits.y_val])

    final_estimator = build_estimator(best_type, best_params, random_state=random_state)

    t_fit_start = time.perf_counter()
    final_estimator.fit(X_train_val, y_train_val)
    fit_duration = time.perf_counter() - t_fit_start
    print(f"  Final fit complete in {fit_duration:.2f}s on {len(X_train_val):,} samples.")

    # 4. ONE-TIME FINAL EVALUATION ON UNTOUCHED TEST PARTITION
    print("\n[TEST EVALUATION] Evaluating final model ONCE on untouched Test set...")
    t_test_start = time.perf_counter()
    test_probs = (
        final_estimator.predict_proba(splits.X_test)
        if hasattr(final_estimator, "predict_proba")
        else None
    )
    test_preds = final_estimator.predict(splits.X_test)
    test_duration = time.perf_counter() - t_test_start

    # Top-1 and Top-3 accuracy
    test_acc = float(accuracy_score(splits.y_test, test_preds))
    macro_p = float(precision_score(splits.y_test, test_preds, average="macro", zero_division=0))
    macro_r = float(recall_score(splits.y_test, test_preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(splits.y_test, test_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(splits.y_test, test_preds, average="weighted", zero_division=0))

    top3_correct = 0
    test_confidences = []
    if test_probs is not None:
        top3_preds = np.argsort(test_probs, axis=1)[:, -3:]
        for i, y_true in enumerate(splits.y_test):
            if y_true in top3_preds[i]:
                top3_correct += 1
            test_confidences.append(float(test_probs[i, test_preds[i]]))
        top3_acc = top3_correct / len(splits.y_test)
    else:
        top3_acc = test_acc
        test_confidences = [1.0] * len(splits.y_test)

    # Detailed per-class metrics
    class_names = [splits.id_to_class[i] for i in range(splits.num_classes)]
    cls_report = classification_report(
        splits.y_test,
        test_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(splits.y_test, test_preds, labels=range(splits.num_classes))

    # Error Analysis: Strongest, Weakest, Most Confused Pairs
    per_class_f1 = {
        cls_name: float(cls_report[cls_name]["f1-score"])
        for cls_name in class_names
        if cls_name in cls_report
    }
    sorted_f1 = sorted(per_class_f1.items(), key=lambda it: it[1])
    weakest_classes = sorted_f1[:5]
    strongest_classes = sorted_f1[-5:][::-1]

    # Find top confused pairs (off-diagonal entries)
    confused_pairs = []
    for i in range(splits.num_classes):
        for j in range(splits.num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_class": splits.id_to_class[i],
                    "predicted_class": splits.id_to_class[j],
                    "error_count": int(cm[i, j]),
                })
    confused_pairs.sort(key=lambda cp: cp["error_count"], reverse=True)

    # 5. Latency Benchmark (Per-sample single vector inference)
    print("\n[BENCHMARK] Profiling single-sample inference latency (1,000 runs)...")
    sample_probe = splits.X_test[0:1]
    latencies = []
    for _ in range(1000):
        t0 = time.perf_counter()
        final_estimator.predict(sample_probe)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_lat = float(np.mean(latencies))
    med_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    throughput = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    print(f"  Test Accuracy:        {test_acc * 100:.2f}%")
    print(f"  Top-3 Test Accuracy:  {top3_acc * 100:.2f}%")
    print(f"  Test Macro F1:        {macro_f1:.4f}")
    print(f"  Test Weighted F1:     {weighted_f1:.4f}")
    print(f"  Mean Latency:         {mean_lat:.3f} ms (p95: {p95_lat:.3f} ms)")
    print(f"  Throughput:           {throughput:.1f} inferences/sec")

    # 6. Build Metadata & Save Model Artifacts
    metadata = ModelMetadata(
        model_name="SignBridge-StaticSignClassifier",
        model_version="1.0.0",
        algorithm=best_name,
        feature_schema_version=FeatureSchema.VERSION,
        feature_dimension=FeatureSchema.TOTAL_DIMENSIONS,
        dataset_identity=splits.manifest.get("dataset_name", "ASL-Alphabet-Natural-RGB"),
        dataset_manifest_hash=splits.manifest_hash,
        class_to_id=splits.class_to_id,
        id_to_class=splits.id_to_class,
        num_classes=splits.num_classes,
        train_samples=len(X_train_val),
        val_samples=len(splits.X_val),
        test_samples=len(splits.X_test),
        random_seed=random_state,
        hyperparameters=best_params,
        training_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        scikit_learn_version=sklearn.__version__,
        metrics={
            "validation_macro_f1": round(val_f1, 4),
            "validation_accuracy": round(val_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "top3_accuracy": round(top3_acc, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "mean_latency_ms": round(mean_lat, 4),
            "median_latency_ms": round(med_lat, 4),
            "p95_latency_ms": round(p95_lat, 4),
            "throughput_samples_per_sec": round(throughput, 1),
            "confidence_mean": round(float(np.mean(test_confidences)), 4),
            "confidence_std": round(float(np.std(test_confidences)), 4),
        },
    )

    classifier = SignClassifier(model=final_estimator, metadata=metadata)
    save_sign_classifier(classifier, target_models)
    print(f"\nModel artifacts saved to: {target_models}")

    # Model serialized size
    model_size_bytes = (target_models / "model.joblib").stat().st_size
    model_size_mb = model_size_bytes / (1024.0 * 1024.0)

    # 7. Write Stage 6 Reports
    target_reports.mkdir(parents=True, exist_ok=True)

    with open(target_reports / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(cls_report, f, indent=2)

    with open(target_reports / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump({
            "classes": class_names,
            "matrix": cm.tolist(),
        }, f, indent=2)

    with open(target_reports / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump({
            "weakest_classes": [{"class": c, "f1": round(f, 4)} for c, f in weakest_classes],
            "strongest_classes": [{"class": c, "f1": round(f, 4)} for c, f in strongest_classes],
            "top_confused_pairs": confused_pairs[:10],
            "confidence_distribution": {
                "mean": round(float(np.mean(test_confidences)), 4),
                "median": round(float(np.median(test_confidences)), 4),
                "std": round(float(np.std(test_confidences)), 4),
                "min": round(float(np.min(test_confidences)), 4),
                "max": round(float(np.max(test_confidences)), 4),
            },
        }, f, indent=2)

    with open(target_reports / "benchmark.json", "w", encoding="utf-8") as f:
        json.dump({
            "mean_latency_ms": round(mean_lat, 4),
            "median_latency_ms": round(med_lat, 4),
            "p95_latency_ms": round(p95_lat, 4),
            "throughput_inferences_per_sec": round(throughput, 1),
            "model_size_mb": round(model_size_mb, 2),
            "model_size_bytes": model_size_bytes,
        }, f, indent=2)

    # 8. Visual Plots
    generate_visual_reports(
        conf_matrix=cm,
        class_names=class_names,
        per_class_f1=per_class_f1,
        test_confidences=test_confidences,
        reports_dir=target_reports,
    )

    print("=" * 80)
    print(" STAGE 6 TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return classifier, metadata.metrics


if __name__ == "__main__":
    train_and_evaluate_pipeline()
