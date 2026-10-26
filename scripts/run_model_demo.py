"""Stage 6: Offline Sign Classifier Inference Demonstration.

Demonstrates offline static sign recognition using the trained model artifact
and feature vectors extracted from the isolated test partition.
Does NOT connect the webcam or UI.
"""

from pathlib import Path
import random
import sys
import numpy as np

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.model.dataset_loader import load_and_validate_dataset
from signbridge.model.persistence import load_sign_classifier


def run_offline_demo(num_samples: int = 5, seed: int = 42) -> None:
    """Loads production model and demonstrates inference on test set feature vectors."""
    model_dir = PROJECT_ROOT / "models" / "sign_classifier"

    print("=" * 80)
    print(" SIGNBRIDGE — STAGE 6: OFFLINE MODEL INFERENCE DEMO")
    print(f" Model Directory: {model_dir}")
    print("=" * 80)

    if not model_dir.exists():
        print(f"ERROR: Model directory {model_dir} not found. Run scripts/model/train.py first.")
        sys.exit(1)

    # 1. Load trained model
    classifier = load_sign_classifier(model_dir)
    print(f"Model loaded: {classifier.metadata.model_name} (v{classifier.metadata.model_version})")
    print(f"Algorithm:    {classifier.metadata.algorithm}")
    print(f"Vocabulary:   {len(classifier.vocabulary)} classes: {', '.join(classifier.vocabulary[:10])}...")

    # 2. Load test set features
    splits = load_and_validate_dataset()
    print(f"Test samples available: {len(splits.X_test):,}")

    rng = random.Random(seed)
    sample_indices = rng.sample(range(len(splits.X_test)), min(num_samples, len(splits.X_test)))

    print("\n" + "=" * 80)
    print(f" INFERENCE RESULTS ON {len(sample_indices)} TEST SAMPLES")
    print("=" * 80)

    for i, idx in enumerate(sample_indices, 1):
        features = splits.X_test[idx]
        true_class_id = int(splits.y_test[idx])
        true_label = splits.id_to_class[true_class_id]

        # Perform inference
        result = classifier.predict(features)
        match_symbol = "[CORRECT]" if result.class_name == true_label else "[MISMATCH]"

        print(f"\nSample #{i} (Test Index: {idx}):")
        print(f"  True Label:       '{true_label}' (Class ID: {true_class_id})")
        print(f"  Predicted Label:  '{result.class_name}' (Class ID: {result.class_id})  {match_symbol}")
        print(f"  Confidence:       {result.confidence * 100:.2f}%")
        print(f"  Top Margin:       {result.top_margin * 100:.2f}%")
        print(f"  Inference Time:   {result.inference_latency_ms:.3f} ms")
        print("  Top-3 Candidates:")
        for rank, (cand_name, cand_score) in enumerate(result.top_candidates, 1):
            print(f"    {rank}. '{cand_name}': {cand_score * 100:.2f}%")

    print("\n" + "=" * 80)
    print(" DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_offline_demo()
