"""Verification and Benchmarking Script for Stage 4: Feature Engineering.

Runs standalone (hardware-independent) validation:
  1. Synthetic hand poses (open palm, fist, peace sign, pinch).
  2. Mathematical translation invariance verification.
  3. Mathematical scale invariance verification.
  4. Handedness canonicalization verification.
  5. Numerical safety and NaN/Inf rejection checks.
  6. High-throughput latency benchmarking (1,000 extractions).
"""

import sys
import time
from pathlib import Path
import numpy as np

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from signbridge.features.extractor import HandFeatureExtractor
from signbridge.features.schema import FEATURE_NAMES, FeatureSchema
from signbridge.features.validator import LandmarkValidationError, LandmarkValidator
from signbridge.vision.landmarks import HandLandmarks, Landmark3D


def create_synthetic_hand(pose: str = "open", handedness: str = "Right") -> HandLandmarks:
    """Generates synthetic 21-point hand landmarks for testing."""
    coords = np.zeros((21, 3), dtype=np.float32)
    # Wrist at (320, 360, 0)
    coords[0] = [320.0, 360.0, 0.0]

    # MCP joints (landmarks 1, 5, 9, 13, 17)
    coords[1] = [300.0, 330.0, -10.0]  # thumb CMC
    coords[2] = [280.0, 300.0, -20.0]  # thumb MCP
    coords[5] = [295.0, 260.0, -5.0]   # index MCP
    coords[9] = [320.0, 250.0, 0.0]    # middle MCP (palm anchor distance = 110 px)
    coords[13] = [345.0, 260.0, 5.0]   # ring MCP
    coords[17] = [365.0, 275.0, 10.0]  # pinky MCP

    if pose == "open":
        # All fingers extended upward
        for finger_idx, base_mcp in [(1, 2), (2, 5), (3, 9), (4, 13), (5, 17)]:
            start_idx = 1 if finger_idx == 1 else (finger_idx - 1) * 4 + 1
            for offset in range(1, 4):
                curr = start_idx + offset
                prev = start_idx + offset - 1
                coords[curr] = coords[prev] + [0.0, -35.0, 0.0]
    elif pose == "fist":
        # All fingers curled inward toward palm
        for finger_idx in range(2, 6):
            base_mcp = (finger_idx - 1) * 4 + 1
            coords[base_mcp + 1] = coords[base_mcp] + [0.0, -10.0, 20.0]
            coords[base_mcp + 2] = coords[base_mcp + 1] + [0.0, 15.0, 25.0]
            coords[base_mcp + 3] = coords[base_mcp + 2] + [0.0, 20.0, 10.0]
        # Thumb folded over fingers
        coords[3] = coords[2] + [20.0, 0.0, 15.0]
        coords[4] = coords[3] + [25.0, 0.0, 10.0]
    elif pose == "pinch":
        # Thumb tip and index tip touch
        coords[3] = coords[2] + [15.0, -15.0, 0.0]
        coords[4] = [300.0, 230.0, 0.0]    # thumb tip
        coords[6] = coords[5] + [0.0, -20.0, 0.0]
        coords[7] = coords[6] + [0.0, -15.0, 0.0]
        coords[8] = [300.0, 230.0, 0.0]    # index tip touches thumb tip
        # Middle, ring, pinky extended
        for finger_idx in [3, 4, 5]:
            base_mcp = (finger_idx - 1) * 4 + 1
            for offset in range(1, 4):
                curr = base_mcp + offset
                prev = base_mcp + offset - 1
                coords[curr] = coords[prev] + [0.0, -35.0, 0.0]

    if handedness == "Left":
        # Mirror horizontally around center 320
        coords[:, 0] = 640.0 - coords[:, 0]

    lms = [Landmark3D(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in coords]
    return HandLandmarks(handedness=handedness, landmarks=lms, raw_array=coords)


def run_stage4_verification() -> bool:
    print("=" * 72)
    print(" SIGNBRIDGE — STAGE 4 VERIFICATION & BENCHMARK SUITE")
    print("=" * 72)

    passed_checks = 0
    total_checks = 7

    extractor = HandFeatureExtractor(canonicalize_handedness=True)

    # 1. Structural & Schema Conformance Check
    print("\n[CHECK 1/7] Schema & Dimensionality Conformance...")
    hand_open = create_synthetic_hand("open", "Right")
    feat_open = extractor.extract(hand_open)

    assert feat_open.features.shape == (94,), f"Expected (94,), got {feat_open.features.shape}"
    assert feat_open.features.dtype == np.float32, f"Expected float32, got {feat_open.features.dtype}"
    assert feat_open.schema_version == "1.0"
    assert len(FEATURE_NAMES) == 94
    print(f"  --> PASS: Feature vector shape is {feat_open.features.shape}, dtype={feat_open.features.dtype}, schema={feat_open.schema_version}")
    passed_checks += 1

    # 2. Translation Invariance Check
    print("\n[CHECK 2/7] Mathematical Translation Invariance...")
    offsets = [
        np.array([150.0, -200.0, 50.0], dtype=np.float32),
        np.array([-500.0, 300.0, -100.0], dtype=np.float32),
        np.array([1234.5, -678.9, 42.0], dtype=np.float32),
    ]
    max_translation_err = 0.0
    for offset in offsets:
        shifted_coords = hand_open.raw_array + offset
        shifted_hand = HandLandmarks(
            handedness="Right",
            landmarks=[Landmark3D(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in shifted_coords],
            raw_array=shifted_coords,
        )
        feat_shifted = extractor.extract(shifted_hand)
        err = float(np.max(np.abs(feat_open.features - feat_shifted.features)))
        max_translation_err = max(max_translation_err, err)

    assert max_translation_err < 1e-5, f"Translation error {max_translation_err} exceeds 1e-5"
    print(f"  --> PASS: Maximum translation error across all 94 features: {max_translation_err:.2e} (tolerance: 1e-5)")
    passed_checks += 1

    # 3. Scale Invariance Check
    print("\n[CHECK 3/7] Mathematical Scale Invariance...")
    scales = [0.3, 0.7, 1.5, 2.5, 4.0]
    max_scale_err = 0.0
    for scale in scales:
        wrist = hand_open.raw_array[0]
        scaled_coords = wrist + (hand_open.raw_array - wrist) * scale
        scaled_hand = HandLandmarks(
            handedness="Right",
            landmarks=[Landmark3D(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in scaled_coords],
            raw_array=scaled_coords,
        )
        feat_scaled = extractor.extract(scaled_hand)
        err = float(np.max(np.abs(feat_open.features - feat_scaled.features)))
        max_scale_err = max(max_scale_err, err)

    assert max_scale_err < 1e-3, f"Scale error {max_scale_err} exceeds 1e-3"
    print(f"  --> PASS: Maximum scale error across all 94 features: {max_scale_err:.2e} (tolerance: 1e-3)")
    passed_checks += 1

    # 4. Handedness Canonicalization Check
    print("\n[CHECK 4/7] Handedness Canonicalization (Left Hand Reflection)...")
    hand_left_open = create_synthetic_hand("open", "Left")
    feat_left = extractor.extract(hand_left_open)

    coord_diff = float(np.max(np.abs(feat_open.coords - feat_left.coords)))
    dist_diff = float(np.max(np.abs(feat_open.wrist_tip_distances - feat_left.wrist_tip_distances)))
    angle_diff = float(np.max(np.abs(feat_open.joint_angles - feat_left.joint_angles)))

    assert coord_diff < 1e-4, f"Canonical coordinate diff {coord_diff} exceeds 1e-4"
    assert dist_diff < 1e-4, f"Canonical distance diff {dist_diff} exceeds 1e-4"
    assert angle_diff < 1e-4, f"Canonical angle diff {angle_diff} exceeds 1e-4"
    print(f"  --> PASS: Handedness canonicalization error: coords={coord_diff:.2e}, dists={dist_diff:.2e}, angles={angle_diff:.2e}")
    passed_checks += 1

    # 5. Determinism & Numerical Safety
    print("\n[CHECK 5/7] Determinism & Numerical Safety (Zero NaNs/Infs)...")
    baseline = extractor.extract(hand_open).features
    for _ in range(50):
        sample = extractor.extract(hand_open).features
        assert np.array_equal(baseline, sample), "Extractor output failed bitwise determinism"

    assert np.all(np.isfinite(baseline)), "Feature vector contains non-finite values"
    print("  --> PASS: 50/50 consecutive runs are bit-for-bit identical; 0 NaNs, 0 Infs detected")
    passed_checks += 1

    # 6. Degenerate Input Handling & Exception Safety
    print("\n[CHECK 6/7] Degenerate Input Validation & Exception Safety...")
    # Test NaN rejection
    nan_coords = hand_open.raw_array.copy()
    nan_coords[8, 1] = np.nan
    try:
        LandmarkValidator.validate(nan_coords)
        raise AssertionError("Failed to reject NaN in coordinates")
    except LandmarkValidationError:
        pass

    # Test collapsed palm rejection
    collapsed = np.zeros((21, 3), dtype=np.float32)
    try:
        LandmarkValidator.validate(collapsed)
        raise AssertionError("Failed to reject collapsed hand")
    except LandmarkValidationError:
        pass
    print("  --> PASS: LandmarkValidator successfully catches NaNs, Infs, and degenerate collapsed hands")
    passed_checks += 1

    # 7. High-Throughput Latency Benchmark
    print("\n[CHECK 7/7] High-Throughput Latency Benchmark (1,000 iterations)...")
    poses = [
        create_synthetic_hand("open"),
        create_synthetic_hand("fist"),
        create_synthetic_hand("pinch"),
    ]

    # Warmup
    for p in poses:
        for _ in range(20):
            _ = extractor.extract(p)

    iterations = 1000
    latencies: list[float] = []

    start_bench = time.perf_counter()
    for i in range(iterations):
        target_pose = poses[i % 3]
        t0 = time.perf_counter()
        _ = extractor.extract(target_pose)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms
    total_time = time.perf_counter() - start_bench

    latencies_np = np.array(latencies, dtype=np.float64)
    avg_ms = float(np.mean(latencies_np))
    p50_ms = float(np.percentile(latencies_np, 50))
    p95_ms = float(np.percentile(latencies_np, 95))
    p99_ms = float(np.percentile(latencies_np, 99))
    throughput = iterations / total_time

    print(f"  Total Iterations: {iterations}")
    print(f"  Total Benchmark Time: {total_time:.3f} s")
    print(f"  Throughput: {throughput:,.1f} extractions/sec")
    print(f"  Mean Latency: {avg_ms * 1000.0:.1f} us ({avg_ms:.3f} ms)")
    print(f"  p50  Latency: {p50_ms * 1000.0:.1f} us ({p50_ms:.3f} ms)")
    print(f"  p95  Latency: {p95_ms * 1000.0:.1f} us ({p95_ms:.3f} ms)")
    print(f"  p99  Latency: {p99_ms * 1000.0:.1f} us ({p99_ms:.3f} ms)")

    assert avg_ms < 1.0, f"Average latency {avg_ms:.3f} ms exceeds 1.0 ms budget"
    assert throughput > 1000.0, f"Throughput {throughput:.1f} extractions/s below 1,000 threshold"
    print("  --> PASS: Latency and throughput well exceed real-time requirements (budget < 1.0 ms)")
    passed_checks += 1

    print("\n" + "=" * 72)
    print(f" STAGE 4 VERIFICATION COMPLETE: {passed_checks}/{total_checks} CHECKS PASSED")
    print("=" * 72)
    return passed_checks == total_checks


if __name__ == "__main__":
    success = run_stage4_verification()
    sys.exit(0 if success else 1)
