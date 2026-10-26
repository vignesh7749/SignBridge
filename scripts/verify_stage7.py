"""Stage 7 Verification Script: Synthetic Simulation Test Harness.

Executes and verifies 10 synthetic prediction sequences covering:
1. Consistent sign recognition (A A A A A -> A)
2. Single-frame jitter tolerance (A A B A A A -> A)
3. Sign transitions (A A A B B B -> A, B)
4. Duplicate suppression (A A A A A A A -> A only)
5. Rapid alternating noise rejection (A B A B A B -> No emission)
6. Hand loss within grace period (A A A -> NO_HAND -> A A A -> A)
7. Hand loss exceeding grace period (A A A -> NO_HAND (long) -> A A A -> Reset + A)
8. Low-confidence rejection (< threshold -> No emission)
9. FPS variation invariance (30 FPS, 24 FPS, 15 FPS -> Accurate dwell-time emissions)
10. Rapid distinct sign sequence (A -> B -> C -> A, B, C)

Also profiles temporal processing latency and dwell-time stabilization delay.
"""

from pathlib import Path
import sys
import time
from typing import List, Optional
import numpy as np

# Ensure src is in Python module search path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.temporal.config import TemporalEngineConfig
from signbridge.temporal.contracts import FramePrediction, StableSign, TemporalState
from signbridge.temporal.engine import TemporalEngine


def generate_frame_stream(
    sequence: list[tuple[str, int, float, bool]],  # (label, class_id, conf, has_hand)
    fps: float = 30.0,
    start_time: float = 0.0,
) -> list[FramePrediction]:
    """Generates a timestamped stream of FramePrediction objects."""
    dt = 1.0 / fps
    stream = []
    t = start_time
    for label, cid, conf, has_hand in sequence:
        if has_hand:
            p = FramePrediction(
                class_id=cid,
                label=label,
                confidence=conf,
                timestamp=t,
                has_hand=True,
                top_margin=conf - 0.1,
            )
        else:
            p = FramePrediction.no_hand(timestamp=t)
        stream.append(p)
        t += dt
    return stream


def run_sequence(
    engine: TemporalEngine,
    stream: list[FramePrediction],
) -> tuple[list[StableSign], list[float]]:
    """Runs a stream of frames through the engine, collecting emitted tokens and per-frame latency."""
    emitted = []
    latencies_ms = []
    for p in stream:
        res = engine.process_frame(p)
        latencies_ms.append(res.processing_latency_ms)
        if res.has_emitted_sign:
            assert res.stable_sign is not None
            emitted.append(res.stable_sign)
    return emitted, latencies_ms


def main() -> int:
    print("=" * 70)
    print("SIGNBRIDGE — STAGE 7 SIMULATION VERIFICATION & BENCHMARK HARNESS")
    print("=" * 70)

    # Standard configuration: 250 ms dwell, 400 ms window, 250 ms hand loss grace
    config = TemporalEngineConfig()

    all_latencies: list[float] = []
    all_dwell_delays: list[float] = []
    tests_passed = 0
    total_tests = 10

    # -------------------------------------------------------------
    # Scenario 1: Consistent Sign (A A A A A A A A A A -> A)
    # -------------------------------------------------------------
    print("\n[SCENARIO 1] Consistent Sign (A held for 15 frames at 30 FPS = 500 ms)")
    engine = TemporalEngine(config)
    stream1 = generate_frame_stream([("A", 0, 0.92, True)] * 15, fps=30.0)
    emitted1, lats1 = run_sequence(engine, stream1)
    all_latencies.extend(lats1)
    labels1 = [s.label for s in emitted1]
    print(f"  Emitted: {labels1}")
    if labels1 == ["A"]:
        print("  [PASS] Scenario 1: Emitted single stable 'A'")
        all_dwell_delays.append(emitted1[0].duration_ms)
        tests_passed += 1
    else:
        print(f"  [FAIL] Scenario 1: Expected ['A'], got {labels1}")

    # -------------------------------------------------------------
    # Scenario 2: Single-frame Jitter (A A A B A A A A A A -> A)
    # -------------------------------------------------------------
    print("\n[SCENARIO 2] Single-Frame Jitter Rejection (A A A B A A A A A A)")
    engine = TemporalEngine(config)
    seq2 = [("A", 0, 0.92, True)] * 3 + [("B", 1, 0.88, True)] + [("A", 0, 0.92, True)] * 10
    stream2 = generate_frame_stream(seq2, fps=30.0)
    emitted2, lats2 = run_sequence(engine, stream2)
    all_latencies.extend(lats2)
    labels2 = [s.label for s in emitted2]
    print(f"  Emitted: {labels2}")
    if labels2 == ["A"]:
        print("  [PASS] Scenario 2: Jitter rejected, emitted single stable 'A'")
        tests_passed += 1
    else:
        print(f"  [FAIL] Scenario 2: Expected ['A'], got {labels2}")

    # -------------------------------------------------------------
    # Scenario 3: Sign Transition (A held 500ms, then B held 500ms)
    # -------------------------------------------------------------
    print("\n[SCENARIO 3] Sign Transition (A x 15 frames, then B x 15 frames)")
    engine = TemporalEngine(config)
    seq3 = [("A", 0, 0.92, True)] * 15 + [("B", 1, 0.94, True)] * 15
    stream3 = generate_frame_stream(seq3, fps=30.0)
    emitted3, lats3 = run_sequence(engine, stream3)
    all_latencies.extend(lats3)
    labels3 = [s.label for s in emitted3]
    print(f"  Emitted: {labels3}")
    if labels3 == ["A", "B"]:
        print("  [PASS] Scenario 3: Emitted sequential 'A' then 'B'")
        tests_passed += 1
    else:
        print(f"  [FAIL] Scenario 3: Expected ['A', 'B'], got {labels3}")

    # -------------------------------------------------------------
    # Scenario 4: Duplicate Suppression (A held for 45 frames = 1.5s)
    # -------------------------------------------------------------
    print("\n[SCENARIO 4] Duplicate Suppression (A continuously held for 45 frames)")
    engine = TemporalEngine(config)
    seq4 = [("A", 0, 0.93, True)] * 45
    stream4 = generate_frame_stream(seq4, fps=30.0)
    emitted4, lats4 = run_sequence(engine, stream4)
    all_latencies.extend(lats4)
    labels4 = [s.label for s in emitted4]
    print(f"  Emitted count: {len(labels4)}, tokens: {labels4}")
    if labels4 == ["A"]:
        print("  [PASS] Scenario 4: Duplicate suppression succeeded (only 1 'A' emitted)")
        tests_passed += 1
    else:
        print(f"  [FAIL] Scenario 4: Expected exactly ['A'], got {labels4}")

    # -------------------------------------------------------------
    # Scenario 5: Alternating Noise (A B A B A B A B -> No emission)
    # -------------------------------------------------------------
    print("\n[SCENARIO 5] Rapid Alternating Noise (A B A B A B A B A B)")
    engine = TemporalEngine(config)
    seq5 = [("A", 0, 0.85, True), ("B", 1, 0.85, True)] * 10
    stream5 = generate_frame_stream(seq5, fps=30.0)
    emitted5, lats5 = run_sequence(engine, stream5)
    all_latencies.extend(lats5)
    labels5 = [s.label for s in emitted5]
    print(f"  Emitted: {labels5}")
    if len(labels5) == 0:
        print("  [PASS] Scenario 5: Rapid noise rejected with zero emissions")
        tests_passed += 1
    else:
        print(f"  [FAIL] Scenario 5: Expected [], got {labels5}")

    # -------------------------------------------------------------
    # Scenario 6: Hand Loss Within Grace Period (A x 5, NO_HAND x 3 = 100ms, A x 10)
    # -------------------------------------------------------------
    print("\n[SCENARIO 6] Hand Loss Within Grace Period (100 ms loss < 250 ms grace)")
    engine = TemporalEngine(config)
    seq6 = (
        [("A", 0, 0.91, True)] * 5
        + [("", -1, 0.0, False)] * 3  # 100 ms at 30 fps
        + [("A", 0, 0.91, True)] * 10
    )
    stream6 = generate_frame_stream(seq6, fps=30.0)
    emitted6, lats6 = run_sequence(engine, stream6)
    all_latencies.extend(lats6)
    labels6 = [s.label for s in emitted6]
    print(f"  Emitted: {labels6}")
    if labels6 == ["A"]:
        print("  [PASS] Scenario 6: Hand loss tolerated within grace period, 'A' emitted")
        tests_passed += 1
    else:
        print(f"  [FAIL] Scenario 6: Expected ['A'], got {labels6}")

    # -------------------------------------------------------------
    # Scenario 7: Hand Loss Exceeding Grace Period (A x 5, NO_HAND x 15 = 500ms, A x 15)
    # -------------------------------------------------------------
    print("\n[SCENARIO 7] Hand Loss Exceeding Grace Period (500 ms loss > 250 ms grace)")
    # Using neutral_release or on_change with long hand loss
    engine = TemporalEngine(config)
    seq7 = (
        [("A", 0, 0.91, True)] * 5  # Dwell ~ 166 ms (not yet committed)
        + [("", -1, 0.0, False)] * 15  # 500 ms hand loss -> resets candidate
        + [("A", 0, 0.91, True)] * 15  # Fresh start for A -> committed
    )
    stream7 = generate_frame_stream(seq7, fps=30.0)
    emitted7, lats7 = run_sequence(engine, stream7)
    all_latencies.extend(lats7)
    labels7 = [s.label for s in emitted7]
    print(f"  Emitted: {labels7}")
    if labels7 == ["A"]:
        print("  [PASS] Scenario 7: State properly reset on long hand loss, fresh 'A' committed")
        tests_passed += 1
    else:
        print(f"  [FAIL] Scenario 7: Expected ['A'], got {labels7}")

    # -------------------------------------------------------------
    # Scenario 8: Low Confidence Filtering (A with 0.50 conf < 0.70 threshold)
    # -------------------------------------------------------------
    print("\n[SCENARIO 8] Low-Confidence Rejection (A with conf=0.55 < 0.70 threshold)")
    engine = TemporalEngine(config)
    seq8 = [("A", 0, 0.55, True)] * 20
    stream8 = generate_frame_stream(seq8, fps=30.0)
    emitted8, lats8 = run_sequence(engine, stream8)
    all_latencies.extend(lats8)
    labels8 = [s.label for s in emitted8]
    print(f"  Emitted: {labels8}")
    if len(labels8) == 0:
        print("  [PASS] Scenario 8: Low confidence predictions filtered out")
        tests_passed += 1
    else:
        print(f"  [FAIL] Scenario 8: Expected [], got {labels8}")

    # -------------------------------------------------------------
    # Scenario 9: FPS Variation (15 FPS, 24 FPS, 30 FPS)
    # -------------------------------------------------------------
    print("\n[SCENARIO 9] FPS Variation Robustness (15 FPS, 24 FPS, 30 FPS)")
    fps_passed = True
    for test_fps in [15.0, 24.0, 30.0]:
        engine_fps = TemporalEngine(config)
        # Hold for 500 ms at test_fps
        frame_count = int(np.ceil(0.50 * test_fps))
        stream_fps = generate_frame_stream([("C", 2, 0.95, True)] * frame_count, fps=test_fps)
        emitted_fps, lats_fps = run_sequence(engine_fps, stream_fps)
        all_latencies.extend(lats_fps)
        labels_fps = [s.label for s in emitted_fps]
        if labels_fps != ["C"]:
            print(f"    Failed at {test_fps} FPS: expected ['C'], got {labels_fps}")
            fps_passed = False
        else:
            print(f"    {test_fps} FPS: Emitted ['C'] correctly at dwell {emitted_fps[0].duration_ms:.1f} ms")

    if fps_passed:
        print("  [PASS] Scenario 9: Behavior invariant to camera FPS variation")
        tests_passed += 1
    else:
        print("  [FAIL] Scenario 9: FPS variation failure")

    # -------------------------------------------------------------
    # Scenario 10: Rapid Sign Changes (A held 400ms -> B held 400ms -> C held 400ms)
    # -------------------------------------------------------------
    print("\n[SCENARIO 10] Sequential Sign Sequence (A -> B -> C without duplicates)")
    engine = TemporalEngine(config)
    seq10 = (
        [("A", 0, 0.92, True)] * 12   # 400 ms
        + [("B", 1, 0.94, True)] * 12  # 400 ms
        + [("C", 2, 0.96, True)] * 12  # 400 ms
    )
    stream10 = generate_frame_stream(seq10, fps=30.0)
    emitted10, lats10 = run_sequence(engine, stream10)
    all_latencies.extend(lats10)
    labels10 = [s.label for s in emitted10]
    print(f"  Emitted sequence: {labels10}")
    if labels10 == ["A", "B", "C"]:
        print("  [PASS] Scenario 10: Sequential signs A -> B -> C recognized cleanly")
        tests_passed += 1
    else:
        print(f"  [FAIL] Scenario 10: Expected ['A', 'B', 'C'], got {labels10}")

    # -------------------------------------------------------------
    # Latency & Delay Profiling Summary
    # -------------------------------------------------------------
    mean_lat = float(np.mean(all_latencies))
    med_lat = float(np.median(all_latencies))
    p95_lat = float(np.percentile(all_latencies, 95))

    print("\n" + "=" * 70)
    print("STAGE 7 TEMPORAL PERFORMANCE BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total Simulation Frames Processed: {len(all_latencies)}")
    print(f"Mean Per-Frame Latency:            {mean_lat:.4f} ms")
    print(f"Median Per-Frame Latency:          {med_lat:.4f} ms")
    print(f"p95 Per-Frame Latency:             {p95_lat:.4f} ms")
    print(f"Estimated Max Temporal Throughput: {1000.0 / mean_lat:.1f} frames/sec")

    print("\nStabilization Delay Profile (configured minimum dwell = 300.0 ms):")
    if all_dwell_delays:
        print(f"  Mean Stabilization Dwell:        {np.mean(all_dwell_delays):.1f} ms")
        print(f"  Min Stabilization Dwell:         {np.min(all_dwell_delays):.1f} ms")
        print(f"  Max Stabilization Dwell:         {np.max(all_dwell_delays):.1f} ms")

    print("\n" + "=" * 70)
    print(f"SIMULATION SUITE RESULT: {tests_passed}/{total_tests} SCENARIOS PASSED")
    print("=" * 70)

    return 0 if tests_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
