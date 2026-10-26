"""Stage 9 Performance and Latency Benchmark.

Executes the full end-to-end SignBridge pipeline on real machine hardware,
measuring stage-by-stage latencies, throughput, frame drops, and end-to-end responsiveness.
"""

from pathlib import Path
import sys
import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signbridge.camera.base import CameraStatus
from signbridge.core.config import CameraConfig, VisionConfig
from signbridge.pipeline import PipelineConfig, SignBridgePipeline
from signbridge.temporal.config import TemporalEngineConfig
from signbridge.translation.config import TranslationConfig


def run_benchmark(num_frames: int = 100) -> dict[str, float]:
    print("=" * 70)
    print(f"SIGNBRIDGE — STAGE 9 END-TO-END BENCHMARK ({num_frames} FRAMES)")
    print("=" * 70)

    config = PipelineConfig(
        camera=CameraConfig(width=1280, height=720, fps=30, flip_horizontal=True),
        vision=VisionConfig(max_num_hands=2, min_detection_confidence=0.65),
        temporal=TemporalEngineConfig(minimum_dwell_ms=250.0),
        translation=TranslationConfig(),
        draw_hud=True,
        draw_skeleton=True,
    )

    pipeline = SignBridgePipeline(config=config)
    pipeline.start()

    # Latency storage arrays (in milliseconds)
    cam_lats: list[float] = []
    det_lats: list[float] = []
    feat_lats: list[float] = []
    cls_lats: list[float] = []
    temp_lats: list[float] = []
    trans_lats: list[float] = []
    hud_lats: list[float] = []
    total_lats: list[float] = []
    frame_intervals: list[float] = []

    successful_frames = 0
    hands_detected_count = 0
    t_prev_frame = time.perf_counter()
    bench_start = time.perf_counter()

    print(f"Benchmarking live pipeline over {num_frames} frames...")

    for i in range(num_frames):
        t_frame_start = time.perf_counter()
        success, output = pipeline.process_next_frame(timeout=1.0)
        t_frame_end = time.perf_counter()

        if not success or output is None:
            continue

        successful_frames += 1
        interval = t_frame_start - t_prev_frame
        if i > 0:
            frame_intervals.append(interval)
        t_prev_frame = t_frame_start

        lats = output.latencies
        cam_lats.append(lats.camera_capture_ms)
        det_lats.append(lats.hand_detection_ms)
        feat_lats.append(lats.feature_extraction_ms)
        cls_lats.append(lats.classifier_ms)
        temp_lats.append(lats.temporal_ms)
        trans_lats.append(lats.translation_ms)
        hud_lats.append(lats.hud_render_ms)
        total_lats.append(lats.total_pipeline_ms)

        if output.primary_hand is not None:
            hands_detected_count += 1

        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames... (current total latency: {lats.total_pipeline_ms:.2f} ms)")

    bench_duration = time.perf_counter() - bench_start
    final_state = pipeline.get_state()
    pipeline.stop()

    # Compute Statistics
    mean_fps = successful_frames / bench_duration if bench_duration > 0 else 0.0
    instantaneous_fps = [1.0 / dt for dt in frame_intervals if dt > 0]
    median_fps = float(np.median(instantaneous_fps)) if instantaneous_fps else mean_fps
    detection_rate = (hands_detected_count / successful_frames * 100.0) if successful_frames > 0 else 0.0

    mean_total = float(np.mean(total_lats)) if total_lats else 0.0
    median_total = float(np.median(total_lats)) if total_lats else 0.0
    p95_total = float(np.percentile(total_lats, 95)) if total_lats else 0.0

    print("\n" + "=" * 70)
    print("BENCHMARK EXECUTION RESULTS")
    print("=" * 70)
    print(f"Total Frames Requested:     {num_frames}")
    print(f"Successful Frames:          {successful_frames}")
    print(f"Dropped Frames:             {final_state.dropped_frames}")
    print(f"Hand Detection Rate:        {detection_rate:.1f}%")
    print(f"Benchmark Wall Duration:    {bench_duration:.2f} s")
    print(f"Mean Pipeline FPS:          {mean_fps:.1f} FPS")
    print(f"Median Pipeline FPS:        {median_fps:.1f} FPS")
    print("-" * 70)
    print("STAGE-BY-STAGE LATENCY BREAKDOWN (Mean / Median / p95):")
    print(f"  1. Camera Read:           {np.mean(cam_lats):.2f} ms / {np.median(cam_lats):.2f} ms / {np.percentile(cam_lats, 95):.2f} ms")
    print(f"  2. Hand Detection:        {np.mean(det_lats):.2f} ms / {np.median(det_lats):.2f} ms / {np.percentile(det_lats, 95):.2f} ms")
    print(f"  3. Feature Extraction:    {np.mean(feat_lats):.4f} ms / {np.median(feat_lats):.4f} ms / {np.percentile(feat_lats, 95):.4f} ms")
    print(f"  4. Classifier Inference:  {np.mean(cls_lats):.4f} ms / {np.median(cls_lats):.4f} ms / {np.percentile(cls_lats, 95):.4f} ms")
    print(f"  5. Temporal Smoothing:    {np.mean(temp_lats):.4f} ms / {np.median(temp_lats):.4f} ms / {np.percentile(temp_lats, 95):.4f} ms")
    print(f"  6. Translation Engine:    {np.mean(trans_lats):.4f} ms / {np.median(trans_lats):.4f} ms / {np.percentile(trans_lats, 95):.4f} ms")
    print(f"  7. HUD Composition:       {np.mean(hud_lats):.2f} ms / {np.median(hud_lats):.2f} ms / {np.percentile(hud_lats, 95):.2f} ms")
    print("-" * 70)
    # Benchmark Active Hand Features & Classifier Inference (1,000 samples)
    from signbridge.features.extractor import HandFeatureExtractor
    from signbridge.model.predictor import get_default_predictor
    from signbridge.vision.landmarks import BoundingBox, DetectedHand, LandmarkPoint

    extractor = HandFeatureExtractor()
    predictor = get_default_predictor()
    dummy_points = [
        LandmarkPoint(i, f"LM_{i}", 0.5 + 0.01 * (i % 5), 0.5 + 0.01 * (i // 5), 0.0, 320, 240, 0.0)
        for i in range(21)
    ]
    raw_coords = np.zeros((21, 3), dtype=np.float32)
    raw_coords[0] = [0.5, 0.7, 0.0]
    raw_coords[9] = [0.5, 0.4, 0.0]
    dummy_hand = DetectedHand(
        handedness="Right",
        landmarks=dummy_points,
        raw_array=raw_coords,
        handedness_confidence=0.98,
        bbox=BoundingBox(100, 100, 400, 400),
        pixel_array=np.zeros((21, 2), dtype=np.int32),
        score=0.98,
    )

    feat_bench_lats = []
    cls_bench_lats = []
    for _ in range(1000):
        t0 = time.perf_counter()
        fv = extractor.extract(dummy_hand)
        feat_bench_lats.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        pred = predictor.predict(fv.features)
        cls_bench_lats.append((time.perf_counter() - t0) * 1000.0)

    print("\nACTIVE HAND INFERENCE BENCHMARK (1,000 SAMPLES):")
    print(f"  Feature Extraction:       {np.mean(feat_bench_lats):.4f} ms / {np.median(feat_bench_lats):.4f} ms / {np.percentile(feat_bench_lats, 95):.4f} ms")
    print(f"  Classifier Inference:     {np.mean(cls_bench_lats):.4f} ms / {np.median(cls_bench_lats):.4f} ms / {np.percentile(cls_bench_lats, 95):.4f} ms")
    print(f"  Active Hand E2E Pipeline: {mean_total + np.mean(feat_bench_lats) + np.mean(cls_bench_lats):.2f} ms")
    print("=" * 70)

    return {
        "num_frames": num_frames,
        "successful_frames": successful_frames,
        "dropped_frames": final_state.dropped_frames,
        "detection_rate": detection_rate,
        "mean_fps": mean_fps,
        "median_fps": median_fps,
        "mean_total_ms": mean_total,
        "median_total_ms": median_total,
        "p95_total_ms": p95_total,
        "feat_mean_ms": float(np.mean(feat_bench_lats)),
        "cls_mean_ms": float(np.mean(cls_bench_lats)),
    }


if __name__ == "__main__":
    run_benchmark(num_frames=60)

