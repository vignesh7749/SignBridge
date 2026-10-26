# SignBridge — Final Visual & State Correction Report

**Project**: SignBridge — Two-Way American Sign Language (ASL) Communication System  
**Status**: Completed & Verified  
**Tests**: 215 Passed (100% Pass Rate)  
**FeatureSchema**: Preserved strictly at v1.0 (94D alphabet feature vector, unaugmented)  
**Stages**: Stages 0 through 12 preserved intact without introducing any new development stage  

---

## 1. Executive Summary

This corrective patch resolves the visual overlapping, stale translation buffer accumulation, and state leakage issues observed during Demo Video playback in SignBridge.

Key achievements:
1. **Strict Live / Demo State Isolation**: Implemented independent operational pipelines (`_live_detector`, `_demo_detector`, separate temporal engines, separate translation engines, and separate sequence smoothers). Transitioning between Live Camera and Demo Video immediately purges all transient buffers, dwell timers, active words, sentences, and speech states.
2. **Dedicated Source Ownership**: Live webcam frames strictly route to the live pipeline; demo video frames strictly route to the demo pipeline. When Demo Video is selected, live camera capture is stopped cleanly.
3. **Truthful Demo vs. Detection Presentation**: Eradicated confusion between demo source metadata and model predictions. The UI explicitly separates `DEMO: HELP` from `DETECTED: BAD (80%)`, adding a `STATUS: MISMATCH / UNVERIFIED` badge rather than faking classification or overwriting ground truth.
4. **Stale Sentence Eradication & Loop Protection**: Demo playback now starts with an empty sentence buffer (`SENTENCE: Waiting for words...`). Word-sign mode (`WORD_SIGNS`) is automatically engaged during demo playback to prevent accidental fingerspelling emissions (e.g., `'A'`). When demo videos loop, sequence buffers reset and duplicate words are suppressed, permanently preventing `A BAD BAD BAD`.
5. **3-Row Non-Overlapping Viewport HUD**: Removed legacy burned-in text layers from `src/signbridge/pipeline/hud.py` and implemented a 3-row layout in `src/signbridge/ui/layout.py`:
   - **Row 1 (`oy+12`)**: Source mode badge (`MODE: DEMO VIDEO [HELP]` / `MODE: LIVE CAMERA`)
   - **Row 2 (`oy+38`)**: Multi-hand telemetry (`HANDS: 2 | PRIMARY: RIGHT 100% | SECONDARY: LEFT 90%`)
   - **Row 3 (`oy+64`)**: Prediction telemetry (`PREDICTION: BAD | CONFIDENCE: 80% [MISMATCH]`)

---

## 2. Root Cause Analysis & Empirical Investigations

### 2.1 Investigation: Why `help.mp4` Predicts `BAD`

A frame-by-frame diagnosis was conducted using `scratch/diagnose_help.py` and `scratch/check_word_classifier.py` on `datasets/samples/word_signs/help.mp4`:
- **Video Properties**: 35 frames, 30 FPS, 640x480 resolution.
- **Vision Pipeline**: MediaPipe Hand Landmark Detector detects both hands from frames 7 to 34 (Primary: Right hand at 100% confidence, Secondary: Left hand at 90% confidence).
- **Classification Flow**:
  1. In frames 7–15, hand shape resembles a fist as hands move into position. In hybrid `AUTO` mode, the static alphabet classifier matched this to letter `'A'`.
  2. At frame 22, the 20-frame trajectory buffer filled. The extracted 2,560D trajectory vector was evaluated by `models/word_sign/model.joblib`.
  3. The model output was `BAD` with 99.9% probability.
- **Root Cause**: The word classifier (`WordSignClassifier`) was trained via `scripts/train_word_sign_model.py` using synthetic trajectory generation with unnormalized spatial palm scaling and artificial translation noise. Because synthetic distributions do not perfectly match human kinematic variations in real video, the trajectory for `help.mp4` fell into the decision region for class `BAD`.
- **Resolution**: In strict accordance with project rules, **recognition was NOT faked**. The system displays the truthful measurement (`DEMO: HELP`, `DETECTED: BAD`, `STATUS: MISMATCH`) and switches router mode to `WORD_SIGNS` so letters like `'A'` are not emitted during word sign playback.

### 2.2 Investigation: Why `SENTENCE: A BAD BAD BAD` Occurred

- **Letter 'A' Emission**: In hybrid `AUTO` mode, transient static hand shapes at the start of the gesture triggered alphabet emissions.
- **Repeated 'BAD' Words**: `VideoFileReader` loops continuously upon reaching EOF. On each loop, the word classifier re-emitted `'BAD'`, appending it repeatedly to `translation_engine.completed_words`.
- **Resolution**:
  1. On demo switch, `SignBridgePipeline` sets `router.set_mode(RecognitionMode.WORD_SIGNS)`.
  2. `SignBridgePipeline._reset_operational_state()` resets the translation engine to empty.
  3. `SignBridgePipeline.process_frame()` detects when the demo video loops (`frame_id < _last_demo_frame_id`) and resets the word smoother and temporal engine.
  4. In `DEMO_VIDEO` mode, consecutive duplicate word additions to the sentence buffer are suppressed.

### 2.3 Investigation: Viewport Text Overlaps

- **Root Cause**: Two independent rendering systems were writing text to the top-left area:
  1. `src/signbridge/pipeline/hud.py` was burning legacy Stage 10 header text (`STAGE 10 REAL-TIME VOICE TRANSLATION`, `VOICE: ON [READY]`, `SIGNBRIDGE`) and transient banners (`WORD COMPLETED: 'BAD'`) directly into the image frame at `y=14` to `y=70`.
  2. `src/signbridge/ui/layout.py` was drawing in-feed pill badges on top of that resized frame at `ox+12, oy+12` and `ox+12, oy+38`.
- **Resolution**:
  1. Cleaned `src/signbridge/pipeline/hud.py` to remove duplicate header text and transient completion banners from the raw frame.
  2. Reorganized `src/signbridge/ui/layout.py` into 3 non-overlapping rows with dedicated vertical spacing:
     - Row 1: `oy + 12`
     - Row 2: `oy + 38`
     - Row 3: `oy + 64`
  3. Placed helper guidance at the bottom of the viewport (`oy + nh - 36`).

---

## 3. Architecture & State Isolation Implementation

### 3.1 Separate Pipelines in `SignBridgePipeline`

```python
class SignBridgePipeline:
    def __init__(self, ...):
        # Dedicated Vision Detectors
        self._live_detector = MediaPipeHandDetector(config=self.config.vision)
        self._demo_detector = MediaPipeHandDetector(config=self.config.vision)
        self.detector = self._live_detector

        # Dedicated Temporal Engines
        self._live_temporal_engine = TemporalEngine(...)
        self._demo_temporal_engine = TemporalEngine(...)
        self.temporal_engine = self._live_temporal_engine

        # Dedicated Translation Engines
        self._live_translation_engine = TranslationEngine(...)
        self._demo_translation_engine = TranslationEngine(...)
        self.translation_engine = self._live_translation_engine

        # Dedicated Word Smoothers
        self._live_word_smoother = WordSignTemporalSmoother(...)
        self._demo_word_smoother = WordSignTemporalSmoother(...)
        self.word_smoother = self._live_word_smoother
```

### 3.2 Clean Transitions

```python
def switch_to_demo_video(self, reader: ICameraSource) -> None:
    # 1. Halt live camera
    if self._active_source == FrameSource.LIVE_CAMERA and self.camera:
        self.camera.stop()
    # 2. Assign dedicated demo instances
    self.detector = self._demo_detector
    self.temporal_engine = self._demo_temporal_engine
    self.translation_engine = self._demo_translation_engine
    self.word_smoother = self._demo_word_smoother
    self.router.word_smoother = self._demo_word_smoother
    # 3. Assign demo reader
    self.camera = reader
    self.camera.start()
    self._active_source = FrameSource.DEMO_VIDEO
    self.router.set_mode(RecognitionMode.WORD_SIGNS)
    # 4. Comprehensive state reset
    self._reset_operational_state()
```

---

## 4. Test Verification Summary

Full regression test run executed via pytest:
```
============================ 215 passed in 33.72s =============================
```

### Key Test Suites Verified:
- `tests/test_final_stabilization.py`: 13 passed
  - `test_manual_speak_bypasses_duplicate_suppression`: PASSED
  - `test_retrigger_after_different_sign`: PASSED
  - `test_speak_words_vs_speak_letters_routing`: PASSED
  - `test_uncertain_state_triggered_on_low_confidence`: PASSED
  - `test_recognition_card_renders_uncertain`: PASSED
  - `test_frame_source_enum`: PASSED
  - `test_camera_and_demo_state_isolation`: PASSED
  - `test_diagnostics_overlay_rendering`: PASSED
  - `test_strict_live_demo_state_reset_and_clean_sentence`: PASSED
  - `test_dedicated_detector_and_engine_instances`: PASSED
  - `test_demo_video_loop_suppresses_duplicate_words`: PASSED
  - `test_demo_target_vs_detected_distinction`: PASSED
  - `test_viewport_hud_rows_non_overlapping_positions`: PASSED
- `tests/test_tts_event_trigger.py`: 7 passed
- `tests/test_word_sign_video_input.py`: 4 passed
- `tests/test_ui.py`: 15 passed
- `tests/test_pipeline.py`: 13 passed
- `tests/test_translation.py`: 19 passed
- `tests/test_camera.py`: 9 passed
- `tests/test_vision.py`: 13 passed

All 215 tests across Stages 0–12 are fully green.
