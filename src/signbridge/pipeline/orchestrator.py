"""End-to-end pipeline orchestrator for SignBridge."""

from collections import deque
import threading
import time
from typing import Any, Optional, Sequence, Union
import cv2
import numpy as np

from signbridge.camera.base import CameraStatus, FramePacket, ICameraSource
from signbridge.camera.capture import CameraManager
from signbridge.core.logger import get_logger
from signbridge.features.extractor import FeatureVector, HandFeatureExtractor
from signbridge.features.validator import LandmarkValidationError
from signbridge.model.base import PredictionResult
from signbridge.model.predictor import SignPredictor, get_default_predictor
from signbridge.pipeline.config import PipelineConfig
from signbridge.pipeline.contracts import (
    FrameProcessingOutput,
    FrameSource,
    PipelineError,
    StageLatencies,
)
from signbridge.pipeline.hud import PipelineHUD
from signbridge.pipeline.state import PipelineRuntimeState
from signbridge.temporal.contracts import FramePrediction, TemporalFrameResult, TemporalState
from signbridge.temporal.engine import TemporalEngine
from signbridge.translation.contracts import TranslationEventType, TranslationResult, TranslationState
from signbridge.translation.engine import TranslationEngine
from signbridge.tts import SpeechController, SpeechEvent, SpeechEventType, TTSConfig, TTSStatus
from signbridge.word_sign.emergency import EmergencyModalManager
from signbridge.vision.detector import HandDetectionResult, IHandDetector, MediaPipeHandDetector
from signbridge.vision.landmarks import DetectedHand
from signbridge.word_sign.model import WordSignClassifier, load_default_word_sign_model
from signbridge.word_sign.router import RecognitionMode, RecognitionRouter, RoutedRecognitionResult
from signbridge.word_sign.temporal import WordSignTemporalSmoother

logger = get_logger("pipeline.orchestrator")


class SignBridgePipeline:
    """Orchestrates end-to-end real-time sign language recognition and translation.

    Coordinates:
      CameraSource -> HandDetector -> FeatureExtractor -> SignPredictor ->
      TemporalEngine -> Router -> TranslationEngine -> Presentation HUD.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        camera: Optional[ICameraSource] = None,
        detector: Optional[IHandDetector] = None,
        extractor: Optional[HandFeatureExtractor] = None,
        predictor: Optional[SignPredictor] = None,
        temporal_engine: Optional[TemporalEngine] = None,
        translation_engine: Optional[TranslationEngine] = None,
        speech_controller: Optional[SpeechController] = None,
        hud: Optional[PipelineHUD] = None,
        router: Optional[RecognitionRouter] = None,
        word_classifier: Optional[WordSignClassifier] = None,
        word_smoother: Optional[WordSignTemporalSmoother] = None,
        emergency_manager: Optional[EmergencyModalManager] = None,
    ) -> None:
        self.config = config or PipelineConfig()

        # Injected or default components
        self.camera = camera or CameraManager(
            config=self.config.camera,
            auto_reconnect=self.config.auto_reconnect,
            max_reconnect_attempts=self.config.max_reconnect_attempts,
        )
        # Dedicated LIVE vs DEMO detection and recognition subsystems
        self._live_detector: IHandDetector = detector or MediaPipeHandDetector(self.config.vision)
        self._demo_detector: IHandDetector = MediaPipeHandDetector(self.config.vision)
        self.detector: IHandDetector = self._live_detector

        self.extractor = extractor or HandFeatureExtractor(canonicalize_handedness=True)
        self.predictor = predictor or (
            SignPredictor(self.config.model_dir) if self.config.model_dir else get_default_predictor()
        )

        self._live_temporal_engine: TemporalEngine = temporal_engine or TemporalEngine(self.config.temporal)
        self._demo_temporal_engine: TemporalEngine = TemporalEngine(self.config.temporal)
        self.temporal_engine: TemporalEngine = self._live_temporal_engine

        self._live_translation_engine: TranslationEngine = translation_engine or TranslationEngine(self.config.translation)
        self._demo_translation_engine: TranslationEngine = TranslationEngine(self.config.translation)
        self.translation_engine: TranslationEngine = self._live_translation_engine

        self.speech_controller = speech_controller or SpeechController(config=self.config.tts)
        self.hud = hud or PipelineHUD()

        # Word-Sign Recognition & Routing
        self.word_classifier = word_classifier or load_default_word_sign_model()
        self._live_word_smoother: Optional[WordSignTemporalSmoother] = word_smoother or (
            WordSignTemporalSmoother(self.word_classifier) if self.word_classifier else None
        )
        self._demo_word_smoother: Optional[WordSignTemporalSmoother] = (
            WordSignTemporalSmoother(self.word_classifier) if self.word_classifier else None
        )
        self.word_smoother = self._live_word_smoother

        self.router = router or RecognitionRouter(
            word_smoother=self.word_smoother,
            initial_mode=RecognitionMode.AUTO,
        )
        self.emergency_manager = emergency_manager or EmergencyModalManager()

        # Operational state
        self._is_running = False
        self._is_paused = False
        self._lock = threading.RLock()
        self._frame_count = 0
        self._dropped_frames = 0
        self._start_time = time.time()
        self._last_error: Optional[PipelineError] = None
        self._active_source: FrameSource = FrameSource.LIVE_CAMERA
        self._live_camera: ICameraSource = self.camera
        self.target_demo_sign: Optional[str] = None

        # Rolling telemetry for FPS and Latency estimation
        self._proc_times: deque[float] = deque(maxlen=60)
        self._last_proc_fps_time = time.time()
        self._proc_frame_counter = 0
        self._current_proc_fps = 0.0

        # Cached latest output and state
        self._latest_output: Optional[FrameProcessingOutput] = None
        self._last_spoken_display: Optional[str] = None
        self._runtime_state: PipelineRuntimeState = PipelineRuntimeState.initial()
        self.last_stable_token: Optional[str] = None

    @property
    def active_source(self) -> FrameSource:
        return self._active_source

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    def get_state(self) -> PipelineRuntimeState:
        """Thread-safe snapshot of the current pipeline operational state."""
        with self._lock:
            self._update_runtime_state()
            return self._runtime_state

    def start(self) -> None:
        """Starts camera capture and activates pipeline."""
        with self._lock:
            if self._is_running:
                logger.warning("Pipeline is already running.")
                return
            logger.info("Starting SignBridgePipeline...")
            self.camera.start()
            if self.config.tts.enabled:
                self.speech_controller.start()
            self._is_running = True
            self._is_paused = False
            self._start_time = time.time()
            self._update_runtime_state()
            logger.info("SignBridgePipeline started successfully.")

    def stop(self) -> None:
        """Gracefully stops camera and pipeline resources."""
        with self._lock:
            if not self._is_running:
                return
            logger.info("Stopping SignBridgePipeline...")
            self._is_running = False
            self.camera.stop()
            if self._live_camera and self._live_camera is not self.camera:
                try:
                    self._live_camera.stop()
                except Exception:
                    pass
            try:
                self._live_detector.close()
            except Exception:
                pass
            try:
                self._demo_detector.close()
            except Exception:
                pass
            self.speech_controller.shutdown()
            self._update_runtime_state()
            logger.info("SignBridgePipeline stopped cleanly.")

    def pause(self) -> None:
        """Pauses frame processing without shutting down camera or clearing text."""
        with self._lock:
            if self._is_paused:
                return
            self._is_paused = True
            if hasattr(self.camera, "pause"):
                self.camera.pause()
            self._update_runtime_state()
            logger.info("SignBridgePipeline paused.")

    def resume(self) -> None:
        """Resumes active frame processing and clears any stale frame buffer."""
        with self._lock:
            if not self._is_paused:
                return
            self._is_paused = False
            if hasattr(self.camera, "resume"):
                self.camera.resume()
            self._update_runtime_state()
            logger.info("SignBridgePipeline resumed.")

    # -------------------------------------------------------------------------
    # Stage 8 User Controls Integration
    # -------------------------------------------------------------------------

    def space(self) -> TranslationResult:
        """Triggers word boundary completion."""
        # 1. If translation engine has active in-progress letters, commit them
        if self.translation_engine.current_word:
            res = self.translation_engine.space()
            if (
                res.event_type == TranslationEventType.WORD_COMPLETED
                and self.config.tts.speak_words
                and res.completed_words
            ):
                last_w = res.completed_words[-1]
                word_str = getattr(last_w, "text", str(last_w))
                self.speech_controller.handle_word_completed(word_str)
            self._update_runtime_state()
            return res

        # 2. If an active word sign candidate is held/active but not yet committed, commit it
        active_ws = self._runtime_state.active_word_sign
        if active_ws:
            already_committed = False
            if self.translation_engine.completed_words:
                last_w = self.translation_engine.completed_words[-1]
                last_text = getattr(last_w, "text", str(last_w))
                if last_text == active_ws:
                    already_committed = True
            if not already_committed:
                res = self.translation_engine.process_word(
                    active_ws,
                    confidence=self._runtime_state.word_sign_confidence or 1.0,
                )
                if self.config.tts.speak_words and res.completed_words:
                    last_w = res.completed_words[-1]
                    word_str = getattr(last_w, "text", str(last_w))
                    self.speech_controller.handle_word_completed(word_str)
                self._update_runtime_state()
                return res

        res = self.translation_engine.space()
        self._update_runtime_state()
        return res

    def backspace(self) -> TranslationResult:
        """Deletes last letter or re-opens previous word."""
        res = self.translation_engine.backspace()
        self._update_runtime_state()
        return res

    def clear_word(self) -> TranslationResult:
        """Clears active in-flight word."""
        res = self.translation_engine.clear_word()
        self._update_runtime_state()
        return res

    def clear_sentence(self) -> TranslationResult:
        """Clears full sentence and active word, interrupting active speech."""
        res = self.translation_engine.clear_sentence()
        if self.word_smoother is not None:
            self.word_smoother.reset()
        self.router.reset()
        self.speech_controller.stop()
        self._update_runtime_state()
        return res

    def reset(self) -> TranslationResult:
        """Resets all translation buffers and state, interrupting active speech."""
        res = self.translation_engine.reset()
        if self.word_smoother is not None:
            self.word_smoother.reset()
        self.router.reset()
        self.temporal_engine.reset()
        self.last_stable_token = None
        self.speech_controller.stop()
        self.speech_controller.reset_spoken_state()
        self._last_spoken_display = None
        self._update_runtime_state()
        return res

    def undo(self) -> TranslationResult:
        """Restores previous translation snapshot."""
        res = self.translation_engine.undo()
        self._update_runtime_state()
        return res

    # -------------------------------------------------------------------------
    # Stage 10 Voice Controls Integration
    # -------------------------------------------------------------------------

    def speak_current_stable_sign(self) -> bool:
        """Explicitly speaks the current stable prediction or active text on demand."""
        st = self.get_state()
        candidate = (
            self.last_stable_token
            or st.active_word_sign
            or st.stable_token
            or st.current_word
            or st.predicted_label
            or st.sentence
        )
        if not candidate:
            logger.debug("speak_current_stable_sign: no candidate text found to speak.")
            return False
        return self.speech_controller.speak_manual(candidate)

    def speak_current_sentence(self) -> bool:
        """Speaks the active finalized sentence, active word, or current sign."""
        st = self.get_state()
        # 1. Primary priority: visible constructed completed sentence
        completed_sentence = " ".join(st.completed_words).strip()
        if completed_sentence:
            return self.speech_controller.speak_manual(completed_sentence)

        # 2. Secondary priority: active in-progress word being assembled
        cur_word = (st.current_word or self.translation_engine.current_word).strip()
        if cur_word:
            return self.speech_controller.speak_manual(cur_word)

        # 3. Tertiary priority: sentence text fallback
        full_sent = (st.sentence or self.translation_engine.sentence_text).strip()
        if full_sent:
            return self.speech_controller.speak_manual(full_sent)

        # 4. Quaternary priority: active word sign or stable token
        candidate = st.active_word_sign or st.stable_token
        if candidate:
            return self.speech_controller.speak_manual(candidate)

        # 5. Fallback: live predicted label if hand is detected
        if st.hand_detected and st.predicted_label and st.predicted_label not in ("—", "UNCERTAIN"):
            return self.speech_controller.speak_manual(st.predicted_label)

        # 6. Fallback: last stable token if available
        if self.last_stable_token:
            return self.speech_controller.speak_manual(self.last_stable_token)

        return False

    def _get_active_demo_target(self) -> Optional[str]:
        """Resolves the normalized uppercase label of the currently playing demo/reference sign."""
        if self._active_source != FrameSource.DEMO_VIDEO:
            return None
        target = getattr(self, "target_demo_sign", None)
        if not target and hasattr(self.camera, "video_path") and self.camera.video_path:
            stem = self.camera.video_path.stem.upper().replace("_SAMPLE", "").replace("_", " ")
            if "GOODBYE" in stem or "BYE" in stem:
                target = "GOODBYE"
            else:
                target = stem
        return target.upper() if target else None

    def speak_text(self, text: str) -> bool:
        """Explicitly enqueues manual speech utterance."""
        return self.speech_controller.speak_manual(text)

    def stop_speech(self) -> None:
        """Stops active speech and flushes pending utterances."""
        self.speech_controller.stop()

    def toggle_voice(self) -> bool:
        """Toggles voice synthesis enabled/disabled."""
        enabled = self.speech_controller.toggle_enabled()
        self._update_runtime_state()
        return enabled

    # -------------------------------------------------------------------------
    # Frame Processing
    # -------------------------------------------------------------------------

    def process_next_frame(self, timeout: float = 1.0) -> tuple[bool, Optional[FrameProcessingOutput]]:
        """Reads latest frame from camera and executes full forward pipeline.

        Returns:
            (success_bool, FrameProcessingOutput_or_None)
        """
        if not self._is_running:
            return False, None

        if self._is_paused:
            # While paused, return cached output or empty tick to keep UI alive
            time.sleep(0.03)
            return True, self._latest_output

        # Read latest frame from camera
        read_start = time.perf_counter()
        ok, packet = self.camera.read(timeout=timeout)
        cam_latency = (time.perf_counter() - read_start) * 1000.0

        if not ok or packet is None:
            # Camera read failed or timed out
            self._dropped_frames += 1
            if self.camera.status == CameraStatus.DISCONNECTED:
                self._last_error = PipelineError(
                    stage="camera",
                    message="Camera disconnected.",
                    recoverable=self.config.auto_reconnect,
                )
            self._update_runtime_state()
            return False, None

        # Execute pipeline on valid frame packet
        output = self.process_frame(packet, camera_capture_ms=cam_latency)
        return True, output

    def process_frame(
        self,
        packet: FramePacket,
        camera_capture_ms: float = 0.0,
    ) -> FrameProcessingOutput:
        """Processes a single FramePacket through all stages with error isolation."""
        t_total_start = time.perf_counter()
        latencies = StageLatencies(camera_capture_ms=camera_capture_ms)

        self._frame_count += 1
        frame_id = packet.frame_id
        timestamp = packet.timestamp
        frame = packet.data

        # Demo video loop detection
        if self._active_source == FrameSource.DEMO_VIDEO:
            if hasattr(self, "_last_demo_frame_id") and self._last_demo_frame_id is not None and frame_id < self._last_demo_frame_id:
                logger.info("Demo video looped (frame %d -> %d). Resetting temporal and sequence buffers.", self._last_demo_frame_id, frame_id)
                if self.word_smoother is not None:
                    self.word_smoother.reset()
                self.temporal_engine.reset()
            self._last_demo_frame_id = frame_id

        det_result: HandDetectionResult
        primary_hand: Optional[DetectedHand] = None
        feat_vector: Optional[FeatureVector] = None
        pred_result: Optional[PredictionResult] = None
        frame_pred: Optional[FramePrediction] = None
        temp_result: TemporalFrameResult
        trans_result: TranslationResult
        frame_error: Optional[PipelineError] = None

        # Stage 3: Hand Detection
        t0 = time.perf_counter()
        try:
            det_result = self.detector.detect(frame, frame_id=frame_id, timestamp=timestamp)
        except Exception as e:
            logger.error("Hand detection failed on frame %d: %s", frame_id, e)
            frame_error = PipelineError(stage="vision", message=str(e))
            det_result = HandDetectionResult(frame_id=frame_id, timestamp=timestamp, hands=[])
        latencies.hand_detection_ms = (time.perf_counter() - t0) * 1000.0

        # Multi-hand selection: pick primary hand and identify secondary hand
        secondary_hand: Optional[DetectedHand] = None
        if det_result.has_hands:
            primary_hand = self._select_primary_hand(det_result.hands)
            if len(det_result.hands) > 1:
                for h in det_result.hands:
                    if h is not primary_hand:
                        secondary_hand = h
                        break

        # Stage 4 & 6: Feature Extraction and Classification
        if primary_hand is not None:
            # Stage 4: Features
            t0 = time.perf_counter()
            try:
                feat_vector = self.extractor.extract(primary_hand, timestamp_ms=timestamp * 1000.0)
            except LandmarkValidationError as e:
                logger.warning("Landmark validation error on frame %d: %s", frame_id, e)
                feat_vector = None
            except Exception as e:
                logger.error("Feature extraction unexpected error on frame %d: %s", frame_id, e)
                frame_error = PipelineError(stage="features", message=str(e))
                feat_vector = None
            latencies.feature_extraction_ms = (time.perf_counter() - t0) * 1000.0

            # Stage 6: Classifier Inference
            t0 = time.perf_counter()
            if feat_vector is not None:
                try:
                    pred_result = self.predictor.predict(feat_vector.features)
                    frame_pred = FramePrediction.from_prediction_result(pred_result, timestamp, has_hand=True)
                except Exception as e:
                    logger.error("Classifier inference failed on frame %d: %s", frame_id, e)
                    frame_error = PipelineError(stage="model", message=str(e))
                    frame_pred = FramePrediction.no_hand(timestamp)
            else:
                frame_pred = FramePrediction.no_hand(timestamp)
            latencies.classifier_ms = (time.perf_counter() - t0) * 1000.0
        else:
            # No hand detected: safely notify downstream
            frame_pred = FramePrediction.no_hand(timestamp)

        # Stage 7: Temporal Stabilization
        t0 = time.perf_counter()
        try:
            temp_result = self.temporal_engine.process_frame(frame_pred)
        except Exception as e:
            logger.error("Temporal processing error on frame %d: %s", frame_id, e)
            self._last_error = PipelineError(stage="temporal", message=str(e))
            temp_result = TemporalFrameResult(
                state=TemporalState.IDLE,
                current_candidate=None,
                candidate_confidence=0.0,
                dwell_duration_ms=0.0,
                stability_ratio=0.0,
                is_hand_present=False,
                stable_sign=None,
                processing_latency_ms=0.0,
                frame_prediction=frame_pred,
            )
        latencies.temporal_ms = (time.perf_counter() - t0) * 1000.0

        # Stage 7.5: Recognition Router (Arbitrates between Alphabet and Word-Signs)
        alpha_candidate = temp_result.stable_sign.label if (temp_result.has_emitted_sign and temp_result.stable_sign is not None) else temp_result.current_candidate
        alpha_conf = temp_result.candidate_confidence
        alpha_emitted = (temp_result.has_emitted_sign and temp_result.stable_sign is not None)

        routed_result = self.router.route_frame(
            detection=det_result,
            alphabet_token_candidate=alpha_candidate,
            alphabet_confidence=alpha_conf,
            alphabet_is_emitted=alpha_emitted,
            current_time=timestamp,
        )

        # Stage 8: Translation Engine
        t0 = time.perf_counter()
        try:
            demo_target = self._get_active_demo_target()
            is_unsupported_demo = (
                self._active_source == FrameSource.DEMO_VIDEO
                and bool(demo_target)
                and len(demo_target) > 1
                and demo_target not in ("HELLO", "HELP", "YES", "NO", "STOP", "THANK YOU", "GOODBYE")
            )
            if routed_result.is_word and routed_result.is_newly_emitted and routed_result.token:
                should_append = True
                if is_unsupported_demo:
                    should_append = False
                elif self._active_source == FrameSource.DEMO_VIDEO:
                    if self.translation_engine.completed_words:
                        last_w = self.translation_engine.completed_words[-1]
                        last_text = getattr(last_w, "text", str(last_w))
                        if last_text == routed_result.token:
                            should_append = False
                elif self.translation_engine.completed_words:
                    last_w = self.translation_engine.completed_words[-1]
                    last_text = getattr(last_w, "text", str(last_w))
                    if last_text == routed_result.token and (timestamp - getattr(self, "_last_word_commit_time", 0.0)) < 2.0:
                        should_append = False
                    elif (not det_result.has_hands) or routed_result.confidence < self.config.min_classification_confidence:
                        should_append = False
                elif (not det_result.has_hands) or routed_result.confidence < self.config.min_classification_confidence:
                    should_append = False

                if should_append:
                    # Clear any intermediate alphabet tokens so they don't pollute sentence
                    if self.translation_engine._current_tokens:
                        self.translation_engine.clear_word()
                    trans_result = self.translation_engine.process_word(
                        routed_result.token, timestamp=timestamp, confidence=routed_result.confidence
                    )
                    self._last_word_commit_time = timestamp
                    if self.emergency_manager is not None:
                        self.emergency_manager.trigger_sign_detected(
                            routed_result.token, confidence=routed_result.confidence, timestamp=timestamp
                        )
                else:
                    trans_result = self._get_current_translation_result()
            elif (
                self.router.mode != RecognitionMode.WORD_SIGNS
                and (not routed_result.is_word)
                and routed_result.is_newly_emitted
                and temp_result.stable_sign is not None
            ):
                should_append_letter = True
                if self._active_source == FrameSource.DEMO_VIDEO:
                    if is_unsupported_demo or (demo_target and len(demo_target) > 1):
                        should_append_letter = False
                elif self.router.mode == RecognitionMode.AUTO:
                    if routed_result.word_prediction and routed_result.word_prediction.confidence >= 0.50:
                        should_append_letter = False
                if should_append_letter:
                    trans_result = self.translation_engine.process_token(temp_result.stable_sign)
                else:
                    trans_result = self._get_current_translation_result()
            else:
                # Check pause boundary if enabled
                if (
                    self.config.translation.auto_word_boundary
                    and self.translation_engine.current_word
                    and self.translation_engine._last_token_timestamp > 0.0
                ):
                    pause_elapsed = timestamp - self.translation_engine._last_token_timestamp
                    if pause_elapsed >= self.config.translation.word_boundary_pause_seconds:
                        trans_result = self.translation_engine.space()
                    else:
                        trans_result = self._get_current_translation_result()
                else:
                    trans_result = self._get_current_translation_result()
        except Exception as e:
            logger.error("Translation processing error on frame %d: %s", frame_id, e)
            frame_error = PipelineError(stage="translation", message=str(e))
            trans_result = self._get_current_translation_result()
        latencies.translation_ms = (time.perf_counter() - t0) * 1000.0

        # Stage 10: Event-based speech dispatch on stably confirmed signs and translation events
        try:
            allow_speech = True
            if self._active_source == FrameSource.DEMO_VIDEO:
                if is_unsupported_demo or (demo_target and demo_target not in ("HELLO", "HELP", "YES", "NO", "STOP", "THANK YOU", "GOODBYE")):
                    allow_speech = False

            if allow_speech:
                if self.router.mode == RecognitionMode.WORD_SIGNS:
                    # In WORD_SIGNS mode, speak newly emitted word signs only
                    if routed_result.is_word and routed_result.is_newly_emitted and routed_result.token:
                        self.last_stable_token = routed_result.token
                        if self.config.tts.speak_words or self.config.tts.speak_stable_signs:
                            self.speech_controller.handle_stable_sign(routed_result.token, timestamp=timestamp)
                elif self.router.mode == RecognitionMode.ALPHABET:
                    # In ALPHABET mode, speak confirmed alphabet letters if enabled
                    if temp_result.has_emitted_sign and temp_result.stable_sign is not None:
                        letter_token = temp_result.stable_sign.label
                        self.last_stable_token = letter_token
                        if self.config.tts.speak_letters or self.config.tts.speak_stable_signs:
                            self.speech_controller.handle_stable_sign(letter_token, timestamp=timestamp)
                else:  # AUTO
                    if routed_result.is_word and routed_result.is_newly_emitted and routed_result.token:
                        self.last_stable_token = routed_result.token
                        if self.config.tts.speak_words or self.config.tts.speak_stable_signs:
                            self.speech_controller.handle_stable_sign(routed_result.token, timestamp=timestamp)
                    elif (not routed_result.is_word) and temp_result.has_emitted_sign and temp_result.stable_sign is not None:
                        if not (routed_result.word_prediction and routed_result.word_prediction.confidence >= 0.50):
                            letter_token = temp_result.stable_sign.label
                            self.last_stable_token = letter_token
                            if self.config.tts.speak_letters or self.config.tts.speak_stable_signs:
                                self.speech_controller.handle_stable_sign(letter_token, timestamp=timestamp)

                # 2. Word boundary speech (if enabled)
                if trans_result.event_type == TranslationEventType.WORD_COMPLETED:
                    if self.config.tts.speak_words and trans_result.completed_words:
                        last_w = trans_result.completed_words[-1]
                        word_str = getattr(last_w, "text", str(last_w))
                        self.speech_controller.handle_word_completed(word_str)
                elif trans_result.event_type in (TranslationEventType.CLEAR_SENTENCE, TranslationEventType.RESET):
                    self.speech_controller.stop()
                    self.speech_controller.reset_spoken_state()
                    self._last_spoken_display = None

                # 3. If hands are absent, allow resetting spoken token state for new gestures
                if not det_result.hands:
                    self.speech_controller.reset_spoken_state()
        except Exception as e:
            logger.error("TTS routing error on frame %d: %s", frame_id, e)

        # Calculate Total Latency
        latencies.total_pipeline_ms = (time.perf_counter() - t_total_start) * 1000.0
        self._proc_times.append(latencies.total_pipeline_ms)

        # Update Processing FPS
        self._proc_frame_counter += 1
        now = time.time()
        elapsed_proc = now - self._last_proc_fps_time
        if elapsed_proc >= 0.5:
            self._current_proc_fps = self._proc_frame_counter / elapsed_proc
            self._proc_frame_counter = 0
            self._last_proc_fps_time = now

        # Update last error state for this frame (clears stale error on clean execution)
        self._last_error = frame_error

        # Update Runtime State Snapshot
        self._update_runtime_state(
            det_result=det_result,
            primary_hand=primary_hand,
            secondary_hand=secondary_hand,
            pred_result=pred_result,
            temp_result=temp_result,
            trans_result=trans_result,
            latencies=latencies,
            routed_result=routed_result,
        )

        # Presentation: HUD & Overlay Rendering
        t0 = time.perf_counter()
        annotated_frame: Optional[np.ndarray] = None
        if self.config.draw_hud or self.config.draw_skeleton:
            annotated_frame = self.hud.render(
                frame=frame,
                state=self._runtime_state,
                primary_hand=primary_hand,
                draw_skeleton=self.config.draw_skeleton,
                draw_hud=self.config.draw_hud,
                draw_bbox=self.config.draw_bbox,
                secondary_hand=secondary_hand,
                all_hands=det_result.hands,
            )
        latencies.hud_render_ms = (time.perf_counter() - t0) * 1000.0

        output = FrameProcessingOutput(
            frame_packet=packet,
            detection_result=det_result,
            primary_hand=primary_hand,
            feature_vector=feat_vector,
            prediction_result=pred_result,
            frame_prediction=frame_pred,
            temporal_result=temp_result,
            translation_result=trans_result,
            latencies=latencies,
            annotated_frame=annotated_frame,
            routed_result=routed_result,
        )
        self._latest_output = output
        return output

    def _select_primary_hand(self, hands: Sequence[DetectedHand]) -> DetectedHand:
        """Multi-hand selection policy (Phase 14).

        Selects the primary hand based on policy:
          - 'highest_confidence': selects hand with highest handedness score.
          - 'right' / 'left': prefers requested hand if present, otherwise highest confidence.
        """
        if not hands:
            raise ValueError("Cannot select primary hand from empty list.")

        pref = self.config.primary_hand_preference.lower()
        if pref in ("right", "left"):
            matching = [h for h in hands if h.handedness.lower() == pref]
            if matching:
                return max(matching, key=lambda h: h.handedness_confidence)

        # Default: highest confidence
        return max(hands, key=lambda h: h.handedness_confidence)

    def _update_runtime_state(
        self,
        det_result: Optional[HandDetectionResult] = None,
        primary_hand: Optional[DetectedHand] = None,
        secondary_hand: Optional[DetectedHand] = None,
        pred_result: Optional[PredictionResult] = None,
        temp_result: Optional[TemporalFrameResult] = None,
        trans_result: Optional[TranslationResult] = None,
        latencies: Optional[StageLatencies] = None,
        routed_result: Optional[RoutedRecognitionResult] = None,
    ) -> None:
        """Constructs an immutable snapshot of current operational state."""
        uptime = time.time() - self._start_time

        with self._lock:
            prev = self._runtime_state
            is_frame_update = det_result is not None

            hand_detected = (primary_hand is not None) if is_frame_update else prev.hand_detected
            handedness = (primary_hand.handedness if primary_hand else None) if is_frame_update else prev.handedness
            detection_confidence = (primary_hand.handedness_confidence if primary_hand else 0.0) if is_frame_update else prev.detection_confidence
            hand_count = (det_result.hand_count if det_result else 0) if is_frame_update else prev.hand_count
            sec_handedness = (secondary_hand.handedness if secondary_hand else None) if is_frame_update else prev.secondary_handedness
            sec_confidence = (secondary_hand.handedness_confidence if secondary_hand else 0.0) if is_frame_update else prev.secondary_confidence
            all_hands_tuple = tuple(det_result.hands) if (is_frame_update and det_result) else prev.all_hands

            if is_frame_update:
                if not hand_detected:
                    predicted_label = None
                    prediction_confidence = 0.0
                    top_margin = 0.0
                    active_word = None
                    word_conf = 0.0
                    self.last_stable_token = None
                else:
                    predicted_label = pred_result.class_name if pred_result else None
                    prediction_confidence = pred_result.confidence if pred_result else 0.0
                    top_margin = pred_result.top_margin if pred_result else 0.0
                    active_word = routed_result.active_symbol if (routed_result and routed_result.is_word) else None
                    word_conf = routed_result.confidence if (routed_result and routed_result.is_word) else 0.0
            else:
                predicted_label = prev.predicted_label
                prediction_confidence = prev.prediction_confidence
                top_margin = prev.top_margin
                active_word = prev.active_word_sign
                word_conf = prev.word_sign_confidence

            pipe_lat = latencies.total_pipeline_ms if latencies else prev.pipeline_latency
            stage_lats = latencies or prev.stage_latencies

            dwell_dur = temp_result.dwell_duration_ms if temp_result else 0.0
            is_uncertain = False
            if hand_detected:
                if routed_result and routed_result.is_word:
                    is_uncertain = (word_conf < self.config.min_classification_confidence)
                else:
                    is_uncertain = (prediction_confidence < self.config.min_classification_confidence)

            if self.speech_controller.last_spoken_text:
                self._last_spoken_display = self.speech_controller.last_spoken_text

            # Calculate active input (current_word) cleanly based on mode
            if not hand_detected:
                effective_current_word = trans_result.current_word if trans_result else self.translation_engine.current_word
            elif self.router.mode == RecognitionMode.WORD_SIGNS:
                # In WORD_SIGNS mode, active word is the word sign in-flight before being committed
                already_committed = False
                completed_list = trans_result.completed_words if trans_result else self.translation_engine.completed_words
                if completed_list and active_word:
                    last_w = completed_list[-1]
                    last_text = getattr(last_w, "text", str(last_w))
                    if last_text == active_word:
                        already_committed = True
                effective_current_word = "" if already_committed else (active_word or "")
            elif self.router.mode == RecognitionMode.AUTO and routed_result and routed_result.is_word and active_word:
                # In AUTO mode with active word sign
                already_committed = False
                completed_list = trans_result.completed_words if trans_result else self.translation_engine.completed_words
                if completed_list:
                    last_w = completed_list[-1]
                    last_text = getattr(last_w, "text", str(last_w))
                    if last_text == active_word:
                        already_committed = True
                effective_current_word = "" if already_committed else (active_word or "")
            else:
                # In ALPHABET mode or fingerspelling in AUTO mode
                effective_current_word = trans_result.current_word if trans_result else self.translation_engine.current_word

            self._runtime_state = PipelineRuntimeState(
                camera_status=self.camera.status,
                camera_fps=self.camera.current_fps,
                processing_fps=self._current_proc_fps,
                is_paused=self._is_paused,
                frame_count=self._frame_count,
                dropped_frames=self._dropped_frames,
                uptime_seconds=uptime,
                pipeline_latency=pipe_lat,
                stage_latencies=stage_lats,
                hand_detected=hand_detected,
                handedness=handedness,
                detection_confidence=detection_confidence,
                hand_count=hand_count,
                secondary_handedness=sec_handedness,
                secondary_confidence=sec_confidence,
                all_hands=all_hands_tuple,
                predicted_label=predicted_label,
                prediction_confidence=prediction_confidence,
                top_margin=top_margin,
                temporal_state=temp_result.state if temp_result else self.temporal_engine.current_state,
                temporal_candidate=temp_result.current_candidate if temp_result else self.temporal_engine.active_candidate,
                dwell_progress=min(1.0, (temp_result.dwell_duration_ms / self.config.temporal.minimum_dwell_ms)) if temp_result else 0.0,
                stable_token=temp_result.stable_sign.label if (temp_result and temp_result.stable_sign) else (self.temporal_engine.last_emitted_sign.label if self.temporal_engine.last_emitted_sign else None),
                current_word=effective_current_word,
                sentence=trans_result.sentence_text if trans_result else self.translation_engine.sentence_text,
                completed_words=trans_result.completed_words if trans_result else tuple(self.translation_engine.completed_words),
                translation_state=trans_result.state.value if trans_result else self.translation_engine.current_state.value,
                last_event=trans_result.event_type.value if trans_result else None,
                error_state=self._last_error.message if self._last_error else None,
                voice_enabled=self.speech_controller.is_enabled,
                voice_status=self.speech_controller.status.value,
                voice_error_message=self.speech_controller.error_message,
                is_speaking=self.speech_controller.is_speaking,
                last_spoken=self.speech_controller.last_spoken_text or self._last_spoken_display,
                recognition_mode=self.router.mode.value,
                active_word_sign=active_word,
                word_sign_confidence=word_conf,
                frame_source=self._active_source.value,
                dwell_duration_ms=dwell_dur,
                is_uncertain=is_uncertain,
            )

    def _reset_operational_state(self) -> None:
        """Immediately and comprehensively purges all runtime buffers and in-flight state."""
        self.temporal_engine.reset()
        self.translation_engine.reset()
        if self.word_smoother is not None:
            self.word_smoother.reset()
        self.speech_controller.stop()
        self.speech_controller.reset_spoken_state()
        self._last_spoken_display = None
        self.last_stable_token = None
        self._last_demo_frame_id = None
        self._last_error = None
        if hasattr(self.hud, "reset"):
            self.hud.reset()
        self._latest_output = None
        cam_stat = getattr(self.camera, "status", CameraStatus.INITIALIZING) if self.camera else CameraStatus.INITIALIZING
        cam_fps = getattr(self.camera, "current_fps", 0.0) if self.camera else 0.0
        self._runtime_state = PipelineRuntimeState(
            camera_status=cam_stat,
            camera_fps=cam_fps,
            processing_fps=0.0,
            is_paused=self._is_paused,
            frame_count=0,
            dropped_frames=0,
            uptime_seconds=0.0,
            pipeline_latency=0.0,
            stage_latencies=StageLatencies(),
            hand_detected=False,
            handedness=None,
            detection_confidence=0.0,
            hand_count=0,
            secondary_handedness=None,
            secondary_confidence=0.0,
            all_hands=(),
            predicted_label=None,
            prediction_confidence=0.0,
            top_margin=0.0,
            temporal_state=TemporalState.IDLE,
            temporal_candidate=None,
            dwell_progress=0.0,
            stable_token=None,
            current_word="",
            sentence="",
            completed_words=(),
            translation_state=TranslationState.EMPTY.value,
            last_event=None,
            error_state=None,
            voice_enabled=self.speech_controller.is_enabled,
            voice_status=self.speech_controller.status.value,
            voice_error_message=self.speech_controller.error_message,
            is_speaking=False,
            last_spoken=None,
            recognition_mode=self.router.mode.value,
            active_word_sign=None,
            word_sign_confidence=0.0,
            frame_source=self._active_source.value,
            dwell_duration_ms=0.0,
            is_uncertain=False,
        )

    def switch_to_demo_video(self, reader: ICameraSource, sign_label: Optional[str] = None) -> None:
        """Exclusively switches pipeline to Demo Video mode, isolating from live camera."""
        with self._lock:
            # 1. Stop / pause live webcam capture
            if self._active_source == FrameSource.LIVE_CAMERA and self.camera is not None:
                self._live_camera = self.camera
                try:
                    if hasattr(self._live_camera, "stop"):
                        self._live_camera.stop()
                    elif hasattr(self._live_camera, "release"):
                        self._live_camera.release()
                except Exception as e:
                    logger.warning("Error stopping live camera: %s", e)

            # 2. Switch to dedicated demo detector and demo recognition engines
            self.detector = self._demo_detector
            self.temporal_engine = self._demo_temporal_engine
            self.translation_engine = self._demo_translation_engine
            if self._demo_word_smoother is not None:
                self.word_smoother = self._demo_word_smoother
                self.router.word_smoother = self._demo_word_smoother

            # 3. Re-assign active camera to demo video reader and start it
            self.camera = reader
            try:
                if hasattr(self.camera, "start"):
                    self.camera.start()
                elif hasattr(self.camera, "play"):
                    self.camera.play()
            except Exception as e:
                logger.error("Error starting demo video reader: %s", e)

            self._active_source = FrameSource.DEMO_VIDEO

            # Track demo target sign label (standardized uppercase)
            if sign_label:
                clean_lbl = str(sign_label).strip().upper()
                if "GOODBYE" in clean_lbl:
                    clean_lbl = "GOODBYE"
                elif "/" in clean_lbl:
                    clean_lbl = clean_lbl.split("/")[0].strip()
                self.target_demo_sign = clean_lbl
            elif hasattr(reader, "video_path") and reader.video_path:
                stem = reader.video_path.stem.upper().replace("_SAMPLE", "").replace("_", " ")
                if "GOODBYE" in stem or "BYE" in stem:
                    self.target_demo_sign = "GOODBYE"
                else:
                    self.target_demo_sign = stem
            else:
                self.target_demo_sign = None

            # 4. For word-sign demonstration videos, configure WORD_SIGNS recognition mode
            self.router.set_mode(RecognitionMode.WORD_SIGNS)

            # 5. Cleanly reset all recognition, temporal, and speech state
            self._reset_operational_state()
            logger.info("Switched to DEMO_VIDEO mode for sign '%s' (live camera halted and state cleared).", self.target_demo_sign)

    def switch_to_live_camera(self) -> None:
        """Exclusively restores Live Camera mode, halting demo video playback."""
        with self._lock:
            # 1. Stop demo video reader
            if self.camera is not None and self.camera is not self._live_camera:
                try:
                    if hasattr(self.camera, "stop"):
                        self.camera.stop()
                    elif hasattr(self.camera, "release"):
                        self.camera.release()
                except Exception as e:
                    logger.warning("Error stopping demo video reader: %s", e)

            # 2. Switch to dedicated live detector and live recognition engines
            self.detector = self._live_detector
            self.temporal_engine = self._live_temporal_engine
            self.translation_engine = self._live_translation_engine
            if self._live_word_smoother is not None:
                self.word_smoother = self._live_word_smoother
                self.router.word_smoother = self._live_word_smoother

            # 3. Re-assign and start live camera
            if self._live_camera is not None:
                self.camera = self._live_camera
            else:
                self.camera = CameraManager(
                    config=self.config.camera,
                    auto_reconnect=self.config.auto_reconnect,
                    max_reconnect_attempts=self.config.max_reconnect_attempts,
                )
                self._live_camera = self.camera

            try:
                if hasattr(self.camera, "start"):
                    self.camera.start()
            except Exception as e:
                logger.error("Error starting live camera: %s", e)

            self._active_source = FrameSource.LIVE_CAMERA
            self.target_demo_sign = None

            # 4. Restore AUTO hybrid recognition mode for live camera
            self.router.set_mode(RecognitionMode.AUTO)

            # 5. Cleanly reset all recognition, temporal, and speech state
            self._reset_operational_state()
            logger.info("Restored LIVE_CAMERA mode (demo playback halted and state cleared).")

    def set_camera_source(self, camera: ICameraSource) -> None:
        """Dynamically switches active camera / video source."""
        from signbridge.video.reader import VideoFileReader
        if isinstance(camera, VideoFileReader):
            self.switch_to_demo_video(camera)
        else:
            with self._lock:
                if self.camera is not None and self.camera is not camera:
                    try:
                        if hasattr(self.camera, "stop"):
                            self.camera.stop()
                        elif hasattr(self.camera, "release"):
                            self.camera.release()
                    except Exception as e:
                        logger.warning("Error releasing previous camera source: %s", e)
                self.camera = camera
                self._live_camera = camera
                self._active_source = FrameSource.LIVE_CAMERA
                try:
                    if hasattr(self.camera, "start"):
                        self.camera.start()
                except Exception as e:
                    logger.warning("Error starting camera source: %s", e)
                self._update_runtime_state()
                logger.info("Pipeline video source switched to: %s", type(camera).__name__)

    def set_recognition_mode(self, mode: Union[RecognitionMode, str]) -> None:
        """Configures operational recognition mode (ALPHABET, WORD_SIGNS, AUTO)."""
        self.router.set_mode(mode)
        if self.router.mode == RecognitionMode.WORD_SIGNS:
            if self.translation_engine._current_tokens:
                self.translation_engine.clear_word()
        elif self.router.mode == RecognitionMode.ALPHABET:
            if self.word_smoother is not None:
                self.word_smoother.reset()
        self._update_runtime_state()

    def _get_current_translation_result(self) -> TranslationResult:
        """Constructs a TranslationResult from current engine state without mutating buffers."""
        return TranslationResult(
            state=self.translation_engine.current_state,
            current_word=self.translation_engine.current_word,
            sentence_text=self.translation_engine.sentence_text,
            completed_words=tuple(self.translation_engine.completed_words),
            event_type=TranslationEventType.IGNORED,
            last_token=None,
            tokens_count=self.translation_engine.tokens_count,
            processing_latency_ms=0.0,
        )

