"""Stage 11: SignBridge Professional Application Controller.

Unified orchestrator managing the OpenCV window, high-precision mouse callback,
keyboard shortcut dispatch, responsive canvas composition, and modal lifecycle.
"""

import sys
from typing import Optional, Tuple
import cv2
import numpy as np

from signbridge.core.config import CameraConfig
from signbridge.pipeline.orchestrator import SignBridgePipeline
from signbridge.pipeline.state import PipelineRuntimeState
from signbridge.ui.demo import SignBridgeDemoDriver
from signbridge.ui.layout import DashboardLayout
from signbridge.ui.overlays import ModalOverlayManager
from signbridge.ui.state import ActiveModal, AppStatus, UIState
from signbridge.ui.theme import HIGH_CONTRAST_THEME, OBSIDIAN_THEME, UITheme


class SignBridgeApp:
    """Professional desktop application runner for the SignBridge translation system."""

    def __init__(
        self,
        pipeline: SignBridgePipeline,
        window_title: str = "SignBridge — Professional Translation Suite",
        width: int = 1280,
        height: int = 800,
        high_contrast: bool = False,
        reduced_motion: bool = False,
        demo_mode: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.window_name = window_title
        self.width = width
        self.height = height

        # Presentation layer controllers
        self.ui_state = UIState(
            high_contrast=high_contrast,
            reduced_motion=reduced_motion,
            demo_mode=demo_mode,
        )
        self.layout = DashboardLayout(width, height)
        self.modals = ModalOverlayManager()
        self.demo_driver = SignBridgeDemoDriver(pipeline)

        # Internal canvas buffer
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._is_running = False

        # Last valid frame retention buffer & anti-flicker telemetry
        self._last_valid_frame: Optional[np.ndarray] = None
        self._consecutive_frame_failures: int = 0
        self.max_frame_failure_grace: int = 15  # ~500ms grace period at 30 FPS

    @property
    def last_valid_frame(self) -> Optional[np.ndarray]:
        """Returns the cached last-valid frame buffer."""
        return self._last_valid_frame

    @property
    def consecutive_frame_failures(self) -> int:
        """Returns count of consecutive missing or dropped frame reads."""
        return self._consecutive_frame_failures

    def run(self, max_frames: Optional[int] = None) -> int:
        """Executes the interactive GUI event loop."""
        # Enable Windows High-DPI Awareness to prevent OS bitmap scaling and blur
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    ctypes.windll.user32.SetProcessDPIAware()
                except Exception:
                    pass

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        self._is_running = True
        self.pipeline.start()

        if self.ui_state.demo_mode:
            self.demo_driver.start()
            self.ui_state.set_toast("Demonstration Mode Active", level="info")

        frame_count = 0

        try:
            while self._is_running:
                # Check for window close via window manager 'X' button
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                # 1. Update demonstration driver if active
                if self.ui_state.demo_mode:
                    msg = self.demo_driver.update()
                    if msg:
                        self.ui_state.set_toast(msg, level="info")

                # 2. Advance pipeline frame
                success, output = self.pipeline.process_next_frame(timeout=0.03)
                p_state = self.pipeline.get_state()

                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    break

                # Extract fresh camera frame for display if available
                fresh_frame = None
                if output is not None:
                    if self.ui_state.show_skeleton and output.annotated_frame is not None:
                        fresh_frame = output.annotated_frame
                    elif output.frame_packet is not None:
                        fresh_frame = output.frame_packet.data

                # Anti-flicker frame retention logic
                if fresh_frame is not None and fresh_frame.size > 0:
                    self._last_valid_frame = fresh_frame
                    self._consecutive_frame_failures = 0
                    cam_img = fresh_frame
                else:
                    if p_state.is_paused:
                        # While paused, hold frozen frame without accumulating failures
                        cam_img = self._last_valid_frame
                    else:
                        self._consecutive_frame_failures += 1
                        # Transient delay (< grace period): retain last valid frame to eliminate flicker
                        if (
                            self._consecutive_frame_failures <= self.max_frame_failure_grace
                            and self._last_valid_frame is not None
                        ):
                            cam_img = self._last_valid_frame
                        else:
                            # Genuine sustained failure / hardware disconnect
                            cam_img = None

                # 2.5 Synchronize video playback telemetry if streaming from video file
                from signbridge.video.reader import VideoFileReader
                from signbridge.video.base import VideoPlaybackState
                if isinstance(self.pipeline.camera, VideoFileReader):
                    reader = self.pipeline.camera
                    self.ui_state.video_current_frame = reader.current_frame_index
                    self.ui_state.video_total_frames = reader.total_frames
                    fps = reader.current_fps or 30.0
                    self.ui_state.video_current_time = reader.current_frame_index / fps if fps > 0 else 0.0
                    self.ui_state.video_duration = reader.total_frames / fps if fps > 0 else 0.0
                    self.ui_state.video_is_playing = (reader.playback_state == VideoPlaybackState.PLAYING)

                # 2.6 Check for emergency modal trigger from word-sign recognition
                if (
                    hasattr(self.pipeline, "emergency_manager")
                    and self.pipeline.emergency_manager.is_modal_active
                    and self.ui_state.active_modal != ActiveModal.EMERGENCY
                ):
                    self.ui_state.open_modal(ActiveModal.EMERGENCY)

                # 3. Select active theme
                theme = HIGH_CONTRAST_THEME if self.ui_state.high_contrast else OBSIDIAN_THEME

                # 4. Render main dashboard layout
                self.layout.render(
                    self._canvas,
                    pipeline_state=p_state,
                    ui_state=self.ui_state,
                    theme=theme,
                    camera_frame=cam_img,
                )

                # 5. Render active modal dialog overlay if open
                if self.ui_state.active_modal != ActiveModal.NONE:
                    settings_dict = self._get_settings_dict()
                    self.modals.render(self._canvas, self.ui_state, theme, settings_dict)

                # 6. Display composite canvas
                cv2.imshow(self.window_name, self._canvas)

                # 7. Process keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    if not self._handle_key(key):
                        break

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

        return 0

    def shutdown(self) -> None:
        """Safely terminates GUI, pipeline, and background threads."""
        self._is_running = False
        if self.demo_driver.is_active:
            self.demo_driver.stop()
        self.pipeline.stop()
        cv2.destroyAllWindows()

    # -------------------------------------------------------------------------
    # Mouse Interaction Callback
    # -------------------------------------------------------------------------

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: None) -> None:
        """Tracks mouse position and dispatches click events."""
        self.ui_state.mouse_x = x
        self.ui_state.mouse_y = y

        # Query buttons from modal if modal active, else from dashboard layout
        if self.ui_state.active_modal != ActiveModal.NONE:
            registered = self.modals.get_registered_buttons()
        else:
            registered = self.layout.get_registered_buttons()

        # Update hovered button
        hovered = None
        for bid, (bx, by, bw, bh) in registered.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                hovered = bid
                break
        self.ui_state.hovered_id = hovered

        if event == cv2.EVENT_LBUTTONDOWN:
            self.ui_state.mouse_down = True
            self.ui_state.clicked_id = hovered
        elif event == cv2.EVENT_LBUTTONUP:
            clicked = self.ui_state.clicked_id
            self.ui_state.mouse_down = False
            self.ui_state.clicked_id = None
            if clicked and clicked == hovered:
                self._dispatch_button_action(clicked)

    def _dispatch_button_action(self, btn_id: str) -> None:
        """Executes corresponding action for clicked button."""
        # Top Header Buttons
        if btn_id == "btn_settings":
            self.ui_state.open_modal(ActiveModal.SETTINGS)
        elif btn_id == "btn_help":
            self.ui_state.open_modal(ActiveModal.HELP)
        elif btn_id == "btn_video_demo":
            self.ui_state.open_modal(ActiveModal.VIDEO_DEMO)

        # Modal Controls
        elif btn_id in ("modal_close", "modal_close_btn", "set_btn_done", "help_btn_done", "onboarding_skip"):
            self.ui_state.close_modal()
        elif btn_id == "onboarding_done":
            self.ui_state.close_modal()
            self.ui_state.set_toast("Ready to translate! Position hand in camera view.", level="success")

        # Emergency Modal Actions
        elif btn_id == "btn_emergency_confirm":
            self.pipeline.emergency_manager.confirm_assistance()
            self.ui_state.close_modal()
            self.ui_state.set_toast("Simulated Emergency: Assistance Confirmed (Local Demo Only)", level="warning", duration=4.0)
        elif btn_id == "btn_emergency_cancel":
            self.pipeline.emergency_manager.cancel()
            self.ui_state.close_modal()
            self.ui_state.set_toast("Emergency alert cancelled", level="info")

        # Video Demo Modal & Sample Loading & Input Sources
        elif btn_id.startswith("btn_sample_"):
            sign_key = btn_id.replace("btn_sample_", "")
            self._load_sample_video(sign_key)
        elif btn_id.startswith("btn_play_sign_"):
            sign_id = btn_id.replace("btn_play_sign_", "")
            self._load_catalog_sign(sign_id)
        elif btn_id.startswith("btn_next_card_"):
            sign_id = btn_id.replace("btn_next_card_", "")
            from signbridge.video.catalog import SignLanguageCatalog
            cat = SignLanguageCatalog()
            active_cat = self.ui_state.demo_category
            next_sign = cat.get_next_sign(sign_id, category=active_cat, available_only=False)
            if next_sign:
                matching = cat.list_signs(category=active_cat, query=self.ui_state.demo_search_query)
                next_idx = next((i for i, s in enumerate(matching) if s.id == next_sign.id), -1)
                if next_idx != -1:
                    self.ui_state.demo_page = (next_idx // 5) + 1
                msg = f"Next sign: '{next_sign.name}'"
                if not next_sign.available:
                    msg += " (Demo video unavailable)"
                self.ui_state.set_toast(msg, level="info")
        elif btn_id == "btn_demo_next_sign":
            from signbridge.video.catalog import SignLanguageCatalog
            cat = SignLanguageCatalog()
            cur_id = self.ui_state.demo_selected_sign_id or ""
            active_cat = self.ui_state.demo_category
            next_s = cat.get_next_sign(cur_id, category=active_cat, available_only=False)
            if next_s:
                self.ui_state.demo_selected_sign_id = next_s.id
                self.ui_state.demo_view_mode = "DETAIL"
                matching = cat.list_signs(category=active_cat, query=self.ui_state.demo_search_query)
                next_idx = next((i for i, s in enumerate(matching) if s.id == next_s.id), -1)
                if next_idx != -1:
                    self.ui_state.demo_page = (next_idx // 5) + 1
                msg = f"Sign: '{next_s.name}' ({next_s.category})"
                if not next_s.available:
                    msg += " — Demo video unavailable"
                self.ui_state.set_toast(msg, level="info")
        elif btn_id == "btn_demo_prev_sign":
            from signbridge.video.catalog import SignLanguageCatalog
            cat = SignLanguageCatalog()
            cur_id = self.ui_state.demo_selected_sign_id or ""
            active_cat = self.ui_state.demo_category
            prev_s = cat.get_prev_sign(cur_id, category=active_cat, available_only=False)
            if prev_s:
                self.ui_state.demo_selected_sign_id = prev_s.id
                self.ui_state.demo_view_mode = "DETAIL"
                matching = cat.list_signs(category=active_cat, query=self.ui_state.demo_search_query)
                prev_idx = next((i for i, s in enumerate(matching) if s.id == prev_s.id), -1)
                if prev_idx != -1:
                    self.ui_state.demo_page = (prev_idx // 5) + 1
                msg = f"Sign: '{prev_s.name}' ({prev_s.category})"
                if not prev_s.available:
                    msg += " — Demo video unavailable"
                self.ui_state.set_toast(msg, level="info")
        elif btn_id.startswith("btn_detail_sign_"):
            sign_id = btn_id.replace("btn_detail_sign_", "")
            self.ui_state.demo_selected_sign_id = sign_id
            self.ui_state.demo_view_mode = "DETAIL"
        elif btn_id == "btn_demo_back_list":
            self.ui_state.demo_view_mode = "LIST"
        elif btn_id.startswith("btn_cat_"):
            cat_map = {
                "btn_cat_all": "All Signs",
                "btn_cat_alpha": "Alphabet",
                "btn_cat_nums": "Numbers",
                "btn_cat_greetings": "Greetings",
                "btn_cat_common": "Common Words",
                "btn_cat_daily": "Daily Communication",
                "btn_cat_phrases": "Common Phrases",
            }
            target_cat = cat_map.get(btn_id, "All Signs")
            self.ui_state.demo_category = target_cat
            self.ui_state.demo_page = 1
            self.ui_state.demo_view_mode = "LIST"
            self.ui_state.set_toast(f"Category: {target_cat.upper()}", level="info")
        elif btn_id == "btn_page_prev":
            self.ui_state.demo_page = max(1, self.ui_state.demo_page - 1)
        elif btn_id == "btn_page_next":
            self.ui_state.demo_page += 1
        elif btn_id == "btn_search_clear":
            self.ui_state.demo_search_query = ""
            self.ui_state.demo_category = "All Signs"
            self.ui_state.demo_page = 1
            self.ui_state.demo_view_mode = "LIST"
            self.ui_state.set_toast("Reset filters to all signs", level="info")
        elif btn_id in ("btn_source_camera", "btn_src_camera"):
            self._switch_to_camera()
        elif btn_id == "btn_src_demo":
            self.ui_state.open_modal(ActiveModal.VIDEO_DEMO)
        elif btn_id == "btn_src_upload":
            self._handle_upload_video()

        # Video Transport Controls
        elif btn_id == "btn_vid_prev_sample":
            self._navigate_sample_video(-1)
        elif btn_id == "btn_vid_next_sample":
            self._navigate_sample_video(1)
        elif btn_id == "btn_vid_play":
            from signbridge.video.reader import VideoFileReader
            if isinstance(self.pipeline.camera, VideoFileReader):
                self.pipeline.camera.play()
                if self.pipeline.is_paused:
                    self.pipeline.resume()
                self.ui_state.set_toast("Playing video")
        elif btn_id == "btn_vid_pause":
            from signbridge.video.reader import VideoFileReader
            if isinstance(self.pipeline.camera, VideoFileReader):
                self.pipeline.camera.pause()
                self.ui_state.set_toast("Video paused")
        elif btn_id == "btn_vid_restart":
            from signbridge.video.reader import VideoFileReader
            if isinstance(self.pipeline.camera, VideoFileReader):
                self.pipeline.camera.restart()
                self.ui_state.set_toast("Restarted video")
        elif btn_id == "btn_vid_stop":
            from signbridge.video.reader import VideoFileReader
            if isinstance(self.pipeline.camera, VideoFileReader):
                self.pipeline.camera.stop()
                self.ui_state.set_toast("Stopped video")
        elif btn_id == "btn_vid_run_rec":
            self._run_video_recognition()

        # Mode Selection
        elif btn_id == "btn_mode_alpha":
            self.pipeline.set_recognition_mode("ALPHABET")
            self.ui_state.set_toast("Mode: ALPHABET FINGERSPELLING")
        elif btn_id == "btn_mode_word":
            self.pipeline.set_recognition_mode("WORD_SIGNS")
            self.ui_state.set_toast("Mode: WORD-LEVEL SIGNS")
        elif btn_id == "btn_mode_auto":
            self.pipeline.set_recognition_mode("AUTO")
            self.ui_state.set_toast("Mode: AUTO (Hybrid Recognition)")

        # Settings Modal Options
        elif btn_id == "set_toggle_voice":
            v_en = self.pipeline.toggle_voice()
            self.ui_state.set_toast(f"Voice Output: {'ENABLED' if v_en else 'MUTED'}")
        elif btn_id == "set_toggle_contrast":
            self.ui_state.toggle_high_contrast()
        elif btn_id == "set_toggle_motion":
            self.ui_state.toggle_reduced_motion()
        elif btn_id == "set_toggle_skeleton":
            self.ui_state.show_skeleton = not self.ui_state.show_skeleton
            self.ui_state.set_toast(f"Skeleton Overlay: {'ON' if self.ui_state.show_skeleton else 'OFF'}")
        elif btn_id == "set_rate_minus":
            new_rate = max(100, self.pipeline.speech_controller.config.rate - 10)
            self.pipeline.speech_controller.provider.set_rate(new_rate)
            self.pipeline.speech_controller.config = self.pipeline.speech_controller.config.__class__(
                **{**self.pipeline.speech_controller.config.__dict__, "rate": new_rate}
            )
            self.ui_state.set_toast(f"Speech Rate: {new_rate} WPM")
        elif btn_id == "set_rate_plus":
            new_rate = min(250, self.pipeline.speech_controller.config.rate + 10)
            self.pipeline.speech_controller.provider.set_rate(new_rate)
            self.pipeline.speech_controller.config = self.pipeline.speech_controller.config.__class__(
                **{**self.pipeline.speech_controller.config.__dict__, "rate": new_rate}
            )
            self.ui_state.set_toast(f"Speech Rate: {new_rate} WPM")
        elif btn_id == "set_vol_minus":
            new_vol = max(0.0, round(self.pipeline.speech_controller.config.volume - 0.1, 1))
            self.pipeline.speech_controller.provider.set_volume(new_vol)
            self.pipeline.speech_controller.config = self.pipeline.speech_controller.config.__class__(
                **{**self.pipeline.speech_controller.config.__dict__, "volume": new_vol}
            )
            self.ui_state.set_toast(f"Speech Volume: {int(new_vol * 100)}%")
        elif btn_id == "set_vol_plus":
            new_vol = min(1.0, round(self.pipeline.speech_controller.config.volume + 0.1, 1))
            self.pipeline.speech_controller.provider.set_volume(new_vol)
            self.pipeline.speech_controller.config = self.pipeline.speech_controller.config.__class__(
                **{**self.pipeline.speech_controller.config.__dict__, "volume": new_vol}
            )
            self.ui_state.set_toast(f"Speech Volume: {int(new_vol * 100)}%")
        elif btn_id == "set_btn_reset":
            self.ui_state.high_contrast = False
            self.ui_state.reduced_motion = False
            self.ui_state.show_skeleton = True
            self.ui_state.set_toast("Settings reset to defaults.")

        # Help Modal Tabs
        elif btn_id.startswith("help_tab_"):
            tab_idx = int(btn_id.split("_")[-1])
            self.ui_state.help_tab = tab_idx

        # Voice Panel Buttons
        elif btn_id == "btn_speak":
            ok = self.pipeline.speak_current_sentence()
            if ok:
                self.ui_state.set_toast("Speaking sign / sentence...", level="success")
            else:
                self.ui_state.set_toast("No sign or text available to speak", level="warning")
        elif btn_id == "btn_stop":
            self.pipeline.stop_speech()
            self.ui_state.set_toast("Speech playback halted", level="info")
        elif btn_id == "btn_voice_toggle":
            v_en = self.pipeline.toggle_voice()
            self.ui_state.set_toast(f"Voice Output: {'ENABLED' if v_en else 'MUTED'}")

        # Quick Actions Grid
        elif btn_id == "btn_space":
            res = self.pipeline.space()
            self.ui_state.set_toast(f"Word break: '{res.sentence_text}'", level="info")
        elif btn_id == "btn_backspace":
            res = self.pipeline.backspace()
            self.ui_state.set_toast(f"Deleted letter: '{res.current_word}'", level="info")
        elif btn_id == "btn_clear":
            self.pipeline.clear_sentence()
            self.ui_state.set_toast("Sentence cleared", level="info")
        elif btn_id == "btn_undo":
            res = self.pipeline.undo()
            self.ui_state.set_toast("Undo executed", level="info")
        elif btn_id == "btn_pause":
            if self.pipeline.is_paused:
                self.pipeline.resume()
                self.ui_state.set_toast("Resumed video processing")
            else:
                self.pipeline.pause()
                self.ui_state.set_toast("Paused video processing")
        elif btn_id == "btn_reset":
            self.pipeline.reset()
            self.ui_state.set_toast("Reset all translation state", level="warning")

    def _load_sample_video(self, sign_key: str) -> None:
        """Loads and streams a curated demonstration video into the pipeline."""
        from signbridge.video.sample_manager import SampleVideoManager
        from signbridge.video.reader import VideoFileReader

        mgr = SampleVideoManager()
        sample = mgr.get_sample_by_sign(sign_key)
        if sample and sample.full_path:
            try:
                reader = VideoFileReader(sample.full_path, loop=True, realtime_pacing=True)
                self.pipeline.switch_to_demo_video(reader, sign_label=sample.sign)
                self._last_valid_frame = None
                self.ui_state.active_input_source = "DEMO"
                self.ui_state.selected_sample_video = sample.sign
                self.ui_state.video_filename = sample.filename
                self.ui_state.close_modal()
                self.ui_state.set_toast(f"Playing Demo Video: {sample.sign}", level="success")
            except Exception as e:
                self.ui_state.set_toast(f"Error loading sample: {e}", level="error")
        else:
            self.ui_state.set_toast(f"Sample '{sign_key}' not found", level="warning")

    def _load_catalog_sign(self, sign_id: str) -> None:
        """Loads and streams a catalog sign or provides reference guidance if video is pending."""
        from signbridge.video.catalog import SignLanguageCatalog
        from signbridge.video.sample_manager import SampleVideoManager
        from signbridge.video.reader import VideoFileReader

        cat = SignLanguageCatalog()
        entry = cat.get_sign_by_id(sign_id)
        if not entry:
            self.ui_state.set_toast(f"Catalog sign '{sign_id}' not found", level="warning")
            return

        if not entry.available or not entry.video:
            self.ui_state.demo_selected_sign_id = entry.id
            self.ui_state.demo_view_mode = "DETAIL"
            self.ui_state.set_toast(
                f"Demo video unavailable for '{entry.name}'. Showing reference mechanics guide.",
                level="warning",
                duration=3.5,
            )
            return

        mgr = SampleVideoManager()
        sample = mgr.get_sample_by_id(entry.id) or mgr.get_sample_by_sign(entry.name)
        if sample and sample.full_path:
            full_path_str = sample.full_path
            sign_name = sample.sign
            filename = sample.filename
        else:
            vpath = entry.get_full_video_path(cat.project_root)
            if vpath and vpath.exists():
                full_path_str = str(vpath)
                sign_name = entry.name.upper()
                filename = vpath.name
            else:
                full_path_str = None
                sign_name = entry.name.upper()
                filename = ""

        if full_path_str:
            try:
                reader = VideoFileReader(full_path_str, loop=True, realtime_pacing=True)
                self.pipeline.switch_to_demo_video(reader, sign_label=sign_name)
                self._last_valid_frame = None
                self.ui_state.active_input_source = "DEMO"
                self.ui_state.selected_sample_video = sign_name
                self.ui_state.video_filename = filename
                self.ui_state.demo_category = entry.category
                self.ui_state.close_modal()
                self.ui_state.set_toast(f"Playing Demo Video: {sign_name} ({entry.category})", level="success")
            except Exception as e:
                self.ui_state.set_toast(f"Error loading demo video: {e}", level="error")
        else:
            self.ui_state.set_toast(f"Video file for '{entry.name}' unavailable", level="warning")

    def _navigate_sample_video(self, direction: int) -> None:
        """Navigates to the next (+1) or previous (-1) playable demonstration video strictly within the active category."""
        from signbridge.video.catalog import SignLanguageCatalog

        cat = SignLanguageCatalog()
        cur_sign_name = (self.ui_state.selected_sample_video or "HELLO").upper()
        entry = cat.get_sign_by_name(cur_sign_name)
        cur_id = entry.id if entry else "asl_word_hello"
        active_cat = entry.category if entry else (
            self.ui_state.demo_category if self.ui_state.demo_category not in ("All Signs", "ALL SIGNS") else None
        )
        if direction > 0:
            next_entry = cat.get_next_sign(cur_id, category=active_cat, available_only=True)
        else:
            next_entry = cat.get_prev_sign(cur_id, category=active_cat, available_only=True)
        if next_entry:
            if next_entry.id == cur_id:
                self.ui_state.set_toast(
                    f"'{entry.name if entry else cur_sign_name}' is the only playable video in {active_cat or 'category'}",
                    level="info",
                )
            else:
                self._load_catalog_sign(next_entry.id)
        else:
            self.ui_state.set_toast(f"No playable demonstration videos in {active_cat or 'category'}", level="warning")

    def _switch_to_camera(self) -> None:
        """Restores live webcam as the pipeline input source."""
        self.pipeline.switch_to_live_camera()
        self._last_valid_frame = None
        self.ui_state.active_input_source = "CAMERA"
        self.ui_state.selected_sample_video = None
        self.ui_state.video_filename = None
        self.ui_state.close_modal()
        self.ui_state.set_toast("Switched to Live Camera", level="info")

    def _run_video_recognition(self) -> None:
        """Ensures video reader and pipeline are active and processing recognition."""
        from signbridge.video.reader import VideoFileReader

        if self.pipeline.is_paused:
            self.pipeline.resume()
        if isinstance(self.pipeline.camera, VideoFileReader):
            if not self.pipeline.camera.is_running:
                self.pipeline.camera.play()
        self.ui_state.video_recognition_active = True
        self.ui_state.set_toast("Running recognition on video frames", level="info")

    def _handle_upload_video(self) -> None:
        """Imports and plays a user video file."""
        from pathlib import Path
        from signbridge.video.sample_manager import SampleVideoManager
        from signbridge.video.reader import VideoFileReader

        file_path = None
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            file_path = filedialog.askopenfilename(
                title="Select ASL Sign Video File",
                filetypes=[
                    ("Video Files", "*.mp4;*.avi;*.mov;*.mkv;*.webm"),
                    ("MP4 Files", "*.mp4"),
                    ("All Files", "*.*"),
                ],
            )
            root.destroy()
        except Exception:
            pass

        mgr = SampleVideoManager()
        if not file_path:
            user_samples = mgr.list_user_samples()
            if user_samples and user_samples[0].full_path:
                reader = VideoFileReader(user_samples[0].full_path, loop=True, realtime_pacing=True)
                self.pipeline.switch_to_demo_video(reader, sign_label=user_samples[0].sign)
                self._last_valid_frame = None
                self.ui_state.active_input_source = "UPLOAD"
                self.ui_state.selected_sample_video = f"User: {user_samples[0].sign}"
                self.ui_state.video_filename = user_samples[0].file
                self.ui_state.set_toast(f"Playing User Video: {user_samples[0].file}", level="success")
                return
            self.ui_state.set_toast("No video selected or dialog cancelled", level="warning")
            return

        val = mgr.add_user_sample(file_path, sign_label="CUSTOM")
        if val.is_valid:
            user_samples = mgr.list_user_samples()
            matching = [
                s for s in user_samples
                if s.source_url == str(Path(file_path).resolve()) or Path(file_path).name in s.file
            ]
            target = matching[-1] if matching else (user_samples[-1] if user_samples else None)
            if target and target.full_path:
                reader = VideoFileReader(target.full_path, loop=True, realtime_pacing=True)
                self.pipeline.switch_to_demo_video(reader, sign_label=target.sign)
                self._last_valid_frame = None
                self.ui_state.active_input_source = "UPLOAD"
                self.ui_state.selected_sample_video = f"User: {target.sign}"
                self.ui_state.video_filename = target.file
                self.ui_state.set_toast(f"Imported & playing: {Path(file_path).name}", level="success")
            else:
                self.ui_state.set_toast(f"Imported {Path(file_path).name}", level="success")
        else:
            self.ui_state.set_toast(f"Upload rejected: {val.error_message}", level="error")

    # -------------------------------------------------------------------------
    # Keyboard Event Handler
    # -------------------------------------------------------------------------

    def _handle_key(self, key: int) -> bool:
        """Processes keyboard input. Returns False if user requested application exit."""
        # Emergency Modal Active Handling
        if self.ui_state.active_modal == ActiveModal.EMERGENCY:
            if key in (ord("y"), ord("Y"), 13, 10):
                self.pipeline.emergency_manager.confirm_assistance()
                self.ui_state.close_modal()
                self.ui_state.set_toast("Simulated Emergency: Assistance Confirmed (Local Demo Only)", level="warning", duration=4.0)
                return True
            elif key in (ord("n"), ord("N"), 27):
                self.pipeline.emergency_manager.cancel()
                self.ui_state.close_modal()
                self.ui_state.set_toast("Emergency alert cancelled", level="info")
                return True
            return True

        # 1. Video Demo Hub Modal Active Handling
        if self.ui_state.active_modal == ActiveModal.VIDEO_DEMO:
            if key == 27:  # ESC
                if self.ui_state.demo_view_mode == "DETAIL":
                    self.ui_state.demo_view_mode = "LIST"
                    return True
                self.ui_state.close_modal()
                return True
            elif key in (ord(","), ord("<")):
                self.ui_state.demo_page = max(1, self.ui_state.demo_page - 1)
                return True
            elif key in (ord("."), ord(">")):
                self.ui_state.demo_page += 1
                return True
            elif key in (13, 10):  # ENTER
                if self.ui_state.demo_view_mode == "DETAIL" and self.ui_state.demo_selected_sign_id:
                    self._load_catalog_sign(self.ui_state.demo_selected_sign_id)
                else:
                    self.ui_state.close_modal()
                return True

        # 2. General Active Modal Handling (ESC or ENTER closes active modal)
        if self.ui_state.active_modal != ActiveModal.NONE:
            if key in (27, 13, 10):
                self.ui_state.close_modal()
            return True

        # 3. Video Player Context Key Handling (ENTER triggers recognition)
        is_video_mode = self.ui_state.active_input_source in ("DEMO", "UPLOAD")
        if is_video_mode and key in (13, 10):
            self._run_video_recognition()
            return True

        # 4. Authoritative Workspace & Global Shortcut Dispatch via Registry
        from signbridge.ui.shortcuts import SHORTCUT_REGISTRY
        action = SHORTCUT_REGISTRY.match_key(key, context="workspace")
        if not action:
            return True

        act = action.action_id
        if act in ("quit_app", "close_or_exit"):
            return False
        elif act == "commit_word":
            res = self.pipeline.space()
            self.ui_state.set_toast(f"Word break: '{res.sentence_text}'")
        elif act == "delete_letter":
            res = self.pipeline.backspace()
            self.ui_state.set_toast(f"Deleted: '{res.current_word}'")
        elif act == "clear_sentence":
            self.pipeline.clear_sentence()
            self.ui_state.set_toast("Cleared sentence")
        elif act == "undo_action":
            self.pipeline.undo()
            self.ui_state.set_toast("Undo executed")
        elif act == "reset_all":
            self.pipeline.reset()
            self.ui_state.set_toast("Reset all buffers", level="warning")
        elif act == "speak_sentence":
            ok = self.pipeline.speak_current_sentence()
            if ok:
                self.ui_state.set_toast("Speaking sign / sentence...", level="success")
            else:
                self.ui_state.set_toast("No sign or sentence to speak", level="warning")
        elif act == "stop_speech":
            self.pipeline.stop_speech()
            self.ui_state.set_toast("Speech stopped")
        elif act == "toggle_voice":
            v_en = self.pipeline.toggle_voice()
            self.ui_state.set_toast(f"Voice: {'ENABLED' if v_en else 'MUTED'}")
        elif act == "mode_alpha":
            self.pipeline.set_recognition_mode("ALPHABET")
            self.ui_state.set_toast("Mode: ALPHABET FINGERSPELLING")
        elif act == "mode_words":
            self.pipeline.set_recognition_mode("WORD_SIGNS")
            self.ui_state.set_toast("Mode: WORD-LEVEL SIGNS")
        elif act == "mode_auto":
            self.pipeline.set_recognition_mode("AUTO")
            self.ui_state.set_toast("Mode: AUTO (Hybrid Recognition)")
        elif act == "open_demo_hub":
            self.ui_state.open_modal(ActiveModal.VIDEO_DEMO)
        elif act == "open_settings":
            self.ui_state.open_modal(ActiveModal.SETTINGS)
        elif act == "open_help":
            self.ui_state.open_modal(ActiveModal.HELP)
        elif act == "toggle_demo_driver":
            self.ui_state.demo_mode = not self.ui_state.demo_mode
            if self.ui_state.demo_mode:
                self.demo_driver.start()
                self.ui_state.set_toast("Demo Mode: STARTED (Auto spelling HELLO WORLD)", level="success")
            else:
                self.demo_driver.stop()
                self.ui_state.set_toast("Demo Mode: STOPPED (Returned to Live Camera)", level="info")
        elif act == "toggle_diagnostics":
            self.ui_state.show_diagnostics = not self.ui_state.show_diagnostics
            self.ui_state.set_toast(f"Diagnostics HUD: {'ON' if self.ui_state.show_diagnostics else 'OFF'}")
        elif act == "toggle_pause":
            if self.pipeline.is_paused:
                self.pipeline.resume()
                self.ui_state.set_toast("Resumed")
            else:
                self.pipeline.pause()
                self.ui_state.set_toast("Paused")
        elif act == "toggle_mirror":
            cfg = self.pipeline.camera.config
            self.pipeline.camera.config = CameraConfig(
                device_index=cfg.device_index,
                width=cfg.width,
                height=cfg.height,
                fps=cfg.fps,
                flip_horizontal=not cfg.flip_horizontal,
            )
            self.ui_state.set_toast(f"Mirror mode: {self.pipeline.camera.config.flip_horizontal}")

        return True

        return True

    def _get_settings_dict(self) -> dict:
        """Extracts runtime settings snapshot for the settings modal."""
        ctrl = self.pipeline.speech_controller
        return {
            "voice_enabled": ctrl.is_enabled,
            "speech_rate": ctrl.config.rate,
            "speech_volume": ctrl.config.volume,
            "speak_words": ctrl.config.speak_words,
        }
