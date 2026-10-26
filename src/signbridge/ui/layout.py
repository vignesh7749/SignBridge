"""Stage 11: Professional Dashboard Layout & Panel Renderers.

Calculates responsive panel geometry and renders the complete unified workspace:
Header, Camera Viewport, Recognition Panel, Translation Panel, Voice Panel,
Action Toolbar, and Footer Status Strip.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from signbridge.pipeline.state import PipelineRuntimeState
from signbridge.temporal.contracts import TemporalState
from signbridge.ui.components import (
    draw_badge,
    draw_button,
    draw_card,
    draw_card_header,
    draw_metric_pill,
    draw_progress_bar,
    draw_status_indicator,
    hit_test,
)
from signbridge.ui.state import AppStatus, UIState
from signbridge.ui.theme import UITheme


class DashboardLayout:
    """Manages responsive workspace geometry and renders all dashboard panels."""

    def __init__(self, target_width: int = 1280, target_height: int = 800) -> None:
        self.width = target_width
        self.height = target_height
        self._buttons: Dict[str, Tuple[int, int, int, int]] = {}

    def get_registered_buttons(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Returns map of button_id -> (x, y, w, h) for mouse hit-testing."""
        return dict(self._buttons)

    def render(
        self,
        canvas: np.ndarray,
        pipeline_state: PipelineRuntimeState,
        ui_state: UIState,
        theme: UITheme,
        camera_frame: Optional[np.ndarray] = None,
    ) -> None:
        """Draws the complete dashboard into canvas."""
        h, w = canvas.shape[:2]
        self.width = w
        self.height = h
        self._buttons.clear()

        # Canvas background fill
        canvas[:] = theme.bg_canvas

        # 1. Render Header
        self._render_header(canvas, pipeline_state, ui_state, theme)

        # 2. Render Left Column: Camera Panel
        left_w = int(w * 0.61)
        right_x = left_w + 24
        right_w = w - right_x - 16
        body_y = 56
        body_h = h - body_y - 38

        self._render_camera_panel(
            canvas,
            16,
            body_y,
            left_w,
            body_h,
            camera_frame,
            pipeline_state,
            ui_state,
            theme,
        )

        # 3. Render Right Column: Recognition, Translation, Voice & Controls
        self._render_sidebar(
            canvas,
            right_x,
            body_y,
            right_w,
            body_h,
            pipeline_state,
            ui_state,
            theme,
        )

        # 4. Render Footer Status Strip
        self._render_footer(canvas, pipeline_state, ui_state, theme)

    def _render_header(
        self,
        canvas: np.ndarray,
        p_state: PipelineRuntimeState,
        u_state: UIState,
        theme: UITheme,
    ) -> None:
        """Draws top header bar with branding, status pill, and quick toggles."""
        w = self.width
        cv2.rectangle(canvas, (0, 0), (w, 48), theme.header_bg, -1)
        cv2.line(canvas, (0, 48), (w, 48), theme.card_border, 1, cv2.LINE_AA)

        # Brand Icon + Title
        cv2.circle(canvas, (24, 24), 8, theme.primary_accent, -1, cv2.LINE_AA)
        cv2.circle(canvas, (24, 24), 4, theme.header_bg, -1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            "SIGNBRIDGE",
            (38, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            theme.font_scale_title,
            theme.text_primary,
            2,
            cv2.LINE_AA,
        )

        mode_badge = "DEMO" if u_state.demo_mode else "PRO"
        draw_badge(
            canvas,
            195,
            15,
            mode_badge,
            fg_color=theme.text_dark,
            bg_color=theme.primary_accent,
            font_scale=0.35,
        )

        # Global Status Indicator
        app_status = self._derive_app_status(p_state, u_state)
        draw_status_indicator(
            canvas,
            255,
            14,
            app_status,
            theme,
            is_speaking=p_state.is_speaking,
        )

        # Telemetry & Voice Badges on Right side
        rx = w - 16

        # Help Button [?]
        btn_help_w = 64
        btn_help_x = rx - btn_help_w
        self._register_button("btn_help", btn_help_x, 10, btn_help_w, 28)
        draw_button(
            canvas,
            btn_help_x,
            10,
            btn_help_w,
            28,
            "Help",
            "H",
            u_state.hovered_id == "btn_help",
            u_state.clicked_id == "btn_help",
            theme,
        )
        rx -= btn_help_w + 8

        # Settings Button [⚙]
        btn_set_w = 84
        btn_set_x = rx - btn_set_w
        self._register_button("btn_settings", btn_set_x, 10, btn_set_w, 28)
        draw_button(
            canvas,
            btn_set_x,
            10,
            btn_set_w,
            28,
            "Settings",
            "O",
            u_state.hovered_id == "btn_settings",
            u_state.clicked_id == "btn_settings",
            theme,
        )
        rx -= btn_set_w + 8

        # Demo Videos Hub Button [📹]
        btn_demo_w = 90
        btn_demo_x = rx - btn_demo_w
        self._register_button("btn_video_demo", btn_demo_x, 10, btn_demo_w, 28)
        draw_button(
            canvas,
            btn_demo_x,
            10,
            btn_demo_w,
            28,
            "Demo Hub",
            "V",
            u_state.hovered_id == "btn_video_demo",
            u_state.clicked_id == "btn_video_demo",
            theme,
            accent_border=True,
        )
        rx -= btn_demo_w + 12

        # Voice Status Pill
        if not p_state.voice_enabled:
            v_col, v_txt = theme.muted_accent, "VOICE: DISABLED"
        elif p_state.is_speaking:
            v_col, v_txt = theme.primary_accent, "VOICE: SPEAKING"
        elif p_state.voice_status == "ERROR":
            v_col, v_txt = theme.error_accent, "VOICE: ERROR"
        else:
            v_col, v_txt = theme.secondary_accent, "VOICE: READY"

        v_w = draw_badge(
            canvas,
            rx - 120,
            12,
            v_txt,
            fg_color=v_col,
            bg_color=theme.card_bg,
            border_color=v_col,
            font_scale=theme.font_scale_small,
        )
        rx -= v_w + 10

        # Camera FPS Pill
        draw_metric_pill(
            canvas,
            rx - 110,
            12,
            "CAM",
            f"{p_state.camera_fps:.0f} FPS",
            theme,
        )

    def _render_camera_panel(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        camera_frame: Optional[np.ndarray],
        p_state: PipelineRuntimeState,
        u_state: UIState,
        theme: UITheme,
    ) -> None:
        """Renders live video stream within a clean framed container."""
        # Card container
        draw_card(canvas, x, y, w, h, theme.card_bg, theme.card_border, theme.border_width)
        draw_card_header(canvas, x, y, w, "Video Viewport", theme)

        # Input Source Buttons (Header Right)
        is_cam = u_state.active_input_source == "CAMERA"
        is_demo = u_state.active_input_source in ("DEMO", "SAMPLE")
        is_upload = u_state.active_input_source == "UPLOAD"

        btn_bw = 105
        btn_y = y + 4
        btn_cam_x = x + w - (btn_bw * 3) - 20
        btn_demo_x = btn_cam_x + btn_bw + 6
        btn_up_x = btn_demo_x + btn_bw + 6

        self._register_button("btn_src_camera", btn_cam_x, btn_y, btn_bw, 22)
        draw_button(
            canvas, btn_cam_x, btn_y, btn_bw, 22, "Live Camera", "C",
            u_state.hovered_id == "btn_src_camera", is_cam, theme,
            accent_border=is_cam,
        )

        self._register_button("btn_src_demo", btn_demo_x, btn_y, btn_bw, 22)
        draw_button(
            canvas, btn_demo_x, btn_y, btn_bw, 22, "Demo Video", "V",
            u_state.hovered_id == "btn_src_demo", is_demo, theme,
            accent_border=is_demo,
        )

        self._register_button("btn_src_upload", btn_up_x, btn_y, btn_bw, 22)
        draw_button(
            canvas, btn_up_x, btn_y, btn_bw, 22, "Upload Video", "U",
            u_state.hovered_id == "btn_src_upload", is_upload, theme,
            accent_border=is_upload,
        )

        # Video viewport bounds (leave bottom space for transport controls in video mode)
        is_video_mode = is_demo or is_upload
        bottom_bar_h = 52 if is_video_mode else 0

        vx = x + 10
        vy = y + 36
        vw = w - 20
        vh = h - 46 - bottom_bar_h

        if camera_frame is not None and camera_frame.size > 0:
            # Scale video preserving aspect ratio to fit inside viewport
            fh, fw = camera_frame.shape[:2]
            scale = min(vw / fw, vh / fh)
            nw, nh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
            ox = vx + (vw - nw) // 2
            oy = vy + (vh - nh) // 2

            # Use INTER_AREA for sharp, alias-free downscaling; INTER_CUBIC for crisp upscaling without pixelation
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            resized = cv2.resize(camera_frame, (nw, nh), interpolation=interp)
            canvas[oy : oy + nh, ox : ox + nw] = resized
            cv2.rectangle(canvas, (ox, oy), (ox + nw, oy + nh), theme.card_border, 1, cv2.LINE_AA)

            # Viewport HUD: Strict 3-Row Non-Overlapping Layout
            # ROW 1: Source Mode Badge
            is_demo_src = p_state.frame_source == "DEMO_VIDEO" or is_demo or is_upload
            if is_demo_src:
                sample_name = (u_state.selected_sample_video or "DEMO").upper()
                row1_text = f"MODE: DEMO VIDEO [{sample_name}]"
                draw_badge(canvas, ox + 12, oy + 12, row1_text, fg_color=theme.text_dark, bg_color=theme.primary_accent, font_scale=0.38)
            else:
                draw_badge(canvas, ox + 12, oy + 12, "MODE: LIVE CAMERA", fg_color=theme.text_dark, bg_color=theme.secondary_accent, font_scale=0.38)

            # ROW 2: Multi-hand Telemetry Badge
            row2_y = oy + 38
            if p_state.hand_detected:
                if p_state.hand_count >= 2:
                    pri_lbl = (p_state.handedness or "RIGHT").upper()
                    sec_lbl = (p_state.secondary_handedness or "LEFT").upper()
                    row2_text = f"HANDS: 2 | PRIMARY: {pri_lbl} {p_state.detection_confidence * 100:.0f}% | SECONDARY: {sec_lbl} {p_state.secondary_confidence * 100:.0f}%"
                else:
                    pri_lbl = (p_state.handedness or "RIGHT").upper()
                    row2_text = f"HANDS: 1 | PRIMARY: {pri_lbl} {p_state.detection_confidence * 100:.0f}%"
                draw_badge(canvas, ox + 12, row2_y, row2_text, theme.secondary_accent, (20, 18, 16), theme.secondary_accent, font_scale=0.38)
            else:
                row2_text = "HANDS: 0 | SEARCHING FOR HANDS"
                draw_badge(canvas, ox + 12, row2_y, row2_text, theme.muted_accent, (22, 18, 16), theme.muted_accent, font_scale=0.38)

            # ROW 3: Prediction & Confidence Badge
            row3_y = oy + 64
            pred_tok = p_state.active_word_sign or p_state.predicted_label
            pred_conf = p_state.word_sign_confidence if p_state.active_word_sign else p_state.prediction_confidence
            if pred_tok:
                conf_pct = int(pred_conf * 100)
                if is_demo_src:
                    sample_name = (u_state.selected_sample_video or "DEMO").upper()
                    match_lbl = " [MATCH]" if pred_tok.upper() == sample_name else " [MISMATCH]"
                    row3_text = f"PREDICTION: {pred_tok} | CONFIDENCE: {conf_pct}%{match_lbl}"
                else:
                    row3_text = f"PREDICTION: {pred_tok} | CONFIDENCE: {conf_pct}%"
                pred_col = theme.primary_accent if conf_pct >= 70 else theme.warning_accent
                draw_badge(canvas, ox + 12, row3_y, row3_text, pred_col, (25, 20, 16), pred_col, font_scale=0.38)
            else:
                row3_text = "PREDICTION: WAITING... | CONFIDENCE: --"
                draw_badge(canvas, ox + 12, row3_y, row3_text, theme.muted_accent, (22, 18, 16), theme.muted_accent, font_scale=0.38)

            # Bottom prompt banner when no hand in view
            if not p_state.hand_detected:
                p_text = "PLACE HAND INSIDE CAMERA AREA" if is_cam else "PROCESSING VIDEO FRAMES"
                draw_badge(canvas, ox + (nw - 240) // 2, oy + nh - 36, p_text, theme.warning_accent, (15, 12, 10), theme.warning_accent)

            # In-feed Diagnostic HUD Overlay (Toggled via G key)
            if u_state.show_diagnostics:
                self._render_diagnostics_overlay(canvas, ox, oy, nw, nh, p_state, theme)

            # Pause Overlay if paused
            if p_state.is_paused:
                overlay = canvas[oy : oy + nh, ox : ox + nw].copy()
                cv2.rectangle(overlay, (0, 0), (nw, nh), (10, 10, 10), -1)
                cv2.addWeighted(overlay, 0.70, canvas[oy : oy + nh, ox : ox + nw], 0.30, 0, canvas[oy : oy + nh, ox : ox + nw])
                cv2.rectangle(canvas, (ox + nw // 2 - 170, oy + nh // 2 - 40), (ox + nw // 2 + 170, oy + nh // 2 + 40), theme.primary_accent, 2, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    "PAUSED",
                    (ox + nw // 2 - 60, oy + nh // 2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.95,
                    theme.primary_accent,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    "Press 'P' to resume video stream",
                    (ox + nw // 2 - 130, oy + nh // 2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    theme.text_secondary,
                    1,
                    cv2.LINE_AA,
                )
        else:
            # Camera unavailable graphic (only displayed upon sustained failure exceeding grace threshold)
            cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), (18, 14, 12), -1)
            msg_title = "PAUSED (WAITING FOR FRAME)" if p_state.is_paused else ("VIDEO READY / STOPPED" if is_video_mode else "CAMERA FEED UNAVAILABLE")
            cv2.putText(
                canvas,
                msg_title,
                (vx + vw // 2 - 150, vy + vh // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                theme.warning_accent if p_state.is_paused else theme.error_accent,
                1,
                cv2.LINE_AA,
            )
            sub_msg = (
                "Press 'P' or 'Play' to resume video stream"
                if p_state.is_paused
                else ("Select Demo Video or Click Play to Begin" if is_video_mode else "Check camera device connection or permissions")
            )
            cv2.putText(
                canvas,
                sub_msg,
                (vx + vw // 2 - 180, vy + vh // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                theme.text_muted,
                1,
                cv2.LINE_AA,
            )

        # ---------------------------------------------------------------------
        # Video Transport & Player Controls (Rendered in Demo & Upload Mode)
        # ---------------------------------------------------------------------
        if is_video_mode:
            py = y + h - 46
            cv2.rectangle(canvas, (x + 10, py - 4), (x + w - 10, py + 38), (22, 18, 16), -1)
            cv2.rectangle(canvas, (x + 10, py - 4), (x + w - 10, py + 38), theme.card_border, 1, cv2.LINE_AA)

            # Play, Pause, Restart, Stop buttons
            tb_w = 64
            tb_h = 28
            cx = x + 16

            self._register_button("btn_vid_play", cx, py, tb_w, tb_h)
            draw_button(canvas, cx, py, tb_w, tb_h, "Play", "", u_state.hovered_id == "btn_vid_play", False, theme)
            cx += tb_w + 6

            self._register_button("btn_vid_pause", cx, py, tb_w, tb_h)
            draw_button(canvas, cx, py, tb_w, tb_h, "Pause", "", u_state.hovered_id == "btn_vid_pause", False, theme)
            cx += tb_w + 6

            self._register_button("btn_vid_restart", cx, py, 74, tb_h)
            draw_button(canvas, cx, py, 74, tb_h, "Restart", "", u_state.hovered_id == "btn_vid_restart", False, theme)
            cx += 74 + 6

            self._register_button("btn_vid_stop", cx, py, tb_w, tb_h)
            draw_button(canvas, cx, py, tb_w, tb_h, "Stop", "", u_state.hovered_id == "btn_vid_stop", False, theme)
            cx += tb_w + 14

            # Status / Detection Telemetry
            det_label = p_state.active_word_sign or p_state.stable_token or p_state.predicted_label or "—"
            conf_pct = int((p_state.word_sign_confidence or p_state.prediction_confidence) * 100)
            cur_sec = int(u_state.video_current_time)
            tot_sec = int(u_state.video_duration)
            time_str = f"{cur_sec//60:02d}:{cur_sec%60:02d} / {tot_sec//60:02d}:{tot_sec%60:02d}"
            frame_str = f"Frame {u_state.video_current_frame}/{u_state.video_total_frames}" if u_state.video_total_frames > 0 else ""

            telem_parts = []
            if frame_str:
                telem_parts.append(f"{frame_str} ({time_str})")
            telem_parts.append(f"Sign: [{det_label}]")
            telem_parts.append(f"Conf: [{conf_pct}%]")
            telem_str = " | ".join(telem_parts)

            cv2.putText(
                canvas,
                telem_str,
                (cx, py + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                theme.secondary_accent if conf_pct >= 75 else theme.text_secondary,
                1,
                cv2.LINE_AA,
            )

            # RUN RECOGNITION button on the right
            rec_bw = 145
            rec_bx = x + w - rec_bw - 18
            self._register_button("btn_vid_run_rec", rec_bx, py, rec_bw, tb_h)
            draw_button(
                canvas, rec_bx, py, rec_bw, tb_h, "Run Recognition", "R",
                u_state.hovered_id == "btn_vid_run_rec", False, theme,
                accent_border=True,
            )

    def _render_diagnostics_overlay(
        self,
        canvas: np.ndarray,
        ox: int,
        oy: int,
        nw: int,
        nh: int,
        p_state: PipelineRuntimeState,
        theme: UITheme,
    ) -> None:
        """Renders live on-screen diagnostic HUD with hand, detection, and classification telemetry."""
        hud_w = 260
        hud_h = 145
        hx = ox + nw - hud_w - 12
        hy = oy + 12

        roi = canvas[hy : hy + hud_h, hx : hx + hud_w]
        if roi.size > 0:
            dark_bg = np.zeros_like(roi)
            cv2.addWeighted(roi, 0.25, dark_bg, 0.75, 0, roi)
        cv2.rectangle(canvas, (hx, hy), (hx + hud_w, hy + hud_h), theme.primary_accent, 1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            "DIAGNOSTIC HUD [G]",
            (hx + 10, hy + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            theme.primary_accent,
            1,
            cv2.LINE_AA,
        )

        lines = [
            f"Hand: {p_state.handedness or 'None'}",
            f"Detection Conf: {p_state.detection_confidence * 100:.1f}%",
            f"Predicted Sign: {p_state.predicted_label or '—'}",
            f"Classifier Conf: {p_state.prediction_confidence * 100:.1f}%",
            f"Stable Sign: {p_state.stable_token or '—'}",
            f"Stability Dwell: {p_state.dwell_duration_ms:.0f} ms",
        ]

        text_y = hy + 38
        for line in lines:
            cv2.putText(
                canvas,
                line,
                (hx + 10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                theme.text_primary,
                1,
                cv2.LINE_AA,
            )
            text_y += 18

    def _render_sidebar(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        p_state: PipelineRuntimeState,
        u_state: UIState,
        theme: UITheme,
    ) -> None:
        """Renders the right control stack: Recognition, Translation, Voice, and Action buttons."""
        card_gap = 12
        card_rec_h = 140
        card_trans_h = 160
        card_voice_h = 110
        card_actions_h = h - card_rec_h - card_trans_h - card_voice_h - (card_gap * 3)

        cy = y

        # 1. Sign Recognition Card
        self._render_recognition_card(canvas, x, cy, w, card_rec_h, p_state, u_state, theme)
        cy += card_rec_h + card_gap

        # 2. Translation Engine Card
        self._render_translation_card(canvas, x, cy, w, card_trans_h, p_state, u_state, theme)
        cy += card_trans_h + card_gap

        # 3. Voice / TTS Card
        self._render_voice_card(canvas, x, cy, w, card_voice_h, p_state, u_state, theme)
        cy += card_voice_h + card_gap

        # 4. Action Toolbar Card
        self._render_actions_card(canvas, x, cy, w, card_actions_h, p_state, u_state, theme)

    def _render_recognition_card(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        p_state: PipelineRuntimeState,
        u_state: UIState,
        theme: UITheme,
    ) -> None:
        """Renders live predicted sign, confidence bar, and dwell progress."""
        draw_card(canvas, x, y, w, h, theme.card_bg, theme.card_border, theme.border_width)
        # Mode Selection Toggles (Top Right of Card)
        btn_mw = 58
        btn_my = y + 4
        self._register_button("btn_mode_alpha", x + w - 190, btn_my, btn_mw, 22)
        draw_button(
            canvas, x + w - 190, btn_my, btn_mw, 22, "Alpha", "1",
            u_state.hovered_id == "btn_mode_alpha",
            p_state.recognition_mode == "ALPHABET",
            theme,
        )
        self._register_button("btn_mode_word", x + w - 126, btn_my, btn_mw, 22)
        draw_button(
            canvas, x + w - 126, btn_my, btn_mw, 22, "Word", "2",
            u_state.hovered_id == "btn_mode_word",
            p_state.recognition_mode == "WORD_SIGNS",
            theme,
        )
        self._register_button("btn_mode_auto", x + w - 62, btn_my, btn_mw, 22)
        draw_button(
            canvas, x + w - 62, btn_my, btn_mw, 22, "Auto", "3",
            u_state.hovered_id == "btn_mode_auto",
            p_state.recognition_mode == "AUTO",
            theme,
            accent_border=(p_state.recognition_mode == "AUTO"),
        )

        # Header title
        is_demo_mode = (
            p_state.frame_source == "DEMO_VIDEO"
            or u_state.active_input_source in ("DEMO", "SAMPLE")
        )
        if is_demo_mode:
            demo_target = (u_state.selected_sample_video or "DEMO").upper()
            cv2.putText(
                canvas,
                f"DEMO SOURCE: {demo_target}",
                (x + 14, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                theme.primary_accent,
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                canvas,
                "SIGN RECOGNITION",
                (x + 14, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                theme.text_secondary,
                1,
                cv2.LINE_AA,
            )

        # Decide what to display based on mode
        is_word_active = (
            p_state.recognition_mode in ("WORD_SIGNS", "AUTO")
            and bool(p_state.active_word_sign)
        )
        if is_word_active:
            pred_char = p_state.active_word_sign or "—"
            curr_conf = p_state.word_sign_confidence
        elif p_state.is_uncertain and p_state.hand_detected:
            pred_char = "UNCERTAIN"
            curr_conf = p_state.prediction_confidence
        else:
            pred_char = p_state.predicted_label or "—"
            curr_conf = p_state.prediction_confidence

        if is_demo_mode:
            demo_target = (u_state.selected_sample_video or "DEMO").upper()
            has_det = pred_char not in ("—", "UNCERTAIN")
            is_match = has_det and (pred_char.upper() == demo_target)

            bw = 148
            bx, by, bh = x + 14, y + 36, 92
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (26, 20, 18), -1)

            if not has_det:
                border_col = theme.card_border
                status_txt = "ANALYZING..."
                status_col = theme.text_muted
            elif is_match:
                border_col = theme.secondary_accent
                status_txt = "STATUS: MATCH"
                status_col = theme.secondary_accent
            else:
                border_col = theme.warning_accent
                status_txt = "STATUS: MISMATCH"
                status_col = theme.warning_accent

            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border_col, 2, cv2.LINE_AA)

            # Sub-labels inside card box
            cv2.putText(
                canvas,
                f"DEMO: {demo_target}",
                (bx + 8, by + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                theme.text_secondary,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"DETECTED: {pred_char[:10]}",
                (bx + 8, by + 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                theme.primary_accent if has_det else theme.text_muted,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                status_txt,
                (bx + 8, by + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                status_col,
                1,
                cv2.LINE_AA,
            )
        else:
            # Dynamic box sizing for words vs letters vs UNCERTAIN
            is_multi_char = len(pred_char) > 2
            bw = 125 if is_multi_char else 70
            bx, by, bh = x + 14, y + 38, 85
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (26, 20, 18), -1)
            if pred_char == "UNCERTAIN":
                border_col = theme.warning_accent
                char_col = theme.warning_accent
            elif curr_conf >= 0.75:
                border_col = theme.secondary_accent
                char_col = theme.primary_accent if p_state.hand_detected else theme.text_muted
            else:
                border_col = theme.card_border
                char_col = theme.primary_accent if p_state.hand_detected else theme.text_muted
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border_col, 2, cv2.LINE_AA)

            if is_multi_char:
                font_sz = 0.44 if pred_char == "UNCERTAIN" else 0.60
                font_th = 1 if pred_char == "UNCERTAIN" else 2
                text_x = bx + 10 if pred_char == "UNCERTAIN" else bx + 8
                cv2.putText(
                    canvas,
                    pred_char[:9],
                    (text_x, by + 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_sz,
                    char_col,
                    font_th,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    canvas,
                    pred_char,
                    (bx + 16, by + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    char_col,
                    3,
                    cv2.LINE_AA,
                )

        # Confidence & Dwell Meters
        mx = bx + bw + 16
        mw = w - (mx - x) - 14

        # Confidence Bar
        conf_pct = int(curr_conf * 100)
        mode_tag = f" [{p_state.recognition_mode}]"
        cv2.putText(
            canvas,
            f"Confidence: {conf_pct}%{mode_tag}",
            (mx, y + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            theme.font_scale_small,
            theme.text_secondary,
            1,
            cv2.LINE_AA,
        )

        conf_color = (
            theme.secondary_accent
            if conf_pct >= 80
            else (theme.warning_accent if conf_pct >= 50 else theme.error_accent)
        )
        draw_progress_bar(
            canvas,
            mx,
            y + 62,
            mw,
            10,
            curr_conf,
            conf_color,
            (30, 24, 20),
            theme.card_border,
        )

        # Temporal Dwell Bar
        dwell_pct = int(p_state.dwell_progress * 100)
        dwell_lbl = "Dwell Stability: Locked" if dwell_pct >= 100 else f"Dwell Stability: {dwell_pct}%"
        cv2.putText(
            canvas,
            dwell_lbl,
            (mx, y + 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            theme.font_scale_small,
            theme.text_secondary,
            1,
            cv2.LINE_AA,
        )
        draw_progress_bar(
            canvas,
            mx,
            y + 100,
            mw,
            8,
            p_state.dwell_progress,
            theme.primary_accent,
            (30, 24, 20),
            theme.card_border,
        )

    def _render_translation_card(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        p_state: PipelineRuntimeState,
        u_state: UIState,
        theme: UITheme,
    ) -> None:
        """Renders active word, accumulated sentence, and word count."""
        draw_card(canvas, x, y, w, h, theme.card_bg, theme.card_border, theme.border_width)
        draw_card_header(canvas, x, y, w, "Translation Engine", theme, accent_color=theme.secondary_accent)

        # Active in-flight word
        cv2.putText(
            canvas,
            "ACTIVE WORD:",
            (x + 14, y + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            theme.font_scale_small,
            theme.text_secondary,
            1,
            cv2.LINE_AA,
        )
        word_str = p_state.current_word if p_state.current_word else "(empty)"
        word_col = theme.primary_accent if p_state.current_word else theme.text_muted
        cv2.putText(
            canvas,
            word_str,
            (x + 115, y + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            word_col,
            2,
            cv2.LINE_AA,
        )

        # Full Sentence (Focal Point)
        cv2.putText(
            canvas,
            "SENTENCE:",
            (x + 14, y + 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            theme.font_scale_small,
            theme.text_secondary,
            1,
            cv2.LINE_AA,
        )

        # Sentence container box
        sx, sy, sw, sh = x + 14, y + 92, w - 28, 55
        cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), (20, 16, 14), -1)
        cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), theme.card_border, 1, cv2.LINE_AA)

        sent_text = p_state.sentence if p_state.sentence else "Waiting for words..."
        sent_col = theme.text_primary if p_state.sentence else theme.text_muted
        cv2.putText(
            canvas,
            sent_text[:40] + ("..." if len(sent_text) > 40 else ""),
            (sx + 10, sy + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            sent_col,
            2,
            cv2.LINE_AA,
        )

    def _render_voice_card(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        p_state: PipelineRuntimeState,
        u_state: UIState,
        theme: UITheme,
    ) -> None:
        """Renders offline voice status and quick Speak / Stop / Toggle actions."""
        draw_card(canvas, x, y, w, h, theme.card_bg, theme.card_border, theme.border_width)
        draw_card_header(canvas, x, y, w, "Voice Output (SAPI5)", theme, accent_color=theme.primary_accent)

        # Voice status badge in card header
        if not p_state.voice_enabled:
            v_badge_col, v_badge_txt = theme.muted_accent, "VOICE: DISABLED"
        elif p_state.is_speaking:
            v_badge_col, v_badge_txt = theme.primary_accent, "VOICE: SPEAKING"
        elif p_state.voice_status == "ERROR":
            v_badge_col, v_badge_txt = theme.error_accent, "VOICE: ERROR"
        else:
            v_badge_col, v_badge_txt = theme.secondary_accent, "VOICE: READY"

        draw_badge(
            canvas,
            x + w - 135,
            y + 6,
            v_badge_txt,
            fg_color=v_badge_col,
            bg_color=theme.card_bg,
            border_color=v_badge_col,
            font_scale=0.34,
        )

        # Status or Error banner
        if p_state.voice_status == "ERROR" or p_state.voice_error_message:
            err_msg = p_state.voice_error_message or "Voice output unavailable — no compatible Windows TTS voice was found."
            cv2.putText(
                canvas,
                err_msg[:50],
                (x + 14, y + 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                theme.error_accent,
                1,
                cv2.LINE_AA,
            )
        else:
            last = f' "{p_state.last_spoken}"' if p_state.last_spoken else ' "—"'
            v_info = f"Last Spoken: {last[:25]}"
            cv2.putText(
                canvas,
                v_info,
                (x + 14, y + 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                theme.font_scale_small,
                theme.text_secondary,
                1,
                cv2.LINE_AA,
            )

        # Voice Action Buttons
        bw = (w - 40) // 3
        bx1 = x + 14
        bx2 = bx1 + bw + 6
        bx3 = bx2 + bw + 6
        by = y + 68
        bh = 30

        # Speak Button
        self._register_button("btn_speak", bx1, by, bw, bh)
        can_speak = bool(
            p_state.active_word_sign
            or p_state.stable_token
            or p_state.current_word
            or p_state.sentence
            or p_state.predicted_label
        )
        draw_button(
            canvas,
            bx1,
            by,
            bw,
            bh,
            "Speak",
            "S",
            u_state.hovered_id == "btn_speak",
            u_state.clicked_id == "btn_speak",
            theme,
            accent_border=True,
            enabled=can_speak,
        )

        # Stop Button
        self._register_button("btn_stop", bx2, by, bw, bh)
        draw_button(
            canvas,
            bx2,
            by,
            bw,
            bh,
            "Stop",
            "X",
            u_state.hovered_id == "btn_stop",
            u_state.clicked_id == "btn_stop",
            theme,
        )

        # Voice Toggle Button
        self._register_button("btn_voice_toggle", bx3, by, bw, bh)
        v_lbl = "Mute" if p_state.voice_enabled else "Unmute"
        draw_button(
            canvas,
            bx3,
            by,
            bw,
            bh,
            v_lbl,
            "T",
            u_state.hovered_id == "btn_voice_toggle",
            u_state.clicked_id == "btn_voice_toggle",
            theme,
        )

    def _render_actions_card(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        p_state: PipelineRuntimeState,
        u_state: UIState,
        theme: UITheme,
    ) -> None:
        """Renders grid of interactive buttons for primary keyboard/mouse actions."""
        draw_card(canvas, x, y, w, h, theme.card_bg, theme.card_border, theme.border_width)
        draw_card_header(canvas, x, y, w, "Quick Actions", theme)

        # 2x3 Grid of Actions
        bw = (w - 36) // 2
        bh = 30
        gap_y = 8
        gy = y + 36

        actions = [
            ("btn_space", "Word Break", "SPACE"),
            ("btn_backspace", "Delete", "BS"),
            ("btn_clear", "Clear", "C"),
            ("btn_undo", "Undo", "U"),
            ("btn_pause", "Pause/Resume", "P"),
            ("btn_reset", "Reset All", "R"),
        ]

        for i, (bid, lbl, hk) in enumerate(actions):
            col = i % 2
            row = i // 2
            bx = x + 14 + col * (bw + 8)
            by = gy + row * (bh + gap_y)

            self._register_button(bid, bx, by, bw, bh)
            draw_button(
                canvas,
                bx,
                by,
                bw,
                bh,
                lbl,
                hk,
                u_state.hovered_id == bid,
                u_state.clicked_id == bid,
                theme,
            )

    def _render_footer(
        self,
        canvas: np.ndarray,
        p_state: PipelineRuntimeState,
        u_state: UIState,
        theme: UITheme,
    ) -> None:
        """Renders bottom status strip with human-friendly guidance or active toasts."""
        w, h = self.width, self.height
        fy = h - 34
        cv2.rectangle(canvas, (0, fy), (w, h), theme.footer_bg, -1)
        cv2.line(canvas, (0, fy), (w, fy), theme.card_border, 1, cv2.LINE_AA)

        # Active Toast or Contextual Advice
        if u_state.toast and not u_state.toast.is_expired:
            msg = f"[NOTICE] {u_state.toast.message}"
            col = (
                theme.secondary_accent
                if u_state.toast.level == "success"
                else (theme.warning_accent if u_state.toast.level == "warning" else theme.primary_accent)
            )
        else:
            msg = u_state.last_action_text
            col = theme.text_secondary

        cv2.putText(
            canvas,
            msg[:70],
            (20, fy + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            theme.font_scale_small,
            col,
            1,
            cv2.LINE_AA,
        )

        # Right Side Help Hint
        hint = "Press [H] for Help & Controls Guide  |  [Q] Quit"
        cv2.putText(
            canvas,
            hint,
            (w - 380, fy + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            theme.font_scale_small - 0.04,
            theme.text_muted,
            1,
            cv2.LINE_AA,
        )

    def _register_button(self, btn_id: str, x: int, y: int, w: int, h: int) -> None:
        self._buttons[btn_id] = (x, y, w, h)

    @staticmethod
    def _derive_app_status(p_state: PipelineRuntimeState, u_state: UIState) -> AppStatus:
        """Determines high-level AppStatus based on pipeline telemetry."""
        if p_state.error_state:
            return AppStatus.MODEL_ERROR
        if p_state.is_paused:
            return AppStatus.PAUSED
        if p_state.is_speaking:
            return AppStatus.SPEAKING
        if p_state.temporal_state == TemporalState.EMITTED:
            return AppStatus.STABLE_EMITTED
        if p_state.temporal_state == TemporalState.STABLE:
            return AppStatus.DWELLING
        if p_state.hand_detected:
            return AppStatus.RECOGNIZING
        return AppStatus.NO_HAND
