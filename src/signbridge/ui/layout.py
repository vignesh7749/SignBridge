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
    draw_rounded_rect,
    draw_status_indicator,
    draw_text,
    draw_vector_icon,
    hit_test,
    scale_and_letterbox_frame,
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

        # Brand Icon + Title in crisp TrueType
        cv2.circle(canvas, (24, 24), 8, theme.primary_accent, -1, cv2.LINE_AA)
        cv2.circle(canvas, (24, 24), 4, theme.header_bg, -1, cv2.LINE_AA)
        draw_text(canvas, "SIGNBRIDGE", (38, 14), font_size=18, color=theme.text_primary, bold=True)

        mode_badge = "DEMO" if u_state.demo_mode else "PRO"
        draw_badge(
            canvas,
            185,
            13,
            mode_badge,
            fg_color=theme.text_dark,
            bg_color=theme.primary_accent,
            font_scale=0.35,
        )

        # Global Status Indicator
        app_status = self._derive_app_status(p_state, u_state)
        draw_status_indicator(
            canvas,
            245,
            13,
            app_status,
            theme,
            is_speaking=p_state.is_speaking,
        )

        # Telemetry & Voice Badges on Right side
        rx = w - 16

        # Help Button [?]
        btn_help_w = 68
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
        btn_set_w = 90
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
        btn_demo_w = 96
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
            canvas, btn_cam_x, btn_y, btn_bw, 22, "Live Camera", "",
            u_state.hovered_id == "btn_src_camera", is_cam, theme,
            accent_border=is_cam,
        )

        self._register_button("btn_src_demo", btn_demo_x, btn_y, btn_bw, 22)
        draw_button(
            canvas, btn_demo_x, btn_y, btn_bw, 22, "Demo Video", "",
            u_state.hovered_id == "btn_src_demo", is_demo, theme,
            accent_border=is_demo,
        )

        self._register_button("btn_src_upload", btn_up_x, btn_y, btn_bw, 22)
        draw_button(
            canvas, btn_up_x, btn_y, btn_bw, 22, "Upload Video", "",
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
                raw_sample = (u_state.selected_sample_video or "DEMO")
                sample_name = raw_sample.split("/")[0].strip().upper()
                row1_text = f"REFERENCE DEMO: [{sample_name}]"
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
                    raw_sample = (u_state.selected_sample_video or "DEMO")
                    sample_name = raw_sample.split("/")[0].strip().upper()
                    is_core_word = sample_name in ("HELLO", "HELP", "YES", "NO", "STOP", "THANK YOU")
                    if is_core_word:
                        match_lbl = " [MATCH]" if pred_tok.upper() == sample_name else " [MISMATCH]"
                        row3_text = f"EVALUATION: {pred_tok} | CONF: {conf_pct}%{match_lbl}"
                        pred_col = theme.secondary_accent if pred_tok.upper() == sample_name else theme.warning_accent
                    else:
                        row3_text = f"EVALUATION: {pred_tok} | CONF: {conf_pct}% [REF DEMO - UNTRAINED SIGN]"
                        pred_col = theme.primary_accent if conf_pct >= 70 else theme.warning_accent
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
                p_box_w, p_box_h = 320, 80
                p_bx, p_by = ox + (nw - p_box_w) // 2, oy + (nh - p_box_h) // 2
                draw_rounded_rect(canvas, p_bx, p_by, p_box_w, p_box_h, radius=8, bg_color=(20, 16, 14), border_color=theme.primary_accent, border_width=2)
                draw_text(canvas, "STREAM PAUSED", (p_bx + p_box_w // 2, p_by + 24), font_size=18, color=theme.primary_accent, bold=True, anchor="cc")
                draw_text(canvas, "Press 'P' or click Pause to resume", (p_bx + p_box_w // 2, p_by + 54), font_size=12, color=theme.text_secondary, anchor="cc")
        else:
            # Camera unavailable graphic (only displayed upon sustained failure exceeding grace threshold)
            draw_rounded_rect(canvas, vx, vy, vw, vh, radius=6, bg_color=(18, 14, 12), border_color=theme.card_border, border_width=1)
            msg_title = "PAUSED (WAITING FOR FRAME)" if p_state.is_paused else ("VIDEO READY / STOPPED" if is_video_mode else "CAMERA FEED UNAVAILABLE")
            col = theme.warning_accent if p_state.is_paused else (theme.primary_accent if is_video_mode else theme.error_accent)
            draw_text(canvas, msg_title, (vx + vw // 2, vy + vh // 2 - 16), font_size=16, color=col, bold=True, anchor="cc")
            sub_msg = (
                "Press 'P' or 'Play' to resume video stream"
                if p_state.is_paused
                else ("Select Demo Video or Click Play to Begin" if is_video_mode else "Check camera device connection or permissions")
            )
            draw_text(canvas, sub_msg, (vx + vw // 2, vy + vh // 2 + 16), font_size=12, color=theme.text_muted, anchor="cc")

        # ---------------------------------------------------------------------
        # Video Transport & Player Controls (Rendered in Demo & Upload Mode)
        # ---------------------------------------------------------------------
        if is_video_mode:
            py = y + h - 46
            cv2.rectangle(canvas, (x + 10, py - 4), (x + w - 10, py + 38), (22, 18, 16), -1)
            cv2.rectangle(canvas, (x + 10, py - 4), (x + w - 10, py + 38), theme.card_border, 1, cv2.LINE_AA)

            # Video Navigation & Transport buttons
            tb_h = 28
            cx = x + 16

            self._register_button("btn_vid_prev_sample", cx, py, 64, tb_h)
            draw_button(canvas, cx, py, 64, tb_h, "Prev", "", u_state.hovered_id == "btn_vid_prev_sample", False, theme, icon="prev")
            cx += 64 + 4

            self._register_button("btn_vid_play", cx, py, 58, tb_h)
            draw_button(canvas, cx, py, 58, tb_h, "Play", "", u_state.hovered_id == "btn_vid_play", False, theme, icon="play")
            cx += 58 + 4

            self._register_button("btn_vid_pause", cx, py, 60, tb_h)
            draw_button(canvas, cx, py, 60, tb_h, "Pause", "", u_state.hovered_id == "btn_vid_pause", False, theme, icon="pause")
            cx += 60 + 4

            self._register_button("btn_vid_restart", cx, py, 68, tb_h)
            draw_button(canvas, cx, py, 68, tb_h, "Restart", "", u_state.hovered_id == "btn_vid_restart", False, theme, icon="replay")
            cx += 68 + 4

            self._register_button("btn_vid_stop", cx, py, 58, tb_h)
            draw_button(canvas, cx, py, 58, tb_h, "Stop", "", u_state.hovered_id == "btn_vid_stop", False, theme, icon="stop")
            cx += 58 + 4

            self._register_button("btn_vid_next_sample", cx, py, 64, tb_h)
            draw_button(canvas, cx, py, 64, tb_h, "Next", "", u_state.hovered_id == "btn_vid_next_sample", False, theme, icon="next")
            cx += 64 + 8

            # RUN RECOGNITION button on the right
            rec_bw = 145
            rec_bx = x + w - rec_bw - 18
            self._register_button("btn_vid_run_rec", rec_bx, py, rec_bw, tb_h)
            draw_button(
                canvas, rec_bx, py, rec_bw, tb_h, "Run Recognition", "",
                u_state.hovered_id == "btn_vid_run_rec", False, theme,
                accent_border=True,
            )

            # Responsive Status / Detection Telemetry between transport buttons and run button
            avail_telem_w = rec_bx - cx - 12
            if avail_telem_w >= 80:
                det_label = p_state.active_word_sign or p_state.stable_token or p_state.predicted_label or "—"
                conf_pct = int((p_state.word_sign_confidence or p_state.prediction_confidence) * 100)
                cur_sec = int(u_state.video_current_time)
                tot_sec = int(u_state.video_duration)
                time_str = f"{cur_sec//60:02d}:{cur_sec%60:02d}/{tot_sec//60:02d}:{tot_sec%60:02d}"

                if avail_telem_w >= 260:
                    telem_str = f"{time_str} | Sign: [{det_label}] | Conf: [{conf_pct}%]"
                elif avail_telem_w >= 160:
                    telem_str = f"{time_str} | [{det_label}] ({conf_pct}%)"
                else:
                    telem_str = f"[{det_label}] {conf_pct}%"

                max_chars = max(10, avail_telem_w // 7)
                if len(telem_str) > max_chars:
                    telem_str = telem_str[:max_chars - 3] + "..."

                draw_text(
                    canvas,
                    telem_str,
                    (cx, py + 7),
                    font_size=11,
                    color=theme.secondary_accent if conf_pct >= 75 else theme.text_secondary,
                    bold=True,
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
        hud_w = 270
        hud_h = 160
        hx = ox + nw - hud_w - 12
        hy = oy + 12

        roi = canvas[hy : hy + hud_h, hx : hx + hud_w]
        if roi.size > 0:
            dark_bg = np.zeros_like(roi)
            cv2.addWeighted(roi, 0.25, dark_bg, 0.75, 0, roi)
        draw_rounded_rect(canvas, hx, hy, hud_w, hud_h, radius=6, bg_color=(20, 16, 14), border_color=theme.primary_accent, border_width=1)

        draw_text(canvas, "DIAGNOSTIC HUD [G]", (hx + 12, hy + 10), font_size=12, color=theme.primary_accent, bold=True)

        lines = [
            f"Hand: {p_state.handedness or 'None'}",
            f"Detection Conf: {p_state.detection_confidence * 100:.1f}%",
            f"Predicted Sign: {p_state.predicted_label or '—'}",
            f"Classifier Conf: {p_state.prediction_confidence * 100:.1f}%",
            f"Stable Sign: {p_state.stable_token or '—'}",
            f"Stability Dwell: {p_state.dwell_duration_ms:.0f} ms",
        ]

        text_y = hy + 32
        for line in lines:
            draw_text(canvas, line, (hx + 12, text_y), font_size=11, color=theme.text_primary)
            text_y += 19

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
        btn_mw = 68
        btn_my = y + 4
        btn_alpha_x = x + w - (btn_mw * 3) - 26
        btn_word_x = btn_alpha_x + btn_mw + 6
        btn_auto_x = btn_word_x + btn_mw + 6

        self._register_button("btn_mode_alpha", btn_alpha_x, btn_my, btn_mw, 22)
        draw_button(
            canvas, btn_alpha_x, btn_my, btn_mw, 22, "Alpha", "1",
            u_state.hovered_id == "btn_mode_alpha",
            p_state.recognition_mode == "ALPHABET",
            theme,
        )
        self._register_button("btn_mode_word", btn_word_x, btn_my, btn_mw, 22)
        draw_button(
            canvas, btn_word_x, btn_my, btn_mw, 22, "Word", "2",
            u_state.hovered_id == "btn_mode_word",
            p_state.recognition_mode == "WORD_SIGNS",
            theme,
        )
        self._register_button("btn_mode_auto", btn_auto_x, btn_my, btn_mw, 22)
        draw_button(
            canvas, btn_auto_x, btn_my, btn_mw, 22, "Auto", "3",
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
            raw_target = (u_state.selected_sample_video or "DEMO")
            demo_target = raw_target.upper()
            if "GOODBYE" in demo_target:
                demo_target = "GOODBYE"
            elif "/" in demo_target:
                demo_target = demo_target.split("/")[0].strip()
            draw_text(
                canvas,
                f"SIGN RECOGNITION [DEMO: {demo_target[:14]}]",
                (x + 14, y + 10),
                font_size=11,
                color=theme.primary_accent,
                bold=True,
            )
        else:
            draw_text(
                canvas,
                "SIGN RECOGNITION",
                (x + 14, y + 10),
                font_size=11,
                color=theme.text_secondary,
                bold=True,
            )

        # Decide what to display based on mode and validated hand detection
        if not p_state.hand_detected:
            pred_char = "—"
            curr_conf = 0.0
        elif p_state.recognition_mode == "WORD_SIGNS":
            if p_state.active_word_sign:
                pred_char = p_state.active_word_sign
                curr_conf = p_state.word_sign_confidence
            elif p_state.is_uncertain:
                pred_char = "UNCERTAIN"
                curr_conf = p_state.word_sign_confidence
            else:
                pred_char = "—"
                curr_conf = 0.0
        elif p_state.recognition_mode == "ALPHABET":
            if p_state.is_uncertain:
                pred_char = "UNCERTAIN"
                curr_conf = p_state.prediction_confidence
            else:
                pred_char = p_state.predicted_label or "—"
                curr_conf = p_state.prediction_confidence
        else:  # AUTO
            if p_state.active_word_sign:
                pred_char = p_state.active_word_sign
                curr_conf = p_state.word_sign_confidence
            elif p_state.is_uncertain:
                pred_char = "UNCERTAIN"
                curr_conf = p_state.prediction_confidence
            else:
                pred_char = p_state.predicted_label or "—"
                curr_conf = p_state.prediction_confidence

        # Determine prediction card box geometry (pure text-based)
        is_multi_char = len(pred_char) > 2
        bw = 135 if is_multi_char else 80
        bx, by, bh = x + 14, y + 38, 85

        if pred_char == "UNCERTAIN":
            border_col = theme.warning_accent
            char_col = theme.warning_accent
        elif curr_conf >= 0.75:
            border_col = theme.secondary_accent
            char_col = theme.primary_accent if p_state.hand_detected else theme.text_muted
        else:
            border_col = theme.card_border
            char_col = theme.primary_accent if p_state.hand_detected else theme.text_muted

        draw_rounded_rect(canvas, bx, by, bw, bh, radius=6, bg_color=(26, 20, 18), border_color=border_col, border_width=2)

        # Display validated prediction prominently as text (clean em-dash when empty)
        if pred_char == "—":
            draw_text(
                canvas,
                "—",
                (bx + bw // 2, by + bh // 2),
                font_size=36,
                color=theme.text_muted,
                bold=True,
                anchor="cc",
            )
        elif is_multi_char:
            draw_text(
                canvas,
                pred_char[:12],
                (bx + bw // 2, by + bh // 2),
                font_size=16 if pred_char == "UNCERTAIN" else 18,
                color=char_col,
                bold=True,
                anchor="cc",
            )
        else:
            draw_text(
                canvas,
                pred_char,
                (bx + bw // 2, by + bh // 2),
                font_size=40,
                color=char_col,
                bold=True,
                anchor="cc",
            )

        # Confidence & Dwell Meters
        mx = bx + bw + 14
        mw = w - (mx - x) - 14

        # Confidence Bar
        conf_pct = int(curr_conf * 100)
        if is_demo_mode:
            raw_target = (u_state.selected_sample_video or "DEMO")
            demo_target = raw_target.upper()
            if "GOODBYE" in demo_target:
                demo_target = "GOODBYE"
            elif "/" in demo_target:
                demo_target = demo_target.split("/")[0].strip()
            is_core_word = demo_target in ("HELLO", "HELP", "YES", "NO", "STOP", "THANK YOU", "GOODBYE")
            is_letter_demo = len(demo_target) == 1
            if not is_core_word and not is_letter_demo:
                mode_tag = " [REFERENCE]"
            elif pred_char not in ("—", "UNCERTAIN"):
                is_match = (pred_char.upper() == demo_target)
                mode_tag = " [MATCH]" if is_match else " [MISMATCH]"
            else:
                mode_tag = " [DEMO]"
        else:
            mode_tag = f" [{p_state.recognition_mode}]"

        draw_text(
            canvas,
            f"Confidence: {conf_pct}%{mode_tag}",
            (mx, y + 40),
            font_size=11,
            color=theme.text_secondary,
        )

        conf_color = (
            theme.secondary_accent
            if conf_pct >= 80
            else (theme.warning_accent if conf_pct >= 50 else theme.error_accent)
        )
        draw_progress_bar(
            canvas,
            mx,
            y + 58,
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
        draw_text(
            canvas,
            dwell_lbl,
            (mx, y + 80),
            font_size=11,
            color=theme.text_secondary,
        )
        draw_progress_bar(
            canvas,
            mx,
            y + 98,
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

        # Active in-flight word (currently assembled character or word)
        draw_text(canvas, "ACTIVE WORD:", (x + 14, y + 40), font_size=11, color=theme.text_secondary, bold=True)
        word_str = p_state.current_word if p_state.current_word else "(empty)"
        word_col = theme.primary_accent if p_state.current_word else theme.text_muted
        draw_text(canvas, word_str, (x + 115, y + 37), font_size=15, color=word_col, bold=True)

        # Word counter reflects actual completed sentence contents
        word_count = len(p_state.completed_words)
        badge_txt = f"{word_count} {'WORD' if word_count == 1 else 'WORDS'}"
        draw_badge(
            canvas,
            x + w - 100,
            y + 6,
            badge_txt,
            fg_color=theme.secondary_accent,
            bg_color=theme.card_bg,
            border_color=theme.secondary_accent,
            font_scale=0.34,
        )

        # Full Accumulated Sentence (Committed text)
        draw_text(canvas, "SENTENCE:", (x + 14, y + 68), font_size=11, color=theme.text_secondary, bold=True)

        # Sentence container box with multi-line wrapping and padding
        sx, sy, sw, sh = x + 14, y + 86, w - 28, 62
        draw_rounded_rect(canvas, sx, sy, sw, sh, radius=6, bg_color=(20, 16, 14), border_color=theme.card_border, border_width=1)

        completed_sentence = " ".join(p_state.completed_words).strip()
        if not completed_sentence:
            draw_text(canvas, "Awaiting sign language input...", (sx + 14, sy + 21), font_size=13, color=theme.text_muted)
        else:
            words = completed_sentence.split()
            lines: list[str] = []
            cur_line: list[str] = []
            cur_len = 0
            max_line_chars = max(18, (sw - 28) // 9)

            for wd in words:
                if cur_len + len(wd) + (1 if cur_line else 0) <= max_line_chars:
                    cur_line.append(wd)
                    cur_len += len(wd) + (1 if cur_line else 0)
                else:
                    if cur_line:
                        lines.append(" ".join(cur_line))
                    cur_line = [wd]
                    cur_len = len(wd)
            if cur_line:
                lines.append(" ".join(cur_line))

            if len(lines) == 1:
                draw_text(canvas, lines[0], (sx + 14, sy + 20), font_size=17, color=theme.text_primary, bold=True)
            elif len(lines) == 2:
                draw_text(canvas, lines[0], (sx + 14, sy + 10), font_size=14, color=theme.text_primary, bold=True)
                draw_text(canvas, lines[1], (sx + 14, sy + 32), font_size=14, color=theme.text_primary, bold=True)
            else:
                draw_text(canvas, lines[0], (sx + 14, sy + 7), font_size=12, color=theme.text_primary, bold=True)
                draw_text(canvas, lines[1], (sx + 14, sy + 24), font_size=12, color=theme.text_primary, bold=True)
                line3 = " ".join(lines[2:])
                if len(line3) > max_line_chars + 4:
                    line3 = line3[:max_line_chars + 1] + "..."
                draw_text(canvas, line3, (sx + 14, sy + 41), font_size=12, color=theme.text_primary, bold=True)

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
            draw_text(canvas, err_msg[:55], (x + 14, y + 42), font_size=11, color=theme.error_accent)
        else:
            last = f' "{p_state.last_spoken}"' if p_state.last_spoken else ' "—"'
            v_info = f"Last Spoken: {last[:30]}"
            draw_text(canvas, v_info, (x + 14, y + 42), font_size=11, color=theme.text_secondary)

        # Voice Action Buttons with equal height (32px)
        bw = (w - 40) // 3
        bx1 = x + 14
        bx2 = bx1 + bw + 6
        bx3 = bx2 + bw + 6
        by = y + 68
        bh = 32

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

        # 2x3 Grid of Actions with equal height (32px)
        bw = (w - 36) // 2
        bh = 32
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

        draw_text(canvas, msg[:85], (20, fy + 8), font_size=12, color=col)

        # Right Side Help Hint
        hint = "Press [H] Help  |  [V] Demo Hub  |  [Q] Quit"
        draw_text(canvas, hint, (w - 20, fy + 8), font_size=12, color=theme.text_muted, anchor="rt")

    def _register_button(self, btn_id: str, x: int, y: int, w: int, h: int) -> None:
        self._buttons[btn_id] = (x, y, w, h)

    @staticmethod
    def _derive_app_status(p_state: PipelineRuntimeState, u_state: UIState) -> AppStatus:
        """Determines high-level AppStatus based on pipeline telemetry."""
        if p_state.camera_status and getattr(p_state.camera_status, "name", "") == "DISCONNECTED":
            return AppStatus.CAMERA_ERROR
        if p_state.error_state:
            err_lower = p_state.error_state.lower()
            if any(k in err_lower for k in ("camera", "device", "capture", "disconnected", "webcam")):
                return AppStatus.CAMERA_ERROR
            elif any(k in err_lower for k in ("voice", "tts", "audio", "speech", "sapi")):
                return AppStatus.VOICE_ERROR
            elif any(k in err_lower for k in ("model", "classifier", "inference", "feature", "landmark")):
                return AppStatus.MODEL_ERROR
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
