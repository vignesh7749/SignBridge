"""Stage 11: Interactive Modal Dialog Overlays.

Renders SettingsModal, HelpModal, and OnboardingModal with clickable controls
and accessibility configurations.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from signbridge.ui.components import (
    draw_button,
    draw_card,
    draw_card_header,
    draw_switch,
    hit_test,
)
from signbridge.ui.state import ActiveModal, UIState
from signbridge.ui.theme import UITheme


class ModalOverlayManager:
    """Manages translucent modal overlays and processes modal-specific clicks."""

    def __init__(self) -> None:
        self._buttons: Dict[str, Tuple[int, int, int, int]] = {}

    def get_registered_buttons(self) -> Dict[str, Tuple[int, int, int, int]]:
        return dict(self._buttons)

    def render(
        self,
        canvas: np.ndarray,
        ui_state: UIState,
        theme: UITheme,
        app_settings: dict,
    ) -> None:
        """Renders active modal over canvas with dimmed backdrop."""
        if ui_state.active_modal == ActiveModal.NONE:
            return

        self._buttons.clear()
        h, w = canvas.shape[:2]

        # 1. Dimmed Backdrop
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 8, 7), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

        # 2. Render Specific Modal Dialog
        if ui_state.active_modal == ActiveModal.SETTINGS:
            self._render_settings_modal(canvas, w, h, ui_state, theme, app_settings)
        elif ui_state.active_modal == ActiveModal.HELP:
            self._render_help_modal(canvas, w, h, ui_state, theme)
        elif ui_state.active_modal == ActiveModal.ONBOARDING:
            self._render_onboarding_modal(canvas, w, h, ui_state, theme)
        elif ui_state.active_modal == ActiveModal.EMERGENCY:
            self._render_emergency_modal(canvas, w, h, ui_state, theme)
        elif ui_state.active_modal == ActiveModal.VIDEO_DEMO:
            self._render_video_demo_modal(canvas, w, h, ui_state, theme)

    def _render_settings_modal(
        self,
        canvas: np.ndarray,
        w: int,
        h: int,
        ui_state: UIState,
        theme: UITheme,
        settings: dict,
    ) -> None:
        """Renders preferences and accessibility configuration dialog."""
        mw, mh = 620, 500
        mx = (w - mw) // 2
        my = (h - mh) // 2

        # Card container
        draw_card(canvas, mx, my, mw, mh, theme.card_bg, theme.primary_accent, border_width=2)
        draw_card_header(canvas, mx, my, mw, "Application Settings & Accessibility", theme)

        # Close [X] button top right
        btn_close_x = mx + mw - 36
        btn_close_y = my + 4
        self._register_button("modal_close", btn_close_x, btn_close_y, 28, 22)
        draw_button(
            canvas,
            btn_close_x,
            btn_close_y,
            28,
            22,
            "X",
            None,
            ui_state.hovered_id == "modal_close",
            ui_state.clicked_id == "modal_close",
            theme,
        )

        sy = my + 44

        # Setting 1: Voice Output Toggle
        self._register_button("set_toggle_voice", mx + 20, sy, mw - 40, 32)
        draw_switch(
            canvas,
            mx + 20,
            sy,
            mw - 40,
            32,
            is_on=settings.get("voice_enabled", True),
            label="Voice Synthesis Output (SAPI5 Offline)",
            is_hovered=ui_state.hovered_id == "set_toggle_voice",
            theme=theme,
        )
        sy += 40

        # Setting 2: High Contrast Mode (Accessibility)
        self._register_button("set_toggle_contrast", mx + 20, sy, mw - 40, 32)
        draw_switch(
            canvas,
            mx + 20,
            sy,
            mw - 40,
            32,
            is_on=ui_state.high_contrast,
            label="High Contrast Display Mode (WCAG AAA)",
            is_hovered=ui_state.hovered_id == "set_toggle_contrast",
            theme=theme,
        )
        sy += 40

        # Setting 3: Reduced Motion (Accessibility)
        self._register_button("set_toggle_motion", mx + 20, sy, mw - 40, 32)
        draw_switch(
            canvas,
            mx + 20,
            sy,
            mw - 40,
            32,
            is_on=ui_state.reduced_motion,
            label="Reduced Motion (Disable Animation Effects)",
            is_hovered=ui_state.hovered_id == "set_toggle_motion",
            theme=theme,
        )
        sy += 40

        # Setting 4: Speak Words at Boundary
        self._register_button("set_toggle_speak_words", mx + 20, sy, mw - 40, 32)
        draw_switch(
            canvas,
            mx + 20,
            sy,
            mw - 40,
            32,
            is_on=settings.get("speak_words", False),
            label="Speak Each Word at Boundary (in addition to Sentence)",
            is_hovered=ui_state.hovered_id == "set_toggle_speak_words",
            theme=theme,
        )
        sy += 40

        # Setting 5: Show Hand Skeleton Overlay
        self._register_button("set_toggle_skeleton", mx + 20, sy, mw - 40, 32)
        draw_switch(
            canvas,
            mx + 20,
            sy,
            mw - 40,
            32,
            is_on=ui_state.show_skeleton,
            label="Show 21 MediaPipe Hand Landmarks & Bones",
            is_hovered=ui_state.hovered_id == "set_toggle_skeleton",
            theme=theme,
        )
        sy += 44

        # Setting 6: Speech Rate Control
        rate = settings.get("speech_rate", 175)
        cv2.putText(
            canvas,
            f"Speech Rate: {rate} words per minute",
            (mx + 20, sy + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            theme.font_scale_body,
            theme.text_primary,
            1,
            cv2.LINE_AA,
        )
        self._register_button("set_rate_minus", mx + mw - 140, sy, 32, 26)
        draw_button(
            canvas,
            mx + mw - 140,
            sy,
            32,
            26,
            "-",
            None,
            ui_state.hovered_id == "set_rate_minus",
            ui_state.clicked_id == "set_rate_minus",
            theme,
        )
        self._register_button("set_rate_plus", mx + mw - 90, sy, 32, 26)
        draw_button(
            canvas,
            mx + mw - 90,
            sy,
            32,
            26,
            "+",
            None,
            ui_state.hovered_id == "set_rate_plus",
            ui_state.clicked_id == "set_rate_plus",
            theme,
        )
        sy += 40

        # Setting 7: Speech Volume Control
        vol_pct = int(settings.get("speech_volume", 1.0) * 100)
        cv2.putText(
            canvas,
            f"Speech Volume: {vol_pct}%",
            (mx + 20, sy + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            theme.font_scale_body,
            theme.text_primary,
            1,
            cv2.LINE_AA,
        )
        self._register_button("set_vol_minus", mx + mw - 140, sy, 32, 26)
        draw_button(
            canvas,
            mx + mw - 140,
            sy,
            32,
            26,
            "-",
            None,
            ui_state.hovered_id == "set_vol_minus",
            ui_state.clicked_id == "set_vol_minus",
            theme,
        )
        self._register_button("set_vol_plus", mx + mw - 90, sy, 32, 26)
        draw_button(
            canvas,
            mx + mw - 90,
            sy,
            32,
            26,
            "+",
            None,
            ui_state.hovered_id == "set_vol_plus",
            ui_state.clicked_id == "set_vol_plus",
            theme,
        )
        sy += 50

        # Action Buttons at bottom: Done & Reset
        self._register_button("set_btn_done", mx + mw - 160, my + mh - 44, 140, 32)
        draw_button(
            canvas,
            mx + mw - 160,
            my + mh - 44,
            140,
            32,
            "Save & Close",
            "ESC",
            ui_state.hovered_id == "set_btn_done",
            ui_state.clicked_id == "set_btn_done",
            theme,
            accent_border=True,
        )

        self._register_button("set_btn_reset", mx + 20, my + mh - 44, 160, 32)
        draw_button(
            canvas,
            mx + 20,
            my + mh - 44,
            160,
            32,
            "Reset to Defaults",
            None,
            ui_state.hovered_id == "set_btn_reset",
            ui_state.clicked_id == "set_btn_reset",
            theme,
        )

    def _render_help_modal(
        self,
        canvas: np.ndarray,
        w: int,
        h: int,
        ui_state: UIState,
        theme: UITheme,
    ) -> None:
        """Renders comprehensive user documentation with tab navigation."""
        mw, mh = 700, 520
        mx = (w - mw) // 2
        my = (h - mh) // 2

        draw_card(canvas, mx, my, mw, mh, theme.card_bg, theme.primary_accent, border_width=2)
        draw_card_header(canvas, mx, my, mw, "SignBridge User Guide & ASL Notice", theme)

        # Close button
        btn_close_x = mx + mw - 36
        btn_close_y = my + 4
        self._register_button("modal_close", btn_close_x, btn_close_y, 28, 22)
        draw_button(
            canvas,
            btn_close_x,
            btn_close_y,
            28,
            22,
            "X",
            None,
            ui_state.hovered_id == "modal_close",
            ui_state.clicked_id == "modal_close",
            theme,
        )

        # Tab navigation bar
        tabs = ["Overview", "Controls", "Hand Guide", "ASL Notice"]
        tab_w = (mw - 40) // 4
        tab_y = my + 38

        for i, t_name in enumerate(tabs):
            tx = mx + 20 + i * tab_w
            tid = f"help_tab_{i}"
            self._register_button(tid, tx, tab_y, tab_w - 6, 28)
            is_active = ui_state.help_tab == i
            draw_button(
                canvas,
                tx,
                tab_y,
                tab_w - 6,
                28,
                t_name,
                None,
                ui_state.hovered_id == tid,
                is_active,
                theme,
                accent_border=is_active,
            )

        # Content area
        cx = mx + 24
        cy = my + 84

        if ui_state.help_tab == 0:  # Overview
            lines = [
                "WHAT IS SIGNBRIDGE?",
                "SignBridge is an end-to-end real-time American Sign Language fingerspelling",
                "recognition and translation application. It bridges the communication gap",
                "between Deaf/hard-of-hearing signers and hearing individuals.",
                "",
                "CORE PIPELINE:",
                "1. OpenCV Video Capture -> Asynchronous 30 FPS camera ingestion.",
                "2. MediaPipe Hand Tracking -> Extracts 21 3D joint landmarks.",
                "3. 94D Feature Engineering -> Translation, scale, and handedness invariant.",
                "4. Static Classifier -> High-precision sign classification (A-Z).",
                "5. Temporal Stabilization -> Eliminates frame-to-frame sign jitter.",
                "6. Translation Engine -> Builds letters into words and sentences.",
                "7. Offline Voice Synthesis -> Speaks completed sentences naturally via SAPI5.",
            ]
        elif ui_state.help_tab == 1:  # Controls
            lines = [
                "KEYBOARD & MOUSE CONTROLS REFERENCE:",
                "[SPACE]       Insert Word Boundary (commits word to sentence)",
                "[S]           Manual Speak: Voicing current active sentence aloud",
                "[X]           Stop Speech: Immediately interrupts active audio playback",
                "[T]           Toggle Voice: Mute or unmute voice synthesis",
                "[BS]          Backspace: Deletes last letter or re-opens previous word",
                "[C]           Clear Sentence: Clears accumulated words and active sentence",
                "[U]           Undo: Restores previous translation state snapshot",
                "[P]           Pause / Resume: Freezes or resumes live camera processing",
                "[M]           Mirror: Toggles horizontal webcam mirroring",
                "[R]           Reset All: Clears all translation buffers and state",
                "[O]           Settings: Opens configuration dialog",
                "[H]           Help: Opens this guide",
                "[Q] / [ESC]   Quit: Cleanly shuts down camera, threads, and application",
            ]
        elif ui_state.help_tab == 2:  # Hand Guide
            lines = [
                "BEST PRACTICES FOR ACCURATE HAND RECOGNITION:",
                "",
                "1. Hand Placement:",
                "   Position your dominant hand in the center of the camera viewport.",
                "   Ensure your entire hand from wrist to fingertips is clearly visible.",
                "",
                "2. Lighting Conditions:",
                "   Use uniform front-facing lighting. Avoid bright backlights or deep shadows.",
                "",
                "3. Steady Signing (Dwell Time):",
                "   Hold the handshape steady for 0.25 seconds. Watch the Dwell Stability",
                "   progress bar on the right fill up to 100% to commit the letter token.",
                "",
                "4. Hand Background:",
                "   A clean, neutral background provides highest MediaPipe tracking stability.",
            ]
        else:  # ASL Linguistic Notice
            lines = [
                "IMPORTANT ASL LINGUISTIC & GRAMMATICAL NOTICE:",
                "",
                "Current Scope: Fingerspelled English Letters (A-Z)",
                "SignBridge Stages 0-11 focus on the American Sign Language manual alphabet.",
                "Fingerspelling is an integral component of ASL used for proper nouns, names,",
                "technical terms, and emphasis.",
                "",
                "Distinction from Natural ASL:",
                "American Sign Language is a distinct natural human language with its own",
                "unique visual-spatial grammar, syntax, facial markers, and dynamic signs.",
                "English fingerspelling is NOT equivalent to full grammatical ASL.",
                "",
                "SignBridge does NOT claim full natural ASL translation at this stage.",
                "Dynamic holistic gesture recognition and grammatical translation belong",
                "to future development stages.",
            ]

        for idx, line in enumerate(lines):
            is_hdr = idx == 0 or line.endswith(":")
            font_col = theme.primary_accent if is_hdr else theme.text_primary
            scale = 0.48 if is_hdr else 0.42
            weight = 2 if is_hdr else 1
            cv2.putText(canvas, line, (cx, cy + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, scale, font_col, weight, cv2.LINE_AA)

        # Close Button at bottom
        self._register_button("help_btn_done", mx + mw - 140, my + mh - 44, 120, 30)
        draw_button(
            canvas,
            mx + mw - 140,
            my + mh - 44,
            120,
            30,
            "Close Guide",
            "ESC",
            ui_state.hovered_id == "help_btn_done",
            ui_state.clicked_id == "help_btn_done",
            theme,
            accent_border=True,
        )

    def _render_onboarding_modal(
        self,
        canvas: np.ndarray,
        w: int,
        h: int,
        ui_state: UIState,
        theme: UITheme,
    ) -> None:
        """Renders 5-step interactive onboarding tutorial."""
        mw, mh = 640, 440
        mx = (w - mw) // 2
        my = (h - mh) // 2

        draw_card(canvas, mx, my, mw, mh, theme.card_bg, theme.primary_accent, border_width=2)
        draw_card_header(canvas, mx, my, mw, "Welcome to SignBridge!", theme)

        # Close [X] button top right
        btn_close_x = mx + mw - 36
        btn_close_y = my + 4
        self._register_button("modal_close", btn_close_x, btn_close_y, 28, 22)
        draw_button(
            canvas,
            btn_close_x,
            btn_close_y,
            28,
            22,
            "X",
            None,
            ui_state.hovered_id == "modal_close",
            ui_state.clicked_id == "modal_close",
            theme,
        )

        cv2.putText(
            canvas,
            "Real-Time Sign Language Translation in 5 Steps",
            (mx + 24, my + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            theme.text_secondary,
            1,
            cv2.LINE_AA,
        )

        steps = [
            ("1. Position Hand", "Place your hand in front of the camera with fingers spread clearly."),
            ("2. Hold a Sign", "Form an ASL alphabet letter and hold it steady for 0.25 seconds."),
            ("3. Watch Dwell", "The Dwell Stability meter locks in when the letter is recognized."),
            ("4. Build Words", "Letters form words. Press [SPACE] or click 'Word Break' when done."),
            ("5. Hear Translation", "Click 'Speak' or press [S] to hear your sentence spoken aloud!"),
        ]

        sy = my + 95
        for i, (title, desc) in enumerate(steps):
            cv2.circle(canvas, (mx + 36, sy + 10), 10, theme.primary_accent, -1, cv2.LINE_AA)
            cv2.putText(canvas, str(i + 1), (mx + 32, sy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, theme.text_dark, 1, cv2.LINE_AA)
            cv2.putText(canvas, title, (mx + 56, sy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, theme.primary_accent, 1, cv2.LINE_AA)
            cv2.putText(canvas, desc, (mx + 56, sy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, theme.text_secondary, 1, cv2.LINE_AA)
            sy += 48

        # Buttons: Start Translating & Skip
        self._register_button("onboarding_done", mx + mw - 180, my + mh - 44, 160, 32)
        draw_button(
            canvas,
            mx + mw - 180,
            my + mh - 44,
            160,
            32,
            "Start Translating",
            "ENTER",
            ui_state.hovered_id == "onboarding_done",
            ui_state.clicked_id == "onboarding_done",
            theme,
            accent_border=True,
        )

        self._register_button("onboarding_skip", mx + 24, my + mh - 44, 120, 32)
        draw_button(
            canvas,
            mx + 24,
            my + mh - 44,
            120,
            32,
            "Skip Tutorial",
            "ESC",
            ui_state.hovered_id == "onboarding_skip",
            ui_state.clicked_id == "onboarding_skip",
            theme,
        )

    def _render_emergency_modal(
        self,
        canvas: np.ndarray,
        w: int,
        h: int,
        ui_state: UIState,
        theme: UITheme,
    ) -> None:
        """Renders the safe simulated emergency assistance modal dialog."""
        mw, mh = 620, 340
        mx = (w - mw) // 2
        my = (h - mh) // 2

        emergency_red = (40, 45, 230)
        draw_card(canvas, mx, my, mw, mh, theme.card_bg, emergency_red, border_width=3)
        draw_card_header(canvas, mx, my, mw, "[!] EMERGENCY ASSISTANCE CONFIRMATION", theme, accent_color=emergency_red)

        # Close [X]
        btn_close_x = mx + mw - 36
        btn_close_y = my + 4
        self._register_button("modal_close", btn_close_x, btn_close_y, 28, 22)
        draw_button(canvas, btn_close_x, btn_close_y, 28, 22, "X", "", ui_state.hovered_id == "modal_close", False, theme)

        # Warning icon and text
        cv2.circle(canvas, (mx + 50, my + 95), 24, emergency_red, -1)
        cv2.putText(canvas, "!", (mx + 45, my + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(
            canvas,
            "ASL Emergency Sign 'HELP' Detected",
            (mx + 90, my + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            theme.text_primary,
            2,
            cv2.LINE_AA,
        )

        lines = [
            "An emergency sign has been recognized with high confidence.",
            "ACADEMIC DEMONSTRATION NOTICE:",
            "- This system operates strictly in simulated evaluation mode.",
            "- NO actual calls to 911 or external emergency networks are made.",
            "Select an action below to test simulated dispatch handling:",
        ]
        sy = my + 130
        for line in lines:
            col = theme.warning_accent if "ACADEMIC" in line else theme.text_secondary
            cv2.putText(canvas, line, (mx + 30, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
            sy += 24

        # Action Buttons
        self._register_button("btn_emergency_confirm", mx + 50, my + mh - 55, 220, 36)
        draw_button(
            canvas,
            mx + 50,
            my + mh - 55,
            220,
            36,
            "GET HELP (Simulated)",
            "Y",
            ui_state.hovered_id == "btn_emergency_confirm",
            ui_state.clicked_id == "btn_emergency_confirm",
            theme,
            accent_border=True,
        )

        self._register_button("btn_emergency_cancel", mx + 310, my + mh - 55, 180, 36)
        draw_button(
            canvas,
            mx + 310,
            my + mh - 55,
            180,
            36,
            "CANCEL (Dismiss)",
            "N",
            ui_state.hovered_id == "btn_emergency_cancel",
            ui_state.clicked_id == "btn_emergency_cancel",
            theme,
        )

    def _render_video_demo_modal(
        self,
        canvas: np.ndarray,
        w: int,
        h: int,
        ui_state: UIState,
        theme: UITheme,
    ) -> None:
        """Renders the video demonstration sample picker and video controls modal."""
        mw, mh = 700, 460
        mx = (w - mw) // 2
        my = (h - mh) // 2

        draw_card(canvas, mx, my, mw, mh, theme.card_bg, theme.primary_accent, border_width=2)
        draw_card_header(canvas, mx, my, mw, "ASL Word Sign Demonstrations & Video Input", theme)

        # Close [X]
        btn_close_x = mx + mw - 36
        btn_close_y = my + 4
        self._register_button("modal_close", btn_close_x, btn_close_y, 28, 22)
        draw_button(canvas, btn_close_x, btn_close_y, 28, 22, "X", "", ui_state.hovered_id == "modal_close", False, theme)

        cv2.putText(
            canvas,
            "Select a curated ASL demonstration sample to evaluate sequence recognition:",
            (mx + 25, my + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            theme.text_secondary,
            1,
            cv2.LINE_AA,
        )

        sample_buttons = [
            ("btn_sample_hello", "HELLO", "Temple wave salute (1-hand)"),
            ("btn_sample_help", "HELP", "Thumbs-up lifted on flat palm (2-hand)"),
            ("btn_sample_yes", "YES", "S-fist vertical nodding (1-hand)"),
            ("btn_sample_no", "NO", "Index & middle snap shut (1-hand)"),
            ("btn_sample_stop", "STOP", "Open palm chop onto base (2-hand)"),
            ("btn_sample_thank_you", "THANK YOU", "Chin touch extending forward (1-hand)"),
        ]

        sy = my + 95
        for bid, sign_name, desc in sample_buttons:
            # Button on left
            self._register_button(bid, mx + 25, sy, 140, 32)
            draw_button(
                canvas,
                mx + 25,
                sy,
                140,
                32,
                f"Play: {sign_name}",
                "",
                ui_state.hovered_id == bid,
                ui_state.clicked_id == bid,
                theme,
                accent_border=("HELP" in sign_name),
            )

            # Description on right
            cv2.putText(canvas, desc, (mx + 180, sy + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, theme.text_primary, 1, cv2.LINE_AA)
            sy += 42

        # Return to Live Camera Button
        self._register_button("btn_source_camera", mx + 25, my + mh - 50, 220, 34)
        draw_button(
            canvas,
            mx + 25,
            my + mh - 50,
            220,
            34,
            "Switch to Live Webcam",
            "M",
            ui_state.hovered_id == "btn_source_camera",
            ui_state.clicked_id == "btn_source_camera",
            theme,
        )

        # Done button
        self._register_button("modal_close_btn", mx + mw - 140, my + mh - 50, 110, 34)
        draw_button(
            canvas,
            mx + mw - 140,
            my + mh - 50,
            110,
            34,
            "Close Hub",
            "ESC",
            ui_state.hovered_id == "modal_close_btn",
            ui_state.clicked_id == "modal_close_btn",
            theme,
            accent_border=True,
        )

    def _register_button(self, btn_id: str, x: int, y: int, w: int, h: int) -> None:
        self._buttons[btn_id] = (x, y, w, h)
