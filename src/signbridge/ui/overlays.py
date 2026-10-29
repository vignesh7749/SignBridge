"""Stage 11: Interactive Modal Dialog Overlays.

Renders SettingsModal, HelpModal, and OnboardingModal with clickable controls
and accessibility configurations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from signbridge.ui.components import (
    draw_badge,
    draw_button,
    draw_card,
    draw_card_header,
    draw_rounded_rect,
    draw_switch,
    draw_text,
    draw_vector_icon,
    hit_test,
    scale_and_letterbox_frame,
)
from signbridge.ui.state import ActiveModal, UIState
from signbridge.ui.theme import UITheme

# ---------------------------------------------------------------------------
# Visual Demonstration Thumbnail Cache & Resolver
# ---------------------------------------------------------------------------

_THUMBNAIL_CACHE: Dict[str, Optional[np.ndarray]] = {}
_SHARED_CATALOG: Any = None
_SHARED_SAMPLE_MGR: Any = None


def get_sign_thumbnail(entry: Any, target_w: int = 78, target_h: int = 54) -> Optional[np.ndarray]:
    """Retrieves or generates a cached preview thumbnail from a sign entry or video."""
    global _SHARED_CATALOG, _SHARED_SAMPLE_MGR
    if not entry:
        return None

    resolved_id = None
    vpath = None

    if isinstance(entry, str):
        clean_str = entry.strip()
        cache_key = f"str_{clean_str.upper()}_{target_w}_{target_h}"
        if cache_key in _THUMBNAIL_CACHE:
            return _THUMBNAIL_CACHE[cache_key]
        if _SHARED_CATALOG is None:
            from signbridge.video.catalog import SignLanguageCatalog
            _SHARED_CATALOG = SignLanguageCatalog()
        cat_entry = _SHARED_CATALOG.get_sign_by_name(clean_str) or _SHARED_CATALOG.get_sign_by_id(clean_str)
        if cat_entry and cat_entry.available:
            resolved_id = cat_entry.id
            vpath = cat_entry.get_full_video_path()
        else:
            if _SHARED_SAMPLE_MGR is None:
                from signbridge.video.sample_manager import SampleVideoManager
                _SHARED_SAMPLE_MGR = SampleVideoManager()
            rec = _SHARED_SAMPLE_MGR.get_sample_by_sign(clean_str) or _SHARED_SAMPLE_MGR.get_sample_by_id(clean_str)
            if rec and rec.full_path:
                resolved_id = rec.filename
                vpath = Path(rec.full_path)
    elif hasattr(entry, "get_full_video_path"):
        if not getattr(entry, "available", False):
            return None
        resolved_id = getattr(entry, "id", str(entry))
        vpath = entry.get_full_video_path()
    elif hasattr(entry, "full_path") and getattr(entry, "full_path", None):
        resolved_id = getattr(entry, "sign", getattr(entry, "file", str(entry)))
        vpath = Path(entry.full_path)
    else:
        return None

    if not vpath or not Path(vpath).exists():
        if isinstance(entry, str):
            _THUMBNAIL_CACHE[f"str_{entry.strip().upper()}_{target_w}_{target_h}"] = None
        return None

    cache_key = f"{resolved_id}_{target_w}_{target_h}"
    if cache_key in _THUMBNAIL_CACHE:
        thumb_val = _THUMBNAIL_CACHE[cache_key]
        if isinstance(entry, str):
            _THUMBNAIL_CACHE[f"str_{entry.strip().upper()}_{target_w}_{target_h}"] = thumb_val
        return thumb_val

    try:
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            _THUMBNAIL_CACHE[cache_key] = None
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = max(1, min(15, total_frames // 3))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None and frame.size > 0:
            thumb = scale_and_letterbox_frame(frame, target_w, target_h, bg_color=(12, 10, 9))
            _THUMBNAIL_CACHE[cache_key] = thumb
            if isinstance(entry, str):
                _THUMBNAIL_CACHE[f"str_{entry.strip().upper()}_{target_w}_{target_h}"] = thumb
            return thumb
    except Exception:
        pass
    _THUMBNAIL_CACHE[cache_key] = None
    if isinstance(entry, str):
        _THUMBNAIL_CACHE[f"str_{entry.strip().upper()}_{target_w}_{target_h}"] = None
    return None


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
            lines = []
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

        if ui_state.help_tab == 1:
            from signbridge.ui.shortcuts import SHORTCUT_REGISTRY
            grouped = SHORTCUT_REGISTRY.get_help_grouped()
            col1_groups = ["Sentence & Word Editing", "Voice & Audio Output", "Recognition Modes"]
            col2_groups = ["Viewport & Video Stream", "System & Dialogs"]

            def _render_help_col(start_x: int, col_groups: list[str]) -> None:
                cur_y = cy
                for grp in col_groups:
                    items = grouped.get(grp, [])
                    if not items:
                        continue
                    cv2.putText(canvas, grp.upper(), (start_x, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, theme.primary_accent, 2, cv2.LINE_AA)
                    cur_y += 18
                    for badge, desc in items:
                        # Shorten description if needed to fit column
                        short_desc = desc.split(":")[0] + ":" + desc.split(":")[1] if ":" in desc else desc
                        if len(short_desc) > 42:
                            short_desc = short_desc[:39] + "..."
                        cv2.putText(canvas, f"{badge:<7} {short_desc}", (start_x, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, theme.text_primary, 1, cv2.LINE_AA)
                        cur_y += 16
                    cur_y += 8

            _render_help_col(cx, col1_groups)
            _render_help_col(cx + 340, col2_groups)
        else:
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
        """Renders the comprehensive ASL Reference Library and Demo Hub modal."""
        mw, mh = min(1000, w - 40), min(660, h - 30)
        mx = (w - mw) // 2
        my = (h - mh) // 2

        draw_card(canvas, mx, my, mw, mh, theme.card_bg, theme.primary_accent, border_width=2)
        draw_card_header(canvas, mx, my, mw, "ASL Demonstration Hub & Reference Library", theme)

        # Close [X] top right
        btn_close_x = mx + mw - 36
        btn_close_y = my + 4
        self._register_button("modal_close", btn_close_x, btn_close_y, 28, 22)
        draw_button(canvas, btn_close_x, btn_close_y, 28, 22, "X", "", ui_state.hovered_id == "modal_close", False, theme)

        # Curated demonstration shortcuts (backward compatibility & quick access)
        for s_name in ("hello", "help", "yes", "no", "stop", "thank_you"):
            self._register_button(f"btn_sample_{s_name}", mx + 20, my + 38, 1, 1)

        # Lazy load catalog
        if not hasattr(self, "_catalog") or self._catalog is None:
            from signbridge.video.catalog import SignLanguageCatalog
            self._catalog = SignLanguageCatalog()

        # -------------------------------------------------------------
        # Category Tabs Row
        # -------------------------------------------------------------
        cat_tabs = [
            ("btn_cat_all", "All Signs", "ALL SIGNS"),
            ("btn_cat_alpha", "Alphabet", "ALPHABET"),
            ("btn_cat_nums", "Numbers", "NUMBERS"),
            ("btn_cat_greetings", "Greetings", "GREETINGS"),
            ("btn_cat_common", "Common Words", "COMMON WORDS"),
            ("btn_cat_daily", "Daily Communication", "DAILY COMM"),
            ("btn_cat_phrases", "Common Phrases", "PHRASES"),
        ]

        tab_y = my + 38
        tab_h = 28
        total_tabs = len(cat_tabs)
        tab_margin = 6
        avail_tw = mw - 40 - (tab_margin * (total_tabs - 1))
        tab_w = max(80, avail_tw // total_tabs)

        for i, (bid, cat_id, label) in enumerate(cat_tabs):
            tx = mx + 20 + i * (tab_w + tab_margin)
            is_active = (ui_state.demo_category.lower() == cat_id.lower())
            is_hover = (ui_state.hovered_id == bid)
            self._register_button(bid, tx, tab_y, tab_w, tab_h)
            draw_button(
                canvas,
                tx,
                tab_y,
                tab_w,
                tab_h,
                label,
                "",
                is_hover,
                is_active,
                theme,
                accent_border=is_active,
            )

        # -------------------------------------------------------------
        # Status / Filter Row
        # -------------------------------------------------------------
        filter_y = tab_y + tab_h + 10
        matching_signs = self._catalog.list_signs(
            category=ui_state.demo_category,
            query=ui_state.demo_search_query,
        )
        total_match = len(matching_signs)
        avail_match = sum(1 for s in matching_signs if s.available)

        status_text = (
            f"Category: {ui_state.demo_category.upper()}  |  "
            f"Showing {total_match} reference entries ({avail_match} with playable video demonstrations)"
            if avail_match > 0 else
            f"Category: {ui_state.demo_category.upper()}  |  "
            f"All {total_match} reference guides available (Video demonstrations pending curation)"
        )
        draw_text(
            canvas,
            status_text,
            (mx + 22, filter_y + 4),
            font_size=12,
            color=theme.text_secondary if avail_match > 0 else theme.warning_accent,
            bold=True,
        )

        # Clear Filter button if query or non-default category
        if ui_state.demo_search_query or ui_state.demo_category != "All Signs":
            cf_w = 95
            cf_x = mx + mw - cf_w - 22
            self._register_button("btn_search_clear", cf_x, filter_y, cf_w, 24)
            draw_button(
                canvas,
                cf_x,
                filter_y,
                cf_w,
                24,
                "Reset Filter",
                "",
                ui_state.hovered_id == "btn_search_clear",
                False,
                theme,
            )

        # -------------------------------------------------------------
        # Content Area: Detail vs. List View
        # -------------------------------------------------------------
        content_y = filter_y + 32
        content_h = mh - (content_y - my) - 54

        if ui_state.demo_view_mode == "DETAIL" and ui_state.demo_selected_sign_id:
            entry = self._catalog.get_sign_by_id(ui_state.demo_selected_sign_id)
            if entry:
                # Rounded card surface
                draw_rounded_rect(
                    canvas,
                    mx + 20,
                    content_y,
                    mw - 40,
                    content_h,
                    radius=8,
                    bg_color=(22, 18, 16),
                    border_color=theme.card_border,
                    border_width=1,
                )

                # Left Pane: Classroom Video Preview / Placeholder Box (Aspect Ratio Preserved)
                pw, ph = 320, 220
                px, py = mx + 36, content_y + 24
                draw_rounded_rect(
                    canvas,
                    px - 2,
                    py - 2,
                    pw + 4,
                    ph + 4,
                    radius=6,
                    bg_color=(14, 11, 10),
                    border_color=theme.card_border,
                    border_width=1,
                )

                if entry.available:
                    preview_frame = get_sign_thumbnail(entry, pw, ph)
                    if preview_frame is not None:
                        canvas[py : py + ph, px : px + pw] = preview_frame
                        # Translucent dark circular backing and centered play vector icon
                        icx, icy = px + pw // 2, py + ph // 2
                        cv2.circle(canvas, (icx, icy), 26, (18, 14, 12), -1, cv2.LINE_AA)
                        cv2.circle(canvas, (icx, icy), 26, theme.card_border, 1, cv2.LINE_AA)
                        draw_vector_icon(canvas, "play", icx, icy, size=24, color=theme.primary_accent)
                    draw_badge(
                        canvas,
                        px + 8,
                        py + ph - 28,
                        "● VERIFIED ASL VIDEO DEMO",
                        theme.text_dark,
                        theme.secondary_accent,
                        font_scale=0.34,
                    )
                else:
                    draw_rounded_rect(canvas, px, py, pw, ph, radius=4, bg_color=(25, 20, 18))
                    draw_vector_icon(canvas, "search", px + pw // 2, py + ph // 2 - 20, size=28, color=theme.warning_accent)
                    draw_text(
                        canvas,
                        "AUTHENTIC VIDEO",
                        (px + pw // 2, py + ph // 2 + 15),
                        font_size=13,
                        color=theme.warning_accent,
                        bold=True,
                        anchor="cc",
                    )
                    draw_text(
                        canvas,
                        "PENDING CURATION",
                        (px + pw // 2, py + ph // 2 + 35),
                        font_size=11,
                        color=theme.text_muted,
                        anchor="cc",
                    )
                    draw_badge(
                        canvas,
                        px + 8,
                        py + ph - 28,
                        "○ REFERENCE GUIDE",
                        (210, 210, 255),
                        (45, 30, 30),
                        font_scale=0.32,
                    )

                # Right Pane: Educational Specifications & Mechanics
                rx = px + pw + 24

                # Header: Name & Language
                draw_text(canvas, entry.name.upper(), (rx, py), font_size=20, color=theme.text_primary, bold=True)
                by = py + 30
                bx = rx
                bx += draw_badge(canvas, bx, by, f"CATEGORY: {entry.category.upper()}", theme.text_dark, theme.primary_accent, font_scale=0.34) + 6
                bx += draw_badge(canvas, bx, by, f"TYPE: {entry.type.upper()}", theme.text_primary, (45, 40, 36), font_scale=0.34) + 6
                draw_badge(canvas, bx, by, f"LANG: {entry.language}", theme.text_dark, theme.secondary_accent, font_scale=0.34)

                sy = by + 28

                # Section 1: Mechanics
                draw_text(canvas, "GESTURE MECHANICS & HANDSHAPE:", (rx, sy), font_size=12, color=theme.primary_accent, bold=True)
                sy += 20
                words = entry.description.split()
                line = ""
                for w_item in words:
                    if len(line) + len(w_item) + 1 > 54:
                        draw_text(canvas, line, (rx, sy), font_size=11, color=theme.text_primary)
                        sy += 18
                        line = w_item
                    else:
                        line = f"{line} {w_item}".strip()
                if line:
                    draw_text(canvas, line, (rx, sy), font_size=11, color=theme.text_primary)
                    sy += 22

                # Section 2: Educational Notes
                draw_text(canvas, "EDUCATIONAL NOTES & LINGUISTIC CONTEXT:", (rx, sy), font_size=12, color=theme.primary_accent, bold=True)
                sy += 20
                notes_text = entry.notes or "Hold hand position clearly in the webcam frame to trigger stable detection."
                words_n = notes_text.split()
                line_n = ""
                for wn in words_n:
                    if len(line_n) + len(wn) + 1 > 54:
                        draw_text(canvas, line_n, (rx, sy), font_size=11, color=theme.text_secondary)
                        sy += 18
                        line_n = wn
                    else:
                        line_n = f"{line_n} {wn}".strip()
                if line_n:
                    draw_text(canvas, line_n, (rx, sy), font_size=11, color=theme.text_secondary)
                    sy += 20

                # Section 3: Attribution
                attr_text = f"Attribution: {entry.source}  |  License: {entry.license}  |  Language: {entry.language}"
                draw_text(canvas, attr_text[:68], (rx, sy), font_size=10, color=theme.text_muted)

                # Detail Action Buttons at bottom of card
                dbtn_y = content_y + content_h - 48
                bx_btn = mx + 40
                if entry.available:
                    play_bid = f"btn_play_sign_{entry.id}"
                    self._register_button(play_bid, bx_btn, dbtn_y, 180, 34)
                    draw_button(
                        canvas,
                        bx_btn,
                        dbtn_y,
                        180,
                        34,
                        "Play Video Demo",
                        "ENTER",
                        ui_state.hovered_id == play_bid,
                        False,
                        theme,
                        accent_border=True,
                    )
                    bx_btn += 190

                self._register_button("btn_demo_prev_sign", bx_btn, dbtn_y, 130, 34)
                draw_button(
                    canvas,
                    bx_btn,
                    dbtn_y,
                    130,
                    34,
                    "< Prev Sign",
                    "",
                    ui_state.hovered_id == "btn_demo_prev_sign",
                    False,
                    theme,
                )
                bx_btn += 140

                self._register_button("btn_demo_next_sign", bx_btn, dbtn_y, 130, 34)
                draw_button(
                    canvas,
                    bx_btn,
                    dbtn_y,
                    130,
                    34,
                    "Next Sign >",
                    "",
                    ui_state.hovered_id == "btn_demo_next_sign",
                    False,
                    theme,
                )

                self._register_button("btn_demo_back_list", mx + mw - 190, dbtn_y, 150, 34)
                draw_button(
                    canvas,
                    mx + mw - 190,
                    dbtn_y,
                    150,
                    34,
                    "Back to Signs",
                    "ESC",
                    ui_state.hovered_id == "btn_demo_back_list",
                    False,
                    theme,
                )
        else:
            # ---------------------------------------------------------
            # List View: Paginated Sign Cards
            # ---------------------------------------------------------
            page_items, total_pages, cur_page = self._catalog.paginate(
                matching_signs,
                page=ui_state.demo_page,
                page_size=5,
            )
            ui_state.demo_page = cur_page

            if not page_items:
                # Empty state
                draw_rounded_rect(
                    canvas,
                    mx + 20,
                    content_y,
                    mw - 40,
                    content_h,
                    radius=8,
                    bg_color=(22, 18, 16),
                    border_color=theme.card_border,
                    border_width=1,
                )
                draw_text(
                    canvas,
                    "No sign references found matching your current filter.",
                    (mx + mw // 2, content_y + content_h // 2 - 10),
                    font_size=14,
                    color=theme.text_secondary,
                    anchor="cc",
                )
                self._register_button("btn_cat_all", mx + mw // 2 - 80, content_y + content_h // 2 + 15, 160, 32)
                draw_button(
                    canvas,
                    mx + mw // 2 - 80,
                    content_y + content_h // 2 + 15,
                    160,
                    32,
                    "View All Signs",
                    "",
                    ui_state.hovered_id == "btn_cat_all",
                    False,
                    theme,
                    accent_border=True,
                )
            else:
                card_h = 74
                card_gap = 8
                card_w = mw - 40

                for i, sign in enumerate(page_items):
                    cy = content_y + i * (card_h + card_gap)
                    cx = mx + 20

                    # Card surface
                    is_card_hover = ui_state.hovered_id in (
                        f"btn_play_sign_{sign.id}",
                        f"btn_detail_sign_{sign.id}",
                        f"btn_next_card_{sign.id}",
                    )
                    card_border = theme.primary_accent if is_card_hover else theme.card_border
                    draw_rounded_rect(
                        canvas,
                        cx,
                        cy,
                        card_w,
                        card_h,
                        radius=6,
                        bg_color=(24, 20, 18),
                        border_color=card_border,
                        border_width=1,
                    )

                    # ---------------------------------------------------------
                    # 4-Zone Strict Non-Overlapping Layout Architecture
                    # ---------------------------------------------------------
                    # Zone 1: Thumbnail Box [cx + 10 .. cx + 86]
                    tw, th = 76, 52
                    tx, ty = cx + 10, cy + 11
                    if sign.available:
                        thumb = get_sign_thumbnail(sign, tw, th)
                        if thumb is not None:
                            canvas[ty : ty + th, tx : tx + tw] = thumb
                            cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (60, 50, 45), 1, cv2.LINE_AA)
                            # Circular dark backing behind centered play vector icon
                            icx, icy = tx + tw // 2, ty + th // 2
                            cv2.circle(canvas, (icx, icy), 12, (18, 14, 12), -1, cv2.LINE_AA)
                            cv2.circle(canvas, (icx, icy), 12, (80, 70, 65), 1, cv2.LINE_AA)
                            draw_vector_icon(canvas, "play", icx, icy, size=12, color=(240, 240, 240))
                        else:
                            draw_rounded_rect(canvas, tx, ty, tw, th, radius=4, bg_color=(32, 26, 24), border_color=(60, 50, 45), border_width=1)
                            draw_text(canvas, "VIDEO", (tx + tw // 2, ty + th // 2), font_size=10, color=theme.secondary_accent, bold=True, anchor="cc")
                    else:
                        draw_rounded_rect(canvas, tx, ty, tw, th, radius=4, bg_color=(32, 26, 24), border_color=(75, 60, 55), border_width=1)
                        draw_text(canvas, "REF GUIDE", (tx + tw // 2, ty + 20), font_size=10, color=theme.warning_accent, bold=True, anchor="cc")
                        draw_text(canvas, "NO VIDEO", (tx + tw // 2, ty + 36), font_size=9, color=theme.text_muted, anchor="cc")

                    # Zone 2: Identity & Badges [cx + 98 .. cx + 270] (Fixed 172px width)
                    nx = tx + tw + 12
                    zone2_w = 160
                    # Row 1: Sign Title
                    draw_text(canvas, sign.name[:18], (nx, cy + 10), font_size=13, color=theme.text_primary, bold=True)

                    # Row 2: Type and Language Badges
                    bx = nx
                    by = cy + 30
                    bx += draw_badge(canvas, bx, by, sign.type.upper()[:8], theme.text_secondary, (40, 36, 34), font_scale=0.30) + 4
                    draw_badge(canvas, bx, by, sign.language[:6], theme.text_dark, theme.primary_accent, font_scale=0.30)

                    # Row 3: Availability Indicator
                    if sign.available:
                        draw_text(canvas, "● VIDEO DEMO", (nx, cy + 52), font_size=10, color=theme.secondary_accent, bold=True)
                    else:
                        draw_text(canvas, "○ REF GUIDE ONLY", (nx, cy + 52), font_size=10, color=theme.warning_accent, bold=True)

                    # Zone 3: Educational Mechanics & Notes [cx + 276 .. btn_start_x - 12]
                    btn_start_x = cx + card_w - 180
                    zone3_x = nx + zone2_w + 14
                    avail_text_w = max(40, btn_start_x - zone3_x - 12)
                    max_chars = max(15, avail_text_w // 7)

                    desc_text = f"Mechanics: {sign.description}"
                    if len(desc_text) > max_chars:
                        desc_text = desc_text[:max_chars - 3] + "..."
                    draw_text(canvas, desc_text, (zone3_x, cy + 14), font_size=11, color=theme.text_primary)

                    notes_text = f"Guide: {sign.notes or sign.category}"
                    if len(notes_text) > max_chars:
                        notes_text = notes_text[:max_chars - 3] + "..."
                    col_n = theme.text_secondary if sign.available else theme.warning_accent
                    draw_text(canvas, notes_text, (zone3_x, cy + 36), font_size=10, color=col_n)

                    # Zone 4: Action Buttons [btn_start_x .. cx + card_w - 12]
                    btn_play_id = f"btn_play_sign_{sign.id}"
                    btn_detail_id = f"btn_detail_sign_{sign.id}"
                    btn_next_id = f"btn_next_card_{sign.id}"

                    if sign.available:
                        self._register_button(btn_play_id, btn_start_x, cy + 20, 95, 34)
                        draw_button(
                            canvas,
                            btn_start_x,
                            cy + 20,
                            95,
                            34,
                            "Play Demo",
                            "",
                            ui_state.hovered_id == btn_play_id,
                            False,
                            theme,
                            accent_border=True,
                            icon="play",
                        )
                        self._register_button(btn_detail_id, btn_start_x + 100, cy + 20, 68, 34)
                        draw_button(
                            canvas,
                            btn_start_x + 100,
                            cy + 20,
                            68,
                            34,
                            "Details",
                            "",
                            ui_state.hovered_id == btn_detail_id,
                            False,
                            theme,
                        )
                    else:
                        # Register hidden play button for backwards compatibility
                        self._register_button(btn_play_id, btn_start_x, cy + 20, 1, 1)

                        self._register_button(btn_next_id, btn_start_x, cy + 20, 95, 34)
                        draw_button(
                            canvas,
                            btn_start_x,
                            cy + 20,
                            95,
                            34,
                            "Next Sign",
                            "",
                            ui_state.hovered_id == btn_next_id,
                            False,
                            theme,
                            accent_border=False,
                            icon="next",
                        )
                        self._register_button(btn_detail_id, btn_start_x + 100, cy + 20, 68, 34)
                        draw_button(
                            canvas,
                            btn_start_x + 100,
                            cy + 20,
                            68,
                            34,
                            "Guide",
                            "",
                            ui_state.hovered_id == btn_detail_id,
                            False,
                            theme,
                        )

        # -------------------------------------------------------------
        # Bottom Modal Control & Pagination Bar
        # -------------------------------------------------------------
        bottom_y = my + mh - 46

        # Left: Return to Live Camera
        self._register_button("btn_source_camera", mx + 20, bottom_y, 200, 32)
        draw_button(
            canvas,
            mx + 20,
            bottom_y,
            200,
            32,
            "Switch to Live Webcam",
            "M",
            ui_state.hovered_id == "btn_source_camera",
            ui_state.clicked_id == "btn_source_camera",
            theme,
        )

        # Center: Pagination (only in LIST view)
        if ui_state.demo_view_mode != "DETAIL":
            pg_w = 60
            pg_cx = mx + mw // 2
            btn_prev_x = pg_cx - 130
            btn_next_x = pg_cx + 70

            self._register_button("btn_page_prev", btn_prev_x, bottom_y, pg_w, 32)
            draw_button(
                canvas,
                btn_prev_x,
                bottom_y,
                pg_w,
                32,
                "< Prev",
                "",
                ui_state.hovered_id == "btn_page_prev",
                False,
                theme,
                enabled=(ui_state.demo_page > 1),
            )

            cur_p = getattr(ui_state, "demo_page", 1)
            tot_p = max(1, (len(matching_signs) + 4) // 5)
            page_str = f"Page {cur_p} of {tot_p} ({len(matching_signs)} signs)"
            draw_text(
                canvas,
                page_str,
                (pg_cx, bottom_y + 16),
                font_size=12,
                color=theme.text_secondary,
                anchor="cc",
            )

            self._register_button("btn_page_next", btn_next_x, bottom_y, pg_w, 32)
            draw_button(
                canvas,
                btn_next_x,
                bottom_y,
                pg_w,
                32,
                "Next >",
                "",
                ui_state.hovered_id == "btn_page_next",
                False,
                theme,
                enabled=(ui_state.demo_page < tot_p),
            )

        # Right: Close Modal
        self._register_button("modal_close_btn", mx + mw - 140, bottom_y, 120, 32)
        draw_button(
            canvas,
            mx + mw - 140,
            bottom_y,
            120,
            32,
            "Close Hub",
            "ESC",
            ui_state.hovered_id == "modal_close_btn",
            ui_state.clicked_id == "modal_close_btn",
            theme,
            accent_border=True,
        )

    def _register_button(self, btn_id: str, x: int, y: int, w: int, h: int) -> None:
        self._buttons[btn_id] = (x, y, w, h)
