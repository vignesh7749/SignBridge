"""Stage 11: Reusable UI Component Primitives.

High-performance, vectorized OpenCV drawing primitives for cards, buttons,
progress meters, badges, switches, and hit-testing detection.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from signbridge.ui.state import AppStatus
from signbridge.ui.theme import ColorBGR, UITheme


def hit_test(mx: int, my: int, bx: int, by: int, bw: int, bh: int) -> bool:
    """Returns True if point (mx, my) lies within the bounding box [bx, by, bx+bw, by+bh]."""
    return bx <= mx <= (bx + bw) and by <= my <= (by + bh)


def draw_card(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    bg_color: ColorBGR,
    border_color: ColorBGR,
    border_width: int = 1,
) -> None:
    """Renders a flat shaded card with crisp borders."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg_color, -1)
    if border_width > 0:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, border_width, cv2.LINE_AA)


def draw_card_header(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    title: str,
    theme: UITheme,
    icon: str = "●",
    accent_color: Optional[ColorBGR] = None,
) -> None:
    """Renders standard panel card header with title and category dot."""
    acc = accent_color or theme.primary_accent
    # Header strip background
    cv2.rectangle(canvas, (x, y), (x + w, y + 28), theme.header_bg, -1)
    cv2.line(canvas, (x, y + 28), (x + w, y + 28), theme.card_border, 1, cv2.LINE_AA)

    # Bullet dot
    cv2.circle(canvas, (x + 14, y + 14), 4, acc, -1, cv2.LINE_AA)

    # Title text
    cv2.putText(
        canvas,
        title.upper(),
        (x + 26, y + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        theme.font_scale_small,
        theme.text_secondary,
        1,
        cv2.LINE_AA,
    )


def draw_button(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    hotkey: Optional[str],
    is_hovered: bool,
    is_active: bool,
    theme: UITheme,
    accent_border: bool = False,
    enabled: bool = True,
) -> None:
    """Renders an interactive button with optional keyboard shortcut pill."""
    if not enabled:
        bg = theme.card_bg
        border = theme.card_border
        text_col = theme.text_muted
    elif is_active:
        bg = theme.btn_active_bg
        border = theme.primary_accent
        text_col = theme.text_primary
    elif is_hovered:
        bg = theme.btn_hover_bg
        border = theme.card_border_focus if accent_border else theme.btn_border
        text_col = theme.text_primary
    else:
        bg = theme.btn_normal_bg
        border = theme.primary_accent if accent_border else theme.btn_border
        text_col = theme.text_primary

    # Button surface
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1, cv2.LINE_AA)

    # Label text
    text_y = y + h // 2 + 5
    cv2.putText(
        canvas,
        label,
        (x + 12, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        theme.font_scale_body,
        text_col,
        1,
        cv2.LINE_AA,
    )

    # Hotkey indicator chip on right side
    if hotkey:
        hk_w = len(hotkey) * 8 + 8
        hk_x = x + w - hk_w - 8
        hk_y = y + (h - 18) // 2
        cv2.rectangle(canvas, (hk_x, hk_y), (hk_x + hk_w, hk_y + 18), (15, 12, 10), -1)
        cv2.rectangle(canvas, (hk_x, hk_y), (hk_x + hk_w, hk_y + 18), border, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            hotkey,
            (hk_x + 4, hk_y + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            theme.font_scale_small - 0.05,
            theme.primary_accent if enabled else theme.text_muted,
            1,
            cv2.LINE_AA,
        )


def draw_progress_bar(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    progress: float,
    fill_color: ColorBGR,
    bg_color: ColorBGR,
    border_color: ColorBGR,
) -> None:
    """Renders a smooth horizontal progress bar."""
    clamped = max(0.0, min(1.0, float(progress)))
    # Track background
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg_color, -1)
    # Fill
    fill_w = int(w * clamped)
    if fill_w > 0:
        cv2.rectangle(canvas, (x, y), (x + fill_w, y + h), fill_color, -1)
    # Border
    cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 1, cv2.LINE_AA)


def draw_badge(
    canvas: np.ndarray,
    x: int,
    y: int,
    text: str,
    fg_color: ColorBGR,
    bg_color: ColorBGR,
    border_color: Optional[ColorBGR] = None,
    pad_x: int = 8,
    pad_y: int = 4,
    font_scale: float = 0.40,
) -> int:
    """Renders a pill badge and returns total rendered width."""
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    bw = text_size[0] + pad_x * 2
    bh = text_size[1] + pad_y * 2

    cv2.rectangle(canvas, (x, y), (x + bw, y + bh), bg_color, -1)
    if border_color:
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), border_color, 1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        text,
        (x + pad_x, y + bh - pad_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        fg_color,
        1,
        cv2.LINE_AA,
    )
    return bw


def draw_status_indicator(
    canvas: np.ndarray,
    x: int,
    y: int,
    status: AppStatus,
    theme: UITheme,
    is_speaking: bool = False,
) -> int:
    """Renders the global application status badge."""
    if status == AppStatus.SPEAKING or is_speaking:
        dot_color = theme.primary_accent
        text = "SPEAKING"
        bg = (50, 40, 10)
    elif status in (AppStatus.RECOGNIZING, AppStatus.DWELLING):
        dot_color = theme.warning_accent
        text = "DWELLING..."
        bg = (10, 35, 45)
    elif status == AppStatus.STABLE_EMITTED:
        dot_color = theme.secondary_accent
        text = "SIGN EMITTED"
        bg = (15, 45, 20)
    elif status == AppStatus.WORD_COMPLETED:
        dot_color = theme.secondary_accent
        text = "WORD READY"
        bg = (15, 45, 20)
    elif status == AppStatus.HAND_DETECTED:
        dot_color = theme.secondary_accent
        text = "HAND ACTIVE"
        bg = (15, 35, 20)
    elif status == AppStatus.PAUSED:
        dot_color = theme.warning_accent
        text = "PAUSED"
        bg = (10, 30, 40)
    elif status in (AppStatus.CAMERA_ERROR, AppStatus.MODEL_ERROR, AppStatus.VOICE_ERROR):
        dot_color = theme.error_accent
        text = status.value
        bg = (10, 10, 45)
    else:
        dot_color = theme.muted_accent
        text = "NO HAND DETECTED"
        bg = (30, 25, 22)

    # Dot
    cv2.circle(canvas, (x + 8, y + 10), 4, dot_color, -1, cv2.LINE_AA)
    # Text badge
    return draw_badge(
        canvas,
        x + 18,
        y,
        text,
        fg_color=dot_color,
        bg_color=bg,
        border_color=dot_color,
        font_scale=theme.font_scale_small,
    ) + 18


def draw_metric_pill(
    canvas: np.ndarray,
    x: int,
    y: int,
    label: str,
    value: str,
    theme: UITheme,
) -> int:
    """Renders a telemetry metric badge (e.g. 'CAM: 30.0 FPS')."""
    full_text = f"{label}: {value}"
    return draw_badge(
        canvas,
        x,
        y,
        full_text,
        fg_color=theme.text_secondary,
        bg_color=theme.card_bg,
        border_color=theme.card_border,
        font_scale=theme.font_scale_small,
    )


def draw_switch(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    is_on: bool,
    label: str,
    is_hovered: bool,
    theme: UITheme,
) -> None:
    """Renders a modern toggle switch with descriptive label."""
    # Label text
    cv2.putText(
        canvas,
        label,
        (x, y + h // 2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        theme.font_scale_body,
        theme.text_primary if is_hovered else theme.text_secondary,
        1,
        cv2.LINE_AA,
    )

    # Switch track
    sw_w = 44
    sw_h = 22
    sw_x = x + w - sw_w
    sw_y = y + (h - sw_h) // 2

    track_bg = theme.secondary_accent if is_on else (45, 35, 30)
    cv2.rectangle(canvas, (sw_x, sw_y), (sw_x + sw_w, sw_y + sw_h), track_bg, -1)
    cv2.rectangle(canvas, (sw_x, sw_y), (sw_x + sw_w, sw_y + sw_h), theme.card_border, 1, cv2.LINE_AA)

    # Knob
    knob_radius = 8
    knob_x = (sw_x + sw_w - 12) if is_on else (sw_x + 12)
    knob_y = sw_y + sw_h // 2
    cv2.circle(canvas, (knob_x, knob_y), knob_radius, (255, 255, 255), -1, cv2.LINE_AA)
