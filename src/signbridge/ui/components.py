"""Stage 11: Reusable Modern UI Component Primitives & Typography Engine.

Provides high-performance, anti-aliased drawing primitives for cards, buttons,
progress meters, badges, switches, vector icons, and TrueType typography.
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from signbridge.ui.state import AppStatus
from signbridge.ui.theme import ColorBGR, UITheme

# ---------------------------------------------------------------------------
# TrueType Typography Cache & Renderer
# ---------------------------------------------------------------------------

_FONT_CACHE: Dict[Tuple[int, bool], Any] = {}


def get_font(size: int = 14, bold: bool = False) -> Any:
    """Retrieves or creates a cached PIL FreeType/TrueType font instance."""
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    try:
        from PIL import ImageFont
        font_names = (
            ["segoeuib.ttf", "arialbd.ttf", "calibrib.ttf"]
            if bold else
            ["segoeui.ttf", "arial.ttf", "calibri.ttf"]
        )
        for name in font_names:
            try:
                f = ImageFont.truetype(name, size)
                _FONT_CACHE[key] = f
                return f
            except Exception:
                continue
        f = ImageFont.load_default()
        _FONT_CACHE[key] = f
        return f
    except Exception:
        return None


def draw_text(
    canvas: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_size: int = 14,
    color: ColorBGR = (255, 255, 255),
    bold: bool = False,
    anchor: str = "lt",
) -> Tuple[int, int]:
    """Draws crisp anti-aliased TrueType text on canvas using system fonts.
    
    Supports anchors: 'lt' (left-top), 'lc' (left-center), 'cc' (center-center), etc.
    Returns (width, height) of rendered text.
    """
    if not text:
        return 0, 0

    x, y = pos
    font = get_font(font_size, bold)
    if font is not None:
        try:
            from PIL import Image, ImageDraw
            bbox = font.getbbox(text)
            tw = max(1, bbox[2] - bbox[0])
            th = max(1, bbox[3] - bbox[1])

            rx, ry = x, y
            if "r" in anchor:
                rx -= tw
            elif "c" in anchor or "m" in anchor:
                rx -= tw // 2
            if "b" in anchor:
                ry -= th
            elif "c" in anchor or "m" in anchor:
                ry -= th // 2

            ch, cw = canvas.shape[:2]
            x1, y1 = max(0, rx), max(0, ry)
            x2, y2 = min(cw, rx + tw + 6), min(ch, ry + th + 6)
            if x2 > x1 and y2 > y1:
                patch = canvas[y1:y2, x1:x2]
                pil_patch = Image.fromarray(patch[:, :, ::-1])
                draw = ImageDraw.Draw(pil_patch)
                color_rgb = (int(color[2]), int(color[1]), int(color[0]))
                draw.text((rx - x1 - bbox[0], ry - y1 - bbox[1]), text, font=font, fill=color_rgb)
                canvas[y1:y2, x1:x2] = np.asarray(pil_patch)[:, :, ::-1]
            return tw, th
        except Exception:
            pass

    # High-quality fallback to OpenCV putText
    scale = font_size / 28.0
    weight = 2 if bold else 1
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, weight)
    rx, ry = x, y + th
    if "r" in anchor:
        rx -= tw
    elif "c" in anchor or "m" in anchor:
        rx -= tw // 2
    if "b" in anchor:
        ry -= th
    elif "c" in anchor or "m" in anchor:
        ry -= th // 2
    cv2.putText(canvas, text, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)
    return tw, th


# ---------------------------------------------------------------------------
# Vector Geometry & Components
# ---------------------------------------------------------------------------

def hit_test(mx: int, my: int, bx: int, by: int, bw: int, bh: int) -> bool:
    """Returns True if point (mx, my) lies within the bounding box [bx, by, bx+bw, by+bh]."""
    return bx <= mx <= (bx + bw) and by <= my <= (by + bh)


def draw_rounded_rect(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    radius: int,
    bg_color: ColorBGR,
    border_color: Optional[ColorBGR] = None,
    border_width: int = 1,
) -> None:
    """Renders a smooth, anti-aliased rounded rectangle."""
    if w <= 0 or h <= 0:
        return
    r = max(1, min(radius, w // 2, h // 2))

    # Fill
    cv2.rectangle(canvas, (x + r, y), (x + w - r, y + h), bg_color, -1)
    cv2.rectangle(canvas, (x, y + r), (x + w, y + h - r), bg_color, -1)
    cv2.circle(canvas, (x + r, y + r), r, bg_color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (x + w - r, y + r), r, bg_color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (x + r, y + h - r), r, bg_color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (x + w - r, y + h - r), r, bg_color, -1, cv2.LINE_AA)

    # Border
    if border_color and border_width > 0:
        cv2.line(canvas, (x + r, y), (x + w - r, y), border_color, border_width, cv2.LINE_AA)
        cv2.line(canvas, (x + r, y + h), (x + w - r, y + h), border_color, border_width, cv2.LINE_AA)
        cv2.line(canvas, (x, y + r), (x, y + h - r), border_color, border_width, cv2.LINE_AA)
        cv2.line(canvas, (x + w, y + r), (x + w, y + h - r), border_color, border_width, cv2.LINE_AA)
        cv2.ellipse(canvas, (x + r, y + r), (r, r), 180, 0, 90, border_color, border_width, cv2.LINE_AA)
        cv2.ellipse(canvas, (x + w - r, y + r), (r, r), 270, 0, 90, border_color, border_width, cv2.LINE_AA)
        cv2.ellipse(canvas, (x + r, y + h - r), (r, r), 90, 0, 90, border_color, border_width, cv2.LINE_AA)
        cv2.ellipse(canvas, (x + w - r, y + h - r), (r, r), 0, 0, 90, border_color, border_width, cv2.LINE_AA)


def draw_card(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    bg_color: ColorBGR,
    border_color: ColorBGR,
    border_width: int = 1,
    radius: int = 6,
) -> None:
    """Renders a flat shaded modern panel card with smooth corners."""
    draw_rounded_rect(canvas, x, y, w, h, radius, bg_color, border_color, border_width)


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
    """Renders modern panel card header with title and category dot."""
    acc = accent_color or theme.primary_accent
    # Header strip background
    cv2.rectangle(canvas, (x, y), (x + w, y + 28), theme.header_bg, -1)
    cv2.line(canvas, (x, y + 28), (x + w, y + 28), theme.card_border, 1, cv2.LINE_AA)

    # Bullet dot
    cv2.circle(canvas, (x + 14, y + 14), 4, acc, -1, cv2.LINE_AA)

    # Title text
    draw_text(
        canvas,
        title.upper(),
        (x + 26, y + 7),
        font_size=12,
        color=theme.text_secondary,
        bold=True,
    )


def draw_vector_icon(
    canvas: np.ndarray,
    name: str,
    cx: int,
    cy: int,
    size: int = 12,
    color: ColorBGR = (255, 255, 255),
) -> None:
    """Draws a clean, anti-aliased vector geometric icon."""
    r = size // 2
    if name == "play":
        w_half = max(3, int(r * 0.75))
        h_half = max(3, int(r * 0.85))
        shift = max(1, int(r * 0.15))
        pts = np.array(
            [[cx - w_half + shift, cy - h_half], [cx - w_half + shift, cy + h_half], [cx + w_half + shift, cy]],
            np.int32,
        )
        cv2.fillPoly(canvas, [pts], color, cv2.LINE_AA)
    elif name == "pause":
        pw = max(2, int(r * 0.35))
        pg = max(2, int(r * 0.45))
        ph = max(3, int(r * 0.85))
        cv2.rectangle(canvas, (cx - pg // 2 - pw, cy - ph), (cx - pg // 2, cy + ph), color, -1)
        cv2.rectangle(canvas, (cx + (pg + 1) // 2, cy - ph), (cx + (pg + 1) // 2 + pw, cy + ph), color, -1)
    elif name == "stop":
        sr = max(2, int(r * 0.70))
        cv2.rectangle(canvas, (cx - sr, cy - sr), (cx + sr, cy + sr), color, -1)
    elif name == "replay":
        cv2.ellipse(canvas, (cx, cy), (r, r), 0, 45, 315, color, 2, cv2.LINE_AA)
        arrow_pts = np.array([[cx + r - 3, cy - 2], [cx + r + 4, cy - 5], [cx + r + 2, cy + 4]], np.int32)
        cv2.fillPoly(canvas, [arrow_pts], color, cv2.LINE_AA)
    elif name == "next":
        nx = max(3, int(r * 0.55))
        ny = max(3, int(r * 0.80))
        pts = np.array([[cx - nx // 2, cy - ny], [cx + nx, cy], [cx - nx // 2, cy + ny]], np.int32)
        cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
    elif name == "prev":
        nx = max(3, int(r * 0.55))
        ny = max(3, int(r * 0.80))
        pts = np.array([[cx + nx // 2, cy - ny], [cx - nx, cy], [cx + nx // 2, cy + ny]], np.int32)
        cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
    elif name in ("close", "x"):
        cv2.line(canvas, (cx - r, cy - r), (cx + r, cy + r), color, 2, cv2.LINE_AA)
        cv2.line(canvas, (cx - r, cy + r), (cx + r, cy - r), color, 2, cv2.LINE_AA)
    elif name == "search":
        cv2.circle(canvas, (cx - 2, cy - 2), max(2, r - 3), color, 2, cv2.LINE_AA)
        cv2.line(canvas, (cx + r - 4, cy + r - 4), (cx + r + 2, cy + r + 2), color, 2, cv2.LINE_AA)
    elif name == "check":
        pts = np.array([[cx - r, cy], [cx - r // 3, cy + r], [cx + r, cy - r]], np.int32)
        cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
    elif name == "warning":
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
        cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy + r - 3), 1, color, -1, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy - r // 3), (cx, cy + r // 3), color, 2, cv2.LINE_AA)
    elif name == "speaker":
        pts = np.array([[cx - r, cy - r // 3], [cx - r // 3, cy - r // 3], [cx + r // 3, cy - r], [cx + r // 3, cy + r], [cx - r // 3, cy + r // 3], [cx - r, cy + r // 3]], np.int32)
        cv2.fillPoly(canvas, [pts], color, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx + r // 2, cy), (r // 2, r // 2), 0, -45, 45, color, 1, cv2.LINE_AA)


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
    icon: Optional[str] = None,
) -> None:
    """Renders a modern interactive button with smooth corners and TrueType text."""
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

    # Rounded button surface
    draw_rounded_rect(canvas, x, y, w, h, radius=5, bg_color=bg, border_color=border, border_width=1)

    # Hotkey calculation (anchored on right side with consistent 6px inset)
    hk_w = 0
    hk_h = 0
    if hotkey:
        font_hk = get_font(10, bold=True)
        if font_hk is not None:
            try:
                bb = font_hk.getbbox(hotkey)
                hk_txt_w = max(1, bb[2] - bb[0])
            except Exception:
                hk_txt_w = len(hotkey) * 7
        else:
            hk_txt_w = len(hotkey) * 7

        hk_w = max(16, hk_txt_w + 10)
        hk_h = min(18, max(14, h - 8))
        hk_x = x + w - hk_w - 6
        hk_y = y + (h - hk_h) // 2
        draw_rounded_rect(canvas, hk_x, hk_y, hk_w, hk_h, radius=3, bg_color=(15, 12, 10), border_color=border, border_width=1)
        draw_text(
            canvas,
            hotkey,
            (hk_x + hk_w // 2, hk_y + hk_h // 2),
            font_size=10,
            color=theme.primary_accent if enabled else theme.text_muted,
            bold=True,
            anchor="cc",
        )

    # Available area for label and icon (with safety margins to avoid badge collision)
    left_inset = 6
    right_margin = (hk_w + 12) if hotkey else 6
    avail_x = x + left_inset
    avail_w = max(10, w - left_inset - right_margin)

    # Determine font size: for short height (<= 24) or compact button width (< 75), use font_size=11
    target_fsize = 11 if (h <= 24 or avail_w < 70) else 12

    # Measure and clamp label text if present to prevent any overlap with hotkey badge
    tw, th = 0, 0
    render_label = label
    if render_label:
        font = get_font(target_fsize, bold=is_active or accent_border)

        def _get_w(txt: str) -> int:
            if font is not None:
                try:
                    bb = font.getbbox(txt)
                    return max(1, bb[2] - bb[0])
                except Exception:
                    pass
            return len(txt) * 7

        tw = _get_w(render_label)
        max_label_w = avail_w - 4
        if tw > max_label_w and len(render_label) > 1:
            # Ellipsis trim to ensure text NEVER overlaps badge
            while len(render_label) > 1 and _get_w(render_label + "...") > max_label_w:
                render_label = render_label[:-1]
            render_label = render_label + "..."
            tw = _get_w(render_label)

        th = target_fsize

    # Layout content (icon, label, or both) centered in available area
    if icon and not render_label:
        # Icon only: exactly centered in available button surface
        draw_vector_icon(canvas, icon, avail_x + avail_w // 2, y + h // 2, size=max(10, min(16, h // 2)), color=text_col)
    elif icon and render_label:
        isize = max(10, min(14, int(h * 0.45)))
        gap = 5
        total_cw = isize + gap + tw
        start_x = avail_x + max(0, (avail_w - total_cw) // 2)
        icx = start_x + isize // 2
        icy = y + h // 2
        draw_vector_icon(canvas, icon, icx, icy, size=isize, color=text_col)
        tx = start_x + isize + gap + tw // 2
        ty = y + h // 2
        draw_text(canvas, render_label, (tx, ty), font_size=target_fsize, color=text_col, bold=is_active or accent_border, anchor="cc")
    elif render_label:
        # Label only: centered horizontally and vertically in available space with anchor="cc"
        tx = avail_x + avail_w // 2
        ty = y + h // 2
        draw_text(canvas, render_label, (tx, ty), font_size=target_fsize, color=text_col, bold=is_active or accent_border, anchor="cc")


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
    """Renders a smooth horizontal rounded progress bar."""
    clamped = max(0.0, min(1.0, float(progress)))
    r = max(2, h // 2)

    # Track background
    draw_rounded_rect(canvas, x, y, w, h, radius=r, bg_color=bg_color, border_color=border_color, border_width=1)

    # Fill
    fill_w = int(w * clamped)
    if fill_w >= r * 2:
        draw_rounded_rect(canvas, x, y, fill_w, h, radius=r, bg_color=fill_color, border_color=None, border_width=0)
    elif fill_w > 0:
        cv2.rectangle(canvas, (x, y), (x + fill_w, y + h), fill_color, -1)


def draw_badge(
    canvas: np.ndarray,
    x: int,
    y: int,
    text: str,
    fg_color: ColorBGR,
    bg_color: ColorBGR,
    border_color: Optional[ColorBGR] = None,
    pad_x: int = 8,
    pad_y: int = 3,
    font_scale: float = 0.40,
    radius: Optional[int] = None,
) -> int:
    """Renders a modern rounded pill badge and returns total rendered width."""
    font = get_font(11, bold=True)
    if font is not None:
        try:
            bbox = font.getbbox(text)
            tw = max(1, bbox[2] - bbox[0])
            th = max(1, bbox[3] - bbox[1])
        except Exception:
            tw, th = len(text) * 7, 12
    else:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

    bw = tw + pad_x * 2
    bh = th + pad_y * 2 + 2
    r = radius if radius is not None else bh // 2

    draw_rounded_rect(canvas, x, y, bw, bh, radius=r, bg_color=bg_color, border_color=border_color, border_width=1)
    draw_text(canvas, text, (x + pad_x, y + pad_y), font_size=11, color=fg_color, bold=True)
    return bw


def draw_status_indicator(
    canvas: np.ndarray,
    x: int,
    y: int,
    status: AppStatus,
    theme: UITheme,
    is_speaking: bool = False,
) -> int:
    """Renders the global application status badge with status indicator dot."""
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

    # Soft glowing indicator dot
    cv2.circle(canvas, (x + 8, y + 10), 5, dot_color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (x + 8, y + 10), 2, (255, 255, 255), -1, cv2.LINE_AA)

    # Pill badge
    return draw_badge(
        canvas,
        x + 18,
        y,
        text,
        fg_color=dot_color,
        bg_color=bg,
        border_color=dot_color,
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
    draw_text(
        canvas,
        label,
        (x, y + (h - 14) // 2),
        font_size=12,
        color=theme.text_primary if is_hovered else theme.text_secondary,
        bold=is_hovered,
    )

    # Switch track
    sw_w = 44
    sw_h = 22
    sw_x = x + w - sw_w
    sw_y = y + (h - sw_h) // 2

    track_bg = theme.secondary_accent if is_on else (45, 35, 30)
    draw_rounded_rect(canvas, sw_x, sw_y, sw_w, sw_h, radius=11, bg_color=track_bg, border_color=theme.card_border, border_width=1)

    # Knob
    knob_radius = 8
    knob_x = (sw_x + sw_w - 12) if is_on else (sw_x + 12)
    knob_y = sw_y + sw_h // 2
    cv2.circle(canvas, (knob_x, knob_y), knob_radius, (255, 255, 255), -1, cv2.LINE_AA)


def scale_and_letterbox_frame(
    frame: np.ndarray,
    target_w: int,
    target_h: int,
    bg_color: ColorBGR = (15, 12, 10),
) -> np.ndarray:
    """Scales frame to fit within target_w x target_h while strictly preserving aspect ratio."""
    fh, fw = frame.shape[:2]
    if fw <= 0 or fh <= 0 or target_w <= 0 or target_h <= 0:
        return np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    scale = min(target_w / fw, target_h / fh)
    nw = max(1, int(fw * scale))
    nh = max(1, int(fh * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    scaled = cv2.resize(frame, (nw, nh), interpolation=interp)

    out = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    ox = (target_w - nw) // 2
    oy = (target_h - nh) // 2
    out[oy : oy + nh, ox : ox + nw] = scaled
    return out
