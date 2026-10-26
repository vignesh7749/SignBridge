"""Stage 11: UI Design System & Theme Definitions.

Defines cohesive modern color palettes, typography hierarchies, layout metrics,
and accessibility-compliant styling parameters for the SignBridge application.
"""

from dataclasses import dataclass
from typing import Tuple

# BGR color tuple type for OpenCV drawing operations
ColorBGR = Tuple[int, int, int]


@dataclass(frozen=True)
class UITheme:
    """Color palette and visual styling parameters in OpenCV BGR format."""

    name: str

    # Base Backgrounds & Surfaces
    bg_canvas: ColorBGR
    card_bg: ColorBGR
    card_border: ColorBGR
    card_border_focus: ColorBGR
    header_bg: ColorBGR
    footer_bg: ColorBGR
    modal_overlay: ColorBGR

    # Accent Colors
    primary_accent: ColorBGR       # Brand Cyan / Teal
    secondary_accent: ColorBGR     # Emerald Green (Success / Stable)
    warning_accent: ColorBGR       # Amber (Caution / Dwelling)
    error_accent: ColorBGR         # Crimson Red (Error / Failure)
    muted_accent: ColorBGR         # Slate Grey (Idle / Disabled)

    # Typography Colors
    text_primary: ColorBGR         # Main headings & values
    text_secondary: ColorBGR       # Sub-labels & metadata
    text_muted: ColorBGR           # Inactive hints & disabled text
    text_dark: ColorBGR            # Dark text on bright badge pills

    # Interactive States
    btn_normal_bg: ColorBGR
    btn_hover_bg: ColorBGR
    btn_active_bg: ColorBGR
    btn_border: ColorBGR

    # Metric & Border Scales
    border_width: int = 1
    border_focus_width: int = 2
    card_radius: int = 8
    font_scale_title: float = 0.70
    font_scale_heading: float = 0.55
    font_scale_body: float = 0.45
    font_scale_small: float = 0.38
    font_scale_focal: float = 1.10


# Standard Obsidian / Dark Slate Professional Theme
OBSIDIAN_THEME = UITheme(
    name="Obsidian Dark",
    bg_canvas=(23, 17, 15),           # Deep Obsidian #0F1117
    card_bg=(39, 29, 26),             # Slate Surface #1A1D27
    card_border=(71, 52, 46),         # Muted Indigo Border #2E3447
    card_border_focus=(255, 229, 0),  # Cyan Accent Border
    header_bg=(28, 20, 18),           # Header Bar
    footer_bg=(28, 20, 18),           # Footer Strip
    modal_overlay=(10, 8, 7),         # Translucent Modal Backdrop

    primary_accent=(255, 229, 0),     # Vibrant Cyan #00E5FF (BGR)
    secondary_accent=(118, 230, 0),   # Emerald Green #00E676 (BGR)
    warning_accent=(0, 179, 255),     # Amber Gold #FFB300 (BGR)
    error_accent=(113, 61, 255),      # Crimson Red #FF3D71 (BGR)
    muted_accent=(120, 100, 95),      # Muted Slate

    text_primary=(255, 255, 255),     # Pure White
    text_secondary=(181, 165, 160),   # Silver Grey #A0A5B5
    text_muted=(120, 110, 105),       # Subdued Grey
    text_dark=(20, 18, 16),           # Near Black for pills

    btn_normal_bg=(52, 40, 35),       # Dark Slate Button
    btn_hover_bg=(72, 56, 48),        # Hover Highlight
    btn_active_bg=(95, 75, 65),       # Active Pressed State
    btn_border=(80, 60, 52),
)

# High Contrast Theme (WCAG AAA Accessibility Compliant)
HIGH_CONTRAST_THEME = UITheme(
    name="High Contrast",
    bg_canvas=(0, 0, 0),              # Pure Black
    card_bg=(15, 15, 15),             # Dark Charcoal
    card_border=(255, 255, 255),      # Pure White High-Visibility Border
    card_border_focus=(0, 255, 255),  # Yellow Focus Border
    header_bg=(0, 0, 0),
    footer_bg=(0, 0, 0),
    modal_overlay=(0, 0, 0),

    primary_accent=(0, 255, 255),     # Pure Yellow (Maximum Contrast on Black)
    secondary_accent=(0, 255, 0),     # Pure Green
    warning_accent=(0, 200, 255),     # Bright Orange
    error_accent=(0, 0, 255),         # Bright Red
    muted_accent=(180, 180, 180),     # High-vis Grey

    text_primary=(255, 255, 255),     # Pure White
    text_secondary=(220, 220, 220),   # Bright Off-White
    text_muted=(180, 180, 180),
    text_dark=(0, 0, 0),

    btn_normal_bg=(25, 25, 25),
    btn_hover_bg=(60, 60, 60),
    btn_active_bg=(100, 100, 100),
    btn_border=(255, 255, 255),
    border_width=2,
    border_focus_width=3,
    font_scale_title=0.75,
    font_scale_heading=0.60,
    font_scale_body=0.50,
    font_scale_small=0.42,
    font_scale_focal=1.20,
)
