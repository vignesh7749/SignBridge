"""Centralized configuration for the unified SignBridge end-to-end pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from signbridge.core.config import CameraConfig, VisionConfig
from signbridge.temporal.config import TemporalEngineConfig
from signbridge.translation.config import TranslationConfig
from signbridge.tts.config import TTSConfig


@dataclass(frozen=True)
class PipelineConfig:
    """Unified runtime settings for the end-to-end SignBridge pipeline."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=lambda: VisionConfig(max_num_hands=2, min_detection_confidence=0.70, min_tracking_confidence=0.65))
    temporal: TemporalEngineConfig = field(default_factory=TemporalEngineConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)

    # Runtime and presentation policies
    model_dir: Optional[Path] = None
    min_classification_confidence: float = 0.65
    primary_hand_preference: str = "highest_confidence"  # 'highest_confidence', 'right', 'left'
    draw_skeleton: bool = True
    draw_hud: bool = True
    draw_bbox: bool = True
    enable_profiling: bool = True
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 3
