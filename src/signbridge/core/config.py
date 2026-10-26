"""Module 1: Application Configuration.

Centralized type-safe settings for SignBridge.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    flip_horizontal: bool = True


@dataclass(frozen=True)
class VisionConfig:
    max_num_hands: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.6
    model_complexity: int = 1
    static_image_mode: bool = False


@dataclass(frozen=True)
class RecognitionConfig:
    model_type: str = "heuristic"  # Options: 'heuristic', 'knn', 'rf', 'mlp', 'lstm'
    model_weights_path: str = "models/sign_classifier.pkl"
    confidence_threshold: float = 0.75
    min_margin_threshold: float = 0.15


@dataclass(frozen=True)
class TemporalConfig:
    window_size: int = 10
    consensus_threshold: float = 0.7
    dwell_time_seconds: float = 0.5
    idle_timeout_seconds: float = 1.5


@dataclass(frozen=True)
class AudioConfig:
    enabled: bool = True
    speech_rate: int = 150
    volume: float = 1.0


@dataclass(frozen=True)
class UIConfig:
    window_title: str = "SignBridge — Real-Time Translation System"
    show_landmarks: bool = True
    show_hud: bool = True
    show_fps: bool = True


@dataclass(frozen=True)
class AppConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    ui: UIConfig = field(default_factory=UIConfig)
