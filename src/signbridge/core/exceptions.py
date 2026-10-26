"""Module 16: Error Handling.

Domain exception hierarchy for the SignBridge translation pipeline.
"""


class SignBridgeError(Exception):
    """Base domain exception for all SignBridge errors."""
    pass


class ConfigurationError(SignBridgeError):
    """Raised when configuration parameters are invalid or missing."""
    pass


class CameraError(SignBridgeError):
    """Base exception for camera acquisition failures."""
    pass


class CameraNotFoundError(CameraError):
    """Raised when the specified video capture device cannot be opened."""
    pass


class CameraFrameCaptureError(CameraError):
    """Raised when frame capture fails or times out from camera stream."""
    pass


class VisionPipelineError(SignBridgeError):
    """Base exception for computer vision and landmark extraction errors."""
    pass


class LandmarkExtractionError(VisionPipelineError):
    """Raised when landmark extraction encounters malformed frames or fails."""
    pass


class ModelError(SignBridgeError):
    """Base exception for sign recognition model failures."""
    pass


class ModelWeightsNotFoundError(ModelError):
    """Raised when model weights file cannot be located on disk."""
    pass


class ModelInferenceError(ModelError):
    """Raised when model forward pass or tensor shape validation fails."""
    pass


class AudioSynthesisError(SignBridgeError):
    """Raised when TTS engine fails to initialize or synthesize speech."""
    pass
