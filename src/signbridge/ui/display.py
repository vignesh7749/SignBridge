"""Module 15: OpenCV Display Coordinator."""

import cv2
import numpy as np

from signbridge.core.config import UIConfig
from signbridge.core.logger import get_logger

logger = get_logger("ui.display")


class DisplayCoordinator:
    """Manages the OpenCV GUI window and user key event handling."""

    def __init__(self, config: UIConfig) -> None:
        self.config = config
        self._window_name = config.window_title

    def show(self, frame: np.ndarray) -> int:
        """Renders frame in OpenCV window and returns pressed key code."""
        cv2.imshow(self._window_name, frame)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        """Destroys OpenCV GUI windows."""
        cv2.destroyAllWindows()
        logger.info("Display windows closed.")
