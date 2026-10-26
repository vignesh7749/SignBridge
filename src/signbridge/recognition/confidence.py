"""Module 8: Confidence Estimation & Scored Predictions."""

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class ScoredPrediction:
    """Represents the outcome of sign recognition inference."""
    label: str
    confidence: float
    is_valid: bool
    probabilities: np.ndarray
    top_candidates: Sequence[tuple[str, float]]

    @property
    def top_margin(self) -> float:
        """Difference between top-1 and top-2 candidate confidence."""
        if len(self.top_candidates) >= 2:
            return self.top_candidates[0][1] - self.top_candidates[1][1]
        return self.confidence
