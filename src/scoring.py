"""Credit-score conversion utilities."""

import numpy as np


def prob_to_credit_score(
    probability_default: np.ndarray,
    base_score: int = 600,
    points_to_double_odds: int = 20,
    base_odds: float = 50.0,
) -> np.ndarray:
    """Convert Probability of Default to a FICO-like score between 300 and 850."""
    factor = points_to_double_odds / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    clipped_probability = np.clip(probability_default, 0.0001, 0.9999)
    odds = (1.0 - clipped_probability) / clipped_probability
    scores = offset + factor * np.log(odds)
    return np.clip(np.round(scores), 300, 850).astype(int)
