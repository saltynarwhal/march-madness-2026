"""Abstract base class for prediction models."""

from abc import ABC, abstractmethod
from typing import TypedDict

from engine.db import TeamDB


def scores_from_margin(margin: float, total: float = 140.0, floor: float = 40.0) -> tuple[float, float]:
    """Compute (score_a, score_b) from margin and total.

    margin > 0 means Team A wins by that many points.
    """
    score_a = max((total + margin) / 2, floor)
    score_b = max((total - margin) / 2, floor)
    return score_a, score_b


class Prediction(TypedDict):
    team_a_score: float
    team_b_score: float
    winner_id: int
    confidence: float  # 0.5 – 1.0


class PredictionModel(ABC):
    """Every model takes two team IDs + the DB and returns a Prediction."""

    name: str

    @abstractmethod
    def predict(
        self,
        team_a_id: int,
        team_b_id: int,
        db: TeamDB,
        round_num: int = 1,
        slot_id: str | None = None,
    ) -> Prediction: ...
