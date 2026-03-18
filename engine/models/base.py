"""Abstract base class for prediction models."""

from abc import ABC, abstractmethod
from typing import TypedDict

from engine.db import TeamDB


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
    ) -> Prediction: ...
