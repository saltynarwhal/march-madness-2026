"""Seeding-only model – higher seed wins, ties broken by W/L%."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

from engine.db import TeamDB
from engine.models.base import Prediction, PredictionModel

BASE_SCORE = 70.0
SEED_MARGIN_PER_LINE = 1.5


class SeedingModel(PredictionModel):
    name = "Seeding Only"

    def __init__(self, db: TeamDB | None = None):
        self._hist_scores: dict[tuple[int, int], tuple[float, float]] = {}
        if db is not None:
            self._hist_scores = db.get_historical_seed_scores()

    def predict(
        self,
        team_a_id: int,
        team_b_id: int,
        db: TeamDB,
        round_num: int = 1,
        slot_id: str | None = None,
    ) -> Prediction:
        seed_a = db.get_seed(team_a_id)
        seed_b = db.get_seed(team_b_id)

        # Determine winner
        if np.isnan(seed_a) and np.isnan(seed_b):
            winner = self._tiebreak(team_a_id, team_b_id, db)
        elif np.isnan(seed_a):
            winner = team_b_id
        elif np.isnan(seed_b):
            winner = team_a_id
        elif seed_a < seed_b:
            winner = team_a_id
        elif seed_b < seed_a:
            winner = team_b_id
        else:
            winner = self._tiebreak(team_a_id, team_b_id, db)

        # Scores: try historical averages first, fall back to formula
        if np.isnan(seed_a):
            logger.warning("Missing seed for team %d, defaulting to 8", team_a_id)
        if np.isnan(seed_b):
            logger.warning("Missing seed for team %d, defaulting to 8", team_b_id)
        sa_seed = int(seed_a) if not np.isnan(seed_a) else 8
        sb_seed = int(seed_b) if not np.isnan(seed_b) else 8

        key = (min(sa_seed, sb_seed), max(sa_seed, sb_seed))
        if key in self._hist_scores:
            fav_score, dog_score = self._hist_scores[key]
        else:
            gap = abs(sa_seed - sb_seed)
            fav_score = BASE_SCORE + gap * SEED_MARGIN_PER_LINE / 2
            dog_score = BASE_SCORE - gap * SEED_MARGIN_PER_LINE / 2

        if winner == team_a_id:
            score_a, score_b = max(fav_score, dog_score), min(fav_score, dog_score)
        else:
            score_a, score_b = min(fav_score, dog_score), max(fav_score, dog_score)

        gap = abs(seed_a - seed_b) if not (np.isnan(seed_a) or np.isnan(seed_b)) else 0
        confidence = min(gap / 15.0, 1.0) * 0.5 + 0.5

        return Prediction(
            team_a_score=round(float(score_a), 1),
            team_b_score=round(float(score_b), 1),
            winner_id=winner,
            confidence=round(confidence, 3),
        )

    @staticmethod
    def _tiebreak(id_a: int, id_b: int, db: TeamDB) -> int:
        """Same seed: pick whichever has the better regular-season W/L%."""
        a_pct = db.get_team(id_a).get("win_pct", 0.5)
        b_pct = db.get_team(id_b).get("win_pct", 0.5)
        return id_a if a_pct >= b_pct else id_b
