"""Advanced metrics model – uses trained regressors from the notebook."""

from pathlib import Path

import joblib
import numpy as np

from engine.db import TeamDB
from engine.models.base import Prediction, PredictionModel

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class AdvancedMetricsModel(PredictionModel):
    name = "Advanced Metrics"

    def __init__(self, models_dir: Path | str | None = None):
        models_dir = Path(models_dir) if models_dir else DATA_DIR / "models"
        self._margin_model = joblib.load(models_dir / "score_margin_model.pkl")
        self._total_model = joblib.load(models_dir / "total_points_model.pkl")
        self._feature_cols: list[str] = joblib.load(models_dir / "feature_cols.pkl")

    def predict(
        self,
        team_a_id: int,
        team_b_id: int,
        db: TeamDB,
        round_num: int = 1,
    ) -> Prediction:
        features = db.compute_matchup_features(team_a_id, team_b_id, round_num=round_num)

        vec = np.array(
            [[features.get(c, 0.0) for c in self._feature_cols]]
        )

        margin = float(self._margin_model.predict(vec)[0])
        total = float(self._total_model.predict(vec)[0])

        score_a = (total + margin) / 2
        score_b = (total - margin) / 2

        score_a = max(score_a, 40.0)
        score_b = max(score_b, 40.0)

        winner = team_a_id if margin >= 0 else team_b_id
        confidence = min(abs(margin) / 30.0, 1.0) * 0.5 + 0.5

        return Prediction(
            team_a_score=round(score_a, 1),
            team_b_score=round(score_b, 1),
            winner_id=winner,
            confidence=round(confidence, 3),
        )
