"""Advanced metrics model – uses trained regressors from the notebook."""

import logging
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)

from engine.db import TeamDB
from engine.models.base import Prediction, PredictionModel, scores_from_margin

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class AdvancedMetricsModel(PredictionModel):
    name = "Comparative Metrics"

    def __init__(self, models_dir: Path | str | None = None):
        models_dir = Path(models_dir) if models_dir else DATA_DIR / "models"
        self._models_dir = models_dir
        self._margin_model = None
        self._total_model = None
        self._feature_cols: list[str] | None = None

        self._prob_model = None
        self._prob_feature_cols: list[str] | None = None
        self._prob_available: bool | None = None

    def _load(self) -> None:
        if self._margin_model is not None and self._total_model is not None and self._feature_cols is not None:
            return
        try:
            self._margin_model = joblib.load(self._models_dir / "score_margin_model.pkl")
            self._total_model = joblib.load(self._models_dir / "total_points_model.pkl")
            self._feature_cols = joblib.load(self._models_dir / "feature_cols.pkl")
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Comparative Metrics artifacts from "
                f"{self._models_dir}. Re-run `march_madness.ipynb` to re-export "
                "`score_margin_model.pkl`, `total_points_model.pkl`, and `feature_cols.pkl` "
                "in a compatible environment."
            ) from exc

    def _load_prob_model(self) -> None:
        if self._prob_available is not None:
            return
        prob_path = self._models_dir / "prob_model.pkl"
        cols_path = self._models_dir / "prob_feature_cols.pkl"
        if not prob_path.exists() or not cols_path.exists():
            self._prob_available = False
            return
        try:
            self._prob_model = joblib.load(prob_path)
            self._prob_feature_cols = joblib.load(cols_path)
            self._prob_available = True
        except Exception as exc:
            logger.warning("Could not load prob_model for calibrated confidence: %s", exc)
            self._prob_available = False

    def _calibrated_confidence(
        self, team_a_id: int, team_b_id: int, winner_id: int,
        db: TeamDB, round_num: int,
    ) -> float | None:
        """Return P(winner wins) from the calibrated classifier, or None."""
        self._load_prob_model()
        if not self._prob_available:
            return None
        assert self._prob_feature_cols is not None
        features = db.compute_matchup_features(team_a_id, team_b_id, round_num=round_num)
        vec = np.array(
            [[float(features.get(c, 0.0)) for c in self._prob_feature_cols]],
            dtype=float,
        )
        p_a = float(self._prob_model.predict_proba(vec)[0, 1])
        return p_a if winner_id == team_a_id else 1.0 - p_a

    def predict(
        self,
        team_a_id: int,
        team_b_id: int,
        db: TeamDB,
        round_num: int = 1,
        slot_id: str | None = None,
    ) -> Prediction:
        self._load()
        assert self._feature_cols is not None
        features = db.compute_matchup_features(team_a_id, team_b_id, round_num=round_num)

        vec = np.array(
            [[features.get(c, 0.0) for c in self._feature_cols]]
        )

        margin = float(self._margin_model.predict(vec)[0])
        total = float(self._total_model.predict(vec)[0])

        score_a, score_b = scores_from_margin(margin, total)

        winner = team_a_id if margin >= 0 else team_b_id

        cal_conf = self._calibrated_confidence(
            team_a_id, team_b_id, winner, db, round_num,
        )
        if cal_conf is not None:
            confidence = max(0.5, min(1.0, cal_conf))
        else:
            confidence = min(abs(margin) / 30.0, 1.0) * 0.5 + 0.5

        return Prediction(
            team_a_score=round(score_a, 1),
            team_b_score=round(score_b, 1),
            winner_id=winner,
            confidence=round(confidence, 3),
        )
