"""Probability-based models (sampled / threshold / Monte Carlo-per-game consensus).

These models expect a trained classification pipeline exported from `march_madness.ipynb`:
  - data/models/prob_model.pkl
  - data/models/prob_feature_cols.pkl

They also load the existing regression artifacts to generate plausible scores:
  - data/models/score_margin_model.pkl
  - data/models/total_points_model.pkl
  - data/models/feature_cols.pkl
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)

from engine.db import TeamDB
from engine.models.base import Prediction, PredictionModel, scores_from_margin

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@dataclass(frozen=True)
class ThresholdConfig:
    """Round-aware thresholds for picking the favorite (Team A) by probability."""

    # For Round of 64: (fav_seed, dog_seed) -> P(fav wins) threshold to pick the favorite.
    # If prob < threshold, pick the upset.
    r64_seed_thresholds: dict[tuple[int, int], float]

    # For Round of 32: pick favorite when P(fav) >= r32_threshold
    r32_threshold: float

    # For Sweet 16 and later: pick favorite when P(fav) >= later_round_threshold
    later_round_threshold: float

    # Close call definition used only for internal labeling / debugging.
    close_call_delta: float = 0.03


DEFAULT_THRESHOLD_CONFIG = ThresholdConfig(
    r64_seed_thresholds={
        (1, 16): 0.50,
        (2, 15): 0.50,
        (3, 14): 0.50,
        (4, 13): 0.50,
        (5, 12): 0.60,
        (6, 11): 0.60,
        (7, 10): 0.601,
        (8, 9): 0.519,
    },
    r32_threshold=0.42,
    later_round_threshold=0.45,
)


class _ProbabilityBackbone:
    """Loads classifier + regressors and exposes helpers."""

    def __init__(self, models_dir: Path | str | None = None):
        self._models_dir = Path(models_dir) if models_dir else (DATA_DIR / "models")

        # Classifier artifacts
        self._prob_model_path = self._models_dir / "prob_model.pkl"
        self._prob_cols_path = self._models_dir / "prob_feature_cols.pkl"

        self._prob_model = None
        self._prob_feature_cols: list[str] | None = None

        # Score regressors (legacy contract)
        self._margin_model_path = self._models_dir / "score_margin_model.pkl"
        self._total_model_path = self._models_dir / "total_points_model.pkl"
        self._score_cols_path = self._models_dir / "feature_cols.pkl"

        self._margin_model = None
        self._total_model = None
        self._score_feature_cols: list[str] | None = None
        self._score_models_ok: bool | None = None
        self._score_models_err: Exception | None = None

    def can_predict_probs(self) -> bool:
        return self._prob_model_path.exists() and self._prob_cols_path.exists()

    def _load_prob_model(self) -> None:
        if self._prob_model is None:
            self._prob_model = joblib.load(self._prob_model_path)
        if self._prob_feature_cols is None:
            self._prob_feature_cols = joblib.load(self._prob_cols_path)

    def _load_score_models(self) -> None:
        if self._margin_model is not None and self._total_model is not None and self._score_feature_cols is not None:
            self._score_models_ok = True
            return
        try:
            self._margin_model = joblib.load(self._margin_model_path)
            self._total_model = joblib.load(self._total_model_path)
            self._score_feature_cols = joblib.load(self._score_cols_path)
            self._score_models_ok = True
        except Exception as exc:
            # Don’t hard-fail probability models: allow winner + probability to still render,
            # and fall back to a simple seed-based score heuristic.
            self._score_models_ok = False
            self._margin_model = None
            self._total_model = None
            self._score_feature_cols = None
            self._score_models_err = exc

    def _fallback_scores(self, team_a_id: int, team_b_id: int, db: TeamDB) -> tuple[float, float, float]:
        """Seed-based placeholder scores when regressors are unavailable."""
        seed_a = db.get_seed(team_a_id)
        seed_b = db.get_seed(team_b_id)
        if np.isnan(seed_a):
            logger.warning("Missing seed for team %d, defaulting to 8", team_a_id)
        if np.isnan(seed_b):
            logger.warning("Missing seed for team %d, defaulting to 8", team_b_id)
        sa = int(seed_a) if not np.isnan(seed_a) else 8
        sb = int(seed_b) if not np.isnan(seed_b) else 8
        margin = max(min((sb - sa) * 1.5, 25.0), -25.0)
        score_a, score_b = scores_from_margin(margin)
        return score_a, score_b, margin

    @staticmethod
    def _safe_float(val, default: float = 0.0) -> float:
        if val is None:
            return default
        if isinstance(val, float) and np.isnan(val):
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _matchup_feature_row(
        self,
        team_a_id: int,
        team_b_id: int,
        db: TeamDB,
        round_num: int,
        feature_cols: list[str],
    ) -> np.ndarray:
        # Use engine’s standard feature computation as a base.
        feats = db.compute_matchup_features(team_a_id, team_b_id, round_num=round_num)
        vec = np.array([[self._safe_float(feats.get(c, 0.0)) for c in feature_cols]], dtype=float)
        return vec

    def predict_prob_favorite_wins(
        self, team_a_id: int, team_b_id: int, db: TeamDB, round_num: int
    ) -> float:
        """Return P(Team A wins) where Team A is the lower seed (favored) per engine convention."""
        if not self.can_predict_probs():
            raise FileNotFoundError(
                "Missing classifier artifacts. Expected "
                f"{self._prob_model_path.name} and {self._prob_cols_path.name} in {self._models_dir}."
            )
        self._load_prob_model()
        assert self._prob_feature_cols is not None
        vec = self._matchup_feature_row(team_a_id, team_b_id, db, round_num, self._prob_feature_cols)
        # Expect sklearn-like predict_proba with class 1 = Team A win
        proba = float(self._prob_model.predict_proba(vec)[0, 1])
        return max(0.0, min(1.0, proba))

    def predict_scores_from_regressors(
        self, team_a_id: int, team_b_id: int, db: TeamDB, round_num: int
    ) -> tuple[float, float, float]:
        """Return (score_a, score_b, margin) using legacy regressors."""
        self._load_score_models()
        if self._score_models_ok:
            assert self._score_feature_cols is not None
            vec = self._matchup_feature_row(team_a_id, team_b_id, db, round_num, self._score_feature_cols)
            margin = float(self._margin_model.predict(vec)[0])  # type: ignore[union-attr]
            total = float(self._total_model.predict(vec)[0])  # type: ignore[union-attr]
            score_a, score_b = scores_from_margin(margin, total)
            return score_a, score_b, margin
        return self._fallback_scores(team_a_id, team_b_id, db)

    def predict_playin_favorite(
        self, team_a_id: int, team_b_id: int, db: TeamDB
    ) -> Prediction:
        """First Four / play-in games are outside the trained matchup model.

        The bracket engine always passes *Team A* as the better (lower) seed. We advance
        that team and use seed-based placeholder scores so R64 resolves consistently.
        """
        score_a, score_b, _ = self._fallback_scores(team_a_id, team_b_id, db)
        return Prediction(
            team_a_score=round(score_a, 1),
            team_b_score=round(score_b, 1),
            winner_id=int(team_a_id),
            confidence=0.58,
        )


class SampledProbabilityModel(PredictionModel):
    """Option A — sample each game winner from model probability."""

    name = "Lean GB (Sampled)"

    def __init__(self, models_dir: Path | str | None = None, random_seed: int = 12345):
        self._core = _ProbabilityBackbone(models_dir=models_dir)
        self._rng = np.random.default_rng(int(random_seed))

    def predict(
        self,
        team_a_id: int,
        team_b_id: int,
        db: TeamDB,
        round_num: int = 1,
        slot_id: str | None = None,
    ) -> Prediction:
        if round_num == 0:
            return self._core.predict_playin_favorite(team_a_id, team_b_id, db)
        p = self._core.predict_prob_favorite_wins(team_a_id, team_b_id, db, round_num)
        draw = float(self._rng.random())
        pick_a = draw < p
        winner = team_a_id if pick_a else team_b_id

        score_a, score_b, _margin = self._core.predict_scores_from_regressors(team_a_id, team_b_id, db, round_num)

        # confidence stored as win-prob of picked team
        conf = p if pick_a else (1.0 - p)
        conf = max(0.5, min(1.0, conf))

        return Prediction(
            team_a_score=round(score_a, 1),
            team_b_score=round(score_b, 1),
            winner_id=winner,
            confidence=round(conf, 3),
        )


class ThresholdProbabilityModel(PredictionModel):
    """Option B — deterministic threshold strategy."""

    name = "Lean GB (Tiered Threshold)"

    def __init__(
        self,
        models_dir: Path | str | None = None,
        thresholds: ThresholdConfig = DEFAULT_THRESHOLD_CONFIG,
    ):
        self._core = _ProbabilityBackbone(models_dir=models_dir)
        self._t = thresholds

    def _threshold_for(self, seed_a: float, seed_b: float, round_num: int) -> float:
        if round_num <= 1:
            # R64 thresholds keyed by (fav, dog) with Team A convention = favored/lower seed.
            if np.isnan(seed_a):
                logger.warning("Missing seed_a in threshold lookup, defaulting to 8")
            if np.isnan(seed_b):
                logger.warning("Missing seed_b in threshold lookup, defaulting to 8")
            sa = int(seed_a) if not np.isnan(seed_a) else 8
            sb = int(seed_b) if not np.isnan(seed_b) else 8
            key = (min(sa, sb), max(sa, sb))
            return float(self._t.r64_seed_thresholds.get(key, 0.50))
        if round_num == 2:
            return float(self._t.r32_threshold)
        return float(self._t.later_round_threshold)

    def predict(
        self,
        team_a_id: int,
        team_b_id: int,
        db: TeamDB,
        round_num: int = 1,
        slot_id: str | None = None,
    ) -> Prediction:
        if round_num == 0:
            return self._core.predict_playin_favorite(team_a_id, team_b_id, db)
        p = self._core.predict_prob_favorite_wins(team_a_id, team_b_id, db, round_num)
        seed_a = db.get_seed(team_a_id)
        seed_b = db.get_seed(team_b_id)
        thr = self._threshold_for(seed_a, seed_b, round_num)

        pick_a = p >= thr
        winner = team_a_id if pick_a else team_b_id

        score_a, score_b, _margin = self._core.predict_scores_from_regressors(team_a_id, team_b_id, db, round_num)

        conf = p if pick_a else (1.0 - p)
        conf = max(0.5, min(1.0, conf))

        return Prediction(
            team_a_score=round(score_a, 1),
            team_b_score=round(score_b, 1),
            winner_id=winner,
            confidence=round(conf, 3),
        )


class MonteCarloConsensusModel(PredictionModel):
    """Option C — full-bracket Monte Carlo, then per-slot consensus.

    This matches the notebook-style “Option C” logic: simulate the entire bracket N times
    and pick the team that appears as winner of each slot most often.

    Precomputed ``win_freq`` is the marginal share of simulations where the consensus team
    won that slot; ``predict()`` does **not** use that as ``confidence``. The dashboard expects
    ``confidence`` ≈ P(predicted winner wins this matchup), so we set it from the classifier
    while using the CSV only to choose ``winner_id``.
    """

    name = "Lean GB (MC Consensus)"

    def __init__(
        self,
        models_dir: Path | str | None = None,
        n_sims: int = 10_000,
        random_seed: int = 12345,
        season: int = 2026,
    ):
        self._core = _ProbabilityBackbone(models_dir=models_dir)
        self._n = int(n_sims)
        self._rng = np.random.default_rng(int(random_seed))
        self._consensus: dict[str, tuple[int, float]] | None = None  # slot_id -> (winner_id, freq)
        self._season = season

        # Load seeds/slots lazily from the repo’s `data/kaggle/` the same way dashboard does.
        self._data_dir = Path(models_dir).resolve().parent if models_dir else DATA_DIR
        self._seeds_df = None
        self._slots_df = None
        self._precomputed_path = self._data_dir / "cache" / f"mc_slot_consensus_{season}.csv"

    def _load_bracket_inputs(self):
        if self._seeds_df is not None and self._slots_df is not None:
            return
        import pandas as pd

        kaggle_dir = self._data_dir / "kaggle"
        self._seeds_df = pd.read_csv(kaggle_dir / "MNCAATourneySeeds.csv")
        self._slots_df = pd.read_csv(kaggle_dir / "MNCAATourneySlots.csv")

    def _ensure_consensus(self, db: TeamDB) -> None:
        if self._consensus is not None:
            return

        # Fast path: load precomputed slot consensus exported from `march_madness.ipynb`
        if self._precomputed_path.exists():
            import pandas as pd

            df = pd.read_csv(self._precomputed_path)
            if "season" in df.columns:
                df = df[df["season"] == self._season]
                if df.empty:
                    raise ValueError(
                        f"MC consensus CSV has no rows for season {self._season}. "
                        f"Re-export from the notebook for the correct season."
                    )
            needed = {"slot_id", "winner_team_id", "win_freq"}
            if needed.issubset(set(df.columns)) and not df.empty:
                self._consensus = {
                    str(r["slot_id"]): (
                        int(r["winner_team_id"]),
                        float(max(0.0, min(1.0, r["win_freq"]))),
                    )
                    for _, r in df.iterrows()
                }
                return

        # Avoid freezing Streamlit startup: if the precomputed file is missing,
        # require the user to export it from the notebook rather than running
        # thousands of full-bracket sims inside the dashboard.
        raise RuntimeError(
            "Missing precomputed Monte Carlo consensus file. "
            "Run the 'Dashboard Export (v4)' cell in `march_madness.ipynb` to create "
            f"`{self._precomputed_path.as_posix()}`. "
            "This prevents long 'Simulating brackets…' delays in Streamlit."
        )

        # Import here to avoid circular import at module load time:
        # engine.bracket -> engine.models.base -> engine.models.__init__ -> engine.models.probability
        from engine.bracket import Bracket

        self._load_bracket_inputs()
        assert self._seeds_df is not None and self._slots_df is not None

        # Inner sampled model for Monte Carlo runs (uses only probabilities, not score regressors).
        class _Sampler(PredictionModel):
            name = "_MC_Sampler"

            def __init__(self, core: _ProbabilityBackbone, rng: np.random.Generator):
                self._core = core
                self._rng = rng

            def predict(
                self,
                team_a_id: int,
                team_b_id: int,
                db: TeamDB,
                round_num: int = 1,
                slot_id: str | None = None,
            ) -> Prediction:
                p = self._core.predict_prob_favorite_wins(team_a_id, team_b_id, db, round_num)
                pick_a = float(self._rng.random()) < p
                winner = team_a_id if pick_a else team_b_id
                # Scores are irrelevant for MC; return placeholders.
                return Prediction(
                    team_a_score=70.0,
                    team_b_score=68.0,
                    winner_id=int(winner),
                    confidence=round(float(max(p, 1.0 - p)), 3),
                )

        sampler = _Sampler(self._core, self._rng)

        # Count winners per slot across simulations.
        counts: dict[str, dict[int, int]] = {}
        for _ in range(self._n):
            b = Bracket(self._seeds_df, self._slots_df, season=2026)
            b.simulate(sampler, db)
            for sid, slot in b.slots.items():
                wid = slot.winner_team_id
                if wid is None:
                    continue
                if sid not in counts:
                    counts[sid] = {}
                counts[sid][int(wid)] = counts[sid].get(int(wid), 0) + 1

        consensus: dict[str, tuple[int, float]] = {}
        for sid, c in counts.items():
            winner_id, wins = max(c.items(), key=lambda kv: kv[1])
            freq = wins / self._n if self._n > 0 else 0.5
            consensus[sid] = (int(winner_id), float(max(0.0, min(1.0, freq))))

        self._consensus = consensus

        # Save for next dashboard run (so it becomes instant)
        try:
            import pandas as pd

            out = pd.DataFrame(
                [
                    {
                        "season": 2026,
                        "slot_id": sid,
                        "winner_team_id": wid,
                        "win_freq": conf,
                        "n_sims": self._n,
                    }
                    for sid, (wid, conf) in consensus.items()
                ]
            )
            self._precomputed_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(self._precomputed_path, index=False)
        except Exception:
            pass

    def predict(
        self,
        team_a_id: int,
        team_b_id: int,
        db: TeamDB,
        round_num: int = 1,
        slot_id: str | None = None,
    ) -> Prediction:
        if round_num == 0:
            return self._core.predict_playin_favorite(team_a_id, team_b_id, db)

        self._ensure_consensus(db)
        assert self._consensus is not None

        # Slot-based consensus picks the winner; matchup probability comes from the classifier
        # (same convention as the dashboard: confidence = P(predicted winner wins)).
        # Safety: consensus files can be misaligned. Never return a winner not in this matchup.
        winner: int | None = None
        if slot_id is not None and slot_id in self._consensus:
            w_id, _w_freq = self._consensus[slot_id]
            if int(w_id) in (int(team_a_id), int(team_b_id)):
                winner = int(w_id)

        sa, sb = db.get_seed(team_a_id), db.get_seed(team_b_id)
        tie_seed = not np.isnan(sa) and not np.isnan(sb) and float(sa) == float(sb)
        if tie_seed:
            # Same numeric seed (e.g. two #1s): bracket order is Kaggle strong/weak, not "favorite".
            # Use lower TeamID as classifier Team A so P(winner) matches a stable convention.
            lo, hi = (team_a_id, team_b_id) if team_a_id <= team_b_id else (team_b_id, team_a_id)
            p_lo = self._core.predict_prob_favorite_wins(lo, hi, db, round_num)
            if winner is None:
                winner = int(lo if p_lo >= 0.5 else hi)
            conf = float(p_lo if winner == int(lo) else 1.0 - p_lo)
        else:
            p_fav = self._core.predict_prob_favorite_wins(team_a_id, team_b_id, db, round_num)
            if winner is None:
                winner = int(team_a_id if p_fav >= 0.5 else team_b_id)
            conf = float(p_fav if winner == int(team_a_id) else 1.0 - p_fav)
        conf = round(max(0.0, min(1.0, conf)), 3)

        score_a, score_b, _margin = self._core.predict_scores_from_regressors(team_a_id, team_b_id, db, round_num)
        return Prediction(
            team_a_score=round(score_a, 1),
            team_b_score=round(score_b, 1),
            winner_id=int(winner),
            confidence=round(float(conf), 3),
        )

