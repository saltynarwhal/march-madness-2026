"""Team fact database for March Madness prediction engine."""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from sklearn.linear_model import LinearRegression


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TeamDB:
    """Central lookup for team facts, seeds, and matchup feature computation."""

    def __init__(self, data_dir=None):
        data_dir = Path(data_dir) if data_dir else DATA_DIR

        self._season = pd.read_csv(data_dir / "season_2026.csv", low_memory=False)
        self._season["kaggle_team_id"] = pd.to_numeric(
            self._season["kaggle_team_id"], errors="coerce"
        )

        teams_path = data_dir / "kaggle" / "MTeams.csv"
        self._teams = pd.read_csv(teams_path) if teams_path.exists() else pd.DataFrame()

        self._team_facts: dict[int, dict] = {}
        for _, row in self._season.iterrows():
            tid = row.get("kaggle_team_id")
            if pd.notna(tid):
                self._team_facts[int(tid)] = row.to_dict()

        for facts in self._team_facts.values():
            facts["win_pct"] = self._parse_record(facts.get("record", ""))

        self._seeds: dict[int, float] = {}

        self._name_index: dict[str, int] = self._build_name_index()

        matchup_path = data_dir / "matchup_dataset.csv"
        self._matchups = (
            pd.read_csv(matchup_path, low_memory=False)
            if matchup_path.exists()
            else pd.DataFrame()
        )
        # Same residual as notebook `make_2026_features`: seed_disagreement = adj_em_diff - E[adj_em|seed_diff]
        self._lr_seed_em: LinearRegression | None = None
        if (
            not self._matchups.empty
            and "adj_em_diff" in self._matchups.columns
            and "seed_diff" in self._matchups.columns
        ):
            df = self._matchups.dropna(subset=["adj_em_diff", "seed_diff"])
            if len(df) >= 10:
                self._lr_seed_em = LinearRegression().fit(
                    df[["seed_diff"]].fillna(0).to_numpy(),
                    df["adj_em_diff"].fillna(0).to_numpy(),
                )

    @classmethod
    def from_season_df(cls, season_df: pd.DataFrame, data_dir=None):
        """Build a TeamDB from an arbitrary season DataFrame (for backtesting).

        The DataFrame must have a ``kaggle_team_id`` column. All other columns
        are treated as team facts (adj_em, barthag, etc.).
        """
        obj = object.__new__(cls)
        data_dir = Path(data_dir) if data_dir else DATA_DIR

        obj._season = season_df.copy()
        obj._season["kaggle_team_id"] = pd.to_numeric(
            obj._season["kaggle_team_id"], errors="coerce"
        )

        teams_path = data_dir / "kaggle" / "MTeams.csv"
        obj._teams = pd.read_csv(teams_path) if teams_path.exists() else pd.DataFrame()

        obj._team_facts = {}
        for _, row in obj._season.iterrows():
            tid = row.get("kaggle_team_id")
            if pd.notna(tid):
                obj._team_facts[int(tid)] = row.to_dict()

        for facts in obj._team_facts.values():
            facts["win_pct"] = cls._parse_record(facts.get("record", ""))

        obj._seeds = {}
        obj._name_index = obj._build_name_index()

        matchup_path = data_dir / "matchup_dataset.csv"
        obj._matchups = (
            pd.read_csv(matchup_path, low_memory=False)
            if matchup_path.exists()
            else pd.DataFrame()
        )
        obj._lr_seed_em = None
        if (
            not obj._matchups.empty
            and "adj_em_diff" in obj._matchups.columns
            and "seed_diff" in obj._matchups.columns
        ):
            df = obj._matchups.dropna(subset=["adj_em_diff", "seed_diff"])
            if len(df) >= 10:
                obj._lr_seed_em = LinearRegression().fit(
                    df[["seed_diff"]].fillna(0).to_numpy(),
                    df["adj_em_diff"].fillna(0).to_numpy(),
                )
        return obj

    # ------------------------------------------------------------------
    # Seed helpers
    # ------------------------------------------------------------------

    def load_seeds(self, seeds_df: pd.DataFrame, season: int = 2026):
        """Populate team_id -> numeric seed mapping from MNCAATourneySeeds."""
        for _, row in seeds_df[seeds_df["Season"] == season].iterrows():
            team_id = int(row["TeamID"])
            self._seeds[team_id] = self._parse_seed(row["Seed"])

    def get_seed(self, team_id: int) -> float:
        return self._seeds.get(int(team_id), np.nan)

    # ------------------------------------------------------------------
    # Team lookups
    # ------------------------------------------------------------------

    def get_team(self, team_id: int) -> dict:
        return self._team_facts.get(int(team_id), {})

    def get_team_name(self, team_id: int) -> str:
        tid = int(team_id)
        facts = self._team_facts.get(tid)
        if facts:
            return (
                facts.get("kaggle_name")
                or facts.get("bart_name")
                or facts.get("team")
                or str(tid)
            )
        row = self._teams[self._teams["TeamID"] == tid]
        if len(row):
            return row.iloc[0]["TeamName"]
        return str(tid)

    def resolve_team(self, name: str) -> int | None:
        """
        Resolve a human-typed team name to a Kaggle team ID.

        Accepts the canonical name (``Duke``), any casing (``duke``),
        a slug (``st_john_s``), or common short-hands (``UConn``).
        Returns ``None`` when no match is found.
        """
        key = self._normalize(name)
        tid = self._name_index.get(key)
        if tid is not None:
            return tid
        return self._name_index.get(key.replace(" ", ""))

    def _build_name_index(self) -> dict[str, int]:
        """Map every reasonable spelling of every team name to its ID."""
        idx: dict[str, int] = {}

        def _add(raw: str, tid: int):
            spaced = self._normalize(raw)
            collapsed = spaced.replace(" ", "")
            idx[spaced] = tid
            idx[collapsed] = tid

        for tid, facts in self._team_facts.items():
            for field in ("kaggle_name", "bart_name", "team"):
                raw = facts.get(field)
                if raw and isinstance(raw, str):
                    _add(raw, tid)

        if not self._teams.empty:
            for _, row in self._teams.iterrows():
                _add(row["TeamName"], int(row["TeamID"]))

        _ALIASES = {
            "uconn": "connecticut",
            "nc st": "nc state",
            "unc": "north carolina",
            "miami": "miami fl",
            "niu": "northern iowa",
            "txam": "texas a m",
            "texas am": "texas a m",
        }
        for alias, canonical in _ALIASES.items():
            if canonical in idx:
                idx[alias] = idx[canonical]

        return idx

    @staticmethod
    def _normalize(name: str) -> str:
        """Collapse a name to a canonical lowercase key for matching."""
        s = str(name).lower().strip()
        s = re.sub(r"[^a-z0-9]+", " ", s).strip()
        return s

    # ------------------------------------------------------------------
    # Feature computation (mirrors build_matchup_dataset in notebook)
    # ------------------------------------------------------------------

    def compute_matchup_features(
        self, team_a_id: int, team_b_id: int, round_num: int = 1
    ) -> dict:
        """
        Build the same feature vector the notebook models were trained on.

        Convention: Team A = lower seed (favored), Team B = higher seed.
        ``round_num`` is the bracket round (1-6) used to derive ``is_late_round``.
        """
        a = self.get_team(team_a_id)
        b = self.get_team(team_b_id)

        a_seed = self.get_seed(team_a_id)
        b_seed = self.get_seed(team_b_id)

        features: dict[str, float] = {}

        # Seed features
        if np.isnan(a_seed) or np.isnan(b_seed):
            if np.isnan(a_seed):
                logger.warning("Missing seed for team %d, defaulting to 8", team_a_id)
            if np.isnan(b_seed):
                logger.warning("Missing seed for team %d, defaulting to 8", team_b_id)
            features["seed_diff"] = 0.0
            features["min_seed"] = 8.0
            features["is_big_gap"] = 0
        else:
            features["seed_diff"] = a_seed - b_seed
            features["min_seed"] = min(a_seed, b_seed)
            features["is_big_gap"] = int(abs(a_seed - b_seed) >= 8)

        features["is_late_round"] = int(round_num >= 3)

        # Barttorvik metric diffs
        for m in (
            "adj_o", "adj_d", "adj_em", "barthag", "adj_t", "wab",
            "off_efg", "def_efg", "off_to", "def_to",
            "off_or", "def_or", "off_ftr", "def_ftr",
            "fg2_pct", "fg3_pct",
        ):
            a_val = self._safe_float(a.get(m))
            b_val = self._safe_float(b.get(m))
            features[f"{m}_diff"] = a_val - b_val

        # Coach diffs
        for m in (
            "coach_appearances", "coach_tourn_wins",
            "coach_final_fours", "coach_win_rate",
        ):
            a_val = self._safe_float(a.get(m))
            b_val = self._safe_float(b.get(m))
            features[f"{m}_diff"] = a_val - b_val

        # Required by data/models/prob_model.pkl (must match notebook `make_2026_features`)
        features["ast_rate_diff"] = self._safe_float(a.get("ast_rate")) - self._safe_float(
            b.get("ast_rate")
        )
        adj_em_d = features.get("adj_em_diff", np.nan)
        if (
            self._lr_seed_em is not None
            and not np.isnan(a_seed)
            and not np.isnan(b_seed)
            and not np.isnan(adj_em_d)
        ):
            sd = float(a_seed - b_seed)
            pred_line = float(self._lr_seed_em.predict(np.array([[sd]]))[0])
            features["seed_disagreement"] = float(adj_em_d - pred_line)
        else:
            features["seed_disagreement"] = 0.0

        return features

    # ------------------------------------------------------------------
    # Historical averages (used by Seeding model)
    # ------------------------------------------------------------------

    def get_historical_seed_scores(self) -> dict[tuple[int, int], tuple[float, float]]:
        """Return (seed_a, seed_b) -> (avg_score_a, avg_score_b) from history."""
        if self._matchups.empty:
            return {}
        df = self._matchups.dropna(subset=["teamA_seed", "teamB_seed"]).copy()
        if "teamA_score" not in df.columns:
            return {}
        result: dict[tuple[int, int], tuple[float, float]] = {}
        for (sa, sb), grp in df.groupby(["teamA_seed", "teamB_seed"]):
            result[(int(sa), int(sb))] = (
                grp["teamA_score"].mean(),
                grp["teamB_score"].mean(),
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_seed(seed_str) -> float:
        if pd.isna(seed_str):
            return np.nan
        m = re.search(r"(\d+)", str(seed_str))
        return float(int(m.group(1))) if m else np.nan

    @staticmethod
    def _parse_record(record_str) -> float:
        if pd.isna(record_str) or not record_str:
            return 0.5
        parts = str(record_str).split("-")
        if len(parts) != 2:
            return 0.5
        try:
            w, l = int(parts[0]), int(parts[1])
            return w / (w + l) if (w + l) > 0 else 0.5
        except ValueError:
            return 0.5

    @staticmethod
    def _safe_float(val, default: float = 0.0) -> float:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default
