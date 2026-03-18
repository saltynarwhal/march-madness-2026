"""Vegas Odds model – real betting lines with AI-estimated fallback."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from engine.db import TeamDB
from engine.models.base import Prediction, PredictionModel

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
LINES_PATH = DATA_DIR / "vegas_lines.csv"
CACHE_PATH = DATA_DIR / "cache" / "vegas_estimated.json"

SYSTEM_PROMPT = (
    "You are an experienced Las Vegas sports-book odds compiler specializing in "
    "NCAA March Madness. Given two teams, produce the betting line a major "
    "sportsbook would post. Respond ONLY with valid JSON (no markdown fencing) "
    "matching this schema:\n"
    '{"favored": str, "spread": float, "total": float, "reasoning": str}\n\n'
    "- 'favored' is the team name you expect to be favored.\n"
    "- 'spread' is the point-spread magnitude (always positive, e.g. 8.5).\n"
    "- 'total' is the over/under combined score (e.g. 143.5).\n"
    "- 'reasoning' is a one-sentence rationale.\n"
    "Use your knowledge of the 2025-26 season, team strength, and typical "
    "March Madness line-making to produce realistic numbers."
)


def _build_user_prompt(name_a: str, seed_a, name_b: str, seed_b) -> str:
    sa = f"({int(seed_a)} seed)" if not np.isnan(seed_a) else ""
    sb = f"({int(seed_b)} seed)" if not np.isnan(seed_b) else ""
    return (
        f"Team A: {name_a} {sa}\n"
        f"Team B: {name_b} {sb}\n\n"
        "Produce the Vegas line for this NCAA Tournament matchup."
    )


class VegasOddsModel(PredictionModel):
    """
    Prediction from Vegas spread + total.

    Looks for real lines in ``data/vegas_lines.csv`` first.  When a matchup
    isn't found there, falls back to an AI estimate via Claude and flags the
    prediction as *estimated*.  All AI estimates are cached so the API is
    only called once per unique matchup.
    """

    name = "Vegas Odds"

    def __init__(
        self,
        lines_path: Path | str | None = None,
        api_key: str | None = None,
        cache_path: Path | str | None = None,
        model_name: str = "claude-sonnet-4-20250514",
    ):
        self._lines_path = Path(lines_path) if lines_path else LINES_PATH
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._cache_path = Path(cache_path) if cache_path else CACHE_PATH
        self._model_name = model_name
        self._lines = self._load_lines()
        self._cache = self._load_cache()
        self._client = None

        self.estimated_count = 0
        self.real_count = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def predict(
        self,
        team_a_id: int,
        team_b_id: int,
        db: TeamDB,
        round_num: int = 1,
    ) -> Prediction:
        name_a = db.get_team_name(team_a_id)
        name_b = db.get_team_name(team_b_id)

        line = self._lookup_line(name_a, name_b)
        if line is not None:
            self.real_count += 1
            return self._line_to_prediction(
                line, name_a, name_b, team_a_id, team_b_id, estimated=False
            )

        cache_key = self._cache_key(name_a, name_b)
        if cache_key in self._cache:
            self.estimated_count += 1
            return self._line_to_prediction(
                self._cache[cache_key], name_a, name_b,
                team_a_id, team_b_id, estimated=True,
            )

        if not self._api_key:
            return self._no_line_fallback(team_a_id, team_b_id, db)

        result = self._call_claude(name_a, name_b, db, team_a_id, team_b_id)
        self._cache[cache_key] = result
        self._save_cache()
        self.estimated_count += 1

        return self._line_to_prediction(
            result, name_a, name_b, team_a_id, team_b_id, estimated=True
        )

    # ------------------------------------------------------------------
    # Line lookup (real Vegas data)
    # ------------------------------------------------------------------

    def _load_lines(self) -> pd.DataFrame:
        """
        Load ``data/vegas_lines.csv``.  Expected columns:

        - ``team_a`` – name matching Kaggle / Barttorvik (e.g. "Duke")
        - ``team_b`` – opponent name
        - ``spread`` – from Team A perspective (negative = A favored)
        - ``total``  – over/under combined score
        - ``source`` – (optional) e.g. "DraftKings", "FanDuel"
        """
        if self._lines_path.exists():
            return pd.read_csv(self._lines_path)
        return pd.DataFrame()

    def _lookup_line(self, name_a: str, name_b: str) -> dict | None:
        if self._lines.empty:
            return None
        low_a, low_b = name_a.lower(), name_b.lower()
        for _, row in self._lines.iterrows():
            ra = str(row.get("team_a", "")).lower()
            rb = str(row.get("team_b", "")).lower()
            if (ra == low_a and rb == low_b) or (ra == low_b and rb == low_a):
                spread = float(row["spread"])
                if ra == low_b:
                    spread = -spread
                return {
                    "favored": name_a if spread < 0 else name_b,
                    "spread": abs(spread),
                    "total": float(row["total"]),
                    "source": row.get("source", "manual"),
                }
        return None

    # ------------------------------------------------------------------
    # AI fallback (Claude)
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _call_claude(
        self, name_a: str, name_b: str, db: TeamDB,
        team_a_id: int, team_b_id: int,
    ) -> dict:
        seed_a = db.get_seed(team_a_id)
        seed_b = db.get_seed(team_b_id)
        client = self._get_client()
        message = client.messages.create(
            model=self._model_name,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": _build_user_prompt(name_a, seed_a, name_b, seed_b),
            }],
        )
        raw = message.content[0].text.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {
                "favored": name_a,
                "spread": 3.0,
                "total": 140.0,
                "reasoning": f"Parse error – raw: {raw[:200]}",
            }
        return data

    # ------------------------------------------------------------------
    # Fallback when no API key and no CSV line
    # ------------------------------------------------------------------

    @staticmethod
    def _no_line_fallback(
        team_a_id: int, team_b_id: int, db: TeamDB
    ) -> Prediction:
        """Seed-based placeholder when neither real lines nor AI are available."""
        seed_a = db.get_seed(team_a_id)
        seed_b = db.get_seed(team_b_id)
        sa = seed_a if not np.isnan(seed_a) else 8
        sb = seed_b if not np.isnan(seed_b) else 8
        gap = sa - sb
        total = 140.0
        margin = gap * 1.5
        score_a = (total - margin) / 2
        score_b = (total + margin) / 2
        winner = team_a_id if score_a > score_b else team_b_id
        return Prediction(
            team_a_score=round(score_a, 1),
            team_b_score=round(score_b, 1),
            winner_id=winner,
            confidence=0.5,
        )

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(name_a: str, name_b: str) -> str:
        return f"{name_a} vs {name_b}"

    def _load_cache(self) -> dict:
        if self._cache_path.exists():
            with open(self._cache_path) as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _line_to_prediction(
        line: dict,
        name_a: str,
        name_b: str,
        team_a_id: int,
        team_b_id: int,
        estimated: bool = False,
    ) -> Prediction:
        favored = line.get("favored", name_a)
        spread = float(line.get("spread", 3.0))
        total = float(line.get("total", 140.0))

        if favored.lower() == name_a.lower():
            margin = spread
            winner = team_a_id
        else:
            margin = -spread
            winner = team_b_id

        score_a = (total + margin) / 2
        score_b = (total - margin) / 2

        score_a = max(score_a, 40.0)
        score_b = max(score_b, 40.0)

        confidence = min(spread / 25.0, 1.0) * 0.5 + 0.5

        return Prediction(
            team_a_score=round(score_a, 1),
            team_b_score=round(score_b, 1),
            winner_id=winner,
            confidence=round(confidence, 3),
        )
