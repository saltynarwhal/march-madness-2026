"""Animal Kingdom model – Claude judges mascot fights."""

from __future__ import annotations

import json
import os
from pathlib import Path

from engine.db import TeamDB
from engine.models.base import Prediction, PredictionModel

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CACHE_PATH = DATA_DIR / "cache" / "animal_kingdom.json"

SYSTEM_PROMPT = (
    "You are a nature documentary narrator and combat analyst. "
    "Given two NCAA team mascots, judge who would win in a hypothetical fight. "
    "Respond ONLY with valid JSON (no markdown fencing) matching this schema:\n"
    '{"team_a_mascot": str, "team_b_mascot": str, '
    '"team_a_score": int, "team_b_score": int, "reasoning": str}\n'
    "Scores must be integers between 50 and 100. "
    "The higher score wins. Scores should never be equal."
)


def _build_user_prompt(name_a: str, name_b: str) -> str:
    return (
        f"Team A: {name_a}\nTeam B: {name_b}\n\n"
        "Identify each team's mascot, then score the fight 50-100."
    )


class AnimalKingdomModel(PredictionModel):
    name = "Animal Kingdom"

    def __init__(
        self,
        api_key: str | None = None,
        cache_path: Path | str | None = None,
        model_name: str = "claude-sonnet-4-20250514",
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._cache_path = Path(cache_path) if cache_path else CACHE_PATH
        self._model_name = model_name
        self._cache = self._load_cache()
        self._client = None

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

        cache_key = self._cache_key(name_a, name_b)
        if cache_key in self._cache:
            return self._cache_to_prediction(self._cache[cache_key], team_a_id, team_b_id)

        result = self._call_claude(name_a, name_b)
        self._cache[cache_key] = result
        self._save_cache()

        return self._cache_to_prediction(result, team_a_id, team_b_id)

    # ------------------------------------------------------------------
    # Claude interaction
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _call_claude(self, name_a: str, name_b: str) -> dict:
        client = self._get_client()
        message = client.messages.create(
            model=self._model_name,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(name_a, name_b)}],
        )
        raw = message.content[0].text.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {
                "team_a_mascot": name_a,
                "team_b_mascot": name_b,
                "team_a_score": 75,
                "team_b_score": 75,
                "reasoning": f"Parse error – raw: {raw[:200]}",
            }
        return data

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
    def _cache_to_prediction(data: dict, team_a_id: int, team_b_id: int) -> Prediction:
        sa = int(data.get("team_a_score", 75))
        sb = int(data.get("team_b_score", 75))
        sa = max(50, min(100, sa))
        sb = max(50, min(100, sb))
        if sa == sb:
            sa += 1

        winner = team_a_id if sa > sb else team_b_id
        spread = abs(sa - sb)
        confidence = min(spread / 50.0, 1.0) * 0.5 + 0.5

        return Prediction(
            team_a_score=float(sa),
            team_b_score=float(sb),
            winner_id=winner,
            confidence=round(confidence, 3),
        )
