"""Bracket tree structure and simulation engine."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from engine.db import TeamDB
from engine.models.base import PredictionModel


ROUND_LABELS = {
    0: "First Four",
    1: "R64",
    2: "R32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}


@dataclass
class BracketSlot:
    slot_id: str
    round_num: int
    region: str
    strong_source: str
    weak_source: str
    strong_team_id: Optional[int] = None
    weak_team_id: Optional[int] = None
    winner_team_id: Optional[int] = None
    pred_winner_id: Optional[int] = None
    strong_pred_score: Optional[float] = None
    weak_pred_score: Optional[float] = None
    is_actual: bool = False
    actual_winner_id: Optional[int] = None
    actual_strong_score: Optional[float] = None
    actual_weak_score: Optional[float] = None


class Bracket:
    """
    Full tournament bracket backed by the Kaggle slot/seed structure.

    Workflow
    --------
    1. ``Bracket(seeds_df, slots_df)``  -- build from data
    2. ``simulate(model, db)``          -- fill every slot with predictions
    3. ``inject_actuals(results_df)``   -- lock in real outcomes
    4. ``simulate(model, db)``          -- re-predict remaining games
    5. ``to_dataframe()``               -- export for eval / display
    """

    def __init__(
        self,
        seeds_df: pd.DataFrame,
        slots_df: pd.DataFrame,
        season: int = 2026,
        template_season: Optional[int] = None,
    ):
        self.season = season

        # ── seed string -> team_id  (e.g. "W01" -> 1181) ──
        self.seed_to_team: dict[str, int] = {}
        for _, row in seeds_df[seeds_df["Season"] == season].iterrows():
            self.seed_to_team[row["Seed"]] = int(row["TeamID"])

        # ── detect play-in positions from seeds (any position with a/b variants) ──
        playin_bases: set[str] = set()
        for seed_str in self.seed_to_team:
            if seed_str.endswith("a"):
                playin_bases.add(seed_str[:-1])

        # ── build R1–R6 slots from template (ignore template play-ins) ──
        if template_season is None:
            available = slots_df["Season"].unique()
            template_season = season if season in available else int(max(available))

        self.slots: dict[str, BracketSlot] = {}
        tmpl = slots_df[slots_df["Season"] == template_season]
        for _, row in tmpl.iterrows():
            sid = row["Slot"]
            if self._parse_round(sid) > 0:
                self.slots[sid] = BracketSlot(
                    slot_id=sid,
                    round_num=self._parse_round(sid),
                    region=self._parse_region(sid),
                    strong_source=row["StrongSeed"],
                    weak_source=row["WeakSeed"],
                )

        # ── create play-in slots from this season's actual seeds ──
        for base in sorted(playin_bases):
            self.slots[base] = BracketSlot(
                slot_id=base,
                round_num=0,
                region=base[0],
                strong_source=f"{base}a",
                weak_source=f"{base}b",
            )

        self._populate_known_teams()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(self, model: PredictionModel, db: TeamDB) -> None:
        """Run *model* through every unresolved slot, round by round."""
        for rnd in sorted({s.round_num for s in self.slots.values()}):
            for slot in self._round_slots(rnd):
                if slot.is_actual:
                    continue

                self._resolve_teams(slot)

                if slot.strong_team_id is None or slot.weak_team_id is None:
                    continue

                team_a_id, team_b_id = self._order_by_seed(
                    slot.strong_team_id, slot.weak_team_id, db
                )
                pred = model.predict(team_a_id, team_b_id, db, round_num=slot.round_num)

                if team_a_id == slot.strong_team_id:
                    slot.strong_pred_score = pred["team_a_score"]
                    slot.weak_pred_score = pred["team_b_score"]
                else:
                    slot.strong_pred_score = pred["team_b_score"]
                    slot.weak_pred_score = pred["team_a_score"]

                slot.pred_winner_id = pred["winner_id"]
                slot.winner_team_id = pred["winner_id"]

    def inject_actuals(self, results_df: pd.DataFrame) -> None:
        """
        Lock in real results.  ``results_df`` columns:

        - ``slot_id``: matches Kaggle slot IDs
        - ``winner_team_id``
        - ``winner_score``  (optional)
        - ``loser_score``   (optional)
        """
        for _, row in results_df.iterrows():
            sid = row["slot_id"]
            if sid not in self.slots:
                continue
            slot = self.slots[sid]
            winner = int(row["winner_team_id"])

            self._resolve_teams(slot)

            slot.actual_winner_id = winner
            slot.winner_team_id = winner
            slot.is_actual = True

            if "winner_score" in row and pd.notna(row["winner_score"]):
                w_score = float(row["winner_score"])
                l_score = float(row.get("loser_score", 0))
                if winner == slot.strong_team_id:
                    slot.actual_strong_score = w_score
                    slot.actual_weak_score = l_score
                else:
                    slot.actual_strong_score = l_score
                    slot.actual_weak_score = w_score

        max_actual_round = max(
            (s.round_num for s in self.slots.values() if s.is_actual),
            default=-1,
        )
        self.reset_from_round(max_actual_round + 1)

    def reset_from_round(self, from_round: int) -> None:
        """Clear predictions for round ``from_round`` and all later rounds."""
        for slot in self.slots.values():
            if slot.round_num >= from_round and not slot.is_actual:
                slot.winner_team_id = None
                slot.strong_pred_score = None
                slot.weak_pred_score = None
                if slot.round_num > 0:
                    slot.strong_team_id = None
                    slot.weak_team_id = None

    def to_dataframe(self, db: TeamDB) -> pd.DataFrame:
        """Export every slot as a flat table row."""
        rows = []
        for slot in sorted(self.slots.values(), key=lambda s: (s.round_num, s.slot_id)):
            strong_name = db.get_team_name(slot.strong_team_id) if slot.strong_team_id else ""
            weak_name = db.get_team_name(slot.weak_team_id) if slot.weak_team_id else ""
            winner_name = db.get_team_name(slot.winner_team_id) if slot.winner_team_id else ""

            strong_seed = db.get_seed(slot.strong_team_id) if slot.strong_team_id else np.nan
            weak_seed = db.get_seed(slot.weak_team_id) if slot.weak_team_id else np.nan

            pred_winner_name = db.get_team_name(slot.pred_winner_id) if slot.pred_winner_id else ""
            actual_winner_name = db.get_team_name(slot.actual_winner_id) if slot.actual_winner_id else ""

            rows.append(
                {
                    "slot_id": slot.slot_id,
                    "round_num": slot.round_num,
                    "round_label": ROUND_LABELS.get(slot.round_num, f"R{slot.round_num}"),
                    "region": slot.region,
                    "strong_team_id": slot.strong_team_id,
                    "strong_team": strong_name,
                    "strong_seed": strong_seed,
                    "weak_team_id": slot.weak_team_id,
                    "weak_team": weak_name,
                    "weak_seed": weak_seed,
                    "strong_pred_score": slot.strong_pred_score,
                    "weak_pred_score": slot.weak_pred_score,
                    "pred_winner": pred_winner_name,
                    "pred_winner_id": slot.pred_winner_id,
                    "winner": winner_name,
                    "winner_id": slot.winner_team_id,
                    "is_actual": slot.is_actual,
                    "actual_winner": actual_winner_name,
                    "actual_winner_id": slot.actual_winner_id,
                    "actual_strong_score": slot.actual_strong_score,
                    "actual_weak_score": slot.actual_weak_score,
                }
            )
        return pd.DataFrame(rows)

    def get_champion(self, db: TeamDB) -> str | None:
        """Return the name of the predicted champion, if resolved."""
        for slot in self.slots.values():
            if slot.round_num == 6 and slot.winner_team_id:
                return db.get_team_name(slot.winner_team_id)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _populate_known_teams(self) -> None:
        """Fill team IDs for play-in and R1 slots using the seed map."""
        for slot in self.slots.values():
            if slot.round_num == 0:
                slot.strong_team_id = self.seed_to_team.get(slot.strong_source)
                slot.weak_team_id = self.seed_to_team.get(slot.weak_source)
            elif slot.round_num == 1:
                if slot.strong_source not in self.slots:
                    slot.strong_team_id = self.seed_to_team.get(slot.strong_source)
                if slot.weak_source not in self.slots:
                    slot.weak_team_id = self.seed_to_team.get(slot.weak_source)

    def _resolve_teams(self, slot: BracketSlot) -> None:
        """Try to fill strong/weak team IDs from upstream winners."""
        if slot.strong_team_id is None and slot.strong_source in self.slots:
            upstream = self.slots[slot.strong_source]
            if upstream.winner_team_id is not None:
                slot.strong_team_id = upstream.winner_team_id

        if slot.weak_team_id is None and slot.weak_source in self.slots:
            upstream = self.slots[slot.weak_source]
            if upstream.winner_team_id is not None:
                slot.weak_team_id = upstream.winner_team_id

    def _round_slots(self, rnd: int) -> list[BracketSlot]:
        return sorted(
            [s for s in self.slots.values() if s.round_num == rnd],
            key=lambda s: s.slot_id,
        )

    @staticmethod
    def _order_by_seed(id_a: int, id_b: int, db: TeamDB) -> tuple[int, int]:
        """Return (favored, underdog) so Team A = lower seed per training convention."""
        sa, sb = db.get_seed(id_a), db.get_seed(id_b)
        if np.isnan(sa) or np.isnan(sb):
            return id_a, id_b
        return (id_a, id_b) if sa <= sb else (id_b, id_a)

    @staticmethod
    def _parse_round(slot_id: str) -> int:
        if slot_id.startswith("R"):
            return int(slot_id[1])
        return 0

    @staticmethod
    def _parse_region(slot_id: str) -> str:
        if slot_id.startswith("R"):
            return re.sub(r"\d", "", slot_id[2:])
        return slot_id[0]
