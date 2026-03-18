"""Load human-friendly actuals CSVs and convert to engine format."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from engine.bracket import Bracket, ROUND_LABELS
from engine.db import TeamDB


ROUND_ALIASES: dict[str, int] = {}
for _num, _label in ROUND_LABELS.items():
    low = _label.lower()
    ROUND_ALIASES[low] = _num
    ROUND_ALIASES[re.sub(r"[^a-z0-9]", "", low)] = _num
    ROUND_ALIASES[re.sub(r"\s+", "_", low)] = _num
    ROUND_ALIASES[str(_num)] = _num

_EXTRA = {
    "play_in": 0, "playin": 0, "first_four": 0, "first four": 0, "ff": 0,
    "64": 1, "r64": 1, "round_of_64": 1, "round of 64": 1,
    "32": 2, "r32": 2, "round_of_32": 2, "round of 32": 2,
    "sweet_16": 3, "sweet 16": 3, "s16": 3, "16": 3,
    "elite_8": 4, "elite 8": 4, "e8": 4, "8": 4,
    "final_four": 5, "final four": 5, "f4": 5, "final4": 5,
    "championship": 6, "ch": 6, "title": 6, "final": 6, "finals": 6,
}
ROUND_ALIASES.update(_EXTRA)


def parse_round(raw: str) -> int | None:
    key = str(raw).strip().lower()
    return ROUND_ALIASES.get(key)


def load_actuals(
    path: str | Path,
    bracket: Bracket,
    db: TeamDB,
) -> pd.DataFrame:
    """
    Read a human-friendly actuals CSV and return a DataFrame ready for
    ``bracket.inject_actuals()``.

    Expected CSV columns
    --------------------
    - ``round``  : human name (e.g. ``R64``, ``Sweet 16``, ``Final Four``)
    - ``winner`` : team name  (e.g. ``Duke``, ``st johns``, ``Michigan St``)
    - ``winner_score`` : (optional) winning team's final score
    - ``loser_score``  : (optional) losing team's final score

    Returns
    -------
    DataFrame with columns ``slot_id, winner_team_id, winner_score, loser_score``
    that ``Bracket.inject_actuals()`` consumes directly.
    """
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

    # Resolve round numbers up front and sort so earlier rounds are processed
    # first.  This ensures that when actuals span multiple rounds (e.g. R64 +
    # R32), play-in / R64 winners are injected before we look up R32 slots.
    parsed: list[tuple[int, int, pd.Series]] = []  # (round_num, csv_line, row)
    errors: list[str] = []

    for i, row in raw.iterrows():
        line = i + 2  # 1-indexed + header
        round_num = parse_round(row.get("round", ""))
        if round_num is None:
            errors.append(f"Line {line}: unrecognized round '{row.get('round')}'")
        else:
            parsed.append((round_num, line, row))

    parsed.sort(key=lambda t: t[0])

    rows: list[dict] = []
    prev_round: int | None = None

    for round_num, line, row in parsed:
        # When we move to a later round, inject everything resolved so far
        # and propagate winners so the next round's slots have teams.
        if prev_round is not None and round_num > prev_round and rows:
            bracket.inject_actuals(pd.DataFrame(rows))
            for s in bracket.slots.values():
                if s.round_num == round_num:
                    bracket._resolve_teams(s)
        prev_round = round_num

        # --- resolve winner ---
        winner_raw = str(row.get("winner", "")).strip()
        winner_id = db.resolve_team(winner_raw)
        if winner_id is None:
            errors.append(
                f"Line {line}: cannot find team '{winner_raw}'. "
                "Check spelling against the bracket output."
            )
            continue

        # --- find the slot that has this team in this round ---
        slot = _find_slot(bracket, round_num, winner_id)
        if slot is None:
            winner_name = db.get_team_name(winner_id)
            errors.append(
                f"Line {line}: no slot in {ROUND_LABELS.get(round_num, f'R{round_num}')} "
                f"contains {winner_name} ({winner_id}). "
                "Was an earlier round's result missing?"
            )
            continue

        entry: dict = {
            "slot_id": slot.slot_id,
            "winner_team_id": winner_id,
        }
        if "winner_score" in row and pd.notna(row.get("winner_score")):
            entry["winner_score"] = float(row["winner_score"])
        if "loser_score" in row and pd.notna(row.get("loser_score")):
            entry["loser_score"] = float(row["loser_score"])

        rows.append(entry)

    if errors:
        print(f"\n{'!'*60}")
        print(f"  {len(errors)} problem(s) while loading actuals:")
        print(f"{'!'*60}")
        for e in errors:
            print(f"  - {e}")
        print()

    result = pd.DataFrame(rows)
    ok = len(rows)
    bad = len(errors)
    print(f"Loaded {ok} actual result(s) from {Path(path).name}" +
          (f" ({bad} skipped)" if bad else ""))

    return result


def _find_slot(bracket: Bracket, round_num: int, team_id: int):
    """Find the BracketSlot in *round_num* where *team_id* is competing."""
    for slot in bracket.slots.values():
        if slot.round_num != round_num:
            continue
        if slot.strong_team_id == team_id or slot.weak_team_id == team_id:
            return slot
    return None
