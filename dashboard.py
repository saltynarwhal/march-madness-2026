"""
March Madness 2026 — Bracket Prediction Dashboard
Run with:  streamlit run dashboard.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_root = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(_root / ".env")
except ImportError:
    pass  # Streamlit Cloud sets env vars via its Secrets manager

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(_root))

from engine.db import TeamDB
from engine.bracket import Bracket, ROUND_LABELS
from engine.models.seeding import SeedingModel
from engine.evaluation import (
    accuracy_table,
    calibration_summary,
    games_graded_count,
    merge_tournament_results_into_bracket_dfs,
    overall_pick_accuracy,
    spread_accuracy_table,
    truth_dataframe_from_tournament_csv,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="March Madness 2026",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"

REGION_NAMES = {"W": "East", "X": "South", "Y": "Midwest", "Z": "West"}

MODEL_COLORS = {
    "Comparative Metrics": "#2563eb",
    "Animal Kingdom": "#dc2626",
    "Vegas Odds": "#059669",
    "Seeding Only": "#7c3aed",
    "Greg_v1": "#f59e0b",
}

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading engine…")
def load_engine():
    db = TeamDB(str(DATA_DIR))
    seeds_df = pd.read_csv(DATA_DIR / "kaggle" / "MNCAATourneySeeds.csv")
    db.load_seeds(seeds_df, season=2026)
    slots_df = pd.read_csv(DATA_DIR / "kaggle" / "MNCAATourneySlots.csv")
    return db, seeds_df, slots_df


def build_models(db: TeamDB):
    models: dict = {}
    models["Seeding Only"] = SeedingModel(db)
    try:
        from engine.models.advanced_metrics import AdvancedMetricsModel
        models["Comparative Metrics"] = AdvancedMetricsModel(str(DATA_DIR / "models"))
    except Exception as exc:
        logger.warning("Failed to load Comparative Metrics model: %s", exc)
    try:
        from engine.models.greg_v1 import GregV1Model
        models["Greg_v1"] = GregV1Model(str(DATA_DIR / "models"))
    except Exception as exc:
        logger.warning("Failed to load Greg_v1 model: %s", exc)
    # Probability-based models (require prob_model.pkl artifacts)
    try:
        from engine.models.probability import (
            SampledProbabilityModel,
            ThresholdProbabilityModel,
            MonteCarloConsensusModel,
        )
        models["Lean GB (Sampled)"] = SampledProbabilityModel(
            str(DATA_DIR / "models"),
            random_seed=12345,
        )
        models["Lean GB (Tiered Threshold)"] = ThresholdProbabilityModel(
            str(DATA_DIR / "models"),
        )
        models["Lean GB (MC Consensus)"] = MonteCarloConsensusModel(
            str(DATA_DIR / "models"),
            n_sims=10_000,
            random_seed=12345,
        )
    except Exception as exc:
        logger.warning("Failed to load probability models: %s", exc)
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if has_api_key:
        try:
            from engine.models.animal_kingdom import AnimalKingdomModel
            models["Animal Kingdom"] = AnimalKingdomModel()
        except Exception as exc:
            logger.warning("Failed to load Animal Kingdom model: %s", exc)
        try:
            from engine.models.vegas_odds import VegasOddsModel
            models["Vegas Odds"] = VegasOddsModel(
                lines_path=DATA_DIR / "vegas_lines.csv",
            )
        except Exception as exc:
            logger.warning("Failed to load Vegas Odds model: %s", exc)
    return models


_CACHE_VERSION = 8  # bump to invalidate cached bracket dataframes

@st.cache_data(show_spinner="Simulating brackets…")
def simulate_all(_db, _seeds_df, _slots_df, actuals_csv: str | None, _v=_CACHE_VERSION):
    models = build_models(_db)
    brackets = {}
    bracket_dfs = {}
    skipped: list[str] = []
    for name, model in models.items():
        try:
            b = Bracket(_seeds_df, _slots_df, season=2026)
            # Apply actuals (including First Four) before any simulation so the model never
            # "predicts" games you already know. load_actuals() calls inject_actuals internally.
            if actuals_csv:
                from engine.actuals import load_actuals
                load_actuals(actuals_csv, b, _db)
            b.simulate(model, _db)
            brackets[name] = b
            bracket_dfs[name] = b.to_dataframe(_db)
        except Exception as exc:
            skipped.append(f"{name}: {exc}")

    # Full tournament results (post-tournament): merge truth onto predictions for UI + accuracy.
    tournament_path = DATA_DIR / "tournament_results.csv"
    if tournament_path.exists():
        truth_df = truth_dataframe_from_tournament_csv(
            tournament_path, _seeds_df, _slots_df, _db
        )
        if not truth_df.empty:
            bracket_dfs = merge_tournament_results_into_bracket_dfs(bracket_dfs, truth_df)

    champions = {}
    for name, b in brackets.items():
        champions[name] = b.get_champion(_db)

    return bracket_dfs, champions, skipped


# ---------------------------------------------------------------------------
# Bracket rendering (HTML/CSS)
# ---------------------------------------------------------------------------

def _team_cell(name: str, seed, score, is_winner: bool, is_actual: bool,
               win_pct: str = "", *, result_line: bool = False) -> str:
    seed_int = int(seed) if pd.notna(seed) else "?"
    score_str = f"{score:.0f}" if pd.notna(score) else ""
    cls = "team-winner" if is_winner else "team-loser"
    if is_actual:
        cls += " actual"
    if result_line:
        cls += " result-line"
    pct_html = f'<span class="win-pct">{win_pct}</span>' if win_pct else ""
    return (
        f'<div class="team-row {cls}">'
        f'<span class="seed">{seed_int}</span>'
        f'<span class="team-name">{name}</span>'
        f'{pct_html}'
        f'<span class="score">{score_str}</span>'
        f'</div>'
    )


def _game_card(row: pd.Series) -> str:
    has_tournament_result = (
        "result_winner_id" in row.index
        and pd.notna(row.get("result_winner_id"))
    )
    is_actual = bool(row.get("is_actual", False))

    # --- Prediction block (model pick + predicted scores) ---
    pred_id = row.get("pred_winner_id")
    if pd.isna(pred_id) and pd.notna(row.get("winner_id")):
        pred_id = row.get("winner_id")
    s_winner = pred_id == row["strong_team_id"] if pd.notna(pred_id) else False
    w_winner = pred_id == row["weak_team_id"] if pd.notna(pred_id) else False
    s_score = row.get("strong_pred_score")
    w_score = row.get("weak_pred_score")

    s_pct = w_pct = ""
    if not is_actual or has_tournament_result:
        conf = row.get("confidence")
        if pd.notna(conf) and conf is not None:
            conf = float(conf)
            if pred_id == row.get("strong_team_id"):
                s_pct = f"{conf * 100:.0f}%"
                w_pct = f"{(1 - conf) * 100:.0f}%"
            elif pred_id == row.get("weak_team_id"):
                w_pct = f"{conf * 100:.0f}%"
                s_pct = f"{(1 - conf) * 100:.0f}%"

    pred_top = _team_cell(
        row["strong_team"] or "TBD", row.get("strong_seed"), s_score, s_winner, False, s_pct
    )
    pred_bot = _team_cell(
        row["weak_team"] or "TBD", row.get("weak_seed"), w_score, w_winner, False, w_pct
    )

    if not has_tournament_result:
        badge = ""
        if is_actual and pd.notna(row.get("actual_strong_score")):
            badge = '<span class="badge-actual">FINAL</span>'
            s_score = row["actual_strong_score"]
            w_score = row["actual_weak_score"]
            s_winner = row["winner_id"] == row["strong_team_id"] if pd.notna(row.get("winner_id")) else False
            w_winner = row["winner_id"] == row["weak_team_id"] if pd.notna(row.get("winner_id")) else False
            pred_top = _team_cell(row["strong_team"] or "TBD", row.get("strong_seed"), s_score, s_winner, True, "")
            pred_bot = _team_cell(row["weak_team"] or "TBD", row.get("weak_seed"), w_score, w_winner, True, "")
        return f'<div class="game-card">{badge}{pred_top}{pred_bot}</div>'

    # --- Actual tournament result (underneath predictions) ---
    rw = row.get("result_winner_id")
    s_win_r = rw == row["strong_team_id"] if pd.notna(rw) else False
    w_win_r = rw == row["weak_team_id"] if pd.notna(rw) else False
    rs_s = row.get("result_strong_score")
    rs_w = row.get("result_weak_score")
    res_top = _team_cell(
        row["strong_team"] or "TBD", row.get("strong_seed"), rs_s, s_win_r, False, "", result_line=True
    )
    res_bot = _team_cell(
        row["weak_team"] or "TBD", row.get("weak_seed"), rs_w, w_win_r, False, "", result_line=True
    )
    try:
        pick_ok = pd.notna(pred_id) and pd.notna(rw) and int(pred_id) == int(rw)
    except (TypeError, ValueError):
        pick_ok = False
    tag = "✓" if pick_ok else "✗"
    badge = f'<span class="badge-actual">ACTUAL {tag}</span>'

    return (
        f'<div class="game-card game-card-split">'
        f'<div class="game-split-label">Predicted</div>'
        f'<div class="game-card-pred">{pred_top}{pred_bot}</div>'
        f'<div class="game-split-label">Tournament result</div>'
        f'<div class="game-card-result">{badge}{res_top}{res_bot}</div>'
        f"</div>"
    )


# Kaggle slot order for *visual* bracket tree (not lexicographic slot_id sort).
# R1: pairings feed R2 as (W1+W8)->R2W1, (W2+W7)->R2W2, (W3+W6)->R2W3, (W4+W5)->R2W4.
_R1_VISUAL_SUFFIXES = [1, 8, 2, 7, 3, 6, 4, 5]
_R2_VISUAL_SUFFIXES = [1, 2, 3, 4]
_R3_VISUAL_SUFFIXES = [1, 2]
_R4_VISUAL_SUFFIXES = [1]


def _series_for_slot(df: pd.DataFrame, region: str, slot_id: str) -> pd.Series | None:
    sub = df[(df["region"] == region) & (df["slot_id"] == slot_id)]
    if sub.empty:
        return None
    return sub.iloc[0]


def render_region_bracket(df: pd.DataFrame, region: str, direction: str = "ltr") -> str:
    """Build HTML for one region's bracket (R64→R32→S16→E8) with proper tree alignment."""
    dir_cls = "dir-rtl" if direction == "rtl" else ""
    # Left-side regions: R64 … E8 left→right. Right-side (rtl): E8 … R64 so Elite 8 sits by center.
    if direction == "rtl":
        r64_c, r32_c, s16_c, e8_c = 4, 3, 2, 1
    else:
        r64_c, r32_c, s16_c, e8_c = 1, 2, 3, 4

    labels = (
        f'<div class="lbl" style="grid-column:{r64_c}">{ROUND_LABELS.get(1, "R64")}</div>'
        f'<div class="lbl" style="grid-column:{r32_c}">{ROUND_LABELS.get(2, "R32")}</div>'
        f'<div class="lbl" style="grid-column:{s16_c}">{ROUND_LABELS.get(3, "Sweet 16")}</div>'
        f'<div class="lbl" style="grid-column:{e8_c}">{ROUND_LABELS.get(4, "Elite 8")}</div>'
    )

    cells: list[str] = []

    for i, suf in enumerate(_R1_VISUAL_SUFFIXES):
        sid = f"R1{region}{suf}"
        row = _series_for_slot(df, region, sid)
        card = _game_card(row) if row is not None else '<div class="game-card game-card-missing"></div>'
        gr = i + 1
        pair = "top" if i % 2 == 0 else "bottom"
        cells.append(
            f'<div class="grid-bracket-slot" data-round="1" data-pair="{pair}" '
            f'style="grid-column:{r64_c};grid-row:{gr}">{card}</div>'
        )

    for j, suf in enumerate(_R2_VISUAL_SUFFIXES):
        sid = f"R2{region}{suf}"
        row = _series_for_slot(df, region, sid)
        card = _game_card(row) if row is not None else '<div class="game-card game-card-missing"></div>'
        r0 = j * 2 + 1
        pair = "top" if j % 2 == 0 else "bottom"
        cells.append(
            f'<div class="grid-bracket-slot" data-round="2" data-pair="{pair}" '
            f'style="grid-column:{r32_c};grid-row:{r0}/span 2">{card}</div>'
        )

    for j, suf in enumerate(_R3_VISUAL_SUFFIXES):
        sid = f"R3{region}{suf}"
        row = _series_for_slot(df, region, sid)
        card = _game_card(row) if row is not None else '<div class="game-card game-card-missing"></div>'
        r0 = j * 4 + 1
        pair = "top" if j % 2 == 0 else "bottom"
        cells.append(
            f'<div class="grid-bracket-slot" data-round="3" data-pair="{pair}" '
            f'style="grid-column:{s16_c};grid-row:{r0}/span 4">{card}</div>'
        )

    for suf in _R4_VISUAL_SUFFIXES:
        sid = f"R4{region}{suf}"
        row = _series_for_slot(df, region, sid)
        card = _game_card(row) if row is not None else '<div class="game-card game-card-missing"></div>'
        cells.append(
            f'<div class="grid-bracket-slot" data-round="4" data-pair="top" '
            f'style="grid-column:{e8_c};grid-row:1/span 8">{card}</div>'
        )

    grid = "".join(cells)
    return (
        f'<div class="region-bracket {dir_cls}">'
        f'<div class="bracket-col-labels">{labels}</div>'
        f'<div class="region-bracket-grid">{grid}</div>'
        f"</div>"
    )


def render_final_four(df: pd.DataFrame) -> str:
    """Build HTML for Final Four + Championship."""
    ff = df[df["round_num"] == 5].sort_values("slot_id")
    ch = df[df["round_num"] == 6]

    html = '<div class="final-four-block">'
    html += '<div class="round round-5">'
    html += '<div class="round-label">Final Four</div>'
    for _, row in ff.iterrows():
        html += _game_card(row)
    html += '</div>'

    html += '<div class="round round-6">'
    html += '<div class="round-label">Championship</div>'
    for _, row in ch.iterrows():
        html += _game_card(row)
    html += '</div>'
    html += '</div>'
    return html


BRACKET_CSS = """
<style>
/* ── Outer bracket layout ── */
.bracket-scroll-wrap {
    overflow-x: auto;
    padding: 4px 0;
}
.bracket-outer-grid {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 0;
}
.bracket-center-col {
    display: flex;
    align-items: center;
}
.region-spacer {
    height: 16px;
}

/* ── Bracket container ── */
.bracket-container {
    display: flex;
    align-items: flex-start;
    justify-content: center;
    gap: 0;
    padding: 12px 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.region-bracket {
    display: block;
}

/* ── Column labels ── */
.bracket-col-labels {
    display: grid;
    grid-template-columns: repeat(4, minmax(150px, 1fr));
    gap: 0 20px;
    margin-bottom: 6px;
}
.bracket-col-labels .lbl {
    text-align: center;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
}

/* ── Region grid ── */
.region-bracket-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(150px, 1fr));
    grid-template-rows: repeat(8, minmax(52px, auto));
    column-gap: 20px;
    row-gap: 0;
    align-items: stretch;
}
.grid-bracket-slot {
    align-self: center;
    min-width: 0;
    position: relative;
}

/* ── Connector lines (LTR regions) ── */
/* Outgoing horizontal stub → right edge of card toward next round */
.region-bracket:not(.dir-rtl) .grid-bracket-slot[data-round="1"]::after,
.region-bracket:not(.dir-rtl) .grid-bracket-slot[data-round="2"]::after,
.region-bracket:not(.dir-rtl) .grid-bracket-slot[data-round="3"]::after {
    content: '';
    position: absolute;
    right: -20px;
    top: 50%;
    width: 10px;
    border-top: 2px solid #475569;
}
/* Incoming vertical bracket on receiving side (R32, S16, E8) */
.region-bracket:not(.dir-rtl) .grid-bracket-slot[data-round="2"]::before,
.region-bracket:not(.dir-rtl) .grid-bracket-slot[data-round="3"]::before,
.region-bracket:not(.dir-rtl) .grid-bracket-slot[data-round="4"]::before {
    content: '';
    position: absolute;
    left: -20px;
    top: 25%;
    height: 50%;
    width: 10px;
    border-left: 2px solid #475569;
    border-top: 2px solid #475569;
    border-bottom: 2px solid #475569;
}

/* ── Connector lines (RTL regions — mirrored) ── */
.dir-rtl .grid-bracket-slot[data-round="1"]::after,
.dir-rtl .grid-bracket-slot[data-round="2"]::after,
.dir-rtl .grid-bracket-slot[data-round="3"]::after {
    content: '';
    position: absolute;
    left: -20px;
    right: auto;
    top: 50%;
    width: 10px;
    border-top: 2px solid #475569;
}
.dir-rtl .grid-bracket-slot[data-round="2"]::before,
.dir-rtl .grid-bracket-slot[data-round="3"]::before,
.dir-rtl .grid-bracket-slot[data-round="4"]::before {
    content: '';
    position: absolute;
    right: -20px;
    left: auto;
    top: 25%;
    height: 50%;
    width: 10px;
    border-right: 2px solid #475569;
    border-top: 2px solid #475569;
    border-bottom: 2px solid #475569;
}

/* ── Game card missing ── */
.game-card-missing {
    min-height: 52px;
    opacity: 0.35;
    border-style: dashed;
}

/* ── Final Four block ── */
.final-four-block {
    display: flex;
    gap: 0;
    align-items: center;
    margin: 0 4px;
}
.round {
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    min-width: 160px;
    padding: 0 3px;
}
.round-label {
    text-align: center;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    padding: 4px 0 8px 0;
}

/* ── Game card ── */
.game-card {
    background: #1e293b;
    border-radius: 6px;
    margin: 4px 0;
    overflow: hidden;
    border: 1px solid #334155;
    position: relative;
    transition: transform 0.15s, box-shadow 0.15s;
}
.game-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    border-color: #60a5fa;
}
.badge-actual {
    position: absolute;
    top: 2px;
    right: 4px;
    font-size: 8px;
    font-weight: 700;
    color: #22c55e;
    letter-spacing: 0.05em;
}

/* ── Team rows ── */
.team-row {
    display: flex;
    align-items: center;
    padding: 5px 8px;
    gap: 6px;
    font-size: 12px;
    border-bottom: 1px solid #334155;
}
.team-row:last-child { border-bottom: none; }
.team-winner {
    background: #1a3a2a;
    color: #4ade80;
    font-weight: 600;
}
.team-winner.actual {
    background: #14412a;
    color: #22c55e;
}
.team-loser {
    color: #94a3b8;
}
.seed {
    font-size: 10px;
    font-weight: 700;
    color: #64748b;
    min-width: 16px;
    text-align: center;
    background: #0f172a;
    border-radius: 3px;
    padding: 1px 3px;
}
.team-winner .seed { color: #86efac; background: #14532d; }
.team-name {
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.win-pct {
    font-size: 9px;
    color: #94a3b8;
    min-width: 28px;
    text-align: right;
    padding-right: 3px;
}
.team-winner .win-pct { color: #86efac; }
.score {
    font-weight: 600;
    font-size: 12px;
    min-width: 24px;
    text-align: right;
}

/* ── Round-specific card margins ── */
.region-bracket-grid .game-card { margin: 2px 0; }
.round-5 .game-card { margin: 24px 0; }
.round-6 .game-card { margin: 24px 0; }

/* ── Split cards (predicted + actual) ── */
.game-card-split {
    padding: 0;
}
.game-split-label {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
    padding: 4px 6px 2px 6px;
    background: #0f172a;
}
.game-card-pred {
    border-bottom: 1px solid #334155;
}
.game-card-result {
    background: #0c1222;
}
.team-row.result-line.team-winner {
    background: #142032;
    color: #38bdf8;
    font-weight: 600;
}
.team-row.result-line.team-loser {
    color: #64748b;
}

/* ── Region header ── */
.region-header {
    text-align: center;
    font-size: 16px;
    font-weight: 800;
    padding: 8px;
    margin-bottom: 4px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-radius: 6px;
}

/* ── Champion banner ── */
.champion-banner {
    text-align: center;
    padding: 24px;
    margin: 16px 0;
    background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
    border-radius: 12px;
    border: 2px solid #facc15;
}
.champion-banner h2 {
    margin: 0 0 4px 0;
    font-size: 28px;
    color: #facc15;
}
.champion-banner p {
    margin: 0;
    color: #94a3b8;
    font-size: 14px;
}

/* ── Stat cards ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin: 16px 0;
}
.stat-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    border: 1px solid #334155;
}
.stat-card .value {
    font-size: 32px;
    font-weight: 800;
    color: #f8fafc;
}
.stat-card .label {
    font-size: 12px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── Matchup detail table ── */
.matchup-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.matchup-table th {
    background: #0f172a;
    color: #94a3b8;
    padding: 8px 12px;
    text-align: left;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 2px solid #334155;
}
.matchup-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #1e293b;
    color: #e2e8f0;
}
.matchup-table tr:hover td { background: #1e293b; }
.pick-correct { color: #4ade80; font-weight: 600; }
.pick-wrong { color: #f87171; font-weight: 600; }
.upset-tag {
    display: inline-block;
    font-size: 9px;
    font-weight: 700;
    background: #dc2626;
    color: #fff;
    padding: 1px 5px;
    border-radius: 3px;
    margin-left: 4px;
    letter-spacing: 0.04em;
}

/* ── First Four ── */
.first-four-row {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin: 12px 0;
}
.first-four-row .game-card {
    min-width: 160px;
}

/* ── Responsive: mobile ── */
@media (max-width: 768px) {
    .bracket-outer-grid {
        grid-template-columns: 1fr;
    }
    .bracket-center-col {
        order: 5;
    }
    .bracket-region-col:first-child { order: 1; }
    .bracket-region-col:last-child { order: 3; }
    .region-spacer { height: 8px; }
    .final-four-block {
        flex-direction: column;
    }
    /* Hide connector lines on mobile */
    .grid-bracket-slot::before,
    .grid-bracket-slot::after {
        display: none !important;
    }
    .bracket-col-labels,
    .region-bracket-grid {
        grid-template-columns: repeat(4, minmax(80px, 1fr));
        column-gap: 4px;
    }
    .team-row { padding: 3px 4px; gap: 3px; }
    .win-pct { display: none; }
}
</style>
"""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar_actuals() -> str | None:
    """Render the actuals-upload portion of the sidebar. Returns path or None."""
    with st.sidebar:
        st.markdown("## 🏀 Controls")
        st.divider()
        actuals_path = None
        full_path = DATA_DIR / "actuals.csv"
        ff_path = DATA_DIR / "first_four_actuals.csv"
        if full_path.exists():
            actuals_path = str(full_path)
            st.success("Using `data/actuals.csv`")
        elif ff_path.exists():
            actuals_path = str(ff_path)
            st.success("Using `data/first_four_actuals.csv` (First Four winners; add more rounds to this file as games finish).")
        st.caption(
            "Post-tournament: add `data/tournament_results.csv` (all rounds, same columns as actuals) "
            "to compare every model pick to real outcomes in the bracket and accuracy tables."
        )
        if not actuals_path:
            uploaded = st.file_uploader(
                "Upload actuals CSV",
                type="csv",
                help="CSV with columns: round, winner, winner_score, loser_score",
            )
            if uploaded:
                tmp = DATA_DIR / "_uploaded_actuals.csv"
                tmp.write_bytes(uploaded.getvalue())
                actuals_path = str(tmp)
                st.success("Actuals uploaded!")
    return actuals_path


def sidebar_models(model_names: list[str]) -> str:
    """Render the model-selector portion of the sidebar. Returns selected name."""
    with st.sidebar:
        selected_model = st.selectbox(
            "Prediction Model",
            model_names,
            index=0,
            help="Switch between models to compare brackets",
        )
        st.divider()
        st.markdown("### Models Available")
        for name in model_names:
            color = MODEL_COLORS.get(name, "#888")
            st.markdown(
                f'<span style="color:{color};font-weight:700">●</span> {name}',
                unsafe_allow_html=True,
            )
        st.divider()
        st.caption("EMBA 693R — March Madness 2026")
    return selected_model


# ---------------------------------------------------------------------------
# Main page sections
# ---------------------------------------------------------------------------

def _html(content: str, height: int = 100):
    """Render styled HTML via an iframe component (preserves <style> blocks)."""
    wrapped = f"""
    <html><head><meta charset="utf-8"></head>
    <body style="margin:0; background:transparent; overflow:hidden;">
    {BRACKET_CSS}
    {content}
    </body></html>
    """
    components.html(wrapped, height=height, scrolling=False)


def header_section(champions: dict, selected: str):
    champ = champions.get(selected, "???")
    _html(
        f"""
        <div class="champion-banner">
            <p style="font-size:12px; letter-spacing:0.1em;">
                {selected.upper()} PREDICTS
            </p>
            <h2>🏆 {champ} 🏆</h2>
            <p>2026 NCAA Champion</p>
        </div>
        """,
        height=140,
    )


def champions_comparison(champions: dict):
    cards = ""
    for name, champ in champions.items():
        color = MODEL_COLORS.get(name, "#888")
        cards += f"""
        <div class="stat-card" style="border-top: 3px solid {color}; flex:1; min-width:200px;">
            <div class="label">{name}</div>
            <div class="value" style="font-size:20px; color:{color};">{champ or '—'}</div>
        </div>
        """
    n = len(champions)
    height = 100 if n <= 3 else 200
    _html(
        f'<div style="display:flex; gap:12px; flex-wrap:wrap;">{cards}</div>',
        height=height,
    )


def pick_accuracy_vs_tournament(bracket_dfs: dict[str, pd.DataFrame]):
    """Compact table: each model's pick accuracy when `tournament_results.csv` is loaded."""
    if not bracket_dfs:
        return
    if not any(
        "result_winner_id" in df.columns and df["result_winner_id"].notna().any()
        for df in bracket_dfs.values()
    ):
        return
    rows = []
    for name, df in bracket_dfs.items():
        acc = overall_pick_accuracy(df)
        n = games_graded_count(df)
        rows.append({
            "Model": name,
            "Graded games": n,
            "Pick accuracy": f"{acc:.1%}" if pd.notna(acc) and n > 0 else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Graded games are matchups where both a model pick (`pred_winner_id`) and a "
        "`tournament_results.csv` result exist. Upload scores in that file for spread MAE on the Accuracy tab."
    )


def bracket_section(df: pd.DataFrame, model_name: str):
    model_color = MODEL_COLORS.get(model_name, "#60a5fa")

    # First Four (actuals when present; otherwise model/chalk fills play-ins)
    ff_df = df[df["round_num"] == 0]
    if not ff_df.empty:
        cards = ""
        for _, row in ff_df.sort_values("slot_id").iterrows():
            cards += _game_card(row)
        _html(
            '<div style="text-align:center; margin-top:8px;">'
            '<span style="font-size:13px; font-weight:700; color:#94a3b8; '
            'letter-spacing:0.1em; text-transform:uppercase;">First Four</span></div>'
            f'<div class="first-four-row">{cards}</div>',
            height=100,
        )

    # Main bracket: Left regions (W, Y), center (FF + CH), Right regions (X, Z)
    left_top = render_region_bracket(df, "W", "ltr")
    left_bot = render_region_bracket(df, "Y", "ltr")
    center = render_final_four(df)
    right_top = render_region_bracket(df, "X", "rtl")
    right_bot = render_region_bracket(df, "Z", "rtl")

    region_colors = {"W": "#3b82f6", "X": "#ef4444", "Y": "#22c55e", "Z": "#f59e0b"}

    def region_hdr(code):
        c = region_colors[code]
        return (f'<div class="region-header" style="color:{c}; '
                f'border: 1px solid {c}33; background: {c}11;">'
                f'{REGION_NAMES[code]}</div>')

    bracket_body = f"""
    <div class="bracket-scroll-wrap">
        <div class="bracket-outer-grid">
            <div class="bracket-region-col">
                {region_hdr("W")}
                <div class="bracket-container">{left_top}</div>
                <div class="region-spacer"></div>
                {region_hdr("Y")}
                <div class="bracket-container">{left_bot}</div>
            </div>
            <div class="bracket-center-col">
                <div class="bracket-container">{center}</div>
            </div>
            <div class="bracket-region-col">
                {region_hdr("X")}
                <div class="bracket-container">{right_top}</div>
                <div class="region-spacer"></div>
                {region_hdr("Z")}
                <div class="bracket-container">{right_bot}</div>
            </div>
        </div>
    </div>
    """
    bracket_full = f"""
    <html><head><meta charset="utf-8"></head>
    <body style="margin:0; background:transparent;">
    {BRACKET_CSS}
    {bracket_body}
    </body></html>
    """
    components.html(bracket_full, height=1200, scrolling=True)


def round_detail_section(df: pd.DataFrame, db: TeamDB):
    """Expandable round-by-round matchup tables."""
    for rnd in sorted(df["round_num"].unique()):
        label = ROUND_LABELS.get(rnd, f"Round {rnd}")
        rnd_df = df[df["round_num"] == rnd].sort_values("slot_id")
        n_games = len(rnd_df)
        with st.expander(f"{label}  ({n_games} games)", expanded=(rnd <= 1)):
            rows_html = ""
            for _, row in rnd_df.iterrows():
                s_seed = int(row["strong_seed"]) if pd.notna(row["strong_seed"]) else "?"
                w_seed = int(row["weak_seed"]) if pd.notna(row["weak_seed"]) else "?"
                s_score = f'{row["strong_pred_score"]:.0f}' if pd.notna(row.get("strong_pred_score")) else "—"
                w_score = f'{row["weak_pred_score"]:.0f}' if pd.notna(row.get("weak_pred_score")) else "—"

                winner = row["winner"] or "TBD"

                conf = row.get("confidence")
                if pd.notna(conf) and conf is not None:
                    odds_str = f"{float(conf) * 100:.0f}%"
                else:
                    odds_str = "—"

                upset_html = ""
                if pd.notna(row.get("strong_seed")) and pd.notna(row.get("weak_seed")):
                    if row.get("winner_id") == row.get("weak_team_id") and row["weak_seed"] > row["strong_seed"]:
                        upset_html = '<span class="upset-tag">UPSET</span>'

                actual_html = ""
                if row.get("is_actual"):
                    pred_correct = row.get("pred_winner_id") == row.get("actual_winner_id")
                    cls = "pick-correct" if pred_correct else "pick-wrong"
                    sym = "✓" if pred_correct else "✗"
                    a_winner = row.get("actual_winner", "")
                    a_s = f'{row["actual_strong_score"]:.0f}' if pd.notna(row.get("actual_strong_score")) else ""
                    a_w = f'{row["actual_weak_score"]:.0f}' if pd.notna(row.get("actual_weak_score")) else ""
                    actual_score = f"{a_s}–{a_w}" if a_s else ""
                    actual_html = (
                        f'<td><span class="{cls}">{sym}</span></td>'
                        f'<td>{a_winner}</td>'
                        f'<td>{actual_score}</td>'
                    )
                else:
                    actual_html = '<td colspan="3" style="color:#475569;">—</td>'

                rows_html += f"""
                <tr>
                    <td>({s_seed}) {row["strong_team"] or "TBD"}</td>
                    <td>({w_seed}) {row["weak_team"] or "TBD"}</td>
                    <td>{s_score}–{w_score}</td>
                    <td><strong>{winner}</strong>{upset_html}</td>
                    <td>{odds_str}</td>
                    {actual_html}
                </tr>"""

            has_actuals = rnd_df["is_actual"].any()
            actual_cols = (
                "<th>Pick</th><th>Actual Winner</th><th>Actual Score</th>"
                if has_actuals else "<th colspan='3'>Actual</th>"
            )

            table_height = 44 + n_games * 38
            _html(f"""
            <table class="matchup-table">
                <thead>
                    <tr>
                        <th>Higher Seed</th><th>Lower Seed</th>
                        <th>Pred. Score</th><th>Predicted Winner</th><th>Win %</th>
                        {actual_cols}
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            """, height=table_height)


def model_comparison_section(bracket_dfs: dict, db: TeamDB):
    """Show where models agree / disagree on each game."""
    model_names = list(bracket_dfs.keys())
    if len(model_names) < 2:
        st.info("Need at least 2 models to compare. Set your ANTHROPIC_API_KEY to enable more models.")
        return

    ref_df = list(bracket_dfs.values())[0]
    rounds = sorted(ref_df["round_num"].unique())

    selected_rnd = st.selectbox(
        "Compare round:",
        rounds,
        format_func=lambda r: ROUND_LABELS.get(r, f"Round {r}"),
        index=min(1, len(rounds) - 1),
    )

    ref_slots = ref_df[ref_df["round_num"] == selected_rnd].sort_values("slot_id")

    rows = []
    for _, slot_row in ref_slots.iterrows():
        sid = slot_row["slot_id"]
        s_seed = int(slot_row["strong_seed"]) if pd.notna(slot_row["strong_seed"]) else "?"
        w_seed = int(slot_row["weak_seed"]) if pd.notna(slot_row["weak_seed"]) else "?"
        matchup = f"({s_seed}) {slot_row['strong_team']}  vs  ({w_seed}) {slot_row['weak_team']}"
        row_data = {"Matchup": matchup}
        picks = set()
        for model_name, mdf in bracket_dfs.items():
            game = mdf[mdf["slot_id"] == sid]
            if not game.empty:
                g = game.iloc[0]
                w = g["winner"] or "—"
                s = ""
                if pd.notna(g.get("strong_pred_score")):
                    s = f" ({g['strong_pred_score']:.0f}–{g['weak_pred_score']:.0f})"
                row_data[model_name] = f"{w}{s}"
                picks.add(g.get("winner_id"))
        row_data["_agree"] = len(picks) <= 1
        rows.append(row_data)

    # Build the comparison as an HTML table with inline highlights
    model_cols = [n for n in model_names if n in bracket_dfs]
    header_cells = "<th>Matchup</th>" + "".join(f"<th>{m}</th>" for m in model_cols)
    body_rows = ""
    n_disagree = 0
    for r in rows:
        is_split = not r["_agree"]
        if is_split:
            n_disagree += 1
        row_style = ' style="background:#422006;"' if is_split else ""
        cells = f'<td>{r["Matchup"]}</td>'
        for m in model_cols:
            val = r.get(m, "—")
            td_style = ' style="color:#fbbf24; font-weight:600;"' if is_split else ""
            cells += f"<td{td_style}>{val}</td>"
        body_rows += f"<tr{row_style}>{cells}</tr>"

    n_total = len(rows)
    agree_pct = (n_total - n_disagree) / n_total * 100 if n_total else 0

    table_height = 50 + n_total * 38
    _html(f"""
    <table class="matchup-table">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{body_rows}</tbody>
    </table>
    <p style="color:#94a3b8; font-size:12px; margin-top:8px;">
        Models agree on <strong>{n_total - n_disagree}/{n_total}</strong>
        ({agree_pct:.0f}%) of {ROUND_LABELS.get(selected_rnd, '')} picks.
        Highlighted rows show disagreements.
    </p>
    """, height=table_height + 40)


def accuracy_section(bracket_dfs: dict):
    """Show accuracy matrix and spread MAE when actuals exist."""
    has_truth = any(
        "result_winner_id" in df.columns and df["result_winner_id"].notna().any()
        for df in bracket_dfs.values()
    )
    has_actuals = any(df["is_actual"].any() for df in bracket_dfs.values())
    if not has_truth and not has_actuals:
        st.info(
            "No results to grade against. Add `data/tournament_results.csv` (full tournament, "
            "same format as First Four actuals) for pick accuracy vs reality, or load "
            "pre-tournament actuals for partial metrics."
        )
        return

    acc_df = accuracy_table(bracket_dfs)
    spread_df = spread_accuracy_table(bracket_dfs)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Win Accuracy")
        if not acc_df.empty:
            display = acc_df.copy()
            model_cols = [c for c in display.columns if c != "window"]
            for c in model_cols:
                display[c] = display[c].apply(
                    lambda v: f"{v:.1%}" if pd.notna(v) else "—"
                )
            st.dataframe(display.rename(columns={"window": "Window"}),
                         use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Spread MAE (points)")
        if not spread_df.empty:
            display = spread_df.copy()
            model_cols = [c for c in display.columns if c != "window"]
            for c in model_cols:
                display[c] = display[c].apply(
                    lambda v: f"{v:.1f}" if pd.notna(v) else "—"
                )
            st.dataframe(display.rename(columns={"window": "Round"}),
                         use_container_width=True, hide_index=True)

    # Per-model accuracy bar chart
    if not acc_df.empty:
        import plotly.graph_objects as go
        model_cols = [c for c in acc_df.columns if c != "window"]
        cum_row = acc_df.iloc[-1] if "Just" not in acc_df.iloc[-1]["window"] else acc_df.iloc[-2]
        fig = go.Figure()
        for m in model_cols:
            val = cum_row[m]
            if pd.notna(val):
                fig.add_trace(go.Bar(
                    x=[m], y=[val * 100],
                    marker_color=MODEL_COLORS.get(m, "#888"),
                    text=f"{val:.1%}", textposition="outside",
                    name=m,
                ))
        fig.update_layout(
            title="Cumulative Accuracy (%)",
            yaxis=dict(range=[0, 105], title="% Correct"),
            showlegend=False,
            height=350,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Calibration metrics (Brier score + ECE)
    cal_df = calibration_summary(bracket_dfs)
    if not cal_df.empty and cal_df["n_games"].sum() > 0:
        st.markdown("#### Calibration (Brier Score & ECE)")
        st.caption(
            "Lower is better for both metrics. "
            "Brier score measures confidence accuracy; ECE measures systematic over/under-confidence."
        )
        display_cal = cal_df.copy()
        display_cal["brier_score"] = display_cal["brier_score"].apply(
            lambda v: f"{v:.4f}" if pd.notna(v) else "—"
        )
        display_cal["ece"] = display_cal["ece"].apply(
            lambda v: f"{v:.4f}" if pd.notna(v) else "—"
        )
        st.dataframe(
            display_cal.rename(columns={
                "model": "Model", "brier_score": "Brier Score",
                "ece": "ECE", "n_games": "Games",
            }),
            use_container_width=True, hide_index=True,
        )


def upset_tracker(df: pd.DataFrame):
    """Show notable upsets predicted by this model."""
    upsets = []
    for _, row in df.iterrows():
        if row["round_num"] == 0:
            continue
        s_seed = row.get("strong_seed")
        w_seed = row.get("weak_seed")
        if pd.isna(s_seed) or pd.isna(w_seed):
            continue
        if row.get("winner_id") == row.get("weak_team_id") and w_seed > s_seed:
            gap = int(w_seed - s_seed)
            conf = row.get("confidence")
            odds_str = f"{float(conf) * 100:.0f}%" if pd.notna(conf) and conf is not None else "—"
            upsets.append({
                "Round": row["round_label"],
                "Upset Winner": f"({int(w_seed)}) {row['weak_team']}",
                "Over": f"({int(s_seed)}) {row['strong_team']}",
                "Seed Gap": gap,
                "Score": f"{row['weak_pred_score']:.0f}–{row['strong_pred_score']:.0f}"
                         if pd.notna(row.get("weak_pred_score")) else "—",
                "Win %": odds_str,
            })

    if upsets:
        upset_df = pd.DataFrame(upsets).sort_values("Seed Gap", ascending=False)
        st.dataframe(upset_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No upsets predicted — chalk bracket!")


def path_to_title(df: pd.DataFrame, db: TeamDB, champion_id):
    """Show the champion's path through the tournament."""
    if champion_id is None:
        return

    path_games = []
    for _, row in df.sort_values("round_num").iterrows():
        if row.get("winner_id") == champion_id:
            opponent = (row["weak_team"] if row["winner_id"] == row["strong_team_id"]
                        else row["strong_team"])
            opp_seed = (row["weak_seed"] if row["winner_id"] == row["strong_team_id"]
                        else row["strong_seed"])
            opp_seed = int(opp_seed) if pd.notna(opp_seed) else "?"

            score = ""
            if pd.notna(row.get("strong_pred_score")):
                if row["winner_id"] == row["strong_team_id"]:
                    score = f"{row['strong_pred_score']:.0f}–{row['weak_pred_score']:.0f}"
                else:
                    score = f"{row['weak_pred_score']:.0f}–{row['strong_pred_score']:.0f}"

            conf = row.get("confidence")
            odds_str = f"{float(conf) * 100:.0f}%" if pd.notna(conf) and conf is not None else "—"
            path_games.append({
                "Round": row["round_label"],
                "Opponent": f"({opp_seed}) {opponent}",
                "Score": score,
                "Win %": odds_str,
                "Status": "✓ ACTUAL" if row.get("is_actual") else "Predicted",
            })

    if path_games:
        st.dataframe(pd.DataFrame(path_games), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    db, seeds_df, slots_df = load_engine()

    actuals_path = sidebar_actuals()

    bracket_dfs, champions, skipped = simulate_all(db, seeds_df, slots_df, actuals_path)
    model_names = list(bracket_dfs.keys())

    selected_model = sidebar_models(model_names)

    if skipped:
        for msg in skipped:
            st.sidebar.warning(f"Skipped: {msg}")

    # ── Header ──
    _html(
        '<h1 style="text-align:center; margin:0 0 4px 0; color:#f8fafc; '
        'font-family:Inter,-apple-system,sans-serif;">March Madness 2026</h1>'
        '<p style="text-align:center; color:#94a3b8; margin:0; font-size:14px;">'
        'Bracket Prediction Engine — EMBA 693R</p>',
        height=80,
    )

    # ── Champion picks from all models ──
    st.markdown("### 🏆 Champion Predictions")
    champions_comparison(champions)
    pick_accuracy_vs_tournament(bracket_dfs)

    st.divider()

    # ── Full bracket ──
    st.markdown(f"### Full Bracket — {selected_model}")
    st.caption(
        "When `data/first_four_actuals.csv` or `data/actuals.csv` exists, First Four winners are "
        "locked in before simulation so R64+ match your post–First Four bracket. "
        "If neither file is present, play-ins use higher-seed chalk."
    )
    header_section(champions, selected_model)
    bracket_section(bracket_dfs[selected_model], selected_model)

    st.divider()

    # ── Round details ──
    tab_rounds, tab_compare, tab_upsets, tab_path, tab_accuracy = st.tabs([
        "📋 Round Details",
        "🔀 Model Comparison",
        "🔥 Upset Tracker",
        "🛤️ Path to Title",
        "📊 Accuracy",
    ])

    with tab_rounds:
        round_detail_section(bracket_dfs[selected_model], db)

    with tab_compare:
        model_comparison_section(bracket_dfs, db)

    with tab_upsets:
        st.markdown(f"#### Upsets Predicted by {selected_model}")
        upset_tracker(bracket_dfs[selected_model])

    with tab_path:
        st.markdown(f"#### {champions.get(selected_model, '???')}'s Road to the Championship")
        champ_df = bracket_dfs[selected_model]
        ch_row = champ_df[champ_df["round_num"] == 6]
        champ_id = ch_row.iloc[0]["winner_id"] if not ch_row.empty else None
        path_to_title(champ_df, db, champ_id)

    with tab_accuracy:
        accuracy_section(bracket_dfs)


if __name__ == "__main__":
    main()
