"""
March Madness 2026 — Bracket Prediction Dashboard
Run with:  streamlit run dashboard.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).resolve().parent
load_dotenv(_root / ".env")

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(_root))

from engine.db import TeamDB
from engine.bracket import Bracket, ROUND_LABELS
from engine.models.seeding import SeedingModel
from engine.evaluation import accuracy_table, spread_accuracy_table

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

REGION_NAMES = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}

MODEL_COLORS = {
    "Advanced Metrics": "#2563eb",
    "Animal Kingdom": "#dc2626",
    "Vegas Odds": "#059669",
    "Seeding Only": "#7c3aed",
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
        models["Advanced Metrics"] = AdvancedMetricsModel(str(DATA_DIR / "models"))
    except Exception:
        pass
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if has_api_key:
        try:
            from engine.models.animal_kingdom import AnimalKingdomModel
            models["Animal Kingdom"] = AnimalKingdomModel()
        except Exception:
            pass
        try:
            from engine.models.vegas_odds import VegasOddsModel
            models["Vegas Odds"] = VegasOddsModel(
                lines_path=DATA_DIR / "vegas_lines.csv",
            )
        except Exception:
            pass
    return models


@st.cache_data(show_spinner="Simulating brackets…")
def simulate_all(_db, _seeds_df, _slots_df, actuals_csv: str | None):
    models = build_models(_db)
    brackets = {}
    bracket_dfs = {}
    skipped: list[str] = []
    for name, model in models.items():
        try:
            b = Bracket(_seeds_df, _slots_df, season=2026)
            b.simulate(model, _db)
            brackets[name] = b
            bracket_dfs[name] = b.to_dataframe(_db)
        except Exception as exc:
            skipped.append(f"{name}: {exc}")

    if actuals_csv:
        from engine.actuals import load_actuals
        for name, b in brackets.items():
            actuals = load_actuals(actuals_csv, b, _db)
            if not actuals.empty:
                b.inject_actuals(actuals)
                b.simulate(models[name], _db)
                bracket_dfs[name] = b.to_dataframe(_db)

    champions = {}
    for name, b in brackets.items():
        champions[name] = b.get_champion(_db)

    return bracket_dfs, champions, skipped


# ---------------------------------------------------------------------------
# Bracket rendering (HTML/CSS)
# ---------------------------------------------------------------------------

def _team_cell(name: str, seed, score, is_winner: bool, is_actual: bool) -> str:
    seed_int = int(seed) if pd.notna(seed) else "?"
    score_str = f"{score:.0f}" if pd.notna(score) else ""
    cls = "team-winner" if is_winner else "team-loser"
    if is_actual:
        cls += " actual"
    return (
        f'<div class="team-row {cls}">'
        f'<span class="seed">{seed_int}</span>'
        f'<span class="team-name">{name}</span>'
        f'<span class="score">{score_str}</span>'
        f'</div>'
    )


def _game_card(row: pd.Series) -> str:
    s_winner = row["winner_id"] == row["strong_team_id"] if pd.notna(row.get("winner_id")) else False
    w_winner = row["winner_id"] == row["weak_team_id"] if pd.notna(row.get("winner_id")) else False
    is_actual = bool(row.get("is_actual", False))

    if is_actual and pd.notna(row.get("actual_strong_score")):
        s_score = row["actual_strong_score"]
        w_score = row["actual_weak_score"]
    else:
        s_score = row.get("strong_pred_score")
        w_score = row.get("weak_pred_score")

    top = _team_cell(row["strong_team"] or "TBD", row.get("strong_seed"), s_score, s_winner, is_actual)
    bot = _team_cell(row["weak_team"] or "TBD", row.get("weak_seed"), w_score, w_winner, is_actual)

    badge = ""
    if is_actual:
        badge = '<span class="badge-actual">FINAL</span>'

    return f'<div class="game-card">{badge}{top}{bot}</div>'


def render_region_bracket(df: pd.DataFrame, region: str, direction: str = "ltr") -> str:
    """Build HTML for one region's bracket (R64->R32->S16->E8)."""
    rounds = [1, 2, 3, 4]

    dir_cls = "dir-rtl" if direction == "rtl" else ""
    html = f'<div class="region-bracket {dir_cls}">'
    for rnd in rounds:
        rnd_df = df[(df["round_num"] == rnd) & (df["region"] == region)]
        rnd_df = rnd_df.sort_values("slot_id")
        round_cls = f"round round-{rnd}"
        html += f'<div class="{round_cls}">'
        html += f'<div class="round-label">{ROUND_LABELS.get(rnd, "")}</div>'
        for _, row in rnd_df.iterrows():
            html += _game_card(row)
        html += '</div>'
    html += '</div>'
    return html


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
.bracket-container {
    display: flex;
    align-items: flex-start;
    justify-content: center;
    gap: 0;
    overflow-x: auto;
    padding: 12px 0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.region-bracket {
    display: flex;
    gap: 0;
    align-items: center;
}
.region-bracket.dir-rtl {
    flex-direction: row-reverse;
}
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
    min-width: 170px;
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
.score {
    font-weight: 600;
    font-size: 12px;
    min-width: 24px;
    text-align: right;
}
.round-1 .game-card { margin: 2px 0; }
.round-2 .game-card { margin: 16px 0; }
.round-3 .game-card { margin: 44px 0; }
.round-4 .game-card { margin: 100px 0; }
.round-5 .game-card { margin: 24px 0; }
.round-6 .game-card { margin: 24px 0; }
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
/* Champion banner */
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
/* Stat cards */
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
/* Matchup detail table */
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
/* First Four */
.first-four-row {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin: 12px 0;
}
.first-four-row .game-card {
    min-width: 170px;
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
        default_path = DATA_DIR / "actuals.csv"
        if default_path.exists():
            actuals_path = str(default_path)
            st.success("Actuals loaded from `data/actuals.csv`")
        else:
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


def bracket_section(df: pd.DataFrame, model_name: str):
    model_color = MODEL_COLORS.get(model_name, "#60a5fa")

    # First Four
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
    <div style="overflow-x:auto;">
        <div style="display:grid; grid-template-columns:1fr auto 1fr; gap:0; min-width:1100px;">
            <div>
                {region_hdr("W")}
                <div class="bracket-container">{left_top}</div>
                <div style="height:16px;"></div>
                {region_hdr("Y")}
                <div class="bracket-container">{left_bot}</div>
            </div>
            <div style="display:flex; align-items:center;">
                <div class="bracket-container">{center}</div>
            </div>
            <div>
                {region_hdr("X")}
                <div class="bracket-container">{right_top}</div>
                <div style="height:16px;"></div>
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
                        <th>Pred. Score</th><th>Predicted Winner</th>
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
    has_actuals = any(df["is_actual"].any() for df in bracket_dfs.values())
    if not has_actuals:
        st.info("No actual results loaded yet. Upload actuals to see accuracy metrics.")
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
            upsets.append({
                "Round": row["round_label"],
                "Upset Winner": f"({int(w_seed)}) {row['weak_team']}",
                "Over": f"({int(s_seed)}) {row['strong_team']}",
                "Seed Gap": gap,
                "Score": f"{row['weak_pred_score']:.0f}–{row['strong_pred_score']:.0f}"
                         if pd.notna(row.get("weak_pred_score")) else "—",
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

            path_games.append({
                "Round": row["round_label"],
                "Opponent": f"({opp_seed}) {opponent}",
                "Score": score,
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

    st.divider()

    # ── Full bracket ──
    st.markdown(f"### Full Bracket — {selected_model}")
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
