"""Evaluation: compare model predictions against actual tournament results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False

from engine.bracket import ROUND_LABELS


# ------------------------------------------------------------------
# Tournament results merge (post-tournament evaluation)
# ------------------------------------------------------------------


def truth_dataframe_from_tournament_csv(
    path: str | Path,
    seeds_df: pd.DataFrame,
    slots_df: pd.DataFrame,
    db,  # TeamDB
) -> pd.DataFrame:
    """
    Build a bracket dataframe where actual_* columns reflect the full tournament
    CSV (same format as ``load_actuals``). Used to merge ``result_*`` onto model
    predictions without changing those predictions.
    """
    from engine.bracket import Bracket
    from engine.actuals import load_actuals

    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    b = Bracket(seeds_df, slots_df, season=2026)
    load_actuals(p, b, db)
    return b.to_dataframe(db)


def merge_tournament_results_into_bracket_dfs(
    bracket_dfs: dict[str, pd.DataFrame],
    truth_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Add ``result_*`` columns (tournament truth) per slot_id so the UI can show
    predictions vs actuals side by side. Does not modify pred_* columns.
    """
    if truth_df.empty or "slot_id" not in truth_df.columns:
        return bracket_dfs

    rename_map = {
        "actual_winner_id": "result_winner_id",
        "actual_winner": "result_winner",
        "actual_strong_score": "result_strong_score",
        "actual_weak_score": "result_weak_score",
    }
    cols = ["slot_id"] + [c for c in rename_map if c in truth_df.columns]
    t = truth_df[cols].rename(columns=rename_map)

    out: dict[str, pd.DataFrame] = {}
    for name, df in bracket_dfs.items():
        drop = [c for c in rename_map.values() if c in df.columns]
        base = df.drop(columns=drop, errors="ignore")
        out[name] = base.merge(t, on="slot_id", how="left")
    return out


def games_graded_count(df: pd.DataFrame) -> int:
    """Rows where we can compare pred_winner_id to tournament result."""
    if "result_winner_id" in df.columns:
        m = df["result_winner_id"].notna() & df["pred_winner_id"].notna()
        return int(m.sum())
    m = df["is_actual"] & df["pred_winner_id"].notna() & df["actual_winner_id"].notna()
    return int(m.sum())


def overall_pick_accuracy(df: pd.DataFrame) -> float:
    """Fraction of graded games where the model picked the real winner."""
    if "result_winner_id" in df.columns:
        mask = df["result_winner_id"].notna() & df["pred_winner_id"].notna()
        truth = "result_winner_id"
    else:
        mask = df["is_actual"] & df["pred_winner_id"].notna() & df["actual_winner_id"].notna()
        truth = "actual_winner_id"
    subset = df.loc[mask]
    if subset.empty:
        return float("nan")
    return float((subset["pred_winner_id"] == subset[truth]).mean())


# ------------------------------------------------------------------
# Core accuracy computation
# ------------------------------------------------------------------


def accuracy_table(
    bracket_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build the cumulative + per-round accuracy matrix.

    Parameters
    ----------
    bracket_dfs : dict
        ``{model_name: bracket.to_dataframe(db)}`` for each model.

    Returns
    -------
    DataFrame with rows = evaluation windows, columns = model names.
    """
    model_names = list(bracket_dfs.keys())

    actual_rounds = sorted(_rounds_with_known_results(bracket_dfs))

    if not actual_rounds:
        return pd.DataFrame(columns=["window"] + model_names)

    rows: list[dict] = []

    for i, rnd in enumerate(actual_rounds):
        cum_labels = [ROUND_LABELS.get(r, f"R{r}") for r in actual_rounds[: i + 1]]
        cum_label = " + ".join(cum_labels)
        just_label = f"Just {ROUND_LABELS.get(rnd, f'R{rnd}')}"

        cum_row: dict = {"window": cum_label}
        just_row: dict = {"window": just_label}

        for name, df in bracket_dfs.items():
            graded = _graded_games(df)
            cum_slice = graded[graded["round_num"].isin(actual_rounds[: i + 1])]
            just_slice = graded[graded["round_num"] == rnd]

            cum_row[name] = _win_accuracy(cum_slice)
            just_row[name] = _win_accuracy(just_slice)

        rows.append(cum_row)
        rows.append(just_row)

    return pd.DataFrame(rows)


def spread_accuracy_table(
    bracket_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """MAE of predicted spread by round."""
    model_names = list(bracket_dfs.keys())

    actual_rounds = sorted(_rounds_with_known_results(bracket_dfs))

    if not actual_rounds:
        return pd.DataFrame(columns=["window"] + model_names)

    rows: list[dict] = []
    for rnd in actual_rounds:
        row: dict = {"window": ROUND_LABELS.get(rnd, f"R{rnd}")}
        for name, df in bracket_dfs.items():
            slc = _graded_games(df)
            slc = slc[slc["round_num"] == rnd]
            row[name] = _spread_mae(slc)
        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------


def plot_accuracy_heatmap(
    acc_df: pd.DataFrame,
    title: str = "Model Accuracy by Round",
    figsize: tuple = (10, 6),
    ax=None,
):
    """Render the accuracy matrix as a seaborn heatmap."""
    if not _HAS_VIZ:
        print("matplotlib / seaborn not available — skipping plot.")
        return None

    data = acc_df.set_index("window")
    numeric = data.apply(pd.to_numeric, errors="coerce")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.heatmap(
        numeric,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        vmin=0.4,
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _graded_games(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where pred can be compared to tournament truth (or legacy is_actual)."""
    if "result_winner_id" in df.columns:
        return df[df["result_winner_id"].notna() & df["pred_winner_id"].notna()].copy()
    return df[df["is_actual"] & df["pred_winner_id"].notna() & df["actual_winner_id"].notna()].copy()


def _rounds_with_known_results(bracket_dfs: dict[str, pd.DataFrame]) -> set[int]:
    rounds: set[int] = set()
    for df in bracket_dfs.values():
        g = _graded_games(df)
        if not g.empty:
            rounds.update(g["round_num"].unique().tolist())
    return rounds


def _win_accuracy(df: pd.DataFrame) -> float:
    """Fraction of games where pred_winner_id matches tournament result."""
    if df.empty:
        return np.nan
    if "result_winner_id" in df.columns:
        mask = df["result_winner_id"].notna() & df["pred_winner_id"].notna()
        truth_col = "result_winner_id"
    else:
        mask = df["pred_winner_id"].notna() & df["actual_winner_id"].notna()
        truth_col = "actual_winner_id"
    subset = df.loc[mask]
    if subset.empty:
        return np.nan
    correct = (subset["pred_winner_id"] == subset[truth_col]).sum()
    return correct / len(subset)


def _spread_mae(df: pd.DataFrame) -> float:
    """MAE between predicted and actual score margins."""
    if df.empty:
        return np.nan
    errors = []
    for _, row in df.iterrows():
        pred_s = row.get("strong_pred_score")
        pred_w = row.get("weak_pred_score")
        if "result_strong_score" in row.index and pd.notna(row.get("result_strong_score")):
            act_s = row.get("result_strong_score")
            act_w = row.get("result_weak_score")
        else:
            act_s = row.get("actual_strong_score")
            act_w = row.get("actual_weak_score")
        if all(pd.notna(v) for v in [pred_s, pred_w, act_s, act_w]):
            errors.append(abs((pred_s - pred_w) - (act_s - act_w)))
    return float(np.mean(errors)) if errors else np.nan


# ------------------------------------------------------------------
# Calibration & Brier Score
# ------------------------------------------------------------------


def calibration_summary(
    bracket_dfs: dict[str, pd.DataFrame],
    n_bins: int = 5,
) -> pd.DataFrame:
    """Compute Brier score and ECE per model from bracket dataframes.

    Confidence semantics: P(predicted winner wins this game), range [0.5, 1.0].
    Outcome: 1 if predicted winner actually won, 0 otherwise.
    """
    rows: list[dict] = []
    for name, df in bracket_dfs.items():
        graded = _graded_games(df)
        if graded.empty or "confidence" not in graded.columns:
            rows.append({"model": name, "brier_score": np.nan, "ece": np.nan, "n_games": 0})
            continue

        conf = graded["confidence"].astype(float)
        mask = conf.notna()
        if mask.sum() == 0:
            rows.append({"model": name, "brier_score": np.nan, "ece": np.nan, "n_games": 0})
            continue

        graded = graded.loc[mask].copy()
        conf = graded["confidence"].astype(float)

        if "result_winner_id" in graded.columns:
            truth = graded["result_winner_id"]
        else:
            truth = graded["actual_winner_id"]
        outcome = (graded["pred_winner_id"] == truth).astype(float)

        brier = float(((conf - outcome) ** 2).mean())
        ece = _expected_calibration_error(conf.values, outcome.values, n_bins=n_bins)
        rows.append({
            "model": name,
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
            "n_games": int(len(graded)),
        })

    return pd.DataFrame(rows)


def _expected_calibration_error(
    confidences: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 5,
) -> float:
    """Weighted average |accuracy - confidence| across equal-width bins in [0.5, 1.0]."""
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    total = len(confidences)
    if total == 0:
        return np.nan
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi if hi < 1.0 else confidences <= hi)
        n = mask.sum()
        if n == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = outcomes[mask].mean()
        ece += (n / total) * abs(avg_acc - avg_conf)
    return float(ece)


# ------------------------------------------------------------------
# Multi-Year Backtesting
# ------------------------------------------------------------------


def backtest_seasons(
    model_builders: dict[str, callable],
    barttorvik_df: pd.DataFrame,
    crosswalk_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    slots_df: pd.DataFrame,
    results_df: pd.DataFrame,
    seasons: list[int] | None = None,
    data_dir: str | None = None,
) -> pd.DataFrame:
    """Backtest models on historical tournament seasons.

    Parameters
    ----------
    model_builders : dict
        {name: callable} where each callable returns a PredictionModel instance.
    barttorvik_df : DataFrame
        Multi-year Barttorvik data with 'season' and 'kaggle_team_id' columns.
    crosswalk_df : DataFrame
        Barttorvik-to-Kaggle name mapping.
    seeds_df : DataFrame
        Kaggle MNCAATourneySeeds.csv.
    slots_df : DataFrame
        Kaggle MNCAATourneySlots.csv.
    results_df : DataFrame
        Kaggle MNCAATourneyCompactResults.csv (actual tournament outcomes).
    seasons : list[int], optional
        Seasons to backtest. Defaults to [2023, 2024, 2025].
    data_dir : str, optional
        Path to data/ directory for model artifact loading.

    Returns
    -------
    DataFrame with columns: season, model, correct, total, accuracy, brier_score
    """
    import re
    from engine.bracket import Bracket
    from engine.db import TeamDB

    if seasons is None:
        seasons = [2023, 2024, 2025]

    rows: list[dict] = []

    for season in seasons:
        # ── Build season-specific TeamDB from Barttorvik data ──
        bart_season = barttorvik_df[barttorvik_df["season"] == season].copy()
        if bart_season.empty:
            print(f"  [SKIP] No Barttorvik data for {season}")
            continue

        # Merge crosswalk to get kaggle_team_id
        if "kaggle_team_id" not in bart_season.columns and crosswalk_df is not None:
            bart_season = bart_season.merge(
                crosswalk_df[["bart_name", "kaggle_team_id"]],
                left_on="team", right_on="bart_name", how="left",
            )

        # Filter to tournament teams only
        tourn_ids = seeds_df.loc[seeds_df["Season"] == season, "TeamID"].unique()
        bart_season["kaggle_team_id"] = pd.to_numeric(
            bart_season["kaggle_team_id"], errors="coerce"
        )
        bart_season = bart_season[bart_season["kaggle_team_id"].isin(tourn_ids)].copy()

        if bart_season.empty:
            print(f"  [SKIP] No tournament teams matched for {season}")
            continue

        db = TeamDB.from_season_df(bart_season, data_dir=data_dir)
        db.load_seeds(seeds_df, season=season)

        # ── Build actuals from Kaggle tournament results ──
        season_results = results_df[results_df["Season"] == season].copy()
        # Map DayNum to round number using seed-round-slot mapping
        # Simpler: use the bracket's slot structure to inject actuals
        actuals_records = []
        for _, game in season_results.iterrows():
            actuals_records.append({
                "winner_id": int(game["WTeamID"]),
                "loser_id": int(game["LTeamID"]),
                "winner_score": int(game["WScore"]),
                "loser_score": int(game["LScore"]),
            })

        # ── Simulate bracket for each model ──
        for model_name, builder in model_builders.items():
            try:
                model = builder()
            except Exception as exc:
                print(f"  [SKIP] {model_name} failed to build: {exc}")
                continue

            bracket = Bracket(seeds_df, slots_df, season=season)
            bracket.simulate(model, db)

            # ── Inject actuals by matching winners to slots ──
            bdf = bracket.to_dataframe(db)

            # Match actual winners: for each slot with a prediction, check if
            # the actual winner matches the predicted winner.
            correct = 0
            total = 0
            confs = []
            outcomes = []

            for _, slot_row in bdf.iterrows():
                pred_winner = slot_row.get("pred_winner_id")
                if pd.isna(pred_winner):
                    continue

                # Find actual winner for this matchup
                strong_id = slot_row.get("strong_team_id")
                weak_id = slot_row.get("weak_team_id")
                if pd.isna(strong_id) or pd.isna(weak_id):
                    continue

                strong_id = int(strong_id)
                weak_id = int(weak_id)

                # Look up actual result: who won between these two teams?
                actual_winner = None
                for game in actuals_records:
                    w, l = game["winner_id"], game["loser_id"]
                    if (w == strong_id and l == weak_id) or (w == weak_id and l == strong_id):
                        actual_winner = w
                        break

                if actual_winner is None:
                    continue  # game not found (play-in or unmatched)

                total += 1
                if int(pred_winner) == actual_winner:
                    correct += 1

                conf = slot_row.get("confidence")
                if pd.notna(conf):
                    confs.append(float(conf))
                    outcomes.append(1.0 if int(pred_winner) == actual_winner else 0.0)

            accuracy = correct / total if total > 0 else np.nan
            brier = float(np.mean([(c - o) ** 2 for c, o in zip(confs, outcomes)])) if confs else np.nan

            rows.append({
                "season": season,
                "model": model_name,
                "correct": correct,
                "total": total,
                "accuracy": round(accuracy, 4) if not np.isnan(accuracy) else np.nan,
                "brier_score": round(brier, 4) if not np.isnan(brier) else np.nan,
            })

        print(f"  {season}: backtested {len(model_builders)} models on {total} games")

    result_df = pd.DataFrame(rows)

    # Add aggregate row per model
    if not result_df.empty:
        agg_rows = []
        for model_name in result_df["model"].unique():
            mdf = result_df[result_df["model"] == model_name]
            total_correct = mdf["correct"].sum()
            total_games = mdf["total"].sum()
            agg_rows.append({
                "season": "ALL",
                "model": model_name,
                "correct": total_correct,
                "total": total_games,
                "accuracy": round(total_correct / total_games, 4) if total_games > 0 else np.nan,
                "brier_score": round(mdf["brier_score"].mean(), 4),
            })
        result_df = pd.concat([result_df, pd.DataFrame(agg_rows)], ignore_index=True)

    return result_df
