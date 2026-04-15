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
