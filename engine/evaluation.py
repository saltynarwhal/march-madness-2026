"""Evaluation: compare model predictions against actual tournament results."""

from __future__ import annotations

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

    actual_rounds = sorted(
        {
            r
            for df in bracket_dfs.values()
            for r in df[df["is_actual"]]["round_num"].unique()
        }
    )

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
            actual = df[df["is_actual"]].copy()
            cum_slice = actual[actual["round_num"].isin(actual_rounds[: i + 1])]
            just_slice = actual[actual["round_num"] == rnd]

            cum_row[name] = _win_accuracy(cum_slice)
            just_row[name] = _win_accuracy(just_slice)

        rows.append(cum_row)
        if i > 0:
            rows.append(just_row)

    return pd.DataFrame(rows)


def spread_accuracy_table(
    bracket_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """MAE of predicted spread by round."""
    model_names = list(bracket_dfs.keys())

    actual_rounds = sorted(
        {
            r
            for df in bracket_dfs.values()
            for r in df[df["is_actual"]]["round_num"].unique()
        }
    )

    if not actual_rounds:
        return pd.DataFrame(columns=["window"] + model_names)

    rows: list[dict] = []
    for rnd in actual_rounds:
        row: dict = {"window": ROUND_LABELS.get(rnd, f"R{rnd}")}
        for name, df in bracket_dfs.items():
            slc = df[(df["is_actual"]) & (df["round_num"] == rnd)]
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


def _win_accuracy(df: pd.DataFrame) -> float:
    """Fraction of games where pred_winner_id == actual_winner_id."""
    if df.empty:
        return np.nan
    mask = df["pred_winner_id"].notna() & df["actual_winner_id"].notna()
    subset = df[mask]
    if subset.empty:
        return np.nan
    correct = (subset["pred_winner_id"] == subset["actual_winner_id"]).sum()
    return correct / len(subset)


def _spread_mae(df: pd.DataFrame) -> float:
    """MAE between predicted and actual score margins."""
    if df.empty:
        return np.nan
    errors = []
    for _, row in df.iterrows():
        pred_s = row.get("strong_pred_score")
        pred_w = row.get("weak_pred_score")
        act_s = row.get("actual_strong_score")
        act_w = row.get("actual_weak_score")
        if all(pd.notna(v) for v in [pred_s, pred_w, act_s, act_w]):
            errors.append(abs((pred_s - pred_w) - (act_s - act_w)))
    return float(np.mean(errors)) if errors else np.nan
