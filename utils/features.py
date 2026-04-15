"""
Feature engineering utilities.

Functions that transform raw data into model-ready features: difference
columns, round flags, regular-season aggregates, box score metrics,
consensus rankings, and 2026 prediction-time feature assembly.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Simple feature flags
# ---------------------------------------------------------------------------

def is_late_round(daynum):
    """Return 1 for roughly Sweet 16 and beyond, else 0.

    Uses DayNum as a proxy for tournament round. The NCAA tournament
    typically starts around DayNum 136; games at DayNum >= 143 correspond
    to Sweet 16 and later rounds.

    Parameters
    ----------
    daynum : numeric
        The DayNum value from the Kaggle game data.

    Returns
    -------
    int
        1 if late round, 0 otherwise.
    """
    try:
        d = float(daynum)
    except (TypeError, ValueError):
        return 0
    return 1 if d >= 143 else 0  # based on typical NCAA DayNum cut points


# ---------------------------------------------------------------------------
# Difference features
# ---------------------------------------------------------------------------

def add_diff_features(df, a_cols, b_cols):
    """Compute Team A minus Team B difference columns.

    For modeling, we often want one number per game: how much better is
    Team A than B on a given metric? Positive means Team A has the advantage.

    Parameters
    ----------
    df : pd.DataFrame
        Matchup-level DataFrame with columns for each team.
    a_cols : list of str
        Column names for Team A metrics.
    b_cols : list of str
        Column names for Team B metrics (parallel to a_cols).

    Returns
    -------
    pd.DataFrame
        The input DataFrame with new ``*_diff`` columns appended.
    """
    for ac, bc in zip(a_cols, b_cols):
        base = ac.replace('_a_', '_').replace('_teamA_', '_')
        diff_col = base + '_diff'
        df[diff_col] = df[ac] - df[bc]
    return df


# ---------------------------------------------------------------------------
# Regular-season aggregates
# ---------------------------------------------------------------------------

def build_reg_season_features(kg_reg, seasons):
    """Compute team-season metrics the seed committee underweights.

    Produces three features per team-season:
      - ``close_win_pct``: win % in games decided by <= 5 pts (clutch/pressure)
      - ``last15_win_pct``: win % in the final 15 regular-season games (momentum)
      - ``sos``: average opponent win % (schedule strength proxy)

    These are orthogonal to efficiency ratings and give the model real
    upset signal.

    Parameters
    ----------
    kg_reg : pd.DataFrame
        Kaggle ``MRegularSeasonCompactResults.csv``.
    seasons : iterable of int
        Which seasons to include.

    Returns
    -------
    pd.DataFrame or None
    """
    if kg_reg is None:
        print("[SKIP] kg_reg not loaded.")
        return None

    reg = kg_reg[kg_reg['Season'].isin(seasons)].copy()

    # One row per team per game (both winner and loser perspectives)
    w = reg[['Season', 'DayNum', 'WTeamID', 'LTeamID', 'WScore', 'LScore']].copy()
    w.columns = ['season', 'daynum', 'team_id', 'opp_id', 'team_score', 'opp_score']
    w['won'] = 1

    lo = reg[['Season', 'DayNum', 'LTeamID', 'WTeamID', 'LScore', 'WScore']].copy()
    lo.columns = ['season', 'daynum', 'team_id', 'opp_id', 'team_score', 'opp_score']
    lo['won'] = 0

    games = pd.concat([w, lo], ignore_index=True)
    games['margin'] = (games['team_score'] - games['opp_score']).abs()
    games['is_close'] = (games['margin'] <= 5).astype(int)

    # Full-season win% per team — used as opponent quality signal for SOS
    team_wpct = (
        games.groupby(['season', 'team_id'])['won'].mean()
        .reset_index()
        .rename(columns={'won': 'opp_win_pct', 'team_id': 'opp_id'})
    )
    games = games.merge(team_wpct, on=['season', 'opp_id'], how='left')

    records = []
    for (season, team_id), grp in games.groupby(['season', 'team_id']):
        close = grp[grp['is_close'] == 1]
        last15 = grp.nlargest(15, 'daynum')
        records.append({
            'season':         int(season),
            'team_id':        int(team_id),
            'close_win_pct':  close['won'].mean() if len(close) >= 3 else grp['won'].mean(),
            'last15_win_pct': last15['won'].mean(),
            'sos':            grp['opp_win_pct'].mean(),
        })

    out = pd.DataFrame(records)
    print(f"Reg-season features: {out.shape[0]:,} team-seasons  |  "
          f"{int(out['season'].min())}–{int(out['season'].max())}")
    return out


# ---------------------------------------------------------------------------
# Detailed box score aggregates
# ---------------------------------------------------------------------------

def build_detailed_box_features(kg_detailed, seasons):
    """Compute team-season box score metrics not available from Barttorvik.

    Produces four features per team-season:
      - ``ft_pct``:   free throw accuracy (Barttorvik tracks FT *rate*, not *makes*)
      - ``ast_rate``: assists / FGM per game (ball movement / system cohesion)
      - ``blk_rate``: blocks / opp FGA per game (rim protection)
      - ``stl_rate``: steals / opp FGA per game (perimeter pressure)

    Parameters
    ----------
    kg_detailed : pd.DataFrame
        Kaggle ``MRegularSeasonDetailedResults.csv``.
    seasons : iterable of int
        Which seasons to include.

    Returns
    -------
    pd.DataFrame or None
    """
    if kg_detailed is None:
        print("[SKIP] MRegularSeasonDetailedResults.csv not available.")
        return None

    det = kg_detailed[kg_detailed['Season'].isin(seasons)].copy()

    # Winner perspective
    w = det[['Season', 'WTeamID', 'WFTM', 'WFTA', 'WAst', 'WFGM', 'WBlk', 'WStl', 'LFGA']].copy()
    w.columns = ['season', 'team_id', 'ftm', 'fta', 'ast', 'fgm', 'blk', 'stl', 'opp_fga']

    # Loser perspective
    el = det[['Season', 'LTeamID', 'LFTM', 'LFTA', 'LAst', 'LFGM', 'LBlk', 'LStl', 'WFGA']].copy()
    el.columns = ['season', 'team_id', 'ftm', 'fta', 'ast', 'fgm', 'blk', 'stl', 'opp_fga']

    games = pd.concat([w, el], ignore_index=True)
    for col in games.columns[2:]:
        games[col] = pd.to_numeric(games[col], errors='coerce')

    records = []
    for (season, team_id), grp in games.groupby(['season', 'team_id']):
        total_fta = grp['fta'].sum()
        total_ftm = grp['ftm'].sum()
        records.append({
            'season':   int(season),
            'team_id':  int(team_id),
            # Season totals FT% (avoids undefined per-game rates when FTA=0)
            'ft_pct':   total_ftm / total_fta if total_fta >= 10 else np.nan,
            # Per-game means (handles schedule-length differences)
            'ast_rate': (grp['ast'] / grp['fgm'].replace(0, np.nan)).mean(),
            'blk_rate': (grp['blk'] / grp['opp_fga'].replace(0, np.nan)).mean(),
            'stl_rate': (grp['stl'] / grp['opp_fga'].replace(0, np.nan)).mean(),
        })

    out = pd.DataFrame(records)
    print(f"Detailed box features: {out.shape[0]:,} team-seasons  |  "
          f"{int(out['season'].min())}–{int(out['season'].max())}")
    _diag_season = int(out['season'].max())
    print(f"Sample distribution ({_diag_season} = March tournament year; latest in this pull):")
    print(out[out['season'] == _diag_season][['ft_pct', 'ast_rate', 'blk_rate', 'stl_rate']]
          .describe().round(3))
    return out


# ---------------------------------------------------------------------------
# Game-by-game variance features
# ---------------------------------------------------------------------------

def build_variance_features(kg_detailed, seasons):
    """Compute game-by-game variance features that capture upset risk.

    Season averages hide a critical factor: **consistency**. A team that
    scores 80 points every game is very different from one that scores
    100 one night and 60 the next — even though both average 80.

    In single-elimination tournaments, one bad game ends your season.
    High-variance teams are dangerous underdogs (can explode for a win)
    and risky favorites (can collapse for a loss).

    Produces per team-season:
      - ``scoring_consistency``: coefficient of variation of points scored
        (std / mean). Lower = more consistent. High CV favorites are upset-prone.
      - ``fg3_volatility``: std dev of 3-point shooting % across games.
        High volatility means the team can go ice-cold from 3 on any given night.
      - ``fg3_reliance``: share of total points that come from 3-pointers.
        High reliance + high volatility = maximum upset risk.
      - ``to_rate``: turnovers per game. Tournament pressure amplifies
        turnover-prone teams — more turnovers under stress.
      - ``to_volatility``: std dev of turnovers per game. Inconsistent
        ball security is a red flag in high-pressure games.

    Parameters
    ----------
    kg_detailed : pd.DataFrame
        Kaggle ``MRegularSeasonDetailedResults.csv`` with per-game box scores.
    seasons : iterable of int
        Which seasons to include.

    Returns
    -------
    pd.DataFrame or None
        One row per team-season with variance features.
    """
    if kg_detailed is None:
        print("[SKIP] MRegularSeasonDetailedResults.csv not available.")
        return None

    det = kg_detailed[kg_detailed['Season'].isin(seasons)].copy()

    # Winner perspective
    w = det[['Season', 'WTeamID', 'WScore', 'WFGM3', 'WFGA3', 'WTO',
             'WFGM', 'WFGA']].copy()
    w.columns = ['season', 'team_id', 'score', 'fgm3', 'fga3', 'to', 'fgm', 'fga']

    # Loser perspective
    el = det[['Season', 'LTeamID', 'LScore', 'LFGM3', 'LFGA3', 'LTO',
              'LFGM', 'LFGA']].copy()
    el.columns = ['season', 'team_id', 'score', 'fgm3', 'fga3', 'to', 'fgm', 'fga']

    games = pd.concat([w, el], ignore_index=True)
    for col in games.columns[2:]:
        games[col] = pd.to_numeric(games[col], errors='coerce')

    # Per-game shooting percentages
    games['fg3_pct'] = games['fgm3'] / games['fga3'].replace(0, np.nan)
    # Share of points from 3-pointers
    games['fg3_share'] = (games['fgm3'] * 3) / games['score'].replace(0, np.nan)

    records = []
    for (season, team_id), grp in games.groupby(['season', 'team_id']):
        n = len(grp)
        if n < 5:
            continue  # need enough games for meaningful variance
        records.append({
            'season':               int(season),
            'team_id':              int(team_id),
            # Scoring consistency: coefficient of variation (lower = more consistent)
            'scoring_consistency':  grp['score'].std() / grp['score'].mean()
                                    if grp['score'].mean() > 0 else np.nan,
            # 3pt volatility: std of 3pt% across games
            'fg3_volatility':       grp['fg3_pct'].std(),
            # 3pt reliance: avg share of points from 3s
            'fg3_reliance':         grp['fg3_share'].mean(),
            # Turnover rate: turnovers per game
            'to_rate':              grp['to'].mean(),
            # Turnover volatility: std of turnovers per game
            'to_volatility':        grp['to'].std(),
            'n_games':              n,
        })

    out = pd.DataFrame(records)
    print(f"Variance features: {out.shape[0]:,} team-seasons  |  "
          f"{int(out['season'].min())}–{int(out['season'].max())}")
    _diag_season = int(out['season'].max())
    print(f"Sample distribution ({_diag_season}):")
    print(out[out['season'] == _diag_season][
        ['scoring_consistency', 'fg3_volatility', 'fg3_reliance', 'to_rate', 'to_volatility']
    ].describe().round(3))
    return out


# ---------------------------------------------------------------------------
# Massey Ordinals consensus
# ---------------------------------------------------------------------------

def build_massey_consensus(kg_massey, seasons):
    """Average ~60 computer ranking systems into a single consensus rank per team.

    Used as a residual vs adj_em (same approach as seed_disagreement) to
    isolate the part of the analytics community's opinion that Barttorvik
    doesn't already explain.

    Parameters
    ----------
    kg_massey : pd.DataFrame
        Kaggle ``MMasseyOrdinals.csv``.
    seasons : iterable of int
        Which seasons to include.

    Returns
    -------
    pd.DataFrame or None
    """
    if kg_massey is None:
        print("[SKIP] MMasseyOrdinals.csv not available.")
        return None

    # Filter to final pre-tournament week to cut 5.8M → ~250K rows
    m = kg_massey[kg_massey['RankingDayNum'] >= 128].copy()
    m = m[m['Season'].isin(seasons)]

    # Keep only the latest available ranking day per (Season, SystemName)
    m = m.sort_values('RankingDayNum')
    m = m.groupby(['Season', 'SystemName', 'TeamID'], as_index=False)['OrdinalRank'].last()

    # Average across all systems → consensus rank per team-season
    consensus = (
        m.groupby(['Season', 'TeamID'])['OrdinalRank']
        .mean()
        .reset_index()
        .rename(columns={'Season': 'season', 'TeamID': 'team_id',
                         'OrdinalRank': 'consensus_rank'})
    )
    consensus['season'] = consensus['season'].astype(int)
    consensus['team_id'] = consensus['team_id'].astype(int)

    n_systems = m.groupby(['Season', 'SystemName']).ngroups
    print(f"Massey consensus: {len(consensus):,} team-seasons  |  "
          f"{int(consensus['season'].min())}–{int(consensus['season'].max())}  |  "
          f"~{n_systems // len(consensus['season'].unique())} systems/season")
    return consensus


# ---------------------------------------------------------------------------
# 2026 prediction-time feature assembly
# ---------------------------------------------------------------------------

def make_2026_features(teamA_id, teamB_id,
                       season_2026, feature_cols,
                       lr_seed_em, lr_cons=None,
                       reg_features_2026=None,
                       detailed_box_2026=None,
                       massey_2026=None):
    """Build a feature row for one 2026 matchup aligned to ``feature_cols``.

    Mirrors the historical ``build_matchup_dataset`` logic but for a single
    future game: looks up each team's stats, computes differences, and
    derives residual-based features (seed_disagreement, consensus_disagreement).

    Parameters
    ----------
    teamA_id, teamB_id : int
        Kaggle TeamIDs (Team A = favored / lower seed).
    season_2026 : pd.DataFrame
        2026 Barttorvik data with ``kaggle_team_id`` column.
    feature_cols : list of str
        The model's expected feature column names.
    lr_seed_em : fitted LinearRegression
        Predicts adj_em_diff from seed_diff (for seed_disagreement residual).
    lr_cons : fitted LinearRegression or None
        Predicts consensus_rank_diff from adj_em_diff (for consensus_disagreement).
    reg_features_2026 : pd.DataFrame or None
        2026 regular-season features (sos, close_win_pct, etc.).
    detailed_box_2026 : pd.DataFrame or None
        2026 detailed box score features (ft_pct, ast_rate, etc.).
    massey_2026 : pd.DataFrame or None
        2026 Massey consensus rankings.

    Returns
    -------
    pd.DataFrame (1 row) or None
        Feature row aligned to ``feature_cols``, or None if data missing.
    """
    def bart_row(tid):
        r = season_2026[season_2026['kaggle_team_id'] == tid]
        return r.iloc[0] if len(r) else None

    def reg_row(tid):
        if reg_features_2026 is None:
            return None
        r = reg_features_2026[reg_features_2026['team_id'] == tid]
        return r.iloc[0] if len(r) else None

    def det_row(tid):
        if detailed_box_2026 is None:
            return None
        r = detailed_box_2026[detailed_box_2026['team_id'] == tid]
        return r.iloc[0] if len(r) else None

    def mas_row(tid):
        if massey_2026 is None:
            return None
        r = massey_2026[massey_2026['team_id'] == tid]
        return r.iloc[0] if len(r) else None

    rA, rB = bart_row(teamA_id), bart_row(teamB_id)
    if rA is None or rB is None:
        return None

    BART = ['adj_o', 'adj_d', 'adj_em', 'barthag', 'adj_t',
            'off_efg', 'def_efg', 'off_to', 'def_to',
            'off_or', 'def_or', 'off_ftr', 'def_ftr', 'fg2_pct', 'fg3_pct', 'wab']
    COACH = ['coach_appearances', 'coach_tourn_wins', 'coach_final_fours', 'coach_win_rate']
    REG = ['sos']
    DETAILED = ['ft_pct', 'ast_rate', 'blk_rate', 'stl_rate']

    feat = {}

    for m in BART + COACH:
        va = pd.to_numeric(rA[m] if m in rA.index else np.nan, errors='coerce')
        vb = pd.to_numeric(rB[m] if m in rB.index else np.nan, errors='coerce')
        feat[f'{m}_diff'] = va - vb

    rA_r, rB_r = reg_row(teamA_id), reg_row(teamB_id)
    for m in REG:
        va = pd.to_numeric(rA_r[m] if rA_r is not None and m in rA_r.index else np.nan, errors='coerce')
        vb = pd.to_numeric(rB_r[m] if rB_r is not None and m in rB_r.index else np.nan, errors='coerce')
        feat[f'{m}_diff'] = va - vb

    rA_d, rB_d = det_row(teamA_id), det_row(teamB_id)
    for m in DETAILED:
        va = pd.to_numeric(rA_d[m] if rA_d is not None and m in rA_d.index else np.nan, errors='coerce')
        vb = pd.to_numeric(rB_d[m] if rB_d is not None and m in rB_d.index else np.nan, errors='coerce')
        feat[f'{m}_diff'] = va - vb

    sA = pd.to_numeric(rA['seed_num'] if 'seed_num' in rA.index else np.nan, errors='coerce')
    sB = pd.to_numeric(rB['seed_num'] if 'seed_num' in rB.index else np.nan, errors='coerce')

    # seed_diff_val is LOCAL — used only to compute seed_disagreement
    seed_diff_val = sA - sB

    feat['min_seed'] = np.nanmin([sA, sB]) if not (np.isnan(sA) or np.isnan(sB)) else np.nan
    feat['is_big_gap'] = int(abs(seed_diff_val) >= 8) if not np.isnan(seed_diff_val) else 0
    feat['is_late_round'] = 0

    adj_em_diff = feat.get('adj_em_diff', np.nan)

    # ── Asymmetric matchup interaction features ────────────────────────────
    # Mirror of build_matchup_dataset interactions — must stay in sync.

    # Offensive vs defensive matchup (asymmetric)
    va_o = pd.to_numeric(rA['adj_o'] if 'adj_o' in rA.index else np.nan, errors='coerce')
    vb_o = pd.to_numeric(rB['adj_o'] if 'adj_o' in rB.index else np.nan, errors='coerce')
    va_d = pd.to_numeric(rA['adj_d'] if 'adj_d' in rA.index else np.nan, errors='coerce')
    vb_d = pd.to_numeric(rB['adj_d'] if 'adj_d' in rB.index else np.nan, errors='coerce')
    feat['matchup_a_offense'] = va_o - vb_d
    feat['matchup_b_offense'] = vb_o - va_d

    # Tempo mismatch (absolute)
    va_t = pd.to_numeric(rA['adj_t'] if 'adj_t' in rA.index else np.nan, errors='coerce')
    vb_t = pd.to_numeric(rB['adj_t'] if 'adj_t' in rB.index else np.nan, errors='coerce')
    feat['tempo_mismatch'] = abs(va_t - vb_t) if not (np.isnan(va_t) or np.isnan(vb_t)) else np.nan

    # Late-round x efficiency interaction (is_late_round is always 0 at prediction time
    # for R64, but kept here so the feature vector shape matches training)
    feat['late_x_em'] = feat['is_late_round'] * adj_em_diff if not np.isnan(adj_em_diff) else np.nan

    # Seed disagreement: residual of adj_em_diff after controlling for seed_diff.
    # Positive = team is better than their seed implies.
    if not np.isnan(seed_diff_val) and not np.isnan(adj_em_diff):
        feat['seed_disagreement'] = adj_em_diff - lr_seed_em.predict([[seed_diff_val]])[0]
    else:
        feat['seed_disagreement'] = np.nan

    # Consensus disagreement: residual of consensus_rank_diff after adj_em_diff.
    rA_m, rB_m = mas_row(teamA_id), mas_row(teamB_id)
    cA = pd.to_numeric(rA_m['consensus_rank'] if rA_m is not None and 'consensus_rank' in rA_m.index else np.nan, errors='coerce')
    cB = pd.to_numeric(rB_m['consensus_rank'] if rB_m is not None and 'consensus_rank' in rB_m.index else np.nan, errors='coerce')
    if lr_cons is not None and not np.isnan(cA) and not np.isnan(cB) and not np.isnan(adj_em_diff):
        cons_rank_diff = cA - cB
        feat['consensus_disagreement'] = cons_rank_diff - lr_cons.predict([[adj_em_diff]])[0]
    else:
        feat['consensus_disagreement'] = np.nan

    row_df = pd.DataFrame([feat])
    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = np.nan
    return row_df[feature_cols]
