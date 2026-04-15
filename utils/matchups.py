"""
Matchup dataset construction.

Functions for building the historical tournament matchup dataset — one row
per game with Team A (favored) vs Team B (underdog), Barttorvik features
for both sides, coach features, difference columns, and target variables.
"""

import numpy as np
import pandas as pd

from .cleaning import parse_seed
from .features import is_late_round


# ---------------------------------------------------------------------------
# Per-team feature lookup
# ---------------------------------------------------------------------------

def get_team_features(team_id, season, bart_merged, prefix):
    """Look up Barttorvik features for a specific team-season.

    Returns a dict with all features prefixed by ``prefix`` (e.g., ``'a_'``
    for Team A). Used during matchup construction to build parallel columns
    for each side of the game.

    Parameters
    ----------
    team_id : int
        Kaggle TeamID.
    season : int
        Tournament season year.
    bart_merged : pd.DataFrame
        Barttorvik data already merged with crosswalk (has ``kaggle_team_id``).
    prefix : str
        Column name prefix (``'a_'`` or ``'b_'``).

    Returns
    -------
    dict
        Feature dict, or empty dict if team not found.
    """
    mask = (bart_merged['kaggle_team_id'] == team_id) & (bart_merged['season'] == season)
    rows = bart_merged[mask]
    if len(rows) == 0:
        return {}
    row = rows.iloc[0]
    feature_cols = [
        'adj_o', 'adj_d', 'adj_em', 'barthag', 'adj_t',
        'off_efg', 'def_efg', 'off_to', 'def_to',
        'off_or', 'def_or', 'off_ftr', 'def_ftr',
        'fg2_pct', 'fg3_pct', 'wab',
    ]
    result = {}
    for col in feature_cols:
        if col in row.index:
            result[f"{prefix}{col}"] = row[col]
    return result


# ---------------------------------------------------------------------------
# Full matchup dataset builder
# ---------------------------------------------------------------------------

def build_matchup_dataset(kg_compact, kg_seeds, crosswalk, bart_teams,
                          tourn_seasons, coach_stats=None, player_agg=None):
    """Assemble the full matchup-level modeling dataset.

    This is the **primary output of Phase 1**. Each row represents one
    historical tournament game with features for both teams, difference
    columns, and three target variables (score_margin, game_result,
    total_points).

    Convention: Team A = lower seed (favored), Team B = higher seed (underdog).

    Parameters
    ----------
    kg_compact : pd.DataFrame
        Kaggle ``MNCAATourneyCompactResults.csv``.
    kg_seeds : pd.DataFrame
        Kaggle ``MNCAATourneySeeds.csv``.
    crosswalk : pd.DataFrame
        Barttorvik-to-Kaggle team name mapping.
    bart_teams : pd.DataFrame
        All Barttorvik team-season data.
    tourn_seasons : iterable of int
        Which seasons to include in the matchup dataset.
    coach_stats : pd.DataFrame or None
        Coach performance lookup (from ``build_coach_stats``).
    player_agg : pd.DataFrame or None
        Player aggregate data (legacy; typically None).

    Returns
    -------
    pd.DataFrame
    """
    # ── 1. Merge Barttorvik with crosswalk to get kaggle_team_id ──────────
    # So we can look up Barttorvik stats by (Season, TeamID) for each game.
    bart_merged = bart_teams.merge(
        crosswalk[['bart_name', 'kaggle_team_id', 'match_method']],
        left_on='team', right_on='bart_name', how='left'
    )
    bart_merged['kaggle_team_id'] = pd.to_numeric(
        bart_merged['kaggle_team_id'], errors='coerce'
    )

    # ── 2. Build seed lookup: Season + TeamID → seed ──────────────────────
    seeds = kg_seeds.copy()
    seeds['seed_num'] = seeds['Seed'].apply(parse_seed)
    seed_lookup = seeds.set_index(['Season', 'TeamID'])['seed_num'].to_dict()

    # ── 3. Build coach lookup: Season + TeamID → coach stats ─────────────
    if coach_stats is not None:
        coach_lookup = coach_stats.set_index(['Season', 'TeamID'])
    else:
        coach_lookup = None

    # ── 4. Build player aggregate lookup ──────────────────────────────────
    if player_agg is not None and 'kaggle_team_id' in player_agg.columns:
        player_lookup = player_agg.set_index(['season', 'kaggle_team_id'])
    else:
        player_lookup = None

    # ── 5. Iterate over tournament games ──────────────────────────────────
    rows = []
    skipped = 0

    games = kg_compact[kg_compact['Season'].isin(tourn_seasons)].copy()

    for _, game in games.iterrows():
        season = int(game['Season'])
        w_id = int(game['WTeamID'])
        l_id = int(game['LTeamID'])
        w_score = int(game['WScore'])
        l_score = int(game['LScore'])

        # Seeds
        w_seed = seed_lookup.get((season, w_id), np.nan)
        l_seed = seed_lookup.get((season, l_id), np.nan)

        # Team A = lower seed (favored), Team B = higher seed (underdog)
        if not np.isnan(w_seed) and not np.isnan(l_seed):
            if w_seed <= l_seed:
                teamA_id, teamB_id = w_id, l_id
                teamA_score, teamB_score = w_score, l_score
                teamA_seed, teamB_seed = w_seed, l_seed
            else:
                teamA_id, teamB_id = l_id, w_id
                teamA_score, teamB_score = l_score, w_score
                teamA_seed, teamB_seed = l_seed, w_seed
        else:
            teamA_id, teamB_id = w_id, l_id
            teamA_score, teamB_score = w_score, l_score
            teamA_seed, teamB_seed = w_seed, l_seed

        # Targets
        score_margin = teamA_score - teamB_score
        game_result = 1 if score_margin > 0 else 0
        total_points = teamA_score + teamB_score

        # ── Barttorvik features for each team ─────────────────────────────
        feat_a = get_team_features(teamA_id, season, bart_merged, 'a_')
        feat_b = get_team_features(teamB_id, season, bart_merged, 'b_')

        # Skip games where Barttorvik data is missing for either team
        if not feat_a or not feat_b:
            skipped += 1
            continue

        # Seed-based features
        seed_diff = (teamA_seed - teamB_seed) if not (np.isnan(teamA_seed) or np.isnan(teamB_seed)) else np.nan
        min_seed = np.nanmin([teamA_seed, teamB_seed]) if not (np.isnan(teamA_seed) or np.isnan(teamB_seed)) else np.nan
        is_big_gap = int(abs(seed_diff) >= 8) if not np.isnan(seed_diff) else 0

        # Round bucket (early vs late) using DayNum as proxy
        daynum = game.get('DayNum', np.nan)
        late_round = is_late_round(daynum)

        row = {
            # Identifiers
            'season':        season,
            'teamA_id':      teamA_id,
            'teamB_id':      teamB_id,
            'teamA_seed':    teamA_seed,
            'teamB_seed':    teamB_seed,
            'seed_diff':     seed_diff,
            'min_seed':      min_seed,
            'is_big_gap':    is_big_gap,
            'is_late_round': late_round,
            'daynum':        daynum,
            # Targets
            'score_margin':  score_margin,
            'game_result':   game_result,
            'total_points':  total_points,
            'teamA_score':   teamA_score,
            'teamB_score':   teamB_score,
        }

        row.update(feat_a)
        row.update(feat_b)

        # ── Coach features ────────────────────────────────────────────────
        if coach_lookup is not None:
            for team_label, team_id in [('a', teamA_id), ('b', teamB_id)]:
                if (season, team_id) in coach_lookup.index:
                    coach_row = coach_lookup.loc[(season, team_id)]
                    for col in ['coach_appearances', 'coach_tourn_wins',
                                'coach_final_fours', 'coach_win_rate']:
                        if col in coach_row.index:
                            row[f"{team_label}_{col}"] = coach_row[col]

        rows.append(row)

    df = pd.DataFrame(rows)

    # ── 6. Compute difference features for all Barttorvik metrics ─────────
    bart_metrics = ['adj_o', 'adj_d', 'adj_em', 'barthag', 'adj_t',
                    'off_efg', 'def_efg', 'off_to', 'def_to',
                    'off_or', 'def_or', 'off_ftr', 'def_ftr',
                    'fg2_pct', 'fg3_pct', 'wab']
    for m in bart_metrics:
        ac, bc = f'a_{m}', f'b_{m}'
        if ac in df.columns and bc in df.columns:
            df[f'{m}_diff'] = df[ac] - df[bc]

    coach_metrics = ['coach_appearances', 'coach_tourn_wins',
                     'coach_final_fours', 'coach_win_rate']
    for m in coach_metrics:
        ac, bc = f'a_{m}', f'b_{m}'
        if ac in df.columns and bc in df.columns:
            df[f'{m}_diff'] = df[ac] - df[bc]

    # ── 7. Asymmetric matchup interaction features ─────────────────────────
    # Unlike symmetric diff features ("who is better overall"), these capture
    # how Team A's offense matches up against Team B's defense specifically.

    # Offensive-vs-defensive matchup: how much does A's offense beat B's defense?
    # Positive = A's offense is stronger than B's defense (favorable matchup for A).
    if 'a_adj_o' in df.columns and 'b_adj_d' in df.columns:
        df['matchup_a_offense'] = df['a_adj_o'] - df['b_adj_d']
    if 'b_adj_o' in df.columns and 'a_adj_d' in df.columns:
        df['matchup_b_offense'] = df['b_adj_o'] - df['a_adj_d']

    # Tempo mismatch: absolute difference in adjusted tempo.
    # Teams that prefer very different paces create stylistic friction.
    if 'a_adj_t' in df.columns and 'b_adj_t' in df.columns:
        df['tempo_mismatch'] = (df['a_adj_t'] - df['b_adj_t']).abs()

    # Late-round x efficiency: in early rounds a large EM gap almost guarantees
    # a win; in later rounds both teams are elite so the interaction weakens.
    if 'is_late_round' in df.columns and 'adj_em_diff' in df.columns:
        df['late_x_em'] = df['is_late_round'] * df['adj_em_diff']

    # 3-point reliance vs perimeter defense (only if variance features merged in).
    # High fg3_reliance against a team with low def_efg = vulnerable matchup.
    if 'a_fg3_reliance' in df.columns and 'b_def_efg' in df.columns:
        df['fg3_vs_defense_a'] = df['a_fg3_reliance'] * df['b_def_efg']
    if 'b_fg3_reliance' in df.columns and 'a_def_efg' in df.columns:
        df['fg3_vs_defense_b'] = df['b_fg3_reliance'] * df['a_def_efg']

    print(f"\nMatchup dataset built: {len(df):,} games "
          f"({skipped} skipped — missing Barttorvik data)")
    return df
