# --- Cleaning & standardization ---
from .cleaning import (
    wrangle_basic,
    parse_seed,
    is_power6_conf,
    normalize_bart_columns,
    build_crosswalk,
    POWER6_CONFS,
    MANUAL_OVERRIDES,
)

# --- Data fetching & caching ---
from .data_fetch import (
    safe_request,
    load_or_fetch,
    fetch_barttorvik_season,
    fetch_all_barttorvik,
    try_kaggle_download,
    load_kaggle_file,
)

# --- Feature engineering ---
from .features import (
    is_late_round,
    add_diff_features,
    build_reg_season_features,
    build_detailed_box_features,
    build_variance_features,
    build_massey_consensus,
    make_2026_features,
)

# --- Matchup construction ---
from .matchups import (
    get_team_features,
    build_matchup_dataset,
)

# --- Visualization helpers ---
from .viz import (
    is_boolean_col,
    team_name,
    get_upset_threshold,
    upset_flag,
    print_game,
    explain_upset,
    FEAT_LABELS,
)

# --- Existing utilities ---
from .datetime_features import add_datetime_features
from .encoding import bin_rare_categories
from .transforms import transform_skew
from .imputation import impute_missing
from .outliers import cap_outliers_iqr
from .regression import run_regression
from .coach_features import build_coach_stats

__all__ = [
    # Cleaning
    "wrangle_basic", "parse_seed", "is_power6_conf",
    "normalize_bart_columns", "build_crosswalk",
    "POWER6_CONFS", "MANUAL_OVERRIDES",
    # Data fetch
    "safe_request", "load_or_fetch",
    "fetch_barttorvik_season", "fetch_all_barttorvik",
    "try_kaggle_download", "load_kaggle_file",
    # Features
    "is_late_round", "add_diff_features",
    "build_reg_season_features", "build_detailed_box_features", "build_variance_features",
    "build_massey_consensus", "make_2026_features",
    # Matchups
    "get_team_features", "build_matchup_dataset",
    # Viz
    "is_boolean_col", "team_name", "get_upset_threshold",
    "upset_flag", "print_game", "explain_upset", "FEAT_LABELS",
    # Existing
    "add_datetime_features", "bin_rare_categories",
    "transform_skew", "impute_missing", "cap_outliers_iqr",
    "run_regression", "build_coach_stats",
]
