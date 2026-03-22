import pandas as pd
import numpy as np
import os
from config import DYNAMIC_COLS, STATIC_COLS
# Columns to extract from ACS data (per datause.txt)
# Note: 'pct_unisured' is a typo in the raw data for 'pct_uninsured'
ACS_COLS = [
    "pct_less_than_9th_grade",
    "pct_lep",
    "pct_hispanic",
    "pct_non_hispanic_black",
    "pct_senior",
    "pct_young_adults",
    "pct_below_poverty",
    "pct_unemployed",
    "pct_unisured",           # raw column name (typo in source)
    "per_capita_income",
    "pct_female_headed_households",
    "pct_overcrowded_housing",
    "pct_households_without_a_vehicle",
    "pct_work_at_home",
    "pct_service",
]


def merge_acs_with_train(
    acs_path: str = "data/raw/acs_yearly.csv",
    train_path: str = "data/raw/train_data.csv",
    output_path: str = "data/data.csv",
) -> pd.DataFrame:
    """
    Extract ACS features and merge them into the weekly train dataset.

    ACS data is yearly; each weekly row in train_data receives the ACS values
    for the matching (zipcode, year) pair.

    Parameters
    ----------
    acs_path    : path to acs_yearly.csv
    train_path  : path to the raw train_data.csv
    output_path : where to write the merged CSV

    Returns
    -------
    Merged DataFrame
    """
    # --- Load ACS and keep only needed columns ---
    acs = pd.read_csv(acs_path, dtype={"zipcode": str})
    acs_keep = ["zipcode", "year"] + ACS_COLS
    acs = acs[acs_keep].copy()
    # Rename the typo column to the correct name
    acs.rename(columns={"pct_unisured": "pct_uninsured"}, inplace=True)

    # --- Load train data ---
    train = pd.read_csv(train_path, dtype={"zipcode": str})
    train["Week_Start_Date"] = pd.to_datetime(train["Week_Start_Date"])
    train["year"] = train["Week_Start_Date"].dt.year

    # --- Merge on (zipcode, year) ---
    merged = train.merge(acs, on=["zipcode", "year"], how="left")
    merged.drop(columns=["year"], inplace=True)

    # --- Save ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved merged data to {output_path}  shape={merged.shape}")
    return merged


def split_train_test(
    data_path: str = "data/data.csv",
    train_path: str = "data/train.csv",
    test_path: str = "data/test.csv",
    train_ratio: float = 0.85,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data.csv into train and test sets by time order (no shuffle).

    Rows are sorted by Week_Start_Date; the first train_ratio fraction
    becomes the training set and the remainder the test set.

    Parameters
    ----------
    data_path   : path to the merged data CSV
    train_path  : output path for training set
    test_path   : output path for test set
    train_ratio : fraction of data for training (default 0.85)

    Returns
    -------
    (train_df, test_df)
    """
    df = pd.read_csv(data_path, dtype={"zipcode": str})

    zipcodes = df["zipcode"].unique()
    rng = np.random.default_rng(random_seed)
    rng.shuffle(zipcodes)
    n_train = int(len(zipcodes) * train_ratio)
    train_zips = set(zipcodes[:n_train])

    train_df = df[df["zipcode"].isin(train_zips)].reset_index(drop=True)
    test_df = df[~df["zipcode"].isin(train_zips)].reset_index(drop=True)

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train: {len(train_zips)} zipcodes, {train_df.shape[0]} rows -> {train_path}")
    print(f"Test:  {len(zipcodes) - len(train_zips)} zipcodes, {test_df.shape[0]} rows -> {test_path}")
    return train_df, test_df


FEATURE_COLS = DYNAMIC_COLS + STATIC_COLS
TARGET_COL = "Weekly_New_Cases_per_100k"


def prepare_features(
    train_path: str = "data/train.csv",
    test_path: str = "data/test.csv",
    x_train_path: str = "data/X_train.csv",
    y_train_path: str = "data/y_train.csv",
    x_test_path: str = "data/X_test.csv",
    y_test_path: str = "data/y_test.csv",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Select features (mobility + ACS) and target from train/test splits.

    Features : MOBILITY_COLS + ACS columns (from datause.txt)
    Target   : Weekly_New_Cases_per_100k

    Returns
    -------
    (X_train, y_train, X_test, y_test)
    """
    train = pd.read_csv(train_path, dtype={"zipcode": str})
    test = pd.read_csv(test_path, dtype={"zipcode": str})

    X_train = train[["zipcode", "Week_Start_Date"] + FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test = test[["zipcode", "Week_Start_Date"] + FEATURE_COLS]
    y_test = test[TARGET_COL]

    X_train.to_csv(x_train_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    print(f"X_train: {X_train.shape} -> {x_train_path}")
    print(f"y_train: {y_train.shape} -> {y_train_path}")
    print(f"X_test:  {X_test.shape}  -> {x_test_path}")
    print(f"y_test:  {y_test.shape}  -> {y_test_path}")
    return X_train, y_train, X_test, y_test


def _zscore_within_zipcode(
    df: pd.DataFrame, cols: list[str], stats_path=None
) -> pd.DataFrame:
    """
    Z-score normalize cols within each zipcode independently.
    Each zipcode's time series is centered and scaled to unit variance.
    std=0 (constant sequence) → set to 0.

    If stats_path is provided, saves per-zipcode mean/std to CSV
    so predictions can be inverse-transformed back to original scale.
    """
    df = df.copy()
    stats = df.groupby("zipcode")[cols].agg(["mean", "std"])

    if stats_path is not None:
        stats_out = stats.copy()
        stats_out.columns = ["_".join(c) for c in stats_out.columns]
        stats_out.reset_index().to_csv(stats_path, index=False)

    for col in cols:
        mean = df["zipcode"].map(stats[col]["mean"])
        std  = df["zipcode"].map(stats[col]["std"]).replace(0, np.nan)
        df[col] = (df[col] - mean) / std
    df[cols] = df[cols].fillna(0)
    return df


def _zscore_across_zipcodes(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Z-score normalize cols using cross-zipcode statistics.

    For each feature:
      1. Compute per-zipcode mean (one value per zipcode).
      2. Compute global mean/std across those zipcode means.
      3. Normalize every row using these global stats and broadcast back.

    This scales features by their cross-zipcode distribution while
    preserving within-zipcode temporal variation for mobility features.
    """
    df = df.copy()
    zip_means = df.groupby("zipcode")[cols].mean()   # shape: (n_zips, n_cols)
    global_mean = zip_means.mean()                   # mean of zipcode means
    global_std = zip_means.std().replace(0, np.nan)  # std across zipcodes
    df[cols] = (df[cols] - global_mean) / global_std
    df[cols] = df[cols].fillna(0)
    return df


def normalize_features(
    x_train_path: str = "data/X_train.csv",
    y_train_path: str = "data/y_train.csv",
    x_test_path: str = "data/X_test.csv",
    y_test_path: str = "data/y_test.csv",
    x_train_out: str = "data/X_train_scaled.csv",
    y_train_out: str = "data/y_train_scaled.csv",
    x_test_out: str = "data/X_test_scaled.csv",
    y_test_out: str = "data/y_test_scaled.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Z-score normalize features and target per zipcode.

    For each zipcode, each numeric column is normalized independently:
        z = (x - mean_zip) / std_zip
    Constant columns within a zipcode (std=0) are set to 0.
    Train and test sets are normalized independently using their own stats.

    Returns
    -------
    (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
    """
    X_train = pd.read_csv(x_train_path, dtype={"zipcode": str})
    X_test = pd.read_csv(x_test_path, dtype={"zipcode": str})
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)

    # Attach zipcode to y for groupby, then detach
    y_train_zip = X_train[["zipcode"]].copy()
    y_train_zip[TARGET_COL] = y_train[TARGET_COL].values
    y_test_zip = X_test[["zipcode"]].copy()
    y_test_zip[TARGET_COL] = y_test[TARGET_COL].values

    # features: cross-zipcode z-score (preserves relative differences between zipcodes)
    X_train_scaled = _zscore_across_zipcodes(X_train, FEATURE_COLS)
    X_test_scaled  = _zscore_across_zipcodes(X_test,  FEATURE_COLS)
    # target: within-zipcode z-score (model learns the temporal shape, not absolute scale)
    # stats saved so predictions can be inverse-transformed back to original scale
    y_train_scaled = _zscore_within_zipcode(
        y_train_zip, [TARGET_COL], stats_path="data/y_train_stats.csv"
    )[[TARGET_COL]]
    y_test_scaled  = _zscore_within_zipcode(
        y_test_zip,  [TARGET_COL], stats_path="data/y_test_stats.csv"
    )[[TARGET_COL]]

    X_train_scaled.to_csv(x_train_out, index=False)
    y_train_scaled.to_csv(y_train_out, index=False)
    X_test_scaled.to_csv(x_test_out, index=False)
    y_test_scaled.to_csv(y_test_out, index=False)

    print(f"X_train_scaled: {X_train_scaled.shape} -> {x_train_out}")
    print(f"y_train_scaled: {y_train_scaled.shape} -> {y_train_out}")
    print(f"X_test_scaled:  {X_test_scaled.shape}  -> {x_test_out}")
    print(f"y_test_scaled:  {y_test_scaled.shape}  -> {y_test_out}")
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled


if __name__ == "__main__":
    merge_acs_with_train()
    split_train_test()
    prepare_features()
    normalize_features()
