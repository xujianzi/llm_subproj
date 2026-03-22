"""
config.py – Central configuration for PRISM model training.
"""

DYNAMIC_COLS = [
    "median_non_home_dwell_time_lag4",
    "non_home_ratio_lag4",
    "full_time_work_behavior_devices_lag4",
    # "CSVI_score",
]

STATIC_COLS = [
    "pct_less_than_9th_grade",
    "pct_lep",
    "pct_hispanic",
    "pct_non_hispanic_black",
    "pct_senior",
    "pct_young_adults",
    "pct_below_poverty",
    "pct_unemployed",
    "pct_uninsured",
    "per_capita_income",
    "pct_female_headed_households",
    "pct_overcrowded_housing",
    "pct_households_without_a_vehicle",
    "pct_work_at_home",
    "pct_service",
]

CONFIG = {
    # data
    "x_train":    "data/X_train_scaled.csv",
    "y_train":    "data/y_train_scaled.csv",
    "x_test":     "data/X_test_scaled.csv",
    "y_test":     "data/y_test_scaled.csv",
    "batch_size": 16,

    # model: "lstm" or "transformer"
    "model":      "lstm",   # lstm | transformer
    "dropout":    0.1,

    # LSTM hyperparams
    "hidden_size":   256,
    "num_layers":    1,
    "bidirectional": False,

    # static MLP hidden size
    "static_hidden_size": 128,

    # Transformer hyperparams
    "d_model":         128,
    "nhead":           4,
    "dim_feedforward": 128,

    # training
    "epochs":       100,
    "lr":           1e-3,
    "weight_decay": 1e-3,
    "patience":     150,
    "save_dir":  "checkpoints",
}
