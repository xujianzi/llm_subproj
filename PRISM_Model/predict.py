"""
predict.py – Load trained model, run inference on test set, plot results.

Displays a 3×5 grid (one subplot per test zipcode) showing
actual vs predicted COVID time series.
"""

from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from loader import ZipcodeDataset
from model import build_model
from config import CONFIG, DYNAMIC_COLS, STATIC_COLS

BASE_DIR = Path(__file__).parent


def predict(model_type: str = CONFIG["model"]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load dataset ──────────────────────────
    ds = ZipcodeDataset(CONFIG["x_test"], CONFIG["y_test"])

    # Load dates separately for x-axis labels
    x_df = pd.read_csv(BASE_DIR / CONFIG["x_test"], dtype={"zipcode": str})
    x_df = x_df.sort_values(["zipcode", "Week_Start_Date"]).reset_index(drop=True)
    x_df["Week_Start_Date"] = pd.to_datetime(x_df["Week_Start_Date"])

    # ── Build and load model ──────────────────
    model_kwargs = {
        "dropout":            CONFIG["dropout"],
        "static_hidden_size": CONFIG["static_hidden_size"],
        "num_layers":         CONFIG["num_layers"],
    }
    if model_type == "lstm":
        model_kwargs.update({
            "hidden_size":   CONFIG["hidden_size"],
            "bidirectional": CONFIG["bidirectional"],
        })
    else:
        model_kwargs.update({
            "d_model":         CONFIG["d_model"],
            "nhead":           CONFIG["nhead"],
            "dim_feedforward": CONFIG["dim_feedforward"],
        })

    model = build_model(
        model_type,
        dyn_input_size=len(DYNAMIC_COLS),
        static_input_size=len(STATIC_COLS),
        **model_kwargs,
    ).to(device)

    ckpt_path = BASE_DIR / CONFIG["save_dir"] / f"best_{model_type}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # ── Load inverse-transform stats ──────────
    stats = pd.read_csv(
        BASE_DIR / "data/y_test_stats.csv", dtype={"zipcode": str}
    ).set_index("zipcode")

    # ── Inference ─────────────────────────────
    n_zips = len(ds)
    fig, axes = plt.subplots(3, 5, figsize=(20, 10), sharey=False)
    axes = axes.flatten()

    with torch.no_grad():
        for i in range(n_zips):
            x_dyn, x_static, y_true = ds[i]

            pred = model(
                x_dyn.unsqueeze(0).to(device),
                x_static.unsqueeze(0).to(device),
            ).squeeze().cpu().numpy()               # (T,)

            y_true = y_true.squeeze().numpy()       # (T,)
            zip_code = ds.zipcodes[i]

            # inverse z-score: x_orig = z * std + mean
            mean = stats.loc[zip_code, "Weekly_New_Cases_per_100k_mean"]
            std  = stats.loc[zip_code, "Weekly_New_Cases_per_100k_std"]
            y_true = y_true * std + mean
            pred   = pred   * std + mean

            dates = x_df.loc[x_df["zipcode"] == zip_code, "Week_Start_Date"].values

            ax = axes[i]
            ax.plot(dates, y_true, label="Actual",    color="steelblue", linewidth=1.5)
            ax.plot(dates, pred,   label="Predicted", color="tomato",    linewidth=1.5, linestyle="--")
            ax.set_title(f"ZIP {zip_code}", fontsize=9)
            ax.set_ylabel("Cases per 100k", fontsize=7)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.tick_params(axis="x", labelsize=7, rotation=30)
            ax.tick_params(axis="y", labelsize=7)

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10)
    fig.suptitle(f"Test Set – Actual vs Predicted ({model_type.upper()})", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(BASE_DIR / f"predict_{model_type}.png", dpi=150)
    plt.show()
    print(f"Saved to predict_{model_type}.png")


if __name__ == "__main__":
    predict()
