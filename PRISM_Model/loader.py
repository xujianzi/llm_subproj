"""
loader.py – ZipcodeDataset

Each sample = one zipcode's full time series.
  x_dyn    : (T, F_dyn)   dynamic mobility features
  x_static : (F_static,)  static ACS features (constant across T)
  y        : (T, 1)       COVID target
"""

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import CONFIG, DYNAMIC_COLS, STATIC_COLS

BASE_DIR = Path(__file__).parent


class ZipcodeDataset(Dataset):
    def __init__(
        self,
        x_path: str = CONFIG["x_train"],
        y_path: str = CONFIG["y_train"],
    ):
        x_df = pd.read_csv(BASE_DIR / x_path, dtype={"zipcode": str})
        y_df = pd.read_csv(BASE_DIR / y_path)

        x_df = x_df.sort_values(["zipcode", "Week_Start_Date"]).reset_index(drop=True)
        y_df = y_df.loc[x_df.index].reset_index(drop=True)

        self.zipcodes = x_df["zipcode"].unique().tolist()

        self.x_dyns, self.x_statics, self.ys = [], [], []
        for zip_code in self.zipcodes:
            mask = x_df["zipcode"] == zip_code
            rows = x_df.loc[mask]

            x_dyn    = rows[DYNAMIC_COLS].values.astype("float32")       # (T, F_dyn)
            x_static = rows[STATIC_COLS].iloc[0].values.astype("float32") # (F_static,)
            y_seq    = y_df.loc[mask].values.astype("float32")            # (T, 1)

            self.x_dyns.append(torch.tensor(x_dyn))
            self.x_statics.append(torch.tensor(x_static))
            self.ys.append(torch.tensor(y_seq))

    def __len__(self):
        return len(self.zipcodes)

    def __getitem__(self, idx):
        return self.x_dyns[idx], self.x_statics[idx], self.ys[idx]

    @property
    def n_dynamic(self):
        return len(DYNAMIC_COLS)

    @property
    def n_static(self):
        return len(STATIC_COLS)

    @property
    def seq_len(self):
        return self.x_dyns[0].shape[0]


def get_loaders(
    x_train: str = CONFIG["x_train"],
    y_train: str = CONFIG["y_train"],
    x_test: str  = CONFIG["x_test"],
    y_test: str  = CONFIG["y_test"],
    batch_size: int = CONFIG["batch_size"],
    num_workers: int = 0,
):
    train_ds = ZipcodeDataset(x_train, y_train)
    test_ds  = ZipcodeDataset(x_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_ds.n_dynamic, train_ds.n_static, train_ds.seq_len


if __name__ == "__main__":
    train_loader, test_loader, n_dynamic, n_static, seq_len = get_loaders()
    x_dyn, x_static, y = next(iter(train_loader))
    print(f"x_dyn:    {x_dyn.shape}")     # (B, T, F_dyn)
    print(f"x_static: {x_static.shape}")  # (B, F_static)
    print(f"y:        {y.shape}")          # (B, T, 1)
    print(f"n_dynamic={n_dynamic}  n_static={n_static}  seq_len={seq_len}")
