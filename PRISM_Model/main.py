"""
main.py – Training entry point for PRISM COVID prediction models.

Modify CONFIG in config.py to switch models and hyperparameters.
"""

from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loader import get_loaders
from model import build_model
from config import CONFIG

BASE_DIR = Path(__file__).parent


# ─────────────────────────────────────────────
# Training / evaluation helpers
# ─────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for x_dyn, x_static, y in loader:
            x_dyn    = x_dyn.to(device)     # (B, T, F_dyn)
            x_static = x_static.to(device)  # (B, F_static)
            y        = y.to(device)          # (B, T, 1)

            pred = model(x_dyn, x_static)    # (B, T, 1)
            loss = criterion(pred, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * x_dyn.size(0)
    return total_loss / len(loader.dataset)


def compute_r2(model, loader, device) -> float:
    """R² = 1 - SS_res / SS_tot, computed over all predictions in the loader."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_dyn, x_static, y in loader:
            pred = model(x_dyn.to(device), x_static.to(device))
            all_preds.append(pred.cpu())
            all_targets.append(y)
    preds   = torch.cat(all_preds).flatten()
    targets = torch.cat(all_targets).flatten()
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    return 1 - (ss_res / ss_tot).item()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    cfg = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {cfg['model']}")

    # ── Data ──────────────────────────────────
    train_loader, test_loader, n_dynamic, n_static, seq_len = get_loaders(
        cfg["x_train"], cfg["y_train"],
        cfg["x_test"],  cfg["y_test"],
        cfg["batch_size"],
    )
    print(f"n_dynamic={n_dynamic}  n_static={n_static}  seq_len={seq_len}  "
          f"train_zips={len(train_loader.dataset)}  test_zips={len(test_loader.dataset)}")

    # ── Model ─────────────────────────────────
    model_kwargs = {
        "dropout":            cfg["dropout"],
        "static_hidden_size": cfg["static_hidden_size"],
        "num_layers":         cfg["num_layers"],
    }
    if cfg["model"] == "lstm":
        model_kwargs.update({
            "hidden_size":   cfg["hidden_size"],
            "bidirectional": cfg["bidirectional"],
        })
    else:
        model_kwargs.update({
            "d_model":         cfg["d_model"],
            "nhead":           cfg["nhead"],
            "dim_feedforward": cfg["dim_feedforward"],
        })

    model = build_model(
        cfg["model"],
        dyn_input_size=n_dynamic,
        static_input_size=n_static,
        **model_kwargs,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # 计算所有参数量
    print(f"Parameters: {n_params:,}")

    # ── Training ──────────────────────────────
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    save_dir = BASE_DIR / cfg["save_dir"]
    save_dir.mkdir(exist_ok=True)
    best_path = save_dir / f"best_{cfg['model']}.pt"

    best_loss = float("inf")
    no_improve = 0

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        test_loss  = run_epoch(model, test_loader,  criterion, optimizer, device, train=False)
        scheduler.step(test_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{cfg['epochs']}  "
                  f"train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if test_loss < best_loss:
            best_loss = test_loss
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch}. Best test_loss={best_loss:.4f}")
                break

    # ── Final evaluation with best checkpoint ─
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    train_r2 = compute_r2(model, train_loader, device)
    test_r2  = compute_r2(model, test_loader,  device)
    print(f"\nTraining done. Best test_loss={best_loss:.4f}  saved to {best_path}")
    print(f"R2  train={train_r2:.4f}  test={test_r2:.4f}")


if __name__ == "__main__":
    main()
