"""
model.py – LSTM and Transformer models for zipcode-level COVID sequence prediction.

LSTMModel architecture:
    x_dyn    (B, T, F_dyn)  ──► LSTM ──► dyn_hidden   (B, T, H)
    x_static (B, F_static)  ──► MLP  ──► static_hidden (B, Hs)
                                              │ unsqueeze+repeat
                                         static_context (B, T, Hs)
    concat(dyn_hidden, static_context)  ──► (B, T, H+Hs) ──► Linear ──► y (B, T, 1)
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# LSTM with static context fusion
# ─────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(
        self,
        dyn_input_size: int,
        static_input_size: int,
        hidden_size: int = 32,
        static_hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=dyn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        dyn_out = hidden_size * (2 if bidirectional else 1)

        self.static_mlp = nn.Sequential(
            nn.Linear(static_input_size, static_hidden_size),
            nn.ReLU(),
        )

        self.head = nn.Linear(dyn_out + static_hidden_size, 1)

    def forward(self, x_dyn: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        # x_dyn:    (B, T, F_dyn)
        # x_static: (B, F_static)

        dyn_hidden, _ = self.lstm(x_dyn)                         # (B, T, H)

        static_hidden = self.static_mlp(x_static)                # (B, Hs)
        static_context = static_hidden.unsqueeze(1).expand(      # (B, T, Hs)
            -1, dyn_hidden.size(1), -1
        )

        fusion = torch.cat([dyn_hidden, static_context], dim=-1) # (B, T, H+Hs)
        return self.head(fusion)                                  # (B, T, 1)


# ─────────────────────────────────────────────
# Transformer with static context fusion
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerModel(nn.Module):
    def __init__(
        self,
        dyn_input_size: int,
        static_input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        static_hidden_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.input_proj = nn.Linear(dyn_input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.static_mlp = nn.Sequential(
            nn.Linear(static_input_size, static_hidden_size),
            nn.ReLU(),
        )

        self.head = nn.Linear(d_model + static_hidden_size, 1)

    def forward(self, x_dyn: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        # x_dyn:    (B, T, F_dyn)
        # x_static: (B, F_static)

        x = self.input_proj(x_dyn)                               # (B, T, d_model)
        x = self.pos_enc(x)
        dyn_hidden = self.encoder(x)                             # (B, T, d_model)

        static_hidden = self.static_mlp(x_static)                # (B, Hs)
        static_context = static_hidden.unsqueeze(1).expand(      # (B, T, Hs)
            -1, dyn_hidden.size(1), -1
        )

        fusion = torch.cat([dyn_hidden, static_context], dim=-1) # (B, T, d_model+Hs)
        return self.head(fusion)                                  # (B, T, 1)


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_model(model_type: str, dyn_input_size: int, static_input_size: int, **kwargs) -> nn.Module:
    if model_type == "lstm":
        return LSTMModel(dyn_input_size=dyn_input_size, static_input_size=static_input_size, **kwargs)
    elif model_type == "transformer":
        return TransformerModel(dyn_input_size=dyn_input_size, static_input_size=static_input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'lstm' or 'transformer'.")


if __name__ == "__main__":
    B, T, F_dyn, F_static = 8, 38, 4, 15
    x_dyn    = torch.randn(B, T, F_dyn)
    x_static = torch.randn(B, F_static)

    lstm = LSTMModel(dyn_input_size=F_dyn, static_input_size=F_static, hidden_size=32, static_hidden_size=16)
    print("LSTM out:", lstm(x_dyn, x_static).shape)          # (8, 38, 1)

    tfm = TransformerModel(dyn_input_size=F_dyn, static_input_size=F_static, d_model=64, nhead=4)
    print("Transformer out:", tfm(x_dyn, x_static).shape)    # (8, 38, 1)
