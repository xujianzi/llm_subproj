

## Model Architecture

Both models share the same dual-stream design: a temporal encoder for dynamic features and an MLP for static features, fused before the prediction head.

### LSTM

```
x_dyn  (B, T, 3)                x_static (B, 15)
       │                                 │
       ▼                                 ▼
    LSTM                           Linear(15 → Hs)
  (3 → H, L layers)                    ReLU
       │                                 │
  dyn_hidden (B, T, H)         static_hidden (B, Hs)
                                          │
                                  unsqueeze + expand
                                          │
                                 static_context (B, T, Hs)
       │                                  │
       └──────────── concat ──────────────┘
                        │
                  fusion (B, T, H+Hs)
                        │
                   Linear(H+Hs → 1)
                        │
                  y_pred (B, T, 1)
```

### Transformer

```
x_dyn  (B, T, 3)                x_static (B, 15)
       │                                 │
  Linear(3 → d_model)            Linear(15 → Hs)
       │                                ReLU
  PositionalEncoding                     │
       │                        static_hidden (B, Hs)
  TransformerEncoder                      │
  (nhead, dim_ff, L layers)       unsqueeze + expand
       │                                  │
  dyn_hidden (B, T, d_model)    static_context (B, T, Hs)
       │                                  │
       └──────────── concat ──────────────┘
                        │
                  fusion (B, T, d_model+Hs)
                        │
                   Linear(d_model+Hs → 1)
                        │
                  y_pred (B, T, 1)
```

