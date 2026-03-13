
import torch
import torch.nn as nn 
from torch.optim import Adam, AdamW, SGD
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

"""
建立网络结构
"""




class TorchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        vocab_size = AutoConfig.from_pretrained(config["pretrain_model_path"]).vocab_size # get bert vocab size
        num_labels = config["num_labels"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        # (B, L, H) 
        elif model_type == "bert":
            self.use_bert = True 
            self.encoder = AutoModel.from_pretrained(config["pretrain_model_path"], return_dict=False) # return_dict=False → 返回 tuple
            hidden_size = self.encoder.config.hidden_size

        self.classify = nn.Sequential(
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(hidden_size, num_labels),
        )
        self.pooling_style = config["pooling_style"]

        # class_weights: list[float] | None — moves to correct device automatically via register_buffer
        weights = config.get("class_weights")
        if weights is not None:
            self.register_buffer("class_weight", torch.tensor(weights, dtype=torch.float))
        else:
            self.register_buffer("class_weight", None)
        self.loss = nn.CrossEntropyLoss  # store class, instantiate in forward with correct weight
    
    def forward(self, x, attention_mask=None, target = None):
        if self.use_bert:
            x = self.encoder(x, attention_mask=attention_mask)
            # return_dict=False -> (last_hidden_state, pooler_output)
            # pooler_output: CLS token through dense+tanh, designed for classification
            x = x[1]  # (B, H)
        else:
            emb = self.embedding(x)  # (B, L, H)
            x = self.encoder(emb)    # (B, L, H)

            if isinstance(x, tuple):  # RNN/LSTM/GRU 类的模型同时会返回隐单元向量
                x = x[0]

            if self.pooling_style == "max":
                x = x.max(dim=1).values   # torch中max返回（values,indices）
            else:
                x = x.mean(dim=1)         # (B, H)
            # 也可以直接使用 CLS 位置的向量: x = x[:, 0, :]
        predict = self.classify(x)  # (B, C)
        if target is not None:
            loss_fn = self.loss(weight=self.class_weight)
            return loss_fn(predict, target.squeeze())
        else:
            return predict


def choose_optimizer(config, model):
    """
    Build optimizer with optional differential learning rates.

    Config keys:
        optimizer     : "adam" | "adamw" | "sgd"
        learning_rate : LR for BERT encoder / all params when head_lr is None
        head_lr       : LR for classification head (None = same as learning_rate)
        weight_decay  : L2 penalty, applied to non-bias/non-LayerNorm params (default 0.01 for adamw)
    """
    opt_name     = config["optimizer"]
    lr           = config["learning_rate"]
    head_lr      = config.get("head_lr")          # None → uniform LR
    weight_decay = config.get("weight_decay", 0.01 if opt_name == "adamw" else 0.0)

    # ── build parameter groups ──────────────────────────────────────────────
    if head_lr is not None and hasattr(model, "classify"):
        classify_ids  = {id(p) for p in model.classify.parameters()}
        backbone_params = [p for p in model.parameters() if id(p) not in classify_ids]
        head_params     = list(model.classify.parameters())
        param_groups = [
            {"params": backbone_params, "lr": lr,      "weight_decay": weight_decay},
            {"params": head_params,     "lr": head_lr, "weight_decay": 0.0},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": lr, "weight_decay": weight_decay}]

    # ── choose optimizer class ───────────────────────────────────────────────
    if opt_name == "adamw":
        return AdamW(param_groups)
    elif opt_name == "adam":
        return Adam(param_groups)
    elif opt_name == "sgd":
        return SGD(param_groups)


def build_scheduler(config, optimizer, num_training_steps):
    """
    Return a LR scheduler, or None if config["scheduler"] is falsy.

    Config keys:
        scheduler     : None | "linear_warmup" | "cosine_warmup"
        warmup_ratio  : float, proportion of total steps used for warmup (default 0.1)

    Usage in training loop:
        scheduler = build_scheduler(config, optimizer, total_steps)
        ...
        optimizer.step()
        if scheduler:
            scheduler.step()
    """
    scheduler_type = config.get("scheduler")
    if not scheduler_type:
        return None

    warmup_steps = int(num_training_steps * config.get("warmup_ratio", 0.1))

    if scheduler_type == "linear_warmup":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    if scheduler_type == "cosine_warmup":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    raise ValueError(f"Unknown scheduler: {scheduler_type!r}")


if __name__ == "__main__":
    # 这里一般是测试用例
    from config import Config

    Config["model_type"] = "lstm"
    
    model = TorchModel(Config)
    x = torch.LongTensor([[0,1,2,3,4], [5,6,7,8,9], [4,5,3,4,2]])  # (B, L)
    print(x.shape) # (3,5)
    predict = model(x)
    print(predict)
    print(predict.shape) # (3,3)
