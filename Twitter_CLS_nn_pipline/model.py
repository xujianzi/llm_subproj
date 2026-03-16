import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput



"""
建立网络结构 — 注册表模式解耦 encoder 选择

结构：
    EncoderBlock（各自独立的 nn.Module）
        BertEncoderBlock      : BERT → pooler_output (B, H)
        RecurrentEncoderBlock : Embedding + RNN/LSTM/GRU + Pooling → (B, H)

    ENCODER_REGISTRY : dict[str, callable]  — 注册表，新增模型只需在此注册
    build_encoder(config) → nn.Module       — 工厂函数

    TorchModel : encoder → classify，forward 无任何 if/else
"""


# ── Encoder Blocks ─────────────────────────────────────────────────────────────

class BertEncoderBlock(nn.Module):
    """
    BERT encoder 封装。
    forward 返回 pooler_output (B, H)：CLS token 经过 dense+tanh 的表示，
    专为分类任务设计。
    """

    def __init__(self, config: dict):
        super().__init__()
        # return_dict=False → 返回 tuple，避免依赖 ModelOutput 类型
        self.encoder = AutoModel.from_pretrained(
            config["pretrain_model_path"], return_dict=False
        )
        self.output_size: int = self.encoder.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # (last_hidden_state, pooler_output, ...)
        out = self.encoder(input_ids, attention_mask=attention_mask)
        return out[1]  # pooler_output: (B, H)


# RNN cell 类型注册表（供 RecurrentEncoderBlock 内部使用）
_CELL_REGISTRY: dict[str, type[nn.Module]] = {
    "rnn":  nn.RNN,
    "lstm": nn.LSTM,
    "gru":  nn.GRU,
}


class RecurrentEncoderBlock(nn.Module):
    """
    Embedding + RNN/LSTM/GRU + Pooling 封装。
    cell_type 由外部注册表决定，内部无 if/else。
    forward 返回池化后的序列表示 (B, H)。
    """

    def __init__(self, config: dict, cell_type: str):
        super().__init__()
        hidden_size = config["hidden_size"]
        vocab_size  = AutoConfig.from_pretrained(config["pretrain_model_path"]).vocab_size
        num_layers  = config["num_layers"]

        self.embedding    = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.encoder      = _CELL_REGISTRY[cell_type](
            hidden_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.pooling_style = config["pooling_style"]
        self.output_size: int = hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(input_ids)      # (B, L, H)
        x = self.encoder(x)
        if isinstance(x, tuple):           # LSTM/GRU 同时返回隐状态，取序列输出
            x = x[0]                       # (B, L, H)
        if self.pooling_style == "max":
            return x.max(dim=1).values     # (B, H)
        return x.mean(dim=1)               # (B, H)


class QwenEncoderBlock(nn.Module):
    """
    Qwen2.5（decoder-only LLM）作 feature extractor 的封装。
    以 float16 加载节省显存，forward 使用 attention_mask 加权均值池化输出 (B, H)。
    encoder 属性暴露给 lora_optimize.apply_lora 使用。
    """

    def __init__(self, config: dict):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            config["pretrain_model_path"],
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.output_size: int = self.encoder.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out    = self.encoder(input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, L, H)  float16
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()               # (B, L, 1) float32
            x    = (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            x = hidden.float().mean(1)                                # (B, H) float32
        return x  # float32，与 classify head 保持一致


# ── Registry ───────────────────────────────────────────────────────────────────

def _recurrent(cell_type: str):
    """生成与 BertEncoderBlock 签名一致的 RecurrentEncoderBlock 构造器。"""
    def _factory(config: dict) -> RecurrentEncoderBlock:
        return RecurrentEncoderBlock(config, cell_type=cell_type)
    return _factory


ENCODER_REGISTRY: dict[str, callable] = {
    "bert":      BertEncoderBlock,
    "roberta":   BertEncoderBlock,
    "rnn":       _recurrent("rnn"),
    "lstm":      _recurrent("lstm"),
    "gru":       _recurrent("gru"),
    "qwen_lora": QwenEncoderBlock,
}


def build_encoder(config: dict) -> nn.Module:
    """
    根据 config["model_type"] 从注册表构建并返回对应的 EncoderBlock。
    新增模型类型只需在 ENCODER_REGISTRY 中注册，无需修改此函数。
    """
    model_type = config["model_type"]
    if model_type not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Available: {list(ENCODER_REGISTRY)}"
        )
    return ENCODER_REGISTRY[model_type](config)


# ── TorchModel ─────────────────────────────────────────────────────────────────

class TorchModel(nn.Module):
    """
    顶层分类模型。
    encoder 由 build_encoder 构建，forward 无任何条件分支。
    """

    def __init__(self, config: dict):
        super().__init__()
        self.encoder  = build_encoder(config)
        hidden_size   = self.encoder.output_size
        num_labels    = config["num_labels"]

        self.classify = nn.Sequential(
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(hidden_size, num_labels),
        )

        weights = config.get("class_weights")
        if weights is not None:
            self.register_buffer("class_weight", torch.tensor(weights, dtype=torch.float))
        else:
            self.register_buffer("class_weight", None)
        self.loss = nn.CrossEntropyLoss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        x      = self.encoder(input_ids, attention_mask=attention_mask)  # (B, H)
        logits = self.classify(x)                                         # (B, C)
        loss   = None
        if labels is not None:
            loss = self.loss(weight=self.class_weight)(logits, labels.squeeze())
        return SequenceClassifierOutput(loss=loss, logits=logits)   #   用于outputs = model(**inputs)以及compute_metrics


# ── smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import Config, PRETRAIN_MODEL_MAP
    
    Config["model_type"] = "qwen_lora"
    Config["pretrain_model_path"] = PRETRAIN_MODEL_MAP[Config["model_type"]]
    model  = TorchModel(Config)
    x      = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [4, 5, 3, 4, 2]])  # (B=3, L=5)
    output = model(x)
    print(model.encoder.encoder.config.model_type)
    print(output.logits)
    print(output.logits.shape)  # (3, 3)
    # print(model.encoder.state_dict().keys())
