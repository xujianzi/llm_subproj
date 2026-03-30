
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
import torch.nn.functional as F
from transformers import AutoModel

"""
文本匹配模型集合，支持五种模式：
  bert          — SentenceBERT，mean pooling，pair cosine loss
  bge_infer     — BGE 直接推理，CLS pooling，不训练
  bge_finetune  — BGE 微调，CLS pooling，in-batch InfoNCE loss
  qwen_infer    — Qwen3-Embedding 直接推理，last-token pooling，不训练
  qwen_finetune — Qwen3-Embedding 微调，last-token pooling，in-batch InfoNCE loss

所有 encoder 统一接口：forward(input_ids, attention_mask) → (B, H) 归一化向量
pair 模型统一接口：forward(ids_a, mask_a, ids_b, mask_b) → (emb_a, emb_b)
evaluate.py 通过 model.encoder 调用，无需感知具体模型类型。
"""


# ─────────────────────────── BERT ───────────────────────────

class SentenceBertEncder(nn.Module):
    """BERT encoder：mean pooling 聚合所有非 PAD token，再 L2 归一化"""

    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, return_dict=True)

    def mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        # last_hidden_state: (B, L, H)  attention_mask: (B, L)
        mask = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
        summed = (last_hidden_state * mask).sum(dim=1)       # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)             # (B, 1)
        return summed / counts                               # (B, H)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(input_ids, attention_mask)
        sentence_embedding = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        return F.normalize(sentence_embedding, p=2, dim=1)  # (B, H)


class PairSentenceBert(nn.Module):
    """双塔 BERT：两个句子共享同一 SentenceBertEncder"""

    def __init__(self, model_name):
        super().__init__()
        self.encoder = SentenceBertEncder(model_name)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        emb_a = self.encoder(input_ids_a, attention_mask_a)
        emb_b = self.encoder(input_ids_b, attention_mask_b)
        return emb_a, emb_b


# ─────────────────────────── BGE ────────────────────────────

class SentenceBgeEncoder(nn.Module):
    """
    BGE encoder：取 CLS token（index 0）作为句子表示，再 L2 归一化。
    BGE 预训练时以 CLS token 做对比学习，所以用 CLS 比 mean pooling 更贴合原始训练方式。
    """

    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, return_dict=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(input_ids, attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]   # (B, H) 取第 0 个 token
        return F.normalize(cls_embedding, p=2, dim=1)        # (B, H)


class PairBge(nn.Module):
    """双塔 BGE：两个句子共享同一 SentenceBgeEncoder，供推理和微调共用"""

    def __init__(self, model_name):
        super().__init__()
        self.encoder = SentenceBgeEncoder(model_name)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        emb_a = self.encoder(input_ids_a, attention_mask_a)
        emb_b = self.encoder(input_ids_b, attention_mask_b)
        return emb_a, emb_b


# ─────────────────────────── Qwen ───────────────────────────

class SentenceQwenEncoder(nn.Module):
    """
    Qwen3-Embedding encoder：取最后一个真实 token 的隐状态作为句子表示。

    Qwen3 是 decoder-only 因果架构，最后一个 token 通过注意力机制聚合了
    全部前序 token 的信息，因此用它作为句子向量。

    注意：loader 中 tokenizer 需设置 padding_side="left"，
    使 batch 内每条序列的末尾位置都是真实 token 而非 PAD，
    避免 padding 污染句子表示。
    """

    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, return_dict=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(input_ids, attention_mask)
        # 左填充时，batch 中每条序列末尾位置（index -1）始终是真实 token。
        # 不能用 attention_mask.sum()-1 作为索引，那只对右填充正确：
        #   右填充 [tok, tok, PAD, PAD] → mask.sum()=2, idx=1 → tok2 ✓
        #   左填充 [PAD, PAD, tok, tok] → mask.sum()=2, idx=1 → PAD  ✗
        last_embedding = outputs.last_hidden_state[:, -1, :]    # (B, H) 后一个真实 token（decoder 最后位置聚合了全部上下文）
        return F.normalize(last_embedding, p=2, dim=1)


class PairQwen(nn.Module):
    """双塔 Qwen3-Embedding，接口与 PairSentenceBert / PairBge 完全一致"""

    def __init__(self, model_name):
        super().__init__()
        self.encoder = SentenceQwenEncoder(model_name)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        emb_a = self.encoder(input_ids_a, attention_mask_a)
        emb_b = self.encoder(input_ids_b, attention_mask_b)
        return emb_a, emb_b


# ──────────────────────────── Loss ──────────────────────────

def pair_cosine_loss(emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor):
    """
    用于 bert 模式的 pair loss（labels 取值 +1/-1）：
      正样本（y= 1）：loss = 1 - cos(a, b)
      负样本（y=-1）：loss = max(0, cos(a, b) - margin)
    """
    return F.cosine_embedding_loss(emb_a, emb_b, labels)


def infonce_loss(
    emb_a: torch.Tensor,
    emb_p: torch.Tensor,
    class_labels: torch.Tensor = None,
    temperature: float = 0.05,
):
    """
    用于 bge_finetune 模式的 in-batch InfoNCE loss。
    emb_a:        (B, H) anchor 向量（已归一化）
    emb_p:        (B, H) positive 向量（已归一化）
    class_labels: (B,)  每个样本所属的标准问索引，用于屏蔽假负样本
    temperature:  越小对难负例惩罚越重，BGE 官方默认 0.05

    同类掩码逻辑：
      对角线 (i==j) 是真正样本，保留。
      同类非对角线 (class_labels[i]==class_labels[j], i!=j) 是假负样本，
      将其相似度设为 -inf，使其在 softmax 中权重归零，不参与梯度。
    """
    sim_matrix = torch.matmul(emb_a, emb_p.T) / temperature  # (B, B)

    if class_labels is not None:
        B = sim_matrix.size(0)
        # 同类矩阵：(B, B) bool，True 表示 i 与 j 属于同一标准问
        same_class = class_labels.unsqueeze(1) == class_labels.unsqueeze(0)
        # 对角线是正样本，不能屏蔽；只屏蔽同类的非对角位置
        diag_mask = torch.eye(B, dtype=torch.bool, device=sim_matrix.device)
        false_negative_mask = same_class & ~diag_mask
        sim_matrix = sim_matrix.masked_fill(false_negative_mask, float("-inf"))

    targets = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    return F.cross_entropy(sim_matrix, targets)


# ───────────────────────── 工厂函数 ─────────────────────────

def choose_model(config):
    """根据 config['model_type'] 返回对应模型实例"""
    model_type = config["model_type"]
    if model_type == "bert":
        return PairSentenceBert(config["model_path"])
    elif model_type in ("bge_infer", "bge_finetune"):
        return PairBge(config["bge_model_path"])
    elif model_type in ("qwen_infer", "qwen_finetune"):
        return PairQwen(config["qwen_model_path"])
    else:
        raise ValueError(
            f"未知 model_type: {model_type}，"
            "可选: bert / bge_infer / bge_finetune / qwen_infer / qwen_finetune"
        )


def choose_optimizer(config, model):
    """根据 config['optimizer'] 返回对应优化器"""
    optimizer = config["optimizer"]
    lr = config["learning_rate"]
    if optimizer in ("adam", "adamw"):
        return AdamW(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"optimizer 须为 adam/adamw/sgd，得到: {optimizer}")


if __name__ == "__main__":
    from config import Config
    import copy

    B, L = 2, 16

    for mode in ["bert", "bge_finetune"]:
        cfg = copy.deepcopy(Config)
        cfg["model_type"] = mode
        model = choose_model(cfg)
        ids_a   = torch.randint(0, 100, (B, L))
        mask_a  = torch.ones(B, L, dtype=torch.long)
        ids_b   = torch.randint(0, 100, (B, L))
        mask_b  = torch.ones(B, L, dtype=torch.long)
        emb_a, emb_b = model(ids_a, mask_a, ids_b, mask_b)
        if mode == "bert":
            labels = torch.tensor([1, -1], dtype=torch.float32)
            loss = pair_cosine_loss(emb_a, emb_b, labels)
        else:
            loss = infonce_loss(emb_a, emb_b)
        print(f"[{mode}] loss: {loss.item():.4f}")
