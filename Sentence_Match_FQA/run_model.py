import torch
import torch.nn.functional as F
from pathlib import Path

from config import Config
from model import choose_model
from loader import DataGenerator, load_schema

"""
加载训练好的模型，对输入文本进行标准问匹配。

流程：
  1. 加载模型结构 + 训练权重
  2. 用 DataGenerator 构建知识库向量矩阵（复用 tokenizer 和 knwdb）
  3. 对输入文本编码后做余弦相似度最近邻检索
  4. 返回最匹配的标准问文本和相似度分数
"""

# ── 测试输入（写死） ──────────────────────────────────────
TEST_QUERIES = [
    "我想查一下话费",
    "怎么办理宽带业务",
    "流量还剩多少",
    "帮我看看账单",
]


def build_knwdb_vectors(train_dg: DataGenerator, model):
    """
    将 DataGenerator.knwdb 中所有问题编码成归一化向量矩阵。
    返回：
      knwdb_vectors        (N, H) 归一化向量矩阵
      pos_to_std_idx       {位置索引 → 标准问整数 idx}
    """
    question_ids, question_masks = [], []
    pos_to_std_idx = {}

    for std_idx, enc_list in train_dg.knwdb.items():
        for (input_ids, attn_mask) in enc_list:
            pos_to_std_idx[len(question_ids)] = std_idx
            question_ids.append(input_ids)
            question_masks.append(attn_mask)

    with torch.no_grad():
        ids_matrix  = torch.stack(question_ids,  dim=0)  # (N, L)
        mask_matrix = torch.stack(question_masks, dim=0)  # (N, L)
        if torch.cuda.is_available():
            ids_matrix  = ids_matrix.cuda()
            mask_matrix = mask_matrix.cuda()
        vectors = model.encoder(ids_matrix, mask_matrix)  # (N, H)
        vectors = F.normalize(vectors, dim=-1)

    return vectors, pos_to_std_idx


def encode_query(query: str, train_dg: DataGenerator):
    """用 DataGenerator 内置的 tokenizer 对输入文本编码"""
    input_ids, attention_mask = train_dg.encode_sentence(query)
    return input_ids.unsqueeze(0), attention_mask.unsqueeze(0)  # (1, L)


def find_best_match(query_vec, knwdb_vectors, pos_to_std_idx, idx_to_std_text):
    """余弦相似度最近邻检索，返回 (标准问文本, 相似度分数)"""
    query_vec = F.normalize(query_vec.squeeze(0), dim=-1)           # (H,)
    similarities = torch.einsum("d,nd->n", query_vec, knwdb_vectors) # (N,)
    hit_pos   = torch.argmax(similarities).item()
    hit_score = similarities[hit_pos].item()
    std_idx   = pos_to_std_idx[hit_pos]
    std_text  = idx_to_std_text[std_idx]
    return std_text, hit_score


def main():
    config     = Config
    model_type = config["model_type"]

    # ── 1. 加载模型 ───────────────────────────────────────
    model = choose_model(config)

    # 推理模式（bge_infer / qwen_infer）直接使用预训练权重，无需加载 .pt
    if model_type not in ("bge_infer", "qwen_infer"):
        model_path = Path(config["model_out_dir"]) / f"{model_type}_best.pt"
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"已加载权重：{model_path}")

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # ── 2. 构建知识库向量 ─────────────────────────────────
    # DataGenerator 负责 tokenizer 初始化（含 padding_side / pad_token 等设置）
    print("正在构建知识库向量...")
    train_dg = DataGenerator(config["train_data_path"], config)
    knwdb_vectors, pos_to_std_idx = build_knwdb_vectors(train_dg, model)

    # 反转 schema：整数 idx → 标准问文本
    schema          = load_schema(config["schema_path"])
    idx_to_std_text = {v: k for k, v in schema.items()}

    # ── 3. 对每条输入做检索 ───────────────────────────────
    print(f"\n{'─' * 55}")
    for query in TEST_QUERIES:
        input_ids, attention_mask = encode_query(query, train_dg)
        if torch.cuda.is_available():
            input_ids      = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        with torch.no_grad():
            query_vec = model.encoder(input_ids, attention_mask)  # (1, H)

        std_text, score = find_best_match(
            query_vec, knwdb_vectors, pos_to_std_idx, idx_to_std_text
        )
        print(f"输入：{query}")
        print(f"匹配：{std_text}  (相似度: {score:.4f})")
        print(f"{'─' * 55}")


if __name__ == "__main__":
    main()
