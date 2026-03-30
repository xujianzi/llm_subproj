
from pathlib import Path

"""
全局配置参数
"""

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path("I:/pretrain_models")

Config = {
    # 模型类型，五选一：
    #   "bert"         — Sentence-BERT 微调（pair cosine loss）
    #   "bge_infer"    — BGE 直接推理，不训练，直接看准确率
    #   "bge_finetune" — BGE 微调（in-batch InfoNCE loss）
    #   "qwen_infer"   — Qwen3-Embedding 直接推理
    #   "qwen_finetune"— Qwen3-Embedding 微调（in-batch InfoNCE loss）
    "model_type"          : "qwen_finetune",

    # Sentence-BERT 预训练模型路径
    "model_path"          : (MODEL_DIR / "bert" / "bert-base-chinese").resolve(),
    # BGE 预训练模型路径（bge_infer / bge_finetune 时使用）
    "bge_model_path"      : (MODEL_DIR / "BAAI" / "bge-small-zh-v1.5").resolve(),
    # Qwen3-Embedding 模型路径（qwen_infer / qwen_finetune 时使用）
    "qwen_model_path"     : (MODEL_DIR / "Qwen" / "Qwen3-Embedding-0.6B").resolve(),

    # 训练好的模型保存目录
    "model_out_dir"       : (BASE_DIR / "output").resolve(),

    # 数据路径
    "train_data_path"     : (BASE_DIR / "data" / "train.json").resolve(),
    "valid_data_path"     : (BASE_DIR / "data" / "valid.json").resolve(),
    # schema.json：标准问题文本 → 整数索引 的映射字典
    "schema_path"         : (BASE_DIR / "data" / "schema.json").resolve(),

    # BERT tokenizer 编码的最大长度（超出截断，不足补 [PAD]）
    "max_length"          : 20,

    # 训练参数
    "epoch"               : 10,
    "batch_size"          : 32,
    "epoch_data_size"     : 200,   # 每个 epoch 随机采样的样本数量
    "learning_rate"       : 2e-5,
    "optimizer"           : "adamw",
    "positive_sample_rate": 0.45,  # 正样本（相同标准问）在训练对中的比例
}
