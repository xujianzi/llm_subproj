from pathlib import Path

"""
配置参数
"""

BASE_DIR = Path(__file__).resolve().parent # 当前文件（config.py）的路径,转成绝对路径,到所在目录

MODEL_DIR = Path("I:/pretrain_models")

# model_type → 预训练模型路径映射
# RNN/LSTM/GRU 不使用预训练权重，但仍需路径来加载 tokenizer 和 vocab_size
PRETRAIN_MODEL_MAP: dict[str, Path] = {
    "bert":      (MODEL_DIR / "bert" / "bert-base-uncased").resolve(),
    "roberta":   (MODEL_DIR / "bert" / "roberta-base").resolve(),
    "rnn":       (MODEL_DIR / "bert" / "bert-base-uncased").resolve(),
    "lstm":      (MODEL_DIR / "bert" / "bert-base-uncased").resolve(),
    "gru":       (MODEL_DIR / "bert" / "bert-base-uncased").resolve(),
    "qwen_lora": (MODEL_DIR / "Qwen" / "Qwen3-0.6B").resolve(),
}

Config = {
    "model_path": (BASE_DIR / "output").resolve(),
    "train_data_path": (BASE_DIR / "data/train_data.json").resolve(),
    "valid_data_path": (BASE_DIR / "data/valid_data.json").resolve(),
    "vocab_path": (BASE_DIR / "chars.txt").resolve(),
    "model_type": "roberta",       # 模型架构：bert | roberta | rnn | lstm | gru | qwen_lora
    "use_lora":   False,            # 是否挂载 LoRA（需骨干网络为 HuggingFace AutoModel）
    "max_length": 120,
    "hidden_size": 256,            # Only for rnn/lstm/gru
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 6,
    "batch_size": 64,
    "pooling_style": "max",
    "optimizer": "adamw",           # adamw | adamw_torch_fused (在3090上更快)
    "learning_rate": 2e-5,          # BERT encoder LR
    "head_lr": 1e-3,                # classifier head LR (None = same as learning_rate)
    "weight_decay": 0.01,
    "scheduler": "linear_warmup",   # None | "linear_warmup" | "cosine_warmup"
    "warmup_ratio": 0.1,            # 训练开始时的学习率 linearly warmup 到 learning_rate
    # class 0=negative(23.9%), 1=neutral(28.7%), 2=positive(47.4%) — inverse-freq weights
    "class_weights": [1.39, 1.16, 0.70],
    "dropout_rate": 0.1,
    "num_labels": 3,
    # pretrain_model_path 由 PRETRAIN_MODEL_MAP[model_type] 动态注入，无需在此填写
    "seed": 987,
    "save_strategy": "no",          # "no" | "epoch" | "step"
    
    # LoRA 超参（仅 qwen_lora 使用）
    "lora_r":       8,
    "lora_alpha":   16,
    "lora_dropout": 0.05,
}
