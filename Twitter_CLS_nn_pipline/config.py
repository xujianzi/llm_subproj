from pathlib import Path

"""
配置参数
"""

BASE_DIR = Path(__file__).resolve().parent # 当前文件（config.py）的路径,转成绝对路径,到所在目录

MODEL_DIR = Path("E:/LLM/code/pretrain_models")

Config = {
    "model_path": (BASE_DIR / "output").resolve(),
    "train_data_path": (BASE_DIR / "data/train_data.json").resolve(),
    "valid_data_path": (BASE_DIR / "data/valid_data.json").resolve(),
    "vocab_path": (BASE_DIR / "chars.txt").resolve(),
    "model_type": "bert",
    "max_length": 120,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 64,
    "pooling_style": "max",
    "optimizer": "adamw",
    "learning_rate": 2e-5,          # BERT encoder LR
    "head_lr": 1e-3,                # classifier head LR (None = same as learning_rate)
    "weight_decay": 0.01,
    "scheduler": "linear_warmup",   # None | "linear_warmup" | "cosine_warmup"
    "warmup_ratio": 0.1,
    # class 0=negative(23.9%), 1=neutral(28.7%), 2=positive(47.4%) — inverse-freq weights
    "class_weights": [1.39, 1.16, 0.70],
    "dropout_rate": 0.1,
    "num_labels": 3,
    "pretrain_model_path": (MODEL_DIR/ "bert" / "bert-base-uncased").resolve(),
    "seed": 987,
}
