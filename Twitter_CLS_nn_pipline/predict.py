"""
predict.py
----------
加载 base model + LoRA adapter，对输入文本做情感分类推理。

用法：
    python predict.py
    或直接修改底部 ADAPTER_PATH 和 samples 后运行。
"""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import PRETRAIN_MODEL_MAP

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
NUM_LABELS = 3

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_MODEL_PATH = str(PRETRAIN_MODEL_MAP["qwen_lora"])
ADAPTER_PATH    = "output/best_qwen_lora/adapter"  # 训练完成后 adapter 的保存路径


def load_model(base_model_path: str, adapter_path: str):
    """
    加载 base model 并挂载 LoRA adapter。

    Returns:
        (model, tokenizer)  model 处于 eval 模式，未移至 GPU
    """
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=NUM_LABELS,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


def predict(texts: list[str], model, tokenizer, device: str = "cuda", max_length: int = 120) -> list[str]:
    """
    对一批文本做推理，返回标签字符串列表。

    Args:
        texts:      输入文本列表
        model:      已加载的 PeftModel
        tokenizer:  对应的 tokenizer
        device:     运行设备
        max_length: 截断长度

    Returns:
        ["positive", "neutral", ...] 与 texts 等长
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    preds = logits.argmax(dim=-1).tolist()
    return [LABEL_MAP[p] for p in preds]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading base model from: {BASE_MODEL_PATH}")
    print(f"Loading adapter from:    {ADAPTER_PATH}")

    model, tokenizer = load_model(BASE_MODEL_PATH, ADAPTER_PATH)
    model = model.to(device)

    samples = [
        "I absolutely love this! It's amazing.",
        "It's okay, nothing special.",
        "Terrible experience, never buying again.",
        "Just received my order, pretty decent quality.",
    ]

    results = predict(samples, model, tokenizer, device=device)
    print("\n── Results ──────────────────────────────")
    for text, label in zip(samples, results):
        print(f"  [{label:<8}] {text}")
