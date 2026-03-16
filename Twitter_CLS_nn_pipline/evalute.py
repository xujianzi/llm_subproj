import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import f1_score, confusion_matrix

"""
compute_metrics  — 总体 accuracy，供日常训练监控使用。
compute_metrics2 — 总体 accuracy + 每类 F1 + 每类 accuracy（召回率），
                   用于诊断各类别表现。

类别定义（来自 loader.py）：
    0 = negative（原标签 -1）
    1 = neutral （原标签  0）
    2 = positive（原标签  1）

Trainer 在每轮 eval 结束后自动调用，传入:
    eval_pred.predictions : np.ndarray (N, num_labels)  — 模型 logits
    eval_pred.label_ids   : np.ndarray (N,)             — 真实标签
返回 dict，key 会自动加上 "eval_" 前缀出现在日志中。
"""

LABEL_NAMES = ["negative", "neutral", "positive"]  # index 0/1/2


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    acc = float((preds == labels).mean())
    return {"accuracy": acc}


def compute_metrics2(eval_pred: EvalPrediction) -> dict:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)

    # 总体 accuracy
    overall_acc = float((preds == labels).mean())

    # 每类 F1（macro 平均也顺带返回）
    f1_per_class = f1_score(labels, preds, average=None, labels=[0, 1, 2])
    f1_macro     = f1_score(labels, preds, average="macro")

    # 每类 accuracy = 该类被正确预测数 / 该类样本总数（即召回率/sensitivity）
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])  # (3, 3)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    result = {
        "accuracy": overall_acc,
        "f1_macro": float(f1_macro),
    }
    for i, name in enumerate(LABEL_NAMES):
        result[f"f1_{name}"]  = float(f1_per_class[i])
        result[f"acc_{name}"] = float(per_class_acc[i])

    return result
