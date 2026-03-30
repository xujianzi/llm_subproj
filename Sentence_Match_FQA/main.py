import torch
import os
import numpy as np
import logging
from pathlib import Path
import copy

from config import Config
from model import choose_model, pair_cosine_loss, infonce_loss, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

"""
模型训练/推理主程序，支持三种模式（通过 config['model_type'] 切换）：

  bert        — 加载 BERT，训练 pair cosine loss，每 epoch 评估
  bge_infer   — 加载 BGE，不训练，直接评估准确率
  bge_finetune— 加载 BGE，训练 in-batch InfoNCE loss，每 epoch 评估
"""


def main(config):
    os.makedirs(config["model_out_dir"], exist_ok=True)
    model_type = config["model_type"]

    # 根据 model_type 自动选择模型（PairSentenceBert 或 PairBge）
    model = choose_model(config)

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu 可用，迁移模型至 gpu")
        model = model.cuda()

    evaluator = Evaluator(config, model, logger)

    # ── 直接推理模式：不训练，直接跑评估 ─────────────────────────
    if model_type in ("bge_infer", "qwen_infer"):
        logger.info("bge_infer 模式：跳过训练，直接评估")
        acc, elapsed_time = evaluator.eval(epoch=0)
        return [{"epoch": 0, "acc": acc, "elapsed_time": elapsed_time}]

    # ── 训练模式（bert / bge_finetune）────────────────────────────
    train_data = load_data(config["train_data_path"], config)
    optimizer = choose_optimizer(config, model)
    eval_metrics = []

    for epoch in range(1, config["epoch"] + 1):
        model.train()
        logger.info(f"epoch {epoch} begin")
        train_loss = []

        for idx, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()

            if model_type in ("bge_finetune", "qwen_finetune"):
                # InfoNCE：batch 含正对 + 类别标签
                # class_labels 用于屏蔽同标准问的假负样本
                ids_a, mask_a, ids_p, mask_p, class_labels = batch_data
                emb_a, emb_p = model(ids_a, mask_a, ids_p, mask_p)
                loss = infonce_loss(emb_a, emb_p, class_labels)
            else:
                # bert：有标签的正/负对
                ids_a, mask_a, ids_b, mask_b, labels = batch_data
                emb_a, emb_b = model(ids_a, mask_a, ids_b, mask_b)
                loss = pair_cosine_loss(emb_a, emb_b, labels)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            if idx % max(1, len(train_data) // 2) == 0:
                logger.info(f"Epoch {epoch}, Batch {idx}, Loss {loss.item():.4f}")

        logger.info(f"epoch average loss: {np.mean(train_loss):.4f}")
        acc, elapsed_time = evaluator.eval(epoch)
        eval_metrics.append({"epoch": epoch, "acc": acc, "elapsed_time": elapsed_time})
        best = max(eval_metrics, key=lambda x: x["acc"])
        logger.info(f"best metrics so far: {best}")

        # 当前 epoch 达到历史最优时保存模型
        if acc == best["acc"]:
            model_path = Path(config["model_out_dir"]) / f"{model_type}_best.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f"best model saved → {model_path}")


    return eval_metrics


if __name__ == "__main__":
    results = []

    # 切换 model_type 即可对比三种方案：
    #   "bert" / "bge_infer" / "bge_finetune" / "qwen_finetune" / "qwen_infer"
    for model_type in [ "qwen_finetune"]:
        for lr in [2e-5]:
            for batch_size in [32]:
                cfg = copy.deepcopy(Config)
                cfg["model_type"] = model_type
                cfg["learning_rate"] = lr
                cfg["batch_size"] = batch_size

                logger.info(
                    f"Start | model={model_type}, lr={lr}, batch_size={batch_size}"
                )
                eval_metrics = main(cfg)
                best = max(eval_metrics, key=lambda x: x["acc"])
                results.append({
                    "model": model_type,
                    "lr": lr,
                    "batch_size": batch_size,
                    "acc": best["acc"],
                    "elapsed_time": best["elapsed_time"],
                    "epoch": best["epoch"],
                })

    def save_to_csv(results):
        import pandas as pd
        df = pd.DataFrame(results).sort_values("acc", ascending=False)
        out_path = Path(Config["model_out_dir"]) / "experiment_results.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"实验结果已保存至 {out_path}")

    save_to_csv(results)
