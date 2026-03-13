import torch
import os
import copy
import random
import numpy as np
import logging
from pathlib import Path

from config import Config
from model import TorchModel, choose_optimizer, build_scheduler
from evalute import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载训练数据
    train_loader, valid_loader = load_data(config)

    # 加载模型
    model = TorchModel(config)

    # 标识是否使用 gpu
    device = torch.device("cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    if device.type == "cuda":
        logger.info("gpu可以使用，迁移模型至gpu")
    model = model.to(device)

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载 LR scheduler（config["scheduler"]=None 时返回 None，训练循环无需改动）
    num_training_steps = config["epoch"] * len(train_loader)
    scheduler = build_scheduler(config, optimizer, num_training_steps)
    if scheduler:
        logger.info(f"scheduler: {config['scheduler']}, warmup_ratio={config.get('warmup_ratio', 0.1)}, total_steps={num_training_steps}")

    # 加载效果测试类
    evaluator = Evaluator(config, model, logger, valid_loader)

    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"epoch {epoch} begin")
        train_loss = []

        for idx, batch_data in enumerate(train_loader):
            input_ids      = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            labels         = batch_data["labels"].to(device)

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            train_loss.append(loss.item())
            if idx % max(1, int(len(train_loader) / 2)) == 0:
                logger.info(f"Epoch {epoch}, Batch {idx}, Loss {loss.item():.4f}")

        logger.info(f"epoch average loss: {np.mean(train_loss):.4f}")
        acc = evaluator.eval(epoch)

    model_path = Path(config["model_path"]) / f"{config['model_type']}_epoch_{epoch}.pth"
    # torch.save(model.state_dict(), model_path)
    logger.info(f"model saved -> {model_path}")
    return acc


if __name__ == "__main__":
    main(Config)

    # results = []
    # for model_type in ["bert", "lstm", "gru", "rnn"]:
    #     for lr in [1e-3, 1e-4, 1e-5]:
    #         for hidden_size in [128]:
    #             for batch_size in [64, 128]:
    #                 for pooling_style in ["avg", "max"]:

    #                     cfg = copy.deepcopy(Config)
    #                     cfg["model_type"]    = model_type
    #                     cfg["learning_rate"] = lr
    #                     cfg["hidden_size"]   = hidden_size
    #                     cfg["batch_size"]    = batch_size
    #                     cfg["pooling_style"] = pooling_style

    #                     acc = main(cfg)

    #                     result = {
    #                         "model":       model_type,
    #                         "lr":          lr,
    #                         "hidden_size": hidden_size,
    #                         "batch_size":  batch_size,
    #                         "pooling":     pooling_style,
    #                         "acc":         acc,
    #                     }
    #                     results.append(result)
    #                     print("-------完成实验：", result)

    # import pandas as pd
    # df = pd.DataFrame(results)
    # df_sorted = df.sort_values("acc", ascending=False)
    # experiment_results = Path(Config["model_path"]) / "experiment_results.csv"
    # df_sorted.to_csv(experiment_results, index=False)
