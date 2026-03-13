
import torch
import os
import random 
import numpy as np
import logging 
from config import Config 
from model import TorchModel, choose_optimizer
from evalute import Evaluator
from loader import load_data
from pathlib import Path

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda() # model 和数据应该同时在GPU或CPU上
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"epoch {epoch} begin")
        train_loss = []
        for idx, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
        
            optimizer.zero_grad()
            input_ids, attenion_mask, labels = batch_data  # # input_ids: (B, L), labels: (B,)
            loss = model(input_ids, attenion_mask, labels) # 这里实际上就是调用 model.forward(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if idx % int(len(train_data) / 2) == 0: # 当 idx是0，或者idx刚好走到epoch中点时会触发一次日志,观察模型状态【有没有学习】
                logger.info(f"Epoch {epoch}, Batch {idx}, Loss {loss.item():.4f}")
        logger.info(f"epoch average loss: {np.mean(train_loss):.4f}")
        acc = evaluator.eval(epoch)
    
    # model_path = Path(config["model_path"]) / f"{config["model_type"]}_epoch_{epoch}.pth"
    # torch.save(model.state_dict(), model_path)
    return acc

if __name__ == "__main__":
    # main(Config)
    import copy
    
    results = []
    for model_type in ["gated_cnn", 'bert', 'lstm']:
        for lr in [1e-3, 1e-4, 1e-5]:
            for hidden_size in [128]:
                for batch_size in [64, 128]:
                    for pooling_style in ["avg", 'max']:

                        cfg = copy.deepcopy(Config)   # 每次实验用一份新的 config 副本,防止Config在多层循环里被反复污染
                        cfg["model_type"] = model_type
                        cfg["learning_rate"] = lr
                        cfg["hidden_size"] = hidden_size
                        cfg["batch_size"] = batch_size
                        cfg["pooling_style"] = pooling_style

                        acc = main(cfg)

                        result = {
                        "model": model_type,
                        "lr": lr,
                        "hidden_size": hidden_size,
                        "batch_size": batch_size,
                        "pooling": pooling_style,
                        "acc": acc,
                        }

                        results.append(result)
                        print("-------完成实验：", result)
    
    import pandas as pd
    df = pd.DataFrame(results)
    df_sorted = df.sort_values("acc", ascending=False)
    experiment_results = Path(Config["model_path"]) / f"experiment_results.csv"
    df_sorted.to_csv(experiment_results, index=False)

