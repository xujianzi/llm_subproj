import os
import copy
import random
import numpy as np
import logging
from pathlib import Path

import torch
from torch.optim import AdamW as TorchAdamW
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

from config import Config, PRETRAIN_MODEL_MAP
from model import TorchModel
from evalute import compute_metrics, compute_metrics2
from loader import load_datasets
from lora_optimize import apply_lora, save_lora_adapter, load_lora_adapter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序（基于 transformers.Trainer）
"""

# scheduler 名称映射：config["scheduler"] → TrainingArguments.lr_scheduler_type
SCHEDULER_MAP = {
    "linear_warmup": "linear",
    "cosine_warmup": "cosine",
    None:            "constant",
}


class CustomTrainer(Trainer):
    """
    支持差异学习率（head_lr）的 Trainer 子类。
    当 config["head_lr"] 不为 None 时，分类头使用独立 LR，其余参数使用
    TrainingArguments.learning_rate。
    """

    _OPTIMIZER_MAP = {
        "adamw":             lambda pg: TorchAdamW(pg),
        "adamw_torch_fused": lambda pg: TorchAdamW(pg, fused=True),
    }

    def __init__(self, *args, head_lr: float | None = None, optimizer_type: str = "adamw", **kwargs):
        super().__init__(*args, **kwargs)
        self._head_lr       = head_lr
        self._optimizer_type = optimizer_type

    def create_optimizer(self):
        if self._head_lr is None or not hasattr(self.model, "classify"):
            return super().create_optimizer()

        lr = self.args.learning_rate
        wd = self.args.weight_decay
        classify_ids = {id(p) for p in self.model.classify.parameters()}
        param_groups = [
            {
                "params": [p for p in self.model.parameters()
                           if id(p) not in classify_ids and p.requires_grad],
                "lr": lr,
                "weight_decay": wd,
            },
            {
                "params": [p for p in self.model.classify.parameters() if p.requires_grad],
                "lr": self._head_lr,
                "weight_decay": 0.0,
            },
        ]
        build = self._OPTIMIZER_MAP.get(self._optimizer_type)
        if build is None:
            raise ValueError(f"Unknown optimizer: {self._optimizer_type!r}, available: {list(self._OPTIMIZER_MAP)}")
        self.optimizer = build(param_groups)
        return self.optimizer


def main(config):
    seed = config.get("seed", Config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.makedirs(config["model_path"], exist_ok=True)

    is_lora = config.get("use_lora", False)

    # 加载数据集（Dataset 对象）
    train_ds, valid_ds, tokenizer = load_datasets(config)
    # Qwen tokenizer 默认无 pad_token，用 eos_token 代替
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DataCollator 传给 Trainer 处理动态 padding
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # 加载模型；LoRA 在 TorchModel 建完后挂载到骨干网络
    model = TorchModel(config)
    if is_lora:
        backbone = model.encoder.encoder
        backbone.config.pad_token_id = tokenizer.pad_token_id
        model.encoder.encoder = apply_lora(backbone, config)
        # for name, p in model.named_parameters(): # 验证会学习的参数
        #     if p.requires_grad:
        #         print(name)

    # lr_scheduler_type
    lr_scheduler_type = SCHEDULER_MAP.get(config.get("scheduler"), "constant")

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(config["model_path"]),
        num_train_epochs=config["epoch"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        lr_scheduler_type=lr_scheduler_type,
        eval_strategy="epoch",                                        # 多久评估一次
        save_strategy="no" if is_lora else config["save_strategy"],   # 多久保存 checkpoint
        load_best_model_at_end=False if is_lora else config["save_strategy"] != "no",  # 指定 Trainer 用哪个评估指标来选择“最佳模型 checkpoint”
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=is_lora,
        save_total_limit=1, # 最多保存多少个 checkpoint
        seed=config["seed"],
        logging_steps=50,
    )

    # Trainer（差异 LR 由 CustomTrainer 处理）
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        compute_metrics=compute_metrics2,
        head_lr=config.get("head_lr"),
        optimizer_type=config.get("optimizer", "adamw"),
    )

    trainer.train()                                      # 内部会 outputs = model(**inputs)
    metrics = trainer.evaluate()
    # print(trainer.state.log_history)                   # Trainer会记录了所有 epoch 的指标
    acc = metrics.get("eval_accuracy", 0.0)
    logger.info(f"Final eval accuracy: {acc:.4f}")

    # LoRA：只保存 adapter 权重 + classify head
    if is_lora:
        adapter_path = Path(config["model_path"]) / "adapter"
        save_lora_adapter(model.encoder.encoder, model.classify, adapter_path)

    return metrics   # full dict: eval_accuracy, eval_f1_macro, eval_f1_*, eval_acc_*, etc.


if __name__ == "__main__":
    Config["pretrain_model_path"] = PRETRAIN_MODEL_MAP[Config["model_type"]]
    main(Config)

