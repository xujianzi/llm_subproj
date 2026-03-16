"""
lora_optimize.py
----------------
LoRA 工具函数，与具体模型架构解耦。

对外接口：
    apply_lora(backbone, config)                        → 将 LoRA 挂载到任意骨干网络，返回 PeftModel
    save_lora_adapter(backbone, classify, adapter_path) → 只保存 adapter 权重 + classify head
    load_lora_adapter(backbone, classify, adapter_path) → 加载 adapter 权重 + classify head

target_modules 选择优先级：
    1. config["lora_target_modules"]（显式指定，优先级最高）
    2. _DEFAULT_TARGET_MODULES[backbone.config.model_type]（按架构自动匹配）
    3. 抛出 ValueError，提示用户显式指定
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

logger = logging.getLogger(__name__)

# HuggingFace model_type → 默认 LoRA target modules
# key 对应 backbone.config.model_type（如 "bert", "roberta", "qwen2"）
_DEFAULT_TARGET_MODULES: dict[str, list[str]] = {
    "bert":    ["query", "key", "value"],
    "roberta": ["query", "key", "value"],
    "qwen2":   ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qwen3":   ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def _get_target_modules(backbone: nn.Module, config: dict) -> list[str]:
    """
    确定 LoRA 注入的目标层名称。
    优先使用 config["lora_target_modules"]，其次按架构自动匹配。
    arch 模型架构名称
    """
    if "lora_target_modules" in config:
        return config["lora_target_modules"]
    arch = getattr(getattr(backbone, "config", None), "model_type", None) 
    if arch in _DEFAULT_TARGET_MODULES:
        return _DEFAULT_TARGET_MODULES[arch]
    raise ValueError(
        f"No default target_modules for architecture {arch!r}. "
        f"Please set config['lora_target_modules'] explicitly. "
        f"Supported architectures: {list(_DEFAULT_TARGET_MODULES)}"
    )


def apply_lora(backbone: nn.Module, config: dict) -> nn.Module:
    """
    将 LoRA 挂载到任意骨干网络（AutoModel 实例），返回 PeftModel。

    backbone : 原始的 HuggingFace AutoModel（BertModel、Qwen2Model 等均可）
    config   : 至少包含 lora_r、lora_alpha；可选 lora_dropout、lora_target_modules
    """
    target_modules = _get_target_modules(backbone, config)
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias="none",
    )
    wrapped = get_peft_model(backbone, lora_cfg)
    wrapped.print_trainable_parameters()
    return wrapped


def save_lora_adapter(backbone: nn.Module, classify: nn.Module, adapter_path: str | Path) -> None:
    """
    只保存 adapter 权重和 classify head 权重，base model 权重不保存。

    保存内容：
        adapter_path/adapter_config.json       — LoRA 配置
        adapter_path/adapter_model.safetensors — LoRA 增量权重
        adapter_path/classify_head.pt          — 分类头权重

    backbone : 经 apply_lora 包装后的 PeftModel
    classify : TorchModel.classify（nn.Sequential）
    """
    adapter_path = Path(adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)

    backbone.save_pretrained(str(adapter_path))
    torch.save(classify.state_dict(), adapter_path / "classify_head.pt")
    logger.info(f"Adapter + classify head saved → {adapter_path}")


def load_lora_adapter(backbone: nn.Module, classify: nn.Module, adapter_path: str | Path) -> nn.Module:
    """
    将 adapter 权重和 classify head 权重加载到已有模型上。

    backbone : 未经 LoRA 包装的原始 AutoModel
    classify : TorchModel.classify（nn.Sequential），in-place 加载权重
    返回值   : 加载了 adapter 的 PeftModel（赋值回 encoder_block.encoder）
    """
    adapter_path = Path(adapter_path)
    wrapped = PeftModel.from_pretrained(backbone, str(adapter_path))
    classify.load_state_dict(
        torch.load(adapter_path / "classify_head.pt", map_location="cpu")
    )
    logger.info(f"Adapter + classify head loaded ← {adapter_path}")
    return wrapped
