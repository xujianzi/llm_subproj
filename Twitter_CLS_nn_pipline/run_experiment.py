"""
run_experiment.py
-----------------
两个解耦模块：
  A) hp_search(model_type, max_trials)  — 对单个模型做超参数网格/随机搜索
  B) model_compare(model_types, use_best_configs)  — 对多个模型做横向对比

在 __main__ 中直接调用所需函数即可，无需命令行参数。

输出文件（均在 output/experiments/ 下）：
  hp_search_<model>_<timestamp>.csv   — A 的全量搜索结果（按 acc 降序）
  best_config_<model>.json            — A 保存的最优超参数配置（供 B 复用）
  model_compare_<timestamp>.csv       — B 的模型对比结果
"""

import copy
import json
import random
import logging
import itertools
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config, PRETRAIN_MODEL_MAP
from main import main                    # main(config) -> metrics dict
from lora_optimize import train_lora     # train_lora(config) -> metrics dict

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(Config["model_path"]) / "experiments"

# ── Search Spaces ──────────────────────────────────────────────────────────────
# BERT/RoBERTa：预训练模型微调，学习率小、对 warmup 敏感
_BERT_SPACE: dict[str, list] = {
    "learning_rate": [1e-5, 2e-5, 3e-5],
    "head_lr":       [1e-4, 5e-4, 1e-3],
    "batch_size":    [32, 64],
    "warmup_ratio":  [0.06, 0.1],
    "weight_decay":  [0.0, 0.01],
    "dropout_rate":  [0.1, 0.2],
    "scheduler":     ["linear_warmup", "cosine_warmup"],
}

# RNN/LSTM/GRU：从头训练，学习率大、hidden_size 和 pooling 影响明显
_RNN_SPACE: dict[str, list] = {
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "hidden_size":   [128, 256, 512],
    "num_layers":    [1, 2, 3],
    "batch_size":    [64, 128],
    "pooling_style": ["max", "avg"],
    "dropout_rate":  [0.1, 0.3],
}

_LORA_SPACE: dict[str, list] = {
    "lora_r":        [4, 8, 16, 32],
    "lora_alpha":    [8, 16, 32, 64],
    "lora_dropout":  [0.05, 0.1],
    "learning_rate": [1e-4, 5e-5, 2e-5],
    "batch_size":    [8, 16],
}

SEARCH_SPACES: dict[str, dict[str, list]] = {
    "bert":      _BERT_SPACE,
    "roberta":   _BERT_SPACE,
    "rnn":       _RNN_SPACE,
    "lstm":      _RNN_SPACE,
    "gru":       _RNN_SPACE,
    "qwen_lora": _LORA_SPACE,
}

# 当 hp_search(max_trials=None) 时各模型的默认上限
# BERT/RoBERTa/qwen_lora 微调成本高，默认随机采样；RNN 类全量网格搜索（None = 无限制）
DEFAULT_MAX_TRIALS: dict[str, int | None] = {
    "bert":      20,
    "roberta":   20,
    "rnn":       None,
    "lstm":      None,
    "gru":       None,
    "qwen_lora": 10,
}


# ── Internal Helpers ───────────────────────────────────────────────────────────

def _make_base_config(model_type: str) -> dict:
    """deepcopy Config，并根据 model_type 自动设置 pretrain_model_path。"""
    if model_type not in PRETRAIN_MODEL_MAP:
        raise ValueError(
            f"No pretrain_model_path mapping for model_type={model_type!r}. "
            f"Please add it to PRETRAIN_MODEL_MAP in config.py. "
            f"Available: {list(PRETRAIN_MODEL_MAP)}"
        )
    cfg = copy.deepcopy(Config)
    cfg["model_type"]         = model_type
    cfg["pretrain_model_path"] = PRETRAIN_MODEL_MAP[model_type]
    return cfg


def _grid_configs(model_type: str, space: dict, max_trials: int | None) -> list[dict]:
    """笛卡尔积展开搜索空间，可选随机采样 max_trials 组。"""
    keys   = list(space.keys())
    combos = list(itertools.product(*[space[k] for k in keys]))
    total  = len(combos)

    if max_trials is not None and max_trials < total:
        combos = random.sample(combos, max_trials)
        logger.info(f"Random sampling {max_trials} / {total} combinations")

    configs = []
    for values in combos:
        cfg = _make_base_config(model_type)
        for k, v in zip(keys, values):
            cfg[k] = v
        configs.append(cfg)
    return configs


def _run_one(cfg: dict, trial_idx: int, total: int, tracked_keys: list[str]) -> dict:
    """运行单次实验，捕获异常，返回结果行 dict。"""
    label = f"[{trial_idx + 1}/{total}] model={cfg['model_type']}"
    hp_summary = " | ".join(f"{k}={cfg.get(k)}" for k in tracked_keys[:4])  # 打印前4个HP
    logger.info(f"{label} → {hp_summary}")

    t0 = time.time()
    try:
        result = train_lora(cfg) if cfg["model_type"] == "qwen_lora" else main(cfg)
        if isinstance(result, dict):
            acc      = result.get("eval_accuracy", None)
            f1_macro = result.get("eval_f1_macro", None)
        else:
            acc      = float(result)
            f1_macro = None
    except Exception as e:
        logger.error(f"{label} FAILED: {e}", exc_info=True)
        acc, f1_macro = None, None

    elapsed = (time.time() - t0) / 60
    row: dict = {
        "model_type":   cfg["model_type"],
        "acc":          acc,
        "f1_macro":     f1_macro,
        "duration_min": round(elapsed, 2),
    }
    for k in tracked_keys:
        row[k] = cfg.get(k)
    return row


def _save_csv(rows: list[dict], path: Path, sort_by: str = "acc") -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False, na_position="last")
    df.to_csv(path, index=False)
    logger.info(f"Saved → {path}")
    return df


def _save_best_config(cfg: dict, model_type: str) -> Path:
    path = RESULTS_DIR / f"best_config_{model_type}.json"
    import numpy as np
    def _to_native(v):
        if isinstance(v, Path):        return str(v)
        if isinstance(v, np.integer):  return int(v)
        if isinstance(v, np.floating): return float(v)
        return v
    serializable = {k: _to_native(v) for k, v in cfg.items()}
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))
    logger.info(f"Best config saved → {path}")
    return path


def _load_best_config(model_type: str) -> dict | None:
    path = RESULTS_DIR / f"best_config_{model_type}.json"
    if not path.exists():
        logger.warning(f"No saved best config for '{model_type}' at {path}")
        return None
    raw = json.loads(path.read_text())
    # 将路径字段还原为 Path 对象
    for k in ("model_path", "train_data_path", "valid_data_path", "vocab_path", "pretrain_model_path"):
        if k in raw:
            raw[k] = Path(raw[k])
    return raw


# ── Module A: Hyperparameter Search ───────────────────────────────────────────

def hp_search(model_type: str, max_trials: int | None = None) -> pd.DataFrame:
    """
    对 model_type 做超参数搜索（网格搜索或随机采样）。

    Args:
        model_type:  要搜索的模型（"bert"/"roberta"/"rnn"/"lstm"/"gru"）
        max_trials:  最多跑几组，None 表示全量网格搜索

    Saves:
        hp_search_<model>_<timestamp>.csv   全量结果（按 acc 降序）
        best_config_<model>.json            最优 HP 配置，供 model_compare 复用

    Returns:
        pd.DataFrame  排序后的结果表
    """
    if model_type not in SEARCH_SPACES:
        raise ValueError(
            f"No search space for model_type={model_type!r}. "
            f"Available: {list(SEARCH_SPACES)}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    space      = SEARCH_SPACES[model_type]
    if max_trials is None:
        max_trials = DEFAULT_MAX_TRIALS.get(model_type)
    configs = _grid_configs(model_type, space, max_trials)
    total   = len(configs)
    tracked = list(space.keys())

    logger.info(f"{'='*60}")
    logger.info(f"HP Search: model={model_type}, trials={total}")
    logger.info(f"{'='*60}")

    rows = [_run_one(cfg, i, total, tracked) for i, cfg in enumerate(configs)]

    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    path = RESULTS_DIR / f"hp_search_{model_type}_{ts}.csv"
    df   = _save_csv(rows, path)

    # 找最优行，保存对应的完整 Config
    valid_df = df.dropna(subset=["acc"])
    if not valid_df.empty:
        best_row = valid_df.iloc[0]
        best_cfg = _make_base_config(model_type)
        for k in tracked:
            best_cfg[k] = best_row[k]
        _save_best_config(best_cfg, model_type)
        logger.info(f"Best: acc={best_row['acc']:.4f}, f1_macro={best_row.get('f1_macro')}")
    else:
        logger.warning("All trials failed, no best config saved.")

    return df


# ── Module B: Model Comparison ─────────────────────────────────────────────────

def model_compare(model_types: list[str], use_best_configs: bool = False) -> pd.DataFrame:
    """
    对多个模型各跑一次，横向对比性能。

    Args:
        model_types:      要对比的模型列表
        use_best_configs: True → 从 best_config_<model>.json 加载HP；
                          False → 使用默认 Config（只覆盖 model_type）

    Saves:
        model_compare_<timestamp>.csv

    Returns:
        pd.DataFrame  排序后的对比结果
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total = len(model_types)

    logger.info(f"{'='*60}")
    logger.info(f"Model Compare: {model_types}, use_best_configs={use_best_configs}")
    logger.info(f"{'='*60}")

    rows = []
    for i, model_type in enumerate(model_types):
        if use_best_configs:
            cfg = _load_best_config(model_type)
            if cfg is None:
                logger.warning(f"Falling back to default Config for {model_type}")
                cfg = _make_base_config(model_type)
        else:
            cfg = _make_base_config(model_type)

        # model_compare 不需要记录 HP 字段（或记录当前 cfg 中相关值）
        tracked = list(SEARCH_SPACES.get(model_type, {}).keys())
        row = _run_one(cfg, i, total, tracked)
        rows.append(row)

    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    path = RESULTS_DIR / f"model_compare_{ts}.csv"
    df   = _save_csv(rows, path)

    # 打印对比摘要
    summary_cols = [c for c in ["model_type", "acc", "f1_macro", "duration_min"] if c in df.columns]
    logger.info("\n" + df[summary_cols].to_string(index=False))

    return df


def main_test(models: list[str], is_compare: bool = False, max_trials: int | None = None) -> None:
    """
    对 models 中的每个模型依次做超参数搜索，
    若 is_compare=True，搜索结束后用最优 HP 做模型横向对比。

    Args:
        models:      模型名称列表，如 ["roberta", "lstm", "gru"]
        is_compare:  是否在 HP 搜索后做模型对比
        max_trials:  每个模型随机采样的组合数（None = 全量网格）
    """
    for model_type in models:
        hp_search(model_type, max_trials=max_trials)

    if is_compare:
        model_compare(models, use_best_configs=True)


if __name__ == "__main__":
    # [A] 超参数搜索：对单个模型搜索最优 HP
    # max_trials=None 表示全量网格搜索；设置整数则随机采样（推荐 BERT 类用随机采样）
    # hp_search("roberta", max_trials=20)
    # hp_search("lstm")

    # [B] 模型对比：对多个模型各跑一次，横向比较
    # use_best_configs=False → 用默认 Config；True → 读取 A 保存的最优 HP
    # model_compare(["roberta", "lstm", "gru", "rnn"], use_best_configs=False)
    # model_compare(["roberta", "lstm", "gru", "rnn"], use_best_configs=True)

    # # [A→B] 完整 pipeline：先搜 HP，再用最优 HP 做模型对比
    # hp_search("roberta", max_trials=20)
    # model_compare(["roberta", "lstm", "gru", "rnn"], use_best_configs=True)

    # ── 修改参数后直接运行即可 ────────────────────────────────────────────────
    main_test(
        models     = ["lstm", "gru", "rnn","bert","roberta"],
        is_compare = True,
        max_trials = 2,       # None = 全量网格搜索
    )
