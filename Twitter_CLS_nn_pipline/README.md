# Twitter Climate-Change Stance Classification

基于多种预训练模型与循环网络的 Twitter 气候变化立场分类流水线，支持 BERT / RoBERTa / RNN / LSTM / GRU / Qwen（LoRA 微调），提供超参数搜索与模型横向对比功能。

## 任务说明

对 Twitter 推文进行三分类：

| 类别 | 原始标签 | 含义 |
|------|---------|------|
| negative | -1 | 气候变化怀疑者（skeptical） |
| neutral  |  0 | 中立 |
| positive | +1 | 支持气候行动（pro-climate） |

## 项目结构

```
Twitter_CLS_nn_pipline/
├── config.py           # 超参数配置与预训练模型路径映射
├── model.py            # 编码器注册表 + TorchModel（encoder → classify）
├── loader.py           # 数据加载，动态 padding，TwitterDataset
├── main.py             # 训练主程序（基于 Trainer，支持差异学习率）
├── lora_optimize.py    # LoRA 挂载 / 保存 / 加载（与具体模型解耦）
├── evalute.py          # 评估指标（accuracy、per-class F1、per-class accuracy）
├── run_experiment.py   # 超参数搜索（A）+ 模型横向对比（B）
├── predict.py          # 加载 LoRA adapter 对新文本做推理
└── data/
    ├── train_data.json
    └── valid_data.json
```

## 快速开始

### 1. 配置

编辑 `config.py`，设置 `model_type` 和 `use_lora`：

```python
"model_type": "roberta",   # bert | roberta | rnn | lstm | gru | qwen_lora
"use_lora":   False,        # True → 在骨干网络上挂载 LoRA
```

确保 `PRETRAIN_MODEL_MAP` 中对应模型的本地路径正确。

### 2. 训练

```python
# main.py 底部
Config["pretrain_model_path"] = PRETRAIN_MODEL_MAP[Config["model_type"]]
main(Config)
```

或直接运行：

```bash
python main.py
```

### 3. LoRA 训练（Qwen 等大模型）

```python
"model_type": "qwen_lora",
"use_lora":   True,
"lora_r":     8,
"lora_alpha": 16,
```

训练结束后 adapter 自动保存至 `output/adapter/`。

### 4. 超参数搜索 & 模型对比

```python
# run_experiment.py
hp_search("roberta", max_trials=20)          # 随机采样 20 组
model_compare(["roberta", "bert", "lstm"])   # 横向对比
```

### 5. 推理

```bash
python predict.py
```

## 实验结果（最优 HP）

| 模型 | Accuracy | F1-macro | 训练时长 |
|------|----------|----------|---------|
| RoBERTa | **83.4%** | **0.812** | 3.4 min |
| RoBERTa-LoRA | 79.4% | **0.782** | 1.4 min |
| BERT    | 80.6% | 0.780 | 3.3 min |
| GRU     | 63.8% | 0.614 | 0.25 min |
| LSTM    | 62.9% | 0.603 | 0.27 min |
| RNN     | 52.3% | 0.502 | 0.21 min |

## 依赖

```
torch
transformers
peft
scikit-learn
pandas
numpy
```
