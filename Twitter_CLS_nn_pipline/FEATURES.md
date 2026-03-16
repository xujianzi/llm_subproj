# 项目特性说明

## 核心特性

### 1. 编码器注册表模式（Encoder Registry）

`model.py` 使用注册表统一管理所有编码器类型，新增模型只需在 `ENCODER_REGISTRY` 中添加一行，无需修改任何现有逻辑：

```python
ENCODER_REGISTRY = {
    "bert":    BertEncoderBlock,
    "roberta": BertEncoderBlock,
    "rnn":     _recurrent("rnn"),
    "lstm":    _recurrent("lstm"),
    "gru":     _recurrent("gru"),
    "qwen_lora": QwenEncoderBlock,
}
```

`TorchModel.forward` 中无任何 `if/else` 条件分支，结构统一干净。

---

### 2. LoRA 与模型架构完全解耦

`lora_optimize.py` 不依赖任何具体模型类。`apply_lora` 接受任意 HuggingFace `AutoModel` 骨干网络，通过读取 `backbone.config.model_type` 自动匹配默认的注入层：

| 架构 | 默认 target_modules |
|------|-------------------|
| bert / roberta | query, key, value |
| qwen2 / qwen3 | q_proj, k_proj, v_proj, o_proj |

也可通过 `config["lora_target_modules"]` 显式覆盖，适配任意新架构。

`use_lora` 配置项独立于 `model_type`，两者正交：

```python
"model_type": "bert",    # 模型架构
"use_lora":   True,      # 是否挂载 LoRA（可选）
```

---

### 3. 差异学习率（Differential Learning Rate）

`CustomTrainer` 为分类头（`classify`）和编码器骨干网络分别设置不同的学习率，避免预训练权重被分类头的大梯度破坏：

```python
"learning_rate": 2e-5,   # 编码器 LR（微小更新）
"head_lr":       1e-3,   # 分类头 LR（较大更新）
```

---

### 4. 动态 Padding（Dynamic Padding）

数据加载时不对单条样本做 padding，交由 `DataCollatorWithPadding` 在每个 batch 内按最长序列动态补齐。相比固定 `max_length` padding，减少无效计算，加快训练速度。

---

### 5. 类别权重（Class Weights）

数据集存在类别不平衡（negative 23.9%、neutral 28.7%、positive 47.4%），通过反频率权重缓解：

```python
"class_weights": [1.39, 1.16, 0.70]
```

权重通过 `register_buffer` 注册到模型，随模型一起移动到 GPU，传入 `CrossEntropyLoss`。

---

### 6. 超参数搜索框架（`run_experiment.py`）

提供两个解耦模块：

- **`hp_search`**：对单个模型做网格搜索或随机采样，结果按 accuracy 降序保存为 CSV，最优配置保存为 JSON。
- **`model_compare`**：加载各模型最优配置，横向对比多模型性能，输出对比表。

三类搜索空间针对不同架构特性独立设计：
- BERT/RoBERTa：关注学习率、warmup、weight decay
- RNN/LSTM/GRU：关注 hidden_size、num_layers、pooling_style
- Qwen + LoRA：关注 lora_r、lora_alpha、batch_size

---

### 7. LoRA Adapter 轻量化保存

训练大模型时仅保存 LoRA 增量权重和分类头，不保存完整 base model（节省磁盘空间）：

```
output/adapter/
├── adapter_config.json        # LoRA 结构配置
├── adapter_model.safetensors  # LoRA 增量权重（仅几 MB）
└── classify_head.pt           # 分类头权重
```

---

### 8. 详细评估指标（`evalute.py`）

每轮 eval 输出：

- 总体 accuracy
- Macro F1
- 每类 F1（negative / neutral / positive）
- 每类 accuracy（即各类召回率）

便于诊断模型在各类别上的表现，识别类别偏差。

---

### 9. 大模型自动标注流水线（LLM-based Auto Labeling）

原始推文数据来自 Twitter 气候变化话题抓取，不含人工标注。项目使用大语言模型对数据进行自动标注，分为两个阶段：

**阶段一：数据预处理（`preprocess.py`）**

对原始 Twitter JSON 流进行清洗，过滤非英文推文，提取正文（优先取 `extended_tweet.full_text`），并执行：
- 去除转发前缀（`RT @user:`）
- 去除 URL、@用户名、emoji
- 最短长度过滤（≥ 10 字符）
- 文本去重（大小写归一化后哈希比对）

清洗后按 9:1 随机划分为训练集和验证集，初始标签全部置为占位值。

**阶段二：LLM 批量标注（`relabel.py`）**

调用大语言模型，对每条推文按气候变化立场打标签（1 / 0 / -1），使用的模型为：

> **Qwen-Flash**（`qwen-flash`，阿里云 DashScope API）
> 备用：**GPT-4o-mini**（OpenAI API，通过 `OPENAI_API_KEY` 切换）

Prompt 设计为零样本（zero-shot）分类，要求模型返回严格的 JSON 整数数组，每批次 20 条：

```
Classify each tweet's stance toward climate change.
Return ONLY a JSON array of integers, one per tweet, in the same order:
  1  = pro-climate  (believes in / supports climate action)
  0  = neutral      (informational, no clear stance, or ambiguous)
 -1  = skeptical    (doubts, denies, or opposes climate change)
```

**工程可靠性设计：**

| 机制 | 说明 |
|------|------|
| 断点续标 | 每批完成后写入 checkpoint 文件，中断后可从断点恢复，避免重复调用 API |
| 指数退避重试 | 单批次失败最多重试 3 次，等待时间为 2¹、2²、2³ 秒 |
| 兜底策略 | 3 次重试全部失败时该批次标签回退为 `0`（neutral），不中断整体流程 |
| 原子写入 | 最终结果先写临时文件再原子重命名，防止写入中断损坏源文件 |
| temperature=0 | 推理温度设为 0，保证分类结果确定性，不引入随机性 |

---

## 使用的技术栈

| 类别 | 技术 |
|------|------|
| 深度学习框架 | PyTorch |
| 预训练模型 | HuggingFace Transformers（BERT、RoBERTa、Qwen2/Qwen3） |
| 参数高效微调 | PEFT（LoRA — Low-Rank Adaptation） |
| 训练框架 | HuggingFace Trainer（含 fp16 混合精度） |
| 数据处理 | HuggingFace DataCollatorWithPadding、PyTorch Dataset/DataLoader |
| 评估 | scikit-learn（F1、confusion matrix） |
| 实验管理 | pandas（结果 CSV 导出与排序） |
| 数据标注 | Qwen-Flash（阿里云 DashScope）/ GPT-4o-mini（OpenAI），OpenAI-compatible API |
| 语言 | Python 3.12 |
