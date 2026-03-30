# Sentence Match FQA — 项目特色

> 基于语义向量检索的中文 FAQ 匹配系统，支持多种预训练模型的直接推理与对比学习微调，单配置项一键切换。

---

## 1. 五种模式，一行切换

通过 `config["model_type"]` 即可在五种运行模式间无缝切换，无需修改任何其他代码：

| model_type | 模型 | 是否训练 | Loss |
|---|---|---|---|
| `bert` | BERT (bert-base-chinese) | ✅ 微调 | Pair Cosine Loss |
| `bge_infer` | BGE (bge-small-zh-v1.5) | ❌ 直接推理 | — |
| `bge_finetune` | BGE | ✅ 微调 | InfoNCE |
| `qwen_infer` | Qwen3-Embedding-0.6B | ❌ 直接推理 | — |
| `qwen_finetune` | Qwen3-Embedding-0.6B | ✅ 微调 | InfoNCE |

`main.py`、`evaluate.py`、`loader.py` 对模型类型完全透明，切换成本为零。

---

## 2. 三种 Pooling 策略，对应三类模型架构

不同预训练模型有不同的最优 pooling 方式，项目针对各自架构特点分别实现：

```
BERT (encoder-only)
  → Mean Pooling：对所有非 PAD token 的隐状态加权平均
  → 充分利用双向注意力，适合短文本语义建模

BGE (encoder-only, 对比预训练)
  → CLS Pooling：取第 0 个 token 的隐状态
  → 与 BGE 预训练目标一致，CLS 天然聚合全句信息

Qwen3-Embedding (decoder-only, 因果注意力)
  → Last-Token Pooling：取最后一个真实 token 的隐状态
  → 因果架构中最后 token 通过注意力聚合了全部前序上下文
  → Tokenizer 强制左填充，确保 batch 末位始终为真实 token
```

---

## 3. InfoNCE 假负样本掩码

In-batch InfoNCE 训练时，同一 batch 内随机采样可能出现来自同一标准问的两个样本——它们在语义上是正样本，但会被 InfoNCE 错误地当作负样本惩罚。

项目通过构造**同类掩码**解决这一问题：

```python
# 同类非对角位置 → 相似度置为 -inf → softmax 权重归零 → 不产生梯度
same_class = class_labels.unsqueeze(1) == class_labels.unsqueeze(0)  # (B, B)
false_negative_mask = same_class & ~torch.eye(B, dtype=torch.bool)
sim_matrix = sim_matrix.masked_fill(false_negative_mask, float("-inf"))
```

这对本数据集尤为重要：仅 29 个标准问、batch_size=32，不处理假负样本时每个 batch 必然命中同类冲突。

---

## 4. 知识库即训练集的双重复用

训练集的所有问题在加载时被编码并存入 `knwdb`（知识库字典）：

```
knwdb = { 标准问idx: [(input_ids, attn_mask), ...] }
```

这个结构在项目中承担**两个完全不同的职责**：

- **训练时**：作为正/负样本的采样来源（正样本 = 同一 idx 下随机取两个问题）
- **评估时**：将全部问题编码为向量矩阵，作为最近邻检索的向量数据库

无需额外的索引构建步骤，训练集天然就是检索库。

---

## 5. 无分类头的向量检索评估

评估不依赖传统分类头（softmax over N classes），而是直接用**余弦相似度最近邻检索**：

```
验证问题 → encoder → 归一化向量
    ↓
与知识库所有向量做点积（归一化后点积 = 余弦相似度）
    ↓
torch.einsum("d,nd->n", test_vec, knwdb_vectors)
    ↓
argmax → 命中标准问索引 → 与 ground truth 对比
```

这与实际生产部署方式完全一致——评估指标直接反映线上效果。

---

## 6. 统一的 Encoder 接口

所有模型（BERT / BGE / Qwen）均通过 `model.encoder(input_ids, attention_mask)` 暴露单句编码能力：

```python
# evaluate.py 中，无论哪种模型，代码完全相同
self.knwbd_vectors = self.model.encoder(ids_matrix, mask_matrix)
test_vecs = self.model.encoder(input_ids, attention_mask)
```

`evaluate.py` 对底层模型类型零感知，新增模型只需实现 encoder 接口即可接入。

---

## 7. 实验结果对比

在电信客服 FAQ 数据集（29 类，464 条验证样本）上的准确率：

| 模式 | 准确率 | 备注 |
|---|---|---|
| **qwen_finetune** | **97.2%** | 最优，微调 9 epoch |
| bge_finetune | 93.97% | 微调 10 epoch |
| qwen_infer | 93.1% | **零样本**，无任何微调 |
| bert | 92.0% | 微调 4 epoch |
| bge_infer | 91.8% | **零样本**，无任何微调 |

> Qwen3-Embedding 和 BGE 的零样本表现（93%+）说明强大的预训练泛化能力；微调后 Qwen3 进一步提升至 97.2%。

---

## 简历描述

**基于对比学习的中文 FAQ 语义匹配系统**（PyTorch / Transformers）

构建了一套支持多模型横向对比的中文 FAQ 语义检索系统，在电信客服数据集（29 类标准问、464 条验证样本）上最终准确率达 **97.2%**。

- **多模型统一框架**：封装 BERT、BGE、Qwen3-Embedding 三类预训练模型，通过单一配置项 `model_type` 在"直接推理 / 对比学习微调"五种模式间无缝切换，`evaluate.py` 对底层模型类型零感知；
- **差异化 Pooling 策略**：针对不同架构分别实现 Mean Pooling（BERT 双向注意力）、CLS Pooling（BGE 对比预训练目标）、Last-Token Pooling（Qwen3 因果架构），并为 decoder-only 模型设置左填充以确保批内末位始终为真实 token；
- **InfoNCE 假负样本处理**：数据集仅 29 类而 batch_size=32，批内必然出现同标准问样本对，通过构造同类掩码将其相似度置为 −∞，从负样本集合中排除，消除对梯度的错误惩罚；
- **训练集双重复用**：`knwdb` 结构同时作为正负样本采样池（训练期）和向量检索数据库（评估/推理期），评估直接以余弦相似度最近邻检索代替分类头，与线上部署方式完全一致。
