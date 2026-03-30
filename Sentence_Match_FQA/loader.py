
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import AutoTokenizer

"""
数据加载器
使用 HuggingFace Tokenizer 对句子编码，knwdb 存储 (input_ids, attention_mask) 对。

训练样本格式随 model_type 不同：
  bert        → [ids_a, mask_a, ids_b, mask_b, label]  有标签正/负对，用于 pair_cosine_loss
  bge_finetune→ [ids_a, mask_a, ids_p, mask_p, class_label]  正对 + 类别标签（用于屏蔽假负样本）
  bge_infer   → 无训练，loader 只用于构建知识库向量和验证集评估

验证集统一返回：(input_ids, attention_mask, label_idx)
"""


class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # 根据 model_type 选择对应的 tokenizer 路径
        model_type = config["model_type"]
        if model_type in ("bge_infer", "bge_finetune"):
            tok_path = config["bge_model_path"]
        elif model_type in ("qwen_infer", "qwen_finetune"):
            tok_path = config["qwen_model_path"]
        else:
            tok_path = config["model_path"]

        # 直接传 Path 对象，新版 huggingface_hub 对字符串做 repo ID 校验但对 Path 对象不做
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)

        # Qwen3-Embedding 是 decoder-only 架构，必须左填充：
        # 左填充保证 batch 中每条序列的最后一列都是真实 token，
        # 使 last-token pooling 能正确取到句子的语义表示。
        # decoder-only 模型默认没有 pad_token，需手动指定为 eos_token，
        # 否则 tokenizer 在填充时报错或产生错误的 input_ids。
        if model_type in ("qwen_infer", "qwen_finetune"):
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"]
        self.max_length = config["max_length"]
        self.data_type = None
        self._load()

    def _load(self):
        self.data = []
        # 知识库：标准问题索引 → [(input_ids, attention_mask), ...] 列表
        self.knwdb = defaultdict(list)

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line, dict):  # 训练集：每行是 {questions, target}
                    self.data_type = "train"
                    label = line["target"]
                    for question in line["questions"]:
                        input_ids, attention_mask = self.encode_sentence(question)
                        self.knwdb[self.schema[label]].append(
                            (input_ids, attention_mask)
                        )
                else:  # 验证集：每行是 [question, label]
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_ids, attention_mask = self.encode_sentence(question)
                    label_idx = torch.tensor(self.schema[label], dtype=torch.long)
                    self.data.append([input_ids, attention_mask, label_idx])

    def encode_sentence(self, sentence: str):
        """
        使用 BERT tokenizer 编码单个句子。
        返回 (input_ids, attention_mask)，均为形状 (max_length,) 的 LongTensor。
        """
        encoded = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)       # (max_length,)
        attention_mask = encoded["attention_mask"].squeeze(0)  # (max_length,)
        return input_ids, attention_mask

    def random_train_sample(self):
        """
        按 model_type 采样训练数据：

        bert：
          按 positive_sample_rate 概率采正/负对，返回 5 个元素
          [ids_a, mask_a, ids_b, mask_b, label(+1/-1)]

        bge_finetune：
          只采正对（InfoNCE 在 batch 内自动构造负样本），返回 4 个元素
          [ids_a, mask_a, ids_p, mask_p]
        """
        standard_question_idx = list(self.knwdb.keys())

        if self.config["model_type"] in ("bge_finetune", "qwen_finetune"):
            # ---- BGE 微调：只需正对 + 类别标签 ----
            # 类别标签用于 infonce_loss 中屏蔽假负样本（同标准问的其他对）
            a_idx = random.choice(standard_question_idx)
            if len(self.knwdb[a_idx]) < 2:
                return self.random_train_sample()
            (ids_a, mask_a), (ids_p, mask_p) = random.sample(self.knwdb[a_idx], 2)
            class_label = torch.tensor(a_idx, dtype=torch.long)
            return [ids_a, mask_a, ids_p, mask_p, class_label]

        else:
            # ---- BERT：有标签的正/负对 ----
            if random.random() <= self.config["positive_sample_rate"]:
                a_idx = random.choice(standard_question_idx)
                if len(self.knwdb[a_idx]) < 2:
                    return self.random_train_sample()
                (ids_a, mask_a), (ids_b, mask_b) = random.sample(self.knwdb[a_idx], 2)
                label = torch.tensor(1, dtype=torch.float)
            else:
                a_idx, b_idx = random.sample(standard_question_idx, 2)
                ids_a, mask_a = random.choice(self.knwdb[a_idx])
                ids_b, mask_b = random.choice(self.knwdb[b_idx])
                label = torch.tensor(-1, dtype=torch.float)
            return [ids_a, mask_a, ids_b, mask_b, label]

    def __len__(self):
        if self.data_type == "train":
            return self.train_data_size
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        """
        训练集：随机采样，返回 [ids_a, mask_a, ids_b, mask_b, label]
        验证集：按顺序返回 [input_ids, attention_mask, label_idx]
        """
        if self.data_type == "train":
            return self.random_train_sample()
        else:
            return self.data[index]


def load_schema(schema_path):
    """加载标准问题 → 整数索引 的映射"""
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return schema


def load_data(data_path, config, shuffle=True):
    """构建 DataGenerator 并包装成 DataLoader"""
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    # 验证训练集加载
    dg = DataGenerator(Config["train_data_path"], Config)
    sample = dg[0]
    print("训练样本数量:", len(dg))
    print("一条训练样本 ids_a shape:", sample[0].shape)
    print("label:", sample[4])
