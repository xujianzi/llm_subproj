
import torch
import torch.nn.functional as F
from loader import load_data
import time

"""
模型评估器
评估流程：
  1. kwnb_to_vector：将训练集知识库里的所有问题用模型编码成向量矩阵
  2. eval：对验证集每条问题编码后，与知识库向量做余弦相似度，取最近邻的标准问题索引
            与 ground truth 对比计算准确率
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        # 训练集用于构建向量知识库
        self.train_data = load_data(config["train_data_path"], config)
        # 验证集用于评估（不打乱顺序）
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct": 0, "wrong": 0}

    def kwnb_to_vector(self):
        """
        将 knwdb（训练集知识库）中所有问题编码成归一化向量矩阵。
        knwdb 结构：{ 标准问题idx: [(input_ids, attention_mask), ...] }

        构建两个并行结构：
          question_ids / question_masks：按插入顺序保存所有问题的 token 序列
          question_idx_to_standard_question_idx：第 i 个问题 → 所属标准问题索引
        """
        self.question_ids = []
        self.question_masks = []
        self.question_idx_to_standard_question_idx = {}

        for std_idx, enc_list in self.train_data.dataset.knwdb.items():
            for (input_ids, attention_mask) in enc_list:
                pos = len(self.question_ids)
                self.question_idx_to_standard_question_idx[pos] = std_idx
                self.question_ids.append(input_ids)
                self.question_masks.append(attention_mask)

        with torch.no_grad():
            ids_matrix = torch.stack(self.question_ids, dim=0)    # (N, max_length)
            mask_matrix = torch.stack(self.question_masks, dim=0) # (N, max_length)
            if torch.cuda.is_available():
                ids_matrix = ids_matrix.cuda()
                mask_matrix = mask_matrix.cuda()
            # 使用 encoder 单独编码（不需要 pair 结构），得到 (N, hidden_size) 的向量矩阵
            self.knwbd_vectors = self.model.encoder(ids_matrix, mask_matrix)
            self.knwbd_vectors = F.normalize(self.knwbd_vectors, dim=-1)  # (N, hidden_size)

    def eval(self, epoch):
        """对验证集做推理，返回 (accuracy, elapsed_time_sec)"""
        self.logger.info(f"开始评估第{epoch}轮模型")
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        # 构建知识库向量矩阵
        self.kwnb_to_vector()

        for batch_data in self.valid_data:
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            # valid batch: (input_ids, attention_mask, label_idx)
            input_ids, attention_mask, labels = batch_data
            with torch.no_grad():
                # 编码验证集问题，得到 (batch_size, hidden_size) 的向量
                test_vecs = self.model.encoder(input_ids, attention_mask)
            self.write_stats(test_vecs, labels)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_time = time.time() - start_time

        acc = self.show_stats()
        self.logger.info(f"评估第{epoch}轮模型耗时：{elapsed_time:.2f}秒")
        return acc, elapsed_time

    def write_stats(self, test_question_vecs, labels):
        """
        对 batch 中每个测试向量，在知识库向量矩阵中做最近邻检索，
        比较命中的标准问题索引与 ground truth 是否一致。
        """
        assert len(test_question_vecs) == len(labels)
        for test_vec, label_idx in zip(test_question_vecs, labels):
            # 归一化后做点积 = 余弦相似度
            test_vec = F.normalize(test_vec, dim=-1)                             # (H,)
            similarities = torch.einsum("d,nd->n", test_vec, self.knwbd_vectors) # (N,)
            hit_idx = torch.argmax(similarities).item()
            hit_std_idx = self.question_idx_to_standard_question_idx[hit_idx]
            if hit_std_idx == label_idx.item():
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1

    def show_stats(self):
        """打印并返回本轮准确率"""
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        total = correct + wrong
        self.logger.info("预测集合条目总量：%d" % total)
        self.logger.info("预测正确：%d，预测错误：%d" % (correct, wrong))
        self.logger.info("预测准确率：%.4f" % (correct / total))
        self.logger.info("--------------------")
        return correct / total
