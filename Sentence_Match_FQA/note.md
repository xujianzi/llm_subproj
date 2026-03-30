帮我建立architecture图，需要展示这个FQA文本项目的架构，
- 包括训练的流程：训练数据和验证数据， 模型的结构策略和loss选择, 输出和评估（直接进行归一化后与knwdb中的向量进行cosine相似度计算）
- 最后的使用：
  - 输入问题和知识库，输出问题的向量表示，进而输出标准问



  config.py
  - model_trained_path 改名为 model_dir（与 main.py 中 config["model_dir"] 一致）
  - model_type 改为 "bert"
  - 补充各字段说明注释

  loader.py
  - 移除 jieba、手工词表加载，改用 AutoTokenizer.from_pretrained
  - knwdb 存储格式从 tensor 改为 (input_ids, attention_mask) 元组
  - random_train_sample 改为按 positive_sample_rate 概率生成有标签的句子对 [ids_a, mask_a, ids_b, mask_b, label]，与 pair_cosine_loss 匹配
  - 验证集 __getitem__ 改为返回 [input_ids, attention_mask, label_idx]
  - 移除 load_vocab，保留 load_schema

  main.py
  - 导入改为 PairSentenceBert, pair_cosine_loss
  - 训练循环解包 5 个元素，先得到 emb_a, emb_b 再算 pair_cosine_loss
  - 超参搜索去掉 hidden_size（BERT 不需要），改为只搜 lr 和 batch_size

  evaluate.py
  - kwnb_to_vector 从 knwdb 中同时取 input_ids 和 attention_mask，用 self.model.encoder 编码（单句推理不需要 pair 结构）
  - eval 验证循环解包 3 个元素，传入 attention_mask 给 encoder