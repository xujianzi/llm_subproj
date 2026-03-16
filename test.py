
# import torch
# import torch.nn as nn
# from torch.optim import Adam, AdamW, SGD
# from transformers import AutoModel, AutoTokenizer, AutoConfig
# from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
# from transformers.modeling_outputs import SequenceClassifierOutput

# """
# 建立网络结构
# """




# class TorchModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         hidden_size = config["hidden_size"]
#         vocab_size = AutoConfig.from_pretrained(config["pretrain_model_path"]).vocab_size # get bert vocab size
#         num_labels = config["num_labels"]
#         model_type = config["model_type"]
#         num_layers = config["num_layers"]
#         self.use_bert = False
#         self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
#         if model_type == "rnn":
#             self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
#         elif model_type == "lstm":
#             self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
#         elif model_type == "gru":
#             self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
#         # (B, L, H) 
#         elif model_type == "bert":
#             self.use_bert = True 
#             self.encoder = AutoModel.from_pretrained(config["pretrain_model_path"], return_dict=False) # return_dict=False → 返回 tuple
#             hidden_size = self.encoder.config.hidden_size

#         self.classify = nn.Sequential(
#             nn.Dropout(config["dropout_rate"]),
#             nn.Linear(hidden_size, num_labels),
#         )
#         self.pooling_style = config["pooling_style"]

#         # class_weights: list[float] | None — moves to correct device automatically via register_buffer
#         weights = config.get("class_weights")
#         if weights is not None:
#             self.register_buffer("class_weight", torch.tensor(weights, dtype=torch.float))
#         else:
#             self.register_buffer("class_weight", None)
#         self.loss = nn.CrossEntropyLoss  # store class, instantiate in forward with correct weight
    
#     def forward(self, input_ids, attention_mask=None, labels=None):
#         x = input_ids
#         if self.use_bert:
#             x = self.encoder(x, attention_mask=attention_mask)
#             # return_dict=False -> (last_hidden_state, pooler_output)
#             # pooler_output: CLS token through dense+tanh, designed for classification
#             x = x[1]  # (B, H)
#         else:
#             emb = self.embedding(x)  # (B, L, H)
#             x = self.encoder(emb)    # (B, L, H)

#             if isinstance(x, tuple):  # RNN/LSTM/GRU 类的模型同时会返回隐单元向量
#                 x = x[0]

#             if self.pooling_style == "max":
#                 x = x.max(dim=1).values   # torch中max返回（values,indices）
#             else:
#                 x = x.mean(dim=1)         # (B, H)
#             # 也可以直接使用 CLS 位置的向量: x = x[:, 0, :]
#         logits = self.classify(x)  # (B, C)
#         loss = None
#         if labels is not None:
#             loss_fn = self.loss(weight=self.class_weight)
#             loss = loss_fn(logits, labels.squeeze())
#         return SequenceClassifierOutput(loss=loss, logits=logits) # 封装模型输出，包含 loss 和 logits，方便 Trainer 计算 loss 和 logits


# if __name__ == "__main__":
#     # 这里一般是测试用例
#     from config import Config

#     Config["model_type"] = "lstm"
    
#     model = TorchModel(Config)
#     x = torch.LongTensor([[0,1,2,3,4], [5,6,7,8,9], [4,5,3,4,2]])  # (B, L)
#     print(x.shape) # (3,5)
#     output = model(x)
#     print(output.logits)
#     print(output.logits.shape) # (3,3)



import torch

print(torch.rand(3))

