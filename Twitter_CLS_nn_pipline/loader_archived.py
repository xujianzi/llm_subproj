
import json
import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
加载数据
"""

class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        '''
        创建词表
        '''
        self.config = config
        self.path = data_path
        self.label2idx, self.idx2label = build_label_index(self.path)
        self.config["class_num"] = len(self.idx2label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self._load()
    
    def _load(self):
        '''加载数据'''
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                tag = line["tag"]
                label = self.label2idx[tag]
                title = line["title"]
                if self.config["model_type"] == "bert":
                    # iput_id = self.tokenizer.encode(title, max_length = self.config["max_length"], pad_to_max_length=True)
                    encoded = self.tokenizer.encode_plus(title, max_length=self.config["max_length"],padding="max_length",truncation=True)
                    input_ids = torch.LongTensor(encoded["input_ids"])  # ['input_ids', 'token_type_ids', 'attention_mask']
                    attention_mask = torch.LongTensor(encoded["attention_mask"])
                else:
                    input_ids = self._encode_sentence(title)
                    input_ids = torch.LongTensor(input_ids)  # torch.tensor(input_ids, dtype=torch.long)
                    # 这里的 mask 对非BERT不会被用到，但为了统一接口给一个全1
                    attention_mask = torch.ones(self.config["max_length"], dtype=torch.long)
                label_idx = torch.LongTensor([label])
                self.data.append([input_ids, attention_mask, label_idx])

    def _encode_sentence(self, text):
        input_ids = []
        for char in text:
            input_ids.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_ids = self._padding(input_ids)
        return input_ids
    def _padding(self, input_ids):
        input_ids = input_ids[:self.config["max_length"]]
        input_ids += [0] * (self.config["max_length"] - len(input_ids)) 
        return input_ids
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def build_label_index(datapath):
    labels = set()
    with open(datapath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            labels.add(item["tag"])
    label2idx = {label: idx for idx, label in enumerate(sorted(labels))}
    idx2label = {idx: label for label, idx in label2idx.items()}
    return label2idx, idx2label

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            token = line.strip()
            token_dict[token] = idx + 1
    return token_dict

def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    data_path = Config["train_data_path"]
    _, idx2label = build_label_index(data_path)
    print(idx2label) # 测试idx
    
    # tokenizer = BertTokenizer.from_pretrained(Config["pretrain_model_path"])
    # print(dir(tokenizer))  # 测试 tokenizer的方法

    # dg = DataGenerator(Config["valid_data_path"], Config)
    # print(dg[1])


