import torch
import torch.nn as nn 
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

'''
SentenceBert内部训练的实现
'''


class SentenceBertEncder(nn.Module):
    def __init__(self, model_name:str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
    
    def mean_pooling(self, last_hidden_state:torch.Tensor, attention_mask:torch.Tensor):
        # last_hidden_state： B, L, H
        # attention_mask： B, L
        mask = attention_mask.unsqueeze(-1).float() # B, L, 1
        summed = (last_hidden_state * mask).sum(dim=1)  # B, H
        counts = mask.sum(dim=1).clamp(min=1e-9)  # B, 1
        return summed / counts  # B, H
    
    def forward(self, input_ids:torch.Tensor, attention_mask:torch.Tensor):
        outputs = self.encoder(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        sentence_embedding = self.mean_pooling(last_hidden_state, attention_mask) # B, H
        sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
        return sentence_embedding

class PairSentenceBert(nn.Module):
    def __init__(self, model_name:str):
        super().__init__()
        self.encoder = SentenceBertEncder(model_name)
    
    def forward(
        self,
        input_ids_a:torch.Tensor,
        attention_mask_a:torch.Tensor,
        input_ids_b:torch.Tensor,
        attention_mask_b:torch.Tensor,
        ) -> torch.Tensor:
        sentence_embedding_a = self.encoder(input_ids_a, attention_mask_a)
        sentence_embedding_b = self.encoder(input_ids_b, attention_mask_b)
        return sentence_embedding_a, sentence_embedding_b

def pair_cosine_loss(
    sentence_embedding_a:torch.Tensor,
    sentence_embedding_b:torch.Tensor,
    labels:torch.Tensor,
) -> torch.Tensor:
    """
    正样本（y=1）：
    loss = 1 - cos(a, b)
    负样本（y=-1）：    
    loss = max(0, cos(a, b) - margin)
    """
    loss = F.cosine_embedding_loss(

        sentence_embedding_a,
        sentence_embedding_b,
        labels,  # [-1, 1]
    )
    return loss

def pair_mse_loss(
    sentence_embedding_a:torch.Tensor,
    sentence_embedding_b:torch.Tensor,
    labels:torch.Tensor,
) -> torch.Tensor:
    # emb_a, emb_b: [B, H]
    # labels: [B]
    cos_sim = F.cosine_similarity(sentence_embedding_a, sentence_embedding_b, dim=1)
    probs = (cos_sim + 1) / 2  # [B]  [0, 1]
    loss = F.mse_loss(
        probs,
        labels, # [0, 1]
    )
    return loss

class TripletSentenceBert(nn.Module):
    def __init__(self, model_name:str):
        super().__init__()
        self.encoder = SentenceBertEncder(model_name)
    
    def forward(
        self,
        input_ids_a:torch.Tensor,
        attention_mask_a:torch.Tensor,
        input_ids_b:torch.Tensor,
        attention_mask_b:torch.Tensor,
        input_ids_c:torch.Tensor,
        attention_mask_c:torch.Tensor,
        ) -> torch.Tensor:
        sentence_embedding_a = self.encoder(input_ids_a, attention_mask_a)
        sentence_embedding_b = self.encoder(input_ids_b, attention_mask_b)
        sentence_embedding_c = self.encoder(input_ids_c, attention_mask_c)
        return sentence_embedding_a, sentence_embedding_b, sentence_embedding_c 

def triplet_loss_fn(emb_anchor, emb_positive, emb_negative, margin=0.5):
    cos_pos = F.cosine_similarity(emb_anchor, emb_positive)
    cos_neg = F.cosine_similarity(emb_anchor, emb_negative)
    loss = F.relu(cos_neg - cos_pos + margin).mean() # 这里是cosine相似度，实际上是余弦距离𝐿=max⁡(𝑑(𝑎, 𝑝)−𝑑(𝑎, 𝑛)+𝑚𝑎𝑟𝑔𝑖𝑛,  0)
    return loss