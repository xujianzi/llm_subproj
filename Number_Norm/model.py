import torch
import torch.nn as nn
import torch.nn.functional as F

class CharLM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        # input_ids: [B, L]
        emb = self.embedding(input_ids)     # [B, L, D]
        output, _ = self.lstm(emb)          # [B, L, H]
        logits = self.fc(output)            # [B, L, V]
        return logits

# 假设 input_ids, target_ids 已经准备好
# input_ids:  [B, L]
# target_ids: [B, L]

model = CharLM(vocab_size=5000, emb_dim=128, hidden_dim=256)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

input_ids = torch.randint(0, 5000, (16, 32))
target_ids = torch.randint(0, 5000, (16, 32))

logits = model(input_ids)                  # [16, 32, 5000]

loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),   # [16*32, 5000]
    target_ids.reshape(-1)                 # [16*32]
)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(loss.item())