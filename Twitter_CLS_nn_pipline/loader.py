"""
loader_v2.py
------------
Data loading for Twitter climate-change stance classification.

Labels stored in JSON:  -1 (skeptical)  0 (neutral)   1 (pro-climate)
Labels fed to model:     0               1              2
  (shifted by +1 so CrossEntropyLoss sees 0-indexed class indices)

Train : 8000 randomly sampled records from train_data.json
Valid : all records from valid_data.json

Design notes
------------
* All model types (bert / lstm / rnn / gru …) share the same BERT tokenizer.
  - BERT tokenizer adds [CLS] (id=101) at position 0 and [SEP] at the end.
  - For BERT:           [CLS] is essential (pooler is built on it).
  - For RNN/LSTM/GRU:   max- or mean-pooling is applied over every position,
                        so [CLS] is just one more token; the model learns to
                        handle it. No need to strip it.

* Dynamic padding via DataCollatorWithPadding:
  - _encode() only truncates to max_length; it never pads.
  - The collator pads each batch to the longest sequence in that batch,
    which is shorter (on average) than always padding to max_length.

* Batch format returned by the DataLoader is a **dict**, not a tuple:
      batch["input_ids"]      : LongTensor (B, L_padded)
      batch["attention_mask"] : LongTensor (B, L_padded)
      batch["labels"]         : LongTensor (B,)

  Update your training loop accordingly:
      input_ids      = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      labels         = batch["labels"].to(device)
"""

import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, DataCollatorWithPadding

from config import Config

# ── label mappings (importable by other modules) ──────────────────────────────
LABEL_MAP   = {-1: 0,  0: 1,  1: 2}    # raw label  → model class index
IDX2LABEL   = { 0: -1, 1: 0,  2: 1}    # model index → raw label
LABEL_NAMES = { 0: "negative", 1: "neutral", 2: "positive"}

TRAIN_SIZE  = 8000


# ── dataset ───────────────────────────────────────────────────────────────────

class TwitterDataset(Dataset):
    """
    __getitem__ returns a plain dict so DataCollatorWithPadding can pad it:
        {
            "input_ids"      : list[int],   # [CLS] + token ids + [SEP], truncated
            "attention_mask" : list[int],   # all 1s (no padding at this stage)
            "labels"         : int,         # 0 / 1 / 2
        }
    token_type_ids is excluded — model.py does not use it.
    """

    def __init__(self, records: list[dict], config: dict, tokenizer: BertTokenizer):
        self.tokenizer = tokenizer
        self.max_len   = config["max_length"]
        self.data      = [self._encode(rec) for rec in records]

    def _encode(self, rec: dict) -> dict:
        label_idx = LABEL_MAP[rec["label"]]

        # truncation=True  → cut to max_length if needed.
        # padding=False    → no padding here; DataCollatorWithPadding handles it.
        # return_tensors=None → plain Python lists (collator converts to tensors).
        # return_token_type_ids=False → single-sentence; not used by model.py.
        enc = self.tokenizer(
            rec["text"],
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_tensors=None,
            return_token_type_ids=False,
        )

        return {
            "input_ids":      enc["input_ids"],       # list[int]
            "attention_mask": enc["attention_mask"],  # list[int]
            "labels":         label_idx,              # int
        }
        # return [enc["input_ids"], enc["attention_mask"], label_idx]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> dict:
        return self.data[index]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── public API ────────────────────────────────────────────────────────────────

def load_data(config: dict, seed: int | None = None):
    """
    Build and return (train_loader, valid_loader).

    Args:
        config : Config dict; mutated in-place to set config["num_labels"] = 3.
        seed   : random seed for train sampling; falls back to config["seed"].

    Returns:
        train_loader : DataLoader, 8000 randomly sampled training records.
        valid_loader : DataLoader, all validation records.
    """
    if seed is None:
        seed = config.get("seed", 42)

    all_train = _load_json(config["train_data_path"])
    all_valid = _load_json(config["valid_data_path"])

    rng           = random.Random(seed)
    train_records = rng.sample(all_train, min(TRAIN_SIZE, len(all_train)))
    # train_records = all_train
    valid_records = all_valid

    tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
    config["num_labels"] = len(LABEL_MAP)  # 3

    train_ds = TwitterDataset(train_records, config, tokenizer)
    valid_ds = TwitterDataset(valid_records, config, tokenizer)

    # DataCollatorWithPadding pads each batch to its own longest sequence.
    # It handles "input_ids" and "attention_mask"; "labels" is stacked as-is.
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt") #导入tokenizer告诉模型用什么做padding

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
        collate_fn=collator,
    )
    return train_loader, valid_loader


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, valid_loader = load_data_v2(Config)

    print(f"Train batches : {len(train_loader):4d}  ({len(train_loader.dataset)} samples)")
    print(f"Valid batches : {len(valid_loader):4d}  ({len(valid_loader.dataset)} samples)")

    batch = next(iter(train_loader))
    L = batch["input_ids"].shape[1]
    print(f"\nBatch keys     : {list(batch.keys())}")
    print(f"input_ids      : {tuple(batch['input_ids'].shape)}  "
          f"(dynamic padding -> L={L}, vs fixed max_length={Config['max_length']})")
    print(f"attention_mask : {tuple(batch['attention_mask'].shape)}")
    print(f"labels         : {tuple(batch['labels'].shape)}  "
          f"values={batch['labels'][:8].tolist()}")
    print(f"  -> raw labels : {[IDX2LABEL[i] for i in batch['labels'][:8].tolist()]}")
    print(f"  -> names      : {[LABEL_NAMES[i] for i in batch['labels'][:8].tolist()]}")
