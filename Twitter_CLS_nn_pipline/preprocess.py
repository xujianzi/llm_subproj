import json
import random
import re
from typing import List, Dict, Optional
from config import Config, BASE_DIR
from pathlib import Path 




INPUT_FILE = BASE_DIR / "data" / "climate_tweets.txt"

TRAIN_FILE = Config["train_data_path"]
VALID_FILE = Config["valid_data_path"]

TRAIN_RATIO = 0.9
RANDOM_SEED = 42

# 先简单设成 正样本
DEFAULT_LABEL = 1

# 只保留英文
TARGET_LANG = "en"

# 最短文本长度
MIN_TEXT_LEN = 10


def remove_emoji(text: str) -> str:
    """
    去掉常见 emoji
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)


def extract_text(tweet: Dict) -> str:
    """
    优先取长文本 full_text，其次取 text
    """
    if "extended_tweet" in tweet and isinstance(tweet["extended_tweet"], dict):
        full_text = tweet["extended_tweet"].get("full_text")
        if full_text:
            return full_text

    # 有些 tweet 可能在 retweeted_status 里有更完整内容
    if "retweeted_status" in tweet and isinstance(tweet["retweeted_status"], dict):
        rt_obj = tweet["retweeted_status"]
        if "extended_tweet" in rt_obj and isinstance(rt_obj["extended_tweet"], dict):
            full_text = rt_obj["extended_tweet"].get("full_text")
            if full_text:
                return full_text
        if rt_obj.get("text"):
            return rt_obj["text"]

    return tweet.get("text", "")


def remove_rt_prefix(text: str) -> str:
    """
    把 'RT @someone: xxx' 前缀去掉，只保留正文
    """
    text = re.sub(r"^RT\s+@\w+:\s*", "", text)
    return text.strip()


def clean_text(text: str) -> str:
    """
    基础清洗
    """
    text = remove_rt_prefix(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)   # 去 URL
    text = re.sub(r"@\w+", "", text)               # 去 @用户名
    text = re.sub(r"&amp;", "&", text)             # HTML 转义简单处理
    text = remove_emoji(text)                      # 去 emoji
    text = re.sub(r"\s+", " ", text)               # 多空格合并
    return text.strip()


def build_label(tweet: Dict, text: str) -> int:
    """
    自动生成标签
    目前先把 climate_tweets.txt 里的数据都设为正样本 1
    """
    return DEFAULT_LABEL


def is_target_language(tweet: Dict, target_lang: str = "en") -> bool:
    """
    语言过滤：
    只保留 tweet['lang'] == target_lang
    """
    return tweet.get("lang") == target_lang


def load_and_process(file_path: str) -> List[Dict]:
    """
    逐行读取并处理
    """
    results = []
    seen = set()

    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                tweet = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 1) 语言过滤
            if not is_target_language(tweet, TARGET_LANG):
                continue

            # 2) 提取文本
            text = extract_text(tweet)
            if not text:
                continue

            # 3) 清洗
            text = clean_text(text)

            # 4) 长度过滤
            if len(text) < MIN_TEXT_LEN:
                continue

            # 5) 去重
            norm_text = text.lower()
            if norm_text in seen:
                continue
            seen.add(norm_text)

            # 6) 生成标签
            label = build_label(tweet, text)

            results.append({
                "text": text,
                "label": label
            })

    return results


def split_dataset(data: List[Dict], train_ratio: float = 0.9):
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    valid_data = data[split_idx:]
    return train_data, valid_data


def save_json(data: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    random.seed(RANDOM_SEED)

    data = load_and_process(INPUT_FILE)

    print(f"Total cleaned samples: {len(data)}")

    train_data, valid_data = split_dataset(data, TRAIN_RATIO)

    print(f"Train size: {len(train_data)}")
    print(f"Valid size: {len(valid_data)}")

    save_json(train_data, TRAIN_FILE)
    save_json(valid_data, VALID_FILE)

    print("Done.")


if __name__ == "__main__":
    main()