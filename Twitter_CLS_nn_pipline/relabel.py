"""
relabel.py
----------
Re-labels train_data.json and valid_data.json using an LLM.

Each tweet is classified as:
  1  = positive / pro-climate  (believes in / supports climate action)
  0  = neutral                 (informational, no clear stance)
 -1  = negative / skeptical    (doubts, denies, or opposes climate concerns)

Usage:
    export OPENAI_API_KEY=sk-...
    python relabel.py

Config knobs (top of file):
    MODEL      – any OpenAI-compatible model name
    BATCH_SIZE – tweets per API call (keep ≤ 30 for reliable JSON output)
    BASE_URL   – change to point at a different OpenAI-compatible endpoint
"""


import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI

from config import Config

# load .env FIRST so os.environ.get picks up its values below
load_dotenv()

# ── tuneable parameters ──────────────────────────────────────────────────────
MODEL      = os.environ.get("LLM_MODEL_QWEN_FLASH", "gpt-4o-mini")
BATCH_SIZE = 20
MAX_RETRY  = 3
BASE_URL   = os.environ.get("DASHSCOPE_BASE_URL", None)
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_PATH = Path(Config["train_data_path"])
VALID_PATH = Path(Config["valid_data_path"])
DATA_DIR   = TRAIN_PATH.parent

VALID_LABELS = {1, 0, -1}


# ── helpers ───────────────────────────────────────────────────────────────────

def build_client() -> OpenAI:
    api_key = (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    if not api_key:
        sys.exit(
            "ERROR: No API key found.\n"
            "Set DASHSCOPE_API_KEY (or OPENAI_API_KEY) in your environment or .env file."
        )
    kwargs = {"api_key": api_key}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    return OpenAI(**kwargs)


def classify_batch(client: OpenAI, texts: list[str]) -> list[int]:
    """
    Ask the LLM to classify a batch of tweets.
    Returns a list of int labels in the same order as *texts*.
    Raises ValueError on parse failure so the caller can retry.
    """
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    prompt = (
        "Classify each tweet's stance toward climate change.\n"
        "Return ONLY a JSON array of integers, one per tweet, in the same order:\n"
        "  1  = pro-climate  (believes in / supports climate action)\n"
        "  0  = neutral      (informational, no clear stance, or ambiguous)\n"
        " -1  = skeptical    (doubts, denies, or opposes climate change)\n\n"
        f"Tweets:\n{numbered}\n\n"
        f"Return only the JSON array, e.g.: [1, 0, -1, ...]"
        f" — exactly {len(texts)} integers."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()

    # strip optional markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    labels = json.loads(raw)

    if not isinstance(labels, list) or len(labels) != len(texts):
        raise ValueError(
            f"Expected list of {len(texts)} labels, got: {raw!r}"
        )
    for lbl in labels:
        if lbl not in VALID_LABELS:
            raise ValueError(f"Invalid label value {lbl!r} — must be 1, 0, or -1")

    return [int(l) for l in labels]


# ── core relabeling logic ─────────────────────────────────────────────────────

def relabel_file(client: OpenAI, src_path: Path, dst_path: Path) -> None:
    """
    Relabels *src_path* and writes result to *dst_path*.
    A checkpoint file is kept next to *dst_path* so the run can be resumed
    if interrupted.
    """
    checkpoint_path = dst_path.with_suffix(".checkpoint.json")

    # load source
    with open(src_path, encoding="utf-8") as f:
        data: list[dict] = json.load(f)
    total = len(data)

    # resume from checkpoint if available
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            labeled: list[dict] = json.load(f)
        start = len(labeled)
        print(f"  Resuming from checkpoint - {start}/{total} already done.")
    else:
        labeled = []
        start = 0

    for batch_start in range(start, total, BATCH_SIZE):
        batch = data[batch_start: batch_start + BATCH_SIZE]
        texts = [item["text"] for item in batch]

        for attempt in range(1, MAX_RETRY + 1):
            try:
                labels = classify_batch(client, texts)
                break
            except Exception as exc:
                print(
                    f"  Batch {batch_start}-{batch_start + len(batch) - 1} "
                    f"attempt {attempt}/{MAX_RETRY} failed: {exc}"
                )
                if attempt == MAX_RETRY:
                    print("  Falling back to label=0 (neutral) for this batch.")
                    labels = [0] * len(batch)
                else:
                    time.sleep(2 ** attempt)  # exponential back-off

        for item, lbl in zip(batch, labels):
            labeled.append({"text": item["text"], "label": lbl})

        # save checkpoint after every batch
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(labeled, f, ensure_ascii=False, indent=2)

        done = min(batch_start + BATCH_SIZE, total)
        print(f"  [{done:>6}/{total}]  last batch labels: {labels}")

        time.sleep(0.3)  # gentle rate-limit buffer

    # write to a temp file first, then rename atomically so a crash
    # mid-write never corrupts the source file
    tmp_path = dst_path.with_suffix(".tmp.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)
    tmp_path.replace(dst_path)

    checkpoint_path.unlink(missing_ok=True)

    # quick stats
    counts = {1: 0, 0: 0, -1: 0}
    for item in labeled:
        counts[item["label"]] = counts.get(item["label"], 0) + 1
    print(
        f"  Done -> {dst_path.name}  "
        f"positive={counts[1]}  neutral={counts[0]}  negative={counts[-1]}"
    )


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    client = build_client()

    for src, dst in [
        (TRAIN_PATH, TRAIN_PATH),
        (VALID_PATH, VALID_PATH),
    ]:
        print(f"\nRelabeling {src.name} ...")
        relabel_file(client, src, dst)

    print("\nAll done.")


if __name__ == "__main__":
    main()
