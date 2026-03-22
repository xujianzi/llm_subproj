import json
from pathlib import Path
from bs4 import BeautifulSoup


def load_file_as_text(path: Path) -> str:
    ext = path.suffix.lower()
    data = path.read_bytes()
    if ext in {".txt", ".md"}:
        return data.decode("utf-8", errors="ignore")
    if ext == ".json":
        return json.dumps(json.loads(data), ensure_ascii=False, indent=2)
    # naive HTML to text fallback
    soup = BeautifulSoup(data.decode("utf-8", errors="ignore"), "html.parser")
    return soup.get_text(separator="\n")
