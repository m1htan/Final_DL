import json
from pathlib import Path
from typing import Any, Dict

CACHE_PATH = Path("db") / "ohcache.json"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_cache() -> Dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}

def save_cache(data: Dict[str, Any]) -> None:
    with CACHE_PATH.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
