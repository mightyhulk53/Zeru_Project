import json, hashlib, random, numpy as np, pandas as pd, pathlib, logging, os
from typing import List, Dict

def set_seeds(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)

def read_jsonl(path: str | pathlib.Path) -> pd.DataFrame:
    """
    Expect the raw file to be a **JSON lines** (one dict per line) OR a giant JSON list.
    We normalise into a flat DataFrame.
    """
    p = pathlib.Path(path)
    with open(p, "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == '[':         # gigantic list
            data = json.load(f)
            return pd.json_normalize(data)
        else:                    # jsonl
            records = [json.loads(line) for line in f]
            return pd.json_normalize(records)

def id_hash(wallet: str) -> int:
    """Stable numeric hash of a wallet address (we never expose raw addr in plots)."""
    return int(hashlib.sha256(wallet.encode()).hexdigest(), 16) % (10**12)

def init_logger():
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        level=logging.INFO)
