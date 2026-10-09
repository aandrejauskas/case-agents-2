"""Helpers para carregar os dados prontos em `data/`."""
import json
from pathlib import Path
from typing import List, Tuple

from common.schemas import Tool

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_tools() -> List[Tool]:
    raw = json.loads((DATA_DIR / "tools_registry.json").read_text(encoding="utf-8"))
    return [Tool(**item) for item in raw]


def load_router_training_data() -> Tuple[List[str], List[str]]:
    raw = json.loads((DATA_DIR / "router_training_data.json").read_text(encoding="utf-8"))
    texts = [item["text"] for item in raw]
    labels = [item["label"] for item in raw]
    return texts, labels


def load_eval_dataset() -> List[dict]:
    return json.loads((DATA_DIR / "eval_dataset.json").read_text(encoding="utf-8"))
