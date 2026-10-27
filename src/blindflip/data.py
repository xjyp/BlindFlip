"""Dataset validation, sampling, and content-based split checks."""

import hashlib
import json
import random
from pathlib import Path

FIELDS = ("question_id", "instruction", "response_a", "response_b")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    ensure_ascii=True).encode()).hexdigest()


def item_hash(item):
    # Candidate order cannot hide a duplicated calibration comparison.
    return digest([item["instruction"], sorted([item["response_a"], item["response_b"]])])


def load_items(path, size=0, seed=43):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, list) or not values:
        raise ValueError("Dataset must be a nonempty JSON list")
    items, seen = [], set()
    for value in values:
        if not isinstance(value, dict) or any(
                not isinstance(value.get(k), str) for k in FIELDS):
            raise ValueError(f"Each example needs string fields: {FIELDS}")
        iid = value["question_id"]
        if not iid or iid in seen:
            raise ValueError(f"Empty or duplicate question_id: {iid!r}")
        seen.add(iid)
        items.append({k: value[k] for k in FIELDS})
    if size < 0 or size > len(items):
        raise ValueError("Sample size must be between zero and dataset size")
    if size:
        selected = set(random.Random(seed).sample(sorted(seen), size))
        items = [item for item in items if item["question_id"] in selected]
    return items


def check_disjoint(items, portfolio, allow_content_overlap=False):
    ids = set(portfolio.get("calibration_item_ids", []))
    hashes = set(portfolio.get("calibration_item_hashes", []))
    id_overlap = [item["question_id"] for item in items if item["question_id"] in ids]
    overlap = [item["question_id"] for item in items if item_hash(item) in hashes]
    if id_overlap:
        raise ValueError(f"Calibration/test ID overlap detected: {id_overlap[:5]}")
    if overlap and not allow_content_overlap:
        raise ValueError(f"Calibration/test content overlap: {len(overlap)} examples, including "
                         f"{overlap[:5]}. Historical replay requires --allow-content-overlap.")
    return {"content_checked": bool(hashes), "content_overlap_count": len(overlap),
            "content_overlap_item_ids": overlap,
            "historical_overlap_allowed": allow_content_overlap}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True,
                              allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)
