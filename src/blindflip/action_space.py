"""Factorized Semantic Action Space (F-SAS)."""

import hashlib
import json

from .templates import DIRECTIONS, WRAPPER_TEMPLATES


def action_tags():
    return [f"{wrapper}|{direction}"
            for direction in DIRECTIONS for wrapper in WRAPPER_TEMPLATES]


def family(tag):
    wrapper, direction = tag.split("|", 1)
    if wrapper not in WRAPPER_TEMPLATES:
        raise ValueError(f"Unknown wrapper: {wrapper}")
    return DIRECTIONS[direction]["family"]


def render(tag, target):
    if target not in ("A", "B"):
        raise ValueError("Target must be A or B")
    wrapper, direction = tag.split("|", 1)
    definition = DIRECTIONS[direction]
    other = "A" if target == "B" else "B"
    claim = definition["claim"]
    values = {"criterion": definition["criterion"], "claim": claim,
              "claim_inline": " ".join(claim.split()).replace('"', "'"),
              "target": target, "other": other, "t": target, "o": other}
    # Preserve the original rendering order and whitespace, including JSON.
    text = WRAPPER_TEMPLATES[wrapper].format(**values)
    return text.replace("{t}", target).replace("{o}", other)


def fingerprint():
    value = json.dumps({"directions": DIRECTIONS, "wrappers": WRAPPER_TEMPLATES},
                       sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(value).hexdigest()
