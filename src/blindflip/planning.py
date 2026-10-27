"""Planning module: MATADOR portfolio construction and validation."""

from dataclasses import dataclass

from .action_space import action_tags, family, fingerprint
from .memory import Memory


CALIBRATION_SCHEMA = "blindflip.calibration.v1"
PORTFOLIO_SCHEMA = "blindflip.portfolio.v1"


def valid_choice(value):
    return type(value) is int and value in (0, 1)


@dataclass(frozen=True)
class PortfolioPlan:
    """Frozen action ordering and the family-covering prefix."""

    ordered_actions: tuple
    family_champions: tuple


class MATADOR:
    """Exact additive-reliability planner with mechanism-family coverage."""

    @staticmethod
    def plan(memory, budget):
        tags = action_tags()
        if not 1 <= budget <= len(tags):
            raise ValueError(f"Budget must be between 1 and {len(tags)}")
        if set(memory.hits) != set(tags):
            raise ValueError("Calibration must contain exactly all 66 action columns")
        key = lambda tag: (len(memory.hits[tag]), tag)
        families = sorted({family(tag) for tag in tags})
        champions = [max((tag for tag in tags if family(tag) == name), key=key)
                     for name in families]
        champions.sort(key=key, reverse=True)
        ordered = champions + [tag for tag in sorted(tags, key=key, reverse=True)
                               if tag not in champions]
        return PortfolioPlan(tuple(ordered), tuple(champions))


def select_portfolio(matrix, budget=2):
    """Build a frozen MATADOR portfolio from surrogate calibration memory."""
    tags = action_tags()
    if not 1 <= budget <= len(tags):
        raise ValueError(f"Budget must be between 1 and {len(tags)}")
    if matrix.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(f"Expected a {CALIBRATION_SCHEMA} matrix")
    if matrix.get("action_space_sha256") != fingerprint():
        raise ValueError("Action space differs from calibration")
    clean = matrix["clean"]
    if not clean or any(value is not None and not valid_choice(value)
                        for value in clean.values()):
        raise ValueError("Clean judgments must be 0, 1, or null")
    eligible = {iid for iid, choice in clean.items() if valid_choice(choice)}
    if not eligible:
        raise ValueError("No clean-valid calibration examples")
    if set(matrix["hits"]) != set(tags):
        raise ValueError("Calibration must contain exactly all 66 action columns")
    hits = {}
    for tag in tags:
        values = matrix["hits"][tag]
        if len(values) != len(set(values)) or not set(values) <= eligible:
            raise ValueError(f"Invalid hit IDs for {tag}")
        hits[tag] = set(values)
    memory = Memory.from_feedback(clean, hits)
    plan = MATADOR.plan(memory, budget)
    ordered = list(plan.ordered_actions)
    return {
        "schema": PORTFOLIO_SCHEMA,
        "simulation": bool(matrix.get("simulation", False)),
        "action_space_sha256": fingerprint(),
        "source": matrix.get("source", {}),
        "calibration_item_ids": sorted(clean),
        "calibration_item_hashes": matrix.get("calibration_item_hashes", []),
        "calibration_cost": matrix.get("calibration_cost", {}),
        "calibration_eligible": len(eligible),
        "budget": budget,
        "selected_actions": ordered[:budget],
        "ordered_actions": ordered,
        "family_champions": list(plan.family_champions),
        "reliability": {tag: memory.reliability(tag) for tag in tags},
        "tie_break": "descending (individual_hits, action_tag)",
    }


def validate_portfolio(portfolio):
    if portfolio.get("schema") != PORTFOLIO_SCHEMA:
        raise ValueError(f"Expected a {PORTFOLIO_SCHEMA} portfolio")
    if portfolio.get("action_space_sha256") != fingerprint():
        raise ValueError("Portfolio action space does not match")
    tags = action_tags()
    ordered = portfolio["ordered_actions"]
    if len(ordered) != len(tags) or set(ordered) != set(tags):
        raise ValueError("Portfolio must contain a permutation of all 66 actions")
    budget = portfolio["budget"]
    if type(budget) is not int or not 1 <= budget <= len(tags):
        raise ValueError("Invalid portfolio budget")
    if portfolio["selected_actions"] != ordered[:budget]:
        raise ValueError("Selected actions must be the frozen ordering's prefix")
