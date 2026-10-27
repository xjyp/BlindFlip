"""Reasoning module: reusable actions and mechanism-family labels."""

from .action_space import action_tags, family, render


MECHANISM_NAMES = {
    "priority": "Priority",
    "eligibility": "Eligibility",
    "quality": "Quality",
}


class Reasoning:
    """F-SAS construction without access to comparison content."""

    @staticmethod
    def actions():
        return action_tags()

    @staticmethod
    def family(tag):
        return MECHANISM_NAMES[family(tag)]

    @staticmethod
    def instantiate(tag, target):
        return render(tag, target)
