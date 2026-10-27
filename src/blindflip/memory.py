"""Memory module: immutable surrogate-calibration experience."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Memory:
    """Per-action successes over clean-valid calibration comparisons."""

    eligible: frozenset
    hits: dict

    @classmethod
    def from_feedback(cls, clean, hits):
        eligible = frozenset(iid for iid, choice in clean.items()
                             if type(choice) is int and choice in (0, 1))
        if not eligible:
            raise ValueError("No clean-valid calibration examples")
        return cls(eligible=eligible,
                   hits={tag: frozenset(values) for tag, values in hits.items()})

    def reliability(self, tag):
        return len(self.hits[tag]) / len(self.eligible)
