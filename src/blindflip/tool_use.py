"""Tool Use module: source-note-only action execution."""

from .reasoning import Reasoning


class ToolUse:
    """Bind a fixed action to the opposite symbolic clean preference."""

    @staticmethod
    def target_from_clean(clean_choice):
        if type(clean_choice) is not int or clean_choice not in (0, 1):
            raise ValueError("Clean preference must be 0 (A) or 1 (B)")
        return "B" if clean_choice == 0 else "A"

    @staticmethod
    def source_note(tag, clean_choice):
        return Reasoning.instantiate(tag, ToolUse.target_from_clean(clean_choice))
