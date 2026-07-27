"""Application-level demos built on top of the ROSClaw runtime."""

_LAZY_IMPORTS = {
    "InstructionParseError": ("rosclaw.apps.mobility", "InstructionParseError"),
    "MoveIntent": ("rosclaw.apps.mobility", "MoveIntent"),
    "MoveRunResult": ("rosclaw.apps.mobility", "MoveRunResult"),
    "run_language_move": ("rosclaw.apps.mobility", "run_language_move"),
    "PatrolPlan": ("rosclaw.apps.behavior_tree", "PatrolPlan"),
    "PatrolRunResult": ("rosclaw.apps.behavior_tree", "PatrolRunResult"),
    "parse_patrol_instruction": ("rosclaw.apps.behavior_tree", "parse_patrol_instruction"),
    "run_patrol_behavior_tree": ("rosclaw.apps.behavior_tree", "run_patrol_behavior_tree"),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = __import__(module_name, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

__all__ = [
    "InstructionParseError",
    "MoveIntent",
    "MoveRunResult",
    "run_language_move",
    "PatrolPlan",
    "PatrolRunResult",
    "parse_patrol_instruction",
    "run_patrol_behavior_tree",
]
