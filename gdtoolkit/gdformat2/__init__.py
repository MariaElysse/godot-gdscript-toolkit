"""gdformat2 — AST-first GDScript formatter.

A parallel command to gdformat that takes an already-parsed Lark Tree and
serializes it into compliant GDScript code, with comments as first-class tree nodes.
"""
from .style_config import StyleConfig  # noqa: F401
from .formatter import format_code  # noqa: F401
from .safety_checks import (  # noqa: F401
    check_tree_invariant,
    check_formatting_stability,
    check_comment_persistence,
)

DEFAULT_CONFIG = {
    "excluded_directories": [".git"],
    "safety_checks": True,
    "use_spaces": False,  # Default to tabs
    "line_length": 100,
}
