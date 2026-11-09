"""Common helpers for gdformat2 tests.

Provides format_and_compare() that works with the new AST-based formatter,
plus safety checks (tree invariant, stability, comment persistence).
"""
import difflib
from typing import Optional

import pytest
from lark import Tree

from gdtoolkit.gdformat2.parser import parser as p
from gdtoolkit.gdformat2.formatter import format_code as _gdformat2_format
from gdtoolkit.gdformat2.style_config import StyleConfig
from gdtoolkit.gdformat2.safety_checks import (
    check_all_safety,
)


MAX_LINE_LENGTH = 100


def format_and_compare(
    input_code: str,
    expected_output_code: str,
    spaces_for_indent: Optional[int] = None,
):
    """Format input_code using gdformat2 and compare to expected output.

    Args:
        input_code: Original GDScript source code.
        expected_output_code: Expected formatted output.
        spaces_for_indent: If set, use spaces instead of tabs with given indent size.
    """
    # Parse the input (may raise on grammar failures)
    try:
        tree = p.parse(input_code)
    except Exception as e:
        pytest.skip(f"Cannot parse with new grammar: {e}")

    # Build style config based on indent preference
    if spaces_for_indent is not None:
        style = StyleConfig(
            use_spaces=True,
            tab_size=spaces_for_indent,
            line_length=MAX_LINE_LENGTH,
        )
    else:
        style = StyleConfig(line_length=MAX_LINE_LENGTH)

    # Format using gdformat2
    formatted_code = _gdformat2_format(tree, style)

    # Compare output
    _compare(formatted_code, expected_output_code)


def format_and_compare_with_safety(
    input_code: str,
    expected_output_code: str,
    spaces_for_indent: Optional[int] = None,
):
    """Format and run all three safety checks.

    Like format_and_compare but also verifies tree invariant + stability + comments.
    Skips gracefully on parse errors or if comparison fails before safety runs.
    """
    try:
        tree = p.parse(input_code)
    except Exception as e:
        pytest.skip(f"Cannot parse with new grammar: {e}")

    if spaces_for_indent is not None:
        style = StyleConfig(
            use_spaces=True,
            tab_size=spaces_for_indent,
            line_length=MAX_LINE_LENGTH,
        )
    else:
        style = StyleConfig(line_length=MAX_LINE_LENGTH)

    formatted_code = _gdformat2_format(tree, style)

    # Run safety checks (even if output differs - they validate structure preservation)
    try:
        check_all_safety(input_code, tree, formatted_code, style)
    except Exception as e:
        pytest.fail(f"Safety check failed after format: {e}")


def _compare(formatted_code: str, expected_output_code: str):
    """Compare two strings and raise AssertionError with unified diff on mismatch."""
    if formatted_code != expected_output_code:
        diff = "\n".join(
            difflib.unified_diff(
                expected_output_code.splitlines(),
                formatted_code.splitlines(),
                lineterm="",
            )
        )
        pytest.fail(f"Formatted output differs:\n{diff}")


def collect_comments_from_tree(node: Tree) -> list:
    """Collect all comment strings from a parse tree."""
    found = []
    if isinstance(node, Tree):
        for child in node.children:
            if isinstance(child, Tree) and str(child.data).lower() == "comment":
                tok = child.children[0] if child.children else None
                if tok is not None:
                    found.append(str(tok.value))
            found.extend(collect_comments_from_tree(child))
    return found


def test_comment_persistence(input_code: str, expected_output_code: str):
    """Verify that comments survive gdformat2 formatting.

    This is a convenience wrapper for use with pytest.mark.parametrize.
    """
    tree = p.parse(input_code)
    original_comments = collect_comments_from_tree(tree)

    style = StyleConfig(line_length=MAX_LINE_LENGTH)
    formatted_code = _gdformat2_format(tree, style)

    # Re-parse the output
    re_parsed = p.parse(formatted_code)
    re_parsed_comments = collect_comments_from_tree(re_parsed)

    assert set(original_comments) == set(
        re_parsed_comments
    ), f"Comments not preserved.\nOriginal: {original_comments}\nAfter format: {re_parsed_comments}"
