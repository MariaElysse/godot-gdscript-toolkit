"""Comment handling utilities for gdformat2.

Comments are now tree nodes embedded in the parse AST (Tree("comment", [...])).
This module provides helpers to work with them.
"""
from typing import List, Optional

from lark import Tree


def gather_comments(tree: Tree) -> List[str]:
    """Extract all comment texts from a parsed tree."""
    return [get_comment_text(node) for node in _find_all_comments(tree)]


def find_standalone_comments(statement: Tree) -> List[str]:
    """Find standalone comment children of a statement node."""
    comments = []
    for child in getattr(statement, "children", []):
        if isinstance(child, Tree) and str(child.data) == "comment":  # type: ignore
            comments.append(get_comment_text(child))
    return comments


def find_inline_comment(statement: Tree) -> Optional[str]:
    """Find inline comment (appearing after code on same line)."""
    for child in getattr(statement, "children", []):
        if isinstance(child, Tree) and str(child.data) == "comment":  # type: ignore
            meta = getattr(child, "meta", None)
            col = getattr(meta, "column", 0) if meta else 0
            if col > 2:  # Appears after some code content
                return get_comment_text(child)
    return None


def _find_all_comments(tree: Tree) -> List[Tree]:
    """Recursively find all comment nodes in a tree."""
    comments = []
    for child in getattr(tree, "children", []):
        if isinstance(child, Tree):
            # lark's `Tree.data` is a plain str, so the `.value` branch is
            # defensive dead code - guarded by `hasattr`, never taken at runtime.
            data = (
                str(child.data)
                if not hasattr(child.data, "value")
                else child.data.value  # type: ignore[attr-defined]
            )  # type: ignore
            if data == "comment":
                comments.append(child)
            elif hasattr(child, "data"):  # It's a non-comment tree node
                comments.extend(_find_all_comments(child))
    return comments


def normalize_comment(raw: str) -> str:
    """Apply the style guide's comment-spacing rule to a raw comment string.

    Regular and documentation comments start with a space (`# text`, `## text`), but
    code region markers keep no space (`#region`, `#endregion`), and a bare marker stays
    bare (`#`).

    Whitespace already present after the marker is preserved rather than collapsed, so
    the indentation of commented-out code survives (`#\\t\\tif x:` stays as written). Only
    trailing whitespace is dropped."""
    i = 0
    while i < len(raw) and raw[i] == "#":
        i += 1
    marker = raw[:i] or "#"
    body = raw[i:].rstrip()
    if not body:
        return marker
    if body.startswith("region") or body.startswith("endregion"):
        return f"{marker}{body}"
    # A comment starts with a space. Whitespace after that is preserved, so the indentation
    # of commented-out code survives (`# \t\tif x:` stays as written). A tab straight after
    # the marker is neither a text comment nor commented-out code - the code signal is a
    # space and then the tab - so the space goes in front of it.
    if body[0] == "\t":
        return f"{marker} {body}"
    if body[0] == " ":
        return f"{marker}{body}"
    return f"{marker} {body}"


def get_comment_text(comment_node: Tree) -> str:
    """Extract the text content from a comment tree node."""
    for child in getattr(comment_node, "children", []):
        if hasattr(child, "value"):
            return normalize_comment(str(child.value))
    return ""
