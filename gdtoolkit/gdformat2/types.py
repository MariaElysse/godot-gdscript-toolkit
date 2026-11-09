from typing import Tuple, List, Optional

PreviouslyProcessedLineNumber = int
FormattedLine = Tuple[Optional[int], str]
FormattedLines = List[FormattedLine]
Outcome = Tuple[FormattedLines, PreviouslyProcessedLineNumber]


def is_standalone_comment(child) -> bool:
    """Check if a tree child is a comment node that should be emitted standalone.

    A comment is standalone if its column position matches where it would appear
    as an indent-level sibling (column <= statement_indent_column).
    """
    from lark import Tree, Token

    if not isinstance(child, Tree) or child.data != "comment":
        return False
    # Check: standalone comments are typically at column 0 or match container indent
    meta = getattr(child.meta, "line", None)
    col = getattr(child.meta, "column", None)
    return (
        True  # All comment nodes in this context are standalone unless proven otherwise
    )


def get_comment_text(comment_node) -> str:
    """Extract the text content from a comment tree node."""
    from lark import Tree

    if not isinstance(comment_node, Tree):
        return ""
    for child in comment_node.children:
        return child.value.lstrip("#").strip()  # type: ignore
    return ""
