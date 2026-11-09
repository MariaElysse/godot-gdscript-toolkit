from ..common.utils import get_line
from .types import FormattedLines, Outcome


def format_simple_statement(
    statement_name: str, statement, indent_string: str, _indent_level: int, style=None
) -> Outcome:
    """Format a simple single-word statement (pass, break, continue, breakpoint).

    Returns [(line_number, "statement_name")] tuple.
    Also handles any comment children of the statement node.
    """
    lines: FormattedLines = []
    stmt_line = get_line(statement)
    inline_comment = None

    # A comment child on the statement's own line rides along on it; one on an earlier line
    # is a standalone comment and gets its own. Emitting every comment on its own line is what
    # moved `extends Node2D  # World` onto two lines, since the head statements are formatted
    # through here and their trailing comment is a child.
    for child in getattr(statement, "children", []):
        from lark import Tree

        if isinstance(child, Tree) and _tree_data_str(child) == "comment":
            comment_text = _get_comment_text(child)
            if get_line(child) == stmt_line:
                inline_comment = comment_text
            else:
                lines.append((get_line(child), f"{indent_string}{comment_text}"))

    # The actual statement
    if inline_comment:
        statement_name = f"{statement_name}  {inline_comment}"
    lines.append((stmt_line, f"{indent_string}{statement_name}"))

    return (lines, stmt_line)


def _tree_data_str(node):
    """Return a tree's rule name as a plain string."""
    from lark import Tree

    if isinstance(node, Tree):
        d = node.data
        return d.value if hasattr(d, "value") else str(d)
    return ""


def _get_comment_text(comment_node) -> str:
    """Extract the text content from a comment tree node."""
    from lark import Tree
    from .comments import normalize_comment

    if not isinstance(comment_node, Tree):
        return ""
    for child in comment_node.children:
        # Child is usually a Token with .value
        return normalize_comment(str(child.value))  # type: ignore
    return ""
