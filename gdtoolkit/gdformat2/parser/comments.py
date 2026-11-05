"""Attach comment tokens collected by the postlexer into the parse tree."""

from typing import List, Optional, Tuple

from lark import Token, Tree


# Container rules that receive variable-length children and can hold comments.
# Fixed-position nodes (e.g. function_decl) never get comments so consumers
# can still index their children safely. These names match the root gdscript.lark.
CONTAINER_RULES = frozenset({
    "start",
    "class_block",
    "suite",
    "match_stmt",
    "property",
    "arguments",
    "elements",
    "dict_entries",
    "parameters",
    "signal_parameters",
    "enum_entries",
    "array_pattern_items",
    "dict_pattern_items",
    "property",
    "untitled_property",
})


# Bracketed constructs whose comments belong to them even when there is no list child to
# hold them (an empty collection's only content is the comment). A comment inserted here
# becomes a direct child of the bracketed node, which the layout helpers read.
BRACKETED_RULES = frozenset({
    "array_literal",
    "dict_literal",
    "array_pattern",
    "dict_pattern",
    "call_suffix",
})


def _span(node) -> Optional[Tuple[int, int]]:
    if isinstance(node, Token):
        if node.start_pos is None:
            return None
        return node.start_pos, node.end_pos
    if isinstance(node, Tree) and not getattr(node.meta, "empty", True):
        return node.meta.start_pos, node.meta.end_pos
    return None


def _comment_node(tok: Token) -> Tree:
    """Create a comment tree node from a COMMENT token."""
    node = Tree("comment", [tok])
    m = node.meta
    m.empty = False
    m.line, m.column, m.start_pos = tok.line, tok.column, tok.start_pos
    m.end_line, m.end_column, m.end_pos = (
        tok.end_line, tok.end_column, tok.end_pos
    )
    return node


def _find_container(tree: Tree, pos: int) -> Tree:
    """Innermost container enclosing ``pos`` (falls back to ``tree``)."""
    target = tree
    node = tree
    while True:
        nxt = None
        for child in node.children:
            if isinstance(child, Tree):
                sp = _span(child)
                if sp and sp[0] <= pos < sp[1]:
                    nxt = child
                    break
        if nxt is None:
            # Nothing lexically encloses the comment. When this node brackets a list
            # container (`array_literal`, `dict_literal`, `call_suffix`, `signal_decl`,
            # `function_decl`, ...), the comment is inside those brackets but before the
            # first item or after the last one, and the list's own span covers only the
            # items, so the comment belongs to the list rather than to the enclosing
            # block. Without this the comment was hoisted out of the expression and
            # emitted as a standalone statement next to it.
            for child in node.children:
                if isinstance(child, Tree) and child.data in CONTAINER_RULES:
                    return child
            if node.data in BRACKETED_RULES:
                # An empty collection's only content is the comment and its list child is
                # a None placeholder, so the bracketed node itself holds the comment
                # instead of it being hoisted out of the expression.
                return node
            return target
        node = nxt
        if node.data in CONTAINER_RULES:
            target = node


def _insert_ordered(container: Tree, comment: Tree, pos: int) -> None:
    for i, child in enumerate(container.children):
        sp = _span(child)
        if sp and sp[0] > pos:
            container.children.insert(i, comment)
            return
    container.children.append(comment)


def attach_comments(tree: Tree, comments: List[Token]) -> Tree:
    """Insert each comment token into ``tree`` as a ``comment`` node.

    The tree must have been built with ``propagate_positions=True`` and the
    postlexer's :py:attr:`GDScriptIndenter.comments` list passed here.
    """
    seen = set()
    for tok in sorted(comments, key=lambda t: t.start_pos):
        if tok.start_pos in seen:  # guard against double collection
            continue
        seen.add(tok.start_pos)
        container = _find_container(tree, tok.start_pos)
        _insert_ordered(container, _comment_node(tok), tok.start_pos)
    return tree
