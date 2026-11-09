"""Top-level formatter module for gdformat2.

Provides `format_code()` as the main entry point, taking an already-parsed
tree and StyleConfig to produce formatted GDScript code.
"""
from typing import List

from lark import Tree, Token

from .style_config import StyleConfig
from .block import format_block
from .class_statement import format_class_statement


def format_code(
    tree: Tree,
    style: StyleConfig,
) -> str:
    """Format a parsed GDScript AST into formatted code.

    Args:
        tree: Already-parsed Lark Tree (with embedded comment nodes).
        style: Formatting configuration (indent rules, line length, etc.).

    Returns:
        Formatted GDScript source code as a string.
    """
    tab_size = 4 if not style.use_spaces else style.tab_size
    indent_string = (
        "\t" * ((tab_size) // max(tab_size, 1))
        if not style.use_spaces
        else " " * (tab_size or 4)
    )

    # Extract top-level items from root start node
    top_items = _extract_top_level(tree)

    # Root-level items have no indentation (column 0)
    formatted_lines, _last_line = format_block(
        top_items, format_class_statement, "", 0, style
    )

    if formatted_lines and formatted_lines[-1][1] != "":
        formatted_lines.append((None, ""))

    formatted_lines = _limit_preserving_continuations(
        formatted_lines, style, indent_string
    )

    return "\n".join(line for _, line in formatted_lines)


def _limit_preserving_continuations(lines, style, indent_string):
    """Drop a `(` block to one indent level when two levels put a line over the limit.

    The guide asks for two indent levels on a continuation line, and it also asks to keep every
    line under 100 characters. Where the two conflict the limit wins, because a long line is the
    one a reader actually suffers from: a block whose content would overflow at two levels is
    written at one, and `continuation_indent` cannot know that when it is called. Arrays,
    dictionaries, enums and subscripts are at one level already, so only a `(` block is ever
    relaxed, and a line deeper inside a nested collection is left to that collection's layout.
    """
    if style is None or not lines:
        return lines
    limit = style.line_length
    result = list(lines)
    for _ in range(8):
        relaxed = _relax_one_block(result, limit, indent_string)
        if not relaxed:
            return result
    return result


def _relax_one_block(lines, limit, indent_string):
    """Dedent one over-long `(` block by a level. True when a block was dedented."""
    unit = indent_string or "\t"
    unit_width = len(unit.replace("\t", "    "))
    for start, end, opener_indent in _paren_blocks(lines):
        widths = [len(text.replace("\t", "    ")) for _n, text in lines[start:end]]
        if not widths or max(widths) <= limit:
            continue
        # Only a block that still has a level to give, and only when giving it helps. A block
        # whose widest line stays over the limit even a level shallower is left at two levels:
        # the code then keeps the indent the guide asks for, and the long line is the
        # `unbreakable-line` check's business - a 96-column parameter name cannot be fixed by
        # indenting it, and dedenting for nothing makes the block harder to read.
        if not any(
            text.startswith(opener_indent + unit + unit)
            for _n, text in lines[start:end]
        ):
            continue
        if max(widths) - unit_width > limit:
            continue
        for index in range(start, end):
            line_no, text = lines[index]
            if text.startswith(unit):
                lines[index] = (line_no, text[len(unit) :])
        return True
    return False


def _paren_blocks(lines):
    """(content start index, content end index, opener indent) for every `(` block."""
    blocks = []
    stack = []
    for index, (_line_no, text) in enumerate(lines):
        stripped = text.lstrip("\t ")
        indent = text[: len(text) - len(stripped)]
        for character in stripped:
            if character == "#":
                break
            if character in "([{":
                stack.append((character, indent, index))
            elif character in ")]}" and stack:
                character_open, opener_indent, opener_index = stack.pop()
                if character_open == "(":
                    blocks.append((opener_index + 1, index, opener_indent))
    return blocks


def _block_is_over_long(lines, start, end, limit):
    return any(
        len(text.replace("\t", "    ")) > limit for _line_no, text in lines[start:end]
    )


def _extract_top_level(tree: Tree):  # noqa: C901
    """Extract top-level file items from the parse tree.

    The grammar produces: start -> (file_item | _NL)* where file_item can be:
      - extends_stmt, class_name_stmt, annotation (direct declarations)
      - class_member (wrapped in class_member node for classes/functions/variables/etc.)
    Comments may appear directly as Tree(comment) nodes.

    Annotations are collected and prepended to the following real declaration,
    matching GDScript decorator semantics. Inline comments inside annotations
    are attached to the annotation's arguments subtree; trailing comments
    after the closing paren are paired with their statement.

    We flatten to a list of: comment nodes, annotations, and actual declarations.
    """
    items = []  # type: List
    pending_annotations = []  # Collect consecutive annotations before a real statement

    # Move a top-level enum's / annotation's same-line comments out of the file scope so
    # their formatter can attach them positionally.
    from gdtoolkit.gdformat2.class_statement import _absorb_member_comments as _absorb

    _root_children = getattr(tree, "children", [])
    for _c in list(_root_children):
        if hasattr(_c, "data") and _tree_data_str(_c).lower() in (
            "class_member",
            "annotation",
        ):
            _absorb(_c, _root_children)

    # Build annotation span map for determining which comments belong inside
    ann_spans = {}
    for child in getattr(tree, "children", []):
        if isinstance(child, Tree) and _tree_data_str(child).lower() == "annotation":
            meta = getattr(child, "meta", None)
            if meta:
                start_pos = getattr(meta, "start_pos", -1)
                end_pos = getattr(meta, "end_pos", -1)
                ann_spans[start_pos] = (child, end_pos)

    for child in getattr(tree, "children", []):
        if not hasattr(child, "data"):
            continue
        data_str = _tree_data_str(child).lower()

        if "comment" in data_str and isinstance(child, Tree):
            # Check if this comment falls inside any annotation's span
            c_start = getattr(getattr(child, "meta", None), "start_pos", -1)

            inside_any_ann = False
            for ann_start, (ann_node, ann_end) in ann_spans.items():
                if ann_start <= c_start < ann_end:
                    # Comment belongs inside an annotation - attach it to the annotation's args
                    _attach_comment_to_annotation(ann_node, child)
                    inside_any_ann = True
                    break

            if not inside_any_ann:
                # Flush any pending annotations first so a same-line comment can
                # attach to the annotation it trails.
                if pending_annotations:
                    for ann in pending_annotations:
                        items.append(("annotation", ann))
                    pending_annotations = []
                # A comment on the same line as the previous declaration is that
                # declaration's inline comment, not a standalone one.
                if _attach_inline_to_previous(items, child):
                    continue
                items.append(("__root_comment__", child))
            continue

        elif data_str == "annotation":
            # Collect consecutive annotations - they'll be prepended to the next real statement
            pending_annotations.append(child)
            continue

        # Direct file-level items (extends, class_name) - flush any pending annotations first
        elif data_str in ("extends_stmt", "class_name_stmt"):
            if pending_annotations:
                for ann in pending_annotations:
                    items.append(("annotation", ann))
                pending_annotations = []
            items.append(("member", child))

        elif "class_member" in data_str:
            # Flush pending annotations before class members
            if pending_annotations:
                for ann in pending_annotations:
                    items.append(("annotation", ann))
                pending_annotations = []

            # class_member wraps the actual declarations - unwrap them
            from gdtoolkit.gdformat2.class_statement import (
                _find_suite_or_empty as _fsoe,
                _tree_data_str as _tds,
            )

            for sub_child in getattr(child, "children", []):
                if hasattr(sub_child, "data"):
                    sd = _tree_data_str(sub_child).lower()
                    # Skip end containers and NL markers - they're not declarations
                    if sd != "end" and not sd.startswith("_nl"):
                        items.append(("member", sub_child))
                        # A comment left in a function's suite that is indented shallower
                        # than the body belongs to the file scope (e.g. a comment between
                        # two top-level functions). Promote it to a top-level comment item
                        # right after the function, otherwise it would be dropped.
                        if sd == "function_decl":
                            _, orphans = _fsoe(sub_child)
                            for oc in orphans:
                                items.append(("__root_comment__", oc))

    # Flush any remaining pending annotations (e.g., annotation at end of file)
    if pending_annotations:
        for ann in pending_annotations:
            items.append(("annotation", ann))

    return items


def _attach_inline_to_previous(items, comment_node) -> bool:
    """If `comment_node` sits on the same line as the previous member or annotation
    item, turn that item into an inline-comment 3-tuple and return True."""
    from .comments import normalize_comment

    tok = next(
        (t for t in getattr(comment_node, "children", []) if isinstance(t, Token)), None
    )
    c_line = (
        getattr(tok, "line", 0)
        if tok is not None
        else getattr(getattr(comment_node, "meta", None), "line", 0)
    )
    if not c_line or not items:
        return False
    prev = items[-1]
    if not (
        isinstance(prev, tuple)
        and prev[0] in ("member", "annotation")
        and len(prev) == 2
    ):
        return False
    prev_node = prev[1]
    try:
        prev_end = _last_code_line(prev_node)
    except Exception:
        return False
    if prev_end != c_line:
        return False
    text = normalize_comment(str(getattr(tok, "value", ""))) if tok is not None else ""
    if not text:
        return False
    items[-1] = (prev[0], prev_node, text)
    return True


def _last_code_line(node) -> int:
    """The line the node's last real token sits on.

    A statement node's span reaches the newline that ends it, so `get_end_line` can report the
    line *after* the statement - which is what stopped `extends Node2D  # World` from keeping its
    trailing comment: the comment is on line 1 and the node claimed to end on line 2, so the
    inline check missed and the comment was emitted standalone.
    """
    from ..common.utils import get_end_line

    ignore = ("_NL", "_INDENT", "_DEDENT", "END", "END_OF_FILE")
    lines = [
        getattr(token, "line", 0)
        for token in node.scan_values(lambda child: isinstance(child, Token))
        if str(getattr(token, "type", "")).upper() not in ignore
    ]
    return max(lines) if lines else get_end_line(node)


def _attach_comment_to_annotation(annotation_node: Tree, comment_node: Tree) -> None:
    """Attach a sibling comment to the annotation node's children.

    Comments that appear inside an annotation's character span (between its
    start and end positions) are re-inserted as children of the annotation.
    This preserves them during formatting so they can be emitted inline.
    """
    # Find arguments subtree to attach comment before closing paren
    args_tree = None
    for _, c in enumerate(getattr(annotation_node, "children", [])):
        if isinstance(c, Tree) and _tree_data_str(c).lower() == "arguments":
            args_tree = c
            break

    # Insert comment after the last argument but before any closing paren token
    if args_tree is not None:
        # Append to arguments as sibling of elements - this keeps it with content
        annotation_node.children.append(comment_node)
    else:
        # No arguments found, append directly to annotation
        annotation_node.children.append(comment_node)


def _tree_data_str(node) -> str:
    if isinstance(node, Tree):
        d = node.data
        return str(d) if not hasattr(d, "value") else d.value  # type: ignore
    return ""
