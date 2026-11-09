"""Format a GDScript decorator annotation expression."""
from typing import Any, Dict, List, Tuple, Union

from lark import Tree, Token

from ..common.utils import get_line, get_end_line
from .types import FormattedLines, Outcome


def format_annotation(  # noqa: C901
    statement: Tree,
    indent_string: str,
    indent_level: int,
    style=None,
) -> Outcome:
    """Format an annotation/decorator like @export_enum(...) or @tool.

    New grammar structure (Earley):
        annotation: ANNOTATION (arguments)?

    Annotations are standalone - they appear on their own line before a statement.
    When followed by another declaration (var, func), the caller should handle
    prepending. For now we output them as single lines at the current indent level.
    """
    # Collect annotation name and arguments from children
    parts = []
    inline_comments = []  # Comments inside args - emit inline with last arg line
    args_list = None  # Individual argument strings, when the annotation has arguments
    has_parens = False  # `@ready()` must keep its (empty) parens

    for child in getattr(statement, "children", []):
        if not hasattr(child, "__class__"):  # Skip None placeholders
            continue
        if isinstance(child, Token):
            ttype = str(getattr(child, "type", "")).upper()
            if ttype == "ANNOTATION":
                parts.append(str(getattr(child, "value", "")))
            elif ttype in ("PARENTHESIS_OPEN", "PARENTHESIS_CLOSE"):
                has_parens = True  # We handle parens via arguments subtree
            else:
                parts.append(str(getattr(child, "value", str(child.type))))
        elif isinstance(child, Tree):
            d = _tree_data_str(child).lower()
            if d == "arguments":
                args_list, extra_comments = _format_annotation_args_with_inline(child)
                inline_comments.extend(extra_comments)
            elif d == "comment":
                # Comment attached inside the annotation (inline with content)
                text = _get_comment_text(child)
                if text:
                    inline_comments.append(text)

    from .comments import normalize_comment as _nc

    line_num = get_line(statement)
    name = "".join(parts)
    end_line = get_end_line_safe(statement)
    max_len = style.line_length if style else 100
    indent_width = len(indent_string.replace("\t", "    "))

    # Classify the annotation's comments by position: after `(` -> opening, on an
    # argument's line -> that argument, after `)` -> closing.
    open_line = open_col = close_line = close_col = None
    args_tree = None
    for child in getattr(statement, "children", []):
        if isinstance(child, Token):
            tt = str(getattr(child, "type", "")).upper()
            if tt == "PARENTHESIS_OPEN":
                open_line, open_col = getattr(child, "line", None), getattr(
                    child, "column", None
                )
            elif tt == "PARENTHESIS_CLOSE":
                close_line, close_col = getattr(child, "line", None), getattr(
                    child, "column", None
                )
        elif isinstance(child, Tree) and _tree_data_str(child).lower() == "arguments":
            args_tree = child

    arg_lines = []
    if args_tree is not None:
        for ac in getattr(args_tree, "children", []):
            if isinstance(ac, Tree) and _tree_data_str(ac).lower() != "comment":
                arg_lines.append(get_line_safe(ac))

    all_comments = []

    def _gather(container):
        for c in getattr(container, "children", []):
            tok = None
            if (
                isinstance(c, Token)
                and str(getattr(c, "type", "")).upper() == "COMMENT"
            ):
                tok = c
            elif hasattr(c, "data") and _tree_data_str(c).lower() == "comment":
                tok = next(
                    (t for t in getattr(c, "children", []) if isinstance(t, Token)),
                    None,
                )
            if tok is not None:
                text = _nc(str(getattr(tok, "value", "")))
                if text:
                    all_comments.append(
                        (
                            getattr(tok, "line", 0) or 0,
                            getattr(tok, "column", 0) or 0,
                            text,
                        )
                    )

    _gather(statement)
    if args_tree is not None:
        _gather(args_tree)
    all_comments.sort()

    open_comments, close_comments, stray = [], [], []
    # Keys are source line numbers, but lookups pass `arg_line` which can be
    # None; `Any` keeps both the sorted() and the .get() call sites honest.
    arg_comments: Dict[Any, List[str]] = {}
    for cln, ccol, ctxt in all_comments:
        if (
            close_line is not None
            and cln == close_line
            and close_col is not None
            and ccol > close_col
        ):
            close_comments.append(ctxt)
        elif (
            open_line is not None
            and cln == open_line
            and open_col is not None
            and ccol > open_col
        ):
            open_comments.append(ctxt)
        elif cln in arg_lines:
            arg_comments.setdefault(cln, []).append(ctxt)
        else:
            stray.append(ctxt)

    def _suffix(texts):
        return ("  " + "  ".join(texts)) if texts else ""

    if args_list is None:
        empty = "()" if has_parens else ""
        return (
            [
                (
                    line_num,
                    f"{indent_string}{name}{empty}{_suffix([t for _l, _c, t in all_comments])}",
                )
            ],
            end_line,
        )

    joined = ", ".join(args_list)
    if indent_width + len(f"{name}({joined})") <= max_len:
        return (
            [
                (
                    line_num,
                    f"{indent_string}{name}({joined}){_suffix([t for _l, _c, t in all_comments])}",
                )
            ],
            end_line,
        )

    # Too long: put the arguments on their own line(s). The arguments sit two levels in, or one
    # when two would not leave them under the limit - an annotation whose single argument is a
    # long string has only the one level to give, and without this the layout declares the
    # joined arguments too wide and gives every argument its own line, comment and all.
    from .expression_utils import continuation_indent_for

    inner_indent = continuation_indent_for(
        indent_string, len(joined), max_len, style.tab_size if style else 4
    )
    inner_width = len(inner_indent.replace("\t", "    "))
    lines: FormattedLines = [(line_num, f"{indent_string}{name}(")]
    if args_list and inner_width + len(joined) <= max_len:
        # The joined arguments still fit on one continuation line; the opening and
        # argument comments join it - unless the comment is what pushes the line over,
        # in which case it moves to a line of its own. The fit test has to include the
        # suffix: a merged comment on an already-long string argument is the whole
        # cause of the over-long line in `multiline_annotations_w_comments`.
        content_comments = open_comments[:]
        for ln in sorted(arg_comments):
            content_comments.extend(arg_comments[ln])
        suffix = _suffix(content_comments + stray)
        joined_with_suffix = inner_width + len(joined) + len(suffix)
        own_line_width = inner_width + len(suffix.strip())
        if suffix and joined_with_suffix > max_len >= own_line_width:
            lines.append((None, f"{inner_indent}{joined}"))
            lines.append((None, f"{inner_indent}{'  '.join(content_comments + stray)}"))
        else:
            lines.append((None, f"{inner_indent}{joined}{suffix}"))
    elif args_list:
        # One argument per line: each line is short, so this shape takes the full two levels
        # even when the joined form above had to settle for one.
        from .expression_utils import continuation_indent

        arg_indent = continuation_indent(indent_string)
        lines[-1] = (line_num, f"{indent_string}{name}({_suffix(open_comments)}")
        for i, arg in enumerate(args_list):
            comma = "," if i < len(args_list) - 1 else ""
            arg_line = arg_lines[i] if i < len(arg_lines) else None
            lines.append(
                (
                    None,
                    f"{arg_indent}{arg}{comma}{_suffix(arg_comments.get(arg_line, []))}",
                )
            )
        last = lines[-1]
        lines[-1] = (last[0], f"{last[1]}{_suffix(stray)}")
    elif _suffix(open_comments + stray):
        lines.append((None, f"{inner_indent}{(open_comments + stray)[0]}"))
    lines.append((None, f"{indent_string}){_suffix(close_comments)}"))
    return (lines, end_line)


def _format_annotation_args_with_inline(args_tree: Tree):
    """Format arguments and collect inline comments.

    Returns (formatted_args_string, list_of_inline_comment_texts).
    """
    # Extract actual expression children (skip commas and placeholders)
    expr_children: List[Union[Tuple[str, str], Tree]] = []
    for child in getattr(args_tree, "children", []):
        if not isinstance(child, Tree):
            continue
        d = _tree_data_str(child).lower()
        if d == "comment":
            # Inline comment inside args - collect it but don't include in main string
            text = _get_comment_text(child)
            expr_children.append(("__inline_comment__", text))
        else:
            expr_children.append(child)

    formatted_parts = []
    inline_comments = []
    for item in expr_children:
        if isinstance(item, tuple) and item[0] == "__inline_comment__":
            inline_comments.append(item[1])
        elif hasattr(item, "data") or hasattr(item, "value"):
            from .expression_to_str import expression_to_str as ets

            try:
                part = ets(item)
                formatted_parts.append(part)
            except Exception:
                formatted_parts.append("?")

    return (formatted_parts, inline_comments)


def _get_comment_text(comment_node: Tree) -> str:
    """Extract the text content from a comment tree node."""
    from .comments import normalize_comment

    for child in getattr(comment_node, "children", []):
        if hasattr(child, "value"):
            return normalize_comment(str(child.value))
    return ""


def _tree_data_str(node) -> str:
    """Return a tree's rule name as a plain string."""
    if isinstance(node, Tree):
        # The lark stubs type `Tree.data` as plain str, but at runtime it can
        # be a Token-like object with a `.value`; the hasattr guard picks the
        # right branch, which mypy cannot see.
        d = node.data
        return str(d) if not hasattr(d, "value") else d.value  # type: ignore[attr-defined]
    return ""


def get_end_line_safe(node) -> int:
    """Get the end line of a node, with safety fallback."""
    try:
        if hasattr(node, "meta") and getattr(node.meta, "end_line", None):
            return node.meta.end_line
        return get_end_line(node)
    except Exception:
        try:
            from ..common.utils import get_end_line as _gel

            return _gel(node)
        except Exception:
            return get_line_safe(node)


def get_line_safe(node) -> int:
    """Get the line of a node, with safety fallback."""
    try:
        if hasattr(node, "meta") and getattr(node.meta, "line", None):
            return node.meta.line
        return get_line(node)
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# Placement rules (mirrors the legacy formatter's annotation.py)
#
# An annotation is either "prependable" - emitted on the same line as the
# declaration it decorates, when that fits - or "standalone", always on its own
# line. `@tool` and `@warning_ignore_start` are the notable standalone cases.
# ---------------------------------------------------------------------------

STANDALONE_ANNOTATIONS = frozenset(
    {
        "abstract",
        "export_category",
        "export_group",
        "export_subgroup",
        "icon",
        "tool",
        "warning_ignore_start",
        "warning_ignore_restore",
    }
)

# `@warning_ignore("...")` is prependable only for these warnings. The list is
# the warnings that apply to the annotated declaration itself; `unused_variable`
# and the rest are excluded on purpose, so they stay standalone.
NON_STANDALONE_WARNING_IGNORES = frozenset(
    {
        "unassigned_variable",
        "unassigned_variable_op_assign",
        "unused_parameter",
        "shadowed_global_identifier",
        "shadowed_variable",
        "shadowed_variable_base_class",
        "unreachable_code",
        "unreachable_pattern",
        "standalone_expression",
        "standalone_ternary",
        "incompatible_ternary",
        "untyped_declaration",
        "inferred_declaration",
        "unsafe_property_access",
        "unsafe_method_access",
        "unsafe_cast",
        "unsafe_call_argument",
        "unsafe_void_return",
        "return_value_discarded",
        "static_called_on_instance",
        "redundant_await",
        "assert_always_true",
        "assert_always_false",
        "integer_division",
        "narrowing_conversion",
        "int_as_enum_without_cast",
        "int_as_enum_without_match",
        "enum_variable_without_default",
        "deprecated_keyword",
        "confusable_identifier",
        "confusable_local_declaration",
        "confusable_local_usage",
        "confusable_capture_reassignment",
        "inference_on_variant",
        "native_method_override",
    }
)

# `@abstract` prepends only to one of these; otherwise it stays standalone.
_ABSTRACT_TARGETS = frozenset(
    {
        "function_decl",
        "class_name_stmt",
        "inner_class",
        "abstract_func_decl",
    }
)


def annotation_name(node) -> str:
    """The annotation's bare name, without the leading `@`."""
    for child in getattr(node, "children", []):
        if (
            isinstance(child, Token)
            and str(getattr(child, "type", "")).upper() == "ANNOTATION"
        ):
            return str(getattr(child, "value", "")).lstrip("@")
    return ""


def _iter_tokens(node):
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, Token):
            yield n
        elif hasattr(n, "children"):
            for c in n.children:
                if c is not None:
                    stack.append(c)


def _annotation_string_arg(node) -> str:
    """First string literal inside the annotation's arguments, quotes stripped."""
    for token in _iter_tokens(node):
        if str(getattr(token, "type", "")).upper() == "STRING":
            return str(getattr(token, "value", "")).strip("\"'")
    return ""


def annotation_to_flat_str(node) -> str:
    """Render an annotation as one line, e.g. `@export_range(1, 100)`."""
    name = annotation_name(node)
    has_parens = any(
        isinstance(c, Token)
        and str(getattr(c, "type", "")).upper() == "PARENTHESIS_OPEN"
        for c in getattr(node, "children", [])
    )
    args_tree = None
    for child in getattr(node, "children", []):
        if isinstance(child, Tree) and _tree_data_str(child).lower() == "arguments":
            args_tree = child
            break
    if args_tree is None:
        return f"@{name}()" if has_parens else f"@{name}"
    args, _comments = _format_annotation_args_with_inline(args_tree)
    return f"@{name}({', '.join(args)})"


def is_prependable_annotation(
    node, next_target_type: str, in_function_body: bool
) -> bool:
    """Whether `node` belongs on the same line as the declaration it decorates.

    `next_target_type` is the statement type of the declaration after the
    annotation run. Inside a function body `@warning_ignore` is always
    standalone, because there it guards the following statement, not a
    declaration.
    """
    name = annotation_name(node)
    if name == "abstract":
        return next_target_type in _ABSTRACT_TARGETS
    if name in STANDALONE_ANNOTATIONS:
        return False
    if name != "warning_ignore":
        return True
    if in_function_body:
        return False
    return _annotation_string_arg(node) in NON_STANDALONE_WARNING_IGNORES
