from functools import partial
from typing import Dict, Callable, Any, List, Optional

from lark import Tree, Token

from ..common.utils import get_line, get_end_line
from .context import ExpressionContext
from .style_config import StyleConfig
from .types import FormattedLines, Outcome
from .block import format_block
from .expression import format_expression
from .var_statement import format_var_statement, _safe_str  # noqa: F401
from .const_statement import format_const_statement
from .statement_utils import format_simple_statement
from .comments import normalize_comment


def _format_annotation(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format an annotation/decorator like @export_enum(...) or @tool."""
    from .annotation import format_annotation as _fmt

    return _fmt(statement, indent_string, indent_level, style)


_HANDLERS: Dict[str, Callable] = {}


def format_class_statement(
    statement: Any, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format a top-level item (comment or class member)."""

    if isinstance(statement, tuple):
        # Handle special wrapper tuples from formatter.py _extract_top_level
        tag, real_node = statement  # type: ignore
        if tag == "__root_comment__":
            return _format_root_comment(real_node, indent_string, indent_level, style)
        elif tag == "__comment__":
            # Standalone comment at block level - format it normally
            text = _get_comment_text(real_node)  # type: ignore
            return (
                [(get_line(real_node), f"{indent_string}{text}")],
                get_end_line_safe(real_node),
            )
        elif tag == "annotation":
            stmt_data = _tree_data_str(real_node).lower()
            handler = _HANDLERS.get(stmt_data, _format_unknown_class_member)
            return handler(real_node, indent_string, indent_level, style)
        elif tag == "member":
            stmt_data = _tree_data_str(real_node).lower()
            handler = _HANDLERS.get(stmt_data, _format_unknown_class_member)
            result = handler(real_node, indent_string, indent_level, style)
            return result

    # Direct node - check if this IS a comment tree (standalone comment at block level)
    stmt_data = _tree_data_str(statement).lower()

    # Only handle as standalone comment if the statement itself is a comment type
    if stmt_data == "comment":
        comment_text = _get_comment_text(statement)  # type: ignore
        return (
            [(get_line(statement), f"{indent_string}{comment_text}")],
            get_end_line_safe(statement),
        )

    handler = _HANDLERS.get(stmt_data, lambda s, i, l, st: ([], 0))  # type: ignore
    result = handler(statement, indent_string, indent_level, style)

    return result


# ─── Handler Functions ──────────────────────────────────────────────────────


def _format_root_comment(
    comment_node: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format a root-level (top-level file) comment.

    Root comments should NOT be indented - they go at column 0.
    """
    text = _get_comment_text(comment_node)
    end_line = get_end_line_safe(comment_node) if hasattr(comment_node, "meta") else 1
    # Root-level comments appear at column 0 (no indentation).
    return ([], end_line)


def _format_member_decl(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format an unwrapped class member."""
    stmt_data = _tree_data_str(statement).lower()
    handler = _HANDLERS.get(stmt_data, _format_unknown_class_member)
    return handler(statement, indent_string, indent_level, style)


def _format_variable_decl(
    statement: Tree,
    indent_string: str,
    indent_level: int,
    style=None,
) -> Outcome:
    """Format a class-level variable declaration (may be static)."""
    # Check if first child is STATIC token
    is_static = False
    for child in getattr(statement, "children", []):
        if isinstance(child, Token) and str(_tree_data_str(child)) == "STATIC":  # type: ignore
            is_static = True
            break

    prefix = "static " if is_static else ""
    return format_var_statement(statement, indent_string, indent_level, style, prefix)


def _format_signal_decl(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format a signal declaration."""
    from .expression_to_str import expression_to_str as _ets

    name = _find_name_token(statement) or "unknown"

    # Signal with no parameters (only NAME and maybe end token)
    params_count = sum(1 for c in statement.children if isinstance(c, Tree))  # type: ignore
    if params_count == 0:
        return format_simple_statement(
            f"signal {name}", statement, indent_string, indent_level
        )

    sig_params = _parameters_with_stripped_defaults(
        [c for c in getattr(statement, "children", []) if hasattr(c, "data")][-1],
        f"{indent_string}signal {name}(",
        indent_string,
        style,
    )
    params_str = _ets(sig_params)
    line_no = get_line_safe(statement)
    full_line = f"{indent_string}signal {name}{params_str}"
    # The parameter list breaks by the same rule as a function header.
    param_layout, has_trailing_comma = _parameter_parts_with_values(sig_params)
    has_param_comments = any(
        _tree_data_str(c) == "comment"
        for c in getattr(sig_params, "children", [])
        if isinstance(c, Tree)
    )
    if (param_layout or has_param_comments) and (
        _display_width(full_line) > (style.line_length if style else 100)
        or has_trailing_comma
        or has_param_comments
    ):
        return _break_param_list(
            sig_params,
            f"{indent_string}signal {name}(",
            line_no,
            ")",
            indent_string,
            style,
        )
    return [(line_no, full_line)]


def _display_width(text: str) -> int:
    """Display width of a line, counting a tab as four columns."""
    return len(text.replace("\t", "    "))


def _absorb_member_comments(member, siblings) -> None:
    """Move a declaration's own comments from its siblings into the declaration node.

    Two shapes:
    - an `enum_decl` inside a `class_member`: a same-line comment after `{`, after an
      entry, or after `}` is left as a `comment` sibling; move those within the enum's
      line span into the enum;
    - an `annotation`: a comment after the closing `)` (which lies outside the
      annotation's span, so it is not attached by the annotation handling) is left as a
      sibling; move those on the annotation's last line past the `)` into the annotation.

    Either way the comment is then attached positionally rather than emitted standalone."""
    d_member = _tree_data_str(member).lower() if hasattr(member, "data") else ""
    if d_member in ("enum_decl", "annotation"):
        decl = member
    else:
        decl = None
        for c in getattr(member, "children", []) or []:
            if isinstance(c, Tree) and _tree_data_str(c).lower() == "enum_decl":
                decl = c
                break
    if decl is None:
        return
    end = get_end_line_safe(decl)
    if _tree_data_str(decl).lower() == "enum_decl":
        start = get_line_safe(decl)

        def _matches(ln, _col):
            return bool(ln) and start <= ln <= end

    else:
        close_col = None
        for c in getattr(decl, "children", []):
            if (
                isinstance(c, Token)
                and str(getattr(c, "type", "")).upper() == "PARENTHESIS_CLOSE"
            ):
                close_col = getattr(c, "column", None)

        def _matches(ln, col):
            return (
                bool(ln)
                and ln == end
                and close_col is not None
                and col is not None
                and col > close_col
            )

    absorbed = []
    for c in list(siblings):
        if c is member:
            continue
        ln = col = None
        if isinstance(c, Token) and str(getattr(c, "type", "")).upper() == "COMMENT":
            ln, col = getattr(c, "line", 0), getattr(c, "column", 0)
        elif hasattr(c, "data") and _tree_data_str(c).lower() == "comment":
            tok = next(
                (t for t in getattr(c, "children", []) if isinstance(t, Token)), None
            )
            if tok is not None:
                ln, col = getattr(tok, "line", 0), getattr(tok, "column", 0)
            else:
                ln = getattr(getattr(c, "meta", None), "line", 0)
        if _matches(ln, col):
            absorbed.append(c)
    if not absorbed:
        return
    decl.children.extend(absorbed)
    siblings[:] = [c for c in siblings if not any(c is a for a in absorbed)]


def _format_enum_decl(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format an enum declaration.

    New grammar structure (flat):
        enum_decl [ENUM_token, NAME?, BRACE_OPEN/COLON, body_tree?, BRACE_CLOSE?]

    For unnamed enums like `enum { A, B }`: no NAME token before braces/colon
    For named enums like `enum Foo { A, B }`: NAME token at position 1

    Multiline format used when:
    - More than 3 entries
    - Any entry has a trailing comma (grammar-based detection)
    """
    children_list = list(getattr(statement, "children", []))

    # Extract enum name (direct NAME child of enum_decl, not nested) and body tree
    name_token = None
    body_tree = None
    has_trailing_comma = False

    for child in children_list:
        if child is None:
            continue
        if isinstance(child, Token):
            ttype = str(child.type).lower()
            # Only consider direct NAME token as enum name (not entry names inside enum_entries)
            # Name comes between ENUM and BRACE_OPEN for named enums like `enum X {`
            if ttype == "name" and name_token is None:
                name_token = child.value
        elif isinstance(child, Tree):
            d = str(
                getattr(
                    getattr(child, "data", None), "value", getattr(child, "data", "")
                )
            )
            if d in ("enum_entries", "enum_body"):
                body_tree = child
                # Check for trailing comma by looking at COMMA tokens after last entry
                all_children = list(getattr(child, "children", []))
                if len(all_children) >= 2:
                    last_entry_idx = -1
                    for idx in range(len(all_children) - 1, -1, -1):
                        ec = all_children[idx]
                        if (
                            isinstance(ec, Tree)
                            and _tree_data_str(ec).lower() == "enum_entry"
                        ):
                            last_entry_idx = idx
                            break
                    # Check if there's a COMMA after the last entry
                    for idx in range(last_entry_idx + 1, len(all_children)):
                        ec = all_children[idx]
                        if (
                            isinstance(ec, Token)
                            and str(getattr(ec, "type", "")).lower() == "comma"
                        ):
                            has_trailing_comma = True
                            break

    # Build output string directly
    parts = ["enum"]
    if name_token:
        parts.append(name_token)

    # Check for brace tokens to handle empty enums
    has_braces = False
    for child in children_list:
        if isinstance(child, Token):
            ttype = str(child.type).lower()
            if ttype in ("brace_open", "brace_close"):
                has_braces = True
                break

    if body_tree is not None or has_braces:
        # Extract enum entries with their optional values from the entries tree
        entry_parts = []
        entry_lines = []
        for echild in getattr(body_tree, "children", []) if body_tree else []:
            d = (
                str(
                    getattr(
                        getattr(echild, "data", None),
                        "value",
                        getattr(echild, "data", ""),
                    )
                )
                if hasattr(echild, "data")
                else ""
            )
            if d == "enum_entry":
                # enum_entry: NAME [EQUAL expression]
                name_val = None
                expr_tree = None
                for eitem in getattr(echild, "children", []):
                    if (
                        isinstance(eitem, Token)
                        and str(getattr(eitem, "type", "")).lower() == "name"
                    ):
                        name_val = str(getattr(eitem, "value", ""))
                    elif isinstance(eitem, Tree):
                        # This is the expression after EQUAL (could be assignment/cast/number/etc.)
                        expr_tree = eitem
                        break

                if name_val:
                    entry_str = name_val
                    if expr_tree is not None:
                        # Use the plain serializer: format_expression returns lines that
                        # already carry the indent (which must not appear after `=`) and
                        # _safe_str only yields the first token of a compound value.
                        try:
                            from .expression_to_str import (
                                standalone_expression_to_str as _sestr,
                            )

                            expr_str = _sestr(expr_tree)
                        except Exception:
                            expr_str = ""
                        if expr_str:
                            entry_str += f" = {expr_str}"
                    entry_parts.append(entry_str)
                    entry_lines.append(get_line_safe(echild))

        entry_parts = [ep for ep in entry_parts if ep.strip()]
        max_len = style.line_length if style else 100
        start_line = get_line_safe(statement)
        end_line = get_end_line_safe(statement)

        # Comments attached to the enum: those moved in from the enclosing class_member
        # (direct children) plus any inside the entries tree. Sort them by position and
        # classify: on the `enum` line -> open, on the closing `}` line -> close, on an
        # entry's line -> that entry, anything else -> trailing.
        raw_comments = []

        def _collect_comments(node):
            for c in getattr(node, "children", []):
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
                if tok is None:
                    continue
                text = normalize_comment(str(getattr(tok, "value", "")))
                if text:
                    raw_comments.append(
                        (
                            getattr(tok, "line", 0) or 0,
                            getattr(tok, "column", 0) or 0,
                            text,
                        )
                    )

        _collect_comments(statement)
        if body_tree is not None:
            _collect_comments(body_tree)
        raw_comments.sort()

        # Brace positions disambiguate a comment that sits on the same line as both braces:
        # after `{` it is the opening comment, after `}` the closing one.
        open_line = open_col = close_line = close_col = None
        for c in children_list:
            if isinstance(c, Token):
                t = str(getattr(c, "type", "")).upper()
                if t == "BRACE_OPEN":
                    open_line, open_col = getattr(c, "line", None), getattr(
                        c, "column", None
                    )
                elif t == "BRACE_CLOSE":
                    close_line, close_col = getattr(c, "line", None), getattr(
                        c, "column", None
                    )

        open_comments, entry_comments, close_comments = [], {}, []
        for cline, ccol, ctext in raw_comments:
            if (
                close_line is not None
                and cline == close_line
                and close_col is not None
                and ccol > close_col
            ):
                close_comments.append(ctext)
            elif (
                open_line is not None
                and cline == open_line
                and open_col is not None
                and ccol > open_col
            ):
                open_comments.append(ctext)
            elif cline in entry_lines:
                entry_comments.setdefault(cline, []).append(ctext)
            else:
                close_comments.append(ctext)

        def _suffix(texts):
            return ("  " + "  ".join(texts)) if texts else ""

        if not entry_parts:
            # Empty enum - `{}`, split onto two lines when the header is too long.
            all_texts = open_comments + close_comments
            single_line = f"{indent_string}{' '.join(parts)} {{}}{_suffix(all_texts)}"
            if _display_width(single_line) <= max_len:
                return ([(get_line(statement), single_line)], end_line)
            return (
                [
                    (get_line(statement), f"{indent_string}{' '.join(parts)} {{"),
                    (None, f"{indent_string}}}{_suffix(all_texts)}"),
                ],
                end_line,
            )
        head = " ".join(parts)

        # The guide: "Write enums with each item on its own line", whose own Bad example is a
        # single-line enum, so an enum with entries is never written on one line - only an
        # empty one is - and never with several entries sharing an inner line.
        inner_indent = indent_string + "\t"
        lines_out = [
            (get_line(statement), f"{indent_string}{head} {{" + _suffix(open_comments))
        ]
        for idx, ep in enumerate(entry_parts):
            eline = entry_lines[idx] if idx < len(entry_lines) else None
            suffix = (
                _suffix(entry_comments.get(eline, []))
                if eline in entry_comments
                else ""
            )
            lines_out.append((None, f"{inner_indent}{ep},{suffix}"))
        lines_out.append((None, f"{indent_string}}}" + _suffix(close_comments)))
        return (lines_out, end_line)

    line_str = f"{indent_string}{' '.join(parts)}"
    return ([(get_line(statement), line_str)], get_end_line_safe(statement))


def _format_func_decl(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format a function declaration (top-level or nested class)."""
    # Check for static modifier - first child is STATIC token in new grammar
    is_static = False
    children_list = list(getattr(statement, "children", []))  # type: ignore
    if children_list:
        first_child = children_list[0]
        if isinstance(first_child, Token):
            v = str(_tree_data_str(first_child)).lower()  # type: ignore
            is_static = v == "static"

    func_name = _find_name_token(statement) or "?"

    # In new grammar: FUNC NAME PARENTHESIS_OPEN ...PARENTHESIS_CLOSE COLON suite
    # Find parameters between LPAR and RPAR tokens (not wrapped in a tree)
    params_between_parens = []
    inside_parens = False
    for child in children_list:
        if isinstance(child, Token):
            t_str = str(_tree_data_str(child)).lower()  # type: ignore
            if t_str == "parenthesis_open" or (isinstance(child, Token) and child.type == "PARENTHESIS_OPEN"):  # type: ignore
                inside_parens = True
            elif t_str == "colon":
                break
        else:
            if hasattr(child, "data"):
                data_str = str(_tree_data_str(child)).lower()  # type: ignore
                if inside_parens and not ("suite" in data_str or "comment" in data_str):
                    params_between_parens.append(child)

    prefix = "static func " if is_static else "func "
    suffix_part = ":"

    # Check for return type (FORWARD_ARROW or ->) - token AFTER colon
    for child in children_list:
        if isinstance(child, Token):
            # Use child.type directly since _tree_data_str returns "" for Tokens
            if str(child.type).lower() == "forward_arrow":
                idx = children_list.index(child)
                if idx + 1 < len(children_list):
                    ret_child = children_list[idx + 1]
                    suffix_part = f" -> {_safe_str(ret_child)}:"
                    break

    # A body-less function (`@abstract func f()`), parsed by the optional suite,
    # gets no trailing colon.
    has_body = any(
        isinstance(c, Tree) and "suite" in _tree_data_str(c).lower()
        for c in children_list
    )
    if not has_body:
        suffix_part = suffix_part.rstrip(":")

    # Find parameters node from the original parsed tree (not artificially created)
    param_tree = None
    for child in children_list:
        if (
            isinstance(child, Tree)
            and str(_tree_data_str(child)).lower() == "parameters"
        ):
            param_tree = child
            break
    if param_tree is not None:
        param_tree = _parameters_with_stripped_defaults(
            param_tree, f"{indent_string}{prefix}{func_name}(", indent_string, style
        )

    expression_context = ExpressionContext(
        f"{prefix}{func_name}",
        get_line_safe(statement),
        suffix_part,
        get_end_line_safe(statement),
    )

    # Format parameters using expression_to_str (handles the "parameters" data_map handler)
    from .expression_to_str import expression_to_str as _ets

    if param_tree is not None:
        params_str = _ets(param_tree)
        header_lines = [
            (
                get_line_safe(statement),
                f"{indent_string}{prefix}{func_name}{params_str}{suffix_part}",
            )
        ]
    else:
        # No parameters - emit with empty parens like "func name():"
        # Use suffix_part which includes return type if present
        header_lines = [
            (
                get_line_safe(statement),
                f"{indent_string}{prefix}{func_name}(){suffix_part}",
            )
        ]

    # Check for inline comment in the suite that should appear after the header
    func_line_no = get_line_safe(statement)
    inline_func_comment = None
    for child in children_list:
        sd = _tree_data_str(child).lower() if hasattr(child, "data") else ""
        if "suite" in str(sd):
            # Look at suite children to find an inline comment before first statement
            suite_children = list(getattr(child, "children", []))
            for sub_child in suite_children:
                s = (
                    _tree_data_str(sub_child).lower()
                    if hasattr(sub_child, "data")
                    else ""
                )
                if "comment" in str(s):
                    col = (
                        getattr(sub_child.meta, "column", -1)
                        if hasattr(sub_child, "meta")
                        else -1
                    )
                    if hasattr(sub_child, "meta") and sub_child.meta.line == int(func_line_no) and col > len(indent_string + prefix + func_name):  # type: ignore
                        for cc in getattr(sub_child, "children", []):
                            if (
                                isinstance(cc, Token)
                                and str(getattr(cc, "type", "")).upper() == "COMMENT"
                            ):
                                inline_func_comment = normalize_comment(str(cc.value))
                                break
                    break  # Only check first comment

    header_line_text = ""
    if param_tree is not None:
        params_str = _ets(param_tree)
        header_line_text = (
            f"{indent_string}{prefix}{func_name}{params_str}{suffix_part}"
        )
        param_layout, has_trailing_comma = _parameter_parts_with_values(param_tree)
        has_param_comments = any(
            _tree_data_str(c) == "comment"
            for c in getattr(param_tree, "children", [])
            if isinstance(c, Tree)
        )
        # Break the header when the flat form overflows (display columns: a tab counts
        # as four) or the author wrote a trailing comma (the same rule the array/dict
        # layout uses), or when a comment sits inside the parameter list, which cannot be
        # rendered on one line.
        if (param_layout or has_param_comments) and (
            _display_width(header_line_text) > (style.line_length if style else 100)
            or has_trailing_comma
            or has_param_comments
        ):
            open_line = f"{indent_string}{prefix}{func_name}("
            if inline_func_comment:
                open_line += f"  {inline_func_comment}"
            header_lines = _break_param_list(
                param_tree,
                open_line,
                get_line_safe(statement),
                f"){suffix_part}",
                indent_string,
                style,
            )
        else:
            if inline_func_comment:
                header_line_text += f"  {inline_func_comment}"
            header_lines = [(get_line_safe(statement), header_line_text)]
    else:
        # No parameters - emit with empty parens like "func name():"
        header_line_text = f"{indent_string}{prefix}{func_name}(){suffix_part}"
        if inline_func_comment:
            header_line_text += f"  {inline_func_comment}"
        header_lines = [(get_line_safe(statement), header_line_text)]

    # Find and format function body (suite content).
    # For top-level functions (indent_level == 0), extract trailing suite comments
    # so they appear at the end of the function's formatted output.
    # For nested functions inside classes, orphans are NOT extracted here -
    # _format_inner_class extracts them before calling format_block which calls this.
    if indent_level == 0:
        # Top-level: extract and append orphaned comments directly after function body
        suite_result = (
            _find_suite_or_empty(statement)
            if not inline_func_comment
            else _find_suite_or_empty(
                statement, skip_inline_comment_line_no=int(func_line_no)
            )
        )
        suite_content, orphaned_comments_list = (
            suite_result if suite_result else ([], [])
        )
    else:
        # Nested: still need suite content but don't extract orphans (parent handles them)
        # The parent (_format_inner_class) has already extracted orphans from this node,
        # so calling _find_suite_or_empty again will just return the non-orphan items.
        # An inline comment on the header sits inside the suite on the header's line, so it
        # must be skipped the same way as at top level; skipping the whole lookup instead
        # dropped the function's body.
        suite_result = (
            _find_suite_or_empty(
                statement, skip_inline_comment_line_no=int(func_line_no)
            )
            if inline_func_comment
            else _find_suite_or_empty(statement)
        )
        if suite_result:
            suite_content, _orphaned = suite_result
        else:
            suite_content = []
        orphaned_comments_list = None

    if suite_content:
        # Proper indentation respecting use_spaces setting
        if not style or not style.use_spaces:
            body_indent = "\t" * ((indent_level // 4) + 1)
        else:
            tab_size = style.tab_size or 4
            num_levels = (indent_level // max(tab_size, 1)) + 1
            body_indent = " " * (tab_size * num_levels)
        body_lines, last_line_no = format_block(
            suite_content,
            _format_func_body_statement,
            body_indent,
            indent_level + 4,
            style,
        )
        header_lines += body_lines

    return (header_lines, get_end_line_safe(statement))


def _ends_on_line(node, line) -> bool:
    """True when `node`'s source span ends on `line`.

    A comment ending on a declaration's last line is that declaration's inline comment;
    a comment on any later line is a standalone comment of its own.
    """
    from ..common.utils import get_end_line

    try:
        return int(get_end_line(node)) == int(line)
    except Exception:
        return getattr(getattr(node, "meta", None), "line", None) == line


def _format_inner_class(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format a nested class declaration."""
    is_static = False
    # Check for STATIC token in inner_class children
    for child in getattr(statement, "children", []):
        if isinstance(child, Token) and str(_tree_data_str(child)).lower() == "static":
            is_static = True
            break

    name_token = _find_name_token(statement) or "Unknown"
    prefix = "static class" if is_static else "class"

    # The guide asks for single-line inner-class declarations
    # (`@abstract class MyNode extends Node:`), so a base written in the header stays on
    # the declaration line instead of becoming an `extends` body statement.
    extends_text = ""
    for child in getattr(statement, "children", []):
        if hasattr(child, "data") and _tree_data_str(child).lower() == "extends_target":
            extends_text = f" extends {_extends_target_to_str(child)}"
            break

    # Check for inline comment that should appear right after the declaration header
    line_no = get_line(statement)
    decl_text = f"{indent_string}{prefix} {name_token}{extends_text}:"
    inline_comment = _find_inline_comment(statement, int(line_no), decl_text)
    if inline_comment:
        decl_text += f"  {inline_comment}"

    formatted_lines: FormattedLines = [(line_no, decl_text)]

    # Get class body content from class_block children
    has_inline_comment = bool(inline_comment)
    # We detect these by looking at comments that appear on member declaration lines
    # with column positions suggesting they're inline (not standalone).
    var_inline_comments: Dict[int, str] = {}
    ann_inline_comments: Dict[int, str] = {}
    class_block_children = []
    first_comment_emitted = False
    # First pass: collect all children from class_block
    orphaned_comments: list = []

    for child in getattr(statement, "children", []):
        if hasattr(child, "data"):
            data_str = str(_tree_data_str(child)).lower()
            if "class_block" in data_str:
                class_block_children = list(getattr(child, "children", []))
            elif data_str in ("class_body", "inline_class_body"):
                # A one-line body (`class Foo: var a; var b`) wraps its members the same
                # way a block does; keep only the members, the semicolons between them are
                # separators the layout does not need.
                class_block_children = [
                    c
                    for c in getattr(child, "children", [])
                    if c is not None and hasattr(c, "data")
                ]

    # Move enum same-line comments out of their class_member so the enum formatter can
    # attach them positionally.
    for member in class_block_children:
        if hasattr(member, "data"):
            _absorb_member_comments(member, class_block_children)

    # Detect inline comments on variable declarations
    for i, sub_child in enumerate(class_block_children):
        if hasattr(sub_child, "data"):
            sd = _tree_data_str(sub_child).lower()
            if "comment" in sd and hasattr(sub_child, "meta"):
                comment_line = getattr(sub_child.meta, "line", 0)
                # Check if the previous sibling is a class_member containing variable
                if i > 0:
                    prev = class_block_children[i - 1]
                    prev_sd = (
                        _tree_data_str(prev).lower() if hasattr(prev, "data") else ""
                    )
                    if prev_sd == "annotation":
                        if _ends_on_line(prev, comment_line):
                            ann_inline_comments[comment_line] = normalize_comment(
                                str(sub_child.children[0].value)
                            )
                    elif prev_sd == "class_member" and hasattr(prev, "children"):
                        for inner in getattr(prev, "children", []):
                            if hasattr(inner, "data"):
                                inner_data = _tree_data_str(inner).lower()
                                if inner_data in ("variable_decl", "local_variable"):
                                    # Only a comment on the declaration's own last line is that
                                    # declaration's inline comment. A comment on a later line is a
                                    # standalone comment: registering it here made the member loop
                                    # skip it, so it was dropped from the output entirely.
                                    if _ends_on_line(inner, comment_line):
                                        var_inline_comments[
                                            comment_line
                                        ] = normalize_comment(
                                            str(sub_child.children[0].value)
                                        )
                                    break

    class_body_stmts = []
    for i, sub_child in enumerate(class_block_children):
        if not hasattr(sub_child, "data"):
            continue
        sd = _tree_data_str(sub_child).lower()

        # Handle comments from class_block specially
        if "comment" in sd:
            comment_line_no = getattr(getattr(sub_child, "meta", None), "line", 0)
            is_inline_var_comment = (
                comment_line_no in var_inline_comments
                and not has_inline_comment
                or first_comment_emitted
            )
            if has_inline_comment and not first_comment_emitted:
                first_comment_emitted = True
                continue
            if is_inline_var_comment or comment_line_no in var_inline_comments:
                continue
            if comment_line_no in ann_inline_comments:
                # Trailing comment on an annotation's line; it rides on that line.
                continue

            class_body_stmts.append(("__comment__", sub_child))
        elif "class_member" in sd:
            # class_member wraps the actual declaration - unwrap it
            for inner in getattr(sub_child, "children", []):
                if hasattr(inner, "data") and not (
                    _tree_data_str(inner).lower() in ("end", "_nl", "comment")
                ):
                    inner_data = _tree_data_str(inner).lower()
                    # Check for orphaned comments INSIDE nested inner_class's block
                    if inner_data == "inner_class":
                        comment_node = _find_orphaned_comment_in_inner_class(inner)
                        class_body_stmts.append(
                            ("member", inner)
                        )  # Add inner_class first
                        if comment_node is not None:
                            # Insert orphan IMMEDIATELY after this inner_class (preserves source order)
                            class_body_stmts.append(("__comment__", comment_node))
                    # Check for orphaned comments INSIDE function suites
                    elif inner_data == "function_decl":
                        _, func_suite_orphans = _find_suite_or_empty(inner)
                        if func_suite_orphans:
                            # Remove orphaned comments from the suite so they won't appear in formatting
                            for child in getattr(inner, "children", []):
                                if hasattr(child, "data") and str(
                                    _tree_data_str(child)
                                ).lower().startswith("suite"):
                                    for oc in func_suite_orphans:
                                        try:
                                            if oc in getattr(child, "children", []):
                                                child.children.remove(oc)
                                        except ValueError:
                                            pass
                            # Add orphans AFTER this function member (matching source order: comment follows the declaration it was attached to)
                        class_body_stmts.append(("member", inner))
                        for oc in func_suite_orphans:
                            class_body_stmts.append(("__comment__", oc))
                    else:
                        # Check if this statement has an inline comment. Look it up by the
                        # declaration's last line, which is the line its inline comment sits
                        # on; the first line only matches when the declaration is one line
                        # long, so a trailing comment on a multiline declaration was lost.
                        from ..common.utils import get_end_line as _gel_member

                        try:
                            inner_line = _gel_member(inner)
                        except Exception:
                            inner_line = getattr(
                                getattr(inner, "meta", None), "line", 0
                            )
                        inline_comment_text = var_inline_comments.get(inner_line)
                        if inline_comment_text:
                            class_body_stmts.append(
                                ("member", inner, inline_comment_text)
                            )
                        else:
                            class_body_stmts.append(("member", inner))
        elif sd == "inner_class":
            # Non-class_member nested inner_class - add as member
            comment_node = _find_orphaned_comment_in_inner_class(sub_child)
            class_body_stmts.append(("member", sub_child))  # Add inner_class first
            if comment_node is not None:
                class_body_stmts.append(("__comment__", comment_node))
        elif sd == "extends_stmt":
            # Base declared as a body statement (`class C:` then `extends Base`).
            class_body_stmts.append(("member", sub_child))
        elif sd == "annotation":
            # An annotation on a class member. format_block joins it onto the
            # member's line when the placement rules allow.
            ann_line = getattr(getattr(sub_child, "meta", None), "line", 0)
            inline = ann_inline_comments.get(ann_line)
            if inline:
                class_body_stmts.append(("annotation", sub_child, inline))
            else:
                class_body_stmts.append(("annotation", sub_child))

    if class_body_stmts:
        # Proper indentation respecting use_spaces setting
        if not style or not style.use_spaces:
            body_indent = "\t" * ((indent_level // 4) + 1)
        else:
            tab_size = style.tab_size or 4
            num_levels = (indent_level // max(tab_size, 1)) + 1
            body_indent = " " * (tab_size * num_levels)
        class_lines, last_line_no = format_block(
            class_body_stmts,
            format_class_statement,
            body_indent,
            indent_level + 4,
            style,
        )
        formatted_lines += class_lines

    return (formatted_lines, get_end_line_safe(statement))


def _format_func_body_statement(
    statement: Any, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format a statement inside a function body."""
    from .function_statement import format_func_statement as fmt_func_stmt

    # If wrapped in tuple (from block), unwrap it first
    if isinstance(statement, tuple):
        tag, real_node = statement  # type: ignore
        return fmt_func_stmt(real_node, indent_string, indent_level, style)

    return fmt_func_stmt(statement, indent_string, indent_level, style)


def _format_bare_string(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format a bare string expression (standalone string not in var/const)."""
    from .expression_to_str import (
        _regular_string_to_str,
        _long_string_to_str,
        _long_rstring_to_str,
    )

    # Find the STRING or LONG_STRING token among children
    for child in getattr(statement, "children", []):
        if isinstance(child, Token):
            ttype = str(child.type).lower()
            if ttype == "regular_string" or ttype == "string":
                return _format_bare_string_token(child, indent_string)
            elif ttype in ("long_string", "long_rstring"):
                # Triple-quoted strings - normalize to double quotes
                raw = str(child.value)
                if raw.startswith("'''") or raw.startswith('"""'):
                    return _format_bare_string_token(child, indent_string)
                return format_simple_statement(
                    raw, statement, indent_string, indent_level
                )

    # Fallback: just show the data type
    stmt_data = (
        _tree_data_str(statement).lower() if hasattr(statement, "data") else "unknown"
    )
    return ([(get_line_safe(statement), f"{indent_string}/* {stmt_data} */")], get_end_line_safe(statement))  # type: ignore


def _format_bare_string_token(token: Token, indent_string: str) -> Outcome:
    """Format a bare string token with proper quote normalization."""
    raw = str(token.value)

    # Get the start position (could be int or tuple depending on Lark version)
    pos = getattr(token, "start_pos", None)
    if isinstance(pos, int):
        line_no = pos
    elif isinstance(pos, tuple) and len(pos) > 0:
        line_no = pos[0]
    else:
        line_no = 1

    # The quote choice and escaping are the plain string rules; a bare string is not a
    # different kind of literal. This used to re-escape the raw content without first
    # undoing the source's escapes, which produced `'x " " \\''` and `"x ""`.
    from .expression_to_str import _string_token_to_str

    result = _string_token_to_str(token)

    if "\n" in result:
        # A multi-line triple-quoted string keeps its own lines.
        lines = [(None, part) for part in result.split("\n")]
        lines[0] = (line_no, f"{indent_string}{lines[0][1]}")
        return (lines, line_no)

    return ([(line_no, f"{indent_string}{result}")], line_no)


def _format_unknown_class_member(
    statement: Any, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Fallback for unknown class member types."""
    stmt_data = (
        _tree_data_str(statement).lower() if hasattr(statement, "data") else "unknown"
    )
    return ([(get_line_safe(statement), f"{indent_string}/* {stmt_data} */")], get_end_line_safe(statement))  # type: ignore


# ─── Populate handlers after all functions are defined ──────────────────────

_HANDLERS.update(
    {
        "variable_decl": _format_variable_decl,
        "constant_decl": format_const_statement,
        "signal_decl": _format_signal_decl,
        "enum_decl": _format_enum_decl,
        "function_decl": _format_func_decl,
        "inner_class": _format_inner_class,
        # Top-level file items
        "extends_stmt": lambda s, i, l, st: format_extends_stmt(s, i, l),
        "class_name_stmt": lambda s, i, l, st: format_class_name_stmt(s, i, l),
        # Simple statements at class/file level
        "pass_stmt": partial(format_simple_statement, "pass"),
        "local_variable": lambda s, i, l, st: format_var_statement(s, i, l, st),  # type: ignore
        # Bare expressions (standalone strings)
        "bare_string": _format_bare_string,
        # Decorator annotations
        "annotation": _format_annotation,
    }
)


def get_line_safe(node) -> int:
    """Safe line number getter."""
    try:
        return get_line(node)
    except Exception:
        return 1


def get_end_line_safe(node) -> int:
    """Safe end line number getter."""
    try:
        return get_end_line(node)
    except Exception:
        return 1


def _find_inline_comment(
    statement: Tree, declaration_line: int, decl_prefix: str
) -> Optional[str]:
    """Find a comment that should appear inline with the declaration.

    Looks for comment children in the statement tree whose column position
    aligns with where an inline comment would be appended (after the colon).
    Returns the comment text or None if no inline comment found.
    """
    # Check for direct comment children that should appear inline
    for child in getattr(statement, "children", []):
        sd = _tree_data_str(child).lower() if hasattr(child, "data") else ""
        if "comment" in str(sd):
            for cc in getattr(child, "children", []):
                if (
                    isinstance(cc, Token)
                    and str(getattr(cc, "type", "")).upper() == "COMMENT"
                ):
                    return normalize_comment(str(cc.value))

    # Also check in class_block children for comments that appear before body
    for child in getattr(statement, "children", []):
        sd = _tree_data_str(child).lower() if hasattr(child, "data") else ""
        if sd == "class_block":
            first_comment_idx = -1
            first_body_idx = len(
                list(getattr(child, "children", []))
            )  # default: no body
            for idx, sub_child in enumerate(getattr(child, "children", [])):
                s = (
                    _tree_data_str(sub_child).lower()
                    if hasattr(sub_child, "data")
                    else ""
                )
                if "comment" in str(s):
                    col = (
                        getattr(sub_child.meta, "column", -1)
                        if hasattr(sub_child, "meta")
                        else -1
                    )
                    # Check column position: inline comments appear after the declaration
                    # text at a column past where we'd append (after colon + spaces)
                    if hasattr(sub_child, "meta"):
                        line_match = sub_child.meta.line == declaration_line
                        col_ok = col > len(decl_prefix) - 1
                        if line_match and col_ok:
                            first_comment_idx = idx
                elif "class_member" in str(s) and first_body_idx == len(
                    list(getattr(child, "children", []))
                ):
                    first_body_idx = idx

            if first_comment_idx >= 0 and first_comment_idx < first_body_idx:
                # Extract the COMMENT token from inside the comment tree at that index
                inline_child = list(getattr(child, "children", [None]))[
                    first_comment_idx
                ]
                for cc in getattr(inline_child, "children", []):
                    if (
                        isinstance(cc, Token)
                        and str(getattr(cc, "type", "")).upper() == "COMMENT"
                    ):
                        return normalize_comment(str(cc.value))

    # Check suite children for function inline comments
    for child in getattr(statement, "children", []):
        sd = _tree_data_str(child).lower() if hasattr(child, "data") else ""
        if "suite" in str(sd):
            for sub_child in getattr(child, "children", []):
                s = (
                    _tree_data_str(sub_child).lower()
                    if hasattr(sub_child, "data")
                    else ""
                )
                if "comment" in str(s):
                    col = (
                        getattr(sub_child.meta, "column", -1)
                        if hasattr(sub_child, "meta")
                        else -1
                    )
                    if (
                        sub_child.meta.line == declaration_line
                        and col > len(decl_prefix) - 1
                    ):
                        for cc in getattr(sub_child, "children", []):
                            if (
                                isinstance(cc, Token)
                                and str(getattr(cc, "type", "")).upper() == "COMMENT"
                            ):
                                return normalize_comment(str(cc.value))

    return None


def _find_var_inline_comment(
    statement: Tree, formatted_lines: list
) -> Optional[str]:  # noqa: F811
    """Find an inline comment for a variable declaration.

    NOTE: This function is kept for reference but inline comments are now handled
    at extraction time in _format_inner_class and _extract_suite_body, where sibling
    access to parent container children is available. The detection happens before
    formatting, then inline comments are passed through the 3-tuple format.
    """
    return None


def _find_name_token(statement: Tree):
    """Find the NAME token in a tree node."""
    for child in getattr(statement, "children", []):
        if isinstance(child, Token) and child.type == "NAME":  # type: ignore
            return str(child.value)  # type: ignore
    return None


def _find_name_token_in_tree(tree_node: Tree):
    """Find a NAME token within a tree."""
    for child in getattr(tree_node, "children", []):
        if isinstance(child, Token) and child.type == "NAME":  # type: ignore
            return str(child.value)  # type: ignore
    return None


def _has_full_call_suffix(node: Tree) -> bool:
    """Check if a tree contains PARENTHESIS_OPEN/CLOSE tokens indicating a function call."""
    from lark import Token

    if isinstance(node, Token):
        return node.type in ("PARENTHESIS_OPEN", "PARENTHESIS_CLOSE")
    if hasattr(node, "children"):
        for c in getattr(node, "children", []):
            if c is not None and _has_full_call_suffix(c):
                return True
    return False


def _find_suite_or_empty(
    statement: Tree, skip_inline_comment_line_no: int = 0, extract_orphans: bool = True
) -> tuple:
    """Find suite content within a function declaration.

    The grammar produces: suite -> _NL _INDENT (suite_line | comment)+ _DEDENT
    where suite children are statements, comment nodes, and end tokens.
    Detects inline comments on local variables and appends them via 3-tuple format.
    Also detects orphaned comments that have no real statements after them in the
    same suite - these comments are extracted to parent scope instead of being
    formatted inside the function body.

    Args:
        statement: Function declaration tree node
        skip_inline_comment_line_no: If non-zero, filter out comments on this line
            (they've been added inline to the header instead)
        extract_orphans: Whether a trailing comment run indented shallower than the
            body is extracted to the parent scope. A lambda body asks for them to stay:
            `_format_lambda_multi` renders the expression and has no enclosing scope to
            hand a comment to, so an extracted one would simply be dropped.

    Returns:
        A 2-tuple of (items, orphaned_comments) where items are the suite statements
        to format and orphaned_comments are comment nodes extracted for parent scope.
    """
    from lark import Tree as LarkTree

    # Build inline comment map for local variables first
    inline_comments: Dict[int, str] = {}

    items = []  # type: List[Any]
    orphaned_comments: List = []  # Comments extracted to parent scope
    seen_positions = (
        set()
    )  # Track positions of processed statements to avoid duplicates

    for child in getattr(statement, "children", []):
        if isinstance(child, LarkTree) and str(_tree_data_str(child)).lower().startswith("suite"):  # type: ignore
            suite_children = list(getattr(child, "children", []))

            # First pass: detect inline comments on local variables
            for i, sub_child in enumerate(suite_children):
                if isinstance(sub_child, LarkTree) and hasattr(sub_child, "data"):
                    sd = _tree_data_str(sub_child).lower()
                    if "comment" in sd and hasattr(sub_child, "meta"):
                        comment_line_no = getattr(
                            getattr(sub_child, "meta", None), "line", 0
                        )
                        comment_col = getattr(
                            getattr(sub_child, "meta", None), "column", 0
                        )
                        # Check if the previous sibling is a suite_line containing local_variable
                        if i > 0:
                            prev = suite_children[i - 1]
                            if (
                                isinstance(prev, LarkTree)
                                and _tree_data_str(prev).lower() == "suite_line"
                            ):
                                # Find end token to determine declaration column position
                                decl_end_col = None
                                decl_end_line = None
                                for deep in getattr(prev, "children", []):
                                    if (
                                        hasattr(deep, "data")
                                        and _tree_data_str(deep).lower() == "end"
                                    ):
                                        decl_end_col = getattr(
                                            getattr(deep, "meta", None), "column", 0
                                        )
                                        decl_end_line = getattr(
                                            getattr(deep, "meta", None), "line", 0
                                        )

                                # Inline comment: on the declaration's LAST line (a
                                # multiline declaration ends on a later line than it
                                # starts on) AND at or past the end column
                                if (
                                    decl_end_col
                                    and decl_end_line
                                    and comment_col >= decl_end_col
                                    and comment_line_no == decl_end_line
                                ):
                                    inline_comments[
                                        comment_line_no
                                    ] = normalize_comment(
                                        str(sub_child.children[0].value)
                                    )

            # Indentation (column) of the suite's real statements. A trailing comment is
            # only "orphaned" to the enclosing scope when it is indented LESS than the
            # body; a comment indented at the body level belongs to this suite.
            body_col = None
            for sub in suite_children:
                if not isinstance(sub, LarkTree):
                    continue
                sub_d = _tree_data_str(sub).lower()
                if "comment" in sub_d or "_nl" in sub_d or sub_d == "end":
                    continue
                col = getattr(getattr(sub, "meta", None), "column", None)
                if col:
                    body_col = col if body_col is None else min(body_col, col)

            # Second pass: extract items, detecting orphaned comments
            for i, sub_child in enumerate(suite_children):
                sd = (
                    _tree_data_str(sub_child).lower()
                    if hasattr(sub_child, "data")
                    else ""
                )

                if not hasattr(sub_child, "data"):
                    continue

                # Direct comment nodes inside suite - add as comments (or skip if inline)
                if "comment" in sd:
                    if skip_inline_comment_line_no > 0 and getattr(getattr(sub_child, "meta", None), "line", 0) == skip_inline_comment_line_no:  # type: ignore
                        continue  # Skip - already added inline to header
                    comment_line_no = getattr(
                        getattr(sub_child, "meta", None), "line", 0
                    )
                    if comment_line_no in inline_comments:
                        continue  # Inline var comment - skip here, handled via member tuple

                    # Check if this comment is orphaned (no real statements after it in suite).
                    # A later *comment* is not a real statement: a comment run trailing the body
                    # must be judged as a whole, or only its last line looks trailing and gets
                    # extracted while the rest stay behind - which is what split a `# ---` box
                    # into two pieces, one inside the function and one after it.
                    has_real_after = False
                    for later in suite_children[i + 1 :]:
                        lsd = (
                            _tree_data_str(later).lower()
                            if hasattr(later, "data")
                            else ""
                        )
                        if not lsd or "_nl" in lsd or lsd == "end" or "comment" in lsd:
                            continue
                        has_real_after = True
                        break

                    comment_col = getattr(getattr(sub_child, "meta", None), "column", 0)
                    if has_real_after:
                        # Comment has real statements after it - keep it in suite
                        items.append(("__comment__", sub_child))
                    elif (
                        extract_orphans
                        and body_col
                        and comment_col
                        and comment_col < body_col
                    ):
                        # Trailing comment indented shallower than the body: it belongs to
                        # the enclosing scope, not this suite.
                        orphaned_comments.append(sub_child)
                    else:
                        # Trailing comment at body indentation - keep it in the suite
                        items.append(("__comment__", sub_child))
                    continue

                # Skip NL markers and end tokens
                if "_nl" in sd or sd == "end":
                    continue

                inner_data = _tree_data_str(sub_child).lower()

                if inner_data == "suite_line":
                    # A trailing comment sits on the statement's LAST line, which for a
                    # multiline statement is the line its `end` marker is on.
                    stmt_end_line = 0
                    for deep_child in getattr(sub_child, "children", []):
                        if (
                            hasattr(deep_child, "data")
                            and _tree_data_str(deep_child).lower() == "end"
                        ):
                            stmt_end_line = getattr(
                                getattr(deep_child, "meta", None), "line", 0
                            )
                    for deep_child in getattr(sub_child, "children", []):
                        dd = str(_tree_data_str(deep_child)).lower()  # type: ignore
                        if (
                            hasattr(deep_child, "data")
                            and "comment" not in dd
                            and dd != "end"
                        ):
                            # Check for inline comment on this statement
                            lookup_line = stmt_end_line or getattr(
                                getattr(deep_child, "meta", None), "line", 0
                            )
                            ic_text = inline_comments.get(lookup_line)
                            if ic_text:
                                items.append(("member", deep_child, ic_text))
                            else:
                                items.append(("member", deep_child))
                elif inner_data == "local_variable":
                    # Direct local_variable inside suite (not wrapped in suite_line)
                    var_line = getattr(getattr(sub_child, "meta", None), "line", 0)
                    ic_text = inline_comments.get(var_line)
                    pos = (
                        sub_child.meta.start_pos
                        if hasattr(sub_child, "meta") and sub_child.meta.start_pos
                        else 0
                    )
                    stmt_key = f"{inner_data}_{pos}"
                    if stmt_key not in seen_positions:
                        seen_positions.add(stmt_key)
                        if ic_text:
                            items.append(("member", sub_child, ic_text))
                        else:
                            items.append(("member", sub_child))
                else:
                    # Direct statement inside suite (e.g., expression_stmt, etc.)
                    # Check for duplicates using position
                    pos = (
                        sub_child.meta.start_pos
                        if hasattr(sub_child, "meta") and sub_child.meta.start_pos
                        else 0
                    )
                    stmt_key = f"{inner_data}_{pos}"
                    if stmt_key not in seen_positions:
                        seen_positions.add(stmt_key)
                        items.append(("member", sub_child))

    return (items, orphaned_comments)


def format_extends_stmt(
    statement: Tree, indent_string: str, indent_level: int = 0
) -> Outcome:
    """Format an extends statement: `extends Target` or `extends Target.Type`."""
    target_parts = []
    for child in getattr(statement, "children", []):
        if isinstance(child, Token):
            ttype = str(getattr(child, "type", "")).lower()
            if ttype == "extends":
                continue
            if ttype == "period":
                target_parts.append(".")
            else:
                target_parts.append(str(child.value))
        elif hasattr(child, "data"):
            d = _tree_data_str(child).lower()
            if d == "end":
                continue
            if d == "extends_target":
                target_parts.append(_extends_target_to_str(child))
            else:
                target_parts.append(_safe_str(child) or _tree_data_str(child))
    return format_simple_statement(
        f"extends {''.join(target_parts)}", statement, indent_string, indent_level
    )


def _parameter_default_value(param: Tree):
    """Return the default-value expression of a `parameter` node, or None."""
    seen_equal = False
    for child in getattr(param, "children", []):
        if child is None:
            continue
        if isinstance(child, Token):
            if str(getattr(child, "type", "")).lower() == "equal":
                seen_equal = True
            continue
        if seen_equal:
            return child
    return None


def _parameter_with_stripped_default(
    param: Tree, header_prefix: str, indent_string: str, style
) -> Tree:
    """`param` with a redundant paren group removed from its default value.

    A group spanning the whole default is redundant - the guide asks to avoid parentheses
    that are not needed for order of operations or for wrapping over multiple lines, and
    nothing outside the group can bind to what is inside it - so `func f(x = (1 + 2))` is
    `func f(x = 1 + 2)`.

    The parameter is rendered flat (`full`) and its default may also be laid out on its own,
    so the strip has to happen on the node before either rendering, or the two forms
    disagree about whether the parentheses are there, and the head of the flat form is
    derived by removing the rendered default from its end. `header_prefix` is the start of
    the declaration line, which the fit test needs.
    """
    value = _parameter_default_value(param)
    if value is None:
        return param
    from .expression_utils import strip_slot_parentheses
    from .expression_to_str import expression_to_str as _ets

    flat = _ets(param)
    value_str = _ets(value)
    head = (
        flat[: len(flat) - len(value_str)]
        if value_str and flat.endswith(value_str)
        else ""
    )
    stripped = strip_slot_parentheses(
        value, f"{header_prefix}{head}", indent_string, style
    )
    if stripped is value:
        return param
    children = list(getattr(param, "children", []))
    for i, child in enumerate(children):
        if child is value:
            children[i] = stripped
            break
    return Tree(param.data, children, getattr(param, "meta", None))


def _parameters_with_stripped_defaults(
    params: Tree, header_prefix: str, indent_string: str, style
) -> Tree:
    """A `parameters` node whose defaults have lost a redundant paren group.

    The header's flat form is one rendering of this node (`x = (1 + 2)` came from
    `expression_to_str` on the node itself), so the strip has to happen on the node rather
    than on the parts the breaking layout works from.
    """
    children = list(getattr(params, "children", []))
    changed = False
    for i, child in enumerate(children):
        if child is None or not hasattr(child, "data"):
            continue
        if "parameter" not in _tree_data_str(child).lower():
            continue
        stripped = _parameter_with_stripped_default(
            child, header_prefix, indent_string, style
        )
        if stripped is not child:
            children[i] = stripped
            changed = True
    if not changed:
        return params
    return Tree(params.data, children, getattr(params, "meta", None))


def _parameter_parts_with_values(params: Tree):
    """Split a `parameters` node into `(full, head, value_node)` per parameter.

    `full` is the whole parameter rendered flat (`a = [1]`); `head` is the parameter
    up to and including its `=`/`:=` and the space after it (`a = `, `b := `,
    `x: Array = `), or the whole parameter when it has no default; `value_node` is
    the default expression, or None. The header layout needs `full` for the
    all-on-one-line attempt and `head` plus `value_node` to explode a default that
    does not fit its own line.

    Returns `(parts, has_trailing_comma)`.
    """
    from .expression_to_str import expression_to_str as _ets

    parts = []
    last_meaningful = None
    for child in getattr(params, "children", []):
        if child is None:
            continue
        last_meaningful = child
        if isinstance(child, Token):
            continue
        if "parameter" not in _tree_data_str(child).lower():
            continue
        # Callers pass a `parameters` node whose defaults have already lost a redundant paren
        # group (`_parameters_with_stripped_defaults`), so the flat form and the default's own
        # node agree.
        full = _ets(child)
        value_node = _parameter_default_value(child)
        if value_node is None:
            parts.append((full, full, None))
            continue
        value_str = _ets(value_node)
        # The default value is the trailing part of the rendered parameter, so the
        # head is what precedes it.
        head = (
            full[: len(full) - len(value_str)]
            if value_str and full.endswith(value_str)
            else full
        )
        parts.append((full, head, value_node))
    has_trailing_comma = (
        isinstance(last_meaningful, Token)
        and str(getattr(last_meaningful, "type", "")).lower() == "comma"
    )
    return parts, has_trailing_comma


def _extends_target_to_str(node) -> str:
    """Render an `extends_target` (STRING/NAME followed by `.NAME` parts).

    `_safe_str` returns only the first token of a multi-part target, which drops
    everything after the first dot, so walk the target explicitly.
    """
    parts = []
    for child in getattr(node, "children", []):
        if child is None:
            continue
        if isinstance(child, Token):
            ttype = str(getattr(child, "type", "")).lower()
            if ttype == "period":
                parts.append(".")
            else:
                parts.append(str(child.value))
        elif hasattr(child, "data"):
            parts.append(_extends_target_to_str(child))
    return "".join(parts)


def _break_param_list(
    params_tree, open_line, line_no, close_text, indent_string, style
):
    """Lines for a parameter list that does not fit on one line.

    `params_tree` is the `parameters`/`signal_parameters` node; `open_line` is the first
    line, already indented and ending with the opening parenthesis; `close_text` closes
    the header. Shared by the function header and the signal declaration, because the
    rule is the same: a source trailing comma forces one parameter per line, otherwise
    every parameter shares one continuation line when it fits. A parameter whose default
    value does not fit its own line is laid out multiline (a signal parameter has no
    default, so this only fires for a function). A comment in the list is emitted where
    it was written, on the parameter's line when it follows one.
    """
    from .expression import (
        _collection_layout,
        _comment_layout_text,
        _format_concrete_expression,
        _node_end_line,
        _node_line,
    )
    from .expression_to_str import expression_to_str as _ets

    max_len = style.line_length if style else 100
    from .expression_utils import continuation_indent

    inner = continuation_indent(indent_string)
    inner_width = _display_width(inner)

    def fits(text):
        return inner_width + _display_width(text) <= max_len

    lines = [(line_no, open_line)]

    layout = _collection_layout(params_tree)
    if any(kind == "comment" for kind, _n, _c in layout):
        # Each parameter keeps the author's own separator and each comment stays on the
        # line it was written on.
        prev_line = line_no
        for kind, node, had_comma in layout:
            if kind == "comment":
                text = _comment_layout_text(node)
                c_line = _node_line(node)
                if lines and c_line and c_line == prev_line:
                    ln, txt = lines[-1]
                    lines[-1] = (ln, f"{txt}  {text}")
                else:
                    lines.append((None, f"{inner}{text}"))
                continue
            lines.append((None, f"{inner}{_ets(node)}{',' if had_comma else ''}"))
            prev_line = _node_end_line(node)
        lines.append((None, f"{indent_string}{close_text}"))
        return lines

    param_layout, has_trailing_comma = _parameter_parts_with_values(params_tree)
    joined = ", ".join(full for full, _, _ in param_layout)
    if not has_trailing_comma and fits(joined):
        lines.append((None, f"{inner}{joined}"))
    else:
        for idx, (full, head, value_node) in enumerate(param_layout):
            is_last = idx == len(param_layout) - 1
            trailing = "," if (not is_last or has_trailing_comma) else ""
            if value_node is None or fits(full + trailing):
                lines.append((None, f"{inner}{full}{trailing}"))
                continue
            value_lines = _format_concrete_expression(
                value_node,
                ExpressionContext(head, line_no, "", None),
                inner,
                style or StyleConfig(),
            )
            for j, (_, text) in enumerate(value_lines):
                lines.append(
                    (None, f"{text}{trailing}" if j == len(value_lines) - 1 else text)
                )
    lines.append((None, f"{indent_string}{close_text}"))
    return lines


def format_class_name_stmt(
    statement: Tree, indent_string: str, indent_level: int = 0
) -> Outcome:
    """Format a class_name statement: `class_name Foo` or `class_name Foo extends Target`."""
    parts = ["class_name"]
    for i, child in enumerate(getattr(statement, "children", [])):
        if isinstance(child, Token):
            ttype = str(getattr(child, "type", "")).lower()
            if ttype == "class_name":  # Skip the CLASS_NAME keyword
                continue
            # The `extends` keyword is re-emitted together with its target below;
            # dropping it produced `class_name Foo Node`, which is not valid GDScript.
            if str(getattr(child, "value", "")) == "extends":
                continue
            parts.append(str(child.value))
        elif hasattr(child, "data"):
            d = str(_tree_data_str(child)).lower()
            if d == "end":
                break  # Stop at the end token
            elif d == "extends_target":
                parts.append("extends")
                parts.append(_extends_target_to_str(child))
    return format_simple_statement(
        " ".join(parts), statement, indent_string, indent_level
    )


def _tree_data_str(node) -> str:
    """Return a tree's rule name or token type as a plain string."""
    from lark import Tree, Token

    if isinstance(node, Token):
        return str(getattr(node, "type", ""))  # type: ignore
    elif isinstance(node, Tree):
        d = node.data
        return str(d) if not hasattr(d, "value") else d.value  # type: ignore
    return ""


def _get_comment_text(comment_node) -> str:
    """Extract the text content from a comment tree node."""
    from .comments import normalize_comment

    for child in getattr(comment_node, "children", []):
        if hasattr(child, "value"):
            return normalize_comment(str(child.value))  # type: ignore
    return ""


def _find_orphaned_comment_in_inner_class(inner_class_node: Tree) -> "Tree | None":
    """Find and remove an orphaned comment inside a nested inner_class's block.

    Returns the comment node if it should be at the parent scope instead,
    or None if there's no orphaned comment.
    Also DETACHES the comment from the inner_class so it won't appear in both places.

    A comment is "orphaned" when:
    1. It appears as a direct child of an inner_class's class_block
    2. There are no other class_members AFTER it within that block
       (meaning it comes after all real declarations)
    3. It is indented shallower than the block's own content, so it belongs to the
       enclosing scope; a comment at the body's column is the class's own and stays
    """
    for gc in getattr(inner_class_node, "children", []):
        gcsd = _tree_data_str(gc).lower()
        if "class_block" not in gcsd:
            continue
        if not hasattr(gc, "children") or len(gc.children) == 0:
            continue

        # Column of the block's own content, so a comment written at that column is not
        # mistaken for one trailing the class at the enclosing scope's column.
        body_col = None
        for member in gc.children:
            msd = _tree_data_str(member).lower()
            if not hasattr(member, "data") or "comment" in msd or msd in ("end", "_nl"):
                continue
            col = getattr(getattr(member, "meta", None), "column", None)
            if col:
                body_col = col if body_col is None else min(body_col, col)

        # Find all comments inside this class_block (iterate backwards to safely remove)
        for i in range(len(gc.children) - 1, -1, -1):
            gg_child = gc.children[i]
            gg_sd = _tree_data_str(gg_child).lower()
            if "comment" not in gg_sd:
                continue

            # Check if there are any real members AFTER this comment
            has_member_after = False
            for later in gc.children[i + 1 :]:
                ls = _tree_data_str(later).lower()
                if ls not in ("end", "_nl") and "comment" not in ls:
                    has_member_after = True
                    break

            if not has_member_after:
                comment_col = getattr(getattr(gg_child, "meta", None), "column", 0)
                if body_col and comment_col and comment_col >= body_col:
                    # Indented at the body's column, so it belongs inside this class.
                    continue
                # This is an orphaned comment - detach it from inner_class
                gc.children.pop(i)
                return gg_child  # Return so caller can add to parent scope
    return None
