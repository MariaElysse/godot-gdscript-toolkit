"""Expression formatter for gdformat2.

Adapted from original gdtoolkit/formatter/expression.py to use tree-based
comment handling instead of line-indexed arrays.
"""
from typing import Any, List, Optional, Tuple

from lark import Tree, Token

from ..common.utils import get_end_line
from .context import ExpressionContext
from .expression_utils import (
    collection_indent,
    continuation_indent,
    continuation_indent_for,
)
from .style_config import StyleConfig
from .types import FormattedLine, FormattedLines


def _extract_all_tokens(node, names: List[str]) -> None:
    """Collect identifier-like Token values under *node*, recursing through Trees.

    Lark Earley may report an empty .type, so identifiers are extracted by value,
    filtering out dots, operators and symbols.
    """
    for child in getattr(node, "children", []):
        if isinstance(child, Token):
            cval = str(getattr(child, "value", "") or "")
            # Keep identifiers (not dots, operators, symbols)
            if (
                cval
                and not any(c in cval for c in "+-*/%&|^~!@#'\"\\`{}[]():;,=<>")
                and cval != "."
            ):
                names.append(cval)
        elif isinstance(child, Tree):
            _extract_all_tokens(child, names)


def _extract_attr_names(node, names: List[str]) -> None:
    """Collect Token values one level under *node*, skipping PERIOD tokens."""
    for child in getattr(node, "children", []):
        if isinstance(child, Token) and child.type == "PERIOD":
            continue
        elif isinstance(child, Tree):
            for sub_ch in getattr(child, "children", []):
                if isinstance(sub_ch, Token):
                    val = str(getattr(sub_ch, "value", "") or "")
                    if val:
                        names.append(val)


def format_expression(
    expression,
    expression_context: ExpressionContext,
    indent_string: str,
    style: Optional[StyleConfig] = None,
) -> tuple:
    """Format an expression tree into formatted lines.

    Args:
        expression: The Lark Tree node for the expression, or a Token if the
            parser produced a bare token (e.g. a simple NAME in a while condition)
        expression_context: Prefix/suffix strings and their line positions
        indent_string: The indentation string to use (spaces or tabs)
        style: Optional StyleConfig for formatting rules

    Returns:
        Tuple of (formatted_lines, end_line_number)
    """
    if isinstance(expression, Token):
        formatted = _format_standalone_expression(  # type: ignore
            expression, expression_context, indent_string, style
        )
        return formatted, get_end_line_safe(expression)

    # For trees that produce their full output via expression_to_str (like type annotations,
    # enum bodies, assignment wrappers with maybe_placeholders), use the whole tree.
    data = (
        str(getattr(expression, "data", None))
        if not hasattr(expression.data, "value")
        else getattr(expression.data, "value", "")
    )
    special_types = (
        "type",
        "return_type",
        "enum_body",
        "enum_entries",
        "assignment",
        "cast",
        "ternary",
        "type_test",
    )
    non_none_children = (
        [c for c in expression.children if c is not None] if expression.children else []
    )
    # Use whole tree if: special type, or all children are None (maybe_placeholders),
    # or first child would be a None placeholder
    uses_whole_tree = (
        data in special_types
        or len(non_none_children) == 0
        or (
            expression.children
            and expression.children[0] is not None
            and expression.children[0] is not expression
        )
    )

    concrete_expression = (
        expression
        if uses_whole_tree
        else (expression.children[0] if expression.children else expression)
    )
    return _format_standalone_expression(
        concrete_expression, expression_context, indent_string, style
    ), get_end_line(expression)


def format_concrete_expression(
    expression: Tree,
    expression_context: ExpressionContext,
    indent_string: str,
    style: Optional[StyleConfig] = None,
) -> tuple:
    return _format_concrete_expression(
        expression, expression_context, indent_string, style
    ), get_end_line(expression)


def _is_foldable(expression) -> bool:
    """Check if an expression can be folded onto a single line."""
    non_foldable_types = [
        "string",
        "rstring",
        "get_node",
        "node_path",
        "string_name",
        "unique_node_path",
        "signal_arg_typed",
        "signal_arg_regular",
        "non_foldable_dot_chain",
        "var_capture_pattern",
        "etc_pattern",
        "wildcard_pattern",
    ]

    if _is_multiline_string(expression):
        return True
    if isinstance(expression, Token):
        return False
    data = (
        str(expression.data)
        if not hasattr(expression.data, "value")
        else expression.data.value  # type: ignore
    )
    return data not in non_foldable_types


def _has_standalone_comment_children(tree_node: Tree) -> bool:
    """Check if a tree node has comment children (standalone comments)."""
    for child in getattr(tree_node, "children", []):
        if isinstance(child, Tree) and str(child.data) == "comment":  # type: ignore
            return True
    return False


def _has_trailing_comma(expression) -> bool:
    from .expression_utils import has_trailing_comma as _htc

    return _htc(expression)


def _has_trailing_comma_in_elements(tree_node: Tree) -> bool:
    """Detect trailing comma in array/dict elements for the Earley grammar.

    In the new grammar, arrays have an 'elements' subtree with children like
    [expr1, COMMA, expr2, COMMA, ..., exprN]. A trailing comma adds one more
    COMMA token after the last expression.
    """
    for child in getattr(tree_node, "children", []):
        if isinstance(child, Tree):
            d = str(getattr(child.data, "value", None)) or str(child.data)
            if d == "elements":
                elems = list(child.children)
                # Count expressions vs commas: N elements should have N-1
                # commas without trailing comma
                expr_count = sum(1 for e in elems if isinstance(e, Tree))
                comma_count = sum(
                    1
                    for e in elems
                    if isinstance(e, Token)
                    and str(getattr(e, "type", "")).upper() == "COMMA"
                )
                # Trailing comma: same number of commas as expressions
                return comma_count >= expr_count
            elif d in (
                "dict_entries",
                "enum_entries",
                "arguments",
                "array_pattern_items",
                "dict_pattern_items",
            ):
                elems = list(child.children)
                expr_count = sum(1 for e in elems if isinstance(e, Tree))
                comma_count = sum(
                    1
                    for e in elems
                    if isinstance(e, Token)
                    and str(getattr(e, "type", "")).upper() == "COMMA"
                )
                return comma_count >= expr_count

    # Fallback: check the old grammar pattern (trailing_comma tree node at end)
    from .expression_utils import has_trailing_comma as _htc

    if len(tree_node.children) > 0:
        last = tree_node.children[-1]
        if isinstance(last, Tree):
            d = str(getattr(last.data, "value", None)) or str(last.data)
            if d == "trailing_comma":
                return True

    return False


def _lambda_body_forces_multiline(lambda_node) -> bool:
    """True if a lambda body cannot be written inline.

    `func(x): pass pass` and `func(): if c: pass` are not valid GDScript, so a body
    with more than one statement, or with a compound statement, must use the block
    layout even when the whole thing would fit on one line. A single-statement body
    can also be unwritable on one line: `func(): return [1,]` has to explode because
    the array it returns carries a trailing comma.
    """
    suite = next(
        (
            c
            for c in getattr(lambda_node, "children", [])
            if isinstance(c, Tree) and _tree_data_str(c) == "suite"
        ),
        None,
    )
    if suite is None:
        return False
    statements = []
    for child in getattr(suite, "children", []):
        if not isinstance(child, Tree):
            continue
        d = _tree_data_str(child)
        if d in ("comment", "end"):
            continue
        if d == "suite_line":
            # Block suite: each line contributes its statement.
            for sub in getattr(child, "children", []):
                if isinstance(sub, Tree) and _tree_data_str(sub) not in (
                    "end",
                    "comment",
                ):
                    statements.append((_tree_data_str(sub), sub))
            continue
        statements.append((d, child))
    if len(statements) > 1:
        return True
    if not statements:
        return False
    stmt_type, stmt_node = statements[0]
    if stmt_type in ("if_stmt", "for_stmt", "while_stmt", "match_stmt"):
        return True
    return _is_forced_multiline(stmt_node)


def _has_inline_forcing_lambda(node, depth=0) -> bool:
    """True if `node` contains a lambda whose body must be a block."""
    if not isinstance(node, Tree) or depth > 25:
        return False
    if _tree_data_str(node) == "lambda" and _lambda_body_forces_multiline(node):
        return True
    return any(
        _has_inline_forcing_lambda(c, depth + 1) for c in getattr(node, "children", [])
    )


def _is_forced_multiline(expression, style=None) -> bool:  # noqa: C901
    """Check if an expression MUST be formatted multiline regardless of line length.

    This handles arrays with trailing commas which the legacy formatter always forces
    to multiline, even when they'd fit on one line.
    """
    # A collection carrying a trailing comma ANYWHERE in the expression forces the whole
    # expression multiline (legacy keeps such a collection exploded, e.g.
    # `not 1 in [1] in [true,]`). A lambda body is a separate block, so don't descend
    # into it.
    def _has_tc_collection(node, depth=0):
        if not isinstance(node, Tree) or depth > 25:
            return False
        d = _tree_data_str(node)
        if d in (
            "array_literal",
            "dict_literal",
            "array",
            "dict",
            "enum_body",
            "array_pattern",
            "dict_pattern",
        ):
            if _has_trailing_comma_in_elements(node):
                return True
        if d == "call_suffix" and _has_trailing_comma_in_elements(node):
            # A trailing comma in the call's arguments forces the whole expression
            # multiline, the same as one in a collection.
            return True
        if d == "lambda":
            return False
        for c in getattr(node, "children", []):
            if _has_tc_collection(c, depth + 1):
                return True
        return False

    if _has_tc_collection(expression):
        return True

    # A comment inside a collection cannot be written on one line, and the collection's
    # layout is the only thing that knows where to put it, so treat it as forced.
    if _has_comment_child(expression):
        return True

    # A comment inside a lambda's own parameter list is the same, but `_has_comment_child`
    # stops at a lambda, so it needs its own check.
    if _lambda_header_has_comment(expression):
        return True

    if _has_inline_forcing_lambda(expression):
        return True

    # Unwrap assignment/cast/ternary/type_test wrappers to find the inner node
    unwrapped = expression
    wrapper_types = ("assignment", "cast", "ternary", "type_test")
    for _ in range(20):  # max unwrap depth
        if isinstance(unwrapped, Tree):
            d = (
                str(getattr(unwrapped.data, "value", unwrapped.data))
                if hasattr(unwrapped.data, "value")
                else str(unwrapped.data)
            )
            if d in wrapper_types:
                non_none = [
                    c for c in getattr(unwrapped, "children", []) if c is not None
                ]
                if len(non_none) == 1:
                    unwrapped = non_none[0]
                else:
                    break
            else:
                break
        else:
            # Unwrapped to a Token - no forced multiline needed for primitives
            return False

    if not isinstance(unwrapped, Tree):
        return False

    # Check if the inner node is an array_literal with trailing comma
    d = (
        str(getattr(unwrapped.data, "value", unwrapped.data))
        if hasattr(unwrapped.data, "value")
        else str(unwrapped.data)
    )
    if d == "array_literal" and _has_trailing_comma_in_elements(unwrapped):
        return True
    # Also check for nested arrays with trailing commas (recursive)
    if d == "array_literal":
        WRAPPERS = frozenset({"assignment", "cast", "ternary", "type_test"})

        def unwrap(node):
            """Unwrap wrapper nodes to get the innermost expression."""
            while isinstance(node, Tree) and _tree_data_str(node) in WRAPPERS:
                for c in getattr(node, "children", []):
                    if c is not None and isinstance(c, Tree):
                        node = c
                        break
                else:
                    break
            return node

        def find_nested_array_with_tc(node, depth=0):
            """Recursively search for array_literals with trailing commas."""
            if not isinstance(node, Tree) or depth > 15:
                return False
            # First: unwrap wrappers to check the actual inner content
            inner = unwrap(node)
            if _tree_data_str(inner) == "array_literal":
                if _has_trailing_comma_in_elements(inner):
                    return True
                # Recurse into THIS array's elements to find deeper nested arrays
                for child in getattr(inner, "children", []):
                    if isinstance(child, Tree) and _tree_data_str(child) == "elements":
                        for ec in getattr(child, "children", []):
                            if ec is not None and not (
                                isinstance(ec, Token)
                                and str(getattr(ec, "type", "")).upper() == "COMMA"
                            ):
                                if find_nested_array_with_tc(ec, depth + 1):
                                    return True
                # Also check bracket tokens' siblings aren't relevant
            else:
                # Not an array_literal itself (could be assignment/cast/etc wrapper or other type)
                # Recurse into all Tree children to find nested arrays
                for child in getattr(node, "children", []):
                    if isinstance(child, Tree):
                        if find_nested_array_with_tc(child, depth + 1):
                            return True
            return False

        if find_nested_array_with_tc(unwrapped):
            return True

    # Check if the inner node is a dict with many entries or long keys
    if d in ("dict_literal", "enum_body"):
        # For dict_literal, count actual dict_entries children (not brace tokens)
        if d == "dict_literal":
            entries_tree = None
            for c in getattr(unwrapped, "children", []):
                if (
                    isinstance(c, Tree)
                    and str(getattr(c.data, "value", c.data)) == "dict_entries"
                ):
                    entries_tree = c
                    break
            # Count entry nodes only: the container also holds the COMMA tokens between
            # them, and counting those made a two-entry dict look like a three-entry one.
            entry_count = len(
                [
                    c
                    for c in (
                        getattr(entries_tree, "children", []) if entries_tree else []
                    )
                    if isinstance(c, Tree)
                    and _tree_data_str(c)
                    in ("dict_entry", "eq_dict_entry", "dict_pattern_entry")
                ]
            )
        else:
            entry_count = len(
                [c for c in getattr(unwrapped, "children", []) if c is not None]
            )
        if entry_count >= 3 or _has_trailing_comma_in_elements(unwrapped):
            return True

    # Check for subscripts with arrays (trailing comma or many elements)
    for child in getattr(unwrapped, "children", []):
        if isinstance(child, Tree):
            cdata = str(getattr(child.data, "value", None)) or str(child.data)
            if cdata == "subscript":
                # Check inside the subscript's inner expression
                sub_inner = [
                    c
                    for c in getattr(child, "children", [])
                    if not (
                        isinstance(c, Token)
                        and str(getattr(c, "type", "")).lower()
                        in ("bracket_open", "bracket_close")
                    )
                ]
                if not sub_inner:
                    continue
                sub_inner = sub_inner[0]
                # Walk wrappers to find array_literal or paren-wrapped additive expressions
                # Node unwrapping walks Tree/Token/None values, not just lists.
                inner: Any = sub_inner
                for _ in range(10):
                    if isinstance(inner, Tree):
                        id2 = str(getattr(inner.data, "value", None)) or str(inner.data)
                        if id2 == "array_literal":
                            # Check element count
                            elems_tree = [
                                c
                                for c in getattr(inner, "children", [])
                                if hasattr(c, "data")
                                and (str(getattr(c.data, "value", None)) or str(c.data))
                                == "elements"
                            ]
                            if elems_tree:
                                ec = list(elems_tree[0].children)
                                expr_count = sum(1 for e in ec if isinstance(e, Tree))
                                comma_count = sum(
                                    1
                                    for e in ec
                                    if isinstance(e, Token)
                                    and str(getattr(e, "type", "")).upper() == "COMMA"
                                )
                                # Arrays with trailing commas should ALWAYS be multiline
                                if comma_count >= expr_count:
                                    return True
                                # Arrays with many elements (>5) should force multiline
                                if expr_count > 5:
                                    return True
                            inner = None  # Done checking - don't continue unwrapping
                        elif id2 in (
                            "additive",
                            "shift_expr",
                            "mdr_expr",
                            "arith_expr",
                            "bitw_or",
                            "bitw_xor",
                            "bitw_and",
                            "bit_and",
                            "bit_or",
                            "bit_xor",
                        ):
                            # Direct additive/arith chain inside subscript -
                            # this triggers paren preservation multiline
                            return True
                        elif (
                            id2 == "primary"
                        ):  # paren-wrapped expressions (new grammar)
                            # Check for paren-wrapped additive/arith chains -
                            # preserve them multiline
                            non_none_inner = [
                                c
                                for c in getattr(inner, "children", [])
                                if c is not None
                            ]
                            inner_expr = None
                            if len(non_none_inner) >= 2:
                                inner_expr = (
                                    non_none_inner[1]
                                    if len(non_none_inner) > 1
                                    else non_none_inner[0]
                                )
                                # Check if this is an additive/arith chain
                                # (preserves readability with parens)
                                d_ie = str(
                                    getattr(inner_expr.data, "value", None)
                                ) or str(inner_expr.data)
                                if d_ie in (
                                    "additive",
                                    "shift_expr",
                                    "mdr_expr",
                                    "arith_expr",
                                    "bitw_or",
                                    "bitw_xor",
                                    "bitw_and",
                                ):
                                    return True
                            # Even if not additive, continue unwrapping through the inner expression
                            # (it may have assignment/cast/etc wrappers around an additive chain)
                            if inner_expr and isinstance(inner_expr, Tree):
                                inner = (
                                    inner_expr  # Continue checking the wrapped content
                                )
                            else:
                                break
                            continue  # Skip elif/else chains to let loop re-evaluate with new inner
                        elif id2 in ("assignment", "cast", "ternary", "type_test"):
                            non_none = [
                                c
                                for c in getattr(inner, "children", [])
                                if c is not None
                            ]
                            if len(non_none) == 1:
                                inner = non_none[0]
                            else:
                                break
                        else:
                            # Unknown type - check children anyway (e.g.,
                            # subscript wrapping other stuff)
                            for ch in getattr(inner, "children", []):
                                if isinstance(ch, Tree):
                                    inner = ch  # Recurse into child
                                    break
                            else:
                                break
                    else:
                        break

    return False


def _is_multiline_string(expression) -> bool:
    if not isinstance(expression, Tree):
        return False
    if str(expression.data) != "string":  # type: ignore
        return False
    first = expression.children[0]
    if isinstance(first, Token) and first.type == "LONG_STRING":  # type: ignore
        return len(str(first.value).splitlines()) > 1  # type: ignore
    return False


def _format_standalone_expression(
    expression, expression_context: ExpressionContext, indent_string: str, style=None
) -> FormattedLines:
    from .expression_utils import remove_outer_parentheses as _rop

    expression = _rop(expression)
    return _format_concrete_expression(
        expression, expression_context, indent_string, style
    )


def get_data_str(node):
    if isinstance(node, Tree):
        d = getattr(getattr(node.data, "value", None), "value", str(node.data))
        return str(d)
    elif hasattr(node, "type"):
        return "Token(%s)" % node.type
    return "?"


def _format_concrete_expression(
    expression, expression_context: ExpressionContext, indent_string: str, style=None
) -> FormattedLines:
    if isinstance(expression, Token):
        return [_format_token_as_line(expression, expression_context, indent_string)]

    # Keep original data for wrapper handling below; unwrap only when needed
    orig_expression = expression

    if _is_foldable(expression):
        max_len = style.line_length if style else 100
        single = _format_concrete_expression_to_single_line(
            expression, expression_context, indent_string
        )
        _, line_text = single[0]
        display_length = len(line_text.replace("\t", "    "))
        has_comments = _has_standalone_comment_children(expression)  # type: ignore
        forced_multiline = (
            _is_forced_multiline(orig_expression, style) if not has_comments else False
        )
        if display_length <= max_len and not has_comments and not forced_multiline:
            return single
        # Line is too long or forced multiline — delegate to multiline formatter.
        # Unwrap wrapper nodes (assignment/cast/ternary/type_test) before calling
        # the multiline formatter so it can find the right handler.
        unwrapped = expression
        data_str = get_data_str(unwrapped)
        WRAPPERS = frozenset({"assignment", "cast", "ternary", "type_test"})
        # A cast/type_test carrying `is`/`as` repetitions must NOT be unwrapped: the
        # repetitions live on this node, so unwrapping to the first child drops them
        # and the chain is laid out as its bare operand.
        has_repetitions = (
            data_str in ("cast", "type_test", "asless_type_test", "actual_type_cast")
            and len([c for c in getattr(unwrapped, "children", []) if c is not None])
            > 1
        )
        if data_str in WRAPPERS and not has_repetitions:
            for c in getattr(unwrapped, "children", []):
                if c is not None and isinstance(c, Tree):
                    unwrapped = _unwrap_wrappers_chain(c)
                    break
        # Only delegate to multiline handler if the unwrapped node has a known type.
        # For simple tokens or unknown types, fall back to single-line output with
        # prefix context preserved (to avoid losing variable names in var declarations).
        multi_data = get_data_str(unwrapped)
        # A cast/type_test is only a multiline chain when it carries repetitions; a
        # plain one is a pass-through and must keep its single-line fallback (a node
        # path value such as `^"../Some/Stuff"` is wrapped in one).
        is_repetition_chain = (
            multi_data in ("cast", "type_test", "asless_type_test", "actual_type_cast")
            and len([c for c in getattr(unwrapped, "children", []) if c is not None])
            > 1
        )
        KNOWN_MULTILINE_TYPES = frozenset(
            {
                "array",
                "dict",
                "enum_body",
                "enum_entries",
                "array_literal",
                "dict_literal",
                # Match patterns mirror the literals, so they get the same layout.
                "array_pattern",
                "dict_pattern",
                "standalone_call",
                "getattr_call",
                "subscr_expr",
                "lambda",
                "postfix",
                "function_call_expr",
                "ternary",
                "primary",
                "preload_expr",
                "await_expr",
                # New grammar operator chain types
                "logic_or",
                "logic_and",
                "logic_not",
                "bit_shift",
                "power",
                "factor",
                "additive",
                "shift_expr",
                "mdr_expr",
                "arith_expr",
                "bitw_or",
                "bitw_xor",
                "bitw_and",
                "bit_and",
                "bit_or",
                "bit_xor",
                "content_test",
                "comparison",
                "and_test",
                "or_test",
                "sign",
                "bit_not",
            }
        )
        if multi_data not in KNOWN_MULTILINE_TYPES and not is_repetition_chain:
            # Token or unknown type — keep single-line output with prefix intact
            return _format_concrete_expression_to_single_line(
                expression, expression_context, indent_string
            )
        result = _format_foldable_to_multiple_lines(
            unwrapped, expression_context, indent_string, style or StyleConfig()
        )
        return result

    return _format_concrete_expression_to_single_line(
        expression, expression_context, indent_string
    )


def _format_token_as_line(token: Token, ec: ExpressionContext, indent) -> FormattedLine:
    """Format a single token as a line."""
    prefix = (
        f"{ec.prefix_string}{token.value}"
        if isinstance(ec.prefix_string, str)
        else ""  # type: ignore
    )
    suffix = ec.suffix_string if isinstance(ec.suffix_string, str) else ""
    return (ec.prefix_line, f"{indent}{prefix}{suffix}")


def _format_concrete_expression_to_single_line(
    expression, expression_context: ExpressionContext, indent_string: str
) -> FormattedLines:
    from .expression_to_str import expression_to_str as _etostr

    if isinstance(expression, Token):
        text = (
            f"{expression_context.prefix_string}{expression.value}"
            f"{expression_context.suffix_string}"  # type: ignore
        )
    else:
        data = (
            str(expression.data)
            if not hasattr(expression.data, "value")
            else expression.data.value  # type: ignore
        )
        try:
            expr_str = _etostr(expression)
            text = (
                f"{expression_context.prefix_string}{expr_str}"
                f"{expression_context.suffix_string}"  # type: ignore
            )
        except Exception:
            text = f"{expression_context.prefix_string}<{data}>{expression_context.suffix_string}"

    return [(expression_context.prefix_line, f"{indent_string}{text}")]


def _format_comma_separated_list(
    a_list, expression_context: ExpressionContext, indent_string: str, style=None
) -> FormattedLines:
    """Format a comma-separated list of expressions across multiple lines."""
    from .expression_to_str import standalone_expression_to_str as _sestr

    max_len = style.line_length if style else 100
    elements = []
    for item in a_list:
        from .expression_utils import is_any_comma as _iac

        if not _iac(item):
            text = _sestr(item)
            display_len = len(text.replace("\t", "    "))
            elements.append((text, display_len))

    # Check if we can fit on one line
    joined = ", ".join(e[0] for e in elements)
    total_len = (
        len(expression_context.prefix_string)
        + len(joined)
        + len(expression_context.suffix_string)  # type: ignore
    )
    total_len += len(indent_string.replace("\t", "    "))  # rough estimate

    if total_len <= max_len and len(elements) < 10:
        text = (
            f"{expression_context.prefix_string}{joined}"
            f"{expression_context.suffix_string}"  # type: ignore
        )
        return [(expression_context.prefix_line, f"{indent_string}{text}")]

    # Multi-line format
    lines: FormattedLines = []
    first = True
    for elem_text, _ in elements:
        if first and expression_context.prefix_string:
            first_line_text = (
                f"{indent_string}{expression_context.prefix_string}{elem_text},"
            )
            lines.append((expression_context.prefix_line, first_line_text))  # type: ignore
            first = False
        elif first:
            lines.append(
                (expression_context.prefix_line, f"{indent_string}{elem_text},")
            )
            first = False
        else:
            indent2 = "    " if "\t" not in indent_string else "\t\t"
            lines.append((None, f"{indent2}{elem_text},"))

    # Add trailing comma and close
    last_indent = "    " if "\t" not in indent_string else "\t\t"
    text = f"\n{last_indent}]  {expression_context.suffix_string}"  # type: ignore
    return lines


def _call_has_block_lambda_argument(call_suffix) -> bool:
    """True when one of a call's arguments is a lambda that must be a block of its own.

    Such an argument cannot share a line with anything: `expression_to_str` renders its
    body flat, so the call has to hand it to the expression layout instead. A collection
    that merely forces the multiline layout (a trailing comma, three entries) is not
    this: it still has an inline spelling.
    """
    return any(_has_inline_forcing_lambda(a) for a in _call_arg_nodes(call_suffix))


def _block_argument_lines(
    arg_nodes, first_line_no: Optional[int], inner_indent: str, style
) -> FormattedLines:
    """Lay a comma-separated argument/element list out one block per item.

    Used when an item cannot share a line with its siblings: a lambda whose body is a
    block, or any other item the multiline layout forces. Each item is laid out by the
    expression formatter and the separator is appended to the last line of the item's own
    block, which is where the comma belongs once the item spans several lines.
    """
    lines: FormattedLines = []
    for index, node in enumerate(arg_nodes):
        sep = "," if index < len(arg_nodes) - 1 else ""
        node_lines = _format_concrete_expression(
            node,
            ExpressionContext("", first_line_no, "", None),
            inner_indent,
            style,
        )
        last = len(node_lines) - 1
        for index_in_node, (line_no, text) in enumerate(node_lines):
            lines.append((line_no, f"{text}{sep}" if index_in_node == last else text))
    return lines


def _format_postfix_multi(  # noqa: C901
    postfix: Tree, expr_ctx: ExpressionContext, indent_string: str, style=None
) -> FormattedLines:
    """Format attribute/call chains across multiple lines.

    Uses a bottom-up strategy similar to the legacy formatter:
    1. First try compact single-line format (all segments concatenated).
       If this fits within line_length including prefix, output as-is.
    2. Check if base chain names fit at outer indent (without nested args expanded).
       If yes → keep dot chain inline, only break call/subscript boundaries that exceed limits.
    3. Otherwise wrap in parens with each segment on its own line.

    Produces output like:
        var x = (
            aaaaa
            . bbbbb[1]
            . cccc()
            . ddd
        )
    or for chains where only call args need breaking:
        var t = aaaaa.bbbbb.cccc[1].ddd()...nnnnnnnnn(
            1, 2, 3, 4, 5, 6, 7, 8, 9
        )
    """
    from .expression_to_str import (
        expression_to_str as _etostr,
        standalone_expression_to_str as _sestr,
    )
    from .expression_to_str import _call_suffix_to_str, _subscription_to_str

    first_line_no = expr_ctx.prefix_line  # type: ignore
    prefix_text = expr_ctx.prefix_string or ""  # type: ignore
    max_len = style.line_length if style else 100
    suffix_text = getattr(expr_ctx, "suffix_string", "") or ""

    def _is_attribute(node):
        if isinstance(node, Token):
            return False
        d = (
            str(getattr(node.data, "value", node.data))
            if not hasattr(node.data, "value")
            else node.data.value  # type: ignore
        )
        return d == "attribute"

    def _is_subscript(node):
        if isinstance(node, Token):
            return False
        d = (
            str(getattr(node.data, "value", node.data))
            if not hasattr(node.data, "value")
            else node.data.value  # type: ignore
        )
        return d in ("subscript",)

    def _is_call_suffix(node):
        if isinstance(node, Token):
            return False
        d = (
            str(getattr(node.data, "value", node.data))
            if not hasattr(node.data, "value")
            else node.data.value  # type: ignore
        )
        return d in ("call_suffix",)

    non_none_children = [c for c in getattr(postfix, "children", []) if c is not None]

    # Group postfix children into logical segments:
    # - First child (base name) is its own segment
    # - Subsequent .attr chains, and their following [subscript]/(call_suffix)
    #   are combined: e.g., ".map" + "(args)" → ".map(args)"
    segments = []  # List of lists of child nodes (each inner list = one segment)
    i = 0
    while i < len(non_none_children):
        seg_children = [non_none_children[i]]
        # Combine with any immediately following subscript or call_suffix
        while i + 1 < len(non_none_children) and (
            _is_subscript(non_none_children[i + 1])
            or _is_call_suffix(non_none_children[i + 1])
        ):
            seg_children.append(non_none_children[i + 1])
            i += 1
        segments.append(seg_children)
        i += 1

    # A plain `name(arg)` call: one segment, a bare base name, and no dot chain.
    # Reaching this formatter already means the single-line form overflows, so put
    # the argument on its own indented line. A single argument that itself must be
    # multiline (a collection with a trailing comma) is laid out in place.
    if len(segments) == 1 and not any(_is_attribute(s) for s in segments[0]):
        _seg = segments[0]
        _calls = [s for s in _seg if _is_call_suffix(s)]
        _bases = [s for s in _seg if not _is_call_suffix(s)]
        if len(_calls) == 1 and _bases and all(isinstance(b, Token) for b in _bases):
            _cs_str = _call_suffix_to_str(_calls[0])
            _inner = _cs_str[_cs_str.index("(") + 1 : _cs_str.rindex(")")].strip()
            _inner_indent = continuation_indent(indent_string)
            _inner_width = len(_inner_indent.replace("\t", "    "))
            _args = _call_arg_nodes(_calls[0])
            _base = prefix_text + "".join(str(b.value) for b in _bases)
            # A comment written inside the parentheses is laid out where it was written:
            # as its own line, or on the argument's line when it follows one. Each
            # argument keeps the author's own separator, since the comment follows it.
            _arg_layout = _collection_layout(_calls[0])
            if any(_k == "comment" for _k, _n, _c in _arg_layout):
                _arg_lines = [(first_line_no, f"{indent_string}{_base}(")]
                _prev_arg_line = _node_line(_calls[0])
                for _k, _n, _had_comma in _arg_layout:
                    if _k == "comment":
                        _ctext = _comment_layout_text(_n)
                        _cline = _node_line(_n)
                        if _arg_lines and _cline and _cline == _prev_arg_line:
                            _ln, _txt = _arg_lines[-1]
                            _arg_lines[-1] = (_ln, f"{_txt}  {_ctext}")
                        else:
                            _arg_lines.append((None, f"{_inner_indent}{_ctext}"))
                        continue
                    _arg_lines.append(
                        (
                            None,
                            f"{_inner_indent}{_etostr(_n)}{',' if _had_comma else ''}",
                        )
                    )
                    _prev_arg_line = _node_end_line(_n)
                _arg_lines.append((None, f"{indent_string}){suffix_text}"))
                return _arg_lines
            if len(_args) == 1 and _is_forced_multiline(_args[0], style):
                return [
                    (first_line_no, f"{indent_string}{_base}("),
                    *_format_concrete_expression(
                        _args[0],
                        ExpressionContext("", first_line_no, "", None),
                        _inner_indent,
                        style or StyleConfig(),
                    ),
                    (None, f"{indent_string}){suffix_text}"),
                ]
            # A single collection argument that overflows is laid out in place the same
            # way, rather than falling back to the paren-wrapped chain layout.
            if len(_args) == 1 and _inner and _inner[0] in "[{":
                return [
                    (first_line_no, f"{indent_string}{_base}("),
                    *_format_concrete_expression(
                        _args[0],
                        ExpressionContext("", first_line_no, "", None),
                        _inner_indent,
                        style or StyleConfig(),
                    ),
                    (None, f"{indent_string}){suffix_text}"),
                ]
            if (
                _inner
                and "," not in _inner
                and _inner[0] not in "[{"
                and _inner_width + len(_inner) <= max_len
            ):
                return [
                    (first_line_no, f"{indent_string}{_base}("),
                    (None, f"{_inner_indent}{_inner}"),
                    (None, f"{indent_string}){suffix_text}"),
                ]
            # Multiple arguments: break after `(` and put them on continuation lines.
            # A trailing comma on the last argument forces one argument per line (the
            # same rule as an array or a parameter list); otherwise the arguments share
            # one continuation line while that fits, which is the legacy layout. When it
            # does not fit they go one per line, because the guide's line length rule is
            # imperative - "Keep individual lines of code under 100 characters"; only its
            # "try to keep lines under 80 characters" is the soft one.
            if _args and _inner and _inner[0] not in "[{":
                _trailing = _has_trailing_comma_in_elements(_calls[0])
                _arg_lines = [(first_line_no, f"{indent_string}{_base}(")]
                if any(_is_forced_multiline(a, style) for a in _args):
                    # An argument that must be a block (a lambda with a multi-statement
                    # body, a collection carrying a trailing comma) cannot be joined onto
                    # a shared continuation line, so every argument gets its own block.
                    _arg_lines.extend(
                        _block_argument_lines(
                            _args, first_line_no, _inner_indent, style or StyleConfig()
                        )
                    )
                elif not _trailing:
                    _joined_args = ", ".join(_etostr(a) for a in _args)
                    if _inner_width + len(_joined_args) <= max_len:
                        # They share one continuation line while that fits.
                        _arg_lines.append((None, f"{_inner_indent}{_joined_args}"))
                    elif all(_inner_width + len(_etostr(a)) <= max_len for a in _args):
                        # One argument per line, each still fitting on its own.
                        for _i, _arg in enumerate(_args):
                            _comma = "," if _i < len(_args) - 1 else ""
                            _arg_lines.append(
                                (None, f"{_inner_indent}{_etostr(_arg)}{_comma}")
                            )
                    else:
                        # An argument is itself wider than a line, so flattening it cannot
                        # help: lay it out in place instead. The call's own parens give it
                        # the continuation it needs.
                        _arg_lines.extend(
                            _block_argument_lines(
                                _args,
                                first_line_no,
                                _inner_indent,
                                style or StyleConfig(),
                            )
                        )
                else:
                    for _i, _arg in enumerate(_args):
                        _comma = "," if (_i < len(_args) - 1 or _trailing) else ""
                        _arg_lines.append(
                            (None, f"{_inner_indent}{_etostr(_arg)}{_comma}")
                        )
                _arg_lines.append((None, f"{indent_string}){suffix_text}"))
                return _arg_lines

    # Check if any segment contains content that forces multiline (e.g.,
    # arrays with trailing commas)
    def _get_array_from_subscript(sub_node):
        """Unwrap through assignment/cast/ternary/type_test to find the
        actual array inside a subscript."""
        for c in getattr(sub_node, "children", []):
            if c is None or not isinstance(c, Tree):
                continue
            # Unwrap wrapper types to find the inner array
            unwrapped = c
            while unwrapped and isinstance(unwrapped, Tree):
                wd = str(getattr(unwrapped.data, "value", unwrapped.data))
                if wd in ("assignment", "cast", "ternary", "type_test"):
                    children_list = [
                        ch
                        for ch in getattr(unwrapped, "children", [])
                        if ch is not None
                    ]
                    unwrapped = children_list[0] if len(children_list) >= 1 else None
                elif wd in ("array", "array_literal"):
                    return unwrapped
                else:
                    break
        return None

    def _segment_needs_multiline(seg):
        """Check if any child in the segment requires forced multiline formatting."""
        for node in seg:
            if isinstance(node, Tree):
                d = str(getattr(node.data, "value", node.data))
                if d in (
                    "dict",
                    "dict_literal",
                    "enum_body",
                    "enum_entries",
                    "array",
                    "array_literal",
                ) and _is_forced_multiline(node, style):
                    # A literal base carrying a trailing comma forces the chain multiline.
                    return True
                if d == "call_suffix" and _has_trailing_comma_in_elements(node):
                    # A trailing comma in the arguments forces the chain multiline.
                    return True
                if d == "subscript":
                    arr = _get_array_from_subscript(node)
                    if arr is not None and (
                        _has_trailing_comma_in_elements(arr)
                        or _has_standalone_comment_children(arr)
                        or _count_array_elements(arr) > 5
                    ):
                        return True
        return False

    def _count_array_elements(arr_node):
        """Count the number of actual element expressions in an array literal."""
        count = 0
        for child in getattr(arr_node, "children", []):
            if isinstance(child, Tree) and _tree_data_str(child).lower() == "elements":
                for ec in getattr(child, "children", []):
                    if ec is not None and not (
                        isinstance(ec, Token)
                        and str(getattr(ec, "type", "")).upper() == "COMMA"
                    ):
                        count += 1
        return count

    has_forced_multiline_children = any(
        _segment_needs_multiline(seg) for seg in segments
    )

    # Compute compact representation (all segments concatenated via expression_to_str)
    compact_parts = []
    for seg in segments:
        comp = _etostr(seg[0])
        for s in seg[1:]:
            comp += _etostr(s)
        compact_parts.append(comp)

    compact_content = "".join(compact_parts)

    prefix_display = len(prefix_text.replace("\t", "    "))
    # The compact form is emitted as `indent_string + prefix + content`, so the
    # indent's display width counts too; leaving it out let a 104-column line through
    # a 100-column limit whenever the chain sat at indent level 1.
    indent_display = len(indent_string.replace("\t", "    "))
    full_line_width = (
        indent_display + prefix_display + len(compact_content.replace("\t", "    "))
    )

    # A call somewhere in the chain whose argument is a lambda with a block body cannot be
    # written on one line either: `expression_to_str` renders the body flat, so the
    # one-liner would not compile. Kept apart from `has_forced_multiline_children`, which
    # selects the paren-wrapped layout - this one wants the bottom-up layout below.
    has_block_lambda_arg = any(
        isinstance(s, Tree)
        and _tree_data_str(s) == "call_suffix"
        and _call_has_block_lambda_argument(s)
        for seg in segments
        for s in seg
    )

    if (
        full_line_width <= max_len
        and not has_forced_multiline_children
        and not has_block_lambda_arg
    ):
        # Compact representation fits - output as single line
        return [(first_line_no, f"{indent_string}{prefix_text}{compact_content}")]

    # ---- Pure-subscript chains (no attribute access) ----
    # Chains like `p[a][b]`, `p[[1, 2, 3,]]`, `p[1 + 1 + ...]` and `{dict}[x]` must NOT be
    # wrapped in outer parens. The base stays inline and each subscript's content expands
    # independently only where it would overflow (or is trailing-comma forced).
    _op_chain_types = frozenset(
        {
            "additive",
            "shift_expr",
            "mdr_expr",
            "arith_expr",
            "bitw_or",
            "bitw_xor",
            "bitw_and",
            "bit_and",
            "bit_or",
            "bit_xor",
            "content_test",
            "comparison",
            "and_test",
            "or_test",
        }
    )

    def _subscript_inner_node(sub_node):
        for c in getattr(sub_node, "children", []):
            if isinstance(c, Token):
                ttype = str(getattr(c, "type", "")).lower()
                if ttype in ("bracket_open", "bracket_close"):
                    continue
            if c is not None:
                return c
        return None

    def _unwrap_inner(node):
        from .expression_to_str import _unwrap_wrappers as _uwr

        return _uwr(node) if isinstance(node, Tree) else node

    def _array_element_texts(arr_node):
        texts = []
        for child in getattr(arr_node, "children", []):
            if child is None or isinstance(child, Token):
                continue
            if _tree_data_str(child) == "elements":
                for ec in getattr(child, "children", []):
                    if ec is None or (
                        isinstance(ec, Token)
                        and str(getattr(ec, "type", "")).upper() == "COMMA"
                    ):
                        continue
                    texts.append(_etostr(ec))
            else:
                texts.append(_etostr(child))
        return texts

    def _op_chain_terms(chain_node):
        terms = []
        pending = None
        for c in getattr(chain_node, "children", []):
            if isinstance(c, Token):
                tv = str(getattr(c, "value", "") or "")
                if tv and tv not in ("(", ")", ","):
                    pending = tv
            elif isinstance(c, Tree):
                txt = _etostr(c)
                if pending is not None:
                    terms.append(f"{pending} {txt}")
                    pending = None
                else:
                    terms.append(txt)
        return terms

    def _subscript_forced(sub_node):
        inner = _unwrap_inner(_subscript_inner_node(sub_node))
        if isinstance(inner, Tree) and _tree_data_str(inner) in (
            "array",
            "array_literal",
        ):
            from .expression_utils import has_trailing_comma as _htc

            try:
                return bool(_htc(inner))
            except Exception:
                return False
        return False

    def _plan_subscript_expansion(sub_node, inner_ind):
        inner = _unwrap_inner(_subscript_inner_node(sub_node))
        d_in = _tree_data_str(inner) if isinstance(inner, Tree) else ""
        if d_in in ("array", "array_literal"):
            from .expression_utils import has_trailing_comma as _htc

            elems = _array_element_texts(inner)
            try:
                has_tc = bool(_htc(inner))
            except Exception:
                has_tc = False
            if has_tc:
                content = [f"{inner_ind}{e}," for e in elems]
            else:
                content = [f"{inner_ind}{', '.join(elems)}"]
            return "[[", content, "]]"
        if d_in in _op_chain_types:
            terms = _op_chain_terms(inner)
            return "[", [f"{inner_ind}{t}" for t in terms], "]"
        return "[", [f"{inner_ind}{_etostr(inner)}"], "]"

    def _format_pure_subscript_chain():
        # A subscript's brackets hold a list, like an array's, so its items keep one level.
        inner_ind = collection_indent(indent_string)
        nodes_flat = [n for seg in segments for n in seg]
        base_node = nodes_flat[0]
        sub_nodes = [n for n in nodes_flat[1:] if _is_subscript(n)]

        base_data = _tree_data_str(base_node) if isinstance(base_node, Tree) else ""
        if base_data in (
            "dict",
            "dict_literal",
            "enum_body",
            "enum_entries",
            "array",
            "array_literal",
        ):
            # Multiline literal base: delegate to its own multiline formatter and attach
            # the (inline) subscripts to the closing bracket line.
            suffix_inline = "".join(_etostr(s) for s in sub_nodes)
            ec = ExpressionContext(
                prefix_text, first_line_no, suffix_inline, expr_ctx.suffix_line
            )
            if base_data in ("dict", "dict_literal", "enum_body", "enum_entries"):
                return _format_dict_multi(
                    base_node, ec, indent_string, style or StyleConfig()
                )
            return _format_array_multi(
                base_node, ec, indent_string, style or StyleConfig()
            )

        result_lines = [
            (first_line_no, f"{indent_string}{prefix_text}{_etostr(base_node)}")
        ]

        def _set_last(text):
            line_no, _ = result_lines[-1]
            result_lines[-1] = (line_no, text)

        for sub in sub_nodes:
            inline = _etostr(sub)
            if (
                not _subscript_forced(sub)
                and len((result_lines[-1][1] + inline).replace("\t", "    ")) <= max_len
            ):
                _set_last(result_lines[-1][1] + inline)
                continue
            open_txt, content_lines, close_txt = _plan_subscript_expansion(
                sub, inner_ind
            )
            _set_last(result_lines[-1][1] + open_txt)
            result_lines.extend((None, c) for c in content_lines)
            result_lines.append((None, f"{indent_string}{close_txt}"))
        return result_lines

    _base0 = [n for seg in segments for n in seg][0]
    _base0_data = _tree_data_str(_base0) if isinstance(_base0, Tree) else ""
    _is_literal_base = _base0_data in (
        "dict",
        "dict_literal",
        "enum_body",
        "enum_entries",
        "array",
        "array_literal",
    )
    if (
        not any(_is_attribute(s) for seg in segments for s in seg)
        and not any(_is_call_suffix(s) for seg in segments for s in seg)
        and any(_is_subscript(s) for seg in segments for s in seg)
        and (_is_literal_base or "\n" not in _etostr(_base0))
        and (full_line_width > max_len or has_forced_multiline_children)
    ):
        return _format_pure_subscript_chain()

    # If there are arrays with trailing commas inside subscripts, we MUST
    # use paren-wrapped multiline format.
    if has_forced_multiline_children:
        # The chain's segments are continuation lines of the `(` this wraps them in, so they get
        # the two levels; the items of a subscript's own brackets, and its closing bracket, hang
        # off that indent by one, which is the single level the guide gives a collection.
        inner_indent2 = continuation_indent(indent_string)
        elem_indent = inner_indent2 + "\t"
        bracket_close_indent = inner_indent2
        result_lines = [(first_line_no, f"{indent_string}{prefix_text}(")]

        # Build a set of subscripts that need expanding (contain arrays with
        # trailing commas/comments/many elements)
        expanding_subscripts = set()
        for seg in segments:
            for s in seg:
                if (
                    isinstance(s, Tree)
                    and str(getattr(s.data, "value", s.data)) == "subscript"
                ):
                    arr = _get_array_from_subscript(s)
                    if arr is not None and (
                        _has_trailing_comma_in_elements(arr)
                        or _has_standalone_comment_children(arr)
                        or _count_array_elements(arr) > 5
                    ):
                        expanding_subscripts.add(id(s))

        for i, seg in enumerate(segments):
            # Check if this segment has any expanding subscript
            seg_has_expanding = any(
                isinstance(s, Tree)
                and str(getattr(s.data, "value", s.data)) == "subscript"
                and id(s) in expanding_subscripts
                for s in seg
            )

            if seg_has_expanding:
                # Extract the name/identifier for this segment (first
                # non-subscript token or attr name)
                prefix_content = ""
                for s in seg:
                    if isinstance(s, Token):
                        ttype = str(getattr(s, "type", "")).lower()
                        if (
                            ttype not in ("bracket_open", "bracket_close")
                            and ttype != "period"
                        ):
                            val = getattr(s, "value", "") or ""
                            prefix_content = str(val)
                            break
                    elif isinstance(s, Tree):
                        d = str(getattr(s.data, "value", s.data))
                        if d == "attribute":
                            for ch in getattr(s, "children", []) or []:
                                if isinstance(ch, Tree):
                                    cd = str(getattr(ch.data, "value", ch.data))
                                    if cd == "attribute_name":
                                        for sub_ch in getattr(ch, "children", []) or []:
                                            if isinstance(sub_ch, Token):
                                                val = str(
                                                    getattr(sub_ch, "value", "") or ""
                                                )
                                                if val:
                                                    prefix_content = val
                            break

                # Build the leading part: either "base[[" (first segment) or ".attr[[]" (subsequent)
                # Subscript wraps a nested array_literal, so we need [[
                if i == 0:
                    leading = f"{inner_indent2}{prefix_content}[["
                else:
                    leading = f"{inner_indent2}.{prefix_content}[["
                result_lines.append((None, leading))

                # Add all array element lines for each expanding subscript in this segment
                for s in seg:
                    if (
                        isinstance(s, Tree)
                        and str(getattr(s.data, "value", s.data)) == "subscript"
                        and id(s) in expanding_subscripts
                    ):
                        arr = _get_array_from_subscript(s)
                        if arr is not None:
                            all_children = list(getattr(arr, "children", []))
                            for child in all_children:
                                if isinstance(child, Token) or child is None:
                                    continue
                                cd = _tree_data_str(child)
                                if cd == "elements":
                                    element_children = list(
                                        getattr(child, "children", [])
                                    )
                                    for idx, ec in enumerate(element_children):
                                        if isinstance(ec, Token):
                                            continue
                                        if ec is None:
                                            continue
                                        elem_text = (
                                            _etostr(ec)
                                            if isinstance(ec, Tree)
                                            else str(getattr(ec, "value", ""))
                                        )
                                        # Trailing comma only on non-last elements
                                        has_trailing_comma_on_last = (
                                            idx == len(element_children) - 1
                                        )
                                        comma = (
                                            ","
                                            if not has_trailing_comma_on_last
                                            else ""
                                        )
                                        result_lines.append(
                                            (None, f"{elem_indent}{elem_text}{comma}")
                                        )

                # Close with double brackets for the nested subscript structure [[]]
                result_lines.append((None, f"{bracket_close_indent}]]"))
            else:
                # A forced-multiline literal base (a collection carrying a trailing
                # comma) is laid out exploded rather than flattened back onto one line.
                base = seg[0] if seg else None
                base_data = (
                    str(getattr(base.data, "value", base.data))
                    if isinstance(base, Tree)
                    else ""
                )
                if (
                    i == 0
                    and len(seg) == 1
                    and base_data
                    in (
                        "dict",
                        "dict_literal",
                        "enum_body",
                        "enum_entries",
                        "array",
                        "array_literal",
                    )
                    and _is_forced_multiline(base, style)
                ):
                    result_lines.extend(
                        _format_concrete_expression(
                            base,
                            ExpressionContext("", first_line_no, "", None),
                            inner_indent2,
                            style or StyleConfig(),
                        )
                    )
                    continue
                # A call whose arguments carry a trailing comma is exploded in place:
                # the name and `(` on the segment line, one argument per line, `)` back
                # at the segment indent.
                tc_call = next(
                    (
                        s
                        for s in seg
                        if isinstance(s, Tree)
                        and _tree_data_str(s) == "call_suffix"
                        and _has_trailing_comma_in_elements(s)
                    ),
                    None,
                )
                if tc_call is not None:
                    # `_etostr` of an attribute carries its leading dot, which the
                    # `.` separator below supplies, so drop it.
                    name_text = "".join(
                        _etostr(s) for s in seg if not _is_call_suffix(s)
                    )
                    name_text = (
                        name_text[1:] if name_text.startswith(".") else name_text
                    )
                    display = f".{name_text}" if i > 0 else name_text
                    # The arguments continue the call's own line, so they get the guide's two
                    # levels on top of the segment's indent.
                    arg_indent = continuation_indent(inner_indent2)
                    result_lines.append((None, f"{inner_indent2}{display}("))
                    for arg in _call_arg_nodes(tc_call):
                        result_lines.append((None, f"{arg_indent}{_etostr(arg)},"))
                    result_lines.append((None, f"{inner_indent2})"))
                    continue
                # Non-expanding segment - use flat output
                parts = []
                for s in seg:
                    if isinstance(s, Token):
                        ttype = str(getattr(s, "type", "")).lower()
                        if (
                            ttype not in ("bracket_open", "bracket_close")
                            and ttype != "period"
                        ):
                            val = getattr(s, "value", "") or ""
                            parts.append(str(val))
                    elif isinstance(s, Tree):
                        d = str(getattr(s.data, "value", s.data))
                        if d == "attribute":
                            for ch in getattr(s, "children", []) or []:
                                if isinstance(ch, Tree):
                                    cd = str(getattr(ch.data, "value", ch.data))
                                    if cd == "attribute_name":
                                        for sub_ch in getattr(ch, "children", []) or []:
                                            if isinstance(sub_ch, Token):
                                                val = str(
                                                    getattr(sub_ch, "value", "") or ""
                                                )
                                                if val:
                                                    parts.append(val)
                        elif d == "call_suffix":
                            parts.append(_etostr(s))
                        else:
                            parts.append(_etostr(s) if s is not None else "")

                seg_str = "".join(parts)
                if i == 0:
                    result_lines.append((None, f"{inner_indent2}{seg_str}"))
                else:
                    display_seg = f".{seg_str}"
                    result_lines.append((None, f"{inner_indent2}{display_seg}"))

        # Append suffix
        if suffix_text:
            result_lines.append((None, f"{indent_string}){suffix_text}".rstrip()))
        else:
            result_lines.append((None, f"{indent_string})"))
        return result_lines

    inner_indent = continuation_indent(indent_string)

    def _format_segment_content(seg):
        """Format a segment's content using expression_to_str."""
        parts = [_etostr(s) for s in seg]
        return "".join(parts)

    # Check if the dot chain base names (without expanded call args) fit at outer indent.
    # This determines if we should use legacy-style inline chain with only problematic parts broken.
    def _compute_base_segment(seg):
        """Compute just the segment name without expanded nested content."""
        parts = []
        for sc in seg:
            if isinstance(sc, Token):  # Base name like 'aaaaa'
                parts.append(str(sc.value))
            elif _is_attribute(sc):
                # Extract attribute names from tree - Lark Earley may report empty .type,
                # so we extract identifiers by value filtering out symbols/operators
                all_names: List[str] = []
                _extract_all_tokens(sc, all_names)
                parts.append("." + ".".join(all_names))
            elif _is_subscript(sc):
                # Only the opening bracket participates in the base-chain width estimate;
                # the subscript content is what gets broken out onto its own lines when
                # the segment does not fit (mirrors the call_suffix "(" case above).
                parts.append("[")
            elif _is_call_suffix(sc):
                parts.append("(")
            else:
                parts.append(_etostr(sc))
        return "".join(parts)

    base_segments = [_compute_base_segment(seg) for seg in segments]

    def _has_segment_real_content(seg):
        """Check if a segment has real suffix content that needs multiline formatting.

        Returns False for segments with only empty call suffixes like .instantiate() --
        these should stay inline even when combined would exceed line length, to avoid
        producing garbled output with blank lines between parens.
        """
        call_trees = [
            s
            for s in seg
            if isinstance(s, Tree)
            and str(getattr(s.data, "value", getattr(s.data, "", ""))) == "call_suffix"
        ]
        sub_trees = [
            s
            for s in seg
            if isinstance(s, Tree)
            and str(getattr(s.data, "value", getattr(s.data, "", ""))) == "subscript"
        ]

        # Check call suffixes - return True only if any has real args content
        for cs in call_trees:
            parts = []
            for c in getattr(cs, "children", []):
                if c is None or isinstance(c, Token):
                    ttype = (
                        str(getattr(c, "type", "")).lower()
                        if hasattr(c, "type")
                        else ""
                    )
                    if ttype not in ("parenthesis_open", "parenthesis_close"):
                        parts.append(str(getattr(c, "value", _etostr(c))))
                elif isinstance(c, Tree):
                    parts.append(_etostr(c))
            cs_content = "".join(parts)
            if cs_content.strip():  # Has real arguments
                return True

        # Check subscript suffixes - always has content (even if empty [])
        if sub_trees:
            return True

        return False

    # Build the full dot chain using base names only (without expanded call args)
    first_base = base_segments[0] if base_segments else ""
    rest_bases = (
        ["." + s.lstrip(".") for s in base_segments[1:]]
        if len(base_segments) > 1
        else []
    )
    full_base_chain = "".join([first_base] + rest_bases)

    # Check if base chain (with prefix, without expanded args) fits at outer indent
    base_width = prefix_display + len(full_base_chain.replace("\t", "    "))

    # Legacy bottom-up hybrid: keep the dot chain inline on one line and break only the
    # sub-parts of segments that exceed limits (call args or subscript content). It is
    # used only when EVERY resulting line fits; otherwise fall back to paren-wrapped
    # per-segment lines (the "standard" multiline form).
    def _format_hybrid_inline():
        result_lines = []
        chain_parts = [prefix_text]  # Start building first line content

        for i, seg in enumerate(segments):
            seg_str = _format_segment_content(seg)

            # Check if appending this segment would exceed line_length. The emitted
            # line is `indent_string + chain_parts + seg_str`, so the indent counts;
            # the first segment is tested too, so a bare `f(...)` whose arguments
            # overflow is broken after `(` instead of falling through to the
            # paren-wrapped layout.
            test_line = "".join(chain_parts) + seg_str
            test_width = len(f"{indent_string}{test_line}".replace("\t", "    "))

            # A call carrying a lambda with a block body has to break even though its
            # flattened text fits: `seg_str` is the flattened form, and there is no
            # length at which a block lambda can share the line.
            _seg_has_block_lambda = any(
                isinstance(s, Tree)
                and _tree_data_str(s) == "call_suffix"
                and _call_has_block_lambda_argument(s)
                for s in seg
            )

            if test_width <= max_len and not _seg_has_block_lambda:
                chain_parts.append(seg_str)
            else:
                # This segment's content exceeds limits - break here.
                # BUT: skip empty suffixes (like .instantiate()) — they should stay inline
                # even when combined with the prefix would exceed line_length. Only break for
                # segments that have real content to format multiline (non-empty call args,
                # subscript content, etc.)

                # Check if this segment has any non-empty suffix content
                seg_has_real_suffix = _has_segment_real_content(seg)

                if not seg_has_real_suffix:
                    # Empty suffix — just append inline despite exceeding length.
                    # The alternative is garbled empty-line output, which is worse.
                    chain_parts.append(seg_str)
                    continue

                # Build the opening line with all segments up to this point + name of breaking seg
                open_chain = "".join(chain_parts)  # Everything before breaking seg

                # Get base name from breaking segment (skip call_suffix/subscript)
                for s in seg:
                    if isinstance(s, Tree):
                        d = str(getattr(s.data, "value", getattr(s.data, "", "")))
                        if d in ("call_suffix", "subscript"):
                            break
                        elif d in ("attribute", "attribute_name"):
                            all_names: List[str] = []
                            _extract_all_tokens(s, all_names)
                            open_chain += "." + ".".join(all_names)
                    elif isinstance(s, Token):
                        open_chain += str(s.value)

                # Handle the multiline content of the breaking segment
                call_trees = [
                    s
                    for s in seg
                    if isinstance(s, Tree)
                    and str(getattr(s.data, "value", getattr(s.data, "", "")))
                    == "call_suffix"
                ]
                sub_trees = [
                    s
                    for s in seg
                    if isinstance(s, Tree)
                    and str(getattr(s.data, "value", getattr(s.data, "", "")))
                    == "subscript"
                ]

                if call_trees:
                    cs_str = _call_suffix_to_str(call_trees[0])  # e.g., "(1, 2, 3)"
                    paren_idx = cs_str.index("(")
                    after_paren = cs_str[paren_idx + 1 :].rstrip(")")

                    result_lines.append(
                        (
                            first_line_no,
                            f"{indent_string}{open_chain}{cs_str[:paren_idx+1]}",
                        )
                    )  # Line with opening (
                    if _call_has_block_lambda_argument(call_trees[0]):
                        # An argument that must be a block (a lambda with a
                        # multi-statement body) is laid out as one, rather than folded
                        # into a single flattened continuation line.
                        result_lines.extend(
                            _block_argument_lines(
                                _call_arg_nodes(call_trees[0]),
                                first_line_no,
                                inner_indent,
                                style or StyleConfig(),
                            )
                        )
                    elif after_paren.strip():
                        result_lines.append((None, f"{inner_indent}{after_paren}"))
                    else:
                        result_lines.append((None, f"{inner_indent}"))
                    close_char = ")"
                elif sub_trees:
                    # Preserve outer parens around inner expressions (e.g., [(1+2)]) for readability
                    from .expression_to_str import _unwrap_wrappers as _uwrappers

                    inner_expr_node = [
                        c
                        for c in getattr(sub_trees[0], "children", [])
                        if not (
                            isinstance(c, Token)
                            and str(getattr(c, "type", "")).lower()
                            in ("bracket_open", "bracket_close")
                        )
                    ]
                    if inner_expr_node:
                        inner_unwrapped = _uwrappers(inner_expr_node[0])
                        # Check for paren-wrapped additive/arith chains - preserve parens
                        d_inner = ""
                        is_paren_wrapped_additive = False

                        # Determine the innermost expression type
                        test_node = inner_unwrapped
                        if isinstance(test_node, Tree):
                            d_inner = _tree_data_str(test_node)
                            # Check if it's a primary with paren-wrapped content (new grammar)
                            if d_inner == "primary":
                                prim_children = [
                                    c
                                    for c in getattr(test_node, "children", [])
                                    if c is not None
                                ]
                                if len(prim_children) >= 3:
                                    first_ch = prim_children[0]
                                    last_ch = prim_children[-1]
                                    inner_ch = (
                                        prim_children[len(prim_children) // 2]
                                        if len(prim_children) == 3
                                        else prim_children[1]
                                    )
                                    if (
                                        isinstance(first_ch, Token)
                                        and str(getattr(first_ch, "type", "")).upper()
                                        == "PARENTHESIS_OPEN"
                                        and isinstance(last_ch, Token)
                                        and str(getattr(last_ch, "type", "")).upper()
                                        == "PARENTHESIS_CLOSE"
                                    ):
                                        # Check inner content of parens is additive/arith chain
                                        inner_data = _tree_data_str(inner_ch)
                                        if inner_data in (
                                            "additive",
                                            "shift_expr",
                                            "mdr_expr",
                                            "arith_expr",
                                            "bitw_or",
                                            "bitw_xor",
                                            "bitw_and",
                                            "bit_and",
                                            "bit_or",
                                            "bit_xor",
                                        ):
                                            is_paren_wrapped_additive = True
                            elif d_inner in (
                                "additive",
                                "shift_expr",
                                "mdr_expr",
                                "arith_expr",
                                "bitw_or",
                                "bitw_xor",
                                "bitw_and",
                                "bit_and",
                                "bit_or",
                                "bit_xor",
                                "content_test",
                                "comparison",
                                "and_test",
                                "or_test",
                            ):
                                is_paren_wrapped_additive = True

                        inner_str = _etostr(inner_unwrapped)
                        if is_paren_wrapped_additive and inner_str:
                            # Preserve paren wrapping: [(content)]
                            sub_inner_content = f"({inner_str})"
                        else:
                            sub_inner_content = inner_str or ""
                    else:
                        sub_inner_content = ""

                    if sub_inner_content.strip():
                        # Check if content contains parens - expand to multiline
                        if sub_inner_content.startswith(
                            "("
                        ) and sub_inner_content.endswith(")"):
                            inner_paren_content = sub_inner_content[1:-1].strip()
                            result_lines.append(
                                (first_line_no, f"{indent_string}{open_chain}[(")
                            )
                            result_lines.append(
                                (None, f"{inner_indent}{inner_paren_content}")
                            )
                            close_char = ")]"
                        else:
                            result_lines.append(
                                (first_line_no, f"{indent_string}{open_chain}[")
                            )
                            result_lines.append(
                                (None, f"{inner_indent}{sub_inner_content}")
                            )
                            close_char = "]"
                    else:
                        result_lines.append(
                            (first_line_no, f"{indent_string}{open_chain}[")
                        )
                        result_lines.append((None, f"{inner_indent}"))
                        close_char = "]"
                else:
                    close_char = ""

                if suffix_text and close_char:
                    close_char += suffix_text.rstrip()
                result_lines.append((None, f"{indent_string}{close_char}"))

                # Format remaining segments after the break as multiline
                for i2 in range(i + 1, len(segments)):
                    s_str = _format_segment_content(segments[i2])
                    result_lines.append((None, f"{inner_indent}{s_str}"))

                return result_lines

        # Every segment fit inline, so there is nothing to break here. Return the whole
        # chain; the caller rejects it when the line overflows and falls back to the
        # paren-wrapped layout. Returning only the prefix dropped the entire chain and
        # emitted `if not` with no condition.
        return [
            (
                first_line_no,
                f"{indent_string}{''.join(chain_parts)}{suffix_text}".rstrip(),
            )
        ]

    # Legacy rule: when a chain contains a lambda in a NON-last element and the last
    # element is a call, the whole chain is paren-wrapped per segment rather than the
    # inline hybrid form. (E.g. `x.map(func(a): return a).filter(func(h): ...)`.)
    def _segment_has_lambda(seg):
        return any(isinstance(s, Tree) and _find_lambda_arg(s) is not None for s in seg)

    _force_paren_for_lambda = (
        any(_segment_has_lambda(seg) for seg in segments[:-1])
        and bool(segments)
        and any(_is_call_suffix(s) for s in segments[-1])
    )

    # Legacy: a chain containing a lambda always takes the bottom-up layout and skips
    # the "every line fits" test, because a lambda body cannot be reflowed to suit the
    # line length. The exemption is for the lambda's own lines: when the chain's *base*
    # line is itself too wide, nothing about the lambda excuses it, and the per-segment
    # layout below is the one that can bring it under the limit - which is what the
    # 112-column `x.aaaa..hhhhh(func(): return 1)` needs.
    _chain_has_lambda = any(_segment_has_lambda(seg) for seg in segments)

    if base_width <= max_len or _chain_has_lambda:
        _hybrid_lines = _format_hybrid_inline()
        _base_line_fits = bool(_hybrid_lines) and (
            len(_hybrid_lines[0][1].replace("\t", "    ")) <= max_len
        )
        if not _force_paren_for_lambda and (
            (_chain_has_lambda and _base_line_fits)
            or all(len(_l.replace("\t", "    ")) <= max_len for _, _l in _hybrid_lines)
        ):
            return _hybrid_lines

    # Base chain doesn't fit → wrap in parens with per-segment lines.
    result_lines = [(first_line_no, f"{indent_string}{prefix_text}(")]

    for i, seg in enumerate(segments):
        if i == 0:
            # A base that is a parenthesized expression which must be a block of its own
            # (a lambda with a multi-statement body) keeps its parentheses, with the
            # opening one glued to the lambda header and the closing one to the last
            # line of the body.
            _base_block = None
            if len(seg) == 1 and isinstance(seg[0], Tree):
                from .expression_to_str import _paren_wrapped_primary_inner

                _inner = _paren_wrapped_primary_inner(seg[0])
                if _inner is not None and _is_forced_multiline(_inner, style):
                    _base_block = _inner
            if _base_block is not None:
                _base_lines = _format_concrete_expression(
                    _base_block,
                    ExpressionContext("(", first_line_no, "", None),
                    inner_indent,
                    style or StyleConfig(),
                )
                if _base_lines:
                    _last = len(_base_lines) - 1
                    for _k, (_ln, _txt) in enumerate(_base_lines):
                        result_lines.append((_ln, f"{_txt})" if _k == _last else _txt))
                    continue
            result_lines.append((None, f"{inner_indent}{_format_segment_content(seg)}"))
            continue

        # For non-first segments, check if call/subscript suffix needs multiline expansion.
        seg_str = _format_segment_content(seg)
        display_seg = seg_str[1:] if seg_str.startswith(".") else seg_str

        # Check if the segment has call_suffix with too-long args
        call_trees = [
            s
            for s in seg
            if isinstance(s, Tree)
            and str(getattr(s.data, "value", getattr(s.data, "", ""))) == "call_suffix"
        ]
        sub_trees = [
            s
            for s in seg
            if isinstance(s, Tree)
            and str(getattr(s.data, "value", getattr(s.data, "", ""))) == "subscript"
        ]

        # Extract the attribute name prefix (everything before suffix)
        attr_parts = []
        for s in seg:
            if _is_attribute(s):
                names: List[str] = []
                _extract_attr_names(s, names)
                attr_parts.extend(["."] + names)
            elif isinstance(s, Token):
                ttype = str(getattr(s, "type", "")).upper()
                if ttype not in ("BRACKET_OPEN", "BRACKET_CLOSE", "PERIOD"):
                    val = str(getattr(s, "value", "") or "")
                    if val:
                        attr_parts.append(val)

        prefix_str = "".join(attr_parts)  # e.g., '. capitalize'
        indent_line = f"{inner_indent}.{display_seg}"
        indent_width = len(indent_line.replace("\t", "    "))
        available_space = max_len - len(inner_indent.replace("\t", "    "))

        close_char = None
        expanded = False

        if call_trees:
            _lambda_arg = _find_lambda_argument(call_trees[0])
            if _lambda_arg is not None and indent_width > max_len:
                # A long lambda passed as the sole call argument gets its own block
                # layout; the call's `(` / `)` stay on the segment-indent lines.
                _call_display = (
                    prefix_str[1:] if prefix_str.startswith(".") else prefix_str
                )
                result_lines.append((None, f"{inner_indent}.{_call_display}("))
                # The lambda continues the call's own line, so it sits two levels in from the
                # segment; its body is a block and takes one more from its header.
                result_lines.extend(
                    _format_lambda_multi(
                        _lambda_arg,
                        ExpressionContext("", first_line_no, "", expr_ctx.suffix_line),
                        continuation_indent(inner_indent),
                        style or StyleConfig(),
                    )
                )
                result_lines.append((None, f"{inner_indent})"))
                expanded = True
            else:
                cs_str = _call_suffix_to_str(call_trees[0])
                paren_idx = cs_str.index("(")
                after_paren = cs_str[paren_idx + 1 :].rstrip(")")
                content_width = (
                    len(after_paren.replace("\t", "    ")) if after_paren.strip() else 0
                )

                # Count arguments to determine if multiline is needed
                arg_count = after_paren.count(",") + 1 if after_paren.strip() else 0

                # Expand args multiline when they exceed available space, have many
                # arguments (>5), or the segment line is already too long to fit.
                if (after_paren.strip() and "," in after_paren) and (
                    content_width > available_space
                    or arg_count > 5
                    or indent_width > max_len
                ):
                    _call_display = (
                        prefix_str[1:] if prefix_str.startswith(".") else prefix_str
                    )
                    result_lines.append((None, f"{inner_indent}.{_call_display}("))
                    # Lay each argument out through the expression formatter. Splitting the
                    # flattened argument string on commas broke any argument that itself
                    # contains one (a dict or a nested call) into pieces.
                    _arg_nodes = _call_arg_nodes(call_trees[0])
                    # The call's arguments continue the call's own line, so they get the two
                    # levels the guide asks for, measured from that segment.
                    _arg_indent = continuation_indent(inner_indent)
                    for _ai, _arg in enumerate(_arg_nodes):
                        _sep = "," if _ai < len(_arg_nodes) - 1 else ""
                        _arg_lines = _format_concrete_expression(
                            _arg,
                            ExpressionContext("", first_line_no, "", None),
                            _arg_indent,
                            style or StyleConfig(),
                        )
                        for _j, (_ln, _txt) in enumerate(_arg_lines):
                            result_lines.append(
                                (
                                    None,
                                    f"{_txt}{_sep}"
                                    if _j == len(_arg_lines) - 1
                                    else _txt,
                                )
                            )
                    close_char = ")"
                    expanded = True
                else:
                    result_lines.append((None, indent_line))
        elif sub_trees and indent_width <= max_len:
            from .expression_to_str import _unwrap_wrappers as _uwrappers

            sub_inner_node = [
                c
                for c in getattr(sub_trees[0], "children", [])
                if not (
                    isinstance(c, Token)
                    and str(getattr(c, "type", "")).lower()
                    in ("bracket_open", "bracket_close")
                )
            ]
            inner_sub_content = ""
            is_paren_wrapped = False

            if sub_inner_node:
                unwrapped = _uwrappers(sub_inner_node[0])
                d_unwrapped = _tree_data_str(unwrapped)
                # Check for paren-wrapped additive/arith chains in new grammar
                if isinstance(unwrapped, Tree) and d_unwrapped == "primary":
                    prim_children = [
                        c for c in getattr(unwrapped, "children", []) if c is not None
                    ]
                    if len(prim_children) >= 3:
                        first_ch = prim_children[0]
                        last_ch = prim_children[-1]
                        inner_ch = (
                            prim_children[len(prim_children) // 2]
                            if len(prim_children) == 3
                            else prim_children[1]
                        )
                        if (
                            isinstance(first_ch, Token)
                            and str(getattr(first_ch, "type", "")).upper()
                            == "PARENTHESIS_OPEN"
                            and isinstance(last_ch, Token)
                            and str(getattr(last_ch, "type", "")).upper()
                            == "PARENTHESIS_CLOSE"
                        ):
                            inner_data = _tree_data_str(inner_ch)
                            if inner_data in (
                                "additive",
                                "shift_expr",
                                "mdr_expr",
                                "arith_expr",
                                "bitw_or",
                                "bitw_xor",
                                "bitw_and",
                                "bit_and",
                                "bit_or",
                                "bit_xor",
                            ):
                                is_paren_wrapped = True
                                inner_sub_content = _etostr(
                                    inner_ch
                                )  # Get the additive content without parens
                else:
                    inner_sub_content = _etostr(unwrapped)

            if not inner_sub_content and sub_inner_node:
                inner_sub_content = _etostr(sub_inner_node[0])

            if inner_sub_content.strip():
                content_width = len(inner_sub_content.replace("\t", "    "))
                # Check if content is paren-wrapped - handle specially
                has_parens = inner_sub_content.startswith(
                    "("
                ) and inner_sub_content.endswith(")")
                if content_width > available_space or is_paren_wrapped:
                    if is_paren_wrapped:
                        result_lines.append((None, f"{inner_indent}{prefix_str}[("))
                        result_lines.append(
                            (None, f"{inner_indent}\t\t{inner_sub_content}")
                        )
                        close_char = ")]"
                    else:
                        result_lines.append((None, f"{inner_indent}{prefix_str}["))
                        result_lines.append(
                            (None, f"{inner_indent}\t{inner_sub_content}")
                        )
                        close_char = "]"
                    expanded = True
                elif has_parens and content_width > available_space // 2:
                    # Partially long paren-wrapped - still expand
                    inner_paren = inner_sub_content[1:-1].strip()
                    result_lines.append(
                        (None, f"{inner_indent}{prefix_str}[({inner_paren})")
                    )
                else:
                    result_lines.append((None, indent_line))
            else:
                result_lines.append((None, indent_line))
        else:
            # No suffix or suffix too short — just output normally
            result_lines.append((None, f"{inner_indent}.{display_seg}"))

        if close_char and expanded:
            result_lines.append((None, f"{inner_indent}{close_char}"))

    # Append suffix
    if suffix_text:
        result_lines.append((None, f"{indent_string}){suffix_text}".rstrip()))
    else:
        result_lines.append((None, f"{indent_string})"))
    return result_lines


def _format_foldable_to_multiple_lines(  # noqa: C901
    expression: Tree,
    expression_context: ExpressionContext,
    indent_string: str,
    style=None,
) -> FormattedLines:
    """Format a foldable expression across multiple lines."""

    def make_expr_ctx(prefix: str, suffix: str = "") -> ExpressionContext:
        return ExpressionContext(
            f"{expression_context.prefix_string}{prefix}",
            expression_context.prefix_line,
            f"{suffix}{expression_context.suffix_string}",  # type: ignore
            expression_context.suffix_line,
        )

    data = _tree_data_str(expression)

    handlers_map = {
        # Old grammar types
        "array": lambda: _format_array_multi(
            expression, make_expr_ctx("["), indent_string, style or StyleConfig()
        ),
        "dict": lambda: _format_dict_multi(
            expression, make_expr_ctx("{"), indent_string, style or StyleConfig()
        ),
        "enum_body": lambda: _format_dict_multi(
            expression, make_expr_ctx("{"), indent_string, style or StyleConfig()
        ),
        # New grammar (Earley) types
        # Note: use empty prefix because array_literal has its own bracket tokens as children
        "array_literal": lambda: _format_array_multi(
            expression, make_expr_ctx(""), indent_string, style or StyleConfig()
        ),
        "dict_literal": lambda: _format_dict_multi(
            expression, make_expr_ctx(""), indent_string, style or StyleConfig()
        ),
        # Match patterns have the same shape as the literals.
        "array_pattern": lambda: _format_array_multi(
            expression, make_expr_ctx(""), indent_string, style or StyleConfig()
        ),
        "dict_pattern": lambda: _format_dict_multi(
            expression, make_expr_ctx(""), indent_string, style or StyleConfig()
        ),
        "enum_entries": lambda: _format_dict_multi(
            expression, make_expr_ctx(""), indent_string, style or StyleConfig()
        ),
        # Other types
        "standalone_call": lambda: _format_call_multi(
            expression, expression_context, indent_string, style or StyleConfig()
        ),
        "preload_expr": lambda: _format_preload_multi(
            expression, expression_context, indent_string, style or StyleConfig()
        ),
        "await_expr": lambda: _format_await_multi(
            expression, expression_context, indent_string, style or StyleConfig()
        ),
        "getattr_call": lambda: _format_getattr_call_multi(
            expression, expression_context, indent_string, style or StyleConfig()
        ),
        "subscr_expr": lambda: _format_subscription_multi(
            expression, make_expr_ctx(""), indent_string, style
        ),
        "lambda": lambda: _format_lambda_multi(
            expression, make_expr_ctx(""), indent_string, style or StyleConfig()
        ),
        "postfix": lambda: globals()["_format_postfix_multi"](
            expression, expression_context, indent_string, style or StyleConfig()
        ),
        # A bare `name(args)` statement is its own rule, but lays out like a postfix.
        "function_call_expr": lambda: globals()["_format_postfix_multi"](
            expression, expression_context, indent_string, style or StyleConfig()
        ),
    }

    # Handle intermediate expression types (additive, bit_shift, mdr_expr, etc.)
    is_repetition_chain = (
        data in ("cast", "type_test", "asless_type_test", "actual_type_cast")
        and len([c for c in getattr(expression, "children", []) if c is not None]) > 1
    )
    if is_repetition_chain or data in (
        "additive",
        "shift_expr",
        "mdr_expr",
        "arith_expr",
        "bitw_or",
        "bitw_xor",
        "bitw_and",
        "bit_and",
        "bit_or",
        "bit_xor",
        "content_test",
        "comparison",
        "and_test",
        "or_test",
        "logic_and",
        "logic_or",
        "bit_shift",
        "power",
        "factor",
    ):
        return _format_operator_chain_multi(
            expression, expression_context, indent_string, style
        )

    # A real ternary expression gets its own multiline layout.
    if data == "ternary" and _is_real_ternary(expression):
        return _format_ternary_multi(
            expression, expression_context, indent_string, style
        )

    # A parenthesized expression `( … )`: format the inner expression; the operator-chain
    # formatter re-adds the surrounding parens itself. Only delegate for operator chains,
    # so other inner content keeps its literal parens.
    if data == "primary":
        prim_children = [
            c for c in getattr(expression, "children", []) if c is not None
        ]
        if (
            len(prim_children) >= 3
            and isinstance(prim_children[0], Token)
            and str(getattr(prim_children[0], "type", "")).upper() == "PARENTHESIS_OPEN"
            and isinstance(prim_children[-1], Token)
            and str(getattr(prim_children[-1], "type", "")).upper()
            == "PARENTHESIS_CLOSE"
        ):
            _inner = _unwrap_wrappers_chain(prim_children[1])
            if get_data_str(_inner) in (
                "additive",
                "shift_expr",
                "mdr_expr",
                "arith_expr",
                "bitw_or",
                "bitw_xor",
                "bitw_and",
                "bit_and",
                "bit_or",
                "bit_xor",
                "content_test",
                "comparison",
                "and_test",
                "or_test",
                "logic_and",
                "logic_or",
                "bit_shift",
                "power",
                "factor",
            ):
                return _format_foldable_to_multiple_lines(
                    prim_children[1], expression_context, indent_string, style
                )

    # Handle assignment/cast/ternary wrappers - unwrap to find inner content
    if data in ("assignment", "cast", "ternary"):

        def find_operator_or_postfix(node):
            """Walk through wrapper types to find an operator chain, postfix,
            or unary (sign/bit_not/logic_not) node."""
            for child in getattr(node, "children", []):
                if child is None or not hasattr(child, "data"):
                    continue
                d = _tree_data_str(child)
                # Check if this IS the target node (operator chain or postfix)
                if d in (
                    "additive",
                    "shift_expr",
                    "mdr_expr",
                    "arith_expr",
                    "bitw_or",
                    "bitw_xor",
                    "bitw_and",
                    "bit_and",
                    "bit_or",
                    "bit_xor",
                    "content_test",
                    "comparison",
                    "and_test",
                    "or_test",
                    "logic_and",
                    "logic_or",
                    "bit_shift",
                    "power",
                    "factor",
                ):
                    return child
                elif d == "postfix":
                    return child
                # Also treat unary operators as targets — we need to handle them specially below
                if d in ("sign", "bit_not", "logic_not"):
                    return child
                # These are wrapper types - recurse into them (including ternary)
                if d in ("assignment", "cast", "ternary", "type_test", "factor"):
                    result = find_operator_or_postfix(child)
                    if result is not None:
                        return result
            return None

        inner_node = find_operator_or_postfix(expression)

        if inner_node is not None:
            inner_data = _tree_data_str(inner_node)
            # Handle operator chains (additive, comparison, etc.)
            if inner_data in (
                "additive",
                "shift_expr",
                "mdr_expr",
                "arith_expr",
                "bitw_or",
                "bitw_xor",
                "bitw_and",
                "bit_and",
                "bit_or",
                "bit_xor",
                "content_test",
                "comparison",
                "and_test",
                "or_test",
                "logic_and",
                "logic_or",
                "bit_shift",
                "power",
                "factor",
            ):
                return _format_operator_chain_multi(
                    inner_node, expression_context, indent_string, style
                )
            elif inner_data == "postfix":
                # Attribute/call chains need their own multiline handler
                return _format_postfix_multi(
                    inner_node,
                    expression_context,
                    indent_string,
                    style or StyleConfig(),
                )
            elif inner_data in ("sign", "bit_not", "logic_not"):
                # Unary operator wrapping an additive/shift/etc chain — needs
                # special multiline format.
                # For: var w = -(1+1+1+...)  →  var w = -(
                #                                       1
                #                                       + 1
                #                                       ...
                #                                   )
                return _format_unary_chain_multi(
                    inner_node, expression_context, indent_string, style
                )
        else:
            # No operator chain found inside — format the whole expression on
            # multiple lines if too long.
            # This handles cases like: var very_long_name = ^"../Some/Stuff"
            # when total line > 100 chars
            from .expression_to_str import expression_to_str as _etostr

            try:
                expr_str = _etostr(expression)
                max_len = style.line_length if style else 100
                display_text = f"{expr_str}{expression_context.suffix_string}"
                # Check if the full line (prefix + content) exceeds max length
                prefix_for_check = expression_context.prefix_string.rstrip()
                check_line = f"{indent_string}{prefix_for_check}{display_text}".replace(
                    "\t", "    "
                )
                if len(check_line) > max_len:
                    # Wrap in parens for multiline: prefix(
                    # then content, then ) on next line
                    new_ec = ExpressionContext(
                        f"{expression_context.prefix_string}(",
                        expression_context.prefix_line,
                        f") {expression_context.suffix_string}",
                        expression_context.suffix_line,
                    )
                    result_lines = [
                        (
                            expression_context.prefix_line,
                            f"{indent_string}{prefix_for_check}(",
                        )
                    ]
                    inner_indent = continuation_indent(indent_string)
                    result_lines.append((None, inner_indent + display_text))
                    result_lines.append((None, indent_string + ")"))
                    return result_lines
                else:
                    # Short enough - output as single line
                    return [
                        (
                            expression_context.prefix_line,
                            f"{indent_string}{prefix_for_check}{display_text}",
                        )
                    ]
            except Exception:
                pass

    # Handle unary operators that got folded (sign, bit_not, logic_not)
    if data in ("sign", "bit_not", "logic_not"):
        from .expression_to_str import expression_to_str as _etostr

        try:
            # Find the inner expression and detect the unary operator prefix
            non_none = [c for c in getattr(expression, "children", []) if c is not None]
            op_prefix = ""
            inner_expr = None
            for child in non_none:
                if isinstance(child, Token):
                    ttype = str(getattr(child, "type", "")).upper()
                    if ttype == "TILDE":
                        op_prefix = "~"
                    elif ttype == "PLUS":
                        op_prefix = "+"
                    elif ttype == "MINUS":
                        op_prefix = "-"
                    elif ttype in ("NOT", "BANG"):
                        # `not` (space before the wrapped parens, unlike `-`/`~`).
                        op_prefix = "not "
                elif isinstance(child, Tree) and inner_expr is None:
                    inner_expr = child

            # Unwrap through wrapper types (type_test → primary → parens → assignment/cast/ternary)
            # to find the actual operator chain
            unwrapped_inner = inner_expr
            while unwrapped_inner is not None and isinstance(unwrapped_inner, Tree):
                ud = _tree_data_str(unwrapped_inner)
                if ud == "type_test":
                    # Go inside type_test
                    tt_children = [
                        c
                        for c in getattr(unwrapped_inner, "children", [])
                        if c is not None
                    ]
                    unwrapped_inner = tt_children[0] if len(tt_children) >= 1 else None
                elif ud == "primary" and isinstance(unwrapped_inner, Tree):
                    # primary wraps parens; find the content between ( and )
                    prim_children = list(getattr(unwrapped_inner, "children", []))
                    open_idx = -1
                    for i, c in enumerate(prim_children):
                        if (
                            isinstance(c, Token)
                            and str(getattr(c, "type", "")).upper()
                            == "PARENTHESIS_OPEN"
                        ):
                            open_idx = i
                            break
                    if open_idx >= 0 and open_idx + 2 <= len(prim_children):
                        unwrapped_inner = prim_children[
                            open_idx + 1
                        ]  # The content after (
                    else:
                        break
                elif ud in ("assignment", "cast", "ternary"):
                    # These are wrapper types with maybe_placeholders
                    w_children = [
                        c
                        for c in getattr(unwrapped_inner, "children", [])
                        if c is not None
                    ]
                    unwrapped_inner = w_children[0] if len(w_children) >= 1 else None
                else:
                    break  # Found a non-wrapper type or reached a Token

            inner_expr_to_use = unwrapped_inner

            # If the inner expression is an operator chain or postfix,
            # delegate to multiline handler with prefix
            if inner_expr_to_use is not None and hasattr(inner_expr_to_use, "data"):
                d = _tree_data_str(inner_expr_to_use)
                # The grammar's binary chain rules, loosest first. The list used to carry
                # the old grammar's names (`and_test`, `or_test`, `arith_expr`, `mdr_expr`,
                # `shift_expr`, `bitw_*`) and to miss `logic_and`/`logic_or`/`bit_shift`/
                # `factor`, so `not (<an and chain>)` found no handler and was flattened
                # onto one over-long line.
                if d in (
                    "logic_or",
                    "logic_and",
                    "content_test",
                    "comparison",
                    "bit_or",
                    "bit_xor",
                    "bit_and",
                    "bit_shift",
                    "additive",
                    "factor",
                ):
                    # Create a modified context with the unary prefix
                    new_ec = ExpressionContext(
                        f"{expression_context.prefix_string}{op_prefix}",
                        expression_context.prefix_line,
                        expression_context.suffix_string,
                        expression_context.suffix_line,
                    )
                    return _format_operator_chain_multi(
                        inner_expr_to_use, new_ec, indent_string, style
                    )
                elif d in ("postfix",):
                    new_ec = ExpressionContext(
                        f"{expression_context.prefix_string}{op_prefix}",
                        expression_context.prefix_line,
                        expression_context.suffix_string,
                        expression_context.suffix_line,
                    )
                    return _format_postfix_multi(
                        inner_expr_to_use, new_ec, indent_string, style or StyleConfig()
                    )

            # For single-line unary expressions (e.g., ~x, +1), format inline.
            # Measure the full line (indent + prefix + content) so the declaration
            # prefix is never dropped from the length decision.
            expr_str = _etostr(expression)
            max_len = style.line_length if style else 100
            display_text = (
                f"{expression_context.prefix_string}{expr_str}"
                f"{expression_context.suffix_string}"
            )
            display_length = len((indent_string + display_text).replace("\t", "    "))
            if display_length <= max_len:
                return [
                    (expression_context.prefix_line, f"{indent_string}{display_text}")
                ]
        except Exception:
            pass

    # A parenthesized expression that does not fit keeps its parentheses: `(` closes the
    # first line, the content gets a line of its own one level in, and `)` comes back to
    # the statement indent. That is the shape the goldens use for an over-long
    # initializer. There is no `primary` entry in the handler map below, so without this
    # branch a paren-wrapped expression is silently flattened onto one over-long line.
    if data == "primary":
        from .class_statement import _display_width
        from .expression_to_str import (
            expression_to_str as _etostr_paren,
            _paren_wrapped_primary_inner,
        )

        _inner = _paren_wrapped_primary_inner(expression)
        if _inner is not None:
            _inner_indent = continuation_indent(indent_string)
            _inner_text = _etostr_paren(_inner)
            _limit = style.line_length if style else 100
            if (
                not _is_forced_multiline(_inner, style)
                and _display_width(_inner_indent + _inner_text) <= _limit
            ):
                return [
                    (
                        expression_context.prefix_line,
                        f"{indent_string}{expression_context.prefix_string}(",
                    ),
                    (None, f"{_inner_indent}{_inner_text}"),
                    (None, f"{indent_string}){expression_context.suffix_string}"),
                ]
            # Too long, or forced multiline, even on a line of its own: lay the content
            # out by its own rules and let that layout supply whatever parentheses it
            # needs, rather than adding a second pair here.
            return _format_standalone_expression(
                _inner,
                ExpressionContext(
                    expression_context.prefix_string,
                    expression_context.prefix_line,
                    expression_context.suffix_string,
                    expression_context.suffix_line,
                ),
                indent_string,
                style or StyleConfig(),
            )

    # Handle par_expr separately to avoid complex inline expressions
    if data == "par_expr" and expression.children:
        new_ec = ExpressionContext(
            f"{expression_context.prefix_string}(",
            expression_context.prefix_line,
            f"){expression_context.suffix_string}",  # type: ignore
            expression_context.suffix_line,
        )
        return _format_standalone_expression(
            expression.children[0], new_ec, indent_string, style or StyleConfig()
        )

    # Handle annotation separately
    if data == "annotation" and expression.children:
        ann_name = expression_to_str_safe(expression.children[0])
        new_ec = ExpressionContext(
            f"@{ann_name}",
            expression_context.prefix_line,
            expression_context.suffix_string,  # type: ignore
            expression_context.suffix_line,
        )
        # The element is a whole FormattedLines list, not a FormattedLine tuple. The
        # nested shape is preserved as-is (callers of this path expect it); the list
        # is therefore annotated more loosely than FormattedLines.
        nested_lines: List[Any] = [
            _format_standalone_expression(
                expression.children[0],
                new_ec,
                indent_string,
                style or StyleConfig(),
            )
        ]
        return nested_lines

    handler = handlers_map.get(data)
    if handler is None:
        # For unhandled types, fall back to single-line conversion using expression_to_str
        from .expression_to_str import expression_to_str as _etostr

        try:
            expr_str = _etostr(expression)
            text = (
                f"{expression_context.prefix_string}{expr_str}"
                f"{expression_context.suffix_string}"  # type: ignore
            )
            return [(expression_context.prefix_line, f"{indent_string}{text}")]
        except Exception:
            return [
                _format_concrete_expression_to_single_line(
                    expression, expression_context, indent_string
                )[0]
            ]

    try:
        result = handler()
    except Exception:
        # Fallback on error - single line with prefix preserved
        from .expression_to_str import expression_to_str as _etostr_fallback

        try:
            expr_str = _etostr_fallback(expression)
            text = f"{expression_context.prefix_string}{expr_str}{expression_context.suffix_string}"
            return [(expression_context.prefix_line, f"{indent_string}{text}")]
        except Exception:
            return [
                _format_concrete_expression_to_single_line(
                    expression, expression_context, indent_string
                )[0]
            ]
    if isinstance(result, ExpressionContext):
        first_child = (
            expression.children[0]
            if hasattr(expression, "children") and expression.children
            else expression  # type: ignore
        )
        # Nested FormattedLines inside the list - same loose annotation as above.
        nested_lines = [
            _format_standalone_expression(
                first_child, result, indent_string, style or StyleConfig()
            )
        ]
        return nested_lines
    return result


def _format_unary_chain_multi(  # noqa: C901
    expression: Tree,
    expression_context: ExpressionContext,
    indent_string: str,
    style=None,
) -> FormattedLines:
    """Format a sign/bit_not/logic_not node containing an operator chain across multiple lines.

    For long unary expressions with additive/shift/etc chains inside parens like:
        var w = -(1+1+1+...long...)

    Produces:
        var w = -(
            1
            + 1
            ...
        )
    """
    # Determine the unary operator prefix
    op_prefix = ""
    non_none = [c for c in getattr(expression, "children", []) if c is not None]

    for child in non_none:
        if isinstance(child, Token):
            ttype = str(getattr(child, "type", "")).upper()
            if ttype == "TILDE":
                op_prefix = "~"
            elif ttype == "MINUS":
                op_prefix = "-"
            elif ttype == "PLUS":
                op_prefix = "+"
            break  # First operator token is the one we want

    if not op_prefix and non_none:
        for subtree in getattr(expression, "iter_subtrees", lambda: [])():
            if isinstance(subtree, Token):
                ttype = str(getattr(subtree, "type", "")).upper()
                if ttype == "TILDE":
                    op_prefix = "~"
                elif ttype == "MINUS":
                    op_prefix = "-"
                elif ttype == "PLUS":
                    op_prefix = "+"
                    break

    # Unwrap through wrapper types to find the inner operator chain
    unwrapped_inner = None
    for child in non_none:
        if isinstance(child, Tree):
            unwrapped_inner = child
            break

    while unwrapped_inner is not None and isinstance(unwrapped_inner, Tree):
        ud = _tree_data_str(unwrapped_inner)
        if ud == "type_test":
            tt_children = [
                c for c in getattr(unwrapped_inner, "children", []) if c is not None
            ]
            unwrapped_inner = tt_children[0] if len(tt_children) >= 1 else None
        elif ud == "primary" and isinstance(unwrapped_inner, Tree):
            prim_children = list(getattr(unwrapped_inner, "children", []))
            # Find content between ( and ) in parenthesized primary
            open_idx = -1
            for i, c in enumerate(prim_children):
                if (
                    isinstance(c, Token)
                    and str(getattr(c, "type", "")).upper() == "PARENTHESIS_OPEN"
                ):
                    open_idx = i
                    break
            if open_idx >= 0 and open_idx + 2 <= len(prim_children):
                unwrapped_inner = prim_children[open_idx + 1]
            else:
                break
        elif ud in ("assignment", "cast", "ternary"):
            w_children = [
                c for c in getattr(unwrapped_inner, "children", []) if c is not None
            ]
            unwrapped_inner = w_children[0] if len(w_children) >= 1 else None
        elif ud == "factor":
            f_children = [
                c for c in getattr(unwrapped_inner, "children", []) if c is not None
            ]
            # factor has optional operators; get the expression part
            expr_child = next((c for c in f_children if isinstance(c, Tree)), None)
            unwrapped_inner = (
                expr_child
                if expr_child
                else (f_children[0] if len(f_children) >= 1 else None)
            )
        else:
            break

    inner_chain_types = (
        "additive",
        "shift_expr",
        "mdr_expr",
        "arith_expr",
        "bitw_or",
        "bitw_xor",
        "bitw_and",
        "bit_and",
        "bit_or",
        "bit_xor",
        "content_test",
        "comparison",
        "and_test",
        "or_test",
    )

    if unwrapped_inner is not None and hasattr(unwrapped_inner, "data"):
        d = _tree_data_str(unwrapped_inner)
        if d in inner_chain_types or d == "postfix":
            # Delegate to the appropriate multiline handler with unary prefix + parens
            new_prefix = f"{expression_context.prefix_string}{op_prefix}"
            if d == "postfix":
                return _format_postfix_multi(
                    unwrapped_inner,
                    ExpressionContext(
                        new_prefix, expression_context.prefix_line, "", None
                    ),
                    indent_string,
                    style or StyleConfig(),
                )
            else:
                # Use operator chain multi with special wrapping
                return _format_unary_operator_chain_multi(
                    unwrapped_inner,
                    new_prefix,
                    expression_context.suffix_string,
                    expression_context.prefix_line,
                    expression_context.suffix_line,
                    indent_string,
                    style,
                )

    # Fallback: format the whole unary node as a single line via expression_to_str.
    # Keep the declaration prefix so e.g. `var c = not not ... !true` is not truncated.
    from .expression_to_str import expression_to_str as _etostr

    prefix = (
        expression_context.prefix_string
        if isinstance(expression_context.prefix_string, str)
        else ""
    )
    try:
        expr_str = _etostr(expression)
        return [(expression_context.prefix_line, f"{indent_string}{prefix}{expr_str}")]
    except Exception:
        return [(expression_context.prefix_line, f"{indent_string}{prefix}?")]


def _format_unary_operator_chain_multi(
    expression: Tree,
    prefix_text: str,
    suffix_text: str,
    prefix_line,
    suffix_line,
    indent_string: str,
    style=None,
) -> FormattedLines:
    """Format an operator chain inside a unary (sign/bit_not/logic_not) context
    across multiple lines.

    Special wrapper around _format_operator_chain_multi that adds the unary operator prefix
    before the opening paren and preserves proper multiline formatting.

    For var w = -(1+1+1+...):
        prefix_text="var w = -" → first line: "var w = -("
        inner content formatted by _format_operator_chain_multi
        suffix added after closing paren
    """
    # We need to create a modified expression context where the prefix includes
    # the unary operator and opening paren is on the same line as prefix.
    # But _format_operator_chain_multi expects just a text prefix, not containing '('.
    # So we do the formatting manually here.

    non_none = [c for c in getattr(expression, "children", []) if c is not None]

    # Extract operands and operators
    from .expression_to_str import _operator_text

    parts = []
    for child in non_none:
        if isinstance(child, Token):
            ttype = str(getattr(child, "type", "")).upper()
            op_types = {
                "PLUS",
                "MINUS",
                "STAR",
                "SLASH",
                "PERCENT",
                "AND_AMPERSAND",
                "AMPERSAND_AMPERSAND",
                "PIPE_PIPE",
                "OR",
                "CARET",
                "LESS_LESS",
                "GREATER_GREATER",
            }
            if ttype in op_types:
                parts.append(("op", _operator_text(getattr(child, "value", ""))))
        elif isinstance(child, Tree):
            d = _tree_data_str(child)
            if d == "type_test":
                val_tokens = [c for c in child.children if isinstance(c, Token)]
                parts.append(
                    ("operand", str(val_tokens[0].value) if val_tokens else "?")
                )

    max_len = style.line_length if style else 100
    first_line_no = prefix_line or 0

    # Build compact segments: [operand, "op operand", "op operand", ...]
    compact_segments = []
    for i in range(0, len(parts), 2):
        _, pval = parts[i]  # operand
        op_prefix = ""
        if i > 0 and i - 1 < len(parts):
            prev_ptype, prev_pval = parts[i - 1]
            if prev_ptype == "op":
                op_prefix = f"{prev_pval} "
        compact_segments.append(f"{op_prefix}{pval}")

    compact_display = " ".join(compact_segments)
    # A chain of unary operators is one unbreakable operand, so the indent is what has to give
    # when the content only fits inside the parentheses at one level.
    inner_indent = continuation_indent_for(
        indent_string, len(compact_display), max_len, style.tab_size if style else 4
    )
    inner_content_width = len((inner_indent + compact_display).replace("\t", "    "))

    # Add opening paren: prefix is like "var w = -" → first output line becomes "var w = -("
    result_lines = [(first_line_no, f"{indent_string}{prefix_text}(")]

    if inner_content_width <= max_len:
        # Compact multiline: content on one line inside parens
        result_lines.append((None, f"{inner_indent}{compact_display}"))
        close = f") {suffix_text}".rstrip() if suffix_text else ")"
        result_lines.append((None, f"{indent_string}{close}"))
    else:
        # Split each operand on its own line. One operand per line is short, so this branch
        # gets the full two levels even when the compact form had to settle for one.
        inner_indent = continuation_indent(indent_string)
        for i, seg in enumerate(compact_segments):
            if i == 0:
                result_lines.append((None, f"{inner_indent}{seg}"))
            else:
                parts_of_seg = seg.split(" ", 1)
                if len(parts_of_seg) == 2 and parts_of_seg[0] in (
                    "+",
                    "-",
                    "*",
                    "/",
                    "%",
                    "and",
                    "or",
                    "^",
                    "|",
                    "&",
                    "<<",
                    ">>",
                ):
                    result_lines.append(
                        (None, f"{inner_indent}{parts_of_seg[0]} {parts_of_seg[1]}")
                    )
                else:
                    result_lines.append((None, f"{inner_indent}{seg}"))
        close = f") {suffix_text}".rstrip() if suffix_text else ")"
        result_lines.append((None, f"{indent_string}{close}"))

    return result_lines


def _tree_data_str(node) -> str:
    """Return a tree's rule name as a plain string."""
    if isinstance(node, Tree):
        d = node.data
        return str(d) if not hasattr(d, "value") else d.value  # type: ignore
    return ""


def _unwrap_wrappers_chain(node):
    """Recursively unwrap assignment/cast/ternary/type_test wrapper nodes.

    A `ternary` node doubles as a pass-through wrapper (no `if` token) and as a real
    ternary expression; stop unwrapping at a real ternary so it can be formatted.
    """
    WRAPPERS = frozenset({"assignment", "cast", "ternary", "type_test"})
    while isinstance(node, Tree) and get_data_str(node) in WRAPPERS:
        if _is_real_ternary(node):
            break
        # A cast/type_test carrying `is`/`as` repetitions is not a pass-through
        # wrapper: its repetitions live on this node, so unwrapping would drop them.
        if (
            get_data_str(node)
            in ("cast", "type_test", "asless_type_test", "actual_type_cast")
            and len([c for c in getattr(node, "children", []) if c is not None]) > 1
        ):
            break
        for c in getattr(node, "children", []):
            if c is not None and isinstance(c, Tree):
                node = c
                break
        else:
            break
    return node


def _is_real_ternary(node) -> bool:
    """True if `node` is a ternary expression (has an `if` token), not a pass-through wrapper."""
    if not isinstance(node, Tree) or get_data_str(node) != "ternary":
        return False
    return any(
        isinstance(c, Token) and str(getattr(c, "type", "")).upper() == "IF"
        for c in getattr(node, "children", [])
    )


def expression_to_str_safe(expression):
    """Safe wrapper around expression_to_str."""
    try:
        from .expression_to_str import expression_to_str as _etostr

        return _etostr(expression)
    except Exception:
        return "?"


# ─── Specific multi-line formatters ───────────────────────────────────────


def _format_array_multi(
    array: Tree, expr_ctx: ExpressionContext, indent_string: str, style=None
) -> FormattedLines:
    """Format an array literal across multiple lines with proper indentation.

    Produces output like:
        var x = [
            1,
            2,
            3,
        ]

    The opening bracket stays on the first line after the prefix, each element
    gets its own line at one indent level deeper than prefix, and closing bracket
    aligns with elements.
    """
    from .expression_to_str import (
        standalone_expression_to_str as _sestr,
        expression_to_str as _etostr,
    )
    from .expression_utils import is_any_comma, has_trailing_comma

    # Extract raw element nodes (not yet serialized) and their text representations
    all_children = list(getattr(array, "children", []))  # type: ignore
    element_nodes = []  # Raw tree/Token children for processing
    for child in all_children:
        if isinstance(child, Token):
            ttype = str(getattr(child, "type", "")).lower()
            if ttype in ("bracket_open", "bracket_close"):
                continue
            if is_any_comma(child):
                continue
        elif child is None:
            continue
        elif hasattr(child, "data"):
            d = _tree_data_str(child)
            if d in ("elements", "array_pattern_items"):
                for ec in getattr(child, "children", []):
                    if (
                        isinstance(ec, Token)
                        and str(getattr(ec, "type", "")).upper() == "COMMA"
                    ):
                        continue
                    if ec is None or is_any_comma(ec):
                        continue
                    element_nodes.append(ec)
            elif not is_any_comma(child):
                element_nodes.append(child)
        else:
            element_nodes.append(child)

    # For inline mode: serialize all elements to text first
    inline_elements = [
        _sestr(ec) if isinstance(ec, Tree) else _sestr(ec) for ec in element_nodes
    ]
    joined = ", ".join(inline_elements)

    max_len = style.line_length if style else 100
    # The opening bracket sits at the statement indent, so that indent counts
    # toward the fit test. Leaving it out let an array re-decide "fits" after the
    # caller had already found the line too long, and it stayed on one line.
    total_width = (
        len(indent_string.replace("\t", "    "))
        + len(expr_ctx.prefix_string)
        + len(joined)
        + len(expr_ctx.suffix_string)
        + 2  # [ ]
    )
    has_tc = has_trailing_comma(array)

    # Check if any nested array element has a trailing comma - forces multiline even if parent fits
    needs_nested_multi = _check_any_child_element_has_trailing_tc(element_nodes)

    # Comments written inside the brackets are layout items of their own, not elements.
    layout = _collection_layout(array)
    has_comments = any(kind == "comment" for kind, _n, _c in layout)

    # Check if we can fit on one line (when not forced multiline by trailing comma or many elements)
    if (
        not has_tc
        and not needs_nested_multi
        and not has_comments
        and total_width <= max_len
        and len(element_nodes) < 10
        and not any(_is_forced_multiline(e, style) for e in element_nodes)
    ):
        return [
            (
                expr_ctx.prefix_line,
                f"{indent_string}{expr_ctx.prefix_string}[{joined}]"
                f"{expr_ctx.suffix_string}",  # type: ignore
            )
        ]

    # Multi-line format with proper indentation
    inner_indent = collection_indent(indent_string)
    prefix_text = (
        expr_ctx.prefix_string if isinstance(expr_ctx.prefix_string, str) else ""
    )

    lines: FormattedLines = []
    # First line: prefix + opening bracket
    lines.append(
        (
            expr_ctx.prefix_line,
            f"{indent_string}{prefix_text}[",
        )
    )  # type: ignore

    # Keep the elements on one continuation line when the whole run fits there. Widths are
    # display widths, so a tab counts as four columns (a tab-counted fit wrongly joined long
    # runs). A trailing comma does not block this: the packed line ends in one, which is what
    # the guide asks of a multiline list, so a packed list carrying one is stable. Refusing to
    # pack when it was present made the second run explode what the first run packed -
    # `long_array_expressions`, `long_dict_expressions`, `assignment_expressions` and
    # `parentheses_expressions_removal` were non-idempotent for exactly that reason.
    inner_width = len(inner_indent.replace("\t", "    "))
    if (
        not has_comments
        and not needs_nested_multi
        and inner_width + len(joined) <= max_len
        and not any(_is_forced_multiline(e, style) for e in element_nodes)
    ):
        lines.append(
            (
                None,
                f"{inner_indent}{joined},",
            )
        )  # type: ignore
    else:
        item_total = len([e for e in layout if e[0] == "item"])
        item_index = -1
        # Source line of whatever was emitted last, so a comment written on the same
        # line rides on that line (`1,  # b`) and one on its own line does not.
        prev_line = _node_line(array)
        for _, (kind, node, had_comma) in enumerate(layout):
            if kind == "comment":
                text = _comment_layout_text(node)
                c_line = _node_line(node)
                if lines and c_line and c_line == prev_line:
                    ln, txt = lines[-1]
                    lines[-1] = (ln, f"{txt}  {text}")
                else:
                    lines.append(
                        (
                            None,
                            f"{inner_indent}{text}",
                        )
                    )
                continue
            item_index += 1
            elem_text = (
                _sestr(node)
                if isinstance(node, Tree)
                else str(getattr(node, "value", node))
            )
            # With a comment in the collection the author's own separator is kept: the
            # comment follows the comma (`1,  # b`), and an element the author left
            # without one (`2  # bbb`) must not gain a comma the comment would sit after.
            comma = ("," if had_comma else "") if has_comments else ","
            # A nested array breaks when its own inline form does not fit here, or
            # when the author wrote a trailing comma inside it. One carrying a comment
            # is laid out in place instead so the comment keeps its position.
            if (
                _node_is_array(node)
                and not _has_comment_child(node)
                and (
                    has_trailing_comma(_unwrap_for_tc_check(node))
                    or inner_width + len(elem_text) > max_len
                )
            ):
                lines.append(
                    (
                        None,
                        f"{inner_indent}[",
                    )
                )  # type: ignore
                lines.extend(
                    _format_array_multi_inner(
                        _unwrap_for_tc_check(node), inner_indent + "\t", style
                    )
                )
                # The last element gets no separator after its closing bracket.
                closing_comma = "," if item_index < item_total - 1 else ""
                lines.append(
                    (
                        None,
                        f"{inner_indent}]{closing_comma}",
                    )
                )  # type: ignore
                prev_line = _node_end_line(node)
                continue
            # An element that must be multiline (a lambda with a multi-statement body,
            # a call whose argument carries a trailing comma) is laid out in place
            # instead of flattened. Like the nested array above, a block element takes
            # the separator only between items: a comma after its last body line would
            # read as a comma inside the block.
            if _is_forced_multiline(node, style):
                elem_lines = _format_concrete_expression(
                    node,
                    ExpressionContext("", expr_ctx.prefix_line, "", None),
                    inner_indent,
                    style or StyleConfig(),
                )
                block_comma = "," if (item_index < item_total - 1 or has_tc) else ""
                for j, (ln, txt) in enumerate(elem_lines):
                    lines.append(
                        (ln, f"{txt}{block_comma}" if j == len(elem_lines) - 1 else txt)
                    )
                prev_line = _node_end_line(node)
                continue
            lines.append(
                (
                    None,
                    f"{inner_indent}{elem_text}{comma}",
                )
            )  # type: ignore
            prev_line = _node_end_line(node)

    # Closing bracket at same indent level as the line with prefix (one less than elements)
    closing = expr_ctx.suffix_string if isinstance(expr_ctx.suffix_string, str) else ""
    lines.append(
        (
            None,
            f"{indent_string}]{closing}",
        )
    )  # type: ignore
    return lines


def _format_array_multi_inner(array: Tree, indent_string: str, style=None) -> list:
    """Format an array literal's contents for multiline mode (no expr_ctx needed).
    Used by parent arrays when nested arrays have trailing commas.
    """
    from .expression_to_str import standalone_expression_to_str as _sestr
    from .expression_utils import is_any_comma, has_trailing_comma

    all_children = list(getattr(array, "children", []))
    element_nodes = []
    for child in all_children:
        if isinstance(child, Token):
            ttype = str(getattr(child, "type", "")).lower()
            if ttype in ("bracket_open", "bracket_close"):
                continue
            if is_any_comma(child):
                continue
        elif child is None:
            continue
        elif hasattr(child, "data"):
            d = _tree_data_str(child)
            if d in ("elements", "array_pattern_items"):
                for ec in getattr(child, "children", []):
                    if (
                        isinstance(ec, Token)
                        and str(getattr(ec, "type", "")).upper() == "COMMA"
                    ):
                        continue
                    if ec is None or is_any_comma(ec):
                        continue
                    element_nodes.append(ec)
            elif not is_any_comma(child):
                element_nodes.append(child)

    result = []
    for node in element_nodes:
        # Recurse deeper: nested array with trailing comma?
        if isinstance(node, Tree) and _tree_data_str(node) == "array_literal":
            inner = _unwrap_for_tc_check(node)
            if (
                isinstance(inner, Tree)
                and _tree_data_str(inner) == "array_literal"
                and has_trailing_comma(inner)
            ):
                nested_lines = _format_array_multi_inner(
                    inner,
                    indent_string + ("\t" if "\t" not in indent_string else "\t"),
                    style,
                )
                result.extend(nested_lines)
                continue

        elem_text = (
            _sestr(node)
            if isinstance(node, Tree)
            else str(getattr(node, "value", node))
        )
        result.append((None, f"{indent_string}{elem_text},"))  # type: ignore
    return result


def _unwrap_for_tc_check(node):
    """Unwrap wrapper nodes to find the innermost expression for trailing comma check."""
    WRAPPERS = frozenset({"assignment", "cast", "ternary", "type_test"})
    while isinstance(node, Tree) and _tree_data_str(node) in WRAPPERS:
        for c in getattr(node, "children", []):
            if c is not None and isinstance(c, Tree):
                node = c
                break
        else:
            break
    return node


def _node_is_array(node) -> bool:
    """True if `node` (possibly wrapped) is an array literal."""
    inner = _unwrap_for_tc_check(node)
    return isinstance(inner, Tree) and _tree_data_str(inner) == "array_literal"


# The variable-length list child of each bracketed construct. A comment written inside
# the brackets lands in this child (see gdtoolkit/parser/comments.py), so the layout
# helpers below read it rather than the bracketed node itself.
_COLLECTION_CONTAINERS = frozenset(
    {
        "elements",
        "array_pattern_items",
        "dict_entries",
        "dict_pattern_items",
        "arguments",
        "parameters",
        "signal_parameters",
        "enum_entries",
    }
)


def _node_start_pos(node):
    """Best-effort source start position of a node (its own or its first child's)."""
    meta = getattr(node, "meta", None)
    pos = getattr(meta, "start_pos", None)
    if pos is not None:
        return pos
    for child in getattr(node, "children", []):
        if child is None or isinstance(child, Token):
            continue
        pos = _node_start_pos(child)
        if pos is not None:
            return pos
    return None


def _node_line(node):
    """1-based source line a node starts on, or 0 when unknown."""
    meta = getattr(node, "meta", None)
    line = getattr(meta, "line", None)
    if line:
        return int(line)
    for child in getattr(node, "children", []):
        if child is None:
            continue
        line = _node_line(child)
        if line:
            return line
    return 0


def _node_end_line(node):
    """1-based source line a node ends on, falling back to its start line."""
    try:
        return int(get_end_line(node))
    except Exception:
        return _node_line(node)


def _collection_container(collection):
    """The list child of a bracketed collection, or None when it has none."""
    for child in getattr(collection, "children", []):
        if isinstance(child, Tree) and _tree_data_str(child) in _COLLECTION_CONTAINERS:
            return child
    return None


def _collection_layout(collection):
    """Ordered layout entries of a bracketed collection.

    Each entry is ``('item', node, had_comma)`` or ``('comment', node, False)``.
    Comments are kept in source position so a caller can attach each one to the line it
    belongs to; rendering a comment as if it were an element produced output like
    ``[# a, 1, 2]``. ``had_comma`` records whether the source wrote a comma straight
    after the item, which decides the separator when an inline comment follows it.
    """
    container = _collection_container(collection)
    if container is not None:
        pool = [c for c in getattr(container, "children", []) if c is not None]
    else:
        # `collection` is itself the list (`parameters`, `signal_parameters`, ...). Keep
        # the comma tokens: they mark which item the author left a separator after.
        pool = [c for c in getattr(collection, "children", []) if c is not None]
    layout = []
    for child in pool:
        if isinstance(child, Token):
            if (
                str(getattr(child, "type", "")).upper() == "COMMA"
                and layout
                and layout[-1][0] == "item"
            ):
                kind, node, _comma = layout[-1]
                layout[-1] = (kind, node, True)
            continue
        if _tree_data_str(child) == "comment":
            layout.append(("comment", child, False))
        else:
            layout.append(("item", child, False))
    return layout


def _has_comment_child(node, depth=0):
    """True when a comment node sits anywhere below `node` (a lambda body aside)."""
    if not isinstance(node, Tree) or depth > 25:
        return False
    if _tree_data_str(node) == "comment":
        return True
    if _tree_data_str(node) == "lambda":
        return False
    for child in getattr(node, "children", []):
        if (
            child is not None
            and isinstance(child, Tree)
            and _has_comment_child(child, depth + 1)
        ):
            return True
    return False


def _lambda_header_has_comment(node, depth=0):
    """True when a lambda below `node` has a comment in its own parameter list.

    `_has_comment_child` stops at a lambda because its body is a separate block, but a
    comment in the lambda's *header* is different: it cannot be written on one line at
    all. Without this the lambda is flattened by `expression_to_str` and the comments are
    dropped. The search stops at each lambda it finds, since the body is formatted on its
    own pass.
    """
    if not isinstance(node, Tree) or depth > 25:
        return False
    if _tree_data_str(node) == "lambda":
        params = next(
            (
                c
                for c in getattr(node, "children", [])
                if isinstance(c, Tree)
                and _tree_data_str(c) in ("parameters", "signal_parameters")
            ),
            None,
        )
        if params is None:
            return False
        return any(
            isinstance(c, Tree) and _tree_data_str(c) == "comment"
            for c in getattr(params, "children", [])
        )
    for child in getattr(node, "children", []):
        if (
            child is not None
            and isinstance(child, Tree)
            and _lambda_header_has_comment(child, depth + 1)
        ):
            return True
    return False


def _comment_layout_text(node) -> str:
    from .comments import get_comment_text

    return get_comment_text(node)


def _call_arg_nodes(call_suffix):
    """The argument expression nodes of a `call_suffix`, skipping commas and parens."""
    from .expression_utils import is_any_comma

    args = []
    for c in getattr(call_suffix, "children", []):
        if c is None or isinstance(c, Token):
            continue
        if _tree_data_str(c) == "arguments":
            for a in getattr(c, "children", []):
                if a is None or isinstance(a, Token) or is_any_comma(a):
                    continue
                args.append(a)
        else:
            args.append(c)
    return args


def _check_any_child_element_has_trailing_tc(element_nodes):
    """Check if any child element in an array has a trailing comma (recursively).

    This detects nested arrays like [[1,2,3],[4,5,6,]] where the inner [4,5,6,]
    has a trailing comma but the outer doesn't.
    """
    from .expression_utils import has_trailing_comma as _htc

    WRAPPERS = frozenset({"assignment", "cast", "ternary", "type_test"})

    def _has_tc(node, depth=0):
        if not isinstance(node, Tree) or depth > 15:
            return False
        # Unwrap wrappers to find inner content
        while isinstance(node, Tree) and _tree_data_str(node) in WRAPPERS:
            for c in getattr(node, "children", []):
                if c is not None and isinstance(c, Tree):
                    node = c
                    break
            else:
                break

        nd = _tree_data_str(node)
        # Direct check: is this an array_literal with trailing comma?
        if nd == "array_literal":
            if _htc(node):
                return True
            # Check inside its elements for deeper nested arrays
            for child in getattr(node, "children", []):
                if isinstance(child, Tree) and _tree_data_str(child) == "elements":
                    for ec in getattr(child, "children", []):
                        if ec is not None and not (
                            isinstance(ec, Token)
                            and str(getattr(ec, "type", "")).upper() == "COMMA"
                        ):
                            if _has_tc(ec, depth + 1):
                                return True
            return False
        # Check children recursively for other node types
        for child in getattr(node, "children", []):
            if isinstance(child, Tree):
                if _has_tc(child, depth + 1):
                    return True
        return False

    for node in element_nodes:
        if _has_tc(node):
            return True
    return False


def _format_dict_multi(  # noqa: C901
    dict_tree: Tree, expr_ctx: ExpressionContext, indent_string: str, style=None
) -> FormattedLines:
    from .expression_to_str import expression_to_str as _etostr

    # Ordered items: ('comment', node) or
    # ('entry', node, rendered_text, key_text, separator, value_node, had_comma).
    # Comments stay in source position so each one is attached to the line it belongs to;
    # rendering one as an entry produced output like `{ # q, "x": 1 }`.
    layout = _collection_layout(dict_tree)
    has_comments = any(kind == "comment" for kind, _n, _c in layout)
    ordered: List[Tuple[Any, ...]] = []
    for kind, node, had_comma in layout:
        if kind == "comment":
            ordered.append(("comment", node))
            continue
        d = _tree_data_str(node)
        if d in ("dict_entry", "eq_dict_entry"):
            kids = [c for c in getattr(node, "children", []) if c is not None]
            key_text = _etostr(kids[0]) if kids else ""
            sep = ":"
            sep_idx = None
            for i, c in enumerate(kids):
                if isinstance(c, Token) and str(getattr(c, "type", "")).upper() in (
                    "COLON",
                    "EQUAL",
                ):
                    sep = "=" if str(getattr(c, "type", "")).upper() == "EQUAL" else ":"
                    sep_idx = i
                    break
            value_node = (
                kids[sep_idx + 1]
                if sep_idx is not None and sep_idx + 1 < len(kids)
                else None
            )
            ordered.append(
                ("entry", node, _etostr(node), key_text, sep, value_node, had_comma)
            )
        elif d == "dict_pattern_entry":
            # A dict-pattern entry is `key` alone (a contains-key pattern) or
            # `key: value`; the key is the first Tree child, the value the second.
            kids = [c for c in getattr(node, "children", []) if c is not None]
            key_node = None
            value_node = None
            is_pair = False
            for c in kids:
                if isinstance(c, Token):
                    if str(getattr(c, "type", "")).upper() == "COLON":
                        is_pair = True
                    continue
                if key_node is None:
                    key_node = c
                elif is_pair and value_node is None:
                    value_node = c
            key_text = _etostr(key_node) if key_node is not None else ""
            if is_pair and value_node is not None:
                ordered.append(
                    (
                        "entry",
                        node,
                        f"{key_text}: {_etostr(value_node)}",
                        key_text,
                        ":",
                        value_node,
                        had_comma,
                    )
                )
            else:
                ordered.append(
                    ("entry", node, key_text, key_text, None, None, had_comma)
                )
        else:
            ordered.append(("entry", node, _etostr(node), None, None, None, had_comma))

    entries = [e for e in ordered if e[0] == "entry"]
    elements = [e[2] for e in entries]

    max_len = style.line_length if style else 100
    compact_joined = ", ".join(elements)

    prefix_text = (
        expr_ctx.prefix_string if isinstance(expr_ctx.prefix_string, str) else ""
    )
    total = len(compact_joined) + len(prefix_text) + len(expr_ctx.suffix_string)  # type: ignore

    # Check if original dict had trailing comma on last entry. A trailing comma is a
    # deliberate "keep this exploded" signal, so it always forces multiline output.
    has_original_trailing_comma = _has_trailing_comma_in_elements(dict_tree)

    # A value that carries a trailing comma anywhere inside forces the whole
    # expression multiline (the legacy rule recurses), so an entry whose value is
    # forced must not be flattened back onto one line. A lambda with a multi-statement
    # body is forced too, and `expression_to_str` would flatten its body to one line.
    any_value_forced = any(
        _is_forced_multiline(v, style)
        for _k, _n, _f, _key, _sep, v, _hc in entries
        if v is not None
    )

    if (
        not has_comments
        and total < max_len
        and len(elements) <= 3
        and not has_original_trailing_comma
        and not any_value_forced
    ):
        # Single-line dictionary: the style guide asks for a space after `{` and before `}`
        # so it is visually distinct from an array.
        if not elements:
            return [
                (
                    expr_ctx.prefix_line,
                    f"{indent_string}{prefix_text}{{}}"
                    f"{expr_ctx.suffix_string}",  # type: ignore
                )
            ]
        return [
            (
                expr_ctx.prefix_line,
                f"{indent_string}{prefix_text}{{ {compact_joined} }}"
                f"{expr_ctx.suffix_string}",  # type: ignore
            )
        ]

    inner_indent = collection_indent(indent_string)
    inner_width = len(inner_indent.replace("\t", "    "))
    closing = expr_ctx.suffix_string if isinstance(expr_ctx.suffix_string, str) else ""

    # First line: prefix + opening brace
    lines: FormattedLines = [
        (expr_ctx.prefix_line, f"{indent_string}{prefix_text}{{"),  # type: ignore
    ]

    if (
        not has_comments
        and entries
        and not any_value_forced
        and inner_width + len(compact_joined) + 1 <= max_len
    ):
        # All entries share a single line inside the braces. A trailing comma does not block
        # this, for the same reason as the array above: the packed line ends in one.
        lines.append((None, f"{inner_indent}{compact_joined},"))
    else:
        prev_line = _node_line(dict_tree)
        entry_total = len(entries)
        entry_index = -1
        for _, item in enumerate(ordered):
            if item[0] == "comment":
                text = _comment_layout_text(item[1])
                c_line = _node_line(item[1])
                if lines and c_line and c_line == prev_line:
                    ln, txt = lines[-1]
                    lines[-1] = (ln, f"{txt}  {text}")
                else:
                    lines.append((None, f"{inner_indent}{text}"))
                continue
            _kind, node, full_text, key_text, sep, value_node, had_comma = item
            entry_index += 1
            # With a comment in the dict the author's own separator is kept, for the same
            # reason as in an array: the comment follows the entry's comma.
            comma = ("," if had_comma else "") if has_comments else ","
            if key_text is None or value_node is None:
                lines.append((None, f"{inner_indent}{full_text}{comma}"))
                prev_line = _node_end_line(node)
                continue
            value_forced = _is_forced_multiline(value_node, style)
            entry_line = (
                f"{key_text}{' = ' if sep == '=' else ': '}{_etostr(value_node)}"
            )
            if inner_width + len(entry_line) + 1 <= max_len and not value_forced:
                lines.append((None, f"{inner_indent}{entry_line}{comma}"))
                prev_line = _node_end_line(node)
                continue
            # Too long for one line: key on its own line, value on the following line(s).
            lines.append(
                (None, f"{inner_indent}{key_text}{' =' if sep == '=' else ':'}")
            )
            value_lines = _format_concrete_expression(
                value_node,
                ExpressionContext("", expr_ctx.prefix_line, "", expr_ctx.suffix_line),
                inner_indent,
                style or StyleConfig(),
            )
            # A value that is a block of its own takes the separator only between
            # entries: a comma after the last line of its body would read as a comma
            # inside the block. (A value that is merely too long stays as it was: the
            # style guide's trailing comma is the documented divergence there.)
            block_comma = comma
            if value_forced:
                block_comma = (
                    ","
                    if (entry_index < entry_total - 1 or has_original_trailing_comma)
                    else ""
                )
            for j, (ln_no, txt) in enumerate(value_lines):
                lines.append(
                    (ln_no, f"{txt}{block_comma}" if j == len(value_lines) - 1 else txt)
                )
            prev_line = _node_end_line(node)

    # Closing brace at the opening brace's indent.
    lines.append((None, f"{indent_string}}}{closing}"))
    return lines


def _format_await_multi(
    expression: Tree, ec: ExpressionContext, indent_string: str, style=None
) -> FormattedLines:
    """Format `await <operand>` by laying out the operand with `await ` as its prefix."""
    operand = next(
        (c for c in getattr(expression, "children", []) if isinstance(c, Tree)), None
    )
    if operand is None:
        from .expression_to_str import standalone_expression_to_str as _sestr

        return [
            (
                ec.prefix_line,
                f"{indent_string}{ec.prefix_string}await {_sestr(expression)}",
            )
        ]
    prefix = (
        f"{ec.prefix_string}await " if isinstance(ec.prefix_string, str) else "await "
    )
    return _format_concrete_expression(
        operand,
        ExpressionContext(prefix, ec.prefix_line, ec.suffix_string, ec.suffix_line),
        indent_string,
        style or StyleConfig(),
    )


def _format_preload_multi(
    expression: Tree, ec: ExpressionContext, indent_string: str, style=None
) -> FormattedLines:
    """Format `preload(<expr>)` when the single-line form overflows.

    Same shape as a broken call: the argument on its own indented line, the
    closing paren back at the statement's indent.
    """
    from .expression_to_str import standalone_expression_to_str as _sestr

    arg = next(
        (c for c in getattr(expression, "children", []) if isinstance(c, Tree)), None
    )
    inner_indent = continuation_indent(indent_string)
    prefix = ec.prefix_string if isinstance(ec.prefix_string, str) else ""
    suffix = ec.suffix_string if isinstance(ec.suffix_string, str) else ""
    return [
        (ec.prefix_line, f"{indent_string}{prefix}preload("),
        (None, f"{inner_indent}{_sestr(arg)}" if arg is not None else inner_indent),
        (None, f"{indent_string}){suffix}"),
    ]


def _format_call_multi(
    call: Tree, ec: ExpressionContext, indent_string: str, style=None
) -> FormattedLines:
    from .expression_to_str import expression_to_str as _etostr

    callee = _etostr(call.children[0]) if len(call.children) > 0 else "?"  # type: ignore
    args_count = max(1, len(call.children) - 1)  # type: ignore

    first_line = f"{indent_string}{callee}("
    lines = [(ec.prefix_line, first_line)]

    for i in range(1, len(call.children)):  # type: ignore
        arg_str = expression_to_str_safe(call.children[i])  # type: ignore
        comma = (
            ","
            if i < args_count - 1
            or (
                len(call.children) > 2
                and call.children[-1]
                and str(call.children[-1].data).endswith("trailing")
            )
            else ""  # type: ignore
        )
        lines.append((None, f"    {arg_str}{comma}"))

    return lines


def _format_getattr_call_multi(getattr_tree, ec, indent_string, style=None):
    from .expression_to_str import expression_to_str as _etostr

    callee = _etostr(getattr_tree.children[0]) if getattr_tree.children else "?"  # type: ignore
    args_count = max(1, len(getattr_tree.children) - 1)  # type: ignore
    lines = [(ec.prefix_line, f"{indent_string}{callee}(")]
    for i in range(1, len(getattr_tree.children)):  # type: ignore
        arg_str = expression_to_str_safe(getattr_tree.children[i])
        comma = "," if i < args_count else ""
        lines.append((None, f"    {arg_str}{comma}"))
    return lines


def _format_ternary_multi(
    expression: Tree,
    expression_context: ExpressionContext,
    indent_string: str,
    style=None,
) -> FormattedLines:
    """Format a ternary expression across multiple lines.

    var f = (
        "cond"
        if true
        else "value"
    )
    """
    from .expression_to_str import expression_to_str as _ets

    non_none = [c for c in getattr(expression, "children", []) if c is not None]
    if len(non_none) < 5:
        suffix = (
            expression_context.suffix_string
            if isinstance(expression_context.suffix_string, str)
            else ""
        )
        prefix = (
            expression_context.prefix_string
            if isinstance(expression_context.prefix_string, str)
            else ""
        )
        return [
            (
                expression_context.prefix_line,
                f"{indent_string}{prefix}{_ets(expression)}{suffix}",
            )
        ]

    cond, body, orelse = non_none[0], non_none[2], non_none[4]
    inner_indent = continuation_indent(indent_string)
    prefix = (
        expression_context.prefix_string
        if isinstance(expression_context.prefix_string, str)
        else ""
    )
    suffix = (
        expression_context.suffix_string
        if isinstance(expression_context.suffix_string, str)
        else ""
    )

    def _branch_lines(label: str, node) -> FormattedLines:
        """One `[label]expr` line, or the expression laid out in place if it must break.

        A branch that cannot be written on one line - it holds a trailing-comma
        collection somewhere - is not flattened. It is laid out a level in, with the
        `if`/`else` keyword in front of its opening parenthesis, which is the only
        place the operator's own keywords can go.
        """
        if not _is_forced_multiline(node, style):
            return [(None, f"{inner_indent}{label}{_ets(node)}")]
        if _is_real_ternary(node):
            # `_format_concrete_expression` would unwrap a real ternary as a
            # pass-through wrapper and land on its condition, flattening the whole
            # branch, so a nested ternary is laid out by the ternary formatter.
            sub = _format_ternary_multi(
                node,
                ExpressionContext("", expression_context.prefix_line, "", None),
                inner_indent,
                style,
            )
        else:
            sub = _format_concrete_expression(
                node,
                ExpressionContext("", expression_context.prefix_line, "", None),
                inner_indent,
                style or StyleConfig(),
            )
        if not sub:
            return [(None, f"{inner_indent}{label}{_ets(node)}")]
        first_line_no, first_text = sub[0]
        # Hoisted out of the f-string: a nested-quote/backslash expression (PEP 701)
        # would require python 3.12+, breaking 3.11 parse and the pinned black/mypy.
        stripped_first = first_text.lstrip("\t")
        return [(first_line_no, f"{inner_indent}{label}{stripped_first}")] + sub[1:]

    return (
        [(expression_context.prefix_line, f"{indent_string}{prefix}(")]
        + _branch_lines("", cond)
        + _branch_lines("if ", body)
        + _branch_lines("else ", orelse)
        + [(None, f"{indent_string}){suffix}")]
    )


def _format_operator_chain_multi(
    expression: Tree,
    expression_context: ExpressionContext,
    indent_string: str,
    style=None,
) -> FormattedLines:
    """Format operator chains (additive, shift, mdr_expr, etc.) across multiple lines.

    Format for short-to-medium chains (fit within line_length on inner content):
        var q = (
            1 + 2 + 3 + 4
        )

    Format for long chains that exceed line_length:
        var q = (
            1
            + 2
            + 3
            + 4
        )
    """

    def _contains_get_node(node):
        if _tree_data_str(node) == "get_node_expr":
            return True
        return any(
            _contains_get_node(c)
            for c in getattr(node, "children", [])
            if isinstance(c, Tree)
        )

    if _contains_get_node(expression):
        # Node paths such as `%Name/Sub` are modelled by the grammar as a division chain.
        # Emit them on a single line so no spaces are inserted into the path.
        from .expression_to_str import expression_to_str as _ets_gn

        suffix = (
            expression_context.suffix_string
            if isinstance(expression_context.suffix_string, str)
            else ""
        )
        return [
            (
                expression_context.prefix_line,
                f"{indent_string}{expression_context.prefix_string}{_ets_gn(expression)}{suffix}",
            )
        ]

    non_none = [c for c in getattr(expression, "children", []) if c is not None]

    # Extract operands and operators by walking children left-to-right.
    # parts alternates: operand, op, operand, op, ...
    from .expression_to_str import expression_to_str as _ets_op, _operator_text

    op_token_types = {
        "PLUS",
        "MINUS",
        "STAR",
        "STAR_STAR",
        "SLASH",
        "PERCENT",
        "LESS_LESS",
        "GREATER_GREATER",
        "AMPERSAND",
        "PIPE",
        "CARET",
        "TILDE",
        "AMPERSAND_AMPERSAND",
        "PIPE_PIPE",
        "AND",
        "OR",
        "IN",
        "IS",
        "AS",
        "NOT",
        "BANG",
        "LESS",
        "GREATER",
        "LESS_EQUAL",
        "GREATER_EQUAL",
        "EQUAL_EQUAL",
        "BANG_EQUAL",
    }
    # Triples of (kind, inline text, node): the node is None for operator entries and a
    # Token or Tree for operand entries, so the columns stay open.
    parts: List[Tuple[str, Any, Any]] = []
    for child in non_none:
        if isinstance(child, Token):
            ttype = str(getattr(child, "type", "")).upper()
            if ttype in op_token_types:
                parts.append(("op", _operator_text(getattr(child, "value", "")), None))
            else:
                # A bare literal operand. `type_test`/`cast` operands inline to a
                # token (`1 is int` gives a NUMBER child), unlike the arithmetic
                # chains whose operands stay wrapped in a rule.
                parts.append(("operand", _ets_op(child), child))
        elif isinstance(child, Tree):
            if _tree_data_str(child) == "comparison_operator":
                # The comparison operator is wrapped in its own tree; unwrap it.
                toks = [
                    c for c in getattr(child, "children", []) if isinstance(c, Token)
                ]
                parts.append(
                    ("op", "".join(str(getattr(t, "value", "")) for t in toks), None)
                )
            else:
                # Any other tree child is an operand. Keep both its inline text and its
                # node, so a long operand can be re-formatted on its own (recursively).
                parts.append(("operand", _ets_op(child), child))

    max_len = style.line_length if style else 100
    first_line_no = expression_context.prefix_line  # type: ignore
    inner_indent = continuation_indent(indent_string)
    prefix_text = expression_context.prefix_string  # type: ignore

    # Build ordered (operator, operand text, operand node) triples.
    items = []
    pending_op = ""
    for kind, value, node in parts:
        if kind == "op":
            # Join consecutive operator tokens into one operator (`not` + `in` -> `not in`).
            pending_op = f"{pending_op} {value}".strip()
        else:
            items.append((pending_op, value, node))
            pending_op = ""

    compact_display = " ".join(f"{op} {text}" if op else text for op, text, _ in items)
    # The indent gives way to the limit when the chain fits on one line inside the parentheses
    # only at one level.
    inner_indent = continuation_indent_for(
        indent_string, len(compact_display), max_len, style.tab_size if style else 4
    )
    inner_content_width = len((inner_indent + compact_display).replace("\t", "    "))

    # If an operand is itself forced multiline (e.g. an array with a trailing comma), the
    # whole chain must be exploded one operand per line rather than kept on one line.
    any_operand_forced = any(
        node is not None and _is_forced_multiline(node) for _op, _text, node in items
    )

    # The closing paren carries the suffix (`:` for an if/while/match condition), or the
    # emitted header loses it and becomes invalid GDScript.
    suffix_text = (
        expression_context.suffix_string
        if isinstance(expression_context.suffix_string, str)
        else ""
    )

    # If the inner content fits, use: (\ncompact) format
    if inner_content_width <= max_len and not any_operand_forced:
        result_lines: FormattedLines = [
            (first_line_no, f"{indent_string}{prefix_text}(")
        ]
        result_lines.append((None, f"{inner_indent}{compact_display}"))
        result_lines.append((None, f"{indent_string}){suffix_text}"))
        return result_lines

    # Long chain: one operand per line. Each operand is re-formatted on its own so
    # nested parenthesized chains expand rather than staying on a single long line. One
    # operand per line is short, so this branch gets the full two levels even when the
    # compact form above had to settle for one.
    inner_indent = continuation_indent(indent_string)
    result_lines = [(first_line_no, f"{indent_string}{prefix_text}(")]
    for i, (op, _text, node) in enumerate(items):
        sub = _format_concrete_expression(
            node,
            ExpressionContext("", first_line_no, "", expression_context.suffix_line),
            inner_indent,
            style or StyleConfig(),
        )
        if not sub:
            continue
        first_ln, first_txt = sub[0]
        if i == 0 and not op:
            result_lines.append((first_ln, first_txt))
        else:
            body = (
                first_txt[len(inner_indent) :]
                if first_txt.startswith(inner_indent)
                else first_txt.lstrip()
            )
            result_lines.append((first_ln, f"{inner_indent}{op} {body}".rstrip()))
        result_lines.extend(sub[1:])
    result_lines.append((None, f"{indent_string}){suffix_text}"))
    return result_lines


def _format_subscription_multi(
    subscr: Tree, expr_ctx: ExpressionContext, indent_string: str, style=None
) -> FormattedLines:
    from .expression_to_str import standalone_expression_to_str as _sestr

    base = _sestr(subscr.children[0]) if len(subscr.children) > 0 else "?"  # type: ignore
    index = _sestr(subscr.children[1]) if len(subscr.children) > 1 else "?"  # type: ignore
    return [(expr_ctx.prefix_line, f"{indent_string}{base}[{index}]")]


def _find_lambda_arg(node, depth=0):
    """Return the first `lambda` node nested inside `node`, or None."""
    if not isinstance(node, Tree) or depth > 15:
        return None
    if _tree_data_str(node) == "lambda":
        return node
    for c in getattr(node, "children", []):
        if not isinstance(c, Tree):
            continue
        found = _find_lambda_arg(c, depth + 1)
        if found is not None:
            return found
    return None


def _find_lambda_argument(call_node):
    """Return the lambda passed as a call's sole argument, or None.

    Only a single-argument call is handled, so multi-argument calls keep their
    existing comma-splitting behaviour.
    """
    for c in getattr(call_node, "children", []):
        if isinstance(c, Tree) and _tree_data_str(c) == "arguments":
            arg_trees = [a for a in getattr(c, "children", []) if isinstance(a, Tree)]
            if len(arg_trees) != 1:
                return None
            return _find_lambda_arg(arg_trees[0])
    return None


def _format_lambda_multi(  # noqa: C901
    lambda_tree: Tree, ec: ExpressionContext, indent_string: str, style=None
) -> FormattedLines:
    """Format a multiline lambda expression with proper parameter splitting and suite formatting.

    When the lambda is long enough to trigger multi-line formatting,
    splits params across lines and formats the body as a block.
    """
    if style is None:
        style = StyleConfig()
    max_len = style.line_length

    # Extract lambda components from flat children structure
    lambda_name = ""
    param_names = []  # List of parameter name strings
    params_tree = None  # the `parameters` node itself, for the comment-aware header
    has_return_type = False
    return_type_str = ""
    suite_children = []
    in_params = False

    for child in getattr(lambda_tree, "children", []):
        if child is None:
            continue
        # Handle both string placeholders and Token objects
        ttype = (
            str(getattr(child, "type", "")).lower() if isinstance(child, Token) else ""
        )
        child_str = (
            str(child.value).strip()
            if isinstance(child, Token)
            else (str(child) if isinstance(child, str) else "")
        )

        if ttype == "parenthesis_open" or child_str == "(":
            in_params = True
            continue
        elif ttype == "parenthesis_close" or child_str == ")":
            in_params = False
            continue
        # forward_arrow and colon are not terminators for this loop — skip them but don't break
        # We need to reach suite which comes AFTER the colon

        if isinstance(child, Tree):
            data_name = str(getattr(child.data, "value", str(child.data)))

            # Handle specific rule types regardless of in_params state
            if data_name in ("parameters", "signal_parameters"):
                from .expression_to_str import lambda_parameter_strings

                param_names = lambda_parameter_strings(child)
                params_tree = child
            elif data_name == "return_type":
                has_return_type = True
                return_type_str = expression_to_str_safe(child)
            elif data_name == "suite":
                suite_children = (
                    list(child.children)
                    if hasattr(child, "children")
                    else []  # type: ignore
                )

        elif isinstance(child, Token):
            # Optional named lambda: func baz(...) — NAME token after FUNC
            if ttype == "name" and not in_params and lambda_name == "":
                lambda_name = str(child.value)

    num_params = len(param_names)

    # Build single-line header for length check
    header_prefix = "func"
    if lambda_name:
        header_prefix += f" {lambda_name}"
    single_line_params_str = "(" + ", ".join(param_names) + ")"
    full_header_single = f"{header_prefix}{single_line_params_str}"
    if has_return_type:
        full_header_single += f" -> {return_type_str}"
    full_header_single += ":"

    params_fit_on_single_line = len(full_header_single) <= max_len

    # Calculate indentation for body (one level deeper than current indent)
    if ec.prefix_string:
        base_indent_count = 0
        try:
            base_indent_count = ec.prefix_string.count("\t") + ec.prefix_string.count(
                "    "
            )
        except Exception:
            pass
        body_indent_level = max(base_indent_count, 1)
    else:
        body_indent_level = 1
    body_indent = "".join([indent_string] * body_indent_level)
    # Body statements sit exactly one level deeper than the lambda header, which is at
    # `indent_string`. (Using indent_string * (level + 1) would over-indent when
    # indent_string is already the header's level, e.g. a lambda passed as a call arg.)
    inner_body_indent = indent_string + "\t"

    # Build body lines by formatting the lambda's suite as a statement block, so the
    # body gets the same treatment as a function body (arrays/dicts/compound statements,
    # `return` spacing and redundant-paren removal) instead of a raw expression_to_str
    # of each statement.
    body_lines = []
    try:
        from .block import format_block as _fmt_block
        from .class_statement import (
            _find_suite_or_empty as _fse,
            _format_func_body_statement as _ffbs,
        )

        _suite_items, _orphans = _fse(lambda_tree, extract_orphans=False)
        if _suite_items:
            _body_units = (
                len(inner_body_indent)
                if ("\t" in inner_body_indent)
                else max(1, len(inner_body_indent) // 4)
            )
            body_lines, _ = _fmt_block(
                _suite_items, _ffbs, inner_body_indent, 4 * _body_units, style
            )
    except Exception:
        body_lines = []

    if not body_lines and suite_children:
        # Fallback for shapes _find_suite_or_empty does not recognise (e.g. a suite of
        # bare tokens): keep the previous raw rendering.
        for sc in suite_children:
            if sc is None:
                continue
            d_name = (
                str(getattr(sc.data, "value", getattr(sc, "data", "")))
                if hasattr(sc, "data")
                else ""
            )
            if isinstance(sc, Token):
                ttype = str(sc.type).lower()
                if ttype in ("_nl", "_indent", "_dedent", "end"):
                    continue
                val = getattr(sc, "value", "")  # type: ignore
                if val:
                    body_lines.append((None, f"{inner_body_indent}{val}"))
            elif d_name == "pass_stmt":
                body_lines.append((None, f"{inner_body_indent}pass"))
            elif hasattr(sc, "data") and data_name not in ("", None):
                try:
                    expr_str = expression_to_str_safe(sc)
                    if expr_str.strip():
                        body_lines.append((None, f"{inner_body_indent}{expr_str}"))
                except Exception:
                    pass

    # A comment inside the parameter list cannot be written on one line, so it forces the
    # split header. The layout is the same rule as a function or signal header - each
    # comment stays on the line it was written on - so it goes through the same helper.
    if params_tree is not None and any(
        isinstance(c, Tree) and _tree_data_str(c) == "comment"
        for c in getattr(params_tree, "children", [])
    ):
        from .class_statement import _break_param_list

        close_text = f") -> {return_type_str.strip()}:" if has_return_type else "):"
        # The header starts on the lambda's own source line, which is what decides whether
        # the first comment was written on that line (`ec.prefix_line` is the enclosing
        # expression's, which can be a different line entirely).
        result = _break_param_list(
            params_tree,
            f"{indent_string}{ec.prefix_string}{header_prefix}(",
            _node_line(lambda_tree),
            close_text,
            indent_string,
            style,
        )
        if body_lines:
            result.extend(body_lines)
        else:
            result.append((None, f"{inner_body_indent}pass"))
        return result

    # A header that fits is written on one line however many parameters it has: the
    # parameter list is only split when the header itself is too long.
    if params_fit_on_single_line and num_params <= 3:
        # Header stays inline; the body still needs its own lines.
        header_prefix_text = ec.prefix_string if ec.prefix_string else ""
        result = [
            (
                ec.prefix_line or None,
                f"{indent_string}{header_prefix_text}{full_header_single}",
            )
        ]
        if body_lines:
            result.extend(body_lines)
        else:
            result.append((None, f"{body_indent}pass"))
        return result

    # Multi-line format: split params and format body as block
    result = []

    # Build prefix for all lines (e.g., "var x1 = func")
    full_prefix = ec.prefix_string if ec.prefix_string else f"func{lambda_name}("
    header_with_prefix = f"{full_prefix}{header_prefix}("  # e.g. "var x1 = func("
    result.append((ec.prefix_line or None, f"{indent_string}{header_with_prefix}"))

    if not param_names:
        # Nothing to split, so the header's own line pair carries the `)`.
        result.append(
            (
                None,
                f"{body_indent}){f' -> {return_type_str.strip()}' if has_return_type else ''}:",
            )
        )

    # Param lines with commas (except last). The parameter list continues the lambda header's
    # line, so it gets the guide's two indent levels; the body below is a block and keeps the
    # one `inner_body_indent` gives it.
    param_indent = continuation_indent(indent_string)
    for pi, pname in enumerate(param_names):
        is_last = pi == len(param_names) - 1
        if not is_last:
            result.append((None, f"{param_indent}{pname},"))
        elif has_return_type:
            # Last param with return type — name on one line, ) -> type: on next
            result.append((None, f"{param_indent}{pname}"))
            return_type_line = f") -> {return_type_str.strip()}:"
            result.append((None, f"{body_indent}{return_type_line}"))
        else:
            # Last param without return type — name, then `):` on its own line
            result.append((None, f"{param_indent}{pname}"))
            result.append((None, f"{body_indent}):"))

    # Body lines
    if body_lines:
        result.extend(body_lines)
    else:
        result.append((None, f"{inner_body_indent}pass"))
    return result


def _format_not_test_multi(not_expr, ec, indent, style=None):
    op = expression_to_str_safe(not_expr.children[0]) if not_expr.children else ""  # type: ignore
    expr = (
        expression_to_str_safe(not_expr.children[1])
        if len(not_expr.children) > 1
        else "?"  # type: ignore
    )
    return [(ec.prefix_line, f"{indent}{op} {expr}")]


def get_end_line_safe(node):
    try:
        from ..common.utils import get_end_line as _gel

        if hasattr(node, "meta"):
            return node.meta.end_line
        return _gel(node)
    except Exception:
        return 1
