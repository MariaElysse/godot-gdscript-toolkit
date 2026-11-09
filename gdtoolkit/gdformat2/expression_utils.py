from typing import List, Optional

from lark import Tree, Token


def continuation_indent(indent_string: str) -> str:
    """The indent for a line that continues the expression opened on an earlier line.

    The guide: "Since continuation lines are still part of the same expression, 2 indent levels
    should be used instead of one." Arrays, dictionaries and enums are the exception and keep a
    single level, which is what `collection_indent` is for. A block body - a suite, a lambda's
    body, an accessor list - is not a continuation and stays at one level too.
    """
    return indent_string + "\t\t"


def collection_indent(indent_string: str) -> str:
    """The indent for the items of an array, a dictionary or an enum: a single level.

    The guide calls these out as the exception to the two levels a continuation line gets,
    "to distinguish continuation lines" from the items of a collection.
    """
    return indent_string + "\t"


def continuation_indent_for(
    indent_string: str, content_width: int, limit: int, tab_size: int = 4
) -> str:
    """The continuation indent for a line `content_width` wide, one level when two would overflow.

    The guide asks for two levels on a continuation line and also to keep every line under the
    limit; the limit wins where they conflict, because a long line is the one a reader suffers
    from. A layout whose content fits on one line inside parentheses at one level but not at two
    has to ask for that here, or its own fit test rejects the block it just built and it falls
    back to a form with a line over the limit - which is what `long_boolean_test_expressions`
    showed for a chain of `not`s.
    """
    two_levels = continuation_indent(indent_string)
    if len(two_levels.replace("\t", " " * tab_size)) + content_width <= limit:
        return two_levels
    return collection_indent(indent_string)


def remove_outer_parentheses(expression) -> object:
    if isinstance(expression, Tree) and str(expression.data).lower() == "par_expr":  # type: ignore
        return remove_outer_parentheses(expression.children[0])
    return expression


def _strip_primary_parens(expression) -> object:
    """If `expression` is a `primary` of the form ``( expression )``, return the inner
    expression; otherwise return it unchanged."""
    while isinstance(expression, Tree):
        raw = expression.data
        d = str(getattr(raw, "value", raw)).lower()
        if d != "primary":
            break
        children = [c for c in getattr(expression, "children", []) if c is not None]
        if (
            len(children) >= 3
            and isinstance(children[0], Token)
            and str(getattr(children[0], "type", "")).upper() == "PARENTHESIS_OPEN"
            and isinstance(children[-1], Token)
            and str(getattr(children[-1], "type", "")).upper() == "PARENTHESIS_CLOSE"
        ):
            expression = children[1]
            continue
        break
    return expression


_WRAPPER_DATA = ("assignment", "cast", "ternary", "type_test")


def strip_wrapped_outer_parentheses(expression) -> object:
    """Descend pass-through wrapper nodes (assignment/cast/ternary/type_test) and strip the
    `( expression )` primary at the bottom, rebuilding the wrapper chain. Returns the input
    unchanged when there is no redundant paren wrap.

    Repeated until nothing more comes off, because each group re-wraps its content in the same
    chain of wrapper nodes, so `((a + b))` is two groups on one slot and needs two removals.

    Only call this where the parentheses are genuinely redundant, i.e. at the whole
    expression's root (e.g. `return (a or b)`); stripping parens that sit under an
    operator would change semantics.
    """
    while True:
        stripped = _strip_one_wrapper_paren_layer(expression)
        if stripped is expression:
            return expression
        expression = stripped


def _strip_one_wrapper_paren_layer(expression) -> object:
    if not isinstance(expression, Tree):
        return expression
    raw = expression.data
    d = str(getattr(raw, "value", raw)).lower()
    if d in _WRAPPER_DATA:
        # A wrapper is pass-through only when it carries nothing else. The grammar wraps every
        # expression in assignment/cast/ternary/type_test, and those scaffolding layers have a
        # single child with the rest of their slots empty. A `cast` holding an `as` token, or a
        # `type_test` holding `is`, is a real operator and the group under it is its left
        # operand: stripping `(a or b) as int` to `a or b as int` re-binds the `or` to the
        # second operand, so it must not happen. The same goes for a ternary with its condition
        # or an assignment with its `=`.
        non_none = [c for c in getattr(expression, "children", []) if c is not None]
        if len(non_none) != 1 or not isinstance(non_none[0], Tree):
            return expression
        new_c = _strip_one_wrapper_paren_layer(non_none[0])
        if new_c is non_none[0]:
            return expression
        children = list(getattr(expression, "children", []))
        for i, c in enumerate(children):
            if c is non_none[0]:
                children[i] = new_c
                break
        return Tree(expression.data, children, getattr(expression, "meta", None))
    return _strip_primary_parens(expression)


def strip_slot_parentheses(
    expression, prefix_text: str, indent_string: str, style
) -> object:
    """Strip a redundant paren group that spans a whole slot, e.g. `var x = (1 + 2)`.

    Only when the value still fits on one line without the group. The guide asks to avoid
    parentheses that are not needed "for order of operations or wrapping over multiple
    lines", and a value that does not fit on one line has a use for them: the group *is* the
    wrapping form, and removing it makes the chain layout break every segment onto its own
    line instead of reproducing the author's wrap. Nothing outside the group can bind to what
    is inside it, so when the value does fit the removal is always safe.

    Call this only where the group is the entire expression of its slot - a declaration's
    value, a condition, an assignment's right-hand side, an argument, a parameter default.
    Never for an operand of an operator (there the group is load-bearing: `(a + b) * c`) or
    for a match pattern (`(1):` is a pattern group, not a value).
    """
    stripped = strip_wrapped_outer_parentheses(expression)
    if stripped is expression:
        return expression
    from .expression import _is_forced_multiline
    from .expression_to_str import expression_to_str

    if _is_forced_multiline(stripped, style):
        return expression
    try:
        text = expression_to_str(stripped)
    except Exception:
        return expression
    limit = style.line_length if style else 100
    if len(f"{indent_string}{prefix_text}{text}".replace("\t", "    ")) > limit:
        return expression
    return stripped


def is_foldable(expression) -> bool:
    non_foldable = [
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
    data = str(expression.data) if not hasattr(expression.data, "value") else expression.data.value  # type: ignore
    return data not in non_foldable


def has_trailing_comma(expression) -> bool:
    """Detect trailing comma in arrays/dicts for both old and new grammar."""
    if (
        not isinstance(expression, Tree)
        or len(getattr(expression, "children", [])) == 0
    ):
        return False
    last = getattr(expression, "children", [None])[-1]
    # Check for trailing comma as a tree node (old grammar)
    if isinstance(last, Tree) and _tree_data_str(last) == "trailing_comma":
        return True
    # Also check for plain COMMA token at the end (new grammar direct children)
    non_none_children = [
        c for c in getattr(expression, "children", []) if c is not None
    ]
    if non_none_children:
        last_non_none = non_none_children[-1]
        if (
            isinstance(last_non_none, Token)
            and str(getattr(last_non_none, "type", "")).upper() == "COMMA"
        ):
            return True

    # For new grammar array_literal/dict_literal/enum_body, check inside elements/subtree.
    # In the Earley grammar, trailing commas are nested inside 'elements' or similar subtrees:
    #   array_literal -> [Token, Tree(elements), Token]
    #   elements      -> [expr1, COMMA, expr2, COMMA, ..., exprN, COMMA, None]
    # Match patterns use array_pattern_items / dict_pattern_items instead, and mirror
    # the literals, so they need the same trailing-comma detection.
    for child in getattr(expression, "children", []):
        if isinstance(child, Tree):
            d = _tree_data_str(child)
            if d in (
                "elements",
                "dict_entries",
                "array_pattern_items",
                "dict_pattern_items",
            ):
                non_none_elems = [
                    c for c in getattr(child, "children", []) if c is not None
                ]
                if len(non_none_elems) >= 2:
                    last_elem = non_none_elems[-1]
                    second_last = non_none_elems[-2]
                    # Trailing comma pattern: ..., COMMA, None (last elem before trailing comma is COMMA)
                    if (
                        isinstance(last_elem, Token)
                        and str(getattr(last_elem, "type", "")).upper() == "COMMA"
                    ):
                        return True
                    # Alternative: last two are COMMA, expression - check if last is also COMMA
                    # This handles the case where trailing comma sits directly after elements
    return False


def is_trailing_comma(expression) -> bool:
    return isinstance(expression, Tree) and _tree_data_str(expression) == "trailing_comma"  # type: ignore


def is_any_comma(expression) -> bool:
    if isinstance(expression, Tree):
        return _tree_data_str(expression) == "trailing_comma"
    return isinstance(expression, Token) and expression.type == "COMMA"  # type: ignore


def is_any_parentheses(expression) -> bool:
    return isinstance(expression, Token) and expression.type in ["LPAR", "RPAR"]  # type: ignore


def _is_multiline_string(expression) -> bool:
    if not isinstance(expression, Tree):
        return False
    data_str = str(expression.data).lower() if hasattr(expression.data, "value") else expression.data  # type: ignore
    if data_str != "string":
        return False
    first = getattr(expression, "children", [None])[0]
    return isinstance(first, Token) and first.type == "LONG_STRING"  # type: ignore


def _tree_data_str(node):
    from lark import Tree as LarkTree

    if isinstance(node, LarkTree):
        d = node.data
        return str(d) if not hasattr(d, "value") else d.value  # type: ignore
    return ""
