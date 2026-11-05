from lark import Tree, Token

from ..common.utils import get_line, get_end_line
from .var_statement import get_end_line_safe
from .context import ExpressionContext
from .types import Outcome
from .expression import format_expression


def format_const_statement(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format a constant declaration.
    
    New grammar structure (flat):
        constant_decl [CONST_token, NAME, COLON, EQUAL?, assignment_tree?]
    """
    stmt_data = _tree_data_str(statement)
    children_list = list(getattr(statement, 'children', []))

    # Emit standalone comment children first
    for child in getattr(statement, 'children', []):
        from lark import Tree as LarkTree
        if isinstance(child, LarkTree) and _tree_data_str(child) == "comment":
            comment_text = _get_comment_text(child)
            return ([(get_line(child), f"{indent_string}{comment_text}")], get_end_line(statement))  # type: ignore

    handlers = {
        "const_assigned": _format_const_statement,
        "const_typed_assigned": _format_const_statement,
        "const_inf": _format_const_statement,
    }
    handler = handlers.get(stmt_data) or _format_const_statement
    return handler(statement, children_list, indent_string, indent_level, style)


def _format_const_statement(
    statement: Tree, children_list, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format a constant declaration for the new flat grammar.
    
    Structure: [CONST_token('const'), NAME('C'), COLON(':'), EQUAL('=')?, assignment_tree]
    Or inference:  [CONST_token, NAME, COLON, EQUAL(=)] with type being None
    """
    name = None
    type_node = None
    expr_node = None
    has_colon_equal = False  # : = means := (inference)
    
    for i, c in enumerate(children_list):
        if c is None:
            continue
        if isinstance(c, Token) and str(getattr(c, 'type', '')).lower() == 'name':
            name = str(getattr(c, 'value', '?'))
        elif isinstance(c, Token) and str(getattr(c, 'type', '')).lower() in ('colon',):
            # Check if next child is EQUAL (meaning : = or :=)
            next_idx = i + 1
            if next_idx < len(children_list) and children_list[next_idx] is not None:
                nc = children_list[next_idx]
                if isinstance(nc, Token) and str(getattr(nc, 'type', '')).lower() == 'equal':
                    has_colon_equal = True
        elif isinstance(c, Token) and str(getattr(c, 'type', '')).lower() == 'equal' and not has_colon_equal:
            pass  # already handled by colon check above
        elif isinstance(c, Tree):
            d = str(getattr(getattr(c, 'data', None), 'value', getattr(c, 'data', '')))
            if d == 'type' and type_node is None:
                type_node = c
            elif d in ('assignment',) and expr_node is None:
                expr_node = c
    
    # Build type hint string using our improved method
    if type_node is not None and not has_colon_equal:
        try:
            from .var_statement import _safe_str
            type_hint = _safe_str(type_node)
        except Exception:
            type_hint = str(getattr(type_node, 'data', '?'))
    else:
        type_hint = ""

    # The slot's own line prefix, needed to judge whether the value fits on one line.
    if has_colon_equal:
        const_prefix = f"const {name} := "
    elif type_hint:
        const_prefix = f"const {name}: {type_hint} = "
    else:
        const_prefix = f"const {name} = "

    # A group spanning the whole value is redundant - the guide asks to avoid parentheses that
    # are not needed for order of operations or for wrapping over multiple lines, and nothing
    # outside the group can bind to what is inside it: `const C = (1 + 2)` is `const C = 1 + 2`.
    # An operand's own group (`(1 + 2) * 3`) stays.
    if expr_node is not None:
        from .expression_utils import strip_slot_parentheses
        expr_node = strip_slot_parentheses(expr_node, const_prefix, indent_string, style)

    # Format the expression value (skip through assignment/cast wrappers)
    from .expression_to_str import expression_to_str as ets
    try:
        if expr_node is not None:
            expr_str = ets(expr_node)
        else:
            expr_str = "?"
    except Exception:
        expr_str = "?"

    # Build output line - detect inference (:=) vs typed assignment (=)
    prefix = None
    if has_colon_equal and expr_str != "?":
        # Inference operator: const c := 1
        prefix = f"const {name} := "
    elif type_hint and expr_str != "?":
        prefix = f"const {name}: {type_hint} = "
    elif type_hint:
        return ([(get_line(statement), f"{indent_string}const {name}: {type_hint}")], get_end_line_safe(statement))
    else:
        prefix = f"const {name} = "

    # A value that does not fit on the line, or that must be multiline regardless of
    # length (a trailing comma anywhere inside), gets the expression layout (a broken
    # array, dict, call or preload) instead of one flattened line.
    max_len = style.line_length if style else 100
    from .expression import _is_forced_multiline
    if expr_node is not None and (
        _is_forced_multiline(expr_node, style)
        or len(f"{indent_string}{prefix}{expr_str}".replace('\t', '    ')) > max_len
    ):
        lines, end_line = format_expression(
            expr_node,
            ExpressionContext(prefix, get_line(statement), "", get_end_line_safe(statement)),
            indent_string, style,
        )
        return (lines, end_line)

    return ([(get_line(statement), f"{indent_string}{prefix}{expr_str}")], get_end_line_safe(statement))


def _tree_data_str(node):
    """Return a tree's rule name as a plain string."""
    from lark import Tree
    if isinstance(node, Tree):
        d = node.data
        return d.value if hasattr(d, 'value') else str(d)
    return ""


def _get_comment_text(comment_node) -> str:
    """Extract the text content from a comment tree node."""
    from lark import Tree
    from .comments import normalize_comment
    if not isinstance(comment_node, Tree):
        return ""
    for child in comment_node.children:
        return normalize_comment(str(child.value))  # type: ignore
    return ""
