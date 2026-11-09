from typing import Dict, Callable

from lark import Tree, Token

from ..common.utils import get_line, get_end_line
from .context import ExpressionContext
from .expression_utils import continuation_indent_for
from .style_config import StyleConfig
from .types import Outcome
from .expression import format_expression
from .statement_utils import format_simple_statement


def format_var_statement(
    statement: Tree, indent_string: str, indent_level: int, style=None, prefix: str = ""
) -> Outcome:
    """Format a variable declaration.

    Handles both class-level variables (variable_decl) and local variables
    (local_variable in the new grammar).

    New grammar structure for local_variable: VAR NAME [EQUAL expression] | VAR NAME COLON type [EQUAL expression]
    """
    stmt_data = _tree_data_str(statement).lower()

    # A property (`var x: set = _set, get = _get`) renders as a header plus one accessor
    # per line, so it takes its own path before the plain declaration shapes below. The
    # inline list arrives as a `property`/`untitled_property` child; the block form puts
    # the `property_accessor_line`s straight on the declaration.
    prop_child = None
    has_accessor_lines = False
    for _child in getattr(statement, "children", []):
        if not hasattr(_child, "data"):
            continue
        _d = _tree_data_str(_child).lower()
        if _d in ("property", "untitled_property"):
            prop_child = _child
            break
        if _d == "property_accessor_line":
            has_accessor_lines = True
    if prop_child is not None or has_accessor_lines:
        return _format_var_property(
            statement, prop_child, indent_string, indent_level, style, prefix
        )

    # Determine what kind of variable this is based on children
    has_colon_after_name = False
    has_type = False
    has_equals = False
    uses_inference_operator = False  # Track if := was detected

    tokens_in_order = []
    for child in getattr(statement, "children", []):  # type: ignore
        if isinstance(child, Token):
            t_str = str(_tree_data_str(child)).lower()
            tokens_in_order.append(t_str)

    # Detect structure from token order - BUT handle := (colon-equals inference operator)
    # := means "typed with inferred value" - treat as untyped assignment (=) not typed (:)
    colon_pos = -1
    equal_pos = -1
    for i, t in enumerate(tokens_in_order):
        if t == "colon":
            colon_pos = i
        elif t == "equal":
            equal_pos = i
            break

    # If COLON comes immediately before EQUAL (no type between them), it's := inference
    has_colon_after_name = False
    if colon_pos >= 0 and equal_pos > colon_pos:
        # Check if there's a non-token child between colon and equal (= the type)
        # Also check children[colon_pos+1] directly since trees can sit at that position
        children_list = list(getattr(statement, "children", []))
        has_type_between = False
        for i in range(colon_pos + 1, equal_pos):
            c = children_list[i] if i < len(children_list) else None
            if c is not None and hasattr(c, "data"):
                has_type_between = True
                break
        # Also check the child immediately after COLON token position (may be a Tree)
        if colon_pos + 1 < len(children_list):
            direct_child = children_list[colon_pos + 1]
            if direct_child is not None and hasattr(direct_child, "data"):
                has_type_between = True

        # If no type between colon and equals, treat as := inference (preserved in output)
        if not has_type_between:
            uses_inference_operator = True  # Mark that we detected :=
            tokens_in_order = [
                t
                for t in tokens_in_order[:colon_pos]
                + ["equal"]
                + tokens_in_order[equal_pos + 1 :]
            ]
        else:
            has_colon_after_name = True
    elif colon_pos >= 0:
        # Colon exists somewhere - determine if it's a type declaration or := inference
        if equal_pos < 0 or colon_pos < equal_pos:
            # Check if there's actually a type between colon and equals (or no equals)
            children_list = list(getattr(statement, "children", []))
            has_type_between = False
            for i in range(
                colon_pos + 1,
                min(
                    equal_pos if equal_pos >= 0 else len(children_list),
                    len(children_list),
                ),
            ):
                c = children_list[i] if i < len(children_list) else None
                if c is not None and hasattr(c, "data"):
                    has_type_between = True
                    break
            # Also check the child immediately after COLON token position (may be a Tree)
            if colon_pos + 1 < len(children_list):
                direct_child = children_list[colon_pos + 1]
                if direct_child is not None and hasattr(direct_child, "data"):
                    has_type_between = True

            if has_type_between:
                # COLON followed by a type annotation - it's a typed declaration
                has_colon_after_name = True

    for t in tokens_in_order:
        if t == "equal":
            has_equals = True
            break

    name_token = None
    type_token = None
    expr_child = None

    children_list = list(getattr(statement, "children", []))  # type: ignore

    # Find NAME token for variable name
    for child in getattr(statement, "children", []):  # type: ignore
        if isinstance(child, Token) and str(_tree_data_str(child)).lower() == "name":  # type: ignore
            name_token = child.value  # type: ignore
            break

    # If no direct NAME token found in statement children, search deeper
    if not name_token:
        for child in getattr(statement, "children", []):  # type: ignore
            if hasattr(child, "data") and str(_tree_data_str(child)).lower() == "type":
                continue
            elif isinstance(child, Token) and hasattr(child, "value"):
                v = str(child.value).lower()  # type: ignore
                if v in ("var", "static"):
                    continue
                name_token = child.value  # type: ignore
                break

    # Find expression after EQUAL token for untyped declarations (VAR NAME = expr)
    if not has_colon_after_name and not expr_child:
        for i, c in enumerate(children_list):
            if isinstance(c, Token) and str(_tree_data_str(c)).lower() == "equal":
                # The expression is the next child after EQUAL (could be a Tree)
                expr_child = (
                    children_list[i + 1] if i + 1 < len(children_list) else None
                )
                break

    if has_colon_after_name:
        # Typed variable declaration: VAR NAME COLON TYPE [EQUAL expr]
        type_idx = (
            tokens_in_order.index("colon") + 1 if "colon" in tokens_in_order else -1
        )
        equal_idx = tokens_in_order.index("equal") if "equal" in tokens_in_order else -1

        # Find the actual TYPE token (comes after COLON)
        children_list = list(getattr(statement, "children", []))  # type: ignore
        for i, c in enumerate(children_list):
            if isinstance(c, Token) and str(_tree_data_str(c)).lower() == "colon":
                type_token = (
                    children_list[i + 1] if i + 1 < len(children_list) else None
                )
                break

        # Find expression after EQUAL (if present)
        for i, c in enumerate(children_list):
            if isinstance(c, Token) and str(_tree_data_str(c)).lower() == "equal":
                expr_child = (
                    children_list[i + 1] if i + 1 < len(children_list) else None
                )
                break

    if has_colon_after_name and type_token is not None:
        # Typed variable (with or without assignment)
        try:
            from .expression import format_expression as fe

            ctx = ExpressionContext(
                "", get_line_safe(type_token), "", get_end_line_safe(type_token)
            )
            result = fe(type_token, ctx, "", style)
            type_str = (
                result[0][0][1] if result and result[0] else _safe_str(type_token)
            )
        except Exception:
            type_str = _safe_str(type_token)
        if expr_child is not None:
            return _format_typed_assigned(
                name_token, type_token, expr_child, indent_string, style, prefix
            )
        else:
            return format_simple_statement(
                f"{prefix}var {name_token}: {type_str}",
                statement,
                indent_string,
                indent_level,
            )

    if has_equals and expr_child is not None:
        # Un-typed with assignment: VAR NAME = EXPR or VAR NAME := EXPR (inference)
        op = ":=" if uses_inference_operator else "="
        return _format_assigned(
            name_token, expr_child, indent_string, style, prefix, operator=op
        )

    elif has_equals and name_token:
        # Un-typed without value: just "var x" (inferred default)
        return format_simple_statement(
            f"{prefix}var {name_token}", statement, indent_string, indent_level
        )

    elif name_token:
        return format_simple_statement(
            f"{prefix}var {name_token}", statement, indent_string, indent_level
        )

    # Fallback
    return ([], 0)


def _safe_str(node):
    """Get string value from a node safely.

    For Tree nodes (like `type`, `return_type`), extracts the token child's value.
    For Token nodes, returns the token value directly.
    Skips tokens that are syntax markers (COLON, EQUAL) - those indicate
    structure, not actual type/values to display.
    """
    if isinstance(node, Token):
        ttype = str(getattr(node, "type", "")).lower()
        # Skip syntax marker tokens - they don't carry semantic value
        if ttype in ("colon", "equal"):  # These are structural, not type values
            return ""
        return str(node.value)  # type: ignore
    elif hasattr(node, "value"):
        val = str(node.value)
        if val in (":", "="):
            return ""
        return val  # type: ignore
    elif hasattr(node, "data") and hasattr(node, "children"):
        data_str = getattr(getattr(node, "data", None), "value", str(node.data))
        children = list(getattr(node, "children", []))
        non_none_children = [c for c in children if c is not None]

        # For `type` trees: use expression_to_str to handle generics like Array[int]
        if data_str == "type" or data_str == "return_type":
            try:
                from .expression import format_expression  # noqa: F401 local import

                return _type_tree_to_str(node)
            except Exception:
                pass

        # If exactly one non-None child that's a Token, use its value
        if len(non_none_children) == 1 and isinstance(non_none_children[0], Token):
            tok = non_none_children[0]
            ttype = str(getattr(tok, "type", "")).lower()
            if ttype in ("colon", "equal"):
                return ""
            return str(tok.value)

        # For Tree nodes like `type`, look for NAME token or other meaningful value
        for child in non_none_children:
            result = _safe_str(child)
            if result and not result.startswith("<"):
                return result
    d = getattr(node, "data", None)
    return f"<{str(d)}>"


def _type_tree_to_str(node, style=None):
    """Format a type tree node (like Array[int], SubClass.NamedEnum).

    Handles both simple types and generic/complex types with brackets.
    Delegates to expression_to_str for proper bracket handling.
    """
    try:
        from .expression import format_expression as fe

        ctx = ExpressionContext("", get_line_safe(node), "", get_end_line_safe(node))
        result = fe(node, ctx, "", style)
        lines = result[0]
        return lines[0][1] if lines else str(getattr(node, "data", "?"))
    except Exception:
        # Fallback: just get the name tokens
        parts = []
        for child in getattr(node, "children", []):
            if isinstance(child, Token) and str(child.type).lower() not in (
                "bracket_open",
                "bracket_close",
            ):
                val = str(getattr(child, "value", ""))
                if val:
                    parts.append(val)
        return "".join(parts)


def _wrap_value_in_parens(lines, indent_string, prefix_text, style):
    """Put a value that cannot be broken on its own line, in parentheses.

    A value that is a single unbreakable token - a long raw string, a node path - leaves the
    statement over the line limit however the expression layout tries. The guide asks for
    lines under 100 characters and allows parentheses "for wrapping over multiple lines", so
    the value moves one level in with `(` and `)` around it. Returns None when that would not
    help, i.e. when the header or the value line would still be too wide.
    """
    from .class_statement import _display_width

    limit = style.line_length if style else 100
    if len(lines) != 1 or not lines[0][1].startswith(prefix_text):
        return None
    if _display_width(lines[0][1]) <= limit:
        return None
    value_text = lines[0][1][len(prefix_text) :]
    header = f"{prefix_text}("
    # The value sits two levels in, or one when two would not leave it under the limit: this
    # exists to *fix* a line that is too wide, so it must not decline the wrap the guide allows
    # for exactly that. A chain of `not`s is one unbreakable operand and needs the one level.
    inner_indent = continuation_indent_for(
        indent_string, _display_width(value_text), limit, style.tab_size if style else 4
    )
    inner = f"{inner_indent}{value_text}"
    if _display_width(header) > limit or _display_width(inner) > limit:
        return None
    return [(lines[0][0], header), (None, inner), (None, f"{indent_string})")]


def _strip_slot_parens(expr, prefix_text, indent_string, style):
    """Remove a redundant paren group that spans a declaration's whole value.

    `var x = (1 + 2)` is `var x = 1 + 2`: the guide asks to avoid parentheses that are not
    needed for order of operations or for wrapping over multiple lines. A group that is an
    operand of the value's own operator chain (`(a + b) * c`) is not a whole slot and is left
    alone, and so is a group a value needs in order to wrap - see
    `expression_utils.strip_slot_parentheses`.
    """
    from .expression_utils import strip_slot_parentheses

    return strip_slot_parentheses(expr, prefix_text, indent_string, style)


def _format_assigned(name, expr, indent_string, style, prefix="", operator="="):
    ec_str = f"{prefix}var {name} {operator} "
    expr = _strip_slot_parens(expr, ec_str, indent_string, style)
    expression_context = ExpressionContext(
        ec_str, get_line_safe(expr), "", get_end_line_safe(expr)
    )
    lines, end_line = format_expression(expr, expression_context, indent_string, style)
    wrapped = _wrap_value_in_parens(
        lines, indent_string, f"{indent_string}{ec_str}", style
    )
    return (wrapped if wrapped is not None else lines, end_line)


def _format_typed_assigned(name, type_node, expr, indent_string, style, prefix=""):
    try:
        from .expression import format_expression as fe

        ctx = ExpressionContext(
            "", get_line_safe(type_node), "", get_end_line_safe(type_node)
        )
        result = fe(type_node, ctx, "", style)
        type_str = result[0][0][1] if result and result[0] else _safe_str(type_node)
    except Exception:
        type_str = _safe_str(type_node)
    ec_str = f"{prefix}var {name}: {type_str} = "
    expr = _strip_slot_parens(expr, ec_str, indent_string, style)
    expression_context = ExpressionContext(
        ec_str, get_line_safe(expr), "", get_end_line_safe(expr)
    )
    lines, end_line = format_expression(expr, expression_context, indent_string, style)
    wrapped = _wrap_value_in_parens(
        lines, indent_string, f"{indent_string}{ec_str}", style
    )
    return (wrapped if wrapped is not None else lines, end_line)


def _var_declaration_head(statement, prefix) -> str:
    """`var NAME[: type][ = value]` for a declaration that carries a property.

    Everything before the accessor list, with the same spelling the source used (`:=`
    for an inferred type, `: type` for an explicit one).
    """
    from .expression_to_str import expression_to_str as _ets

    kids = [c for c in getattr(statement, "children", []) if c is not None]
    name = ""
    type_str = ""
    value_str = ""
    inference = False
    index = 0
    while index < len(kids):
        child = kids[index]
        if isinstance(child, Token):
            t = str(child.type).upper()
            if t == "NAME" and not name:
                name = str(child.value)
            elif t == "COLON":
                nxt = kids[index + 1] if index + 1 < len(kids) else None
                if isinstance(nxt, Token) and str(nxt.type).upper() == "EQUAL":
                    inference = True
                    index += 1
            index += 1
            continue
        d = _tree_data_str(child).lower()
        if d == "type":
            type_str = _ets(child)
        elif d in ("property", "untitled_property", "property_accessor_line"):
            break
        else:
            value_str = _ets(child)
        index += 1
    head = f"{prefix}var {name}"
    if type_str:
        head += f": {type_str}"
    if value_str:
        head += f" := {value_str}" if inference else f" = {value_str}"
    return head


def _accessor_entry(acc, comma, explicit_name=None):
    """Split one accessor into `(name, param, suite, target, had_parens, comma)`.

    `set = __set` has a target and no suite; `get: pass` and `set(x): pass` have a suite
    and the parenthesised parameter when the source wrote one.
    """
    name = explicit_name or ""
    param = ""
    suite = None
    target = ""
    had_parens = False
    in_parens = False
    for child in getattr(acc, "children", []):
        if child is None:
            continue
        if isinstance(child, Token):
            t = str(child.type).upper()
            if t in ("NAME", "GET_SET"):
                if not name:
                    name = str(child.value)
                elif in_parens and not param:
                    param = str(child.value)
                else:
                    target = str(child.value)
            elif t == "PARENTHESIS_OPEN":
                in_parens = True
                had_parens = True
            elif t == "PARENTHESIS_CLOSE":
                in_parens = False
        elif hasattr(child, "data"):
            suite = child
    return (name, param, suite, target, had_parens, comma)


def _property_accessor_entries(prop):
    """One entry per accessor of a `property` / `untitled_property`, in source order."""
    entries = []
    pending_name = None  # the contextual keyword of an `untitled_property`
    for child in getattr(prop, "children", []):
        if isinstance(child, Token):
            t = str(child.type).upper()
            if t == "GET_SET" and pending_name is None:
                pending_name = str(child.value)
            elif t == "COMMA" and entries:
                entries[-1] = entries[-1][:5] + (True,)
            continue
        if not hasattr(child, "data"):
            continue
        d = _tree_data_str(child)
        if d == "property_accessor_line":
            kids = [c for c in getattr(child, "children", []) if c is not None]
            acc = next((c for c in kids if hasattr(c, "data")), None)
            comma = any(
                isinstance(c, Token) and str(c.type).upper() == "COMMA" for c in kids
            )
            if acc is not None:
                entries.append(_accessor_entry(acc, comma, pending_name))
                pending_name = None
        elif d == "property_accessor":
            entries.append(_accessor_entry(child, False, pending_name))
            pending_name = None
        elif d == "property_accessor_body":
            entries.append(_accessor_entry(child, False, pending_name))
            pending_name = None
    return entries


def _property_layout(container):
    """`('item', (accessor_node, explicit_name), comma)` / `('comment', node, False)`.

    The accessors and comments of a property in source order, so a comment can be placed
    on the line it was written on rather than being hoisted out of the declaration.
    """
    entries = []
    pending_name = None  # the contextual keyword of an `untitled_property`
    for child in getattr(container, "children", []):
        if isinstance(child, Token):
            t = str(child.type).upper()
            if t == "GET_SET" and pending_name is None:
                pending_name = str(child.value)
            elif t == "COMMA" and entries and entries[-1][0] == "item":
                kind, payload, _comma = entries[-1]
                entries[-1] = (kind, payload, True)
            continue
        if not hasattr(child, "data"):
            continue
        d = _tree_data_str(child)
        if d == "comment":
            entries.append(("comment", child, False))
        elif d == "property_accessor_line":
            kids = [c for c in getattr(child, "children", []) if c is not None]
            acc = next(
                (
                    c
                    for c in kids
                    if hasattr(c, "data")
                    and _tree_data_str(c) not in ("comment", "end")
                ),
                None,
            )
            comma = any(
                isinstance(c, Token) and str(c.type).upper() == "COMMA" for c in kids
            )
            if acc is not None:
                entries.append(("item", (acc, pending_name), comma))
                pending_name = None
        elif d in ("property_accessor", "property_accessor_body"):
            entries.append(("item", (child, pending_name), False))
            pending_name = None
    return entries


def _format_var_property(
    statement, prop, indent_string, indent_level, style, prefix
) -> Outcome:
    """`var x: set = _set, get = _get` renders as a header plus one accessor per line.

    `prop` is the `property`/`untitled_property` child for the inline list, or None for
    the block form, whose accessors sit directly on the declaration. A comment in the list
    stays on the line it was written on, on the accessor's line when it follows one.
    """
    from .block import format_block
    from .class_statement import _find_suite_or_empty, _format_func_body_statement
    from .expression import _comment_layout_text, _node_end_line, _node_line

    container = prop if prop is not None else statement
    lines = [
        (
            get_line_safe(statement),
            f"{indent_string}{_var_declaration_head(statement, prefix)}:",
        )
    ]
    inner = indent_string + "\t"
    stmt_col = getattr(getattr(statement, "meta", None), "column", 0) or 0
    prev_line = _node_line(statement)
    for kind, payload, comma in _property_layout(container):
        if kind == "comment":
            text = _comment_layout_text(payload)
            c_line = _node_line(payload)
            c_col = getattr(getattr(payload, "meta", None), "column", 0) or 0
            # A comment written at the declaration's own column belongs to the enclosing
            # scope, so it goes back out to that column rather than the accessor indent.
            at_declaration_level = bool(stmt_col and c_col and c_col <= stmt_col)
            if not at_declaration_level and lines and c_line and c_line == prev_line:
                ln, txt = lines[-1]
                lines[-1] = (ln, f"{txt}  {text}")
            else:
                # A blank the author wrote in front of the comment survives, as it does for
                # a standalone comment anywhere else. An inline accessor list spans several
                # source lines once it is one accessor per line, so the gap measured here is
                # a real one - dropping it made the second pass lose a blank the golden has.
                if prev_line and c_line and c_line - prev_line > 1:
                    lines.append((None, ""))
                lines.append(
                    (None, f"{indent_string if at_declaration_level else inner}{text}")
                )
            prev_line = c_line
            continue
        acc, explicit_name = payload
        name, param, suite, target, had_parens, _comma = _accessor_entry(
            acc, comma, explicit_name
        )
        suffix = "," if comma else ""
        if suite is None:
            lines.append((None, f"{inner}{name} = {target}{suffix}"))
            prev_line = _node_end_line(acc)
            continue
        header = f"{inner}{name}{'(' + param + ')' if had_parens else ''}:{suffix}"
        items, accessor_orphans = _find_suite_or_empty(acc)
        # A comment written straight after the accessor's `:` belongs on that line.
        if (
            items
            and isinstance(items[0], tuple)
            and items[0][0] == "__comment__"
            and _node_line(items[0][1]) == _node_line(acc)
        ):
            header = f"{header}  {_comment_layout_text(items[0][1])}"
            items = items[1:]
        lines.append((None, header))
        if items:
            body_lines, _ = format_block(
                items,
                _format_func_body_statement,
                inner + "\t",
                indent_level + 8,
                style or StyleConfig(),
            )
            lines.extend(body_lines)
        # The line the body really ends on: the accessor's own span reaches past it into
        # the next line's hidden dedent token, which would hide the blank before an
        # orphan comment.
        body_end = 0
        for item in items:
            node = item[1] if isinstance(item, tuple) else item
            body_end = max(body_end, _node_line(node))
        prev_line = body_end or _node_end_line(acc)
        # A comment in the accessor's body that is indented shallower than the body belongs
        # to the enclosing scope. It is emitted here, in source order right after the
        # accessor it trailed, rather than with the rest at the end of the property: an
        # accessor list can hold several accessors, and a comment written between two of
        # them belongs between them - emitting it last moved it past the accessor that
        # followed, which then made the formatter unstable on its own output.
        for orphan in accessor_orphans or []:
            o_line = _node_line(orphan)
            if prev_line and o_line and o_line - prev_line > 1:
                lines.append((None, ""))
            o_col = getattr(getattr(orphan, "meta", None), "column", 0) or 0
            # Its own column says which enclosing scope it belongs to: one written at the
            # declaration's column goes back out to the declaration's indent, while one at
            # the accessor list's column stays on that list.
            o_indent = (
                indent_string if (stmt_col and o_col and o_col <= stmt_col) else inner
            )
            lines.append((None, f"{o_indent}{_comment_layout_text(orphan)}"))
            prev_line = o_line
    return (lines, get_end_line_safe(statement))


def _suite_owner(prop, name):
    """The accessor node of `prop` whose name is `name` (used to locate its suite)."""
    for child in getattr(prop, "children", []):
        if not hasattr(child, "data"):
            continue
        d = _tree_data_str(child)
        target = child
        if d == "property_accessor_line":
            target = next(
                (
                    c
                    for c in getattr(child, "children", [])
                    if c is not None and hasattr(c, "data")
                ),
                child,
            )
        if any(
            isinstance(c, Token)
            and str(c.type).upper() in ("NAME", "GET_SET")
            and str(c.value) == name
            for c in getattr(target, "children", [])
        ):
            return target
    return prop


def _tree_data_str(node) -> str:
    from lark import Tree as LarkTree

    if isinstance(node, LarkTree):
        d = node.data
        return str(d) if not hasattr(d, "value") else d.value  # type: ignore
    elif isinstance(node, Token):
        # For tokens, we want the token TYPE name
        return str(getattr(node, "type", ""))  # type: ignore
    return ""


def get_line_safe(node) -> int:
    try:
        from ..common.utils import get_end_line as _gel

        if hasattr(node, "meta"):
            return node.meta.end_line
        elif isinstance(node, Token):
            return 1
        return _gel(node)
    except Exception:
        return 1


def get_end_line_safe(node) -> int:
    try:
        from ..common.utils import get_end_line as _gel

        if hasattr(node, "meta"):
            return node.meta.end_line
        elif isinstance(node, Token):
            return 1
        return _gel(node)
    except Exception:
        return 1
