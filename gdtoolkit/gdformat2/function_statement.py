from functools import partial
from typing import Dict, Callable, Any, List, Tuple

from lark import Tree, Token

from ..common.utils import get_line
from .context import ExpressionContext
from .expression_utils import continuation_indent
from .types import Outcome, FormattedLines
from .expression import format_expression
from .block import format_block
from .statement_utils import format_simple_statement


def _format_assert_statement(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format an assert statement with its expression arguments.

    Grammar: assert_stmt -> ASSERT PARENTHESIS_OPEN [expr] [COMMA expr] PARENTHESIS_CLOSE
    """
    from .expression_to_str import standalone_expression_to_str as _sets
    from .expression import _is_forced_multiline

    # Find expression children between the parens
    arg_nodes = []
    found_open = False
    for child in getattr(statement, 'children', []):
        if isinstance(child, Token):
            ttype = str(getattr(child, 'type', '')).lower()
            if ttype == "parenthesis_open":
                found_open = True
            elif ttype == "parenthesis_close":
                break  # Stop at closing paren
            continue

        child_data_str = _tree_data_str(child).lower() if hasattr(child, 'data') else ''

        if child is None or child_data_str in ("comment", "end"):
            continue

        if found_open:
            arg_nodes.append(child)

    # An argument is a whole slot between its own brackets, so a redundant group around all
    # of it comes off: `assert((1 + 2))` is `assert(1 + 2)`.
    from .expression_utils import strip_slot_parentheses
    arg_nodes = [strip_slot_parentheses(a, "assert(", indent_string, style) for a in arg_nodes]

    # An argument that must be a block of its own (a lambda with a multi-statement body,
    # here wrapped in parentheses and then called) cannot be flattened: `_sets` would put
    # its body on one line, which is not valid GDScript. Lay it out as a block instead.
    if (getattr(style, 'line_length', 100) is not None
            and any(_is_forced_multiline(a, style) for a in arg_nodes)):
        from .expression import _block_argument_lines
        lines = [(get_line_safe(statement), f"{indent_string}assert(")]
        lines.extend(
            _block_argument_lines(arg_nodes, get_line_safe(statement),
                                 continuation_indent(indent_string), style)
        )
        lines.append((None, f"{indent_string})"))
        return (lines, get_end_line_safe(statement))

    args = []
    for child in arg_nodes:
        try:
            args.append(_sets(child))
        except Exception:
            pass

    return format_simple_statement(f"assert({', '.join(args)})", statement, indent_string, indent_level)


# Placeholder function declarations (actual implementations are defined below)
# These allow imports to work before handlers are registered.


_HANDLERS: Dict[str, Callable] = {}


def format_func_statement(
    statement: Any, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format a statement within a function body."""
    
    # Unwrap tuple if needed (from block wrapper)
    actual_stmt = statement
    if isinstance(statement, tuple):
        tag, actual_stmt = statement  # type: ignore
    
    stmt_data = _tree_data_str(actual_stmt).lower()

    handler = _HANDLERS.get(stmt_data)
    if handler is None:
        return format_simple_statement(stmt_data.upper(), actual_stmt, indent_string, indent_level)
    
    return handler(actual_stmt, indent_string, indent_level, style)


def _format_function_call_expr(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format a function call expression (NAME followed by call_suffix).
    
    This is an explicit grammar rule added to handle function calls that the Earley
    parser might otherwise split across partial parse branches.
    """
    from .expression import format_expression
    from .context import ExpressionContext
    
    # statement.data = "function_call_expr" with children [NAME, call_suffix]
    func_name = None
    for child in getattr(statement, 'children', []):
        if isinstance(child, Token) and child.type == "NAME":  # type: ignore
            func_name = str(child.value)  # type: ignore
            break
    
    if func_name is None:
        return format_simple_statement("?", statement, indent_string, indent_level)

    # A comment inside the parentheses belongs on the line it was written on, which only
    # the expression layout knows how to do; `expression_to_str` would flatten it into the
    # argument list as if it were an argument. The same applies to an argument that forces
    # the call multiline (a lambda with a multi-statement body, a trailing comma): a
    # single-line `expression_to_str` renders such a lambda body as one flat line.
    from .expression import _has_comment_child, _is_forced_multiline
    if _has_comment_child(statement) or _is_forced_multiline(statement, style):
        lines, _end = format_expression(
            statement,
            ExpressionContext("", get_line_safe(statement), "", get_line_safe(statement)),
            indent_string, style,
        )
        return (lines, get_end_line_safe(statement))

    # Find call_suffix to get arguments
    args_str = ""
    for child in getattr(statement, 'children', []):
        if hasattr(child, 'data') and str(_tree_data_str(child)).lower() == "call_suffix":  # type: ignore
            # Format the call_suffix content (parens + arguments)
            from .expression_to_str import expression_to_str as ets
            try:
                args_str = ets(child)
            except Exception:
                args_str = "()"
            break
    
    # A statement-level call has no `var x = ` in front of it to hang the arguments off, and this
    # handler used to render it as one line whatever its width. That is why a `print_debug` the
    # author had already exploded came back as a single 110-column line, and why no fixture caught
    # it: every over-long call in the corpus is a value (`var x = call(...)`), which reaches the
    # expression layout through `_format_assigned` instead.
    rendered = f"{func_name}{args_str}"
    limit = style.line_length if style else 100
    if len(indent_string.replace("\t", "    ")) + len(rendered) > limit:
        lines, _end = format_expression(
            statement,
            ExpressionContext("", get_line_safe(statement), "", get_line_safe(statement)),
            indent_string, style,
        )
        return (lines, get_end_line_safe(statement))

    return format_simple_statement(rendered, statement, indent_string, indent_level)


# ─── Handler Functions ──────────────────────────────────────────────────────

def _format_expr_statement(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    expr = getattr(statement, 'children', [None])[0] if statement.children else None  # type: ignore
    if expr is None:
        return ([], get_line_safe(statement))

    expression_context = ExpressionContext(
        "", get_line_safe(statement), "", get_line_safe(statement)
    )
    lines, end_line = format_expression(expr, expression_context, indent_string, style)
    return (lines, get_end_line_safe(statement))


def _format_assignment_statement(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format an assignment statement: `target = value`, `target += value`, etc.

    Grammar: assignment_stmt: assignable assignment_operator assignment
    The left side is an inlined Token for a bare name, otherwise an `assignable` tree.
    """
    from .expression_to_str import expression_to_str as _ets

    children = [c for c in getattr(statement, 'children', []) if c is not None]
    if len(children) != 3:
        return format_simple_statement(_ets(statement), statement, indent_string, indent_level)

    lhs, op_tree, rhs = children
    op = "="
    op_children = getattr(op_tree, 'children', None)
    if op_children:
        op_tokens = [c for c in op_children if isinstance(c, Token)]
        if op_tokens:
            op = str(op_tokens[0].value)

    # Render the target sequentially: `d` + `["key"]` -> `d["key"]` (no space before a
    # subscript/attribute), matching the style guide's `dict["key"] = 5`.
    if isinstance(lhs, Tree):
        lhs_str = "".join(_ets(c) for c in getattr(lhs, 'children', []) if c is not None)
    else:
        lhs_str = str(getattr(lhs, 'value', lhs))

    expression_context = ExpressionContext(
        f"{lhs_str} {op} ", get_line_safe(statement), "", get_end_line_safe(statement)
    )
    # The right-hand side is a whole slot, so a group spanning all of it comes off:
    # `x = (1 + 2)` is `x = 1 + 2`. A group that is an operand of the value's own operator
    # chain is not affected - see `_format_if_elif_branch`.
    from .expression_utils import strip_slot_parentheses
    rhs = strip_slot_parentheses(rhs, f"{lhs_str} {op} ", indent_string, style)
    lines, end_line = format_expression(rhs, expression_context, indent_string, style)
    return (lines, get_end_line_safe(statement))


def _format_return_statement(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format return/break/continue statements.
    
    Grammar: flow_stmt: RETURN [expression] | BREAK | CONTINUE | BREAKPOINT
    The token type tells us which keyword to use; for break/continue there's no expression.
    """
    children = getattr(statement, 'children', [])  # type: ignore
    
    # Check for the keyword token
    keyword = None
    has_expr = False
    expr_node = None
    
    for child in children:
        if isinstance(child, Token):
            ttype = str(getattr(child, 'type', '')).upper()
            if ttype == "RETURN":
                keyword = "return"
            elif ttype == "BREAK":
                keyword = "break"
            elif ttype == "CONTINUE":
                keyword = "continue"
            elif ttype == "BREAKPOINT":
                keyword = "breakpoint"
        elif isinstance(child, Tree):
            # Expression after RETURN
            has_expr = True
            expr_node = child
    
    if keyword is None:
        return format_simple_statement("?", statement, indent_string, indent_level)
    
    # Check for expression (only for 'return')
    if keyword == "return" and has_expr and expr_node is not None:
        # Redundant outer parentheses on the whole return value are removed, e.g.
        # `return (a or b)` -> `return a or b`.
        from .expression_utils import strip_wrapped_outer_parentheses as _strip
        expr_node = _strip(expr_node)
        expression_context = ExpressionContext(
            "return ", get_line_safe(statement), "", get_end_line_safe(statement)
        )
        lines, end_line = format_expression(expr_node, expression_context, indent_string, style)
        return (lines, get_end_line_safe(statement))
    
    # Simple keyword without expression
    return format_simple_statement(keyword, statement, indent_string, indent_level)


def _format_if_statement(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format an if/elif/else compound statement.
    
    Grammar: if_stmt: IF expression COLON suite (ELIF expression COLON suite)* (ELSE COLON suite)?
    Children are flat interleaved tokens and trees:
      [IF_token, expr_tree1, COLON_token, suite_tree1, ELIF_token?, expr_tree2?, ...]
    """
    children = getattr(statement, 'children', [])  # type: ignore
    formatted_lines: FormattedLines = []
    i = 0
    
    while i < len(children):
        child = children[i]
        
        # Detect IF keyword token - start of a new branch
        if isinstance(child, Token) and getattr(child, 'type', '').upper() == 'IF':
            # Expression is at index + 1 (first non-token Tree child)
            expr_pos = i + 1
            # Suite is typically at index + 3 (after IF, expr, COLON)
            suite_pos = i + 3
            
            header_lines = _format_if_elif_branch(
                "if", children, expr_pos, indent_string, indent_level, style
            )
            formatted_lines += header_lines
            
            # Advance past IF branch (IF + expr + COLON + suite)
            i = suite_pos + 1 if suite_pos < len(children) else len(children)
        elif isinstance(child, Token) and getattr(child, 'type', '').upper() == 'ELIF':
            # Expression after ELIF
            expr_pos = i + 1
            # Suite after COLON (index + 3 from ELIF token)
            suite_pos = i + 3
            
            header_lines = _format_if_elif_branch(
                "elif", children, expr_pos, indent_string, indent_level, style
            )
            formatted_lines += header_lines
            
            i = suite_pos + 1 if suite_pos < len(children) else len(children)
        elif isinstance(child, Token) and getattr(child, 'type', '').upper() == 'ELSE':
            # ELSE has no expression, body starts after COLON
            header_lines = _format_else_header(
                children, i, indent_string, indent_level, style
            )
            formatted_lines += header_lines
            
            i = len(children)  # ELSE is always last
        else:
            # Skip non-keyword tokens (COLON, etc.) and Trees that aren't branches
            if isinstance(child, Token):
                pass  # Tokens are handled by the keyword-based loop above
            elif hasattr(child, 'data'):
                pass  # These are expr/suite trees - skipped as part of branch processing
            i += 1
    
    return (formatted_lines, get_end_line_safe(statement))


def _format_if_elif_branch(
    keyword: str,
    children: list,
    expr_pos: int,
    indent_string: str,
    indent_level: int,
    style=None,
) -> FormattedLines:
    """Format an if/elif branch header and body."""
    formatted_lines: FormattedLines = []
    
    # Extract expression
    expr_children = []
    for c in children[expr_pos:expr_pos + 3]:  # Check expr + next few positions
        if isinstance(c, Token):
            break
        elif hasattr(c, 'data'):
            data_str = _tree_data_str(c).lower()
            if data_str in ("suite", "colon"):
                break
            if c is not None:
                expr_children.append(c)
    
    if expr_children:
        # Format the expression as the header line
        from .context import ExpressionContext
        # The guide asks to avoid parentheses that are not needed for order of operations or
        # for wrapping over multiple lines, and a group spanning the whole condition is one:
        # `if (x + 1):` is `if x + 1:`. Nothing outside the group can bind to what is inside
        # it. A group that is an *operand*, as in `if (a and b) or not c:`, is left alone.
        from .expression_utils import strip_slot_parentheses
        expr_tree = strip_slot_parentheses(expr_children[0], f"{keyword} ", indent_string, style)
        expression_context = ExpressionContext(
            f"{keyword} ", get_line_safe(children[expr_pos - 1] if expr_pos > 0 else children[-1]),
            ":", get_end_line_safe(expr_tree)
        )
        header_lines, last_line_no = format_expression(expr_tree, expression_context, indent_string, style)
    else:
        line_no = get_end_line_safe(children[expr_pos - 1] if expr_pos > 0 else children[-1])
        header_lines = [(line_no, f"{indent_string}{keyword} :")]
    
    # Extract body from suite (at position + 3 from keyword token)
    suite_pos = expr_pos + 2
    if suite_pos < len(children):
        suite_child = children[suite_pos]
        if hasattr(suite_child, 'data') and _tree_data_str(suite_child).lower() == "suite":
            body_stmts = _extract_suite_body(suite_child)
            if body_stmts:
                body_indent_string = _get_body_indent(style, indent_level)
                body_lines, last_line_no = format_block(
                    body_stmts,
                    format_func_statement,
                    body_indent_string, indent_level + 4, style
                )
                formatted_lines += header_lines + body_lines
            else:
                formatted_lines = header_lines
        else:
            formatted_lines = header_lines
    else:
        formatted_lines = header_lines
    
    return formatted_lines


def _format_else_header(
    children: list,
    keyword_pos: int,
    indent_string: str,
    indent_level: int,
    style=None,
) -> FormattedLines:
    """Format an else branch header and body. ELSE has no expression."""
    formatted_lines: FormattedLines = []
    
    # Find the suite after COLON
    for i in range(keyword_pos + 1, len(children)):
        child = children[i]
        if hasattr(child, 'data') and _tree_data_str(child).lower() == "suite":
            body_stmts = _extract_suite_body(child)
            line_no = get_line_safe(children[keyword_pos])
            header_lines = [(line_no, f"{indent_string}else:")]
            
            if body_stmts:
                body_indent_string = _get_body_indent(style, indent_level)
                body_lines, last_line_no = format_block(
                    body_stmts,
                    format_func_statement,
                    body_indent_string, indent_level + 4, style
                )
                formatted_lines = header_lines + body_lines
            else:
                formatted_lines = header_lines
            break
    
    return formatted_lines


def _extract_suite_body(suite_tree) -> list:
    """Extract statement children from a suite tree.
    
    Handles the grammar where suite contains suite_line nodes with actual statements,
    plus comment trees at suite level. Detects inline comments on local variables
    and appends them to the member tuple.
    """
    # Build inline comment map for local variables first
    from typing import Dict
    inline_comments: Dict[int, str] = {}
    suite_children = list(getattr(suite_tree, 'children', []))
    
    for i, child in enumerate(suite_children):
        if isinstance(child, Tree) and hasattr(child, 'data'):
            data_str = _tree_data_str(child).lower()
            if "comment" in data_str and hasattr(child, 'meta'):
                comment_line_no = getattr(getattr(child, 'meta', None), 'line', 0)
                comment_col = getattr(getattr(child, 'meta', None), 'column', 0)
                # Check if previous sibling is a suite_line containing local_variable
                if i > 0:
                    prev = suite_children[i - 1]
                    if isinstance(prev, Tree) and _tree_data_str(prev).lower() == "suite_line":
                        decl_end_col = None
                        decl_end_line = None
                        for sub in getattr(prev, 'children', []):
                            if hasattr(sub, 'data') and _tree_data_str(sub).lower() == "end":
                                decl_end_col = getattr(getattr(sub, 'meta', None), 'column', 0)
                                decl_end_line = getattr(getattr(sub, 'meta', None), 'line', 0)
                        # The comment must sit on the declaration's LAST line at or past
                        # its end column. Comparing against the statement's first line
                        # missed a trailing comment on a multiline declaration (the
                        # closing bracket's line), which then became a stray line of its
                        # own instead of staying on that bracket.
                        if (decl_end_col and decl_end_line and comment_col >= decl_end_col
                                and comment_line_no == decl_end_line):
                            comment_text = _get_comment_text(child)
                            inline_comments[comment_line_no] = comment_text
    
    stmts = []
    for child in getattr(suite_tree, 'children', []):
        if child is None:
            continue
        if isinstance(child, Tree) and hasattr(child, 'data'):
            data_str = _tree_data_str(child).lower()
            
            # Comment tree at suite level - emit as __comment__
            if "comment" in data_str:
                comment_line_no = getattr(getattr(child, 'meta', None), 'line', 0)
                # If this is an inline comment for a local variable, skip it here
                if comment_line_no in inline_comments:
                    continue
                stmts.append(("__comment__", child))
                continue
            
            # suite_line: extract inner statements, skip end markers
            if data_str == "suite_line":
                # A trailing comment sits on the statement's LAST line, which for a
                # multiline statement is the line its `end` marker is on.
                stmt_end_line = 0
                for sub_child in getattr(child, 'children', []):
                    if hasattr(sub_child, 'data') and _tree_data_str(sub_child).lower() == "end":
                        stmt_end_line = getattr(getattr(sub_child, 'meta', None), 'line', 0)
                for sub_child in getattr(child, 'children', []):
                    if sub_child is None or not hasattr(sub_child, 'data'):
                        continue
                    idata = _tree_data_str(sub_child).lower()
                    # Skip end markers and comment trees inside suite_line
                    if idata == "end" or (isinstance(sub_child, Tree) and "comment" in idata):
                        continue
                    # Check for inline comment on this statement
                    lookup_line = stmt_end_line or getattr(getattr(sub_child, 'meta', None), 'line', 0)
                    ic_text = inline_comments.get(lookup_line)
                    if ic_text:
                        stmts.append(("member", sub_child, ic_text))
                    else:
                        stmts.append(("member", sub_child))
            # Nested suite - recurse
            elif data_str == "suite":
                inner_stmts = _extract_suite_body(child)
                if inner_stmts:
                    stmts.extend(inner_stmts)
            else:
                # Direct compound statement or other tree node (while, for, match, etc.)
                # Treat as a member statement
                if data_str not in ("colon",):
                    stmts.append(("member", child))
    
    return stmts


def _format_for_statement(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format for/for_typed statements.
    
    Grammar: FOR NAME [COLON type] IN expression COLON suite
    Children: [FOR_token, NAME_token?, COLON_token?, type_tree, IN_token, expr_tree, COLON_token, suite_tree]
    """
    children = getattr(statement, 'children', [])  # type: ignore
    var_name = ""
    in_pos = -1
    has_type = False
    
    for i, child in enumerate(children):
        if isinstance(child, Token) and child.type == "NAME" and not var_name:
            var_name = str(child.value)
        elif isinstance(child, Token) and child.type == "IN":
            in_pos = i
            break  # Found IN; check for type between NAME and IN
    
    # Check for type annotation (COLON + tree between NAME position and IN position)
    name_found = False
    if var_name:
        name_pos = -1
        for ci, cc in enumerate(children):
            if isinstance(cc, Token) and cc.type == "NAME":
                name_pos = ci
                break
        
        # Look for a non-None Tree between NAME and IN positions that isn't the main expression
        has_type = False
        type_tree = None
        if in_pos > 0 and name_pos >= 0:
            # Check children from (name_pos+1) to in_pos for a TYPE or other tree
            for ci in range(name_pos + 1, min(in_pos, len(children))):
                c = children[ci]
                if isinstance(c, Tree) and hasattr(c, 'data'):
                    # This could be the type annotation or an intermediate expression
                    data_str = _tree_data_str(c).lower()
                    if data_str in ("type", "assignment"):
                        # If it's at position + 1 after NAME (before IN), it's likely the type
                        if ci == name_pos + 2:  # Skip COLON token at name_pos+1
                            has_type = True
                            type_tree = c
                            break
    
    # Build prefix with or without type annotation
    if has_type and var_name:
        from .expression_to_str import expression_to_str as ets
        try:
            type_name = ets(type_tree)  # type: ignore
            prefix = f"for {var_name}: {type_name} in "
        except Exception:
            prefix = f"for {var_name}: ? in "
    else:
        prefix = f"for {var_name} in "
    
    # Expression is after IN token
    expr_position = in_pos + 1 if in_pos >= 0 else -1
    return _format_branch(prefix, ":", expr_position, statement, indent_string, indent_level, style)


def _format_for_statement_typed(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    var_name = ""
    type_name = "?"
    for child in getattr(statement, 'children', []):  # type: ignore
        if isinstance(child, Token):
            if not var_name and child.type == "NAME":  # type: ignore
                var_name = str(child.value)  # type: ignore
            elif not type_name:
                type_name = str(child.value)  # type: ignore
    
    prefix = f"for {var_name}: {type_name} in "
    return _format_branch(prefix, ":", 2, statement, indent_string, indent_level, style)


def _format_match_statement(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format a match compound statement.
    
    Grammar: MATCH expression COLON body
    Children: [MATCH_token, expr_tree, COLON_token, body_tree]
    Body tree could be `match_branch`, `_match_body`, or `suite`
    """
    children = getattr(statement, 'children', [])  # type: ignore
    formatted_lines: FormattedLines = []
    
    # Extract expression at position 1
    expr_pos = 1
    header_lines: List[tuple] = []
    if expr_pos < len(children):
        expr_tree = children[expr_pos]
        if isinstance(expr_tree, Tree) and hasattr(expr_tree, 'data'):
            from .context import ExpressionContext
            # The subject is a whole slot (`match (x):` is `match x:`). A branch *pattern* is
            # not, and keeps its parentheses: `(1):` is a pattern group, and `_layout_match_line`
            # is deliberately not given this treatment.
            from .expression_utils import strip_slot_parentheses
            expr_tree = strip_slot_parentheses(expr_tree, "match ", indent_string, style)
            line_no = get_line_safe(statement)
            end_line = get_end_line_safe(expr_tree)
            expression_context = ExpressionContext(
                "match ", line_no, ":", end_line
            )
            header_lines, last_line_no = format_expression(
                expr_tree, expression_context, indent_string, style
            )
    else:
        formatted_lines = [(get_line_safe(statement), f"{indent_string}match :")]
        return (formatted_lines, get_end_line_safe(statement))
    
    # Extract and format body branches
    branch_indent = _get_body_indent(style, indent_level + 1)
    formatted_lines = header_lines.copy()
    
    if len(children) > 3:
        body_child = children[3]
        data_str = _tree_data_str(body_child).lower() if hasattr(body_child, 'data') else ""
        
        # Handle direct match_branch containers (all branches at positions 3+)
        if isinstance(body_child, Tree) and data_str == "match_branch":
            result = _format_match_branch(body_child, branch_indent, indent_level + 4, style)
            formatted_lines.extend(result[0])
            # Check for subsequent direct match_branch children
            for i in range(4, len(children)):
                child = children[i]
                if isinstance(child, Tree) and hasattr(child, 'data'):
                    cs = _tree_data_str(child).lower()
                    if cs == "match_branch":
                        result = _format_match_branch(child, branch_indent, indent_level + 4, style)
                        formatted_lines.extend(result[0])
        # Body is a suite (rare for match, more common for if/for/while)
        elif data_str == "suite":
            body_stmts = _extract_suite_body(body_child)
            if body_stmts:
                body_indent_string = _get_body_indent(style, indent_level + 1)
                body_lines, last_line_no = format_block(
                    body_stmts,
                    format_func_statement,
                    body_indent_string, indent_level + 4, style
                )
                formatted_lines.extend(header_lines + body_lines)
        # Body is _match_body (with _INDENT/_DEDENT postlexer tokens)
        elif data_str == "_match_body":
            body_stmts = _extract_match_body(body_child)
            if body_stmts:
                body_indent_string = _get_body_indent(style, indent_level + 1)
                body_lines, last_line_no = format_block(
                    body_stmts,
                    format_func_statement,
                    body_indent_string, indent_level + 4, style
                )
                formatted_lines.extend(header_lines + body_lines)
    
    return (formatted_lines, get_end_line_safe(statement))


# ─── Branch helper ──────────────────────────────────────────────────────────

def _format_branch(
    prefix: str, suffix: str, expr_position: int | None, 
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    formatted_lines: FormattedLines = []

    if expr_position is not None and len(statement.children) > expr_position:  # type: ignore
        expr = getattr(statement, 'children', [None])[expr_position]  # type: ignore
        if expr is not None:
            # A group spanning the whole condition or iterable is redundant: `while (x + 1):`
            # is `while x + 1:`, and `for i in (a + b):` is `for i in a + b:`. See the note in
            # `_format_if_elif_branch`.
            from .expression_utils import strip_slot_parentheses
            expr = strip_slot_parentheses(expr, prefix, indent_string, style)
            expression_context = ExpressionContext(
                prefix, get_line_safe(statement), suffix, get_end_line_safe(statement)
            )
            header_lines, last_line_no = format_expression(expr, expression_context, indent_string, style)
        else:
            header_lines = [(get_line_safe(statement), f"{indent_string}{prefix}{suffix}")]
            last_line_no = get_line_safe(statement)
    else:
        header_lines = [(get_line_safe(statement), f"{indent_string}{prefix}{suffix}")]
        last_line_no = get_line_safe(statement)

    # Format body statements after the expression.
    # We need to skip COLON tokens and unwrap suite/suite_line containers.
    offset = (expr_position + 1) if expr_position is not None else 0
    
    def _extract_body_children(children, start_offset):
        """Extract actual statement children from compound statement body.
        
        Handles the new grammar where suite/suite_line trees wrap actual statements:
          for_stmt -> [..., COLON, SUITE] 
          suite    -> [_NL, _INDENT, suite_line*, _DEDENT]
          suite_line -> [local_variable | pass_stmt | function_call_expr | compound_statement | ... | end?]
        """
        stmts = []
        for child in children[start_offset:]:
            if not hasattr(child, 'data'):
                continue
            data_str = _tree_data_str(child).lower()
            
            # Skip comments - handled by block formatting
            if "comment" in data_str:
                stmts.append(("__comment__", child))
                continue
            
            # Unwrap suite/suite_line containers to get actual statements inside
            if data_str == "suite":
                for sub_child in getattr(child, 'children', []):
                    if sub_child is None:
                        continue
                    stype = str(sub_child.type).lower() if isinstance(sub_child, Token) else _tree_data_str(sub_child).lower()
                    # Skip postlexer tokens and comment trees
                    if (isinstance(sub_child, Token) and 
                        (stype.startswith("_nl") or stype.startswith("_indent") or stype.startswith("_dedent"))):
                        continue
                    if isinstance(sub_child, Tree) and "comment" in stype:
                        stmts.append(("__comment__", sub_child))
                        continue
                    # Recursively extract from nested containers (suite_line)
                    if stype == "suite_line":
                        for inner in getattr(sub_child, 'children', []):
                            if inner is None or not hasattr(inner, 'data'):
                                continue
                            idata = _tree_data_str(inner).lower()
                            # Skip end markers and comment trees inside suite_line
                            if idata == "end" or (isinstance(inner, Tree) and "comment" in idata):
                                continue
                            stmts.append(("member", inner))
                    else:
                        # Direct compound statement node (if/while/match/etc.) inside suite
                        # Not wrapped in suite_line - add directly as a member
                        if data_str not in ("colon", "end") and hasattr(sub_child, 'data'):
                            stmts.append(("member", sub_child))
            elif data_str == "suite_line":
                for sub_child in getattr(child, 'children', []):
                    if sub_child is None:
                        continue
                    stype = str(sub_child.type).lower() if isinstance(sub_child, Token) else _tree_data_str(sub_child).lower()
                    # Skip postlexer tokens, comment trees, and end markers
                    if (isinstance(sub_child, Token) and 
                        (stype.startswith("_nl") or stype.startswith("_indent") or stype.startswith("_dedent") or stype == "colon")):
                        continue
                    if isinstance(sub_child, Tree) and ("comment" in stype or stype == "end"):
                        continue
                    stmts.append(("member", sub_child))
            else:
                # Direct compound statement or regular statement node inside suite
                # (if/while/match/local_variable/pass_stmt/etc.)
                if data_str not in ("colon", "end"):
                    stmts.append(("member", child))
        return stmts
    
    body_stmts = _extract_body_children(getattr(statement, 'children', []), offset)
    
    if body_stmts:
        body_indent_string = _get_body_indent(style, indent_level)
        body_lines, last_line_no = format_block(
            body_stmts,
            format_func_statement,
            body_indent_string, indent_level + 4, style
        )
        formatted_lines += header_lines + body_lines
    else:
        formatted_lines = header_lines

    return (formatted_lines, last_line_no)


# ─── Match branch helpers ───────────────────────────────────────────────────

def _extract_match_body(body_tree) -> list:
    """Extract statements from a match body (_match_body with _INDENT/_DEDENT).
    
    Grammar: _match_body -> _NL _INDENT (_NL | match_item)+ _DEDENT
    Contains: postlexer tokens + comment trees + match_branch/suite_line nodes
    """
    stmts = []
    for child in getattr(body_tree, 'children', []):
        if child is None:
            continue
        
        # Skip postlexer tokens (_NL, _INDENT, _DEDENT)
        if isinstance(child, Token):
            ttype = str(getattr(child, 'type', '')).lower()
            if (ttype.startswith("_nl") or ttype.startswith("_indent") or 
                ttype.startswith("_dedent")):
                continue
        # Skip standalone comment trees at _match_body level
        elif isinstance(child, Tree) and hasattr(child, 'data'):
            data_str = _tree_data_str(child).lower()
            if "comment" in data_str:
                stmts.append(("__comment__", child))
            elif data_str == "_func_suite":
                # Nested suite inside match item
                for sc in getattr(child, 'children', []):
                    if isinstance(sc, Tree) and hasattr(sc, 'data'):
                        idata = _tree_data_str(sc).lower()
                        if idata == "suite_line" or (idata != "end" and idata != "comment"):
                            stmts.append(("member", sc))
                    elif isinstance(sc, Token):
                        pass  # Skip tokens in nested suite
        else:
            continue
    
    return stmts


def _extract_match_branch_body(branch_tree) -> list:
    """Extract body from a match_branch node.
    
    Grammar: match_branch: match_pattern (COMMA match_pattern)* [WHEN expression] COLON suite
    Children include patterns, optional WHEN expr, COLON, and suite
    """
    # Find the COLON token position and get everything after it as body
    for i, child in enumerate(getattr(branch_tree, 'children', [])):
        if isinstance(child, Token) and getattr(child, 'type', '').upper() == 'COLON':
            suite_pos = i + 1
            break
    else:
        return []
    
    # Check if there's a suite after the colon
    if suite_pos < len(getattr(branch_tree, 'children', [])):
        body_child = getattr(branch_tree, 'children', [])[suite_pos]
        if hasattr(body_child, 'data') and _tree_data_str(body_child).lower() == "suite":
            return _extract_suite_body(body_child)
    
    return []


def format_match_branch_statement(
    statement: Any, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format a match branch (pattern: body)."""
    if isinstance(statement, tuple):
        tag, actual_stmt = statement  # type: ignore
    else:
        actual_stmt = statement
    
    stmt_data = _tree_data_str(actual_stmt).lower()
    
    if not hasattr(actual_stmt, 'data'):
        return format_simple_statement(str(stmt_data), actual_stmt, indent_string, indent_level)
    
    if stmt_data == "match_branch":
        return _format_match_branch(actual_stmt, indent_string, indent_level, style)
    elif stmt_data == "comment":
        # Standalone comment within match body
        text = _get_comment_text(actual_stmt)
        formatted_lines: FormattedLines = [(None, f"{indent_string}{text}")]
        return (formatted_lines, get_end_line_safe(actual_stmt))
    else:
        # Could be a nested statement inside the match suite
        from .function_statement import format_func_statement as _fmt_func
        return _fmt_func(statement, indent_string, indent_level, style)


def _layout_match_line(node, prefix, suffix, line_no, indent_string, style):
    """Lay `node` out as an expression for a match branch header, or None on failure.

    Returns the formatted lines, or None so the caller falls back to the inline form when
    the layout path cannot handle the node.
    """
    try:
        lines, _ = format_expression(
            node, ExpressionContext(prefix, line_no, suffix, line_no), indent_string, style
        )
        return lines
    except Exception:
        return None


def _pattern_list_with_backslashes(items, indent_string, suffix, line_no, max_len):
    """Pack a bare pattern list across lines, breaking each with a backslash.

    A pattern list has no brackets of its own to continue it, and wrapping it in parentheses
    is not available: Godot rejects `(1, 1, 1):` with "Expected closing ")" after grouping
    expression". The guide prefers parentheses for a wrap but only *prefers* them, and a
    backslash continuation is the one break that is legal here, so that is what a list too
    long for one line gets.

    The backslash has to follow the comma (`1, \\` then a newline): the continuation joins the
    next line's first pattern onto the list, where a backslash directly after a pattern would
    splice two patterns into one invalid one.

    Returns None when the list already fits in one group, so the caller keeps the inline form.
    """
    if len(items) < 2:
        return None
    continuation = indent_string + "\t\t"
    # Every group but the last ends in `, \`, and the last carries the colon (and any guard),
    # so measure against the wider continuation indent with room for both.
    budget = max_len - len(suffix.replace('\t', '    ')) - 3
    groups = []
    current = ""
    for item in items:
        candidate = f"{current}, {item}" if current else item
        if current and len(f"{continuation}{candidate}".replace('\t', '    ')) > budget:
            groups.append(current)
            current = item
        else:
            current = candidate
    groups.append(current)
    if len(groups) < 2:
        return None
    lines = []
    for index, group in enumerate(groups):
        indent = indent_string if index == 0 else continuation
        tail = suffix if index == len(groups) - 1 else ", \\"
        lines.append((line_no if index == 0 else None, f"{indent}{group}{tail}"))
    return lines


def _join_match_halves(pattern_lines, guard_lines, indent_string, max_len):
    """Join a laid-out pattern and its laid-out `when` guard into one branch header.

    `pattern_lines` is the pattern on its own and `guard_lines` the guard on its own, each
    starting at the branch indent. The pattern's closing bracket and the guard's opening
    bracket share a line (`... ] when [`), which is where the golden puts the guard's `[`
    too, and the guard's items keep the indentation their own layout gave them.

    Returns None when the joined line would still be over the limit, so the caller can fall
    back to the shape that keeps the pattern inline.
    """
    joined = f"{pattern_lines[-1][1]} when {guard_lines[0][1].lstrip()}"
    if len(joined.replace('\t', '    ')) > max_len:
        return None
    return list(pattern_lines[:-1]) + [(pattern_lines[-1][0], joined)] + list(guard_lines[1:])


def _format_match_branch(
    statement: Tree, indent_string: str, indent_level: int, style=None
) -> Outcome:
    """Format a single match branch.
    
    Grammar: match_branch: match_pattern (COMMA match_pattern)* [WHEN expression] COLON suite
    Children are interleaved patterns, optional WHEN expr, COLON token, and suite tree.
    """
    children = getattr(statement, 'children', [])  # type: ignore
    formatted_lines: FormattedLines = []
    
    # Extract patterns - format as expression string
    pattern_parts = []
    pattern_nodes = []
    when_expr = None  # Store WHEN guard for later
    i = 0
    while i < len(children):
        child = children[i]
        if isinstance(child, Token):
            ttype = str(getattr(child, 'type', '')).upper()
            if ttype == "COMMA":
                pattern_parts.append(", ")
            elif ttype == "WHEN":
                # Look ahead for guard expression
                if i + 1 < len(children) and isinstance(children[i + 1], Tree):
                    when_expr = children[i + 1]
                break
            elif ttype == "COLON":
                break  # COLON means we're past all patterns
        elif hasattr(child, 'data'):
            data_str = _tree_data_str(child).lower()
            if data_str == "when_expr":
                when_expr = child  # Save WHEN guard expression
                break
            elif isinstance(child, Tree) and child is not None:
                from .expression_to_str import match_pattern_to_str
                pattern_nodes.append(child)
                try:
                    pattern_parts.append(match_pattern_to_str(child))
                except Exception:
                    pattern_parts.append("?")
        i += 1

    # Format the header line (pattern1, pattern2: body)
    pattern_str = "".join(pattern_parts).strip()
    if not pattern_str.strip():
        formatted_lines = [(get_line_safe(statement), f"{indent_string}?:")]
        return (formatted_lines, get_end_line_safe(statement))

    # Find COLON and suite
    colon_pos = -1
    for i, child in enumerate(children):
        if isinstance(child, Token) and getattr(child, 'type', '').upper() == "COLON":
            colon_pos = i
            break

    line_no = get_line_safe(statement)
    max_len = style.line_length if style else 100
    guard_inline = ""
    if when_expr is not None:
        from .expression_to_str import expression_to_str as ets
        try:
            guard_inline = ets(when_expr)
        except Exception:
            guard_inline = "?"
    guard_suffix = f" when {guard_inline}" if when_expr is not None else ""
    inline_line = f"{indent_string}{pattern_str}{guard_suffix}:"

    # The pattern goes through the expression layout only when it does not fit, so a long
    # operator chain, dot chain, array, call or dict inside it breaks the way it would
    # anywhere else. A pattern that fits keeps the inline form (which also preserves the
    # parentheses the author wrote). Several comma-separated patterns stay on one line,
    # and when there is a `when` guard it is the guard that gets the layout, with the
    # patterns as its prefix (the golden's shape).
    from .expression import _is_forced_multiline
    needs_break = len(inline_line.replace('\t', '    ')) > max_len
    header_lines = None
    if when_expr is not None:
        if needs_break and len(pattern_nodes) == 1:
            # Both halves can be long. Lay each out on its own and join them, so a pattern
            # that needs breaking breaks rather than staying inline while only the guard is
            # rewrapped - which is what leaves a 145-column line in `multiline_match_statements`.
            _pattern_lines = _layout_match_line(pattern_nodes[0], "", "", line_no, indent_string, style)
            _guard_lines = _layout_match_line(when_expr, "", ":", line_no, indent_string, style)
            if _pattern_lines and _guard_lines:
                header_lines = _join_match_halves(
                    _pattern_lines, _guard_lines, indent_string, max_len
                )
        if header_lines is None and (needs_break or _is_forced_multiline(when_expr, style)):
            header_lines = _layout_match_line(
                when_expr, f"{pattern_str} when ", ":", line_no, indent_string, style
            )
    elif len(pattern_nodes) == 1:
        if needs_break or _is_forced_multiline(pattern_nodes[0], style):
            header_lines = _layout_match_line(
                pattern_nodes[0], "", ":", line_no, indent_string, style
            )
    if header_lines is None and needs_break and len(pattern_nodes) > 1:
        # A bare pattern list: nothing to break except with a backslash.
        from .expression_to_str import match_pattern_to_str
        header_lines = _pattern_list_with_backslashes(
            [match_pattern_to_str(node) for node in pattern_nodes],
            indent_string, f"{guard_suffix}:", line_no, max_len,
        )
    if header_lines is None:
        header_lines = [(line_no, f"{indent_string}{pattern_str}{guard_suffix}:")]
    formatted_lines.extend(header_lines)
    
    # Extract body from suite (after COLON token)
    if colon_pos >= 0 and colon_pos + 1 < len(children):
        body_child = children[colon_pos + 1]
        if hasattr(body_child, 'data') and _tree_data_str(body_child).lower() == "suite":
            body_stmts = _extract_suite_body(body_child)
            if body_stmts:
                body_indent_string = _get_body_indent(style, indent_level + 2)
                body_lines, last_line_no = format_block(
                    body_stmts,
                    format_func_statement,
                    body_indent_string, indent_level + 4, style
                )
                formatted_lines.extend(body_lines)
    
    return (formatted_lines, get_end_line_safe(statement))


# ─── Populate handlers after all functions are defined ──────────────────────

_HANDLERS.update({
    "local_variable": lambda s, i, l, st: format_var_statement(s, i, l, st),  # type: ignore
    "constant_decl": lambda s, i, l, st: format_const_statement(s, i, l, st),  # type: ignore
    "flow_stmt": _format_return_statement,  # Simplified - return is most common
    "assert_stmt": _format_assert_statement,
    "pass_stmt": partial(format_simple_statement, "pass"),
    "expression_stmt": _format_expr_statement,
    "function_call_expr": _format_function_call_expr,
    "assignment_stmt": _format_assignment_statement,
    "if_stmt": _format_if_statement,
    "while_stmt": lambda s, i, l, st: _format_branch("while ", ":", 1, s, i, l, st),
    "for_stmt": _format_for_statement,
    "match_stmt": _format_match_statement,
    "annotation": lambda s, i, l, st: _format_annotation_statement(s, i, l, st),
})


def _format_annotation_statement(statement: Tree, indent_string: str, indent_level: int, style=None):
    """Format an annotation inside a function body (e.g. `@warning_ignore(...)`)."""
    from .annotation import format_annotation
    return format_annotation(statement, indent_string, indent_level, style)


def get_line_safe(node) -> int:
    try:
        return get_line(node)  # type: ignore
    except Exception:
        return 1


def get_end_line_safe(node) -> int:
    from ..common.utils import get_end_line as _gel
    try:
        return _gel(node)
    except Exception:
        return get_line_safe(node)


# Lazy imports to avoid circular dependencies
def format_const_statement(statement, indent_string, indent_level, style=None):
    from .const_statement import format_const_statement as _fcs
    return _fcs(statement, indent_string, indent_level, style)


def format_var_statement(statement, indent_string, indent_level, style=None, prefix=""):
    from .var_statement import format_var_statement as _fvs
    return _fvs(statement, indent_string, indent_level, style, prefix)


def _get_comment_text(comment_node) -> str:
    """Extract the text content from a comment tree node."""
    from .comments import normalize_comment
    for child in getattr(comment_node, 'children', []):
        if hasattr(child, 'value'):
            return normalize_comment(str(child.value))
    return ""


def _get_body_indent(style, current_level):
    """Get proper body indentation string based on style config.
    
    Respects use_spaces setting: spaces for indent or tabs (default).
    The tab_size controls how many space characters per indent level,
    and the number of indent levels is derived from current_level.
    """
    if not style:
        # Default to 4-space indent
        return " " * ((current_level // 4) + 1)
    
    if style.use_spaces:
        tab_size = style.tab_size or 4
        num_levels = (current_level // max(tab_size, 1)) + 1
        return " " * tab_size * num_levels
    else:
        # Tab-based indent (default GDScript convention)
        num_tabs = (current_level // 4) + 1
        return "\t" * num_tabs


def _tree_data_str(node) -> str:
    from lark import Tree
    if isinstance(node, Tree):
        d = node.data
        return str(d) if not hasattr(d, 'value') else d.value  # type: ignore
    return ""
