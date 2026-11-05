from typing import List

from lark import Tree, Token

from .expression_utils import remove_outer_parentheses  # noqa: F401

# The style guide prefers the plain English boolean operators.
_BOOLEAN_OPERATOR_ALIASES = {'&&': 'and', '||': 'or', '!': 'not'}


def _operator_text(value) -> str:
    """Map C-style boolean operators to their plain English equivalents, per the
    style guide (`&&` -> `and`, `||` -> `or`, `!` -> `not`)."""
    return _BOOLEAN_OPERATOR_ALIASES.get(str(value), str(value))


def _regroup(digits: str, size: int, minimum_separators: int, even_groups: bool) -> str:
    """Group `digits` from the right in chunks of `size`, joined by underscores.

    Grouped only when the grouping is worth having: it needs at least `minimum_separators`
    underscores, and with `even_groups` the digit count also has to divide evenly by `size`.
    A run that fails either test keeps no separators at all, so whatever the author wrote
    goes - the guide's Bad example is exactly that (`12_345`).
    """
    bare = digits.replace('_', '')
    if not bare:
        return ''
    head = len(bare) % size
    if head and even_groups:
        return bare
    groups = ([bare[:head]] if head else []) + [bare[i:i + size] for i in range(head, len(bare), size)]
    return '_'.join(groups) if len(groups) - 1 >= minimum_separators else bare


def _split_exponent(text: str) -> tuple:
    """Split a decimal literal into its mantissa and its exponent part (`''` when none)."""
    for i, ch in enumerate(text):
        if ch in 'eE':
            return text[:i], text[i:]
    return text, ''


# The grouping the guide's examples show, as (group size, separators needed to bother).
#
# Decimal digits go in threes and are grouped from a million up: that is the guide's own note
# ("numbers lower than 1000000 generally don't need separators"), and every grouping of a
# million or more needs two separators, so the two readings agree. A short leading group is
# what thousands-grouping looks like, and the guide's own `1_234_567_890` has one.
#
# Hexadecimal and binary digits go in fours and are grouped only when the digits divide evenly
# into fours. The guide shows `0xffff_f8f8_0000` grouped and its own `0xfb8c0b` ungrouped, and
# only the even-division test satisfies both: grouping six digits from the right would split
# them 2-4 (`0xff_9900`), and five digits 1-4 (`0xa_fa01`).
_DECIMAL_GROUP = (3, 2)
_BITWISE_GROUP = (4, 1)


def normalize_number(text: str) -> str:
    """Normalise a numeric literal per the style guide.

    Hexadecimal digits go lower case, and a floating-point literal keeps its leading and
    trailing zero (`.5` -> `0.5`, `13.` -> `13.0`).

    Underscores survive only where they make a literal readable, per `_DECIMAL_GROUP` and
    `_BITWISE_GROUP`. A fraction and an exponent are not magnitudes, so neither carries a
    grouping, and a literal too short to group loses every separator it had.
    """
    if text[:2].lower() == '0x':
        return '0x' + _regroup(text[2:].lower(), *_BITWISE_GROUP, even_groups=True)
    if text[:2].lower() == '0b':
        return '0b' + _regroup(text[2:], *_BITWISE_GROUP, even_groups=True)
    mantissa, exponent = _split_exponent(text)
    whole, dot, fraction = mantissa.partition('.')
    if not whole:
        whole = '0'
    if dot and not fraction:
        fraction = '0'
    number = _regroup(whole, *_DECIMAL_GROUP, even_groups=False)
    if dot:
        number += '.' + fraction.replace('_', '')
    return number + exponent.replace('_', '')


def standalone_expression_to_str(expression):
    expression = remove_outer_parentheses(expression)
    return expression_to_str(expression)


def _expression_with_parens(expression):
    """Format an expression WITHOUT stripping parentheses.
    
    Used for operands in operator chains where parens affect readability
    (e.g., '(1+1)+1' should keep outer parens, not strip to '1+1+1').
    """
    return expression_to_str(expression)


def expression_to_str(expression):
    # Handle Token input
    if isinstance(expression, Token):
        token_handlers = {
            "NUMBER": lambda t: normalize_number(str(t.value)),
            "LONG_STRING": _long_string_to_str,
            "LONG_RSTRING": _long_rstring_to_str,
            "REGULAR_STRING": _regular_string_to_str,
            "REGULAR_RSTRING": _regular_rstring_to_str,
            # Plain STRING tokens from the grammar (handles all string types)
            "STRING": _string_token_to_str,
        }
        if expression.type in token_handlers:
            return token_handlers[expression.type](expression)
        # For RULE tokens or other types, just return the value
        val = str(getattr(expression, 'value', str(expression.type)))
        return val if not isinstance(expression.value, Token) else "?"

    # Handle None input (shouldn't happen but be safe)
    if expression is None:
        return "?"

    data_map = {
        # New grammar passthrough types (wrap expression without changing representation)
        "assignment": _passthrough,
        # cast with AS (e.g., `1.0 as int`) — don't use _passthrough
        # because it stops at the Token(AS) and loses the type part
        # `cast` carries the `as` repetitions (`?cast: ternary ([AS type])*`); the
        # old lambda emitted only the first pair and dropped the rest.
        "cast": _type_test_or_cast_to_str,
        # ternary needs _format_intermediate_expr for proper formatting
        "ternary": _format_intermediate_expr,
        # New grammar: array and dict literals
        "array_literal": _array_to_str,
        "dict_literal": _dict_to_str,
        # New grammar: attribute chains (.foo.bar)
        # Grammar: attribute -> PERIOD (NAME | STRING)+ (PERIOD (NAME | STRING))*
        "attribute": lambda e: _attribute_to_str(e),
        # New grammar: comma-separated lists inside expressions
        "elements": lambda e: standalone_expression_to_str(e.children[0]) if len([c for c in getattr(e, 'children', []) if c is not None and not isinstance(c, Token)]) == 1 else _comma_list_to_str(e),
        # New grammar: dict entries list
        "dict_entries": lambda e: _arguments_to_str([c for c in getattr(e, 'children', []) if c is not None]),
        # New grammar: single dict entry (NAME EQUAL expr or STRING COLON expr)
        "dict_entry": _dict_entry_from_children,
        # New grammar: type annotation (NAME [. NAME]* [BRACKET_OPEN types BRACKET_CLOSE])
        "type": _annotation_to_type_str,
        # Postfix expressions (function calls, subscriptions)
        "postfix": _postfix_to_str,
        "call_suffix": _call_suffix_to_str,
        "arguments": lambda e: _arguments_from_tree(e),
        # Primary expressions (function calls, subscript, etc. with paren tokens)
        "primary": _primary_to_str,
        # Old grammar types
        "expr": lambda e: standalone_expression_to_str(e.children[0]),
        "assnmnt_expr": _operator_chain_based_expression_to_str,
        "test_expr": _operator_chain_based_expression_to_str,
        "asless_test_expr": _operator_chain_based_expression_to_str,
        "or_test": _operator_chain_based_expression_to_str,
        "asless_or_test": _operator_chain_based_expression_to_str,
        "and_test": _operator_chain_based_expression_to_str,
        "asless_and_test": _operator_chain_based_expression_to_str,
        "asless_actual_not_test": lambda e: "{}{}{}".format(
            expression_to_str(e.children[0]),
            "" if e.children[0].value == "!" else " ",
            expression_to_str(e.children[1]),
        ),
        "not_in_op": lambda _: "not in",
        "content_test": _operator_chain_based_expression_to_str,
        "asless_content_test": _operator_chain_based_expression_to_str,
        "comparison": _operator_chain_based_expression_to_str,
        "asless_comparison": _operator_chain_based_expression_to_str,
        "bitw_or": _operator_chain_based_expression_to_str,
        "asless_bitw_or": _operator_chain_based_expression_to_str,
        "bitw_xor": _operator_chain_based_expression_to_str,
        "asless_bitw_xor": _operator_chain_based_expression_to_str,
        "bitw_and": _operator_chain_based_expression_to_str,
        "asless_bitw_and": _operator_chain_based_expression_to_str,
        "shift_expr": _operator_chain_based_expression_to_str,
        "asless_shift_expr": _operator_chain_based_expression_to_str,
        "arith_expr": _operator_chain_based_expression_to_str,
        "asless_arith_expr": _operator_chain_based_expression_to_str,
        "mdr_expr": _operator_chain_based_expression_to_str,
        "asless_mdr_expr": _operator_chain_based_expression_to_str,
        "sign": lambda e: _format_sign(e),
        "asless_actual_neg_expr": lambda e: f"-{expression_to_str(e.children[1])}",
        "asless_actual_bitw_not": lambda e: f"~{expression_to_str(e.children[1])}",
        "pow_expr": _operator_chain_based_expression_to_str,
        "asless_pow_expr": _operator_chain_based_expression_to_str,
        "type_test": _type_test_to_str,
        "asless_type_test": _type_test_to_str,
        "actual_type_cast": _type_test_to_str,
        "await_expr": lambda e: "{} {}".format(
            " ".join(str(t.value) for t in e.children[:-1]),
            expression_to_str(e.children[-1]),
        ),
        "standalone_call": _standalone_call_to_str,
        "getattr_call": _getattr_call_to_str,
        "getattr": lambda e: "".join(map(expression_to_str, e.children)),
        "subscr_expr": _subscription_to_str,
        "subscript": _subscription_to_str,  # New grammar subscript rule
        "primary": lambda e: _format_primary_with_paren_removal(e),
        "par_expr": _format_paren_expr,
        "array": _array_to_str,
        "dict": _dict_to_str,
        "c_dict_element": _dict_element_to_str,
        "eq_dict_element": _dict_element_to_str,
        "string": lambda e: expression_to_str(e.children[0]),
        "rstring": _rstring_to_str,
        "get_node_expr": lambda e: _format_get_node_expr(e),
        "get_node": lambda e: f"${expression_to_str(e.children[0])}",
        "path": lambda e: "".join([name_token.value for name_token in e.children]),
        "node_path": lambda e: f"^{expression_to_str(e.children[0])}",
        "unique_node_path": lambda e: "".join(
            [expression_to_str(n) for n in e.children]
        ),
        "string_name": lambda e: f"&{expression_to_str(e.children[0])}",
        "lambda": _lambda_to_str,
        "lambda_header": _lambda_header_to_str,
        # fake expressions (formatter-generated):
        "func_args": _args_to_str,
        "func_arg_variadic": lambda e: f"...{expression_to_str(e.children[0])}",
        "func_arg_regular": lambda e: "{}{}".format(
            str(e.children[0].value),
            " = {}".format(standalone_expression_to_str(e.children[1]))
            if len(e.children) > 1
            else "",
        ),
        "func_arg_inf": lambda e: "{} := {}".format(
            str(e.children[0].value), standalone_expression_to_str(e.children[1])
        ),
        "func_arg_typed": lambda e: "{}: {}{}".format(
            str(e.children[0].value),
            str(e.children[1].value),
            f" = {standalone_expression_to_str(e.children[2])}"
            if len(e.children) > 2
            else "",
        ),
        "enum_body": _enum_body_to_str,
        "enum_element": _enum_element_to_str,
        "signal_args": _args_to_str,
        "signal_arg_regular": lambda e: str(e.children[0].value),
        "signal_arg_typed": lambda e: "{}: {}".format(
            str(e.children[0].value),
            str(e.children[1].value),
        ),
        "comma_separated_list": lambda e: _arguments_to_str(e.children),
        "contextless_comma_separated_list": lambda e: _arguments_to_str(e.children),
        "contextless_operator_chain_based_expression": (
            _operator_chain_based_expression_to_str
        ),
        "trailing_comma": lambda _: "",
        "annotation": _annotation_to_str,
        "annotation_args": _annotation_args_to_str,
        # New grammar parameter nodes (flat interleaved structure with maybe_placeholders):
        "parameters": _parameters_to_str,
        "parameter": _param_to_str,
        "signal_parameters": _parameters_to_str,
        "signal_parameter": _signal_param_to_str,
        "non_foldable_dot_chain": lambda e: "".join(map(expression_to_str, e.children)),
        "actual_getattr_call": _getattr_call_to_str,
        "actual_subscr_expr": _subscription_to_str,
        # Intermediate expression grammar nodes (unary/binary operators):
        "sign": _format_intermediate_expr,
        "bit_not": _format_intermediate_expr,
        "factor": _format_factor_expr,
        "additive": _format_intermediate_expr,
        "shift_expr": _format_intermediate_expr,
        "mdr_expr": _format_intermediate_expr,
        "arith_expr": _format_intermediate_expr,
        "comparison": _format_intermediate_expr,
        "content_test": _format_intermediate_expr,
        "and_test": _format_intermediate_expr,
        "logic_and": _format_intermediate_expr,
        "or_test": _format_intermediate_expr,
        "logic_or": _format_intermediate_expr,
        "property_custom_getter_args": lambda _: "()",
        # Match pattern types (grammar rules used in match branches):
        "match_pattern": (
            lambda e: f"var {expression_to_str(e.children[1])}"
            if len(getattr(e, 'children', [])) == 2
               and isinstance(e.children[0], Token)
               and str(getattr(e.children[0], 'type', '')).upper() == "VAR"
            else _passthrough(e)
        ),
        # Dict pattern entry: key COLON value (not key = value like dict literals)
        "dict_pattern_entry": lambda e: (
            f"{expression_to_str(e.children[0])}: {expression_to_str(e.children[-1])}"
            if len(getattr(e, 'children', [])) >= 2
            else _passthrough(e)
        ),
        # Dict pattern items container (comma-separated dict_pattern_entry children)
        "dict_pattern_items": lambda e: ", ".join(
            expression_to_str(c) for c in getattr(e, 'children', [])
            if c is not None and isinstance(c, Tree)
        ),
        # Array pattern items container (comma-separated match_pattern children,
        # including PERIOD_PERIOD rest token)
        "array_pattern_items": (
            lambda e: ", ".join(
                expression_to_str(c) for c in getattr(e, 'children', [])
                if c is not None and isinstance(c, Tree)
            )
            + (", .." if any(isinstance(c, Token) and str(getattr(c, 'type', '')).upper() == "PERIOD_PERIOD" for c in getattr(e, 'children', [])) else "")
        ),
        "list_pattern": lambda e: ", ".join(map(expression_to_str, e.children)),
        "test_pattern": _operator_chain_based_expression_to_str,
        "or_pattern": _operator_chain_based_expression_to_str,
        "and_pattern": _operator_chain_based_expression_to_str,
        "not_pattern": lambda e: "{}{}{}".format(
            expression_to_str(e.children[0]),
            "" if e.children[0].value == "!" else " ",
            expression_to_str(e.children[1]),
        ),
        "comp_pattern": _operator_chain_based_expression_to_str,
        "bitw_or_pattern": _operator_chain_based_expression_to_str,
        "bitw_xor_pattern": _operator_chain_based_expression_to_str,
        "bitw_and_pattern": _operator_chain_based_expression_to_str,
        "shift_pattern": _operator_chain_based_expression_to_str,
        "arith_pattern": _operator_chain_based_expression_to_str,
        "mdr_pattern": _operator_chain_based_expression_to_str,
        "neg_pattern": lambda e: f"-{expression_to_str(e.children[1])}",
        "bitw_not_pattern": lambda e: f"~{expression_to_str(e.children[1])}",
        "attr_pattern": lambda e: ".".join(map(expression_to_str, e.children[::2])),
        "call_pattern": lambda e: "{}({})".format(
            expression_to_str(e.children[0]), expression_to_str(e.children[1])
        ),
        "par_pattern": lambda e: f"({expression_to_str(e.children[0])})",
        # `preload(<expr>)` is its own grammar node; without this the generic
        # fallback joins the tokens with spaces (`preload ( "..." )`).
        "preload_expr": lambda e: "preload({})".format(
            expression_to_str(next(
                (c for c in getattr(e, 'children', []) if isinstance(c, Tree)), None
            ))
        ),
        # `super(...)` is its own grammar node for the same reason, and had the same gap: the
        # generic fallback rendered `super ( party, item_data )`. Its one Tree child is the
        # `arguments` node, which is absent for a bare `super()`.
        "super_expr": lambda e: "super({})".format(
            ", ".join(
                expression_to_str(child) for child in getattr(e, 'children', [])
                if isinstance(child, Tree)
            )
        ),
        "var_capture_pattern": lambda e: f"var {expression_to_str(e.children[0])}",
        "etc_pattern": lambda _: "..",
        "wildcard_pattern": lambda _: "_",
        "array_pattern": _array_to_str,
        "array_pattern_items": _pattern_items_to_str,
        "dict_pattern": _dict_to_str,
        "dict_pattern_items": _pattern_items_to_str,
        "dict_pattern_entry": _dict_pattern_entry_to_str,
        "kv_pair_pattern": lambda e: "{}: {}".format(
            expression_to_str(e.children[0]), expression_to_str(e.children[1])
        ),
    }

    # expression.data can be either a string or Token('RULE', 'name')
    # In Lark with maybe_placeholders=True, some grammar nodes use Rule tokens
    key = getattr(expression.data, 'value', str(expression.data)) if not isinstance(expression.data, str) else expression.data
    if key in data_map:
        return data_map[key](expression)
    # Fallback: try treating as intermediate expr node by name
    return _format_intermediate_expr(expression)


def _type_test_or_cast_to_str(expression: Tree) -> str:
    """Render `x is [not] T is [not] T ...` and `x as T as T ...` chains.

    The operand is the first non-None child; each repetition is `IS [NOT] type`
    or `AS type`, so a fixed operand/operator stride (which the generic chain
    loop uses) cannot walk these children.
    """
    children = [c for c in getattr(expression, 'children', []) if c is not None]
    if not children:
        return "?"
    out = [expression_to_str(children[0])]
    for c in children[1:]:
        if isinstance(c, Token):
            ttype = str(getattr(c, 'type', '')).upper()
            if ttype == 'AS':
                out.append(' as ')
            elif ttype == 'IS':
                out.append(' is ')
            elif ttype == 'NOT':
                out.append('not ')
            else:
                out.append(str(getattr(c, 'value', '')))
        else:
            out.append(expression_to_str(c))
    return ''.join(out)


def _type_test_to_str(expression: Tree) -> str:
    """`type_test` with no repetition keeps the generic chain path (it preserves
    outer parens around a compound inner expression); a chain uses the dedicated
    renderer."""
    non_none = [c for c in getattr(expression, 'children', []) if c is not None]
    if len(non_none) == 1:
        return _operator_chain_based_expression_to_str(expression)
    return _type_test_or_cast_to_str(expression)


def _operator_chain_based_expression_to_str(expression: Tree) -> str:
    # Filter out None placeholders from maybe_placeholders
    non_none_children = [c for c in getattr(expression, 'children', []) if c is not None]
    
    d = str(getattr(expression.data, 'value', expression.data))
    
    # Special case: type_test wrapping a paren-wrapped primary that contains compound expressions
    # (and/or/comparison) - preserve outer parens for semantic correctness after unary operators.
    # Only applies when the inner content is NOT a simple literal/constant.
    if d in ('type_test', 'asless_type_test') and len(non_none_children) == 1:
        inner_node = non_none_children[0]
        if isinstance(inner_node, Tree):
            inner_data = str(getattr(inner_node.data, 'value', inner_node.data))
            if inner_data == 'primary':
                # Check for paren-wrapped pattern
                pri_all = [c for c in getattr(inner_node, 'children', []) if c is not None]
                pri_tokens = [c for c in pri_all if isinstance(c, Token)]
                if (len(pri_tokens) >= 2 and
                    getattr(pri_tokens[0], 'type', '').upper() == 'PARENTHESIS_OPEN' and
                    getattr(pri_tokens[-1], 'type', '').upper() == 'PARENTHESIS_CLOSE'):
                    # Check if inner content is a compound expression (not just literals)
                    middle = pri_all[1:-1]
                    has_compound = any(
                        hasattr(c, 'data') and str(getattr(c.data,'value',str(c.data))) in
                        ('and_test','or_test','comparison','arith_expr','additive','shift_expr','mdr_expr','bitw_or','bitw_xor','bitw_and','power','content_test',
                         # New grammar rule names
                         'logic_or','logic_and','factor','bit_shift','bit_and','bit_or','bit_xor')
                        for c in middle if isinstance(c, Tree)
                    )
                    # Also check for paren-wrapped sub-expressions (nested parens)
                    has_nested_parens = any(
                        hasattr(c, 'data') and str(getattr(c.data,'value',str(c.data))) == 'primary'
                        for c in middle if isinstance(c, Tree)
                    )
                    if has_compound or has_nested_parens:
                        # Preserve parens - format inner content
                        parts = []
                        for c in middle:
                            if isinstance(c, Token):
                                tt = getattr(c,'type','').upper()
                                if 'PAR' not in tt and tt != 'NONE':
                                    parts.append(str(c.value))
                            elif hasattr(c, 'data'):
                                try:
                                    result = expression_to_str(c)
                                    # Strip redundant outer parens from sub-expressions
                                    stripped = remove_outer_parentheses(_parse_result_as_tree(result)) if isinstance(result, str) else result
                                    parts.append(expression_to_str(stripped) if isinstance(stripped, Tree) else (str(result) if isinstance(result,str) else '?'))
                                except Exception:
                                    pass
                        inner_formatted = ''.join(parts)
                        return f"({inner_formatted})"
    
    # If only one child (no operator chain), just format it
    if len(non_none_children) <= 1:
        return expression_to_str(non_none_children[0]) if non_none_children else "?"
    
    # Otherwise process as operator chain: operand op operand op ...
    first_expr = non_none_children[0]
    chain = []
    for i in range(1, len(non_none_children) - 1, 2):
        op = non_none_children[i]
        expr = non_none_children[i + 1] if i + 1 < len(non_none_children) else None
        
        if expr is None:
            continue
        
        # Get operator string safely (Token or Tree)
        if isinstance(op, Token):
            op_str = str(op.value)
            padding = "" if op_str.startswith(".") else " "
        elif hasattr(op, 'data'):
            # For Tree operators like comparison_operator, extract their token children
            try:
                op_str_val = expression_to_str(op)
            except (KeyError, AttributeError):
                # Fallback: extract tokens from operator tree directly
                for c in getattr(op, 'children', []):
                    if isinstance(c, Token) and str(getattr(c, 'type', '')).lower() != 'rule':
                        op_str = str(c.value)
                        break
                else:
                    op_str = str(getattr(op.data, 'value', op.data))
                padding = "" if op_str.startswith(".") else " "
                chain.append(f"{padding}{op_str}{padding}{expression_to_str(expr)}")
                continue
            # expression_to_str succeeded - use its output as the operator string
            op_str = str(op_str_val)
            padding = "" if op_str.startswith(".") else " "
        else:
            op_str = str(op)
            padding = " "
        
        chain.append(f"{padding}{expression_to_str(op)}{padding}{expression_to_str(expr)}")
    return f"{expression_to_str(first_expr)}{''.join(chain)}"


def _standalone_call_to_str(call: Tree) -> str:
    from lark import Token as LarkToken
    # Filter out None children (from maybe_placeholders)
    non_none = [c for c in call.children if c is not None]
    callee = expression_to_str(non_none[0]) if len(non_none) > 0 else "?"
    
    # Extract actual arguments - skip LPAR/RPAR tokens
    args_list = []
    inside_parens = False
    for arg in non_none[1:]:
        data_type = str(arg.type) if isinstance(arg, Token) else None  # type: ignore
        
        if data_type == "PARENTHESIS_OPEN":
            inside_parens = True
        elif data_type == "PARENTHESIS_CLOSE":
            inside_parens = False
        else:
            if inside_parens or not data_type:  # Include tree nodes and tokens that aren't parens
                args_list.append(arg)
    
    arguments = _arguments_to_str(args_list) if args_list else ""
    return f"{callee}({arguments})"


def _primary_to_str(primary: Tree) -> str:
    from lark import Token as LarkToken
    # Primary expressions may have tokens (parens) mixed with sub-expressions
    # Filter out None children and format the non-token content
    parts = []
    for c in primary.children:
        if c is None:
            continue
        parts.append(expression_to_str(c))
    return "".join(parts)


def _postfix_to_str(postfix: Tree) -> str:
    from lark import Token as LarkToken
    # Postfix expressions: base_expr call_suffix | base_expr [expr] | etc.
    parts = []
    for c in postfix.children:
        if c is None:
            continue
        s = expression_to_str(c)
        parts.append(s)
    return "".join(parts) if parts else "?"


def _call_suffix_to_str(call_suffix: Tree) -> str:
    # Format call suffix (parenthesized args after function name)
    from lark import Token as LarkToken
    parts = []
    for c in getattr(call_suffix, 'children', []):
        if c is None:
            continue
        if isinstance(c, Token) and str(c.type).lower() == "parenthesis_open":  # type: ignore
            parts.append("(")
        elif isinstance(c, Token) and str(c.type).lower() == "parenthesis_close":  # type: ignore
            parts.append(")")
        else:
            parts.append(expression_to_str(c))
    return "".join(parts)


def _arguments_from_tree(args_tree: Tree) -> str:
    from .expression_utils import is_any_comma  # noqa: F821
    elements = []
    for c in getattr(args_tree, 'children', []):
        if c is None:
            continue
        if not isinstance(c, Token) and hasattr(c, 'data') and str(_tree_data_str(c)).lower() != "trailing_comma":  # type: ignore
            elements.append(standalone_expression_to_str(c))
    return ", ".join(elements)


def _getattr_call_to_str(call: Tree) -> str:
    a_getattr = expression_to_str(call.children[0])
    arguments = _arguments_to_str(call.children[1:])
    return f"{a_getattr}({arguments})"


def _arguments_to_str(arguments):
    from .expression_utils import is_any_comma  # noqa: F821, local import to avoid circular
    return ", ".join(
        standalone_expression_to_str(arg)
        for arg in arguments
        if not is_any_comma(arg)
    )


def _annotation_to_str(annotation: Tree) -> str:
    name = expression_to_str(annotation.children[0])
    if len(annotation.children) > 1:
        return f"@{name}{expression_to_str(annotation.children[1])}"
    return f"@{name}"


def _annotation_args_to_str(annotation: Tree) -> str:
    from .expression_utils import is_any_comma, has_trailing_comma  # noqa: F821
    elements = [
        standalone_expression_to_str(child)
        for child in annotation.children
        if not is_any_comma(child)
    ]
    trailing = "," if has_trailing_comma(annotation) else ""
    return "({}{})".format(", ".join(elements), trailing)


def _lambda_to_str(a_lambda: Tree) -> str:
    """Format a lambda expression from the new flat grammar structure.
    
    For short lambdas, produces inline format: func(params): body
    For long lambdas (params + body exceed line length), produces block format
    with params split across lines and expanded body statements.
    
    New grammar: lambda: FUNC (NAME)? PARENTHESIS_OPEN [parameters] PARENTHESIS_CLOSE
                  [FORWARD_ARROW return_type] COLON suite
    
    With maybe_placeholders=True, children include string placeholders for inline tokens:
      - Named: "func", NAME_str, "(", parameters_tree, ")", ":"(maybe_none), suite_tree
      - Anonymous: "func", "(", parameters_tree, ")", ":"(maybe_none), suite_tree
    """
    from .style_config import StyleConfig  # noqa: local import
    style = StyleConfig()
    max_len = style.line_length
    
    children_list = list(getattr(a_lambda, 'children', []))

    # Extract lambda name (optional) — appears as string placeholder after "func"
    lambda_name = ""
    header_index = 0  # Position we start processing from
    
    for i, child in enumerate(children_list):
        if child is None:
            continue
        if isinstance(child, str) and child == 'func':
            # Next non-None string that isn't "(" is the optional lambda name
            header_index = i + 1
            for j in range(i + 1, len(children_list)):
                next_child = children_list[j]
                if next_child is None:
                    continue
                if isinstance(next_child, str) and next_child != '(':
                    # This is the lambda name (e.g., "baz")
                    lambda_name = f" {next_child}"
                    header_index = j + 1
                elif isinstance(next_child, str) and next_child == '(':
                    header_index = j + 1
                break
            break

    # Extract parameters from the parameters tree
    param_names = []
    for child in children_list[header_index:]:
        if isinstance(child, Tree):
            d = str(getattr(getattr(child, 'data', None), 'value', child.data))
            if d in ('parameters', 'signal_parameters'):
                param_names = lambda_parameter_strings(child)
                break

    # Extract return type separately (it comes after closing paren, before colon)
    has_return_type = False
    return_type_str = ""
    for child in children_list:
        if isinstance(child, Tree):
            d = str(getattr(child.data, 'value', str(child.data)))
            if d == 'return_type':
                from .expression_to_str import expression_to_str as ets  # avoid circular
                has_return_type = True
                return_type_str = f" -> {ets(child)}"
                break

    num_params = len(param_names)

    # Extract suite body statements
    body_statements = []  # List of formatted statement strings
    for child in children_list[header_index:]:
        if isinstance(child, Tree):
            d = str(getattr(getattr(child, 'data', None), 'value', child.data))
            if d == 'suite':
                body_statements = _extract_suite_statements(child)
                break

    # Determine single-line length to decide inline vs block format
    params_str_inline = "(" + ", ".join(param_names) + ")" if param_names else "()"
    # Correct GDScript order: func[name](params) -> return_type:
    single_line_header = f"func{lambda_name}{params_str_inline}"
    if has_return_type:
        single_line_header += return_type_str
    single_line_header += ":"
    body_text_single = " ".join(body_statements) if body_statements else "pass"
    full_single_line = single_line_header + body_text_single
    needs_multiline = len(full_single_line) > max_len or (len(body_statements) > 1 and num_params >= 3)
    
    # Build multiline output if needed, otherwise inline
    if not needs_multiline:
        return f"{single_line_header} {body_text_single}"
    
    # === MULTI-LINE FORMAT ===
    lines = []
    header_prefix_str = f"func{lambda_name}("
    lines.append(header_prefix_str)
    
    # Format each parameter on its own line
    for pi, pname in enumerate(param_names):
        is_last = (pi == len(param_names) - 1)
        if not is_last:
            lines.append(f"{pname},")
        else:
            # Last param: append name, then closing paren with return type if any
            if has_return_type:
                lines.append(pname)
                lines.append(")" + return_type_str + ":")
            else:
                lines.append(f")")
    
    # Format body statements (each on its own indented line)
    indent_str = "\t"  # We'll adjust indentation in context
    for stmt in body_statements:
        if stmt.strip():
            lines.append(f"{indent_str}{stmt}")
    
    return "\n".join(lines)


def _lambda_header_to_str(lambda_header: Tree) -> str:
    has_name = isinstance(lambda_header.children[0], Token) and lambda_header.children[0].type in ["NAME", "GET", "SET"]
    optional_name = f" {str(lambda_header.children[0].value)}" if has_name else ""
    args_idx = 1 if has_name else 0
    arguments = expression_to_str(lambda_header.children[args_idx])
    type_hint_index = 2 if has_name else 1
    expected_len = 3 if has_name else 2
    has_type_hint = len(lambda_header.children) >= (expected_len - 1) and not has_name
    optional_type_hint = f" -> {lambda_header.children[type_hint_index]}" if has_type_hint or (len(lambda_header.children) > type_hint_index) else ""
    return f"func{optional_name}{arguments}{optional_type_hint}:"


def _check_trailing_comma_in_array(array: Tree) -> bool:
    """Check for trailing comma in an array's direct elements.
    
    For the new grammar structure, elements_tree children are:
      [Tree(expr), Token(COMMA), Tree(expr), Token(COMMA), ...]
    With trailing comma: last child is Token(COMMA).
    Without trailing comma: last child is Tree(expr).
    """
    from .expression_utils import has_trailing_comma as orig_has_tc
    from lark import Token
    
    # Direct check on the array itself (old grammar with trailing_comma tree)
    if orig_has_tc(array):
        return True
    
    # Check inside the direct 'elements' child tree (new grammar structure)
    for child in getattr(array, 'children', []):
        if isinstance(child, Tree) and str(getattr(child, 'data', '')).lower() == 'elements':
            children_list = [c for c in getattr(child, 'children', []) if c is not None]
            if children_list:
                last = children_list[-1]
                if isinstance(last, Token) and str(getattr(last, 'type', '')).upper() == "COMMA":
                    return True
    
    return False

def _array_to_str(array: Tree) -> str:
    from .expression_utils import is_any_comma  # noqa: F821
    elements = []
    for child in getattr(array, 'children', []):
        if child is None:
            continue
        if isinstance(child, Token):
            ttype = str(child.type).lower()
            if ttype in ('bracket_open', 'bracket_close'):
                continue  # Skip bracket tokens - we add our own brackets
            continue  # Skip other tokens (commas etc.)
        if not is_any_comma(child):
            d = str(getattr(getattr(child, 'data', None), 'value', getattr(child, 'data', ''))) if hasattr(child, 'data') else ""
            if d in ('elements', 'array_pattern_items'):
                # The elements tree contains: [expr1, COMMA, expr2, COMMA, ...]
                for ec in getattr(child, 'children', []):
                    if isinstance(ec, Token) and str(getattr(ec, 'type', '')).upper() == "COMMA":
                        continue
                    if ec is None or is_any_comma(ec):
                        continue
                    elements.append(standalone_expression_to_str(ec))
            else:
                elements.append(standalone_expression_to_str(child))
    
    # Check trailing comma at DIRECT array level only
    has_tc = _check_trailing_comma_in_array(array)
    trailing = "," if has_tc else ""
    compact = ", ".join(elements)
    
    # Always return compact format here - multiline is handled by expression.py's
    # _format_array_multi which has proper context-aware indentation.
    # This function is used for inline serialization (e.g., inside subscripts).
    return "[{}{}]".format(compact, trailing)


def _dict_to_str(a_dict: Tree) -> str:
    from .expression_utils import is_any_comma, has_trailing_comma  # noqa: F821
    
    # Find the entries subtree if present (grammar wraps entries in this node)
    entries_tree = None
    for child in getattr(a_dict, 'children', []):
        d = str(getattr(child.data, 'value', child.data)) if isinstance(child, Tree) else ''
        if d in ('dict_entries', 'dict_pattern_items'):
            entries_tree = child
            break
    
    children_to_iterate = getattr(entries_tree, 'children', []) if entries_tree is not None else list(getattr(a_dict, 'children', []))
    elements = []
    for child in children_to_iterate:
        if child is None:
            continue
        if isinstance(child, Token):
            ttype = str(child.type).lower()
            if ttype == 'comma':
                continue  # Skip comma tokens
            if ttype in ('brace_open', 'brace_close'):
                continue  # Skip brace tokens - we add our own braces
        if not is_any_comma(child):
            elements.append(expression_to_str(child))
    
    compact = ", ".join(elements)

    if not elements:
        return "{}"
    # Single-line dictionary: the style guide asks for a space after `{` and before `}`
    # so it is visually distinct from an array.
    # A dict with three or more entries used to be turned into a multiline string here,
    # with a hardcoded one-tab indent. That string then reached a caller that had no line
    # structure to put it in (a call argument inside a paren-wrapped chain, say) and came
    # out as a single line containing embedded newlines at the wrong indent. The multiline
    # decision belongs to the layout (`_format_dict_multi`), not to this inline renderer.
    return "{{ {} }}".format(compact)


def _subscription_to_str(subscription: Tree) -> str:
    # subscript grammar: BRACKET_OPEN expression BRACKET_CLOSE
    # Find the expression (skip bracket tokens)
    expr = None
    for child in getattr(subscription, 'children', []):
        if isinstance(child, Token):
            ttype = str(child.type).lower()
            if ttype == 'bracket_open' or ttype == 'bracket_close':
                continue
        elif hasattr(child, 'data'):
            expr = child
            break
    
    if expr is None:
        return "[?]"
    
    # Format expression inline - avoid recursion issues by using _passthrough directly
    subscr_str = _format_subscription_expr(expr)
    return "[{}]".format(subscr_str)


def _format_subscription_expr(node: Tree) -> str:
    """Format an expression for use inside a subscript, avoiding full recursion."""
    from .expression_utils import remove_outer_parentheses  # noqa: F821
    
    # Try to unwrap wrapper trees (assignment/cast/ternary/type_test) first
    unwrapped = _unwrap_wrappers(node)
    if unwrapped is not None and unwrapped != node:
        return expression_to_str(unwrapped)
    
    # Fallback: use the direct children of the subscription
    for child in getattr(node, 'children', []):
        if isinstance(child, Token) or (isinstance(child, Tree) and hasattr(child.data)):
            d = str(getattr(getattr(child, 'data', None), 'value', getattr(child, 'data', '')))
            if d.lower() in ('assignment', 'cast', 'ternary', 'type_test'):
                continue
    
    return expression_to_str(node)


def _dict_element_to_str(dict_element: Tree) -> str:
    template = "{}: {}" if dict_element.data.startswith("c_dict_") else "{} = {}"
    return template.format(
        standalone_expression_to_str(dict_element.children[0]),
        standalone_expression_to_str(dict_element.children[1]),
    )


def _enum_body_to_str(enum_body: Tree) -> str:
    if len(enum_body.children) == 0:
        return "{}"
    elements = map(expression_to_str, enum_body.children)
    return "{{ {} }}".format(", ".join(elements))


def _enum_element_to_str(enum_element: Tree) -> str:
    name = standalone_expression_to_str(enum_element.children[0])
    if len(enum_element.children) > 1:
        value = standalone_expression_to_str(enum_element.children[1])
        return f"{name} = {value}"
    return name


def _args_to_str(args: Tree) -> str:
    return "({})".format(", ".join(map(expression_to_str, args.children)))


def _param_to_str(param: Tree) -> str:
    """Format a single function parameter node.

    Grammar produces flat children (no None slots between named tokens):
        parameter -> NAME [COLON type] [EQUAL expr]
    We walk sequentially to extract components in order.
    Handles := inference operator when COLON is immediately followed by EQUAL.
    """
    name = "?"
    param_type = ""
    default_val = None
    variadic = False  # `...name` variadic parameter
    uses_inference_operator = False  # Track := (colon with no type between)

    # Walk children sequentially: first non-None NAME token is the name,
    # then optionally COLON + type tree, then optionally EQUAL + expr tree
    children_list = [c for c in getattr(param, 'children', []) if c is not None]
    i = 0
    while i < len(children_list):
        child = children_list[i]
        if isinstance(child, Token):
            ttype = str(getattr(child, 'type', '')).lower()
            if ttype == "period_period_period":
                variadic = True
            elif ttype == "name" and name == "?":
                name = str(getattr(child, 'value', '?'))
            elif ttype == "colon":
                # Check if next child is a type Tree or EQUAL (inference operator :=)
                if i + 1 < len(children_list) and isinstance(children_list[i + 1], Tree):
                    param_type = expression_to_str(children_list[i + 1])
                elif i + 1 < len(children_list):
                    next_token = children_list[i + 1]
                    # If immediately followed by EQUAL, it's := inference operator
                    if isinstance(next_token, Token) and str(getattr(next_token, 'type', '')).lower() == "equal":
                        uses_inference_operator = True
            elif ttype == "equal":
                # Next non-token child is the default value expression
                for j in range(i + 1, len(children_list)):
                    next_child = children_list[j]
                    if isinstance(next_child, Tree):
                        try:
                            from .expression_to_str import standalone_expression_to_str as sats
                            default_val = sats(next_child)
                        except Exception:
                            pass
                        break
        i += 1

    if variadic:
        name = "..." + name

    # Build output string based on what we found
    if param_type and default_val:
        return f"{name}: {param_type} = {default_val}"
    elif param_type:
        return f"{name}: {param_type}"
    elif uses_inference_operator and default_val:
        # := inference operator: VAR_NAME := value
        return f"{name} := {default_val}"
    elif default_val:
        return f"{name} = {default_val}"
    return name


def _signal_param_to_str(param: Tree) -> str:
    """Format a single signal parameter node (no type defaults for signals).

    Grammar produces flat children with maybe_placeholders:
        signal_parameter -> NAME [COLON type]
    where None slots appear for missing optional parts.
    """
    name = "?"
    param_type = ""

    for child in getattr(param, 'children', []):
        if child is None:
            continue
        if isinstance(child, Token):
            ttype = str(getattr(child, 'type', '')).lower()
            if ttype == "name" and name == "?":
                name = str(getattr(child, 'value', '?'))
                continue
            if ttype == "colon":
                # Type comes after COLON - get next tree child
                for nc in getattr(param, 'children', []):
                    if nc is not None and isinstance(nc, Tree):
                        param_type = expression_to_str(nc)
                        break
                break
        elif isinstance(child, Tree) and name != "?" and not param_type:
            # This is the type tree after NAME token (without colon in some cases?)
            data_name = getattr(child.data, 'value', str(child.data)).lower()
            if "type" in data_name or "expr" in data_name or "test" in data_name:
                param_type = expression_to_str(child)

    if param_type:
        return f"{name}: {param_type}"
    return name


def _parameters_to_str(params: Tree) -> str:
    """Format parameters/signal_parameters as parenthesized comma-separated list.

    Grammar produces flat interleaved children:
        parameters -> (parameter | COMMA)* [trailing_comma?]
    where Nones appear from maybe_placeholders inside parameter nodes.
    """
    elements = []
    for child in getattr(params, 'children', []):
        if child is None:
            continue
        # Skip comma tokens and trailing commas - we add our own commas
        ttype = str(getattr(child, 'type', '')).lower() if isinstance(child, Token) else ""
        if ttype in ("comma", "trailing_comma") or child is None:
            continue
        # Only process parameter/signal_parameter trees
        data_name = getattr(child.data, 'value', str(getattr(child, 'data', '')))
        if isinstance(child, Tree) and ("parameter" in data_name.lower()):
            param_str = expression_to_str(child)
            elements.append(param_str)
    return "({})".format(", ".join(elements))


def lambda_parameter_strings(parameters_tree) -> list:
    """Render a lambda/function parameter list as one string per parameter.

    Yields strings such as `['x', 'y: int', 'z = 1', '...rest']`. Only
    `parameter` subtrees are parameters: `lark.Token` subclasses `str`, so a
    walk that tests `isinstance(child, str)` before the tree check mistakes the
    `:` and `=` punctuation tokens for parameter names and renders
    `func(x: int)` as `func(x, :)`.
    """
    if not isinstance(parameters_tree, Tree):
        return []
    result = []
    for child in getattr(parameters_tree, 'children', []):
        if not isinstance(child, Tree):
            continue
        data_name = str(getattr(child.data, 'value', child.data))
        if 'parameter' in data_name.lower():
            result.append(expression_to_str(child))
    return result


def _long_string_to_str(string: Token):
    """Format a long/triple-quoted string token.

    A single-line `'''...'''` string is normalized to a plain quoted string the way the
    legacy formatter did it: drop two characters from each end and run the regular
    string rules, so `'''x'''` becomes `"x"` and `'''x "'''` becomes `'x "'`. A
    triple-double-quoted string is kept as written, and any string containing a newline
    keeps its triple quotes because it cannot be written on one line.
    """
    actual_string = string.value
    if actual_string.startswith('r'):
        return _prefixed_string_to_str(string)
    if (actual_string.startswith("'''") and actual_string.endswith("'''")
            and len(actual_string) >= 6):
        if "\n" in actual_string:
            return actual_string
        return _regular_string_to_str(Token("REGULAR_STRING", actual_string[2:-2]))
    return actual_string

def _is_triple_quoted(value):
    """Check if a token value is a triple-quoted string."""
    for prefix in ('"""', "'''"):
        if value.startswith(prefix) and value.endswith(prefix) and len(value) >= 6:
            return True
    return False


def _long_rstring_to_str(rstring: Token) -> str:
    # Same rule as the one-line raw string: the body is literal, so it is kept verbatim and
    # only a `'''` delimiter with nothing to re-escape becomes `"""`.
    return _raw_string_to_str(str(rstring.value))


def _statement_to_str(node) -> str:
    """Render a statement node with the real statement formatter.

    `expression_to_str` on a raw statement tree mis-spaces a type annotation and
    the inference operator (`var x : int`, `var x : = 1`); the statement formatter
    gives `var x: int` and `var x := 1`.
    """
    from .function_statement import format_func_statement
    try:
        result = format_func_statement(node, "", 0, None)
    except Exception:
        return expression_to_str(node)
    lines = result[0] if isinstance(result, tuple) else result
    texts = [
        text.strip() for _line_no, text in lines
        if isinstance(text, str) and text.strip() and text.strip() != 'END'
    ]
    return " ".join(texts) if texts else expression_to_str(node)


def _suite_to_inline_str(suite: Tree) -> str:
    """Format a suite as an inline string for lambda expressions.
    
    For single statements, returns the formatted statement directly.
    For multiple statements, joins them with semicolons (inline format).
    """
    from .expression_to_str import expression_to_str
    from ..formatter.function_statement_to_str import function_statement_to_str as _fsts
    
    # Find actual statement children (skip NL/indent/dedent tokens)
    statements = []
    for child in getattr(suite, 'children', []):
        if child is None:
            continue
        d = str(getattr(getattr(child, 'data', None), 'value', getattr(child, 'data', ''))) if hasattr(child, 'data') else ''
        
        # Skip structural tokens
        if isinstance(child, Token):
            ttype = str(child.type).lower()
            if ttype in ('_nl', '_indent', '_dedent', 'end'):
                continue
            statements.append(('token', child))
            continue
        
        # Get statement content
        if d == 'suite_line':
            # Unwrap suite_line to get actual statement
            for sub in getattr(child, 'children', []):
                if sub is None:
                    continue
                sd = str(getattr(getattr(sub, 'data', None), 'value', getattr(sub, 'data', ''))) if hasattr(sub, 'data') else ''
                if isinstance(sub, Token):
                    ttype = str(sub.type).lower()
                    if ttype not in ('_nl', '_indent', '_dedent', 'end'):
                        statements.append(('token', sub))
                elif sd == 'pass_stmt':
                    statements.append(('pass', sub))
                elif sd in ('flow_stmt', 'return_stmt'):
                    statements.append(('stmt', sub))
                else:
                    statements.append(('expr', sub))
        elif d == 'pass_stmt':
            statements.append(('pass', child))
        elif d == 'flow_stmt':  # return, break, continue
            statements.append(('stmt', child))
        elif hasattr(child, 'data'):  # expression statements
            statements.append(('expr', child))
    
    if not statements:
        return "pass"
    
    parts = []
    for kind, node in statements:
        if kind == 'token':
            val = str(getattr(node, 'value', ''))
            if val: parts.append(val)
        elif kind == 'pass':
            parts.append('pass')
        elif kind == 'stmt' or kind == 'expr':
            try:
                part_str = _statement_to_str(node)
                parts.append(part_str)
            except Exception:
                pass

    return " ".join(parts)


def _extract_suite_statements(suite: Tree) -> list:
    """Extract formatted statement strings from a suite tree for multiline lambda body.
    
    Returns a list of string representations for each statement in the suite,
    suitable for formatting as separate lines in block-style lambda output.
    """
    statements = []
    for child in getattr(suite, 'children', []):
        if child is None:
            continue
        d = str(getattr(getattr(child, 'data', None), 'value', getattr(child, 'data', ''))) if hasattr(child, 'data') else ''
        
        # Skip structural tokens
        if isinstance(child, Token):
            ttype = str(child.type).lower()
            if ttype in ('_nl', '_indent', '_dedent', 'end'):
                continue
            val = getattr(child, 'value', '')  # type: ignore
            if val and not any(t in ttype for t in ['_nl', '_indent', '_dedent']):
                statements.append(str(val))
            continue
        
        # Get statement content
        if d == 'suite_line':
            # Unwrap suite_line to get actual statement
            for sub in getattr(child, 'children', []):
                if sub is None:
                    continue
                sd = str(getattr(getattr(sub, 'data', None), 'value', getattr(sub, 'data', ''))) if hasattr(sub, 'data') else ''
                if isinstance(sub, Token):
                    ttype = str(sub.type).lower()
                    if ttype not in ('_nl', '_indent', '_dedent', 'end'):
                        val = getattr(sub, 'value', '')  # type: ignore
                        if val:
                            statements.append(str(val))
                elif sd == 'pass_stmt':
                    statements.append('pass')
                elif sd in ('flow_stmt', 'return_stmt'):
                    try:
                        stmt_str = _statement_to_str(sub)
                        if stmt_str.strip():
                            statements.append(stmt_str)
                    except Exception:
                        pass
                else:
                    try:
                        expr_str = _statement_to_str(sub)
                        if expr_str.strip():
                            statements.append(expr_str)
                    except Exception:
                        pass
        elif d == 'pass_stmt':
            statements.append('pass')
        elif d in ('flow_stmt', 'return_stmt'):  # return, break, continue at top level of suite
            try:
                stmt_str = _statement_to_str(child)
                if stmt_str.strip():
                    statements.append(stmt_str)
            except Exception:
                pass
        elif hasattr(child, 'data'):  # expression statements / var declarations
            try:
                expr_str = _statement_to_str(child)
                if expr_str.strip() and not any(t in d for t in ['_nl', '_indent']):
                    statements.append(expr_str)
            except Exception:
                pass
    
    return statements


def _prefixed_string_to_str(string: Token) -> str:
    """Format a STRING token with an r/&/^ prefix (e.g., r"...", &"...", ^"...").

    The Earley grammar includes these prefixes as part of the STRING terminal. The
    content is normalized exactly like a plain string literal (quote choice and
    escapes) and the prefix is kept, so `r'x'` becomes `r"x"` and
    `r'''x'''` becomes `r"x"`.
    """
    value = string.value
    for prefix in ('r', '&', '^'):
        if value.startswith(prefix) and len(value) > 1 and value[1] in ('"', "'"):
            inner = value[len(prefix):]
            if prefix == 'r':
                # A raw body is literal, so the escape-rewriting rules below would change
                # its value; only the delimiter may move.
                return _raw_string_to_str(value)
            if _is_triple_quoted(inner):
                return prefix + _long_string_to_str(Token("STRING", inner))
            return prefix + _regular_string_to_str(Token("STRING", inner))
    return value


def _string_token_to_str(token) -> str:
    """Normalize any string token (plain, prefixed, or triple-quoted) to canonical form."""
    value = str(token.value)
    for prefix in ('r', '&', '^'):
        if value.startswith(prefix) and len(value) > 1 and value[1] in ('"', "'"):
            return _prefixed_string_to_str(token)
    if _is_triple_quoted(value):
        return _long_string_to_str(token)
    return _regular_string_to_str(token)


def _regular_string_to_str(string: Token) -> str:
    actual_string = string.value
    actual_string_data = actual_string[1:-1]
    quotes_num = actual_string_data.count('"')
    aposes_num = actual_string_data.count("'")
    target = '"' if quotes_num <= aposes_num else "'"
    if target == '"' and actual_string.startswith("'"):
        actual_string_data = actual_string_data.replace("\\'", "'")
        actual_string_data = actual_string_data.replace('"', '\\"') 
    if target == "'" and actual_string.startswith('"'):
        actual_string_data = actual_string_data.replace('\\"', '"')
        actual_string_data = actual_string_data.replace("'", "\\'")
    return f"{target}{actual_string_data}{target}"


def _rstring_to_str(node) -> str:
    """Render a `rstring` node, whose child is the raw string's plain token.

    Prefixing `r` to the child *after* rendering it through the plain string rules would
    re-escape a body the `r` makes literal - the very bug `_raw_string_to_str` avoids. The
    child's own text is used instead, so the `r` is part of what gets normalized.
    """
    children = getattr(node, 'children', None) or []
    child = children[0] if children else node
    if isinstance(child, Token) and len(str(child)) > 1 and str(child)[0] in ('"', "'"):
        return _raw_string_to_str("r" + str(child))
    return f"r{expression_to_str(child)}"


def _raw_string_to_str(value: str) -> str:
    """Normalize a raw string's delimiters without ever touching its body.

    The body of a raw string is literal, so the normal string rules must not run on it:
    `r"x \\'"` is four characters, and dropping that backslash - which is what
    `_regular_string_to_str` does when it moves a string between quote styles - changes the
    value. Godot agrees: `r"x \\'"` has length 4 and `r"x '"` has length 3. So a body holding
    a backslash is kept verbatim, and only a body that needs no escaping is re-delimited the
    way a plain string would be (double quotes unless the body contains one).
    """
    if not value.startswith("r"):
        return value
    body = value[1:]
    inner = None
    for quotes in ('"""', "'''"):
        if body.startswith(quotes) and body.endswith(quotes) and len(body) >= 6:
            inner = body[3:-3]
            break
    if inner is None:
        if len(body) < 2 or body[0] not in ('"', "'"):
            return value
        inner = body[1:-1]
    if "\\" in inner:
        return value
    if "\n" not in inner:
        if '"' not in inner:
            return f'r"{inner}"'
        if "'" not in inner:
            return f"r'{inner}'"
        return value
    if '"""' not in inner:
        return f'r"""{inner}"""'
    return value


def _regular_rstring_to_str(rstring: Token) -> str:
    return _raw_string_to_str(str(rstring.value))


def _unwrap_wrappers(node: Tree, depth=0) -> Tree:
    """Unwrap assignment/cast/ternary/type_test wrapper trees to find the actual expression.
    
    Returns the unwrapped node or None if no non-wrapping child is found.
    Stops recursing at max_depth to prevent infinite loops.
    """
    WRAPPERS = frozenset({'assignment', 'cast', 'ternary', 'type_test'})
    d = str(getattr(getattr(node, 'data', None), 'value', getattr(node, 'data', ''))).lower()
    if depth > 10 or d not in WRAPPERS:
        return node
    # Get first non-None Tree child (skipping None placeholders from maybe_placeholders)
    for c in getattr(node, 'children', []):
        if c is None:
            continue
        if isinstance(c, Token):
            continue  # Skip operator tokens
        if isinstance(c, Tree):
            return _unwrap_wrappers(c, depth + 1)
    # Fallback: use iter_subtrees to find actual Tree nodes
    for subtree in getattr(node, 'iter_subtrees', lambda: [])():
        if isinstance(subtree, Tree):
            return _unwrap_wrappers(subtree, depth + 1)
    return node


def _is_paren_expr(node: Tree) -> bool:
    """Check if a primary tree is just parentheses around an expression.
    
    New grammar: primary: PARENTHESIS_OPEN expression PARENTHESIS_CLOSE
    Returns True when this pattern matches (3 children: LPAR, expr, RPAR).
    """
    if not isinstance(node, Tree):
        return False
    d = str(getattr(getattr(node, 'data', None), 'value', getattr(node, 'data', ''))).lower()
    if d != 'primary':
        return False
    children = [c for c in getattr(node, 'children', []) if c is not None]
    if len(children) != 3:
        return False
    first, middle, last = children[0], children[1], children[2]
    # Check LPAR ... expr ... RPAR pattern
    if isinstance(first, Token) and str(getattr(first, 'type', '')).upper() == 'PARENTHESIS_OPEN':
        if isinstance(last, Token) and str(getattr(last, 'type', '')).upper() == 'PARENTHESIS_CLOSE':
            return True
    return False


def _paren_wrapped_primary_inner(node):
    """Inner expression of the outermost paren-wrapped `primary`, or None.

    Walks the single-child wrapper chain (`assignment`/`cast`/`ternary`/`type_test`)
    down to the `primary`, because a parenthesized expression is wrapped in those.
    """
    wrappers = ('assignment', 'cast', 'ternary', 'type_test')
    current = node
    for _ in range(12):
        if not isinstance(current, Tree):
            return None
        d = str(getattr(getattr(current, 'data', None), 'value', getattr(current, 'data', ''))).lower()
        if d == 'primary':
            if not _is_paren_expr(current):
                return None
            children = [c for c in getattr(current, 'children', []) if c is not None]
            return children[1] if len(children) >= 3 else None
        if d not in wrappers:
            return None
        nxt = next((c for c in getattr(current, 'children', [])
                    if c is not None and not isinstance(c, Token)), None)
        if nxt is None:
            return None
        current = nxt
    return None


def match_pattern_to_str(node) -> str:
    """Render a match pattern, keeping the parentheses the author wrote.

    A parenthesized pattern keeps them (`(1)` stays `(1)`, the golden does this), where
    an ordinary expression strips a redundant pair. `_format_primary_with_paren_removal`
    already keeps the parens around an operator chain, so only a paren-wrapped primary
    whose inner expression would otherwise be unwrapped needs them re-added here.
    """
    text = expression_to_str(node)
    if len(text) > 1 and text.startswith('(') and text.endswith(')'):
        return text
    inner = _paren_wrapped_primary_inner(node)
    if inner is None:
        return text
    return f"({expression_to_str(inner)})"


def _remove_parentheses(node: Tree) -> Tree:
    """Remove unnecessary parentheses from a tree.
    
    Recursively strips outer (expr) patterns in primary nodes, preserving
    only the inner expression. This matches the original formatter's behavior
    where redundant parens are stripped.
    
    Returns the unwrapped node.
    """
    current = _unwrap_wrappers(node)
    # Strip parentheses: while it's a paren-wrapped primary, unwrap further
    while isinstance(current, Tree) and _is_paren_expr(current):
        children = [c for c in getattr(current, 'children', []) if c is not None]
        current = children[1]  # The inner expression
        current = _unwrap_wrappers(current)
    return current


def _format_paren_expr(node: Tree) -> str:
    """Format a parenthesized expression, stripping unnecessary parens."""
    unwrapped = _remove_parentheses(node)
    inner_str = standalone_expression_to_str(unwrapped.children[0]) if isinstance(unwrapped, Tree) and hasattr(unwrapped, 'children') else '?'
    return f"({inner_str})"


def _format_get_node_expr(node: Tree) -> str:
    """Format get_node_expr for node paths like $Path/To, %Unique/Child.
    
    Grammar (new Earley grammar):
        get_node_expr: DOLLAR (SLASH | PERCENT)? node_path_segment ((SLASH | PERCENT) node_path_segment)*
                     | PERCENT node_path_segment ((SLASH | PERCENT) node_path_segment)*
    
    With maybe_placeholders=True, children may include None placeholders.
    """
    non_none = [c for c in getattr(node, 'children', []) if c is not None]
    if not non_none:
        return "$?"
    
    # Determine prefixes: detect $ and/or % and /root patterns
    first_child = non_none[0]
    has_dollar = False
    has_percent = False
    has_leading_slash = False  # Detect $/root pattern (absolute path)
    prefix = "$"
    
    if isinstance(first_child, Token):
        ft = str(getattr(first_child, 'type', '')).upper()
        if ft == 'DOLLAR':
            has_dollar = True
        elif ft == 'PERCENT':
            has_percent = True
    
    # Check for $/root pattern (SLASH immediately after DOLLAR)
    if len(non_none) >= 3:
        second_child = non_none[1]
        third_child = non_none[2]
        if (isinstance(first_child, Token) and str(getattr(first_child, 'type', '')).upper() == 'DOLLAR'):
            if isinstance(second_child, Token) and str(getattr(second_child, 'type', '')).upper() == 'SLASH':
                has_leading_slash = True
    
    # Check for $% pattern (dollar + percent as unique node prefix)
    if len(non_none) >= 2:
        second_child = non_none[1]
        if isinstance(first_child, Token) and str(getattr(first_child, 'type', '')).upper() == 'DOLLAR':
            if isinstance(second_child, Token) and str(getattr(second_child, 'type', '')).upper() == 'PERCENT':
                has_percent = True
    
    # Collect all path segments
    prefix_parts = []
    if has_dollar:
        prefix_parts.append("$")
    if has_percent and not has_leading_slash:  # Only add % when not a $/root pattern
        prefix_parts.append("%")
    
    segments = []
    skip_indices = set()
    for i, c in enumerate(non_none):
        if c is None:
            continue
        ttype = str(getattr(c, 'type', '')).upper()
        # Skip prefix tokens (first child always skipped)
        if isinstance(first_child, Token) and c is first_child:
            skip_indices.add(i)
            continue
        if ttype in ('DOLLAR', 'PERCENT'):
            # These change the prefix for subsequent segments - but they were already handled above
            skip_indices.add(i)
            continue
        # Skip SLASH tokens (they are separators, not part of output)
        if ttype == 'SLASH':
            skip_indices.add(i)
            continue
        if isinstance(c, Token):
            segments.append(str(getattr(c, 'value', '?')))
        elif hasattr(c, 'data'):
            d = str(getattr(getattr(c, 'data', None), 'value', getattr(c, 'data', '')))
            if d == 'node_path_segment':
                # node_path_segment can be STRING | NAME | node_name_keyword
                for sc in getattr(c, 'children', []):
                    if isinstance(sc, Token):
                        segments.append(str(getattr(sc, 'value', '?')))
                    elif isinstance(sc, str):
                        segments.append(sc)
            else:
                # Try to format as expression
                try:
                    seg_str = expression_to_str(c)
                    segments.append(seg_str)
                except Exception:
                    pass
    
    # Build the result: prefix + (optional leading /) + segments joined by /
    path_str = '/'.join(segments) if segments else ''
    if has_leading_slash:
        return f"{''.join(prefix_parts)}/{path_str}"
    return f"{''.join(prefix_parts)}{path_str}"


def _format_primary_with_paren_removal(node: Tree) -> str:
    """Format a primary expression, stripping unnecessary parentheses.
    
    New grammar: primary includes PARENTHESIS_OPEN expr PARENTHESIS_CLOSE as one option.
    We need to strip these when they wrap a single sub-expression,
    matching the original formatter's behavior.
    """
    non_none = [c for c in getattr(node, 'children', []) if c is not None]
    
    # Check for paren-wrapped pattern: (expr)
    if _is_paren_expr(node):
        children = [c for c in node.children if c is not None]
        inner = children[1]  # The expression between parens
        
        # Don't strip parentheses around additive/arith chains - they aid readability
        # in binary expressions like '(1+1)+1' which should keep the parens.
        d_inner = ''
        if isinstance(inner, Tree):
            # Unwrap the scaffolding the grammar wraps every expression in - assignment/cast/
            # ternary/type_test layers whose optional slots are empty - to reach the real
            # expression the keep list below has to judge.
            test_node = inner
            for _ in range(10):
                if not isinstance(test_node, Tree):
                    break  # Hit a Token - stop unwrapping
                td = str(getattr(test_node.data, 'value', test_node.data))
                if td in ('assignment', 'cast', 'ternary', 'type_test'):
                    nc = [c for c in getattr(test_node, 'children', []) if c is not None]
                    # A layer carrying an operator of its own - `as`, `is`, `if`/`else`, `=` -
                    # is not scaffolding: the parenthesised group sits under it as its left
                    # operand, so this layer is the expression's real top and belongs in the
                    # keep list below, which decides whether the group is load-bearing.
                    if len(nc) != 1:
                        d_inner = td
                        break
                    test_node = nc[0]
                else:
                    d_inner = td
                    break
            else:
                # Reached max depth or no valid data - use inner node's own type
                d_inner = str(getattr(inner.data, 'value', inner.data))

        paren_types_to_keep = ('additive', 'shift_expr', 'mdr_expr', 'arith_expr',
                               'bitw_or', 'bitw_xor', 'bitw_and', 'content_test',
                               'comparison', 'and_test', 'or_test', 'power',
                               # A real cast, type test, ternary or assignment binds looser
                               # than what the group holds, so the group decides the result:
                               # `(a or b) as int` is not `a or b as int`, which re-binds the
                               # `or`, and `(a if b else c) as int` is not `a if b else c as int`.
                               'cast', 'type_test', 'ternary', 'assignment',
                               # New grammar rule names
                               'logic_or', 'logic_and', 'logic_not',
                               'bit_shift', 'factor', 'bit_and', 'bit_or', 'bit_xor')
        if d_inner in paren_types_to_keep:
            # Format inner without stripping parens, then wrap with our own parens
            inner_str = expression_to_str(inner)
            return f"({inner_str})"
        
        return standalone_expression_to_str(inner)
    
    # Otherwise, format normally based on the first non-token child
    result_parts = []
    for c in non_none:
        if isinstance(c, Token):
            ttype = str(getattr(c, 'type', '')).upper()
            # Skip paren tokens - we handle them above
            if ttype == 'PARENTHESIS_OPEN' or ttype == 'PARENTHESIS_CLOSE':
                continue
            result_parts.append(str(getattr(c, 'value', '')))
        elif hasattr(c, 'data'):
            d = str(getattr(getattr(c, 'data', None), 'value', getattr(c, 'data', '')))
            if d in ('get_node_expr',):
                result_parts.append(_format_get_node_expr(c))
            else:
                try:
                    result_parts.append(expression_to_str(c))
                except Exception:
                    pass
        elif isinstance(c, str):
            # String placeholder from maybe_placeholders
            result_parts.append(c)
    
    return ''.join(result_parts)


def _format_factor_expr(node: Tree) -> str:
    """Format a factor expression, with special handling for node paths.
    
    The Earley grammar parses $Path/To as:
      factor -> type_test(get_node_expr($Some)) / type_test(Path)
    This is because SLASH serves dual purpose (path separator and division).
    We detect this pattern and format the whole thing as a single node path.
    """
    non_none = [c for c in getattr(node, 'children', []) if c is not None]
    
    # Check if first child contains get_node_expr
    first_has_getnode = _has_get_node_expr(non_none[0]) if len(non_none) > 0 else False
    slash_indices = []
    for i, child in enumerate(non_none):
        if isinstance(child, Token) and str(getattr(child, 'type', '')).upper() == 'SLASH':
            slash_indices.append(i)
    
    # If first operand is get_node_expr and we have slashes after it,
    # format as a single node path (no spaces around /)
    if first_has_getnode and slash_indices:
        parts = []
        for i, child in enumerate(non_none):
            if _has_get_node_expr(child) or isinstance(child, Token) and str(getattr(child, 'type', '')).upper() == 'SLASH':
                # For get_node_expr: format it
                # For SLASH: add a slash character (no spaces)
                if _has_get_node_expr(child):
                    parts.append(_get_inner_expr(child))
                elif isinstance(child, Token) and str(getattr(child, 'type', '')).upper() == 'SLASH':
                    parts.append('/')
            else:
                try:
                    parts.append(expression_to_str(child))
                except Exception:
                    parts.append('?')
        return ''.join(parts)
    
    # Regular factor handling - use intermediate expression format
    return _format_intermediate_expr(node)


def _has_get_node_expr(child) -> bool:
    """Check if a child (or its wrapper) contains a get_node_expr."""
    d = str(getattr(getattr(child, 'data', None), 'value', getattr(child, 'data', ''))).lower()
    if d == 'get_node_expr':
        return True
    # Check inside type_test/postfix wrappers
    if d in ('type_test', 'await_expr', 'postfix'):
        for subc in getattr(child, 'children', []):
            if _has_get_node_expr(subc):
                return True
    return False


def _get_inner_expr(child) -> str:
    """Get the inner expression string from a node, unwrapping wrappers."""
    d = str(getattr(getattr(child, 'data', None), 'value', getattr(child, 'data', ''))).lower()
    if d == 'get_node_expr':
        return _format_get_node_expr(child)
    # If wrapped in type_test etc., unwrap and get inner
    non_none = [c for c in getattr(child, 'children', []) if c is not None]
    for subc in non_none:
        sd = str(getattr(getattr(subc, 'data', None), 'value', getattr(subc, 'data', ''))).lower()
        if sd == 'get_node_expr':
            return _format_get_node_expr(subc)
        elif sd in ('type_test', 'await_expr'):
            # Recurse deeper
            inner = _unwrap_wrappers(child)
            try:
                return expression_to_str(inner)
            except Exception:
                pass
    # Fallback: format normally but strip parens
    unwrapped = _unwrap_wrappers(child)
    if isinstance(unwrapped, Tree) and hasattr(unwrapped, 'children'):
        inner = _remove_parentheses(unwrapped)
        return standalone_expression_to_str(inner)
    try:
        return expression_to_str(child)
    except Exception:
        return '?'


def _passthrough(node):
    """Passthrough: get the first non-None Tree child and recursively format it.
    
    Skips operator tokens (MINUS, PLUS, TILDE, etc.) to find actual expression
    children. Handles both Tree nodes and Token RULE placeholders.
    With maybe_placeholders=True, some slots in node.children are None but
    contain nested Tree structures — use iter_subtrees() to find them.
    """
    if not hasattr(node, 'children') or not node.children:
        return "?"

    # Try normal children iteration first
    for c in node.children:  # type: ignore
        if c is None:
            continue
        if isinstance(c, Token):
            token_type = str(getattr(c, 'type', '')).upper()
            op_types = {"MINUS", "PLUS", "TILDE", "BANG", "NOT",
                        "STAR", "SLASH", "PERCENT", "AND_AMPERSAND",
                        "PIPE_PIPE", "OR", "CARET", "LESS_LESS",
                        "GREATER_GREATER", "AMPERSAND", "PIPE",
                        "ASTERISK_STAR", "RULE", "FUNC"}
            if token_type in op_types:
                continue  # Skip operator tokens and RULE placeholders
            return str(c.value)
        return expression_to_str(c)

    # If all children are None (maybe_placeholders structure),
    # recurse into subtrees to find actual Tree nodes
    for subtree in getattr(node, 'iter_subtrees', lambda: [])():
        if isinstance(subtree, Token):
            token_type = str(getattr(subtree, 'type', '')).upper()
            op_types = {"MINUS", "PLUS", "TILDE", "BANG", "NOT",
                        "STAR", "SLASH", "PERCENT", "AND_AMPERSAND",
                        "PIPE_PIPE", "OR", "CARET", "LESS_LESS",
                        "GREATER_GREATER", "AMPERSAND", "PIPE",
                        "ASTERISK_STAR", "RULE", "FUNC"}
            if token_type in op_types:
                continue
            return str(subtree.value)
        return expression_to_str(subtree)

    return "?"


def _format_sign(node: Tree) -> str:
    """Format unary sign expressions (+/- prefix operators).
    
    Grammar structure: sign: (PLUS | MINUS) sign | bit_not | power
    Recursive - each sign can contain another sign.
    E.g., +1 → [Token('PLUS', '+'), Tree(type_test, ...)]
         --x → [Token('-'), Token('-'), ...]
    """
    prefix = ""
    for c in getattr(node, 'children', []):
        if c is None:
            continue
        if isinstance(c, Token):
            ttype = str(getattr(c, 'type', '')).upper()
            if ttype == 'PLUS':
                prefix += "+"
            elif ttype == 'MINUS':
                prefix += "-"
            # Other tokens (TILDE etc.) are handled by bit_not
        elif isinstance(c, Tree):
            # Could be another sign (recursive) or the base expression
            d = getattr(getattr(c, 'data', None), 'value', str(c.data)) if hasattr(getattr(c, 'data', None), 'value') else str(c.data)
            if d == 'sign':
                prefix += _format_sign(c)
            elif isinstance(c, Token):
                ttype = str(getattr(c, 'type', '')).upper()
                if ttype == 'MINUS':
                    prefix += "-"
                elif ttype == 'TILDE':
                    prefix += "~"
                # For other tokens inside sign (like NAME), include them
            else:
                return f"{prefix}{expression_to_str(c)}"
    
    # If no operator found in children, recurse into subtrees to find the base expression
    if not prefix:
        for subtree in getattr(node, 'iter_subtrees', lambda: [])():
            if isinstance(subtree, Token):
                ttype = str(getattr(subtree, 'type', '')).upper()
                if ttype == 'PLUS':
                    prefix += "+"
                elif ttype == 'MINUS':
                    prefix += "-"
                continue
            return f"{prefix}{expression_to_str(subtree)}"
    
    # Get the base expression from subtrees
    for subtree in getattr(node, 'iter_subtrees', lambda: [])():
        if isinstance(subtree, Token):
            ttype = str(getattr(subtree, 'type', '')).upper()
            if ttype not in ('PLUS', 'MINUS'):
                return f"{prefix}{str(subtree.value)}"
        elif hasattr(subtree, 'data'):
            d = getattr(getattr(subtree, 'data', None), 'value', str(subtree.data)) if hasattr(getattr(subtree, 'data', None), 'value') else str(subtree.data)
            if d not in ('sign',):
                return f"{prefix}{expression_to_str(subtree)}"
    
    return "?"


def _format_intermediate_expr(node: Tree) -> str:
    """Format intermediate grammar nodes (sign, bit_not, factor, additive, etc.).

    These are unary/binary operator wrappers that don't have their own handlers.
    Walks children collecting operator tokens and expression Trees in order,
    then formats them as a flat operand-operator chain. Also handles trailing
    operator chains where grammar uses *(...) rules for multiplicative ops,
    +(...) for additive, etc.
    """
    non_none = [c for c in getattr(node, 'children', []) if c is not None]

    # For ternary expressions: expr_if IF expr_then ELSE expr_else
    dn = getattr(getattr(node, 'data', None), 'value', str(node.data)) \
        if hasattr(getattr(node, 'data', None), 'value') else str(node.data)
    if dn == 'ternary' and len(non_none) >= 4:
        # Check for IF token
        has_if = any(isinstance(c, Token) and str(getattr(c,'type','')).upper() == 'IF' for c in non_none)
        if has_if:
            # Find expr_if (first Tree child), IF token, expr_then (second Tree child after IF), ELSE token, expr_else
            tree_children = [c for c in non_none if isinstance(c, Tree)]
            if len(tree_children) >= 3:
                expr_if = expression_to_str(tree_children[0])
                # Find the ELSE branch - could be a nested ternary at position >= 2 or direct tree
                expr_else = expression_to_str(tree_children[-1])
                # Get the middle part (if_expr)
                if len(tree_children) >= 3:
                    return f"{expr_if} if {expression_to_str(tree_children[1])} else {expr_else}"
            # Fallback to default

    # For unary operators: operator immediately followed by expr (no space between them)
    # This covers sign, bit_not, logic_not, and similar unary wrappers
    if len(non_none) >= 2:
        first = non_none[0]
        second = non_none[-1]
        
        # Check if this is a pure unary pattern: single operator token + expression tree
        is_unary = (
            isinstance(first, Token)
            and isinstance(second, Tree)
            and dn in ('sign', 'bit_not', 'logic_not')
        )
        if is_unary:
            op_str = _operator_text(str(getattr(first, 'value', '')))
            expr_str = expression_to_str(second)
            
            # Handle unary plus: keep it for numeric literals, strip for simple names
            # This matches the legacy gdformat behavior exactly.
            if op_str == '+':
                # Check what kind of value follows: number/literal vs identifier
                second_data = getattr(getattr(second, 'data', None), 'value', str(second.data)) \
                    if hasattr(getattr(second, 'data', None), 'value') else str(second.data)
                is_numeric_or_literal = (
                    second_data == 'type_test' and len([c for c in getattr(second, 'children', []) if isinstance(c, Token) and not any(x in str(getattr(c,'type','')).upper() for x in ('NONE',''))]) >= 1
                )
                # If the inner value is a plain number/string/null literal, keep unary +
                if _is_simple_expression(expr_str):
                    inner = [c for c in getattr(second, 'children', []) if isinstance(c, Token)][0] if second.children else None
                    if inner and str(getattr(inner, 'type', '')).upper() in ('NUMBER', 'STRING'):
                        return f"{op_str}{expr_str}"  # Keep + for numeric/string literals
                    return expr_str  # Drop + for variable names etc.
            
            # The style guide prefers `not` (with a space) over `!`.
            if op_str == 'not':
                return f"not {expr_str}"
            
            # Other unary operators (like -, ~)
            return f"{op_str}{expr_str}"

    # For binary/operator chains: operand OP operand OP ...
    parts = []
    for child in non_none:
        if isinstance(child, Token):
            op_str = _operator_text(getattr(child, 'value', str(child.type)))
            parts.append(f" {op_str} ")
        elif isinstance(child, Tree):
            dn = getattr(child.data, 'value', str(child.data)) if hasattr(child.data, 'value') else str(child.data)
            # For single-Token wrapper nodes (comparison_operator, _comp_op), extract the operator
            if len(getattr(child, 'children', [])) == 1:
                only_child = child.children[0]
                if isinstance(only_child, Token):
                    parts.append(f" {_operator_text(getattr(only_child, 'value', str(only_child.type)))} ")
                else:
                    formatted = expression_to_str(child)
                    parts.append(formatted)
            else:
                # Recurse into multi-child nodes
                formatted = expression_to_str(child)
                parts.append(formatted)
    
    # Join and clean up
    result = "".join(parts)
    while "  " in result:
        result = result.replace("  ", " ")
    return result.strip()


def _is_simple_expression(expr_str: str) -> bool:
    """Check if an expression string is 'simple' (no operators, just a value).
    
    Simple expressions include:
    - Bare variable names: x
    - Numbers: 42, 0.5, 0xff
    - String literals: "hello", 'world'
    - Special values: null, true, false
    - Self/owner keywords: self, owner, null (without any operators)
    
    Not simple if:
    - Starts with an operator: +a, -(a), !b
    - Contains binary operators: a + b, x.y
    """
    stripped = expr_str.strip()
    # Simple expressions don't start with operators or contain dots (attributes)
    first_char = stripped[0] if stripped else ''
    
    skip_ops = {'+', '-', '~', '!', '('}
    if first_char in skip_ops:
        return False
    
    # Don't count attribute chains as simple: a.b, self.x
    if '.' in stripped and not stripped.startswith('"') and not stripped.startswith("'"):
        return False
    
    # Don't count calls as simple: func()
    if '(' in stripped:
        return False
    
    return True


def _comma_list_to_str(node: Tree) -> str:
    """Format comma-separated expression list (e.g., elements inside array literal)."""
    from .expression_utils import is_any_comma  # noqa: F821, local import to avoid circular
    elements = [
        standalone_expression_to_str(child)
        for child in getattr(node, 'children', [])
        if child is not None and not is_any_comma(child) and not isinstance(child, Token)
    ]
    return ", ".join(elements)


def _attribute_to_str(node: Tree) -> str:
    """Format an attribute chain like `.foo.bar`.
    
    Grammar: attribute -> PERIOD attribute_name (PERIOD attribute_name)*
    attribute_name is just a NAME token in most cases.
    """
    parts = []
    for c in getattr(node, 'children', []):
        if c is None:
            continue
        if isinstance(c, Token):
            ttype = str(c.type).lower()
            if ttype == 'period':
                parts.append('.')
            elif ttype == 'name':
                parts.append(str(c.value))
            # Skip other tokens (colon etc.)
        else:  # Tree child like attribute_name
            for sub_c in getattr(c, 'children', []):
                if isinstance(sub_c, Token) and str(sub_c.type).lower() == 'name':
                    parts.append(str(sub_c.value))
    return ''.join(parts)


def _pattern_items_to_str(node) -> str:
    """Join the children of an `array_pattern_items`/`dict_pattern_items` with `, `.

    The rest token (`..`) is kept, since it is part of the pattern.
    """
    parts = []
    for child in getattr(node, 'children', []):
        if child is None:
            continue
        if isinstance(child, Token):
            if str(getattr(child, 'type', '')).upper() == 'PERIOD_PERIOD':
                parts.append('..')
            continue
        parts.append(expression_to_str(child))
    return ", ".join(parts)


def _dict_pattern_entry_to_str(node) -> str:
    """A dict-pattern entry: `key` alone (a contains-key pattern) or `key: value`."""
    non_none = [c for c in getattr(node, 'children', []) if c is not None]
    if (len(non_none) == 1 and isinstance(non_none[0], Token)
            and str(getattr(non_none[0], 'type', '')).upper() == 'PERIOD_PERIOD'):
        return '..'
    key = None
    value = None
    for child in non_none:
        if isinstance(child, Token):
            continue
        if key is None:
            key = child
        else:
            value = child
    if key is None:
        return "?"
    if value is None:
        return expression_to_str(key)
    return f"{expression_to_str(key)}: {expression_to_str(value)}"


def _dict_entry_from_children(node: Tree) -> str:
    """Format a single dict entry from the new grammar.
    
    New grammar produces: dict_entry -> (NAME | STRING) EQUAL expression
                          or: dict_entry -> NAME COLON expression  (in dict_entries context)
    The children may include the key, separator token(s), and value expression.
    With maybe_placeholders=True, tokens appear as string placeholders ("." or "=").
    Preserves the original separator (::= or =) from the source.
    """
    from .expression_utils import is_any_comma  # noqa: F821
    non_none = [c for c in getattr(node, 'children', []) if c is not None]
    
    if len(non_none) == 0:
        return "?"
    
    # Find key and separator type
    key_expr = None
    value_idx = -1
    separator = '='
    for i, c in enumerate(non_none):
        # Handle both Token objects AND string placeholders (from maybe_placeholders=True)
        if isinstance(c, Token) and str(c.type).lower() in ('colon', 'equal'):
            value_idx = i + 1
            separator = ':' if str(c.type).lower() == 'colon' else '='
            break
        elif isinstance(c, str):
            # String placeholder: ":" for colon, "=" for equal
            if c in (':', '='):
                value_idx = i + 1
                separator = ':' if c == ':' else '='
                break
        elif not isinstance(c, Token):
            key_expr = standalone_expression_to_str(c)
    
    # Get value (last non-token child that's an expression)
    if key_expr is None and len(non_none) > 0:
        # Key might be a NAME token directly
        for c in non_none:
            ttype = str(getattr(c, 'type', '')).lower() if isinstance(c, Token) else ''
            if ttype == 'name':
                key_expr = str(getattr(c, 'value', '?'))
                break
    
    # One space around the `=` operator; `:` reads as `key: value`.
    sep_text = " = " if separator == '=' else ": "

    # Find value expression after separator
    if value_idx > 0 and value_idx < len(non_none):
        value = non_none[value_idx]
        if not isinstance(value, Token) and hasattr(value, 'data'):
            return f"{key_expr}{sep_text}{standalone_expression_to_str(value)}"

    # Fallback: find last tree child as value
    for c in reversed(non_none):
        if not isinstance(c, Token) and not isinstance(c, str) and hasattr(c, 'data'):
            return f"{key_expr}{sep_text}{standalone_expression_to_str(c)}"

    return key_expr if key_expr else "?"


def _annotation_to_type_str(node: Tree) -> str:
    """Format a type annotation (e.g., Array[int], Dictionary[String, Node]).
    
    Grammar: type -> NAME (PERIOD NAME)* [BRACKET_OPEN type (COMMA [type])* BRACKET_CLOSE]
    Children are typically: NAME tokens and PERIOD/BRACKET_OPEN/BRACKET_CLOSE tokens,
    with optional TYPE nodes inside brackets.
    """
    parts = []
    for c in getattr(node, 'children', []):
        if c is None:
            continue
        if isinstance(c, Token):
            ttype = str(c.type).lower()
            if ttype == 'period':
                parts.append('.')
            elif ttype == 'bracket_open':
                parts.append('[')
            elif ttype == 'bracket_close':
                parts.append(']')
            elif ttype == 'comma':
                parts.append(', ')  # Space after comma for readability
            else:
                # NAME tokens (or other identifiers)
                parts.append(str(c.value))
        elif not isinstance(c, Token):
            # Nested type inside brackets like Array[int]
            parts.append(expression_to_str(c) if hasattr(c, 'data') else str(c))
    return ''.join(parts)


def _tree_data_str(node) -> str:
    from lark import Tree
    if isinstance(node, Tree):
        d = node.data
        return str(d) if not hasattr(d, 'value') else d.value  # type: ignore
    return ""
