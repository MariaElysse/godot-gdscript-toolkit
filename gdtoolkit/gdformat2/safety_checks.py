"""Safety checks for gdformat2.

Three validation layers after formatting:
1. Tree invariant — formatted output re-parses to structurally equivalent AST
2. Formatting stability — double-formatting produces identical output (idempotency)
3. Comment persistence — all comments preserved in formatted output
"""
import difflib

from lark import Token, Tree, Transformer

from .parser import parser
from .style_config import StyleConfig
from .formatter import format_code
from .comments import gather_comments
from .exceptions import (
    TreeInvariantViolation,
    FormattingStabilityViolation,
    CommentPersistenceViolation,
)


# Tree transformer that normalizes AST for comparison by stripping cosmetic differences
class LoosenTreeTransformer(Transformer):
    """Loosens tree structure to compare semantic equivalence rather than exact matches."""

    def par_expr(self, args):
        return args[0] if len(args) > 0 else args

    def primary(self, args):
        """A wrapping parenthesis is transparent for the comparison.

        The guide adds one around a long expression to bring the line under the limit, and
        it removes one that is unnecessary, so the same expression appears with and without
        its wrapper depending on which rule ran. `?primary` inlines the single-child forms,
        so the only `primary` node that reaches the transformer is the `( expression )` form.
        """
        if (
            len(args) == 3
            and isinstance(args[0], Token)
            and isinstance(args[2], Token)
            and str(args[0]) == "("
            and str(args[2]) == ")"
        ):
            return args[1]
        return Tree("primary", args)

    def comment(self, args):
        """Compare comment *text* canonically, since the guide's spacing rule rewrites it.

        `#text` becomes `# text` on formatting, which is a required change rather than a
        What a comment says is checked by `check_comment_persistence`.
        """
        from .comments import normalize_comment

        canonical = []
        for arg in args:
            if isinstance(arg, Token):
                canonical.append(Token(str(arg.type), normalize_comment(str(arg))))
            else:
                canonical.append(arg)
        return Tree("comment", canonical)

    def asless_actual_neg_expr(self, args):
        if isinstance(args[1], Token) and str(args[1].type) == "NUMBER":  # type: ignore
            return Token("NUMBER", f"-{args[1].value}")  # type: ignore
        return Tree("asless_actual_neg_expr", args)

    def asless_comparison(self, args):
        return Tree("comparison", args)

    def asless_and_test(self, args):
        return Tree("and_test", args)

    def asless_or_test(self, args):
        return Tree("or_test", args)

    def asless_bitw_or(self, args):
        return Tree("bitw_or", args)

    def asless_bitw_xor(self, args):
        return Tree("bitw_xor", args)

    def asless_bitw_and(self, args):
        return Tree("bitw_and", args)

    def asless_shift_expr(self, args):
        return Tree("shift_expr", args)

    def asless_type_test(self, args):
        return Tree("type_test", args)

    def asless_content_test(self, args):
        return Tree("content_test", args)

    def asless_test_expr(self, args):
        return Tree("test_expr", args)

    def asless_arith_expr(self, args):
        return Tree("arith_expr", args)

    def asless_mdr_expr(self, args):
        return Tree("mdr_expr", args)

    def asless_pow_expr(self, args):
        return Tree("pow_expr", args)

    def signal_stmt(self, args):
        if len(args) > 1 and (
            not hasattr(args[1], "children") or len(args[1].children) == 0
        ):
            return Tree("signal_stmt", args[:-1])
        return Tree("signal_stmt", args)

    def start(self, args):
        # Filter out annotation and property_body_def nodes (handled separately by parser)
        return Tree(
            "start",
            [
                arg
                for arg in args
                if not isinstance(arg, Tree)
                or str(arg.data).startswith("class_member")  # type: ignore
                or str(arg.data) in ("extends_stmt", "class_name_stmt")  # type: ignore
            ],
        )

    def class_def(self, args):
        return Tree(
            "class_def",
            [
                arg
                for arg in args
                if not isinstance(arg, Tree)
                or str(arg.data).startswith("class_member")  # type: ignore
            ],
        )

    def inner_class(self, args):
        return Tree(
            "inner_class",
            [
                arg for arg in args if not isinstance(arg, Tree)
            ],  # Filter annotations at class level
        )


def _node_name(node) -> str:
    """A tree's rule name as a plain string (`Tree.data` may be a token in lark)."""
    return str(getattr(node, "data", ""))


def _is_structural_end(node) -> bool:
    """`end` is the grammar's statement terminator, a tree in this grammar and a token in
    a few places. It carries no content, so the comparison drops it either way."""
    if isinstance(node, Token):
        return str(node) == "end"
    return isinstance(node, Tree) and _node_name(node) == "end"


def _collapse_single_child(node):
    """Drop non-terminal nodes that have exactly one non-terminal child.

    Such a node records which rung of the grammar's precedence ladder a piece of text
    passed through, not what it means. A parenthesised expression re-enters the expression
    grammar at the top, so `( "..." % i )` gains `assignment`/`cast`/`ternary` wrappers
    that the same expression carries without its parenthesis - and the guide both adds and
    removes those parentheses. Collapsing the wrappers on both sides is what lets the
    comparison see through the guide's own reshaping while still catching a token that was
    dropped, duplicated or reordered, since none of those is a single-child difference.
    """
    if not isinstance(node, Tree):
        return node
    children = [_collapse_single_child(child) for child in node.children]
    if str(node.data) == "suite_line":
        # `suite_line` is a statement plus its content-ending token, so a block body holds
        # the same statements whether or not each one is wrapped. Unwrapping it compares
        # the body, not which wrapper the block form happened to use.
        statements = [
            child
            for child in children
            if isinstance(child, Tree) and not _is_structural_end(child)
        ]
        if len(statements) == 1:
            return statements[0]
    if (
        str(node.data) == "sign"
        and len(children) == 2
        and isinstance(children[0], Token)
        and str(children[0]) == "+"
    ):
        # A unary plus is a no-op the guide drops, so `+x` and `x` are the same value.
        return children[1]
    if len(children) == 1 and isinstance(children[0], Tree):
        return children[0]
    return Tree(node.data, children)


# Logical operators whose spelling the guide fixes: the word form is preferred, so the
# formatter rewrites the symbolic one. Value *and* type are normalized, because
# `lark.Token` compares both, and `BANG_EQUAL` (`!=`) keeps its own type so it is untouched.
_LOGICAL_OPERATOR_WORDS = {
    "BANG": ("NOT", "not"),
    "AMPERSAND_AMPERSAND": ("AND", "and"),
    "PIPE_PIPE": ("OR", "or"),
}


def _canonical_token(token):
    """The spelling the comparison uses for a token the guide re-spells.

    `lark.Token` compares its type and its value, so a normalized token gets both.
    """
    operator = _LOGICAL_OPERATOR_WORDS.get(str(token.type))
    if operator is not None:
        return Token(operator[0], operator[1])
    if str(token.type) == "NUMBER":
        # The guide's digit-grouping rule rewrites `1_000` as `1000`: same number, fewer
        # separators. Dropping the separators compares the number, not its grouping. Hex
        # digits are case-insensitive, and the formatter lowercases them.
        return Token(str(token.type), str(token).replace("_", "").lower())
    if str(token.type) == "STRING":
        # The quote style is the formatter's choice (`'x'` and `"x"` are the same string),
        # and so is which quote it escapes. Compare the decoded value where Python can
        # decode it, and the raw text otherwise (a GDScript-only escape).
        import ast

        try:
            return Token(str(token.type), repr(ast.literal_eval(str(token))))
        except (ValueError, SyntaxError):
            return token
    return token


def _strip_placeholders_and_trailing_comma(node):
    """Strip the two things a parse tree carries that are punctuation rather than content.

    - The `None` a slot leaves behind when it is empty. Both parses come from the same
      grammar, so an empty slot is a fact about the grammar, and anything actually removed
      from a slot shows up as missing tokens instead.
    - A trailing comma at the end of a group - the one the guide adds to a multiline
      collection, an accessor list or an argument list. Only a comma that is *last* in its
      group is dropped, so a comma separating two items still has to match.
    """
    if not isinstance(node, Tree):
        return _canonical_token(node) if isinstance(node, Token) else node
    children = [
        _strip_placeholders_and_trailing_comma(child) for child in node.children
    ]
    children = [
        child
        for child in children
        if child is not None
        and not _is_structural_end(child)
        and not (isinstance(child, Token) and str(child) == ";")
    ]
    while (
        len(children) >= 2
        and isinstance(children[-1], Token)
        and str(children[-1]) == ","
    ):
        children.pop()
    return Tree(node.data, children)


def _normalize_for_comparison(node):
    """Reduce a parse tree to what the comparison is allowed to care about.

    Order matters: an empty slot leaves a `None` behind, which gives its parent a second
    child and would otherwise stop the single-child collapse from seeing through a
    precedence rung. So the placeholders go first, then the collapsing, then the trailing
    commas - which the collapse cannot affect, since it only removes single-child parents.
    """
    return _strip_placeholders_and_trailing_comma(
        _collapse_single_child(_strip_placeholders_and_trailing_comma(node))
    )


def check_tree_invariant(
    original_code: str,
    formatted_code: str,
) -> None:
    """Verify that formatted code re-parses to a structurally equivalent AST.

    Raises TreeInvariantViolation if the ASTs differ semantically.
    """
    # Re-parse both versions of the code (with comments attached automatically)
    original_tree = parser.parse(original_code)
    formatted_tree = parser.parse(formatted_code)

    # Apply loosening transformer to ignore cosmetic differences
    looser = LoosenTreeTransformer()
    loose_original = _normalize_for_comparison(looser.transform(original_tree))
    loose_formatted = _normalize_for_comparison(looser.transform(formatted_tree))

    if loose_original != loose_formatted:
        diff = "\n".join(
            difflib.unified_diff(
                str(loose_original.pretty()).splitlines(),
                str(loose_formatted.pretty()).splitlines(),
            )
        )
        raise TreeInvariantViolation(diff)


def check_formatting_stability(
    formatted_code: str,
    style: StyleConfig,
) -> None:
    """Verify that double-formatting produces identical output (idempotency).

    Raises FormattingStabilityViolation if the output changes on second format.
    """
    # Re-parse and re-format
    re_parsed_tree = parser.parse(formatted_code)
    formatted_again = format_code(re_parsed_tree, style)

    if formatted_code != formatted_again:
        diff = "\n".join(
            difflib.unified_diff(
                formatted_code.splitlines(), formatted_again.splitlines()
            )
        )
        raise FormattingStabilityViolation(diff)


def check_comment_persistence(
    original_tree: Tree,
    formatted_code: str,
) -> None:
    """Verify that all comments from the original tree appear in formatted output.

    Uses tree-based comment comparison (comments as embedded nodes rather than arrays).

    Raises CommentPersistenceViolation if any comment is missing.
    """
    # Get comments from original tree
    original_comments = gather_comments(original_tree)

    # Re-parse formatted code and get its comments
    formatted_tree = parser.parse(formatted_code)
    formatted_comments = gather_comments(formatted_tree)

    for orig_comment in original_comments:
        if not any(orig_comment in fc for fc in formatted_comments):
            raise CommentPersistenceViolation(f"Missing comment: {orig_comment}")


def check_all_safety(
    original_code: str,
    original_tree: Tree,
    formatted_code: str,
    style: StyleConfig,
) -> None:
    """Run all three safety checks at once.

    Raises the first violation encountered, if any.
    """
    check_comment_persistence(original_tree, formatted_code)
    check_tree_invariant(original_code, formatted_code)
    check_formatting_stability(formatted_code, style)
