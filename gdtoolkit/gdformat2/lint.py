"""Lint the GDScript Style Guide rules that formatting cannot enforce.

gdformat2 enforces most of the official style guide by rewriting the code, so the formatter is
where a violation of those rules is fixed. A handful of rules cannot be enforced that way: they
need a judgement no formatter can make, or the only fix is a source change the author has to
choose. They are reported here instead, each with the guide section it comes from and the
smallest action that clears it.

    python -m gdtoolkit.gdformat2.lint <path>... [--disable=<check>,...]
    python -m gdtoolkit.gdformat2.lint --list-checks

The checks, and why the formatter cannot do them:

- `code-order` - moving a declaration changes what the rest of the class sees, and a `var`
  initializer can reference an earlier one, so reordering is not safe to automate.
- `continuation-indent` - the guide asks for two indent levels on a continuation line, and
  gdformat2 writes them, so this reports code that has not been through the formatter. A block
  is judged as a block, and one level is accepted where two would put a line over the limit,
  which is the one case where the two guide rules conflict.
- `unbreakable-line` - a line whose widest token cannot fit the limit no matter how it is
  wrapped. gdformat2 wraps everything it can, so a line it could not shorten is reported with
  the token that is too wide.
- `commented-out-code` - the guide asks to remove disabled code, but whitespace alone cannot
  tell a comment from commented-out code, so it is deliberately not touched by the formatter.
- `redundant-type-hint` and `ambiguous-inferred-type` - which of `:=` and an explicit type to
  write is a typing choice, not a formatting one (`var health := 0` could be int or float).
- `file-name` - the file name is outside the file's contents.
- `signal-past-tense` - the guide asks for the past tense, which no regex can decide for an
  arbitrary verb.
- `byte-order-mark` - a BOM makes the file fail to parse rather than being stripped, so it has
  to be reported before the parser ever sees the file.

Identifier naming (snake_case functions and variables, PascalCase classes, CONSTANT_CASE
constants) is the job of `gdlint`, whose `function-name`, `class-name`, `signal-name` and the
other name checks already cover it from a regular expression.
"""
import os
import re
import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from lark import Token, Tree

from .parser import parser as _parser

GUIDE_BASE_URL = (
    "https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/gdscript_styleguide.html"
)

# The guide's own code order ("Code order"), used both to rank a declaration and to tell the
# reader the order they should have written. Every entry is (key, label).
CODE_ORDER = (
    ("tools", "@tool / @icon / @static_unload"),
    ("class name", "class_name"),
    ("extends", "extends"),
    ("doc comment", "the doc comment (`##`)"),
    ("signals", "signals"),
    ("enums", "enums"),
    ("constants", "constants"),
    ("static variables", "static variables"),
    ("export variables", "@export variables"),
    ("public variables", "regular variables, public first"),
    ("private variables", "private variables"),
    ("onready variables", "@onready variables"),
    ("static functions", "static functions, `_static_init()` first"),
    ("virtual methods", "virtual methods, `_init()`..`_physics_process()` first"),
    ("methods", "remaining methods, public first"),
    ("inner classes", "inner classes"),
)

_ORDER_KEYS = tuple(key for key, _ in CODE_ORDER)
_ORDER_SUGGESTION = "Move it to the {} section. Guide order: {}.".format(
    "{}", " / ".join(label for _, label in CODE_ORDER)
)

# The built-in virtual methods, in the guide's own order, so a method that is one of them is
# ranked before the class's other methods.
_VIRTUAL_METHODS = (
    "_init",
    "_enter_tree",
    "_ready",
    "_process",
    "_physics_process",
)

_NODE_ACCESS = ("get_node", "get_node_or_null", "find_child", "find_children")

_KEYWORDS = frozenset(
    """
    and as assert async await break breakpoint class class_name const continue elif else enum
    extends false for func if in is match namespace not null or pass preload return self signal
    static super true var void while yield
    """.split()
)

DEFAULT_CONFIG = MappingProxyType(
    {
        # check control
        "disable": [],
        # `unbreakable-line`: the same limit the formatter uses, and the same reading of a tab
        # as four columns, so the two agree about which line is too long.
        "max-line-length": 100,
        "tab-characters": 4,
        # `code-order`
        "code-order": list(_ORDER_KEYS),
    }
)

CHECKS = (
    "code-order",
    "continuation-indent",
    "unbreakable-line",
    "commented-out-code",
    "redundant-type-hint",
    "ambiguous-inferred-type",
    "file-name",
    "signal-past-tense",
    "byte-order-mark",
)


@dataclass(frozen=True)
class Problem:
    """One style guide rule that has to be fixed by hand."""

    check: str
    line: int
    column: int
    message: str
    suggestion: str
    section: str = ""

    @property
    def guide_url(self) -> str:
        return f"{GUIDE_BASE_URL}#{self.section}" if self.section else GUIDE_BASE_URL

    def __str__(self) -> str:
        return "{}. {} (see {})".format(self.message, self.suggestion, self.guide_url)

    def format_with(self, path: str) -> str:
        """The report line a tool prints, positioned at the problem's line and column."""
        return f"{path}:{self.line}:{self.column}: {self.check}: {self}"


class _Context:
    """What every check gets: the source, its lines, the tree and the file name."""

    def __init__(self, source: str, tree: Optional[Tree], filename: Optional[str], config):
        self.source = source
        self.lines = source.splitlines()
        self.tree = tree
        self.filename = filename
        self.config = config

    @property
    def max_line_length(self) -> int:
        return self.config["max-line-length"]

    @property
    def tab_characters(self) -> int:
        return self.config["tab-characters"]

    def display_width(self, text: str) -> int:
        return len(text.replace("\t", " " * self.tab_characters))

    def indentation_width(self, line: str) -> int:
        return self.display_width(line[: len(line) - len(line.lstrip())])


def lint_code(
    gdscript_code: str,
    filename: Optional[str] = None,
    config: MappingProxyType = DEFAULT_CONFIG,
) -> List[Problem]:
    """Every guide rule in `gdscript_code` that has to be fixed by hand.

    `filename` is what `file-name` checks; without it that check is skipped. A source that does
    not parse skips the tree-based checks rather than raising, so a broken file still gets its
    text-based problems reported.
    """
    try:
        tree = _parser.parse(gdscript_code)
    except Exception:
        tree = None
    context = _Context(gdscript_code, tree, filename, config)
    problems: List[Problem] = []
    for check in _RUNNERS:
        problems.extend(check(context))
    problems = [
        problem for problem in problems if problem.check not in _disabled(config)
    ]
    problems = _suppress_ignored(problems, context)
    return sorted(problems, key=lambda problem: (problem.line, problem.column, problem.check))


def lint_path(path: str, config: MappingProxyType = DEFAULT_CONFIG) -> List[Problem]:
    """`lint_code` for a file on disk, with its path used for the `file-name` check."""
    with open(path, "r", encoding="utf-8", errors="replace") as file:
        return lint_code(file.read(), filename=path, config=config)


def _disabled(config) -> frozenset:
    disable = config.get("disable", []) if hasattr(config, "get") else []
    return frozenset(disable)


def _suppress_ignored(problems: List[Problem], context: _Context) -> List[Problem]:
    """Drop the problems an inline `gdlint: ignore=<check>` asks to ignore.

    The convention is `gdlint`'s, and `gdformat2:` is accepted as a synonym: the comment
    silences the named checks on its own line and on the following line, which is how a problem
    on the line after a trailing comment is ignored.
    """
    ignored: Dict[str, set] = {}
    pattern = r"#\s*(?:gdlint|gdformat2)\s*:\s*ignore\s*=\s*([^,]+(?:,[^,]+)*)"
    for line_no, line in enumerate(context.lines, start=1):
        for match in re.finditer(pattern, line):
            for check in (name.strip() for name in match.group(1).split(",")):
                ignored.setdefault(check, set()).update((line_no, line_no + 1))
    if not ignored:
        return problems
    return [
        problem
        for problem in problems
        if problem.line not in ignored.get(problem.check, ())
    ]


# ─── helpers over the parse tree ──────────────────────────────────────────────


def _rule(node) -> str:
    """A tree's rule name, lower-cased, for either a tree or a token."""
    if isinstance(node, Token):
        return str(getattr(node, "type", "")).lower()
    data = getattr(node, "data", "")
    return str(getattr(data, "value", data)).lower()


def _children(node) -> List:
    return [child for child in getattr(node, "children", []) if child is not None]


def _subtree(node, rule: str) -> Optional[Tree]:
    for child in _children(node):
        if isinstance(child, Tree) and _rule(child) == rule:
            return child
    return None


def _tokens(node) -> Iterable[Token]:
    for child in _children(node):
        if isinstance(child, Token):
            yield child
        elif isinstance(child, Tree):
            yield from _tokens(child)


def _token(node, type_name: str) -> Optional[Token]:
    for token in _tokens(node):
        if str(getattr(token, "type", "")).upper() == type_name:
            return token
    return None


def _line_of(node) -> int:
    meta = getattr(node, "meta", None)
    return getattr(meta, "line", 1) or 1


def _column_of(node) -> int:
    meta = getattr(node, "meta", None)
    return getattr(meta, "column", 0) or 0


_WRAPPER_RULES = ("assignment", "cast", "ternary", "type_test")


def _unwrap(node):
    """The real expression under the grammar's pass-through wrapper layers.

    The grammar wraps every expression in assignment/cast/ternary/type_test layers whose
    optional slots are empty, so a declaration's value is a chain of them. The unwrapping stops
    at a layer that carries an operator of its own, because that layer is the expression.
    Returns the token the chain ends at when the value is a literal.
    """
    current = node
    for _ in range(12):
        if not isinstance(current, Tree):
            return current
        if _rule(current) not in _WRAPPER_RULES:
            return current
        children = _children(current)
        if len(children) != 1:
            return current
        current = children[0]
    return current


# ─── the checks ───────────────────────────────────────────────────────────────


def _check_code_order(context: _Context) -> List[Problem]:
    """The guide's "Code order", for the file and for every inner class."""
    if context.tree is None:
        return []
    order = tuple(context.config.get("code-definitions-order", _ORDER_KEYS))
    problems = []
    for members in _member_blocks(context.tree):
        problems.extend(_code_order_problems(members, order))
    return problems


def _member_blocks(tree: Tree) -> Iterable[List]:
    """Each block of class members: the file's own, then every inner class's."""
    yield _children(tree)
    for node in tree.iter_subtrees():
        if _rule(node) == "inner_class":
            body = _subtree(node, "class_body") or _subtree(node, "class_block")
            if body is not None:
                yield _children(body)


def _code_order_problems(members: List, order: Sequence[str]) -> List[Problem]:
    problems = []
    seen_rank = -1
    seen_section = order[0] if order else ""
    seen_member = False
    pending_annotations: List[str] = []
    signals_rank = order.index("signals") if "signals" in order else 0
    for member in members:
        if isinstance(member, Token):
            continue
        rule = _rule(member)
        if rule == "annotation":
            pending_annotations.append(_annotation_name(member))
            continue
        if rule == "comment":
            # Only the class's own `##` doc comment has a place in this order, and only while
            # nothing but the head has been written: a `#` comment, or a member's own `##`
            # comment, is not ranked and must not look like one that is out of place.
            if seen_member or not _comment_text(member).startswith("##"):
                continue
            section = "doc comment"
            rank = order.index(section) if section in order else -1
        elif rule in ("class_member", "variable_decl", "constant_decl", "signal_decl",
                      "enum_decl", "function_decl", "inner_class", "extends_stmt",
                      "class_name_stmt"):
            section = _section_of(member, pending_annotations)
            rank = order.index(section) if section in order else -1
            if rank < 0:
                pending_annotations = []
                continue
        else:
            pending_annotations = []
            continue
        pending_annotations = []
        if rank >= signals_rank:
            seen_member = True
        if rank < seen_rank:
            problems.append(
                Problem(
                    check="code-order",
                    line=_line_of(member),
                    column=_column_of(member),
                    message="the guide puts {} before {}".format(section, seen_section),
                    suggestion=_ORDER_SUGGESTION.format(section),
                    section="code-order",
                )
            )
        else:
            seen_rank = rank
            seen_section = section
    return problems


def _comment_text(comment) -> str:
    token = _token(comment, "COMMENT")
    return str(getattr(token, "value", "")) if token is not None else ""


def _annotation_name(annotation: Tree) -> str:
    token = _token(annotation, "ANNOTATION")
    return str(getattr(token, "value", "")).lstrip("@") if token is not None else ""


def _section_of(member, annotations: Sequence[str]) -> str:
    """Which of the guide's code order sections a class member belongs to."""
    rule = _rule(member)
    if rule == "class_member":
        children = _children(member)
        inner = children[0] if children else None
        if inner is None:
            return ""
        return _section_of(inner, annotations)
    if rule in ("extends_stmt", "classname_extends_stmt"):
        return "extends"
    if rule == "class_name_stmt":
        return "class name"
    if rule == "signal_decl":
        return "signals"
    if rule == "enum_decl":
        return "enums"
    if rule == "constant_decl":
        return "constants"
    if rule == "inner_class":
        return "inner classes"
    if rule == "variable_decl":
        return _variable_section(member, annotations)
    if rule == "function_decl":
        return _function_section(member)
    return ""


def _variable_section(variable: Tree, annotations: Sequence[str]) -> str:
    is_static = _token(variable, "STATIC") is not None
    if any(name.startswith("export") for name in annotations):
        return "export variables"
    if "onready" in annotations:
        return "onready variables"
    if is_static:
        return "static variables"
    name = _token(variable, "NAME")
    is_private = name is not None and str(name.value).startswith("_")
    return "private variables" if is_private else "public variables"


def _function_section(function: Tree) -> str:
    is_static = _token(function, "STATIC") is not None
    name = _token(function, "NAME")
    method = str(getattr(name, "value", "")) if name is not None else ""
    if method in _VIRTUAL_METHODS:
        return "virtual methods"
    if method.startswith("_") and method not in _VIRTUAL_METHODS:
        # The guide lists overridden virtual methods before the class's own; an overridden
        # custom method cannot be told from a private one without resolving the base class, so
        # a leading underscore counts as virtual-adjacent and stays before the rest.
        return "virtual methods"
    if is_static:
        return "static functions"
    return "methods"


def _check_continuation_indent(context: _Context) -> List[Problem]:
    """The guide's two indent levels for a continuation line.

    A continuation line starts inside brackets opened on an earlier line, or follows a line
    ending in a backslash. Arrays, dictionaries and enums are the guide's exception and take a
    single level, so only a `(` block or a backslash continuation is checked.

    A `(` block is judged as a block, because its indent is a property of the block: the guide
    asks for two levels, and gdformat2 drops a whole block to one when two would put a line over
    the limit, which is the one case where one level is right. Judging line by line would flag
    every line of such a block, and would miss that the block is the unit the formatter chose.
    """
    problems = []
    blocks, backslashes = _continuation_blocks(context)
    for start, end, opener_indent, opener in blocks:
        if opener != "(" or start >= end:
            continue
        if context.lines[start - 1].rstrip().endswith(":"):
            # The bracket opens a suite rather than a continuation: a lambda written inside
            # parentheses has a body at one level from its own header, and that body is the
            # block's first content line here.
            continue
        content = context.lines[start:end]
        two_levels = opener_indent + 2 * context.tab_characters
        one_level = opener_indent + context.tab_characters
        # The formatter's rule, as the same predicate: one level is right for a block that fits
        # at one level and would not fit at two. A block that is over the limit either way keeps
        # two levels, so its long line is the `unbreakable-line` check's business rather than an
        # indent problem.
        widths = [context.display_width(line) for line in content]
        relaxes = bool(widths) and (
            max(widths) <= context.max_line_length
            < max(widths) + context.tab_characters
        )
        expected = one_level if relaxes else two_levels
        actual = context.indentation_width(content[0])
        if actual < expected:
            problems.append(
                Problem(
                    check="continuation-indent",
                    line=start + 1,
                    column=0,
                    message=(
                        "the guide asks for two indent levels on a continuation line, and this "
                        "block uses {}".format("one" if actual == one_level else "none")
                    ),
                    suggestion=(
                        "indent the block two levels from the line that opens the expression "
                        "(arrays, dictionaries and enums are the exception and keep one)"
                    ),
                    section="format-multiline-statements-for-readability",
                )
            )
    for line_index, opener_indent in backslashes:
        following = context.lines[line_index + 1]
        levels = (context.indentation_width(following) - opener_indent) // context.tab_characters
        if levels >= 2:
            continue
        if (context.display_width(following) + (2 - levels) * context.tab_characters
                > context.max_line_length):
            continue
        problems.append(
            Problem(
                check="continuation-indent",
                line=line_index + 2,
                column=0,
                message=(
                    "the guide asks for two indent levels on a continuation line, and this "
                    "uses {}".format(levels if levels else "none")
                ),
                suggestion=(
                    "indent it two levels from the line that ends in a backslash"
                ),
                section="format-multiline-statements-for-readability",
            )
        )
    return problems


def _continuation_blocks(context: _Context):
    """The `(`/`[`/`{` blocks and the backslash continuations in the source.

    Returns `(blocks, backslashes)`: each block is `(content start index, content end index,
    opener indent, opener character)`, and each backslash continuation is `(line index, the
    indent of the line that ends in a backslash)`. A bracket or a backslash inside a string or a
    comment is not one, which is why the quote state carries across lines: a triple-quoted string
    spans lines, and counting a bracket inside one made every line after it look like a
    continuation (that is what `rstrings` showed).
    """
    blocks = []
    backslashes = []
    stack: List[Tuple[str, int, int]] = []  # (character, opener indent, opener index)
    closers = {")": "(", "]": "[", "}": "{"}
    quote = ""
    for index, line in enumerate(context.lines):
        indent = context.indentation_width(line)
        position = 0
        while position < len(line):
            character = line[position]
            if quote:
                if character == "\\":
                    position += 2
                    continue
                if quote in ('"""', "'''") and line[position: position + 3] == quote:
                    quote = ""
                    position += 3
                    continue
                if quote in ('"', "'") and character == quote:
                    quote = ""
            elif character in "\"'":
                if line[position: position + 3] in ('"""', "'''"):
                    quote = line[position: position + 3]
                    position += 3
                    continue
                quote = character
            elif character == "#":
                break
            elif character in "([{":
                stack.append((character, indent, index))
            elif character in closers and stack and stack[-1][0] == closers[character]:
                opener, opener_indent, opener_index = stack.pop()
                blocks.append((opener_index + 1, index, opener_indent, opener))
            position += 1
        if not quote and line.rstrip().endswith("\\") and index + 1 < len(context.lines):
            backslashes.append((index, indent))
    return blocks, backslashes


def _check_unbreakable_line(context: _Context) -> List[Problem]:
    """A line over the limit that a reflow cannot shorten, with the run that fills it.

    gdformat2 wraps everything it can, so what is left over the limit is a line whose widest
    unbreakable run of tokens fills it: a name, a string, a comment, or a run of operators
    written without spaces (`---1`). Nothing can be moved off that run, so the only fix is a
    shorter name, a shorter string or a wrapped comment.

    A run that fits the limit with room to spare is left alone, because the formatter can
    usually put it on a line of its own - that is a formatting problem, not one of these.
    """
    if context.tree is None:
        return []
    problems = []
    for line_number, (run_width, kind, value) in _widest_runs(context).items():
        line = context.lines[line_number - 1]
        available = context.max_line_length - context.indentation_width(line)
        if run_width < available - _DECORATION_MARGIN:
            # The run leaves room for the `var ` or `= (` beside it, so the formatter can
            # still move things off this line: that is a formatting problem, not one of these.
            continue
        if kind == "COMMENT":
            what = "the comment is {} columns".format(run_width)
            action = "wrap the comment across lines"
        elif kind == "STRING" or '"' in value or "'" in value:
            what = "the string `{}` is {} columns".format(_ellipsis(value), run_width)
            action = "shorten the string, or split it and join the parts"
        elif kind == "NUMBER":
            what = "the number `{}` is {} columns".format(_ellipsis(value), run_width)
            action = "shorten the number"
        else:
            what = "the run `{}` is {} columns".format(_ellipsis(value), run_width)
            action = "shorten the identifier name"
        problems.append(
            Problem(
                check="unbreakable-line",
                line=line_number,
                column=0,
                message=(
                    "the line is {} columns and {} on its own, so no reflow fits it under "
                    "{}".format(context.display_width(line), what, context.max_line_length)
                ),
                suggestion=action,
                section="line-length",
            )
        )
    return problems


# How much room to leave beside the widest run before a reflow is considered unable to help:
# with less than this left over, the `var ` or `= (` that has to sit beside the run no longer
# fits, so a shorter run is the only fix. Eight columns is the widest of those decorations.
_DECORATION_MARGIN = 8


def _widest_runs(context: _Context) -> Dict[int, Tuple[int, str, str]]:
    """Per line, the widest run of tokens with no whitespace between them.

    A run is measured from where it starts, in display columns, so a tab counts as the width
    the formatter gives it. Only over-long lines are returned.
    """
    tokens = [token for token in _tokens(context.tree) if getattr(token, "line", None)]
    tokens.sort(key=lambda token: (token.line, token.column))
    on_line: List[Token] = []
    widest: Dict[int, Tuple[int, str, str]] = {}
    for token in tokens + [None]:  # type: ignore[list-item]
        if on_line and (token is None or token.line != on_line[0].line):
            width, kind, value = _run_of(context, on_line)
            if width > widest.get(on_line[0].line, (0, "", ""))[0]:
                widest[on_line[0].line] = (width, kind, value)
            on_line = []
        if token is not None:
            on_line.append(token)
    return {
        line: entry
        for line, entry in widest.items()
        if context.display_width(context.lines[line - 1]) > context.max_line_length
    }


def _run_of(context: _Context, tokens: Sequence[Token]) -> Tuple[int, str, str]:
    """The widest whitespace-free run among `tokens`, as (width, kind, text)."""
    best: List[Token] = []
    best_width = 0
    run: List[Token] = []
    for token in list(tokens) + [None]:  # type: ignore[list-item]
        if run and (token is None or token.column != run[-1].column + len(str(run[-1].value))):
            width = _run_width(run)
            if width > best_width:
                best, best_width = run, width
            run = []
        if token is not None:
            run.append(token)
    return best_width, _run_kind(best), "".join(str(token.value) for token in best)


def _run_width(tokens: Sequence[Token]) -> int:
    """How many columns a run of tokens occupies, none of which can move off it."""
    return sum(len(str(token.value)) for token in tokens)


def _run_kind(tokens: Sequence[Token]) -> str:
    if len(tokens) == 1:
        return str(getattr(tokens[0], "type", "")).upper()
    return ""


def _ellipsis(value: str, limit: int = 24) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _check_commented_out_code(context: _Context) -> List[Problem]:
    """A comment that reads as GDScript rather than as prose.

    The guide asks to remove disabled code; the formatter leaves every comment alone because
    whitespace cannot tell `# This is a comment.` from `# print(...)`. A comment that parses as
    a statement is the case that can be told apart.
    """
    problems = []
    for line_number, line in enumerate(context.lines, start=1):
        stripped = line.strip()
        if not stripped.startswith("#") or stripped.startswith("#!"):
            continue
        text = stripped.lstrip("#").strip()
        if not _looks_like_code(text):
            continue
        problems.append(
            Problem(
                check="commented-out-code",
                line=line_number,
                column=context.indentation_width(line),
                message="this comment reads as disabled code, not as a comment",
                suggestion="delete it, or say what it is there for in prose",
                section="comment-spacing",
            )
        )
    return problems


def _looks_like_code(text: str) -> bool:
    """Whether a comment's text is a GDScript statement rather than a sentence."""
    if not text or text.startswith(("http://", "https://", "TODO", "FIXME", "NOTE", "XXX")):
        return False
    words = text.split()
    if len(words) == 1 and not any(character in text for character in "()[]{}="):
        return False
    if not _has_code_shape(text):
        return False
    return not _parses_as_a_comment(text)


def _has_code_shape(text: str) -> bool:
    first_word = re.match(r"[A-Za-z_]\w*", text)
    if first_word is not None and first_word.group(0) in _KEYWORDS:
        return True
    return any(marker in text for marker in ("(", "=", "[", "{", '"', "->", ":="))


def _parses_as_a_comment(text: str) -> bool:
    """Whether the text parses as English rather than as GDScript.

    A sentence usually does not parse, so a parse failure means "this is code"; a sentence that
    happens to parse (`# (see)` ) is accepted as prose.
    """
    for attempt in (text, "func _f():\n\t{}".format(text)):
        try:
            _parser.parse(attempt)
        except Exception:
            continue
        return False
    return True


def _check_redundant_type_hint(context: _Context) -> List[Problem]:
    """`var direction: Vector3 = Vector3(1, 2, 3)`, where `:=` says the same thing."""
    problems = []
    for declaration in _declarations(context):
        type_node = _subtree(declaration, "type")
        if type_node is None:
            continue
        declared = _type_text(type_node)
        value = _value_of(declaration)
        if value is None or not declared:
            continue
        called = _called_name(value)
        if called is None or called != declared:
            continue
        problems.append(
            Problem(
                check="redundant-type-hint",
                line=_line_of(declaration),
                column=_column_of(declaration),
                message=(
                    "the type `{}` repeats what `{}(...)` already says".format(declared, called)
                ),
                suggestion="write `:=` instead of the explicit type",
                section="inferred-types",
            )
        )
    return problems


def _check_ambiguous_inferred_type(context: _Context) -> List[Problem]:
    """`:=` where the inferred type is not the one the author meant."""
    problems = []
    for declaration in _declarations(context):
        if _subtree(declaration, "type") is not None:
            continue
        if _token(declaration, "COLON") is None:
            # `var x = 1` states no type at all, which is a different (and older) style; only
            # `:=` claims to infer one.
            continue
        value = _value_of(declaration)
        if value is None:
            continue
        expression = _unwrap(value)
        if isinstance(expression, Token) and str(expression.type).upper() == "NUMBER":
            reason = "a number literal is int or float, and only the type says which"
        elif _is_node_access(expression):
            reason = "the compiler infers only `Node` from a node path"
        else:
            continue
        problems.append(
            Problem(
                check="ambiguous-inferred-type",
                line=_line_of(declaration),
                column=_column_of(declaration),
                message="`:=` infers a type that may not be the intended one: {}".format(reason),
                suggestion="write the type explicitly, or initialize with a typed value",
                section="inferred-types",
            )
        )
    return problems


def _declarations(context: _Context) -> Iterable[Tree]:
    if context.tree is None:
        return
    for node in context.tree.iter_subtrees():
        if _rule(node) in ("variable_decl", "local_variable", "constant_decl"):
            yield node


def _type_text(type_node: Tree) -> str:
    return "".join(str(token.value) for token in _tokens(type_node))


def _value_of(declaration: Tree) -> Optional[Tree]:
    """The declaration's initializer, skipping the name and any type."""
    tokens_seen = 0
    for child in _children(declaration):
        if isinstance(child, Token):
            tokens_seen += 1
            continue
        if _rule(child) in ("type", "parameters", "suite"):
            continue
        return child
    return None


def _base_name(node) -> str:
    """The name a `postfix` chain starts from, whether the base is a token or a subtree."""
    if isinstance(node, Token):
        if str(getattr(node, "type", "")).upper() != "NAME":
            return ""
        return str(getattr(node, "value", ""))
    token = _token(node, "NAME")
    return str(token.value) if token is not None else ""


def _called_name(value) -> Optional[str]:
    """The name of the constructor a value calls, e.g. `Vector3` in `Vector3(1, 2, 3)`."""
    expression = _unwrap(value)
    if not isinstance(expression, Tree) or _rule(expression) != "postfix":
        return None
    children = _children(expression)
    if len(children) < 2 or not all(
        isinstance(child, Tree) and _rule(child) == "call_suffix" for child in children[1:]
    ):
        return None
    return _base_name(children[0]) or None


def _is_node_access(expression) -> bool:
    rule = _rule(expression)
    if rule in ("get_node_expr", "node_path", "unique_node_path"):
        return True
    if rule == "postfix":
        children = _children(expression)
        if children and _base_name(children[0]) in _NODE_ACCESS:
            return True
    return False


def _check_file_name(context: _Context) -> List[Problem]:
    """The guide's snake_case file names, and a `class_name` that matches."""
    if not context.filename:
        return []

    stem = os.path.splitext(os.path.basename(context.filename))[0]
    suggestions = []
    if not re.fullmatch(r"[a-z][a-z0-9]*(_[a-z0-9]+)*", stem):
        suggestions.append("name the file in snake_case, e.g. `yaml_parser.gd`")
    class_name = _class_name_of(context.tree)
    if class_name:
        expected = _snake_case(class_name)
        if expected != stem:
            suggestions.append(
                "rename it to `{}.gd` to match `class_name {}`".format(expected, class_name)
            )
    if not suggestions:
        return []
    return [
        Problem(
            check="file-name",
            line=1,
            column=0,
            message="the file name `{}.gd` does not follow the guide".format(stem),
            suggestion="; ".join(dict.fromkeys(suggestions)),
            section="file-names",
        )
    ]


def _class_name_of(tree: Optional[Tree]) -> str:
    if tree is None:
        return ""
    for node in tree.iter_subtrees():
        if _rule(node) == "class_name_stmt":
            token = _token(node, "NAME")
            return str(token.value) if token is not None else ""
    return ""


def _snake_case(name: str) -> str:
    """`YAMLParser` -> `yaml_parser`, the conversion the guide's file names use.

    A boundary goes between a lower-case (or digit) letter and an upper-case one, and before the
    last capital of an acronym followed by a word (`YAML|Parser`).
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name)
    return spaced.lower()


def _check_signal_past_tense(context: _Context) -> List[Problem]:
    """The guide's "use the past tense to name signals".

    Only the endings that cannot be a past tense are reported: `_ing` always, and a plural
    `s`/`es` on a name of more than one word, which keeps a noun like `status` out of it.
    """
    if context.tree is None:
        return []
    problems = []
    for node in context.tree.iter_subtrees():
        if _rule(node) != "signal_decl":
            continue
        token = _token(node, "NAME")
        name = str(getattr(token, "value", "")) if token is not None else ""
        last_word = name.rsplit("_", 1)[-1]
        if last_word.endswith("ing") or ("_" in name and last_word.endswith(("s", "es"))):
            problems.append(
                Problem(
                    check="signal-past-tense",
                    line=_line_of(node),
                    column=_column_of(node),
                    message="`{}` is not in the past tense".format(name),
                    suggestion=(
                        "name the signal for what happened, e.g. `{}`".format(
                            name.rsplit("_", 1)[0] + "_changed" if "_" in name else "changed"
                        )
                    ),
                    section="signals",
                )
            )
    return problems


def _check_byte_order_mark(context: _Context) -> List[Problem]:
    """The guide's "UTF-8 without a byte order mark"."""
    if not context.source.startswith("\ufeff"):
        return []
    return [
        Problem(
            check="byte-order-mark",
            line=1,
            column=0,
            message=(
                "the file starts with a UTF-8 byte order mark, which makes it fail to parse "
                "rather than being ignored"
            ),
            suggestion="save the file as UTF-8 without a byte order mark",
            section="encoding-and-special-characters",
        )
    ]


_RUNNERS = (
    _check_code_order,
    _check_continuation_indent,
    _check_unbreakable_line,
    _check_commented_out_code,
    _check_redundant_type_hint,
    _check_ambiguous_inferred_type,
    _check_file_name,
    _check_signal_past_tense,
    _check_byte_order_mark,
)


# ─── command line ─────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    """`python -m gdtoolkit.gdformat2.lint <path>...`"""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--list-checks" in arguments:
        for check in CHECKS:
            print(check)
        return 0
    disable: List[str] = []
    for argument in list(arguments):
        if argument.startswith("--disable="):
            disable.extend(argument.split("=", 1)[1].split(","))
            arguments.remove(argument)
    if not arguments:
        print(__doc__.strip())
        return 1
    config = MappingProxyType({**DEFAULT_CONFIG, "disable": disable})
    paths: List[str] = []
    for argument in arguments:
        if os.path.isdir(argument):
            for root, _directories, filenames in os.walk(argument):
                paths.extend(
                    os.path.join(root, name) for name in sorted(filenames)
                    if name.endswith(".gd")
                )
        else:
            paths.append(argument)
    found = 0
    for path in paths:
        try:
            problems = lint_path(path, config=config)
        except OSError as error:
            print(f"{path}: {error}", file=sys.stderr)
            continue
        for problem in problems:
            print(problem.format_with(path))
            found += 1
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
