"""Golden tests for gdformat2 using input/output pair files.

These tests run gdformat2 (AST-based formatter) against the same golden test
input-output pairs used by the original formatter, verifying byte-identical output.
Tests that fail to parse with the new grammar are skipped gracefully.
"""
import difflib
import os
import re
from typing import Dict, Set

import pytest

from gdtoolkit.gdformat2.parser import parser as p
from gdtoolkit.gdformat2.formatter import format_code
from gdtoolkit.gdformat2.style_config import StyleConfig
from .common_gdformat2 import format_and_compare, MAX_LINE_LENGTH

DATA_DIR = "input-output-pairs"

# Fixture pairs excluded from the run entirely.
# `bug_326_multistatement_lambda_corner_case` is the only one left and it is not a formatter
# gap: its golden is not valid GDScript (Godot rejects it with "Unindent doesn't match the
# previous indentation level" while the fixture's input parses), so the valid inline-chain
# layout gdformat2 emits cannot match it. The legacy suite excludes it too.
GDFORMAT2_EXCEPTIONS: Set[str] = {
    "bug_326_multistatement_lambda_corner_case",
}

# Golden cases where gdformat2 intentionally diverges from the legacy formatter because
# the official GDScript Style Guide
# (https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/gdscript_styleguide.html)
# requires different output. Maps the case name to the reason for the divergence.
# `test_style_guide_divergence` below still checks that the difference is limited to the
# documented normalisation, so a real regression cannot hide here.
GDFORMAT2_STYLE_GUIDE_DIVERGENCES: Dict[str, str] = {
    # Legacy omits the trailing comma on a multiline list the author did not write one
    # for; the style guide requires it. (Blank lines now match - gdformat2 preserves the
    # author's blank lines, squeezed to one.)
    # Same: the golden omits the trailing comma on a multiline enum the author did not
    # write one for.
    "long_dict_expressions": "multiline trailing comma",
    # The golden omits the trailing comma on the joined continuation line the author
    # did not write one for; the style guide requires it.
    "long_array_expressions": "multiline trailing comma",
    "assignment_expressions": "multiline trailing comma",
    # The guide's documentation comments "must immediately precede" the member they
    # document, and its own example puts the class doc comment right after `extends` with
    # no blank before the next member. A doc run is joined to the member it heads; a blank
    # between the class doc block and a member's own doc block stays. Only a blank line
    # moves, which is what the canon absorbs.
    "class_doc_comment_joins_first_member": "doc comment joined to the member it documents",
    # The style guide asks for a space after `{` and before `}` in single-line
    # dictionaries and a trailing comma in multiline ones; the legacy goldens omit both.
    "short_dict_expressions": "single-line dict brace spacing",
    "typed_dicts": "single-line dict brace spacing; multiline trailing comma",
    "utf_8_characters": "single-line dict brace spacing",
    # Same dict brace spacing; the `(1)` pattern's parentheses are kept now, which is the
    # only other difference.
    "simple_match_statements": "single-line dict brace spacing",
    # The golden writes the comment of an empty collection at the collection's own indent
    # (its text is `# TODO: fix indent`, legacy's own note); gdformat2 indents it as an
    # element, which the canon's comment-column rule accepts.
    "bug_136_empty_comma_separated_list": "comment indentation inside an empty collection",
    # The style guide prefers `and`/`or`/`not` over `&&`/`||`/`!`; the goldens keep the
    # C-style operators.
    "short_boolean_test_expressions": "boolean operators converted to and/or/not",
    # The style guide lower-cases hexadecimal digits; the golden keeps the author's case.
    "simple_atom_expressions": "hexadecimal digits lower-cased",
}


# Fixtures where a style-guide rule that the legacy formatter does not implement makes
# gdformat2 emit something other than the golden, so the golden cannot match byte for byte and
# the difference is more than the normalisations `_style_guide_canon` absorbs. They are checked
# by `test_guide_divergence`, which requires the same program once every change the guide's
# formatting rules may make is collapsed away.
GDFORMAT2_GUIDE_DIVERGENCES: Dict[str, str] = {
    # The goldens write a space after the dot of a wrapped chain segment (`. append(`), which
    # is legacy's shape; gdformat2 writes `.append(`, since the guide's whitespace rule forbids
    # the extra space and its own example writes `position.x = 5`.
    "bug_160_multiline_call_chaining": "no space after the dot in a wrapped chain",
    "long_atom_expressions": (
        'line length: the golden keeps a long `^"..."` node path on one 107-column line'
    ),
    "long_boolean_test_expressions": (
        "line length: the golden keeps a twenty-two-`not` chain on one 104-column line; it also "
        "carries the boolean-operator conversion"
    ),
    "comment_at_the_end_of_file_237": (
        "comment spacing: the guide asks for a space after the marker, so `##\\ttext` becomes "
        "`## \\ttext`"
    ),
    # "Write enums with each item on its own line", whose own Bad example is a single-line
    # enum, so every enum with entries is exploded (and takes the style guide's trailing
    # comma). The goldens keep theirs inline.
    "complex_trailing_comma_scenarios": "enums one item per line",
    "enum_comments": "enums one item per line",
    "enum_operations": "enums one item per line",
    "parentheses_expressions_preservation": "enums one item per line",
    "regular_enum": "enums one item per line",
    "type_hints": "enums one item per line",
    "regular_long_enum": "enums one item per line; multiline trailing comma",
    # "For inner classes, use single-line declarations", whose own example is
    # `@abstract class MyNode extends Node:`. gdformat2 keeps a base declared in the header on
    # the declaration line; the goldens split it into a body statement (`class D:` then
    # `\textends "res://dummy.gd".X`).
    "bug_334_multiline_lambda": "inner-class extends kept on the declaration line",
    "complex_extends_statements": "inner-class extends kept on the declaration line",
    "test_corner_cases": "inner-class extends kept on the declaration line",
    # The guide's separator rule: a large literal's digits are grouped (decimal in threes,
    # hexadecimal and binary in fours) and a literal whose grouping would need fewer than two
    # separators keeps none. The golden preserves the author's separators verbatim.
    "underscored_numeric_literals": "numeric separators grouped or dropped by magnitude",
    # "Keep individual lines of code under 100 characters": the golden puts a merged inline
    # comment after an already-long argument (122 and 115 columns), where gdformat2 moves the
    # comment to a line of its own, because the comment alone is what pushes the line over.
    "multiline_annotations_w_comments": "line length: an overflowing annotation comment moves off the line",
    # Line length again, on a match pattern: the golden joins a call pattern's arguments on
    # one 179-column line, where gdformat2 breaks them, laying an argument that is itself
    # wider than a line out in place. The fixture also carries the multiline trailing comma
    # the canon map used to absorb, which `_guide_collapse` absorbs through the canon.
    "multiline_match_statements": "line length: a call pattern's arguments break, and a multiline trailing comma",
    # A dot chain whose base is itself wider than a line: the golden keeps a 112-column first
    # line because a lambda in the chain exempts it from the fit test, where gdformat2 breaks
    # the chain per segment. The lambda only exempts its own lines.
    "bug_326_multistatement_lambda_corner_case_workaround": "line length: a dot chain with a too-wide base breaks per segment",
    "addition_n_subtraction_expressions": "continuation lines indented two levels",
    # A multiline collection packs its items on one continuation line while they fit,
    # closing with the trailing comma the guide asks of a multiline list. The golden leaves
    # them one per line because the author wrote a trailing comma, which is a shape, not a
    # rule: the comma forces multiline, and a packed list is multiline. Packing regardless of
    # the comma is also what makes formatting stable - the first run adds the comma, and the
    # second run previously exploded what the first had packed.
    # A call *statement* whose arguments do not fit: it wraps, and its arguments take the
    # guide's two indent levels rather than the golden's one. Before `c5e4990` it was not
    # wrapped at all, whatever its width.
    "long_call_statements": "continuation lines indented two levels",
    "force_multiline_array": "a multiline collection packs its items while they fit",
    "force_multiline_dict": "a multiline collection packs its items while they fit",
    "simple_function_annotations": "a multiline collection packs its items while they fit",
    "complex_unistatement_lambdas": "a multiline collection packs its items while they fit; "
    "single-line dict brace spacing",
    "long_multistatement_lambdas": "continuation lines indented two levels; "
    "multiline trailing comma",
    "bug_156_trailing_comma_corner_case": "continuation lines indented two levels",
    "long_arithmetic_expression": "continuation lines indented two levels",
    "multiplication_division_remainder_expressions": "continuation lines indented two levels",
    "negation_n_bitwise_not_expressions": "continuation lines indented two levels",
    "or_xor_and_shift_bitwise_expressions": "continuation lines indented two levels",
    "power_expressions": "continuation lines indented two levels",
    "type_cast_expressions": "continuation lines indented two levels",
    "type_test_expressions": "continuation lines indented two levels",
    # The guide asks for two indent levels on a continuation line, and gdformat2 writes them
    # now, so every wrapped call, chain, ternary, preload, annotation argument or parameter
    # list sits one level deeper than the golden, which keeps legacy's single level. Arrays,
    # dictionaries, enums and subscripts keep one level in both, because the guide names them
    # as the exception to that rule.
    "array_n_dict_expression_comments": "continuation lines indented two levels",
    "attribute_expressions": "continuation lines indented two levels",
    "await_expressions": "continuation lines indented two levels",
    "bug_148_complex_func_header": "continuation lines indented two levels",
    "bug_195_dot_chain_corner_case": "continuation lines indented two levels",
    "bug_205_static_func_corner_case": "continuation lines indented two levels",
    "bug_339_multiline_lambda": "continuation lines indented two levels",
    "call_expressions": "continuation lines indented two levels",
    "complex_multistatement_lambdas": "continuation lines indented two levels",
    "complex_signal_statements": "continuation lines indented two levels",
    "enhanced_operator_chain": "continuation lines indented two levels",
    "inline_lambdas_w_comments": "continuation lines indented two levels",
    "long_attribute_expression_corner_case": "continuation lines indented two levels",
    "multiline_annotations": "continuation lines indented two levels",
    "multiline_content_tests": "continuation lines indented two levels",
    "multiline_preload_workaround": "continuation lines indented two levels",
    "simple_function_definitions": "continuation lines indented two levels",
    "standalone_comments_in_expressions": "continuation lines indented two levels",
    "type_cast_corner_case_expressions": "continuation lines indented two levels",
    "variadic_functions": "continuation lines indented two levels",
    "complex_dot_chains": "continuation lines indented two levels; single-line dict brace spacing",
    "complex_function_definitions": "continuation lines indented two levels; multiline trailing comma",
    "if_corner_case": "continuation lines indented two levels; boolean operators converted to and/or/not",
    "long_unistatement_lambdas": "continuation lines indented two levels; multiline trailing comma",
    "parentheses_expressions_removal": "continuation lines indented two levels; single-line dict brace spacing; multiline trailing comma",
}


def pytest_generate_tests(metafunc):
    this_directory = os.path.dirname(os.path.abspath(__file__))
    if (
        "test_name" in metafunc.fixturenames
        and metafunc.function.__name__ == "test_input_output_pair"
    ):
        tests_dir = os.path.join(this_directory, DATA_DIR)

        # Get all available test names (excluding known exceptions)
        all_test_names = {
            f.split(".")[0]
            for f in os.listdir(tests_dir)
            if f.endswith(".in.gd")
            and f.split(".")[0] not in GDFORMAT2_EXCEPTIONS
            and f.split(".")[0] not in GDFORMAT2_STYLE_GUIDE_DIVERGENCES
            and f.split(".")[0] not in GDFORMAT2_GUIDE_DIVERGENCES
        }

        metafunc.parametrize(
            "test_name",
            sorted(all_test_names),
            ids=lambda name: f"gdformat2_{name}",
        )


def test_input_output_pair(test_name):
    """Test that gdformat2 produces expected output for one input-output pair."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(this_dir, DATA_DIR, f"{test_name}.in.gd")
    out_path = os.path.join(this_dir, DATA_DIR, f"{test_name}.out.gd")

    with open(in_path, encoding="utf-8") as f:
        input_code = f.read()
    with open(out_path, encoding="utf-8") as f:
        expected_output = f.read()

    format_and_compare(input_code, expected_output)


def _style_guide_canon(text: str):
    """Canonical form for comparing a golden with gdformat2 output while ignoring the
    style-guide normalisations gdformat2 intentionally applies: blank lines (not
    reproduced), spaces just inside single-line dict braces, the trailing comma of a
    multiline list, and C-style vs plain-English boolean operators. Line structure is
    preserved so alignment stays meaningful."""
    lines = []
    for raw in text.splitlines():
        raw = raw.replace("&&", "and").replace("||", "or")
        raw = re.sub(r"!(?!=)", "not ", raw)
        # Lower-case hexadecimal digits (leading and trailing float zeros are already
        # identical in both spellings, so they need no normalisation here).
        raw = re.sub(r"0[xX][0-9a-fA-F_]+", lambda m: m.group(0).lower(), raw)
        line = re.sub(r"\{\s+", "{", raw)
        line = re.sub(r"\s+\}", "}", line)
        # Strip a comma immediately before a closing bracket/brace on the same line
        # (e.g. a single-line list `[1, 2, 3,]`).
        line = re.sub(r",(\s*)([\]\}])", r"\1\2", line)
        # A comment's own column carries no code semantics, and one golden records
        # legacy's column for a comment inside an empty collection (bug_136). Compare a
        # comment-only line by its text; every other line keeps its indentation.
        stripped = line.strip()
        if stripped.startswith("#"):
            line = stripped
        lines.append(line.rstrip())

    # Strip the trailing comma of a multiline list: the comma at the end of the last
    # element line, where the next non-blank line closes the list.
    for i, cur in enumerate(lines):
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines) and lines[j].strip()[:1] in ("]", "}") and cur.endswith(","):
            lines[i] = cur[:-1]

    return [line for line in lines if line.strip()]


@pytest.mark.parametrize(
    "test_name",
    sorted(GDFORMAT2_STYLE_GUIDE_DIVERGENCES),
    ids=lambda n: f"DIVERGENCE_{n}",
)
def test_style_guide_divergence(test_name):
    """Document intentional divergences from the legacy goldens.

    The difference must be limited to the documented style-guide normalisation (blank
    lines and/or single-line dict brace spacing). If it grows to include other content,
    or if it disappears, this test fails: a real bug cannot hide here, and a fixture that
    starts matching should be removed from GDFORMAT2_STYLE_GUIDE_DIVERGENCES.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(this_dir, DATA_DIR, f"{test_name}.in.gd"), encoding="utf-8"
    ) as f:
        input_code = f.read()
    with open(
        os.path.join(this_dir, DATA_DIR, f"{test_name}.out.gd"), encoding="utf-8"
    ) as f:
        expected_output = f.read()

    formatted = format_code(
        p.parse(input_code), StyleConfig(line_length=MAX_LINE_LENGTH)
    )

    reason = GDFORMAT2_STYLE_GUIDE_DIVERGENCES[test_name]
    assert formatted != expected_output, (
        f"{test_name} now matches the legacy golden; remove it from "
        f"GDFORMAT2_STYLE_GUIDE_DIVERGENCES ({reason})."
    )
    assert _style_guide_canon(formatted) == _style_guide_canon(expected_output), (
        f"{test_name} diverges by more than the documented style-guide change "
        f"({reason}):\n"
        + "\n".join(
            difflib.unified_diff(
                _style_guide_canon(expected_output),
                _style_guide_canon(formatted),
                "expected(canon)",
                "formatted(canon)",
                lineterm="",
            )
        )
    )


def _display_width(text: str) -> int:
    """Display width of a line, counting a tab as four columns."""
    return len(text.replace("\t", "    "))


_COMMENT_TEXT = re.compile(r"#[^\n]*")

# An `extends` target: a quoted path (which may itself contain a colon, as in
# `"res://dummy.gd".X`) or a plain name, optionally followed by dotted names. Matching it
# as one unit keeps the clause removable even though it is not whitespace-delimited.
_EXTENDS_TARGET = r'(?:"[^"]*"|\'[^\']*\'|[A-Za-z_]\w*)(?:\.(?:[A-Za-z_]\w*))*'

# A numeric literal, so the separators inside one can be dropped wherever they sit. They are
# formatting, not code: the guide's separator rule groups a large literal's digits, so a
# golden's `0xff_99_00` and gdformat2's `0xff9900` are the same number. The `\b` matters -
# without it the digit run inside an identifier (`x2_000`) would be rewritten too.
_NUMERIC_LITERAL = re.compile(
    r"\b0[xX][0-9a-fA-F_]+|\b0[bB][01_]+|\b\d[\d_]*(?:\.[\d_]*)?(?:[eE][-+]?[\d_]+)?"
)


def _guide_code(text: str) -> str:
    """The text with every comment removed, so where a comment sits cannot matter."""
    return _COMMENT_TEXT.sub("", text)


def _guide_comments(text: str) -> list:
    """Every comment's text with runs of whitespace collapsed, sorted.

    The guide may move a comment (onto its own line when it would make a line too long, or
    keep it beside the item it was written next to once that item is exploded), so position is
    not part of the comparison - but the text and the number of comments are.
    """
    fragments = []
    for comment in _COMMENT_TEXT.findall(text):
        # A golden that wrote two comments on one line leaves them as one comment textually, so
        # compare the pieces either side of each `#` rather than whole comments.
        for part in comment.split("#"):
            part = re.sub(r"\s+", " ", part).strip()
            if part:
                fragments.append(part)
    return sorted(fragments)


def _guide_extends(text: str) -> list:
    """Every `extends` target in the text, sorted.

    `_guide_collapse` drops the whole clause, because the guide moves it between the class
    header and the class body, so the targets are compared separately: a changed base class
    still has to fail.
    """
    flat = re.sub(r"\s+", " ", text)
    return sorted(re.findall(r"\bextends\s+(" + _EXTENDS_TARGET + ")", flat))


def _guide_collapse(text: str) -> str:
    """Remove everything the guide's formatting rules may change, so two renderings of the
    same code compare equal.

    Whitespace, grouping parentheses, underscores inside numeric literals and an `extends`
    clause are formatting, not code: a guide divergence must be invisible here. The clause is
    dropped whole rather than just its target, because the guide moves it between the class
    header and the class body, and `_guide_extends` checks the targets separately so a changed
    base class is still a failure. The canon is applied first, so its normalisations (trailing
    comma, single-line dict brace spacing, boolean spelling, hex case, comment column) are
    accounted for too. A backslash line continuation is removed as well: it is how a bare
    match pattern list wraps, and a list split across lines is the same list.
    """
    collapsed = "\n".join(_style_guide_canon(text))
    collapsed = re.sub(r"\\[ \t]*\n", "", collapsed)
    # A chain broken per segment puts its dot at the start of a line. The legacy goldens write a
    # space after that dot (`. name`) and gdformat2 writes none (`.name`, which is what the
    # guide's whitespace rule asks for), so the space is optional here: it is the wrap, not the
    # code. The pattern needs the newline, so a string's dots are only touched if it spans
    # lines, and it takes only blanks after the dot so a line end is never swallowed.
    collapsed = re.sub(r"\n[ \t]*\.[ \t]*", ".", collapsed)
    collapsed = re.sub(r"[()]", " ", collapsed)
    collapsed = re.sub(r"\s*\bextends\s+" + _EXTENDS_TARGET, "", collapsed)
    collapsed = _NUMERIC_LITERAL.sub(lambda m: m.group(0).replace("_", ""), collapsed)
    # Replacing a paren with a space (above) can leave a space in front of a comma, which no
    # rendering of the same list has: `... + 1\n),` becomes `+ 1 ,`.
    collapsed = re.sub(r"\s+,", ",", collapsed)
    collapsed = re.sub(r"\s+", " ", collapsed).strip()
    # Whitespace just inside brackets or braces is formatting: a single-line list written
    # `{ X }` or `[ X ]` and the same list written one item per line differ only by where the
    # newlines fall.
    collapsed = re.sub(r"\{\s+", "{", collapsed)
    collapsed = re.sub(r"\[\s+", "[", collapsed)
    collapsed = re.sub(r"\s+\]", "]", collapsed)
    return re.sub(r"\s+}", "}", collapsed)


@pytest.mark.parametrize(
    "test_name",
    sorted(GDFORMAT2_GUIDE_DIVERGENCES),
    ids=lambda name: f"GUIDE_{name}",
)
def test_guide_divergence(test_name):
    """Document where a style-guide rule makes gdformat2 differ from the legacy golden.

    The goldens are legacy's output and the legacy suite passes against them unchanged, so a
    fixture where the guide and legacy disagree cannot match. Four things are checked, and
    each of them has caught a mistake already: the output still differs (an entry cannot go
    stale), it differs by more than `_style_guide_canon` alone absorbs (otherwise it belongs
    in `GDFORMAT2_STYLE_GUIDE_DIVERGENCES`), it is the same code once every formatting change
    the guide allows is collapsed, and it never has more over-long lines than the golden.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(this_dir, DATA_DIR, f"{test_name}.in.gd"), encoding="utf-8"
    ) as f:
        input_code = f.read()
    with open(
        os.path.join(this_dir, DATA_DIR, f"{test_name}.out.gd"), encoding="utf-8"
    ) as f:
        expected_output = f.read()

    formatted = format_code(
        p.parse(input_code), StyleConfig(line_length=MAX_LINE_LENGTH)
    )
    reason = GDFORMAT2_GUIDE_DIVERGENCES[test_name]

    assert formatted != expected_output, (
        f"{test_name} now matches the legacy golden; remove it from "
        f"GDFORMAT2_GUIDE_DIVERGENCES ({reason})."
    )
    assert _style_guide_canon(formatted) != _style_guide_canon(expected_output), (
        f"{test_name} differs only by what `_style_guide_canon` absorbs, so it belongs in "
        f"GDFORMAT2_STYLE_GUIDE_DIVERGENCES rather than here ({reason})."
    )
    assert _guide_collapse(_guide_code(formatted)) == _guide_collapse(
        _guide_code(expected_output)
    ), (
        f"{test_name} diverges by more than the guide's formatting rules allow ({reason}):\n"
        f"  expected:  {_guide_collapse(_guide_code(expected_output))[:220]}\n"
        f"  formatted: {_guide_collapse(_guide_code(formatted))[:220]}"
    )
    # `_guide_collapse` drops the `extends` clause, because the guide moves it between the
    # class header and the class body; the targets are compared on their own so that a changed
    # base class is still a failure.
    assert _guide_extends(_guide_code(formatted)) == _guide_extends(
        _guide_code(expected_output)
    ), (
        f"{test_name} changes an `extends` target ({reason}): golden "
        f"{_guide_extends(_guide_code(expected_output))}, formatted "
        f"{_guide_extends(_guide_code(formatted))}."
    )
    assert _guide_comments(formatted) == _guide_comments(expected_output), (
        f"{test_name} changes which comments survive ({reason}):\n"
        f"  expected:  {_guide_comments(expected_output)}\n"
        f"  formatted: {_guide_comments(formatted)}"
    )

    def over_long(text: str):
        return [
            line for line in text.splitlines() if _display_width(line) > MAX_LINE_LENGTH
        ]

    assert len(over_long(formatted)) <= len(over_long(expected_output)), (
        f"{test_name} has more over-long lines than the golden ({reason}): golden "
        f"{len(over_long(expected_output))}, formatted {len(over_long(formatted))}."
    )


# Basic formatting tests adapted from old test_basics.py
@pytest.mark.parametrize(
    "input_code,expected_output",
    [
        ("", ""),
        ("pass", "pass\n"),
    ],
)
def test_formatting_basic(input_code: str, expected_output: str):
    format_and_compare(input_code, expected_output)


def test_indentation_tabs():
    """Test that default formatting uses tabs."""
    input_code = "\n".join(
        [
            "extends Node",
            "func foo():",
            "  if True:",
            "    pass",
            "  if False:",
            "    while True:",
            "      pass",
        ]
    )
    expected_output = "\n".join(
        [
            "extends Node",
            "",
            "",
            "func foo():",
            "\tif True:",
            "\t\tpass",
            "\tif False:",
            "\t\twhile True:",
            "\t\t\tpass",
            "",
        ]
    )
    format_and_compare(input_code, expected_output)


def test_indentation_spaces_3():
    """Test that spaces-based indentation works with custom tab size."""
    input_code = "\n".join(
        [
            "extends Node",
            "func foo():",
            "  if True:",
            "    pass",
            "  if False:",
            "    while True:",
            "      pass",
        ]
    )
    expected_output = "\n".join(
        [
            "extends Node",
            "",
            "",
            "func foo():",
            "   if True:",
            "      pass",
            "   if False:",
            "      while True:",
            "         pass",
            "",
        ]
    )
    format_and_compare(input_code, expected_output, spaces_for_indent=3)


@pytest.mark.parametrize(
    "literal,expected",
    [
        # The guide's own examples: decimal grouped in threes, hexadecimal and binary in fours,
        # and `12345` left alone because a grouping there would need only one separator.
        ("1_234_567_890", "1_234_567_890"),
        ("1234567890", "1_234_567_890"),
        ("0xffff_f8f8_0000", "0xffff_f8f8_0000"),
        ("0xfffff8f80000", "0xffff_f8f8_0000"),
        ("0x0f0f1e1e2d2d", "0x0f0f_1e1e_2d2d"),
        ("0b1101_0010_1010", "0b1101_0010_1010"),
        ("0b110100101010", "0b1101_0010_1010"),
        ("12345", "12345"),
        # Its Bad example: a separator that makes nothing more readable goes.
        ("12_345", "12345"),
        # The threshold differs by base. A decimal literal is grouped from a million up, the cut
        # the guide's note draws (`1000000` needs a grouping, `999999` does not), and the short
        # leading group that produces is what thousands-grouping looks like.
        ("1000000", "1_000_000"),
        ("999999", "999999"),
        # Hexadecimal and binary are grouped in fours, but only when the digits divide evenly into
        # fours - grouping six from the right would split them 2-4 and contradict the guide, whose
        # own `0xfb8c0b` is written with no separator while `0xffff_f8f8_0000` is grouped. So an
        # eight-digit value keeps its separator and anything shorter loses it.
        ("0xffff_ffff", "0xffff_ffff"),
        ("0xffffffff", "0xffff_ffff"),
        ("0xff_99_00", "0xff9900"),
        ("0b11_11_11", "0b111111"),
        ("0b0101111", "0b0101111"),
        ("0xaf01", "0xaf01"),
        ("0x9ff", "0x9ff"),
        # A fraction and an exponent are not magnitudes, so they carry no grouping, and the
        # float keeps its trailing zero.
        ("1_000.000", "1000.000"),
        ("1.5_000", "1.5000"),
        ("1e1_0", "1e10"),
        ("13.", "13.0"),
        (".5", "0.5"),
        # Hexadecimal digits are lower-cased whatever the grouping does.
        ("0xAfA01", "0xafa01"),
        ("0xFB8C0B", "0xfb8c0b"),
    ],
)
def test_number_normalization(literal: str, expected: str):
    """The style guide's underscore rule, which the goldens cannot encode (they keep the
    author's separators verbatim)."""
    from gdtoolkit.gdformat2.expression_to_str import normalize_number

    assert normalize_number(literal) == expected


@pytest.mark.parametrize(
    "source,expected_line",
    [
        # A group that spans a whole slot is redundant, so it comes off: the guide asks to avoid
        # parentheses that are not needed for order of operations or for wrapping over multiple
        # lines, and nothing outside the group can bind to what is inside it.
        ("func f():\n\tvar x = (1 + 2)\n", "\tvar x = 1 + 2"),
        ("func f():\n\tvar x: int = (1 + 2)\n", "\tvar x: int = 1 + 2"),
        ("func f():\n\tconst C = (1 + 2)\n", "\tconst C = 1 + 2"),
        ("func f():\n\tx = (1 + 2)\n", "\tx = 1 + 2"),
        ("func f():\n\tx += (1 + 2)\n", "\tx += 1 + 2"),
        ("func f():\n\tvar x = ((1 + 2))\n", "\tvar x = 1 + 2"),
        ("func f(a):\n\tif (a + 1):\n\t\tpass\n", "\tif a + 1:"),
        ("func f(a):\n\tif ((a and b)):\n\t\tpass\n", "\tif a and b:"),
        ("func f(a):\n\twhile (a + 1):\n\t\tpass\n", "\twhile a + 1:"),
        (
            "func f():\n\tfor i in (range(3) + x):\n\t\tpass\n",
            "\tfor i in range(3) + x:",
        ),
        ("func f():\n\tassert((1 + 2))\n", "\tassert(1 + 2)"),
        ("func f(x = (1 + 2)):\n\tpass\n", "func f(x = 1 + 2):"),
        ("func f():\n\tmatch (x + 1):\n\t\t1:\n\t\t\tpass\n", "\tmatch x + 1:"),
        ("func f():\n\treturn (a or b)\n", "\treturn a or b"),
    ],
)
def test_unnecessary_parentheses_removed(source: str, expected_line: str):
    """The guide's "avoid unnecessary parentheses", applied to a whole slot."""
    formatted = format_code(p.parse(source), StyleConfig(line_length=100))
    assert expected_line in formatted.splitlines(), formatted


@pytest.mark.parametrize(
    "source,kept_line",
    [
        # The group is an operand of an operator that binds looser, so it decides the result:
        # `(1 + 2) * 3` is not `1 + 2 * 3`, and `-(1 + 2)` is not `-1 + 2`.
        ("func f():\n\tvar x = (1 + 2) * 3\n", "\tvar x = (1 + 2) * 3"),
        ("func f():\n\tvar x = -(1 + 2)\n", "\tvar x = -(1 + 2)"),
        # A boolean group is how the guide's own example clears an ambiguity.
        (
            "func f(a):\n\tif (a and b) or not c:\n\t\tpass\n",
            "\tif (a and b) or not c:",
        ),
        # `as` and `is` take their whole left operand, so the group under one is load-bearing.
        ("func f():\n\tvar x = (a or b) as int\n", "\tvar x = (a or b) as int"),
        ("func f():\n\tvar x = (a or b) is int\n", "\tvar x = (a or b) is int"),
        # A ternary binds looser than the cast too, so `(a if b else c) as int` must not become
        # `a if b else c as int`, which casts only the `c` and re-reads as a different program.
        (
            "func f():\n\tvar x = (a if b else c) as int\n",
            "\tvar x = (a if b else c) as int",
        ),
        # A match pattern is not a value slot: `(1):` is a pattern group.
        ("func f(x):\n\tmatch x:\n\t\t(1):\n\t\t\tpass\n", "\t\t(1):"),
    ],
)
def test_necessary_parentheses_kept(source: str, kept_line: str):
    """Parentheses the guide allows - for order of operations, or as a pattern - survive."""
    formatted = format_code(p.parse(source), StyleConfig(line_length=100))
    assert kept_line in formatted.splitlines(), formatted


def test_wrapping_parentheses_kept():
    """A group a value needs in order to wrap is not redundant.

    The guide allows parentheses "for wrapping over multiple lines", so this value keeps its
    group: without it the chain layout breaks every segment onto its own line, which changes
    the shape of the declaration for no gain."""
    source = (
        "func _start_building_placement(building_prototype, Constants):\n"
        "\tvar _active_blueprint_node = (\n"
        "\t\tload(Constants.BUILDING_BLUEPRINTS[building_prototype.resource_path]).instantiate()\n"
        "\t)\n"
    )
    formatted = format_code(p.parse(source), StyleConfig(line_length=100))
    lines = formatted.splitlines()
    assert "\tvar _active_blueprint_node = (" in lines, formatted
    assert "\t)" in lines, formatted


# Comment preservation tests (verified working independently)
@pytest.mark.parametrize(
    "input_code",
    [
        """# Header comment
var x : int = 42 # inline after var
func foo():
    # func body comment
    pass""",
        """# File doc
class MyNode:
    var a: int # member comment
    func _ready():
        # in function
        pass""",
    ],
)
def test_comment_persistence(input_code: str):
    """Verify comments survive gdformat2 formatting."""
    from .common_gdformat2 import test_comment_persistence as tc

    tc(input_code, "")
