"""Tests for the gdformat2 lint module: the guide rules formatting cannot enforce."""
import pathlib

import pytest

from gdtoolkit.gdformat2.formatter import format_code
from gdtoolkit.gdformat2.lint import CHECKS, Problem, lint_code
from gdtoolkit.gdformat2.parser import parser as parser_module
from gdtoolkit.gdformat2.style_config import StyleConfig

FIXTURES = pathlib.Path(__file__).parent / "input-output-pairs"


def checks_of(source: str, filename: str = "ok_name.gd"):
    return [problem.check for problem in lint_code(source, filename=filename)]


# ─── code order ───────────────────────────────────────────────────────────────


def test_code_order_accepts_the_guides_order():
    source = (
        "@tool\n"
        "class_name Weapon\n"
        "extends Node2D\n"
        "## A weapon.\n"
        "\n"
        "signal door_opened\n"
        "\n"
        "enum Kind { SWORD }\n"
        "\n"
        "const MAX_AMMO = 30\n"
        "\n"
        "static var shared := 1\n"
        "\n"
        "@export var damage: int = 5\n"
        "var _private: int = 2\n"
        "\n"
        "@onready var bar: Node = get_node(\"UI/LifeBar\")\n"
        "\n"
        "static func helper() -> void:\n"
        "\tpass\n"
        "\n"
        "func _ready() -> void:\n"
        "\tpass\n"
        "\n"
        "func public_method() -> void:\n"
        "\tpass\n"
        "\n"
        "class Inner:\n"
        "\tvar x: int = 1\n"
    )
    assert "code-order" not in checks_of(source)


@pytest.mark.parametrize("source,line", [
    # A signal after a constant, an enum after a signal, a method between variables.
    ("const A = 1\n\nsignal opened\n", 3),
    ("enum Kind { SWORD }\n\nsignal opened\n", 3),
    ("var a: int = 1\n\nfunc f() -> void:\n\tpass\n\nvar b: int = 2\n", 6),
    # A private variable after a public one is fine; a public one after a private is not.
    ("var _private: int = 1\n\nvar public_one: int = 2\n", 3),
    # An inner class belongs last.
    ("class Inner:\n\tvar x: int = 1\n\nvar after: int = 2\n", 4),
])
def test_code_order_reports_the_line_out_of_order(source, line):
    problems = [p for p in lint_code(source, filename="ok_name.gd") if p.check == "code-order"]
    assert [problem.line for problem in problems] == [line]


def test_code_order_lists_the_guide_order_in_the_suggestion():
    problem = next(
        p for p in lint_code("const A = 1\n\nsignal opened\n", filename="ok_name.gd")
        if p.check == "code-order"
    )
    for section in ("@tool", "class_name", "extends", "signals", "enums", "constants",
                    "static variables", "@export variables", "private variables",
                    "@onready variables", "static functions", "virtual methods",
                    "inner classes"):
        assert section in problem.suggestion
    assert "signals" in problem.message and "constants" in problem.message


def test_code_order_applies_inside_an_inner_class():
    source = "class Inner:\n\tconst A = 1\n\n\tsignal opened\n"
    problems = [p for p in lint_code(source, filename="ok_name.gd") if p.check == "code-order"]
    assert [problem.line for problem in problems] == [4]


@pytest.mark.parametrize("source", [
    # A plain comment carries no order, wherever it sits.
    "var a: int = 1\n\n# a note about b\nvar b: int = 2\n",
    "signal opened\n\n# a note\nfunc f() -> void:\n\tpass\n",
    # A member's own doc comment is not the class's, so it is not out of place either.
    "signal opened\n\n## Emitted when a door opens.\nfunc f() -> void:\n\tpass\n",
])
def test_code_order_ignores_comments_that_carry_no_order(source):
    assert "code-order" not in checks_of(source)


# ─── continuation indent ──────────────────────────────────────────────────────


def test_continuation_indent_wants_two_levels():
    """A block is reported once, at its first content line: the indent is the block's."""
    source = "func f() -> void:\n\tcall_a_function(\n\t\tfirst_argument,\n\t\tsecond_argument\n\t)\n"
    problems = [
        p for p in lint_code(source, filename="ok_name.gd") if p.check == "continuation-indent"
    ]
    assert [problem.line for problem in problems] == [3]


def test_continuation_indent_accepts_one_level_when_two_would_overflow():
    """The limit wins over the indent: a block that fits at one level and not at two is right.

    The chain is 90 columns, so two levels (three tabs, 12 columns) would put it at 102 while
    the one level it has (two tabs, 8) sits at 98 - the shape the formatter produces for a
    chain of `not`s in `long_boolean_test_expressions`.
    """
    chain = "not " * 22 + "tt"
    source = "func f() -> void:\n\tvar c = (\n\t\t" + chain + "\n\t)\n"
    assert "continuation-indent" not in checks_of(source)


def test_continuation_indent_still_reports_a_block_that_is_over_either_way():
    """A block over the limit at one level too keeps the two levels, so the indent is reported.

    The long line itself is the `unbreakable-line` check's business, not the indent's.
    """
    chain = "not " * 30 + "true"
    source = "func f() -> void:\n\tvar c = (\n\t\t" + chain + "\n\t)\n"
    assert "continuation-indent" in checks_of(source)


def test_continuation_indent_reports_a_backslash_continuation():
    source = "func f() -> void:\n\tvar x = 1 + \\\n\t1\n"
    problems = [
        p for p in lint_code(source, filename="ok_name.gd") if p.check == "continuation-indent"
    ]
    assert [problem.line for problem in problems] == [3]


def test_continuation_indent_accepts_two_levels():
    source = (
        "func f() -> void:\n"
        "\tcall_a_function(\n"
        "\t\t\tfirst_argument,\n"
        "\t\t\tsecond_argument\n"
        "\t)\n"
    )
    assert "continuation-indent" not in checks_of(source)


def test_continuation_indent_leaves_arrays_and_dicts_alone():
    source = "var party = [\n\t\"Godot\",\n\t\"Godette\",\n]\n"
    assert "continuation-indent" not in checks_of(source)


def test_continuation_indent_leaves_a_closing_bracket_alone():
    source = "func f() -> void:\n\tcall_a_function(\n\t\t\targument\n\t)\n"
    assert "continuation-indent" not in checks_of(source)


# ─── unbreakable line ─────────────────────────────────────────────────────────


def test_unbreakable_line_reports_a_name_that_fills_the_line():
    problem = next(
        p for p in lint_code("var " + "a" * 100 + " = 1\n", filename="ok_name.gd")
        if p.check == "unbreakable-line"
    )
    assert "identifier" in problem.suggestion
    assert "100 columns" in problem.message
    assert problem.section == "line-length"


@pytest.mark.parametrize("source", [
    "var x = " + " + ".join("item_%02d" % index for index in range(14)) + "\n",
    "func f() -> void:\n\tcall(first_argument, second_argument, third_argument, fourth_arg)\n",
    "var names = [" + ", ".join('"name_%02d"' % index for index in range(10)) + "]\n",
])
def test_unbreakable_line_stays_quiet_when_a_reflow_can_help(source):
    assert "unbreakable-line" not in checks_of(source)


def test_unbreakable_line_reports_a_comment_that_needs_wrapping():
    problem = next(
        p for p in lint_code("# " + "word " * 25 + "\n", filename="ok_name.gd")
        if p.check == "unbreakable-line"
    )
    assert "wrap the comment" in problem.suggestion


def test_unbreakable_line_matches_every_line_the_formatter_leaves_long():
    """The check is the complement of the formatter, so the two must agree.

    For each fixture whose formatted output has a line over the limit, the lines the check
    reports are exactly those lines: gdformat2 wrapped everything it could.
    """
    config = StyleConfig(line_length=100)
    checked = 0
    for path in sorted(FIXTURES.glob("*.in.gd")):
        if path.name.startswith("bug_326_multistatement_lambda_corner_case."):
            continue
        formatted = format_code(parser_module.parse(path.read_text()), config)
        over_long = [
            index + 1
            for index, line in enumerate(formatted.splitlines())
            if len(line.replace("\t", "    ")) > 100
        ]
        reported = sorted({
            problem.line
            for problem in lint_code(formatted, filename=path.name)
            if problem.check == "unbreakable-line"
        })
        assert reported == over_long, path.name
        checked += 1
    assert checked > 100


# ─── typing ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("source", [
    "var health := 0\n",
    "var ratio := 1.5\n",
    "var bar := get_node(\"UI/LifeBar\")\n",
    "func f() -> void:\n\tvar speed := 0\n",
])
def test_ambiguous_inferred_type_reports_an_ambiguity(source):
    assert "ambiguous-inferred-type" in checks_of(source)


@pytest.mark.parametrize("source", [
    "var health: int = 0\n",
    "var direction := Vector3(1, 2, 3)\n",
    "const MAX_AMMO = 30\n",
    "var speed = 0\n",
    "var health_bar: ProgressBar = get_node(\"UI/LifeBar\")\n",
])
def test_ambiguous_inferred_type_accepts_a_stated_type(source):
    assert "ambiguous-inferred-type" not in checks_of(source)


def test_redundant_type_hint_reports_a_repeated_constructor():
    problem = next(
        p for p in lint_code("var direction: Vector3 = Vector3(1, 2, 3)\n", filename="ok_name.gd")
        if p.check == "redundant-type-hint"
    )
    assert "Vector3" in problem.message
    assert "`:=`" in problem.suggestion


@pytest.mark.parametrize("source", [
    "var direction: Vector3 = other_direction\n",
    "var health: int = 0\n",
    "var direction: Vector3 = Vector3(1, 2, 3) as Variant\n",
])
def test_redundant_type_hint_leaves_a_useful_hint_alone(source):
    assert "redundant-type-hint" not in checks_of(source)


# ─── comments, names and encoding ─────────────────────────────────────────────


@pytest.mark.parametrize("source", [
    '# print("This is disabled code")\n',
    "# var x = 1\n",
])
def test_commented_out_code_reports_code_in_a_comment(source):
    assert "commented-out-code" in checks_of(source)


@pytest.mark.parametrize("source", [
    "# This is a comment.\n",
    "# See the docs for details.\n",
    "# TODO: fix this later\n",
    "# A weapon is a node that can be picked up.\n",
])
def test_commented_out_code_leaves_prose_alone(source):
    assert "commented-out-code" not in checks_of(source)


def test_signal_past_tense_reports_a_present_tense_name():
    problems = [p for p in lint_code("signal door_opens\n", filename="ok_name.gd")
                if p.check == "signal-past-tense"]
    assert [problem.line for problem in problems] == [1]


@pytest.mark.parametrize("name", ["door_opened", "score_changed", "tree_exited", "status"])
def test_signal_past_tense_leaves_a_past_tense_name_alone(name):
    assert "signal-past-tense" not in checks_of("signal {}\n".format(name))


def test_file_name_reports_a_mismatch_with_the_class_name():
    problem = next(
        p for p in lint_code("class_name YAMLParser\n", filename="parser.gd")
        if p.check == "file-name"
    )
    assert "yaml_parser.gd" in problem.suggestion


def test_file_name_accepts_a_matching_snake_case_name():
    assert "file-name" not in checks_of("class_name YAMLParser\n", filename="yaml_parser.gd")


def test_file_name_is_skipped_without_a_file_name():
    assert "file-name" not in [problem.check for problem in lint_code("class_name YAMLParser\n")]


def test_byte_order_mark_is_reported_even_when_the_file_does_not_parse():
    # A BOM is the reason the file does not parse, so the check has to work without a tree.
    assert "byte-order-mark" in checks_of("\ufeffvar x = 1\n")


# ─── controlling the checks ───────────────────────────────────────────────────


def test_problems_are_sorted_by_position():
    problems = lint_code("signal door_opens\nvar health := 0\n", filename="ok_name.gd")
    assert [problem.line for problem in problems] == sorted(p.line for p in problems)


def test_a_disable_list_silences_a_check():
    from types import MappingProxyType

    from gdtoolkit.gdformat2.lint import DEFAULT_CONFIG

    config = MappingProxyType({**DEFAULT_CONFIG, "disable": ["signal-past-tense"]})
    assert "signal-past-tense" not in [
        problem.check for problem in lint_code("signal door_opens\n", filename="ok_name.gd",
                                               config=config)
    ]


@pytest.mark.parametrize("comment", [
    "# gdlint: ignore=signal-past-tense",
    "# gdformat2: ignore=signal-past-tense",
])
def test_an_inline_ignore_silences_the_next_line(comment):
    source = "{}\nsignal door_opens\n".format(comment)
    assert "signal-past-tense" not in checks_of(source)


def test_every_check_is_named_in_the_public_list():
    assert set(CHECKS) == {
        "code-order",
        "continuation-indent",
        "unbreakable-line",
        "commented-out-code",
        "redundant-type-hint",
        "ambiguous-inferred-type",
        "file-name",
        "signal-past-tense",
        "byte-order-mark",
    }


def test_a_problem_carries_a_guide_link():
    problem = next(
        p for p in lint_code("var health := 0\n", filename="ok_name.gd")
        if p.check == "ambiguous-inferred-type"
    )
    assert isinstance(problem, Problem)
    assert problem.guide_url.endswith("#inferred-types")
    assert problem.format_with("file.gd").startswith("file.gd:1:")
