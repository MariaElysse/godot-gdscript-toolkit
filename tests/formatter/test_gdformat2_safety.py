"""The formatter's own invariants, run over every fixture pair.

`tests/formatter/common_gdformat2.py` has carried `format_and_compare_with_safety` since the
suite was written, wrapping the three checks in `gdtoolkit/gdformat2/safety_checks.py`, but
nothing called it: `test_input_output_pair` compares the formatted output with the golden and
stops there, so the suite could pass while the formatter changed a file on a second run or lost a
comment. These tests run the three checks themselves, over the same fixture pairs, in under two
minutes: the corpus is parsed and formatted once in a session fixture, and the checks reuse it.

`check_tree_invariant` used to be left out, because it compared the two loose trees textually and
so tripped on every rule that changes the tree's *shape* - it reported 37 of the 119 pairs, each
one a difference the guide asks for. Its comparison now understands the reshaping the guide does:
a wrapping parenthesis is transparent (the guide adds and removes them), a trailing comma and the
`end`/`;` statement markers are punctuation, an empty slot is a fact about the grammar rather than
content, a single-child wrapper is a grammar rung rather than a decision, and the spellings the
guide fixes (`!` for `not`, `1_000` for `1000`, `'x'` for `"x"`) are canonicalized. That leaves
five pairs it still reports, listed in `KNOWN_TREE_INVARIANT_DIVERGENCES` with why, and the test
asserts that set in both directions. It was checked against real damage too - a dropped operand,
a swapped argument, a flipped comparison and an `and` turned into an `or` are all still caught.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from gdtoolkit.gdformat2.formatter import format_code  # noqa: E402
from gdtoolkit.gdformat2.parser import parser as parser_module  # noqa: E402
from gdtoolkit.gdformat2.safety_checks import (  # noqa: E402
    FormattingStabilityViolation,
    TreeInvariantViolation,
    check_comment_persistence,
    check_formatting_stability,
    check_tree_invariant,
)
from gdtoolkit.gdformat2.style_config import StyleConfig  # noqa: E402

FIXTURES = pathlib.Path(__file__).parent / "input-output-pairs"
# The one pair whose golden is not valid GDScript, so its input tests nothing about the
# formatter (`GDFORMAT2_EXCEPTIONS` in the golden suite names the same pair).
SKIP = ("bug_326_multistatement_lambda_corner_case.in.gd",)

# Fixtures that still change when the formatter runs twice, with why. The test looks in both
# directions: one of these that becomes stable fails, and so does a fixture that becomes
# unstable, so the list cannot rot - the same discipline as the divergence maps.
#
# It is empty, and that is a checked claim rather than a stale one: the last two entries were
# both comment attachment in a property's accessor list, and they went stable once an accessor's
# trailing comments were emitted in source order and at the scope their column names (see
# `_format_property` in `var_statement.py`).
KNOWN_NON_IDEMPOTENT = {}

# Pairs `check_tree_invariant` still reports, each because the guide lays the construct out in a
# shape that re-parses differently rather than because something was lost. An inline accessor
# list becomes one accessor per line, which is the block form of the `property` rule; a lambda's
# body becomes a block, so a comment trailing it is attached inside it; a long lambda's body is
# emitted as a call path that re-parses as a different call node.
KNOWN_TREE_INVARIANT_DIVERGENCES = {
    "inline_properties":
        "an inline accessor list is laid out one accessor per line, which re-parses as the "
        "block form (`property` + `property_accessor_line`) instead of `untitled_property`",
    "inline_properties_w_comments":
        "as `inline_properties`, with the accessor list's own comments",
    "multiline_properties_w_comments":
        "an accessor's trailing comment is emitted at the scope its column names, one level "
        "shallower or deeper than it was written",
    "inline_lambdas_w_comments":
        "a comment trailing an inline lambda's block body is attached inside the body",
    "long_unistatement_lambdas":
        "a long lambda's body is emitted as a call path that re-parses as a different call node",
}


@pytest.fixture(scope="session")
def formatted():
    """Every fixture's tree and formatted output, computed once for all the tests here."""
    config = StyleConfig(line_length=100)
    prepared = {}
    for path in sorted(FIXTURES.glob("*.in.gd")):
        if path.name in SKIP:
            continue
        tree = parser_module.parse(path.read_text())
        prepared[path.name] = (tree, format_code(tree, config))
    return prepared


@pytest.mark.parametrize("name", sorted(
    path.name for path in FIXTURES.glob("*.in.gd") if path.name not in SKIP
))
def test_formatting_is_stable(name, formatted):
    """Running the formatter on its own output changes nothing, or the change is recorded.

    `check_formatting_stability` re-formats the output itself, so this is one extra pass over the
    corpus in total. The recorded fixtures are asserted to be *still* unstable, which is what
    makes `KNOWN_NON_IDEMPOTENT` a record rather than a dumping ground: a fixture that gets fixed
    fails this test until it is struck off the list, and one that becomes unstable fails too.
    """
    _tree, once = formatted[name]
    config = StyleConfig(line_length=100)
    if name.replace(".in.gd", "") in KNOWN_NON_IDEMPOTENT:
        with pytest.raises(FormattingStabilityViolation):
            check_formatting_stability(once, config)
    else:
        check_formatting_stability(once, config)


@pytest.mark.parametrize("name", sorted(
    path.name for path in FIXTURES.glob("*.in.gd") if path.name not in SKIP
))
def test_comments_survive_formatting(name, formatted):
    """Every comment in the input is still there afterwards."""
    tree, once = formatted[name]
    check_comment_persistence(tree, once)


@pytest.mark.parametrize("name", sorted(
    path.name for path in FIXTURES.glob("*.in.gd") if path.name not in SKIP
))
def test_tree_is_invariant(name, formatted):
    """The output re-parses to the same program, once the guide's reshaping is accounted for.

    `KNOWN_TREE_INVARIANT_DIVERGENCES` is asserted in both directions like the idempotency set:
    a pair recorded here that stops reporting fails until it is struck off, and a pair that
    starts reporting fails too, so neither a fresh regression nor a stale record can hide.
    """
    _tree, once = formatted[name]
    source = (FIXTURES / name).read_text()
    if name.replace(".in.gd", "") in KNOWN_TREE_INVARIANT_DIVERGENCES:
        with pytest.raises(TreeInvariantViolation):
            check_tree_invariant(source, once)
    else:
        check_tree_invariant(source, once)
