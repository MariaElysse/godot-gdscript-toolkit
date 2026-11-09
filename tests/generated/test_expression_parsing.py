import re

import pytest
import hypothesis.extra.lark
import hypothesis.strategies as st
from hypothesis import settings, given, HealthCheck
from lark import Lark

from gdtoolkit.parser import parser
from gdtoolkit.formatter import format_code, check_formatting_safety


MAX_LINE_LENGTH = 100

with open("gdtoolkit/parser/gdscript.lark", "r", encoding="utf-8") as handle:
    gdscript_grammar = handle.read()
    simplified_gdscript_grammar = gdscript_grammar.replace(".2", "")
    simplified_gdscript_grammar = simplified_gdscript_grammar.replace(
        '["+"] atom',
        '["+"] " " atom " "',
    )
    # hypothesis does not support the \p syntax
    simplified_gdscript_grammar = re.sub(
        r"NAME:.*$",
        r"%import common.CNAME -> NAME",
        simplified_gdscript_grammar,
        flags=re.MULTILINE,
    )
    print(simplified_gdscript_grammar)
    for keyword in ["as", "is not", "is", "not", "await", "and", "else", "if", "or", "in"]:
        simplified_gdscript_grammar = simplified_gdscript_grammar.replace(
            f'"{keyword}"', f'" {keyword} "'
        )
    # lambdas may generate _NL/_INDENT suites containing real newlines, while
    # this harness embeds the whole expression into a single-line
    # "func foo(): <expr>" body, so drop lambdas from generated expressions
    simplified_gdscript_grammar = simplified_gdscript_grammar.replace(
        "     | lambda\n", ""
    )
    simplified_gdscript_grammar = simplified_gdscript_grammar.replace("TYPE:", "XTYPE:")
    # NB: glue the type with a *leading* space only. A trailing space would
    # emit `A [ A ]`, which the real lexer cannot match as the TYPE_HINT
    # terminal (it only accepts `A[A]`/`A [A]`-style single spaces inside).
    simplified_gdscript_grammar = simplified_gdscript_grammar.replace(
        "%ignore WS_INLINE", 'TYPE: " " XTYPE'
    )
    # The real TYPE_HINT terminal allows no space after '[' or around the
    # inner types, so nest bare XTYPEs there; only the outer TYPE keeps the
    # token-separating leading space (the real lexer ignores it anyway).
    simplified_gdscript_grammar = simplified_gdscript_grammar.replace(
        'TYPE_HINT: TYPE ["[" TYPE [[" "] "," [" "] TYPE] "]"]',
        'TYPE_HINT: TYPE ["[" XTYPE [[" "] "," [" "] XTYPE] "]"]',
    )
    simplified_gdscript_grammar = simplified_gdscript_grammar.replace(
        "%ignore COMMENT", ""
    )
    gdscript_lark = Lark(simplified_gdscript_grammar, regex=True)


def format_and_check_safety(input_code):
    formatted_code = format_code(input_code, max_line_length=MAX_LINE_LENGTH)
    check_formatting_safety(input_code, formatted_code, MAX_LINE_LENGTH)


@pytest.mark.generated
@settings(
    deadline=None,
    suppress_health_check=(HealthCheck.filter_too_much, HealthCheck.too_slow),
    max_examples=500,
)
@given(
    hypothesis.extra.lark.from_lark(
        gdscript_lark,
        start="expr",
        explicit={"_INDENT": st.just("\n\t"), "_DEDENT": st.just("")},
    )
)  # type: ignore
def test_expression_parsing(expression):
    print(expression)
    gdscript_code = "func foo():{}\n".format(expression)
    parser.parse(gdscript_code)  # just checking if not throwing


@pytest.mark.generated
@settings(
    deadline=None,
    suppress_health_check=(HealthCheck.filter_too_much, HealthCheck.too_slow),
    max_examples=500,
)
@given(
    hypothesis.extra.lark.from_lark(
        gdscript_lark,
        start="expr",
        explicit={"_INDENT": st.just("\n\t"), "_DEDENT": st.just("")},
    )
)  # type: ignore
def test_expression_formatting(expression):
    print(expression)
    gdscript_code = "func foo():{}\n".format(expression)
    format_and_check_safety(gdscript_code)
