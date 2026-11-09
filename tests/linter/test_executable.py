import io
import subprocess
import sys
from contextlib import redirect_stdout

import pytest

from ..common import write_file
from gdtoolkit.linter.problem_printer import (
    print_problem_github,
    print_problem,
)
from gdtoolkit.linter.problem import Problem


def test_valid_file_no_config(tmp_path):
    dummy_file = write_file(tmp_path, "script.gd", "pass")
    assert subprocess.run(["gdlint", dummy_file], check=False).returncode == 0


def test_valid_file_w_config(tmp_path):
    dummy_file = write_file(tmp_path, "script.gd", "pass")
    assert subprocess.run(["gdlint", "-d"], cwd=tmp_path, check=False).returncode == 0
    assert (
        subprocess.run(["gdlint", dummy_file], cwd=tmp_path, check=False).returncode
        == 0
    )


def test_invalid_file_no_config(tmp_path):
    dummy_file = write_file(tmp_path, "script.gd", "var Xx = 1")
    assert subprocess.run(["gdlint", dummy_file], check=False).returncode != 0


def test_invalid_file_w_config(tmp_path):
    dummy_file = write_file(tmp_path, "script.gd", "var Xx = 1")
    assert subprocess.run(["gdlint", "-d"], cwd=tmp_path, check=False).returncode == 0
    assert (
        subprocess.run(["gdlint", dummy_file], cwd=tmp_path, check=False).returncode
        != 0
    )


def test_pretty_printing_missing_file():
    outcome = subprocess.run(
        ["gdlint", "nonexistent.gd"], check=False, capture_output=True
    )
    assert outcome.returncode == 1
    assert len(outcome.stdout.decode().splitlines()) == 0
    assert len(outcome.stderr.decode().splitlines()) > 0
    assert "Traceback" not in outcome.stderr.decode()


def test_pretty_printing_unexpected_token(tmp_path):
    dummy_file = write_file(tmp_path, "script.gd", "pass x")
    outcome = subprocess.run(["gdlint", dummy_file], check=False, capture_output=True)
    assert outcome.returncode == 1
    assert len(outcome.stdout.decode().splitlines()) == 0
    assert len(outcome.stderr.decode().splitlines()) > 0
    assert "Traceback" not in outcome.stderr.decode()


def test_linter_problem_report_in_global_scope(tmp_path):
    dummy_file = write_file(tmp_path, "script.gd", "var foo = 1\nconst X = 1")
    outcome = subprocess.run(["gdlint", dummy_file], check=False, capture_output=True)
    assert outcome.returncode == 1
    assert len(outcome.stdout.decode().splitlines()) == 0
    assert len(outcome.stderr.decode().splitlines()) >= 0
    assert "Definition out of order in global scope" in outcome.stderr.decode()


def test_github_format_flag_is_accepted(tmp_path):
    """The --output-format flag should be accepted without error."""
    dummy_file = write_file(tmp_path, "script.gd", "pass")
    outcome = subprocess.run(
        ["gdlint", "--output-format", "github", dummy_file], check=False
    )
    assert outcome.returncode == 0


def test_github_format_output_on_stdout(tmp_path):
    """GitHub format should write lint errors to stdout, not stderr."""
    # var Xx = 1 triggers a constant-name violation
    dummy_file = write_file(tmp_path, "script.gd", "var Xx = 1")
    outcome = subprocess.run(
        ["gdlint", "--output-format", "github", dummy_file],
        check=False,
        capture_output=True,
    )
    assert outcome.returncode != 0
    # stdout should have the GitHub annotation line
    stdout_lines = outcome.stdout.decode().strip()
    assert len(stdout_lines) > 0
    assert stdout_lines.startswith("::error file=")
    # stderr summary should still go to stderr
    assert "problem" in outcome.stderr.decode().lower()


def test_github_format_contains_file_line_col(tmp_path):
    """GitHub format annotations should include file, line, and column."""
    dummy_file = write_file(tmp_path, "script.gd", "var Xx = 1")
    outcome = subprocess.run(
        ["gdlint", "--output-format", "github", dummy_file],
        check=False,
        capture_output=True,
    )
    annotation = outcome.stdout.decode().strip()
    # File path may be absolute (homebrew gdlint resolves it)
    assert "script.gd" in annotation
    assert ",line=" in annotation
    assert ",col=" in annotation


def test_text_format_unchanged(tmp_path):
    """Default text format should still output to stderr."""
    dummy_file = write_file(tmp_path, "script.gd", "var Xx = 1")
    outcome = subprocess.run(
        ["gdlint", dummy_file],
        check=False,
        capture_output=True,
    )
    assert outcome.returncode != 0
    stderr_text = outcome.stderr.decode()
    assert "script.gd:1:" in stderr_text
    assert "Xx" in stderr_text
    # stdout should be empty for text format
    assert len(outcome.stdout.decode().strip()) == 0


def test_github_format_explicit_text_flag(tmp_path):
    """--output-format text should behave identically to no flag."""
    dummy_file = write_file(tmp_path, "script.gd", "var Xx = 1")
    outcome_with_flag = subprocess.run(
        ["gdlint", "--output-format", "text", dummy_file],
        check=False,
        capture_output=True,
    )
    outcome_default = subprocess.run(
        ["gdlint", dummy_file],
        check=False,
        capture_output=True,
    )
    assert outcome_with_flag.stderr == outcome_default.stderr


def test_print_problem_github_basic(capsys):
    """Direct function test: basic GitHub output format."""
    problem = Problem(
        name="function-argument-name",
        description='Function argument name "x" is not valid',
        line=42,
        column=5,
    )
    print_problem_github(problem, "test.gd")
    captured = capsys.readouterr()
    assert captured.err == ""  # should only go to stdout
    assert "::error file=test.gd,line=42,col=5::" in captured.out


def test_print_problem_github_no_column(capsys):
    """Direct function test: falls back to col 1 when column is None."""
    problem = Problem(
        name="unnecessary-pass",
        description="Unnecessary 'pass' statement",
        line=10,
        column=None,
    )
    print_problem_github(problem, "foo.gd")
    captured = capsys.readouterr()
    assert ",col=1::" in captured.out


def test_print_problem_github_no_commas_in_description(capsys):
    """Direct function test: commas in description are replaced with semicolons."""
    problem = Problem(
        name="test",
        description="This has, a comma, in it",
        line=5,
        column=1,
    )
    print_problem_github(problem, "a.gd")
    captured = capsys.readouterr()
    # The original commas from the description should become semicolons
    assert "This has; a comma; in it" in captured.out
    # But no raw commas from the original text (the ", line X" suffix is fine)
    output_after_msg = captured.out.split("::error")[1]
    msg_and_metadata = output_after_msg.split(")")[0] + ")"
    # Check that description commas are gone - look for semicolons instead
    assert "has, a" not in captured.out
    assert "a comma," not in captured.out
