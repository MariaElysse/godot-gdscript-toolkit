import sys
from .problem import Problem


def print_problem(problem: Problem, file_path: str) -> None:  # TODO: colors
    print(
        "{}:{}: Error: {} ({})".format(
            file_path,
            problem.line,
            problem.description,
            problem.name,
        ),
        file=sys.stderr,
    )


def print_problem_github(problem: Problem, file_path: str) -> None:
    """Output a lint problem as a GitHub workflow command.

    Produces lines like:
      ::error file=script.gd,line=42,col=5::Function argument name \"x\" is
      not valid (function-argument-name)

    Written to stdout so GitHub Actions parses them as annotations.
    """
    col = problem.column if problem.column else 1
    print(
        "::error file={},line={},col={}::{}, line {} ({})".format(
            file_path,
            problem.line,
            col,
            problem.description.replace(",", ";"),
            problem.line,
            problem.name,
        ),
        file=sys.stdout,
    )
