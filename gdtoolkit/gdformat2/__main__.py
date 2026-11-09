"""GDScript 2 Formatter — AST-first serializer.

Usage:
  gdformat2 <path>... [options]
  gdformat2 --dump-default-config

Options:
  -c --check                 Don't write the files back,
                             just check if formatting is possible.
  -d --diff                  Don't write the files back,
                             just suggest formatting changes
                             (implies --check).
  -f --fast                  Skip safety checks.
  -l --line-length=<int>     How many characters per line to allow.
  -s --use-spaces=<int>      Use spaces for indent instead of tabs.
  -h --help                  Show this screen.
  --version                  Show version.
  --dump-default-config      Dump default config to 'gdformat2rc' file.

Examples:
  echo 'pass' | gdformat2 -   # reads from STDIN
"""
import sys
import os
import pathlib
import difflib
from typing import List, Optional, Tuple
from types import MappingProxyType
from importlib.metadata import version as pkg_version

from docopt import docopt
import lark
import yaml

from .parser import parser
from .style_config import StyleConfig
from .formatter import format_code
from .safety_checks import (
    check_all_safety,
    TreeInvariantViolation,
    FormattingStabilityViolation,
    CommentPersistenceViolation,
)
from .comments import gather_comments
from ..common.utils import find_gd_files_from_paths

CONFIG_FILE_NAME = "gdformat2rc"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    arguments = docopt(
        __doc__,
        version="gdformat2 {}".format(pkg_version("gdtoolkit")),
    )

    if arguments["--dump-default-config"]:
        _dump_default_config()

    if arguments["--diff"]:
        arguments["--check"] = True

    config_file_path = _find_config_file()
    config = _load_config_file_or_default(config_file_path)
    files = find_gd_files_from_paths(
        arguments["<path>"], excluded_directories=frozenset(config.get("excluded_directories", [".git"]))  # type: ignore
    )

    line_length = (
        int(arguments["--line-length"])
        if arguments["--line-length"]
        else config.get("line_length", 100)
    )
    use_spaces = None
    spaces_for_indent = None
    if arguments["--use-spaces"]:
        use_spaces = True
        spaces_for_indent = int(arguments["--use-spaces"])

    safety_checks_enabled = (
        not arguments["--fast"]
        if arguments.get("--fast")
        else config.get("safety_checks", True)
    )

    # GDScript indents with tabs, so tabs are the default and `--use-spaces` is the
    # opt-out. This used to read `spaces_for_indent is None`, which is True when the flag
    # is absent, so the CLI emitted four spaces where the library default, the dumped
    # config, the `--use-spaces` help text and the tests all use tabs.
    style = StyleConfig(
        line_length=line_length,
        use_spaces=use_spaces is True,
        tab_size=spaces_for_indent or 4,
    )

    if files == ["-"]:
        _format_stdin(style, safety_checks_enabled)
    elif arguments["--check"]:
        _check_files_formatting(
            files, style, arguments["--diff"], safety_checks_enabled
        )
    else:
        _format_files(files, style, safety_checks_enabled)


def _dump_default_config() -> None:
    import os as _os

    assert not _os.path.isfile(CONFIG_FILE_NAME)
    with open(CONFIG_FILE_NAME, "w", encoding="utf-8") as handle:
        config = {
            "excluded_directories": [".git"],
            "safety_checks": True,
            "use_spaces": False,  # Default to tabs (False)
            "line_length": 100,
        }
        handle.write(yaml.dump(config))
    sys.exit(0)


def _find_config_file() -> Optional[str]:
    search_dir = pathlib.Path(os.getcwd())
    config_path = None
    while search_dir != pathlib.Path(os.path.abspath(os.sep)):
        file_path = str(search_dir / CONFIG_FILE_NAME)
        if os.path.isfile(file_path):
            config_path = file_path
            break
        hidden_file_path = str(search_dir / ".{}".format(CONFIG_FILE_NAME))
        if os.path.isfile(hidden_file_path):
            config_path = hidden_file_path
            break
        search_dir = search_dir.parent
    return config_path


def _load_config_file_or_default(config_path: Optional[str]) -> dict:
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle.read())
            return dict(loaded) if loaded else {}
    return {
        "excluded_directories": [".git"],
        "safety_checks": True,
        "use_spaces": False,
        "line_length": 100,
    }


def _format_stdin(style: StyleConfig, safety_checks_enabled: bool) -> None:
    code = sys.stdin.read()
    success, _, formatted_code = _format_code(code, style, safety_checks_enabled)
    if not success:
        sys.exit(1)
    print(formatted_code, end="")


def _check_files_formatting(
    files: List[str], style: StyleConfig, print_diff: bool, safety_enabled: bool
) -> None:
    formattable_files = set()
    failed_files = set()

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                code = handle.read()
                success, _, formatted_code = _format_code(code, style, safety_enabled)
                if success and formatted_code != code:
                    print(f"would reformat {file_path}", file=sys.stderr)
                    if print_diff:
                        print(
                            "\n".join(
                                difflib.unified_diff(
                                    code.splitlines(),
                                    formatted_code.splitlines(),
                                    file_path,
                                    file_path,
                                    lineterm="",
                                )
                            ),
                            file=sys.stderr,
                        )
                    formattable_files.add(file_path)
                elif not success:
                    failed_files.add(file_path)
        except OSError as e:
            print(f"Cannot open file {file_path!r}: {e.strerror}", file=sys.stderr)
            failed_files.add(file_path)

    if len(formattable_files) == 0 and len(failed_files) == 0:
        print(
            "{} file{} would be left unchanged".format(
                len(files), "s" if len(files) != 1 else ""
            )
        )
        sys.exit(0)
    elif len(failed_files) > 0 and len(formattable_files) == 0:
        sys.exit(1)

    formattable_num = len(formattable_files)
    left_unchanged_num = len(files) - formattable_num
    print(
        "{} file{} would be reformatted, {} file{} would be left unchanged.".format(
            formattable_num,
            "s" if formattable_num != 1 else "",
            left_unchanged_num,
            "s" if left_unchanged_num != 1 else "",
        ),
        file=sys.stderr,
    )
    sys.exit(1)


def _format_files(files: List[str], style: StyleConfig, safety_enabled: bool) -> None:
    formatted_files = set()
    failed_files = set()

    for file_path in files:
        try:
            with open(file_path, "r+", encoding="utf-8") as handle:
                code = handle.read()
                success, _, formatted_code = _format_code(code, style, safety_enabled)
                if success and formatted_code != code:
                    print(f"reformatted {file_path}")
                    formatted_files.add(file_path)
                    handle.seek(0)
                    handle.truncate(0)
                    handle.write(formatted_code)
                elif not success:
                    failed_files.add(file_path)
        except OSError as e:
            print(f"Cannot open file {file_path!r}: {e.strerror}", file=sys.stderr)
            failed_files.add(file_path)

    reformatted_num = len(formatted_files)
    left_unchanged_num = len(files) - reformatted_num
    print(
        "{} file{} reformatted, {} file{} left unchanged.".format(
            reformatted_num,
            "s" if reformatted_num != 1 else "",
            left_unchanged_num,
            "s" if left_unchanged_num != 1 else "",
        )
    )
    sys.exit(0 if len(failed_files) == 0 else 1)


def _format_code(
    code: str, style: StyleConfig, safety_enabled: bool = True
) -> Tuple[bool, bool, str]:
    success = True
    actually_formatted = False
    formatted_code = code

    try:
        # Parse with comments embedded as tree nodes
        parse_tree = parser.parse(code)
        comment_tree = parser.parse_comments(code)  # For safety check reference only

        # Format using the AST serializer
        formatted_code = format_code(parse_tree, style)

        if formatted_code != code:
            actually_formatted = True
            if safety_enabled:
                check_all_safety(
                    original_code=code,
                    original_tree=parse_tree,
                    formatted_code=formatted_code,
                    style=style,
                )
    except lark.exceptions.UnexpectedToken as e:
        success = False
        print(
            f"STDIN:\n{lark.exceptions.UnexpectedToken.__doc__ or str(e)}",
            file=sys.stderr,
        )
    except TreeInvariantViolation as e:
        success = False
        print("Failed to format, formatted code parse tree differs", file=sys.stderr)
    except FormattingStabilityViolation as e:
        success = False
        print("Failed to format, formatted code is unstable", file=sys.stderr)
    except CommentPersistenceViolation as e:
        success = False
        print(f"Failed to format, some comments are missing ({e})", file=sys.stderr)

    return success, actually_formatted, formatted_code


if __name__ == "__main__":
    main()
