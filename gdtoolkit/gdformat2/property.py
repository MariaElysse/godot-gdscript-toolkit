"""Property formatting for gdformat2.

Handles inline property accessors (get = __set, get: ... , set(x): ...)
in variable declarations.
"""
from lark import Tree, Token

from ..common.utils import get_line
from .types import FormattedLines


def has_inline_property_body(statement: Tree) -> bool:
    for child in getattr(statement, "children", []):
        if (
            isinstance(child, Tree)
            and str(child.data).endswith("property")
            or (
                hasattr(child, "data")
                and hasattr(child.data, "value")
                and "inline" in str(child.data.value)
            )
        ):  # type: ignore
            return True
    return False


def append_property_body_to_formatted_line(
    line_text: str,
    inline_prop: Tree,
    context_indent_string: str,
    context_indent_level: int,
    style=None,
) -> tuple:
    """Append property body to an existing formatted variable declaration line.

    Returns (new_lines_list, end_line_number).
    """
    lines = [line_text]  # type: FormattedLines
    for child in getattr(inline_prop, "children", []):
        if hasattr(child, "data"):
            handler = _get_property_handler(str(child.data))  # type: ignore
            if handler:
                formatted = handler(
                    child, context_indent_string, context_indent_level + 4, style
                )
                lines.append(formatted[0][1] if formatted else "")

    return ([line_text + ":\n" + "\n".join(lines[1:])], get_line(inline_prop))


def _get_property_handler(data_name: str):
    handlers = {
        "property_delegate_set": _format_property_delegate,
        "property_delegate_get": _format_property_delegate,
        "property_custom_getter": _format_property_getter,
        "property_custom_setter": _format_property_setter,
    }
    return handlers.get(data_name)


def format_property_body(
    property_tree: Tree, indent_string: str, indent_level: int, style=None
):
    """Format a property body (get/set blocks)."""
    lines = []  # type: FormattedLines

    for child in getattr(property_tree, "children", []):
        if hasattr(child, "data"):
            handler = _get_property_handler(str(child.data))  # type: ignore
            if handler is None:
                continue
            formatted_lines = handler(child, indent_string, indent_level + 4, style)
            lines.append(formatted_lines[0][1] if formatted_lines else "")

    return (lines, get_line(property_tree))


def _format_property_delegate(
    prop: Tree, indent_string: str, indent_level: int, style=None
) -> tuple:
    """Format a property delegate (e.g., `get = __set`)."""
    # property_delegate_set or property_delegate_get
    name = ""
    target = ""
    for child in getattr(prop, "children", []):
        if isinstance(child, Token):
            if not name:
                name = str(child.value)  # type: ignore
            elif not target:
                target = str(child.value)  # type: ignore

    return ([(get_line(prop), f"{indent_string}{name} = {target}")], get_line(prop))


def _format_property_getter(
    prop: Tree, indent_string: str, indent_level: int, style=None
) -> tuple:
    """Format a custom getter block."""
    lines = [(get_line(prop), f"{indent_string}get():")]
    # Format body statements (after the `get:` header)
    for child in getattr(prop, "children", []):
        if hasattr(child, "data"):
            data_str = str(child.data).lower()  # type: ignore
            if "getter" not in data_str and data_str not in ("end", "_nl"):
                lines.append((None, f"{indent_string}    {data_str}"))
    return (lines, get_line(prop))


def _format_property_setter(
    prop: Tree, indent_string: str, indent_level: int, style=None
) -> tuple:
    """Format a custom setter block."""
    arg_name = ""
    for child in getattr(prop, "children", []):
        if isinstance(child, Token):
            arg_name = str(child.value)  # type: ignore

    lines = [(get_line(prop), f"{indent_string}set({arg_name}):")]
    for child in getattr(prop, "children", []):
        if hasattr(child, "data"):
            data_str = str(child.data).lower()  # type: ignore
            if "setter" not in data_str and data_str not in ("end", "_nl"):
                lines.append((None, f"{indent_string}    {data_str}"))
    return (lines, get_line(prop))
