"""Block formatting for gdformat2.

Handles blank lines between statements, indentation, and delegates to
statement formatters. Each statement may carry comment children that are
emitted inline/standalone by the component handler.
"""
from typing import List, Callable, Any

from .types import FormattedLines
from .style_config import StyleConfig
from .annotation import (
    annotation_name,
    annotation_to_flat_str,
    is_prependable_annotation,
)


# Comments and annotations attach to what follows, so they are transparent when
# computing the forced blank lines between two real statements.
_TRANSPARENT = ("comment", "annotation")


def _prev_real_type(item_types: list, idx: int) -> str:
    for k in range(idx - 1, -1, -1):
        if item_types[k] not in _TRANSPARENT:
            return item_types[k]
    return ""


def _next_real_type(item_types: list, idx: int) -> str:
    for k in range(idx, len(item_types)):
        if item_types[k] not in _TRANSPARENT:
            return item_types[k]
    return ""


def _comment_text_of(item) -> str:
    node = item[1] if isinstance(item, tuple) else item
    return _get_comment_text(node)


def _ends_with_doc_run(statements, item_types, i) -> bool:
    """True when the items before `i` end in a `##` doc comment run.

    The guide's documentation comments "must immediately precede a script member", and a
    doc comment on a class goes right after `extends`/`class_name` and immediately before
    the first member. So a doc run is joined to the declaration it documents: the blank
    line an author left between them is dropped, and the declaration's own spacing was
    emitted before the run instead.
    """
    head = None
    k = i - 1
    while k >= 0 and item_types[k] == "comment":
        head = statements[k]
        k -= 1
    return head is not None and _comment_text_of(head).startswith("##")


def _blank_lines_before_item(
    i,
    statements,
    item_types,
    actual_node,
    prev_end_line,
    is_nested,
    style,
    deferred=None,
):
    """Blank lines to emit before item `i`, plus any count carried to the next item.

    Returns `(lines, deferred)`. `deferred` is `None` unless the item is a comment
    run that already accounted for the following declaration's spacing; that
    declaration then emits the deferred count instead of recomputing its own.
    """
    if i == 0:
        return [], None
    prev_item = statements[i - 1]
    prev_node = prev_item[1] if isinstance(prev_item, tuple) else prev_item
    gap = get_line_safe(actual_node) - _content_end_line(prev_node) - 1
    kind = item_types[i]
    prev_real = _prev_real_type(item_types, i)
    next_real = _next_real_type(item_types, i)

    if kind in _TRANSPARENT:
        if item_types[i - 1] in _TRANSPARENT and gap <= 0:
            # Inside a comment/annotation run: the head already carried the spacing.
            return [], deferred
        if kind == "annotation":
            # A joined annotation heads the declaration's line, so only the spacing
            # forced by the previous statement applies to it.
            forced_prev = (
                _get_blank_lines(prev_real, prev_real, style, prev_end_line, is_nested)
                if prev_real
                else []
            )
            before = max(len(forced_prev), 1 if gap >= 1 else 0)
            return [(None, "")] * before, deferred
        if not next_real:
            # Trailing comment: nothing follows, so no spacing is forced, but a
            # blank the author wrote is still preserved (squeezed to one line).
            return [(None, "")] * (1 if gap >= 1 else 0), None
        # A comment run. Blanks forced by the PREVIOUS statement go before the run;
        # blanks forced by the NEXT declaration go after it. A blank the author
        # wrote in front of the run pulls that declaration's spacing in front too.
        forced_prev = (
            _get_blank_lines(prev_real, prev_real, style, prev_end_line, is_nested)
            if prev_real
            else []
        )
        forced_next = _get_blank_lines(
            next_real, next_real, style, prev_end_line, is_nested
        )
        before = max(
            len(forced_prev),
            len(forced_next) if gap >= 1 else 0,
            1 if gap >= 1 else 0,
        )
        # The run ends at the last comment in it, and the gap that decides whether the author
        # detached it from what follows is measured from there. Measuring from the run's FIRST
        # line counts the run's own length as if it were a gap: a three-line `# ---` box looked
        # detached from the declaration it labels, so that declaration emitted its two blank
        # lines a second time, inside the box.
        run_end_node = actual_node
        for j in range(i + 1, len(item_types)):
            if item_types[j] not in _TRANSPARENT:
                break
            run_end_node = (
                statements[j][1] if isinstance(statements[j], tuple) else statements[j]
            )
        after = 0
        if not _comment_text_of(actual_node).startswith("##"):
            # A `##` run is a doc comment, which documents the declaration right below it
            # and so never keeps a blank in between. Any other comment run does keep one
            # the author wrote, and takes the declaration's spacing with it.
            for j in range(i + 1, len(item_types)):
                if item_types[j] not in _TRANSPARENT:
                    after_node = (
                        statements[j][1]
                        if isinstance(statements[j], tuple)
                        else statements[j]
                    )
                    if (
                        get_line_safe(after_node) - _content_end_line(run_end_node) - 1
                        >= 1
                    ):
                        after = len(forced_next)
                    break
        return [(None, "")] * before, after

    if _ends_with_doc_run(statements, item_types, i):
        return [], None
    forced = (
        _get_blank_lines(prev_real, next_real, style, prev_end_line, is_nested)
        if next_real
        else []
    )
    source_blanks = 1 if gap >= 1 else 0
    if deferred is not None:
        count = max(deferred, source_blanks)
    else:
        count = max(len(forced), source_blanks)
    return [(None, "")] * count, None


def _render_annotations_standalone(annotations, indent_string, indent_level, style):
    """Render each annotation on its own line(s), never joined to a declaration.

    `annotations` is a list of `(node, inline_comment_text)` pairs; a trailing
    comment on the annotation's source line rides along on its first line.
    """
    from .annotation import format_annotation

    out: list = []
    for ann, text in annotations:
        ann_lines, _ = format_annotation(ann, indent_string, indent_level, style)
        ann_lines = [ln for ln in ann_lines if isinstance(ln, tuple)]
        if text and ann_lines:
            first = ann_lines[0]
            ann_lines[0] = (first[0], f"{first[1]}  {text}")
        out += ann_lines
    return out


def _join_annotations(clean_lines, annotations, style, indent_string, indent_level):
    """Place `annotations` on the same line as the declaration, or above it.

    Mirrors the legacy formatter: the joined line is used when it fits, unless
    the declaration is a `func` (a bare function always keeps its annotations on
    their own lines, except for `@abstract` on a function or `class_name`), or an
    annotation carries its own trailing comment.
    """
    if not clean_lines:
        return clean_lines
    first = clean_lines[0]
    target_text = first[1]
    stripped = target_text.strip()
    indent_prefix = target_text[: len(target_text) - len(target_text.lstrip())]
    texts = [annotation_to_flat_str(a) for a, _c in annotations]
    annotations_string = " ".join(texts)
    indent_width = len(indent_prefix.replace("\t", "    "))
    single_len = indent_width + len(annotations_string) + len(stripped)
    max_len = style.line_length if style else 100
    names = [annotation_name(a) for a, _c in annotations]
    has_own_comment = any(c for _a, c in annotations)
    is_abstract_func = (
        len(names) == 1 and names[0] == "abstract" and stripped.startswith("func")
    )
    is_abstract_class_name = (
        len(names) == 1 and names[0] == "abstract" and stripped.startswith("class_name")
    )
    enforces_standalone = (
        (stripped.startswith("func") or stripped.startswith("static func"))
        and not is_abstract_func
        and not is_abstract_class_name
    )
    if single_len <= max_len and not enforces_standalone and not has_own_comment:
        clean_lines[0] = (first[0], f"{indent_prefix}{annotations_string} {stripped}")
        return clean_lines
    return (
        _render_annotations_standalone(annotations, indent_string, indent_level, style)
        + clean_lines
    )


def format_block(  # noqa: C901
    statements: List[Any],
    statement_formatter: Callable,
    context_indent_string: str,
    context_indent_level: int,
    style: StyleConfig,
) -> tuple:
    """Format a block of items.

    Each item can be either a Tree node or a (tag, real_node) tuple from formatter.py.
    Comments embedded as children are handled by each component's statement_formatter.
    Blank lines between statements follow the StyleConfig rules.
    __comment__ tag items are emitted directly without going through statement_formatter.
    """
    formatted_lines: FormattedLines = []

    if not statements:
        return ([], context_indent_level)

    prev_end_line = get_line_safe(statements[0]) if statements else context_indent_level

    # Annotations that should end up on the same line as the declaration they
    # decorate are held here until that declaration is rendered. Inside a
    # function body `@warning_ignore` is never joined, so the scope matters.
    in_function_body = (
        getattr(statement_formatter, "__name__", "") == "format_func_statement"
    )
    pending_annotations: list = []
    # Spacing already emitted on behalf of the next declaration by a comment run.
    deferred = None

    # Per-item type info, for source-gap-aware blank lines: a blank line the
    # author wrote is preserved (squeezed to one), and a comment with no blank before
    # the next declaration attaches to it (the declaration's forced blanks go before it).
    # Comments and annotations attach to what follows, so they are transparent when
    # computing the forced blank lines between two real statements.
    item_types: list = []
    for item in statements:
        node = item[1] if isinstance(item, tuple) else item
        tag = item[0] if isinstance(item, tuple) else None
        if tag in ("__comment__", "__root_comment__") or tag == "annotation":
            item_types.append("comment" if tag != "annotation" else "annotation")
        else:
            item_types.append(_get_stmt_type_from_node(node))

    for i, item in enumerate(statements):
        actual_node = item
        # Optional inline comment to append (3-tuple format: tag, node, inline_comment)
        inline_comment = None

        # Check for comment marker first
        is_comment_item = False
        if isinstance(item, tuple):
            tag = item[0]
            actual_node = item[1]
            # Handle extended 3-tuple format: (tag, node, inline_comment)
            if len(item) >= 3:
                inline_comment = item[2] if item[2] else None
            if tag == "__comment__":
                is_comment_item = True
            elif tag == "__root_comment__":
                is_comment_item = True

        # Determine if we're at global scope (indent level 0 means top-level file items)
        is_nested = context_indent_level > 0

        # A prependable annotation is held back and joined onto the declaration
        # that follows. The first annotation of a run carries the spacing; the
        # rest of the run and the declaration itself emit none.
        if (
            not is_comment_item
            and isinstance(item, tuple)
            and item[0] == "annotation"
            and is_prependable_annotation(
                actual_node, _next_real_type(item_types, i), in_function_body
            )
        ):
            if not pending_annotations:
                ann_blanks, deferred = _blank_lines_before_item(
                    i,
                    statements,
                    item_types,
                    actual_node,
                    prev_end_line,
                    is_nested,
                    style,
                    deferred,
                )
                formatted_lines += ann_blanks
            pending_annotations.append((actual_node, inline_comment))
            continue

        # Add blank lines before all non-first items (comments included). When a
        # declaration is about to receive pending annotations, its spacing was
        # already emitted at the head of the annotation run.
        if pending_annotations:
            blank_lines = []
            deferred = None
        elif i > 0 and item_types[i - 1] == "annotation":
            # The item before this one is a *standalone* annotation (a prependable run would
            # have left `pending_annotations` set), and it already emitted the spacing on its
            # own behalf. Emitting the declaration's spacing again put it between the
            # annotation and the declaration it decorates, which detaches it:
            # `@warning_ignore("unused_signal")`, a blank line, then `signal ...`.
            blank_lines = []
            deferred = None
        else:
            blank_lines, deferred = _blank_lines_before_item(
                i,
                statements,
                item_types,
                actual_node,
                prev_end_line,
                is_nested,
                style,
                deferred,
            )
        formatted_lines += blank_lines

        if is_comment_item:
            # A comment interrupts the annotation run: the annotations can no
            # longer join the declaration, so emit them standalone first.
            if pending_annotations:
                formatted_lines += _render_annotations_standalone(
                    pending_annotations,
                    context_indent_string,
                    context_indent_level,
                    style,
                )
                pending_annotations = []
            # Emit the comment directly without going through statement_formatter
            text = _get_comment_text(actual_node)
            comment_indent = _comment_indent(
                actual_node,
                context_indent_string,
                is_trailing=all(
                    item_types[k] == "comment" for k in range(i + 1, len(item_types))
                ),
            )
            formatted_lines.append((None, f"{comment_indent}{text}"))

        # Only call statement_formatter for non-comment items
        if not is_comment_item:
            result = statement_formatter(
                actual_node, context_indent_string, context_indent_level, style
            )

            # Ensure we get a proper tuple back
            lines = []
            last_processed_line_no = prev_end_line

            if isinstance(result, list):
                lines = result  # Just lines, no line number
            elif isinstance(result, tuple) and len(result) == 2:
                lines, last_processed_line_no = result  # type: ignore

            # Filter out any non-tuple entries from lines (safety check)
            clean_lines = [ln for ln in lines if isinstance(ln, tuple)]

            # The declaration's own inline comment belongs on its line, so attach
            # it before any annotation join (which may reorder the first line). A
            # multiline declaration's last line is the one the comment was written on
            # (e.g. the closing bracket of a list), so it goes there.
            if inline_comment and clean_lines:
                text = clean_lines[-1][1]
                joined = f"{text}  {inline_comment}"
                # The guide prefers a comment on its own line when writing it inline would
                # push the line past the limit ("This is a long comment that would make the
                # line below too long if written inline... Inline comments are best used for
                # short comments"), so drop it to the next line whenever that fits.
                from .class_statement import _display_width

                limit = style.line_length if style else 100
                own_line = f"{context_indent_string}{inline_comment}"
                if _display_width(joined) > limit >= _display_width(own_line):
                    clean_lines[-1] = (clean_lines[-1][0], text)
                    clean_lines.append((None, own_line))
                else:
                    clean_lines[-1] = (clean_lines[-1][0], joined)

            # Join held-back annotations onto the rendered declaration (or emit
            # them standalone when the joined line would overflow).
            if pending_annotations:
                clean_lines = _join_annotations(
                    clean_lines,
                    pending_annotations,
                    style,
                    context_indent_string,
                    context_indent_level,
                )
                pending_annotations = []

            formatted_lines += clean_lines

            prev_end_line = (
                last_processed_line_no
                if last_processed_line_no and isinstance(last_processed_line_no, int)
                else get_end_line_safe(actual_node)
            )  # type: ignore

    return (formatted_lines, prev_end_line)


def _last_statement_type(formatted_lines: FormattedLines) -> str:
    """Determine the type of the last formatted statement based on its output.

    For multi-line statements (e.g., function declarations), checks ALL lines
    since the last line may be body content rather than a declaration keyword.

    Declaration-level keywords take priority over body content (like pass).
    """
    if not formatted_lines or not isinstance(formatted_lines[-1], tuple):
        return ""

    try:
        # Collect ALL matches from all lines, preferring STRONG declarations
        # over weak patterns. Iterate BACKWARDS so LATEST declaration wins
        # (e.g., function_decl header takes priority over pass in body).
        found_declarations = []
        found_weak = []

        for line_tuple in reversed(formatted_lines):
            stripped = str(line_tuple[1]).strip()

            if not stripped or stripped.startswith("#"):
                continue

            # Strong declaration patterns (declaration header level)
            if stripped.startswith("static func ") or stripped.startswith("func "):
                found_declarations.append("function_decl")
            elif stripped.startswith("class ") and not stripped.startswith(
                "class_name"
            ):
                found_declarations.append("inner_class")
            elif stripped.startswith("signal "):
                found_declarations.append("signal_decl")
            elif stripped.startswith("enum "):
                found_declarations.append("enum_decl")
            elif stripped.startswith("static var ") or stripped.startswith("var "):
                found_declarations.append("variable_decl")
            elif stripped.startswith("const "):
                found_declarations.append("constant_decl")
            elif stripped.startswith("extends"):
                found_declarations.append("extends_stmt")
            elif stripped.startswith("class_name"):
                found_declarations.append("class_name_stmt")

            # Weak patterns that may appear in function/class bodies
            if stripped == "pass":
                found_weak.append("pass_stmt")

        # Return the first strong declaration found (most specific match)
        if found_declarations:
            return found_declarations[0]

        # Fall back to weak patterns only if no declarations were found
        if found_weak:
            return "pass_stmt"
    except (IndexError, TypeError):
        pass

    return ""


def _get_blank_lines(
    prev_name: str,
    next_name: str,
    style: StyleConfig,
    prev_end: int,
    is_nested_scope: bool = False,
) -> FormattedLines:
    """Calculate blank lines needed between statements based on config.

    Rules derived from expected formatting output (blank_line_adding.out.gd):
    - Consecutive same-type at ANY scope: 0 blanks (grouping)
    - Different-type transitions global: 2 blanks for most, 1 for comment transitions
    - Comments inside nested scopes follow their position in source order
    """
    forced_blanks = 0

    # Top-level declaration types at file level
    TOP_LEVEL_TYPES = {
        "extends_stmt",
        "class_name_stmt",
        "variable_decl",
        "constant_decl",
        "function_decl",
        "signal_decl",
        "enum_decl",
        "inner_class",
        "abstract_func_decl",
    }
    at_file_scope = not is_nested_scope
    is_global_scope = at_file_scope and prev_name in TOP_LEVEL_TYPES

    # Definitions (functions/classes) are surrounded by 2 blank lines at file scope
    # (legacy global rule), on either side and regardless of the neighbouring kind. This
    # must not depend on `prev_name` being a declaration: a plain statement before a
    # definition (e.g. a top-level `pass` before a `class`) still gets the two blank
    # lines. A leading definition, and a comment before one (which has its own rule
    # below, one blank for a function), are excluded.
    definition_types = {"function_decl", "inner_class"}
    if (
        at_file_scope
        and prev_name not in ("", "comment")
        and (prev_name in definition_types or next_name in definition_types)
    ):
        return 2 * [(None, "")]

    # Nested (inside a function/class) definitions get one blank line around them. This
    # must precede the same-type/identical-kind rules, which would otherwise collapse a
    # func -> func (or class -> class) transition inside a class to zero blanks.
    if (
        prev_name not in ("", "comment")
        and not is_global_scope
        and (prev_name in definition_types or next_name in definition_types)
    ):
        return [(None, "")]

    # Helper: determine if both are the same structural type group
    def _same_type_group(p, n):
        """Check if two statement types belong to the same grouping."""
        var_types = {"variable_decl", "constant_decl"}
        func_types = {"function_decl"}
        class_types = {"inner_class"}

        return (
            (p in var_types and n in var_types)
            or (p in func_types and n in func_types)
            or (p in class_types and n in class_types)
        )

    # --- Same type transitions: 0 blanks at any scope ---
    if _same_type_group(prev_name, next_name):
        return []

    # --- Consecutive statements of the identical kind: 0 blanks at any scope ---
    # Covers statement kinds outside the grouped var/function/class sets above,
    # e.g. two `assert` calls in a function body or two empty `enum` declarations.
    if prev_name and prev_name == next_name:
        return []

    # --- Comment → same-type declaration: inherit preceding spacing ---
    if prev_name == "comment" and next_name in TOP_LEVEL_TYPES:
        # Comments at global scope follow a special rule:
        # comment before function → 1 blank; comment before class/variable → 2 blanks
        if next_name in {"function_decl", "signal_decl"}:
            forced_blanks = 1
        else:
            forced_blanks = 2

    # --- Different types at global scope (not involving comments as prev) ---
    elif is_global_scope and prev_name != "":

        # Variable/constant → function/signal: 2 blank lines
        if prev_name in {"variable_decl", "constant_decl"} and next_name in {
            "function_decl",
            "signal_decl",
        }:
            forced_blanks = 2

        # Variable/const → class: 2 blank lines
        elif (
            prev_name in {"variable_decl", "constant_decl"}
            and next_name == "inner_class"
        ):
            forced_blanks = 2

        # Function/signal → variable/constant: 2 blank lines
        elif prev_name in {"function_decl", "signal_decl"} and next_name in {
            "variable_decl",
            "constant_decl",
        }:
            forced_blanks = 2

        # Function/signal → class: 2 blank lines
        elif (
            prev_name in {"function_decl", "signal_decl"} and next_name == "inner_class"
        ):
            forced_blanks = 2

        # Class → function/signal: 1 blank line (not 2, matching expected output)
        elif prev_name == "inner_class" and next_name in {
            "function_decl",
            "signal_decl",
        }:
            forced_blanks = 1

        # extends_stmt transitions. The style guide's class example has one blank line
        # between `extends Node` and the first member; definitions still get two.
        elif prev_name == "extends_stmt":
            if next_name in {"function_decl", "inner_class"}:
                forced_blanks = 2
            elif next_name in {
                "variable_decl",
                "constant_decl",
                "signal_decl",
                "enum_decl",
            }:
                forced_blanks = 1

    # --- Different types in nested scope (inside functions/classes) ---
    elif not is_global_scope and prev_name != "":
        # In nested scopes only definitions (functions/classes) are separated from their
        # neighbours by a blank line; runs of simple statements (e.g. `extends` followed
        # by `pass`) stay grouped.
        if prev_name in definition_types or next_name in definition_types:
            forced_blanks = max(forced_blanks, 1)

    empty_line = [(None, "")]
    return forced_blanks * empty_line


def _is_top_level_transition(prev_name: str, next_name: str) -> bool:
    """Determine if this transition is at the top level of a file.

    Top-level transitions happen between declarations that start at indent level 0:
    extends_stmt, class_name_stmt, variable_decl, constant_decl, function_decl,
    signal_decl, enum_decl, inner_class. Comments don't count as statements.
    """
    top_level_types = {
        "extends_stmt",
        "class_name_stmt",
        "variable_decl",
        "constant_decl",
        "function_decl",
        "signal_decl",
        "enum_decl",
        "inner_class",
    }
    # Both previous and next should be known top-level types for this to be
    # considered a global scope transition
    return (
        prev_name in top_level_types
        or prev_name == "extends_stmt"
        or next_name in top_level_types
    )


def get_line_safe(node) -> int:
    try:
        if isinstance(node, tuple):
            _, actual = node  # type: ignore
            from ..common.utils import get_line as _gl

            return _gl(actual)
        else:
            from ..common.utils import get_line as _gl

            return _gl(node)
    except Exception:
        return 1


def _content_end_line(node) -> int:
    """Source end line of a node's real content.

    `meta.end_line` can overshoot past the content onto a trailing `_DEDENT` line, so
    walk the node and take the greatest line of its non-structural tokens instead."""
    from lark import Token as _Token

    best = get_line_safe(node)
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, _Token):
            if str(getattr(n, "type", "")).startswith("_"):
                continue
            ln = getattr(n, "line", None)
            if isinstance(ln, int):
                # A multi-line token (e.g. a triple-quoted string) reports its start
                # line, so add the newlines it contains to get its end line.
                ln += str(getattr(n, "value", "")).count("\n")
                best = max(best, ln)
            continue
        for c in getattr(n, "children", []):
            if c is not None:
                stack.append(c)
    return best


def get_end_line_safe(node) -> int:
    try:
        if isinstance(node, tuple):
            _, actual = node  # type: ignore
            from ..common.utils import get_end_line as _gel

            return _gel(actual)
        else:
            from ..common.utils import get_end_line as _gel

            return _gel(node)
    except Exception:
        return get_line_safe(node)


def _tree_data_str(node) -> str:
    from lark import Tree

    if isinstance(node, Tree):
        d = node.data
        return str(d) if not hasattr(d, "value") else d.value  # type: ignore
    return ""


def _get_stmt_type_from_node(node) -> str:
    """Get a statement type identifier from a raw tree node.

    Used by format_block to determine blank line rules based on the actual
    statement type (not just 'member' for all tuple-wrapped items).
    """
    if not hasattr(node, "data"):
        return ""
    data = str(_tree_data_str(node)).lower()
    # Map grammar node types to blank-line-friendly identifiers
    type_map = {
        "local_variable": "variable_decl",
        "variable_decl": "variable_decl",
        "constant_decl": "constant_decl",
        "function_decl": "function_decl",
        "signal_decl": "signal_decl",
        "enum_decl": "enum_decl",
        "inner_class": "inner_class",
        "extends_stmt": "extends_stmt",
        "class_name_stmt": "class_name_stmt",
        "pass_stmt": "pass_stmt",
    }
    # A body-less function (`@abstract func f()`, parsed with the optional suite)
    # is its own kind: the legacy formatter does not treat it as a definition for
    # blank-line purposes, so a run of abstract functions stays tightly spaced.
    if data == "function_decl":
        has_suite = any(
            hasattr(c, "data") and "suite" in str(_tree_data_str(c)).lower()
            for c in getattr(node, "children", [])
        )
        if not has_suite:
            return "abstract_func_decl"
    return type_map.get(data, data)


def _get_comment_text(comment_node) -> str:
    """Extract the text content from a comment tree node."""
    from .comments import normalize_comment

    for child in getattr(comment_node, "children", []):
        if hasattr(child, "value"):
            return normalize_comment(str(child.value))
    return ""


def _comment_column(comment_node) -> int:
    """1-based source column of a comment's `#`, or 0 when unknown."""
    from lark import Token as _Token

    for child in getattr(comment_node, "children", []):
        if isinstance(child, _Token):
            col = getattr(child, "column", None)
            if isinstance(col, int):
                return col
    meta = getattr(comment_node, "meta", None)
    col = getattr(meta, "column", None) if meta is not None else None
    return col if isinstance(col, int) else 0


def _comment_indent(comment_node, context_indent_string, is_trailing):
    """Indentation for a standalone comment inside a block.

    Mirrors the legacy formatter: a comment whose source line starts in column 0
    (the raw line begins with `#`) is emitted with no indentation; otherwise it
    takes the block indent. A trailing comment run that sits shallower than its
    block dedents to its own column, because it belongs to whatever follows the
    block (an `elif`/`else` clause, or the enclosing scope) rather than to it.
    """
    col = _comment_column(comment_node)
    if col == 1:
        return ""
    if not is_trailing or col <= 0:
        return context_indent_string
    source_indent = col - 1
    if source_indent < len(context_indent_string):
        return context_indent_string[:source_indent]
    return context_indent_string
