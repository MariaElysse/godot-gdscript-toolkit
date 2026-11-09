from dataclasses import dataclass


@dataclass(frozen=True)
class StyleConfig:
    """Immutable formatting configuration for GDScript code."""

    # The guide's line length rule: "Keep individual lines of code under 100 characters."
    # This is the CLI's default as well; while it said 79 here, a library caller got
    # narrower output than `gdformat2` for the same input.
    line_length: int = 100
    tab_size: int = 4
    use_spaces: bool = False  # GDScript convention is tabs
    blank_lines_before_func_at_global: int = 2
    blank_lines_after_func_at_global: int = 2
    blank_lines_after_func_nested: int = 1
    blank_lines_before_class_at_global: int = 2
    blank_lines_after_class_at_global: int = 2
    blank_lines_after_class_nested: int = 1
    inline_comment_offset: int = 2  # spaces before inline comment
