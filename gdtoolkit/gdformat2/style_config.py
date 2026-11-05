from dataclasses import dataclass, field


@dataclass(frozen=True)
class StyleConfig:
    """Immutable formatting configuration for GDScript code."""

    # GDScript style guide recommends 79-char lines (PEP 8 compatible)
    line_length: int = 79
    tab_size: int = 4
    use_spaces: bool = False  # GDScript convention is tabs
    blank_lines_before_func_at_global: int = 2
    blank_lines_after_func_at_global: int = 2
    blank_lines_after_func_nested: int = 1
    blank_lines_before_class_at_global: int = 2
    blank_lines_after_class_at_global: int = 2
    blank_lines_after_class_nested: int = 1
    inline_comment_offset: int = 2  # spaces before inline comment
