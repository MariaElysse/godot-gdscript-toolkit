from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExpressionContext:
    """Expression formatting context tracking prefix/suffix strings with their line positions."""

    prefix_string: str
    prefix_line: Optional[int]  # earliest line number of prefix string
    suffix_string: str
    suffix_line: Optional[int]  # earliest line number of suffix string
