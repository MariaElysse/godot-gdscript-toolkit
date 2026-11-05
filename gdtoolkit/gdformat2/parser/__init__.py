"""The gdformat2 front end: its own grammar, parser and comment attachment.

`gdformat2` does not share the grammar with the legacy formatter. The legacy tooling
(`gdtoolkit.parser`, used by `gdtoolkit.formatter`, `gdtoolkit.linter` and `gdtoolkit.gd2py`)
keeps the LALR grammar and its tree shapes; gdformat2 parses with the Earley grammar here and
attaches comments into the tree, which the legacy tree has no place for.
"""

from .comments import attach_comments, CONTAINER_RULES  # noqa: F401
from .parser import parser  # noqa: F401
