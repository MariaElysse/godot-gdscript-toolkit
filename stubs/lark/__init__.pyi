from typing import Any, Callable, Iterator, List, Optional

from lark.exceptions import UnexpectedInput as UnexpectedInput
from lark.lark import Lark as Lark
from lark.lexer import Token as Token
from lark.tree import Meta as Meta

Node = Any  # TODO: use one from gdtoolkit and fix accordingly

Discard: Exception

class Tree:
    children: List[Node]
    line: int
    value: Any  # TODO: remove and fix accordingly
    column: int
    end_line: int
    data: str
    meta: Meta
    def __init__(self, name: str, children: List[Node], meta: Optional[Meta] = ...): ...
    def iter_subtrees(self): ...
    def find_pred(
        self, pred: Callable[[Tree], bool]
    ) -> Iterator[Tree]: ...
    def find_data(self, data: str) -> Iterator[Tree]: ...
    def pretty(self, indent_str: str = ...) -> str: ...

class Transformer:
    def transform(self, tree: Tree) -> Tree: ...

__version__: str
