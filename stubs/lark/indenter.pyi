from typing import Any, Iterator, List

from lark.exceptions import LarkError
from lark.lexer import Token

class DedentError(LarkError): ...

class Indenter:
    NL_type: str
    OPEN_PAREN_types: List[str]
    CLOSE_PAREN_types: List[str]
    INDENT_type: str
    DEDENT_type: str
    tab_len: int
    paren_level: int
    indent_level: List[int]
    def __init__(self) -> None: ...
    def handle_NL(self, token: Token) -> Iterator[Token]: ...
    def _process(self, stream: Any) -> Iterator[Token]: ...
