from typing import Any, Optional

class Token(str):
    type: str
    value: Any
    line: int
    column: int
    end_line: int
    end_column: int
    start_pos: int
    end_pos: int
    def __new__(
        cls,
        type_: str,
        value: Any,
        start_pos: Optional[int] = ...,
        line: Optional[int] = ...,
        column: Optional[int] = ...,
        end_line: Optional[int] = ...,
        end_column: Optional[int] = ...,
        end_pos: Optional[int] = ...,
    ) -> Token: ...
    @classmethod
    def new_borrow_pos(cls, type_: str, value: Any, borrow_from: Token) -> Token: ...
