from typing import Any, Optional

class Lark:
    def __init__(self, grammar: Any, **options): ...
    @classmethod
    def open(
        cls: Any, grammar_filename: str, rel_to: Optional[str] = ..., **options
    ) -> Lark: ...
