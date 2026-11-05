import re
from typing import Iterator, List
from collections import defaultdict

from lark.indenter import Indenter
from lark.lexer import Token

_COMMENT_RE = re.compile(r"#[^\r\n]*")


class GDScriptIndenter(Indenter):
    NL_type = "_NL"
    OPEN_PAREN_types = ["PARENTHESIS_OPEN", "BRACKET_OPEN", "BRACE_OPEN"]
    CLOSE_PAREN_types = ["PARENTHESIS_CLOSE", "BRACKET_CLOSE", "BRACE_CLOSE"]
    LAMBDA_LINE_EXTENSION_types = ["IF", "WHILE", "FOR", "MATCH"]
    LAMBDA_SEPARATOR_types = ["COMMA"]
    INDENT_type = "_INDENT"
    DEDENT_type = "_DEDENT"
    # TODO: guess tab length
    tab_len = 4

    def __init__(self):
        super().__init__()
        self.processed_tokens = []
        self.undedented_lambdas_at_paren_level = defaultdict(int)
        self.comments: List[Token] = []

    # -- comment collection --------------------------------------------------

    def collect_comment(self, token: Token) -> Token:
        """Lexer callback for stand-alone (ignored) ``COMMENT`` tokens."""
        self.comments.append(
            Token("COMMENT", str(token), token.start_pos, token.line,
                  token.column, token.end_line, token.end_column, token.end_pos)
        )
        return token

    def _extract_comments(self, nl: Token) -> None:
        """Pull every ``#...`` comment out of an ``_NL`` token text."""
        text = str(nl)
        for m in _COMMENT_RE.finditer(text):
            off = m.start()
            before = text[:off]
            nl_count = before.count("\n")
            line = nl.line + nl_count
            if nl_count:
                column = off - before.rfind("\n")
            else:
                column = nl.column + off
            value = m.group()
            self.comments.append(
                Token("COMMENT", value, start_pos=nl.start_pos + off,
                      line=line, column=column, end_line=line,
                      end_column=column + len(value), end_pos=nl.start_pos + off + len(value))
            )

    # -- indentation ---------------------------------------------------------

    def handle_NL(self, token: Token) -> Iterator[Token]:
        self._extract_comments(token)  # collect comments from this newline block

        if self.paren_level > 0:
            # NL handling inside parens - required for multiline lambdas
            yield from self._handle_NL_in_parens(token)
            return

        yield token

        text = str(token)
        if "\n" not in text:
            # A comment at the very end of the file with no newline after it.
            return

        indent_str = text.rsplit("\n", 1)[1]
        if "#" in indent_str:
            # The last line is a comment that reaches EOF: no indentation info.
            return

        indent = indent_str.count(" ") + indent_str.count("\t") * self.tab_len

        if indent > self.indent_level[-1]:
            self.indent_level.append(indent)
            yield Token.new_borrow_pos(self.INDENT_type, indent_str, token)
        else:
            while indent < self.indent_level[-1]:
                self.indent_level.pop()
                yield Token.new_borrow_pos(self.DEDENT_type, indent_str, token)
            if indent != self.indent_level[-1]:
                raise DedentError(
                    "Unexpected dedent to column %s at line %s. Expected dedent to %s"
                    % (indent, token.line, self.indent_level[-1])
                )

    def process(self, stream):
        # pylint: disable=invalid-name
        """Reset comments before each parse run."""
        self.comments = []
        return super().process(stream)

    # -- original methods (unchanged) -----------------------------------------

    def _process(self, stream):
        self.processed_tokens = []
        self.undedented_lambdas_at_paren_level = defaultdict(int)

        had_newline = False
        for produced_token in super()._process(self._record_stream(stream)):
            if (
                produced_token.type in self.CLOSE_PAREN_types
                or produced_token.type in self.LAMBDA_SEPARATOR_types
            ):
                # dedenting all undedented lambas (more than one if nested) at current paren level
                while self.undedented_lambdas_at_paren_level[self.paren_level] > 0:
                    yield from self._dedent_lambda_at_token(had_newline, produced_token)
                    had_newline = False
            had_newline = produced_token.type == self.NL_type
            yield produced_token

    def _record_stream(self, stream):
        # A one-token lookahead, so `get`/`set` can be recognised as the contextual
        # accessor keyword from both neighbours.
        buffer = []
        prev_type = None
        for token in stream:
            buffer.append(token)
            if len(buffer) < 2:
                continue
            first, second = buffer[0], buffer[1]
            buffer = [second]
            emitted = self._maybe_accessor_keyword(prev_type, first, second)
            prev_type = emitted.type
            self.processed_tokens.append(emitted)
            yield emitted
        if buffer:
            self.processed_tokens.append(buffer[0])
            yield buffer[0]

    @staticmethod
    def _maybe_accessor_keyword(prev_type, token, next_token):
        """Rewrite `get`/`set` in the accessor slot into the ``GET_SET`` terminal.

        ``var x: set = _set`` declares a property, while ``var x: set`` is a variable of a
        type named ``set``; the parser cannot tell them apart from a ``NAME`` alone, and a
        postlexer forces lark's basic lexer, which has no lookahead. So the decision is
        made here, where both neighbours are visible: a ``get``/``set`` name right after a
        colon and followed by the accessor's ``=``, ``(`` or ``:``.
        """
        if (token.type == "NAME" and str(token.value) in ("get", "set")
                and prev_type == "COLON"
                and next_token.type in ("EQUAL", "PARENTHESIS_OPEN", "COLON")):
            return Token.new_borrow_pos("GET_SET", str(token.value), token)
        return token

    def _in_multiline_lambda(self):
        return self.undedented_lambdas_at_paren_level[self.paren_level] > 0

    # pylint: disable=invalid-name
    def _handle_NL_in_parens(self, token: Token):
        # Adapted from lark/indendeter.py as that normally disables line handling
        # when paren_level > 0.
        # NOTE: we never raise DedentError here as it doesn't make sense in parens
        indent_str = token.rsplit("\n", 1)[1]  # tabs and spaces
        indent = indent_str.count(" ") + indent_str.count("\t") * self.tab_len

        if indent > self.indent_level[-1] and (
            self._current_token_is_just_after_lambda_header()
            or self._in_multiline_lambda()
        ):
            self.indent_level.append(indent)
            self.undedented_lambdas_at_paren_level[self.paren_level] += 1
            yield token
            yield Token.new_borrow_pos(self.INDENT_type, indent_str, token)
        elif indent <= self.indent_level[-1] and self._in_multiline_lambda():
            yield token
            while indent < self.indent_level[-1] and self._in_multiline_lambda():
                self.indent_level.pop()
                self.undedented_lambdas_at_paren_level[self.paren_level] -= 1
                yield Token(self.DEDENT_type, None, None, token.line, None, token.line)
                # If we are still in a situation that can handle newlines, emit an extra
                # one with the same rationale as above
                if self._in_multiline_lambda():
                    yield token
            # Otherwise do nothing as other expressions don't handle newlines

    def _dedent_lambda_at_token(self, had_newline: bool, token: Token):
        self.indent_level.pop()
        self.undedented_lambdas_at_paren_level[self.paren_level] -= 1
        if not had_newline:
            yield Token.new_borrow_pos(self.NL_type, "N/A", token)
        yield Token.new_borrow_pos(self.DEDENT_type, "N/A", token)

    def _current_token_is_just_after_lambda_header(self):
        extra_rpars = [0]
        pattern_functions = [
            lambda t: t.type == "COLON",
            lambda t: t.type == "PARENTHESIS_CLOSE",
            lambda t: t.type == "PARENTHESIS_OPEN" and extra_rpars[0] == 0,
            lambda t: t.type == "FUNC",
        ]

        def lpar_accept_function(token: Token) -> bool:
            if token.type == "PARENTHESIS_CLOSE":
                extra_rpars[0] += 1
            elif token.type == "PARENTHESIS_OPEN":
                if extra_rpars[0] <= 0:
                    return False
                extra_rpars[0] -= 1
            return True

        accept_functions = [
            lambda t: t.type == "_NL",
            # Skip the declaration's return type, which sits between the header's
            # COLON and the closing parenthesis: `[FORWARD_ARROW return_type]`, where
            # `return_type: type | VOID` and `type: NAME (PERIOD NAME)* [...]`.
            lambda t: (
                t.type in ["_NL", "FORWARD_ARROW", "VOID", "NAME", "PERIOD",
                           "BRACKET_OPEN", "BRACKET_CLOSE", "COMMA"]
                or t.value == "->"
            ),
            lpar_accept_function,
            lambda t: t.type in ["_NL", "NAME"],
        ]
        i = 0
        for processed_token in reversed(self.processed_tokens):
            if i >= len(pattern_functions):
                return True
            if pattern_functions[i](processed_token):
                i += 1
                continue
            if not accept_functions[i](processed_token):
                return False
        return i >= len(pattern_functions)
