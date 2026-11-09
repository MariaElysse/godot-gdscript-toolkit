"""
Initializes and caches the GDScript parsers, using Lark.
Provides a function to parse GDScript code
and to get an intermediate representation as a Lark Tree.
Comments are collected by the postlexer and attached into the tree at their
correct positions via :py:func:`attach_comments`.
"""
import os
from typing import Any, Dict

from lark import Lark, Tree

from .indenter import GDScriptIndenter
from .comments import attach_comments


# TODO: when upgrading to Python 3.8, replace with functools.cached_property
# pylint: disable=too-few-public-methods
class CachedProperty:
    """A property that is only computed once per instance and then replaces
    itself with an ordinary attribute. Deleting the attribute resets the
    property.
    """

    def __init__(self, func):
        self.__doc__ = func.__doc__
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class Parser:
    """Parses GDScript code using lark parsers.
    The parsers are only created once, upon using them for the first time.

    Comments collected by the postlexer (:py:attr:`indenter.comments`) are
    automatically attached into the parse tree as ``Tree("comment", [...])``
    nodes when :py:meth:`parse` is called with *gather_metadata* (position
    info) enabled.
    """

    def __init__(self):
        self._directory = os.path.dirname(__file__)
        self._use_grammar_cache = True
        self._cache_dirpath: str = os.path.join(_get_cache_directory(), "gdtoolkit")
        self.indenter = GDScriptIndenter()  # shared postlexer

    def parse(self, code: str) -> Tree:
        """Parses GDScript code and returns intermediate representation as a Lark Tree.

        Comments are attached into the tree at their correct positions.
        This requires ``propagate_positions=True`` which is always used now.
        """
        adjusted_code = code + "\n"
        tree = self._parser.parse(adjusted_code)
        _merge_inline_class_body_members(tree)
        # Attach comments collected by the postlexer
        if self.indenter.comments:
            attach_comments(tree, list(self.indenter.comments))
        return tree

    def parse_comments(self, code: str) -> Tree:
        """Parses GDScript code and returns comments - both standalone, and inline."""
        # pylint: disable=no-member
        return self._comment_parser.parse(code)

    def disable_grammar_caching(self) -> None:
        self._use_grammar_cache = False

    def _get_parser(
        self,
        name: str,
        grammar_filename: str = "gdscript.lark",
    ) -> Lark:
        # Earley doesn't support caching — skip cache dir entirely
        grammar_filepath: str = os.path.join(self._directory, grammar_filename)

        # Earley doesn't support caching — skip cache dir entirely
        # Match GDScriptParser pattern exactly for lexer/callback setup
        options: Dict[str, Any] = {
            "parser": "earley",
            "ambiguity": "resolve",
            "propagate_positions": True,
            "maybe_placeholders": True,
        }
        callbacks: dict = {}
        callbacks["COMMENT"] = self.indenter.collect_comment
        options.update(
            {
                "postlex": self.indenter,
                "lexer_callbacks": callbacks,
            }
        )

        a_parser = Lark.open(
            grammar_filepath,
            start="start",
            regex=True,
            **options,
        )

        return a_parser

    @CachedProperty
    def _parser(self) -> Lark:
        return self._get_parser("parser")

    @CachedProperty
    def _comment_parser(self) -> Lark:
        return self._get_parser("parser_comments", grammar_filename="comments.lark")


def _rule_name(node) -> str:
    if not isinstance(node, Tree):
        return ""
    data = node.data
    # lark's `Tree.data` is a plain str, so the `.value` branch is defensive
    # dead code - guarded by `hasattr`, never taken at runtime.
    return str(data) if not hasattr(data, "value") else data.value  # type: ignore[attr-defined]


def _start_line(node) -> int:
    meta = getattr(node, "meta", None)
    line = getattr(meta, "line", None)
    return int(line) if line else 0


def _merge_inline_class_body_members(tree: Tree) -> None:
    """Move a one-line class body's later members inside the class.

    In ``class Foo: var a; var b`` the first member's own optional ``end`` can take the
    semicolon, which leaves the later members as declarations *next to* the class rather
    than in it; the formatter then drops them. They sit on the class's own source line,
    so move them into its ``inline_class_body``.
    """
    for node in list(tree.iter_subtrees()):
        children = getattr(node, "children", None)
        if not children:
            continue
        index = 0
        while index < len(children):
            candidate = children[index]
            inner = None
            if _rule_name(candidate) == "inner_class":
                inner = candidate
            elif _rule_name(candidate) == "class_member":
                inner = next(
                    (c for c in candidate.children if _rule_name(c) == "inner_class"),
                    None,
                )
            if inner is None:
                index += 1
                continue
            body = next(
                (c for c in inner.children if _rule_name(c) == "inline_class_body"),
                None,
            )
            if body is None:
                index += 1
                continue
            class_line = _start_line(inner)
            while index + 1 < len(children):
                sibling = children[index + 1]
                if (
                    _rule_name(sibling) != "class_member"
                    or _start_line(sibling) != class_line
                ):
                    break
                children.pop(index + 1)
                body.children.append(sibling)
            index += 1


def _get_cache_directory() -> str:
    """Returns the cache directory based on the user's operating system"""
    import sys

    if sys.platform in ["linux", "linux2"]:
        return os.path.join(os.path.expanduser("~"), ".cache")
    elif sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Caches")
    elif sys.platform == "win32":
        return os.path.expandvars(r"%LOCALAPPDATA%")
    return ""


parser = Parser()
