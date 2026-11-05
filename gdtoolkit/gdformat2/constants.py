"""Constants for gdformat2.

Default formatting rules that StyleConfig uses as baseline values.
These mirror the original formatter's DEFAULT_CONFIG but are used only
for config file defaults (CLI args override them).
"""
from types import MappingProxyType


TAB_INDENT_SIZE = 4
INLINE_COMMENT_OFFSET = 2

DEFAULT_SURROUNDING_EMPTY_LINES_TABLE = MappingProxyType({
    "function_decl": 1,
})

GLOBAL_SCOPE_SURROUNDING_EMPTY_LINES_TABLE = MappingProxyType({
    "function_decl": 2,
    "inner_class": 2,
})
