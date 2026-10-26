"""LangChain integration for Claude Code, built on the Claude Agent SDK."""

from importlib import metadata

from claude_code_langchain.chat_models import (
    ChatClaudeCode,
    convert_messages,
    convert_to_claude_tool,
)
from claude_code_langchain.tools import (
    ALL_TOOLS,
    NETWORK_TOOLS,
    READ_ONLY_TOOLS,
    SHELL_TOOLS,
    WRITE_TOOLS,
    ClaudeTool,
    normalize_tools,
)

try:
    __version__ = metadata.version(__package__ or "claude-code-langchain")
except metadata.PackageNotFoundError:  # pragma: no cover, local checkouts
    __version__ = "0.0.0"

__all__ = [
    "ALL_TOOLS",
    "NETWORK_TOOLS",
    "READ_ONLY_TOOLS",
    "SHELL_TOOLS",
    "WRITE_TOOLS",
    "ChatClaudeCode",
    "ClaudeTool",
    "__version__",
    "convert_messages",
    "convert_to_claude_tool",
    "normalize_tools",
]
