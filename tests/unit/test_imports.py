import claude_code_langchain

EXPECTED = [
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


def test_all():
    assert sorted(claude_code_langchain.__all__) == sorted(EXPECTED)
    for name in EXPECTED:
        assert hasattr(claude_code_langchain, name)
