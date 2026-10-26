from claude_code_langchain import (
    ALL_TOOLS,
    READ_ONLY_TOOLS,
    ClaudeTool,
    normalize_tools,
)


def test_enum_values_are_cli_tool_names():
    assert ClaudeTool.READ.value == "Read"
    assert ClaudeTool.WEB_SEARCH.value == "WebSearch"
    assert len(ALL_TOOLS) == len(set(ALL_TOOLS))


def test_normalize_tools_mixes_strings_and_enums_and_dedupes():
    assert normalize_tools([*READ_ONLY_TOOLS, "Read", "Bash"]) == [
        "Read",
        "Glob",
        "Grep",
        "Bash",
    ]
