# claude-code-langchain

[![PyPI version](https://img.shields.io/pypi/v/claude-code-langchain.svg)](https://pypi.org/project/claude-code-langchain/)
[![Python versions](https://img.shields.io/pypi/pyversions/claude-code-langchain.svg)](https://pypi.org/project/claude-code-langchain/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/moesio-f/claude-code-langchain/blob/main/LICENSE)

A LangChain chat model that runs Claude through the **Claude Code CLI**, using the
[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python).

> **Not affiliated with Anthropic.** This is an independent community project.
> "Anthropic", "Claude", and "Claude Code" are trademarks of Anthropic PBC.

Authentication is delegated entirely to the `claude` CLI, so the model runs with whatever
the CLI is already configured with. For products or services that serve other users,
Anthropic requires API-key authentication — see the
[Claude Code legal and compliance page](https://code.claude.com/docs/en/legal-and-compliance).

> Built on the ideas of [ChamaruAmasara/langchain-claude-code](https://github.com/ChamaruAmasara/langchain-claude-code),
> rewritten for the current stack: the new `claude-agent-sdk` (replacing the deprecated
> `claude-code-sdk`) and LangChain 1.0+.

## What the rewrite changes

| Area | Original (`claude-code-sdk`) | Here (`claude-agent-sdk`) |
|---|---|---|
| SDK options | `ClaudeCodeOptions` | `ClaudeAgentOptions` |
| Tool calling | Tool schemas injected into the system prompt, JSON parsed out of the reply | Real Anthropic tool use: tools are served over an in-process MCP server, so `AIMessage.tool_calls` comes from actual `tool_use` blocks |
| Structured output | Prompted, best-effort | The CLI's `--json-schema` output format, validated against your schema |
| Extended thinking | A "think step by step" hint appended to the prompt | The SDK's native `thinking` config, returned as a `reasoning` content block |
| Effort | `--effort` via `extra_args` | First-class `effort` option, `low` → `max` |
| Token usage | Not available | `usage_metadata`, including cache reads/writes, plus cost in `response_metadata` |
| Built-in tools | Implicitly on, gated by `max_turns` | Explicit `builtin_tools`; off by default, so plain chat stays plain |
| LangChain | 0.3 | 1.0+ (v1 content blocks, `usage_metadata`, standard `bind_tools` / `with_structured_output`) |

## Requirements

- Python 3.10+
- Node.js 18+ and the Claude Code CLI: `npm install -g @anthropic-ai/claude-code && claude`
- The CLI authenticated with the method you intend to use (API key, Claude subscription, or
  a supported cloud provider)

```bash
pip install claude-code-langchain
```

## Quick start

```python
from claude_code_langchain import ChatClaudeCode

llm = ChatClaudeCode(model="claude-opus-5")
print(llm.invoke("What is the capital of France?").text)
```

`model` accepts a model id (`claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5`) or a CLI
alias (`opus`, `sonnet`, `haiku`). Leave it unset to use whatever the CLI is configured with.

### Messages, streaming, async

```python
from langchain_core.messages import HumanMessage, SystemMessage

llm.invoke([SystemMessage("Be concise."), HumanMessage("Explain generators.")])

for chunk in llm.stream("Count from 1 to 5"):
    print(chunk.text, end="", flush=True)

await llm.ainvoke("Hello")  # also: astream, batch, abatch
```

### Tool calling

```python
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return "12C and sunny"


response = llm.bind_tools([get_weather]).invoke("What is the weather in Paris?")
response.tool_calls
# [{'name': 'get_weather', 'args': {'city': 'Paris'}, 'id': 'toolu_…', 'type': 'tool_call'}]
```

Your tools are exposed to the CLI over an in-process MCP server, but they are never executed
there — the call is handed back to you, exactly like any other LangChain chat model, so it
works with LangGraph's `create_agent` and the usual `ToolNode` loop.

### Structured output

```python
from pydantic import BaseModel


class Person(BaseModel):
    """A person."""

    name: str
    age: int


llm.with_structured_output(Person).invoke("Bob is 42.")
# Person(name='Bob', age=42)
```

This uses the CLI's native JSON-schema output mode, so the response is constrained rather
than merely requested. Pass `include_raw=True` to get `{"raw", "parsed", "parsing_error"}`.

### Thinking and effort

```python
llm = ChatClaudeCode(
    model="claude-opus-5",
    thinking={"type": "adaptive", "display": "summarized"},
    effort="high",
)
message = llm.invoke("Prove that sqrt(2) is irrational.")
message.content  # [{'type': 'reasoning', ...}, {'type': 'text', ...}]
message.text  # just the answer
```

### Agentic mode

Claude Code's own tools are **off by default**, so the model behaves like a normal chat model.
Turn them on when you want an agent that can touch your filesystem:

```python
from claude_code_langchain import READ_ONLY_TOOLS, ChatClaudeCode, ClaudeTool

reader = ChatClaudeCode(
    builtin_tools=READ_ONLY_TOOLS,  # Read, Glob, Grep
    allowed_tools=READ_ONLY_TOOLS,  # run without asking for permission
    max_turns=10,
    cwd="/path/to/project",
)
reader.invoke("Find every TODO comment in this project")

full = ChatClaudeCode(
    builtin_tools="claude_code",  # the complete Claude Code toolset
    permission_mode="acceptEdits",
    max_turns=20,
    cwd="/path/to/project",
)
```

Presets: `READ_ONLY_TOOLS`, `WRITE_TOOLS`, `SHELL_TOOLS`, `NETWORK_TOOLS`, `ALL_TOOLS`, plus the
`ClaudeTool` enum for individual names.

### Sessions

Every response carries the CLI session id, which you can resume later:

```python
first = llm.invoke("Remember the number 7.")
session = first.response_metadata["session_id"]

resumed = ChatClaudeCode(session_id=session).invoke("What number did I mention?")
```

## Configuration

| Option | Purpose |
|---|---|
| `model`, `system_prompt` | Model id/alias; system prompt used when no `SystemMessage` is present |
| `thinking`, `effort` | Extended thinking config; `low` / `medium` / `high` / `xhigh` / `max` |
| `max_turns`, `max_budget_usd` | Agent loop limits (defaults to a single turn) |
| `builtin_tools`, `allowed_tools`, `disallowed_tools`, `permission_mode` | Which Claude Code tools exist, which run unattended, and how permissions are handled |
| `mcp_servers` | Additional MCP servers for the session |
| `cwd`, `add_dirs`, `env`, `cli_path`, `settings`, `setting_sources` | Where and how the CLI subprocess runs |
| `session_id`, `fork_session` | Resume or branch a previous session |
| `extra_args` | Any other CLI flag |

`setting_sources` defaults to `[]` so a run is not affected by your local Claude settings or
`CLAUDE.md`. Pass `None` to load user, project and local settings like the interactive CLI.

## Compared with `ChatAnthropic`

| Feature | `ChatAnthropic` | `ChatClaudeCode` |
|---|---|---|
| `invoke` / `stream` / `batch` / async | ✅ | ✅ |
| System messages, image input | ✅ | ✅ |
| Tool calling, structured output | ✅ | ✅ |
| Extended thinking, effort | ✅ | ✅ |
| Token usage and cost | ✅ | ✅ (plus USD cost per call) |
| Authentication | API key | Delegated to the `claude` CLI |
| Built-in filesystem/bash/web tools, MCP servers, subagents, skills | ❌ | ✅ |
| `temperature`, `top_p`, `top_k`, `stop` | ✅ | Accepted for compatibility, ignored by the CLI |
| Multi-turn history | Sent verbatim | Flattened into one prompt (see below) |

### Known limitations

- The CLI owns conversation state, so a message list is flattened into a single prompt with
  labelled turns. Prior tool calls and results are rendered as text. Use `session_id` when you
  want the CLI's own memory of a conversation instead.
- Sampling parameters and stop sequences are not exposed by the CLI.
- Streaming yields text only; thinking arrives on the final non-streamed message.
- Each call spawns a `claude` subprocess, so latency is higher than a direct API call.

## Development

```bash
uv sync --group dev
uv run pytest                      # unit tests, no CLI needed
uv run ruff check . && uv run mypy src
CLAUDE_CODE_LANGCHAIN_INTEGRATION=1 uv run pytest tests/integration   # runs the CLI
```

## Related projects

Other LangChain adapters for Claude Code:

| Project | Notes |
|---|---|
| [ChamaruAmasara/langchain-claude-code](https://github.com/ChamaruAmasara/langchain-claude-code) | The original project this one was inspired by; on PyPI as `langchain-claude-code-cli` |
| [thehumanworks/langchain-claude-code](https://github.com/thehumanworks/langchain-claude-code) | Wrapper for the Claude Agent SDK; on PyPI as `langchain-claude-code` |
| [brhodes/langchain-claude-agent-sdk](https://codeberg.org/brhodes/langchain-claude-agent-sdk) | `langchain-claude-agent-sdk` on PyPI (hosted on Codeberg rather than GitHub) |
| [langchain-ai/langchain#40131](https://github.com/langchain-ai/langchain/issues/40131) | Proposal to add an Agent SDK-backed model to `langchain-anthropic` |

Each takes a different approach to authentication and conversation state. Some accept OAuth
tokens directly; review the auth method and Anthropic's terms before using one in production.

## Credits

Original project: [langchain-claude-code](https://github.com/ChamaruAmasara/langchain-claude-code)
by Chamaru Amasara. This package is an independent rewrite on the Claude Agent SDK and
LangChain 1.0+.
