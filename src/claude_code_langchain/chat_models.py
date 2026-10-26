import asyncio
import json
import queue
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from operator import itemgetter
from typing import Any, Literal, cast

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SdkMcpTool,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
)
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import ToolCall
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field

from claude_code_langchain.tools import ClaudeTool, normalize_tools

MCP_SERVER_NAME = "langchain"
"""Name of the in-process MCP server used to expose ``bind_tools`` tools."""

TOOL_PREFIX = f"mcp__{MCP_SERVER_NAME}__"
"""Prefix the CLI gives to tools served by the in-process MCP server."""

_TOOL_HELD = (
    "The tool call was captured by the host application. Stop here; the result "
    "will be provided in a later turn."
)


def _image_block(item: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a LangChain image block to an Anthropic image block."""
    if item.get("type") == "image_url":  # LangChain v0 style
        image = item.get("image_url", {})
        url = image if isinstance(image, str) else image.get("url", "")
        if url.startswith("data:"):
            header, _, data = url.partition(",")
            media_type = header.split(":", 1)[-1].split(";")[0]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            }
        return {"type": "image", "source": {"type": "url", "url": url}}

    if item.get("type") != "image":
        return None
    if "source" in item:  # already an Anthropic block
        return item
    if item.get("base64"):  # LangChain v1 standard block
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": item.get("mime_type", "image/png"),
                "data": item["base64"],
            },
        }
    if item.get("url"):
        return {"type": "image", "source": {"type": "url", "url": item["url"]}}
    return None


def _message_text(message: BaseMessage) -> str:
    """Best-effort plain text for a message, ignoring non-text blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "".join(parts)


def _label(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "Human"
    if isinstance(message, AIMessage):
        return "Assistant"
    if isinstance(message, ToolMessage):
        return "Tool result"
    return message.type.capitalize()


def _render_turn(message: BaseMessage) -> str:
    """Render a non-system message as a labelled transcript entry."""
    text = _message_text(message)
    if isinstance(message, AIMessage) and message.tool_calls:
        calls = "\n".join(
            f"[called tool {call['name']} with {json.dumps(call['args'])}]"
            for call in message.tool_calls
        )
        text = f"{text}\n{calls}".strip()
    if isinstance(message, ToolMessage):
        return f"Tool result for {message.tool_call_id}:\n{text}"
    return f"{_label(message)}: {text}"


def convert_messages(
    messages: Sequence[BaseMessage],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert LangChain messages to a system prompt plus one user message.

    The Claude Agent SDK drives a stateful CLI session rather than a stateless
    chat completion, so the conversation is flattened into a single user turn.
    Returns ``(system_prompt, content_blocks)``.
    """
    system_parts: list[str] = []
    turns: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            system_parts.append(_message_text(message))
        else:
            turns.append(message)

    images: list[dict[str, Any]] = []
    for message in turns:
        if isinstance(message.content, list):
            for item in message.content:
                if isinstance(item, dict):
                    block = _image_block(item)
                    if block is not None:
                        images.append(block)

    if len(turns) == 1 and isinstance(turns[0], HumanMessage):
        text = _message_text(turns[0])
    else:
        text = "\n\n".join(_render_turn(message) for message in turns)

    content: list[dict[str, Any]] = [*images, {"type": "text", "text": text}]
    return ("\n\n".join(p for p in system_parts if p) or None, content)


def convert_to_claude_tool(
    tool: dict[str, Any] | type | Callable[..., Any] | BaseTool,
) -> dict[str, Any]:
    """Convert a LangChain tool to an Anthropic-style tool schema."""
    if isinstance(tool, dict) and "input_schema" in tool:
        return tool
    function = convert_to_openai_tool(tool)["function"]
    schema = function.get("parameters") or {"type": "object", "properties": {}}
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "input_schema": schema,
    }


def _json_schema(schema: dict[str, Any] | type) -> dict[str, Any]:
    """Extract a plain JSON schema from a dict, TypedDict or Pydantic model."""
    if isinstance(schema, dict) and schema.get("type") == "object":
        return schema
    return convert_to_claude_tool(schema)["input_schema"]


def _mcp_server(schemas: list[dict[str, Any]]) -> Any:
    """Build an in-process MCP server exposing the bound tool schemas.

    The handlers are only reached if the CLI executes a call before the caller
    stops consuming the stream; they never produce a real result, since tool
    execution belongs to the LangChain caller.
    """

    def make(schema: dict[str, Any]) -> SdkMcpTool[Any]:
        async def handler(_args: Any) -> dict[str, Any]:
            return {"content": [{"type": "text", "text": _TOOL_HELD}]}

        return SdkMcpTool(
            name=schema["name"],
            description=schema.get("description", ""),
            input_schema=schema["input_schema"],
            handler=handler,
        )

    return create_sdk_mcp_server(MCP_SERVER_NAME, tools=[make(s) for s in schemas])


def _usage_metadata(usage: dict[str, Any] | None) -> UsageMetadata | None:
    if not usage:
        return None
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_creation = usage.get("cache_creation_input_tokens", 0) or 0
    input_tokens = (usage.get("input_tokens", 0) or 0) + cache_read + cache_creation
    output_tokens = usage.get("output_tokens", 0) or 0
    metadata: UsageMetadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cache_read or cache_creation:
        metadata["input_token_details"] = {
            "cache_read": cache_read,
            "cache_creation": cache_creation,
        }
    return metadata


class _Turn:
    """Everything collected from one Claude Agent SDK query."""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.thinking: list[str] = []
        self.tool_calls: list[ToolCall] = []
        self.result: ResultMessage | None = None
        self.usage: dict[str, Any] | None = None
        self.model: str | None = None
        self.session_id: str | None = None

    def add_assistant(self, message: AssistantMessage) -> None:
        """Record an assistant message."""
        if self.tool_calls:
            return
        self.model = message.model or self.model
        self.usage = message.usage or self.usage
        self.session_id = message.session_id or self.session_id
        for block in message.content:
            if isinstance(block, TextBlock):
                self.text.append(block.text)
            elif isinstance(block, ThinkingBlock) and block.thinking:
                self.thinking.append(block.thinking)
            elif isinstance(block, ToolUseBlock) and block.name.startswith(TOOL_PREFIX):
                self.tool_calls.append(
                    create_tool_call(
                        name=block.name[len(TOOL_PREFIX) :],
                        args=block.input,
                        id=block.id,
                    )
                )

    def add_result(self, message: ResultMessage) -> None:
        self.result = message
        if not self.tool_calls:
            self.usage = message.usage or self.usage

    def to_message(self) -> AIMessage:
        text = "".join(self.text)
        content: str | list[str | dict[Any, Any]] = text
        if self.thinking:
            content = [
                *(
                    {"type": "reasoning", "reasoning": thought}
                    for thought in self.thinking
                ),
                {"type": "text", "text": text},
            ]

        metadata: dict[str, Any] = {
            "model_name": self.model,
            "session_id": self.session_id,
        }
        extra: dict[str, Any] = {}
        result = self.result
        if result is not None:
            metadata.update(
                session_id=result.session_id,
                num_turns=result.num_turns,
                duration_ms=result.duration_ms,
                is_error=result.is_error,
                stop_reason=result.stop_reason,
                total_cost_usd=result.total_cost_usd,
            )
            if result.structured_output is not None:
                extra["structured_output"] = result.structured_output

        return AIMessage(
            content=content,
            tool_calls=self.tool_calls,
            response_metadata={k: v for k, v in metadata.items() if v is not None},
            additional_kwargs=extra,
            usage_metadata=_usage_metadata(self.usage),
        )


def _run_sync(factory: Callable[[], Any]) -> Any:
    """Run a coroutine from sync code, even inside a running event loop."""
    result: list[Any] = [None]
    error: list[BaseException | None] = [None]

    def target() -> None:
        try:
            result[0] = asyncio.run(factory())
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            error[0] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join()
    if error[0] is not None:
        raise error[0]
    return result[0]


def _iter_sync(factory: Callable[[], AsyncIterator[Any]]) -> Iterator[Any]:
    """Drain an async iterator from sync code via a worker thread."""
    done = object()
    items: queue.Queue[Any] = queue.Queue()

    async def pump() -> None:
        try:
            async for item in factory():
                items.put(item)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            items.put(exc)
        finally:
            items.put(done)

    thread = threading.Thread(target=lambda: asyncio.run(pump()), daemon=True)
    thread.start()
    try:
        while True:
            item = items.get()
            if item is done:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        thread.join()


class ChatClaudeCode(BaseChatModel):
    """Claude models served through the Claude Code CLI (Claude Agent SDK).

    Requires the ``claude`` CLI to be installed and authenticated. Authentication is
    delegated to the CLI.

    Setup:
        .. code-block:: bash

            npm install -g @anthropic-ai/claude-code
            claude   # sign in once
            pip install claude-code-langchain

    Basic use:
        .. code-block:: python

            from claude_code_langchain import ChatClaudeCode

            llm = ChatClaudeCode(model="claude-opus-5")
            llm.invoke("What is the capital of France?")

    Tool calling:
        .. code-block:: python

            llm.bind_tools([get_weather]).invoke("Weather in Paris?").tool_calls

    Structured output:
        .. code-block:: python

            llm.with_structured_output(Person).invoke("Bob is 42.")

    Agentic mode (Claude Code's own tools):
        .. code-block:: python

            from claude_code_langchain import READ_ONLY_TOOLS, ClaudeTool

            agent = ChatClaudeCode(
                builtin_tools=READ_ONLY_TOOLS,
                allowed_tools=READ_ONLY_TOOLS,
                max_turns=10,
                cwd="/path/to/project",
            )
            agent.invoke("Find every TODO comment in this project")
    """

    model: str | None = None
    """Model id or alias (e.g. ``"claude-opus-5"``, ``"sonnet"``).

    ``None`` uses whatever model the CLI is configured to use.
    """

    system_prompt: str | dict[str, Any] | None = None
    """System prompt used when the messages contain no ``SystemMessage``.

    Also accepts the SDK preset form,
    ``{"type": "preset", "preset": "claude_code"}``.
    """

    thinking: dict[str, Any] | None = None
    """Extended thinking config, e.g. ``{"type": "adaptive"}``."""

    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    """How much effort Claude spends on the response."""

    max_turns: int | None = None
    """Maximum agent turns per call. Defaults to 1 (a single model response)."""

    max_budget_usd: float | None = None
    """Stop the query once this much has been spent."""

    builtin_tools: list[str | ClaudeTool] | Literal["claude_code"] | None = None
    """Claude Code tools made available to the model.

    ``None`` (default) disables all built-in tools, giving plain chat behaviour.
    ``"claude_code"`` enables the full Claude Code preset.
    """

    allowed_tools: list[str | ClaudeTool] | None = None
    """Tools that may run without asking for permission."""

    disallowed_tools: list[str | ClaudeTool] | None = None
    """Tools that are removed from the model's context entirely."""

    permission_mode: (
        Literal["default", "acceptEdits", "plan", "bypassPermissions", "dontAsk"] | None
    ) = None
    """Permission mode for tool execution."""

    mcp_servers: dict[str, Any] = Field(default_factory=dict)
    """Extra MCP servers to expose to the session."""

    cwd: str | None = None
    """Working directory of the CLI session."""

    add_dirs: list[str] = Field(default_factory=list)
    """Additional directories the session may access."""

    cli_path: str | None = None
    """Path to the ``claude`` executable. Defaults to the bundled CLI."""

    settings: str | None = None
    """Path to an extra settings JSON file."""

    setting_sources: list[Literal["user", "project", "local"]] | None = Field(
        default_factory=list
    )
    """Filesystem settings to load. Defaults to ``[]`` (isolated); ``None``
    loads user, project and local settings like the interactive CLI."""

    env: dict[str, str] = Field(default_factory=dict)
    """Environment variables for the CLI subprocess."""

    extra_args: dict[str, str | None] = Field(default_factory=dict)
    """Extra CLI flags, e.g. ``{"verbose": None}``."""

    session_id: str | None = None
    """Resume a previous CLI session by id."""

    fork_session: bool = False
    """When resuming, branch into a new session instead of continuing."""

    streaming: bool = False
    """Stream by default when the caller does not ask for it explicitly."""

    max_tokens: int | None = None
    """Accepted for ``ChatAnthropic`` compatibility; the CLI manages this."""

    temperature: float | None = None
    """Accepted for ``ChatAnthropic`` compatibility; not supported by the CLI."""

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "effort": self.effort,
            "thinking": self.thinking,
            "max_turns": self.max_turns,
            "permission_mode": self.permission_mode,
            "builtin_tools": self.builtin_tools,
        }

    def _build_options(
        self,
        *,
        system_prompt: str | None,
        partial_messages: bool = False,
        **kwargs: Any,
    ) -> ClaudeAgentOptions:
        schemas: list[dict[str, Any]] = kwargs.get("tools") or []
        allowed = normalize_tools(list(self.allowed_tools or []))
        mcp_servers = dict(self.mcp_servers)
        max_turns = self.max_turns or 1
        if schemas:
            mcp_servers[MCP_SERVER_NAME] = _mcp_server(schemas)
            allowed += [f"{TOOL_PREFIX}{schema['name']}" for schema in schemas]
            # The caller executes tool calls, so the CLI must not keep going
            # after it has requested one.
            max_turns = 1

        if self.builtin_tools == "claude_code":
            builtin: Any = {"type": "preset", "preset": "claude_code"}
        elif self.builtin_tools:
            builtin = normalize_tools(list(self.builtin_tools))
        else:
            builtin = []

        options = ClaudeAgentOptions(
            model=kwargs.get("model", self.model),
            system_prompt=cast(Any, system_prompt or self.system_prompt),
            tools=builtin,
            allowed_tools=allowed,
            disallowed_tools=normalize_tools(list(self.disallowed_tools or [])),
            mcp_servers=mcp_servers,
            permission_mode=self.permission_mode,
            max_turns=max_turns,
            max_budget_usd=self.max_budget_usd,
            thinking=cast(Any, self.thinking),
            effort=self.effort,
            output_format=kwargs.get("output_format"),
            cwd=self.cwd,
            add_dirs=list(self.add_dirs),
            cli_path=self.cli_path,
            settings=self.settings,
            setting_sources=self.setting_sources,
            env=dict(self.env),
            extra_args=dict(self.extra_args),
            include_partial_messages=partial_messages,
            fork_session=self.fork_session,
        )
        session_id = kwargs.get("session_id", self.session_id)
        if session_id:
            options.resume = session_id
        return options

    def _prompt_and_options(
        self,
        messages: list[BaseMessage],
        *,
        partial_messages: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, ClaudeAgentOptions]:
        system_prompt, content = convert_messages(messages)
        options = self._build_options(
            system_prompt=system_prompt,
            partial_messages=partial_messages,
            **kwargs,
        )
        if len(content) == 1:  # text only
            return content[0]["text"], options

        async def stream() -> AsyncIterator[dict[str, Any]]:
            yield {"type": "user", "message": {"role": "user", "content": content}}

        return stream(), options

    async def _acollect(self, messages: list[BaseMessage], **kwargs: Any) -> _Turn:
        prompt, options = self._prompt_and_options(messages, **kwargs)
        turn = _Turn()
        stream: Any = query(prompt=prompt, options=options)
        try:
            async for message in stream:
                if isinstance(message, AssistantMessage):
                    turn.add_assistant(message)
                    if turn.tool_calls:
                        # Executing the tool is the caller's job, so end the CLI
                        # turn here instead of letting it answer itself.
                        break
                elif isinstance(message, ResultMessage):
                    turn.add_result(message)
        finally:
            await stream.aclose()
        return turn

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        turn = _run_sync(lambda: self._acollect(messages, **kwargs))
        return ChatResult(generations=[ChatGeneration(message=turn.to_message())])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        turn = await self._acollect(messages, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=turn.to_message())])

    async def _achunks(
        self, messages: list[BaseMessage], **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        if kwargs.get("tools"):
            # Tool calls only materialize as whole blocks; stream them as one chunk.
            turn = await self._acollect(messages, **kwargs)
            message = turn.to_message()
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=message.content,
                    tool_call_chunks=[
                        create_tool_call_chunk(
                            name=call["name"],
                            args=json.dumps(call["args"]),
                            id=call["id"],
                            index=index,
                        )
                        for index, call in enumerate(message.tool_calls)
                    ],
                    response_metadata=message.response_metadata,
                    additional_kwargs=message.additional_kwargs,
                    usage_metadata=message.usage_metadata,
                )
            )
            return

        prompt, options = self._prompt_and_options(
            messages, partial_messages=True, **kwargs
        )
        async for sdk_message in query(prompt=prompt, options=options):
            if isinstance(sdk_message, StreamEvent):
                event = sdk_message.event
                if event.get("type") != "content_block_delta":
                    continue
                text = (event.get("delta") or {}).get("text", "")
                if text:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=text))
            elif isinstance(sdk_message, ResultMessage):
                turn = _Turn()
                turn.add_result(sdk_message)
                final = turn.to_message()
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        response_metadata=final.response_metadata,
                        additional_kwargs=final.additional_kwargs,
                        usage_metadata=final.usage_metadata,
                    )
                )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in _iter_sync(lambda: self._achunks(messages, **kwargs)):
            if run_manager and chunk.text:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for chunk in self._achunks(messages, **kwargs):
            if run_manager and chunk.text:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | Literal["auto", "any"] | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tools so the model can request them via ``AIMessage.tool_calls``.

        The tools are exposed to the CLI through an in-process MCP server. The
        model's call is captured and returned to the caller — executing it stays
        the caller's responsibility, as with any other LangChain chat model.

        Args:
            tools: Tools as ``BaseTool``, Pydantic models, functions or schemas.
            tool_choice: ``"any"`` or a tool name to require a tool call.
            **kwargs: Extra arguments bound to every call.
        """
        schemas = [convert_to_claude_tool(tool) for tool in tools]

        hint: str | None = None
        if tool_choice in ("any", "required"):
            hint = "You must call one of the available tools."
        elif isinstance(tool_choice, str) and tool_choice != "auto":
            hint = f"You must call the tool `{tool_choice}`."

        bound = self
        if hint and isinstance(self.system_prompt, (str, type(None))):
            prompt = f"{self.system_prompt}\n\n{hint}" if self.system_prompt else hint
            bound = self.model_copy(update={"system_prompt": prompt})
        return bound.bind(tools=schemas, **kwargs)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[Any, dict[str, Any] | BaseModel]:
        """Return a runnable whose output matches ``schema``.

        Uses the CLI's native JSON-schema output format, so the response is
        constrained rather than merely requested.
        """
        json_schema = _json_schema(schema)
        llm = self.bind(output_format={"type": "json_schema", "schema": json_schema})

        model: type[BaseModel] | None = (
            schema
            if isinstance(schema, type) and issubclass(schema, BaseModel)
            else None
        )

        def parse(message: BaseMessage) -> Any:
            data = message.additional_kwargs.get("structured_output")
            if data is None:
                data = json.loads(str(message.text))
            return model.model_validate(data) if model is not None else data

        parser = RunnableLambda(parse)
        if not include_raw:
            return llm | parser

        parse_raw = RunnablePassthrough.assign(
            parsed=itemgetter("raw") | parser,
            parsing_error=lambda _: None,
        )
        parse_failed = RunnablePassthrough.assign(parsed=lambda _: None)
        return {"raw": llm} | parse_raw.with_fallbacks(
            [parse_failed], exception_key="parsing_error"
        )
