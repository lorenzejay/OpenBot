"""CrewAI as a Bot: a conversational CrewAI Agent, served over AG-UI.

The Bot is a real `crewai.Agent`, run the way CrewAI runs one in a chat session: a conversational
Flow (`conversational=True`, a `converse` route, `stream_turn`) whose one handler gives the
conversation to the Agent and appends its reply. That is the same bridge CrewAI builds for
`Agent(conversational=True).as_flow()` (see `crewai.agent.conversation`), and `ag_ui_crewai`
serves it with `add_crewai_flow_fastapi_endpoint(..., conversational=True)`, which maps AG-UI's
`threadId` to CrewAI's session and runs the turn through `stream_turn`. Nothing of the protocol is
written here.

Three things OpenBot needs that CrewAI's own bridge does not do, and that this file adds:

- THE BOT'S ROLE. The server sends its instructions as system messages, and the conversational
  path drops system messages before the Flow sees them. They are kept aside here and given to the
  Agent as system messages, right after CrewAI's own system prompt.
- THE WHOLE THREAD, EVERY RUN. A CrewAI session keeps its history in memory; OpenBot's server
  sends the whole thread on every run, and a fresh process must answer from it. ag-ui-crewai seeds
  the session from the run's messages, and the Agent is built anew for each turn, so nothing
  depends on an earlier run having happened in this process.
- THE CALLER'S TOOLS ARE HANDED BACK. A CrewAI Agent runs its tools itself. OpenBot's tools are
  run by the server: the Bot asks for one, the run ends, and the next run carries the result. Each
  AG-UI tool becomes a CrewAI tool that records the call and ends the turn, and the turn's reply
  is then the assistant message that asks for it.

The contract with the rest of OpenBot is the one every Bot meets: serve AG-UI on a port, answer
`/health`, and refuse anybody who does not carry the server's token.
"""

from __future__ import annotations

import hmac
import json
import mimetypes
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

# The spec file every language in the box reads: one level above this Bot in the repository, and
# one level above /app/src in the image the Dockerfile builds.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from model_providers import bot_settings
from run_model_key import RunModelKey

import ag_ui_crewai.endpoint as crewai_endpoint
from ag_ui.core import Message, Tool
from ag_ui_crewai import add_crewai_flow_fastapi_endpoint
from crewai import LLM, Agent
from crewai.agent.conversation import AgentConversationState
from crewai.flow import ConversationConfig, Flow, listen
from crewai.flow.conversational import ConversationMessage
from crewai.tools import BaseTool
from crewai.utilities.pydantic_schema_utils import create_model_from_schema
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    field_serializer,
    field_validator,
)

# The one header OpenBot's server sends when it calls a managed Bot. Same name the TypeScript Bots
# check, because a Bot is a Bot whatever it is written in.
TOKEN_HEADER = "x-openbot-agent-token"

# Where the run's system messages and tools wait for the turn, in the Flow's state. Not names a
# CrewAI or AG-UI key uses, so neither ever reads or overwrites them.
INSTRUCTIONS_KEY = "openbot_instructions"
CALLER_TOOLS_KEY = "openbot_caller_tools"

# What the Bot is to CrewAI. CrewAI opens its system prompt with "You are {role}. {backstory}" and
# the goal; these say only that the server's instructions, sent right after, are who it is.
ROLE = "the assistant your instructions describe"
BACKSTORY = "Your instructions follow, and they take precedence."
GOAL = "Help the person you are talking to, following your instructions."


def _expected_token() -> str:
    return (os.environ.get("MANAGED_AGENT_TOKEN") or "").strip()


def _model() -> str:
    """The provider and model OpenBot chose, in the form CrewAI's `LLM` wants.

    `BOT_PROVIDER` and `BOT_MODEL` are set by the shell from the model screen. CrewAI addresses a
    model as `provider/model`, and reads that provider's key and base URL from the environment
    itself, which is why nothing here touches a key.

    The model half is whatever the endpoint publishes, slashes included. The first path component
    is taken as the provider and the rest is sent as the model name, so a name that already
    contains a slash still needs the chosen provider in front: `qwen/qwen3-8b` on an
    OpenAI-compatible endpoint is `openai/qwen/qwen3-8b`, and `openai/gpt-5.6-terra` is
    `openai/openai/gpt-5.6-terra`. Treating a slash as "already a provider" dropped the prefix,
    and the call then either went to a provider nobody configured or sent only the second half to
    the endpoint.
    """
    settings = bot_settings("agent-crewai")
    return f"{settings.provider}/{settings.model}"


def _llm() -> LLM:
    return LLM(model=_model())


# --- The run's system messages, kept for the Agent --------------------------------------------


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


_prepare_crewai_inputs = crewai_endpoint.crewai_prepare_inputs


def _openbot_prepare_crewai_inputs(
    *,
    state: dict,
    messages: list[Message],
    tools: list[Tool],
    context: list[Any] | None = None,
    forwarded_props: Any = None,
):
    """ag-ui-crewai's inputs, plus every system message of the run and its tools, set aside.

    The conversational path seeds the session with the run's messages less every system message,
    and the server's are the Bot's role and the guidance for the tools it holds. They are kept
    under a key of their own here, in order, and the turn hands them to the Agent.
    """
    inputs = _prepare_crewai_inputs(
        state=state,
        messages=messages,
        tools=tools,
        context=context,
        forwarded_props=forwarded_props,
    )
    inputs[INSTRUCTIONS_KEY] = [
        text
        for message in messages
        if getattr(message, "role", None) in ("system", "developer")
        and (text := _text(getattr(message, "content", None)).strip())
    ]
    inputs[CALLER_TOOLS_KEY] = [tool.model_dump() for tool in tools]
    return inputs


crewai_endpoint.crewai_prepare_inputs = _openbot_prepare_crewai_inputs


# --- The caller's tools, handed back ----------------------------------------------------------


class HandBack:
    """Which of the caller's tools the Agent called in one turn, for the server to run.

    One per turn, never shared. The lock is for the Agent calling more than one tool at once,
    which CrewAI runs on a pool.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.names: list[str] = []

    def record(self, name: str) -> None:
        with self._lock:
            self.names.append(name)

    def calls(self, messages: list[Any], names: dict[str, str]) -> list[dict[str, Any]]:
        """The calls to hand back, as the model made them: its own ids and arguments.

        The model's call id, not one made up here, because the next run carries the tool's result
        under that id and CrewAI sends it to the model with the call it answers. `names` maps the
        name the model was shown to the name the server knows.
        """
        if not self.names:
            return []
        asked = next(
            (
                message.get("tool_calls")
                for message in reversed(messages)
                if isinstance(message, dict)
                and message.get("role") == "assistant"
                and message.get("tool_calls")
            ),
            None,
        ) or []
        handed_back = []
        for call in asked:
            function = call.get("function") or {}
            name = names.get(function.get("name"))
            if name is None:
                continue
            arguments = function.get("arguments")
            handed_back.append(
                {
                    "id": call.get("id") or f"call_{uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments
                        if isinstance(arguments, str)
                        else json.dumps(arguments or {}),
                    },
                }
            )
        return handed_back


class CallerTool(BaseTool):
    """An AG-UI tool, as a CrewAI tool that is not run here.

    Calling it records the call and ends the turn (`result_as_answer`), so the Agent never sees a
    result it would have to invent. The server runs the tool, and the next run carries the result
    in the thread like any other tool message.
    """

    result_as_answer: bool = True
    _hand_back: HandBack = PrivateAttr()
    caller_name: str

    def __init__(self, *, hand_back: HandBack, **data: Any) -> None:
        # The name the server knows it by, whatever CrewAI makes of `name` for the model.
        super().__init__(caller_name=data["name"], **data)
        self._hand_back = hand_back

    def _run(self, **arguments: Any) -> str:
        self._hand_back.record(self.caller_name)
        return "Handed to the caller to run."


# CrewAI reports every tool it runs, and ag-ui-crewai turns that report into a tool call with its
# result. For a caller's tool that is a call already "answered" (by the stand-in above), under an id
# the model never made, next to the real one the turn hands back: the server would store a phantom
# call and a result for it, and the provider would be sent an orphaned tool result on the next run.
# So a report of a caller's tool, by the Agent of a turn in progress, is not passed on. Any other
# tool an Agent here is given (one a person adds to this file) is reported as before.
_caller_tools_by_agent: dict[str, frozenset[str]] = {}
_caller_tools_lock = threading.Lock()
_is_backend_tool_event = crewai_endpoint.is_backend_tool_event


def _is_caller_tool_event(event: Any) -> bool:
    with _caller_tools_lock:
        names = _caller_tools_by_agent.get(str(getattr(event, "agent_id", None)))
    return bool(names) and getattr(event, "tool_name", None) in names


def _reported_backend_tool_event(event: Any) -> bool:
    return _is_backend_tool_event(event) and not _is_caller_tool_event(event)


crewai_endpoint.is_backend_tool_event = _reported_backend_tool_event


@contextmanager
def _handing_back(agent: Agent, tools: list[CallerTool]) -> Iterator[None]:
    """While the Agent's turn runs, its reports of these tools stay here."""
    key = str(agent.id)
    with _caller_tools_lock:
        _caller_tools_by_agent[key] = frozenset(
            name for tool in tools for name in (tool.name, tool.caller_name)
        )
    try:
        yield
    finally:
        with _caller_tools_lock:
            _caller_tools_by_agent.pop(key, None)


def _caller_tools(tools: list[dict[str, Any]], hand_back: HandBack) -> list[CallerTool]:
    built: list[CallerTool] = []
    for tool in tools:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = tool.get("parameters") or {"type": "object", "properties": {}}
        built.append(
            CallerTool(
                hand_back=hand_back,
                name=name,
                description=tool.get("description") or name,
                args_schema=create_model_from_schema(
                    {"title": f"{name}_arguments", **parameters}
                ),
            )
        )
    return built


# --- The conversation -------------------------------------------------------------------------


# What a provider accepts on each message, and on each tool call. AG-UI messages carry more (ids,
# metadata, encrypted values), CrewAI's session adds its own, and a strict provider refuses a
# request with a field it does not know rather than ignoring it.
_MESSAGE_FIELDS = {
    "user": {"role", "content", "name", "files"},
    "assistant": {"role", "content", "name", "tool_calls"},
    "tool": {"role", "content", "name", "tool_call_id"},
}
_TOOL_CALL_FIELDS = ("id", "type", "function")


def _message_dict(message: Any) -> dict[str, Any] | None:
    """A session message as CrewAI's `kickoff` takes it, or None for one that is not a turn.

    The session holds `ConversationMessage` models, except the turn's own user message, which
    CrewAI appends as a plain dict. System messages are not turns: the run's are handed to the
    Agent separately, ahead of the conversation.
    """
    if isinstance(message, BaseModel):
        message = message.model_dump(exclude_unset=True)
    allowed = _MESSAGE_FIELDS.get(message.get("role"))
    if allowed is None:
        return None
    kept = {
        key: deepcopy(value)
        for key, value in message.items()
        if key in allowed and value is not None
    }
    if isinstance(kept.get("tool_calls"), list):
        kept["tool_calls"] = [
            {key: call[key] for key in _TOOL_CALL_FIELDS if key in call}
            for call in kept["tool_calls"]
            if isinstance(call, dict)
        ]
    # `content` stays even when empty: CrewAI requires it, and a message that only calls tools
    # has none.
    kept.setdefault("content", None)
    return kept


class OpenBotConversationMessage(ConversationMessage):
    """CrewAI's session message, with the AG-UI id it arrived under.

    ag-ui-crewai builds every MESSAGES_SNAPSHOT from the session and keeps a message's `id` when it
    has one; CrewAI's own message has none, so every earlier message of the thread would come back
    under a new id. A snapshot is authoritative (entries it repeats replace the client's copies),
    and a whole thread under new ids is a thread shown twice.
    """

    id: str | None = None


class OpenBotConversationState(AgentConversationState):
    messages: list[OpenBotConversationMessage] = Field(default_factory=list)
    openbot_instructions: list[str] = Field(default_factory=list)
    openbot_caller_tools: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("messages", mode="before")
    @classmethod
    def content_may_be_empty(cls, value: Any) -> Any:
        """An assistant message that only calls tools has no content, and AG-UI leaves it out.

        CrewAI's session message requires the field, so the thread's tool calls would refuse the
        whole run on the turn after the server ran a tool.
        """
        if isinstance(value, list):
            return [
                {**message, "content": None}
                if isinstance(message, dict) and "content" not in message
                else message
                for message in value
            ]
        return value

    @field_serializer("messages", mode="wrap")
    def messages_as_session_messages(
        self, value: Any, handler: SerializerFunctionWrapHandler
    ) -> Any:
        """Messages appended to the list after validation are serialized as session messages too.

        The session's list is appended to in place, which pydantic never sees: the current user
        message arrives as a plain dict, and CrewAI's own appends are its base `ConversationMessage`.
        CrewAI serializes the state on every step, so each of those logged a pydantic "serialized value
        may not be as expected" warning on every run. Coerced here, they serialize as the messages
        they are, and the list itself is left as it is.
        """
        if isinstance(value, list):
            value = [_session_message(message) for message in value]
        return handler(value)


def _session_message(message: Any) -> Any:
    """One entry of the session's list, as the message the state declares."""
    if isinstance(message, OpenBotConversationMessage):
        return message
    if isinstance(message, BaseModel):
        message = message.model_dump()
    if isinstance(message, dict):
        return OpenBotConversationMessage.model_validate({"content": None, **message})
    return message


def _with_request_media(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The newest user message's pictures and files, kept as a message of their own.

    CrewAI makes the newest user message the turn's request and keeps only its text; every other
    message goes to the model whole. ag-ui-crewai already splits a run that ends on the person's
    message this way. A run that ends on a tool's result (the server ran a tool and the Bot
    continues) does not get that split, so it is made here: the media just before the request, as
    the person sent it.
    """
    for index in range(len(conversation) - 1, -1, -1):
        message = conversation[index]
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            return conversation
        text = [part for part in content if isinstance(part, dict) and part.get("type") == "text"]
        media = [part for part in content if not (isinstance(part, dict) and part.get("type") == "text")]
        if not media:
            return conversation
        return [
            *conversation[:index],
            {**message, "content": media},
            {**message, "content": "\n".join(part.get("text", "") for part in text)},
            *conversation[index + 1 :],
        ]
    return conversation


def _with_tool_names(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each tool result names the tool it answers, read off the call it answers.

    AG-UI's tool message carries only the call's id. Gemini's function response needs the
    function's name, and CrewAI's Gemini client reads it from the message, so without it every run
    after the server ran a tool was refused ("function_response.name: Name cannot be empty").
    Gemini's only: an OpenAI-compatible server that accepts no field beyond the spec's refuses a
    tool message that carries a name, and the others go by the id.
    """
    names: dict[str, str] = {}
    named: list[dict[str, Any]] = []
    for message in conversation:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if call.get("id") and function.get("name"):
                names[call["id"]] = function["name"]
        if message.get("role") == "tool" and not message.get("name"):
            name = names.get(message.get("tool_call_id") or "")
            if name:
                message = {**message, "name": name}
        named.append(message)
    return named


def _gemini_part(part: Any) -> Any:
    """An `image_url` part as the inline data CrewAI's Gemini client reads.

    That client turns only `text`, `inlineData` and `fileData` parts into Gemini parts and drops
    the rest without a word, so a picture the person sent never reached the model ("No image
    provided"). A data URL carries its own type; an address is handed over as a file by URI with
    the type its name implies, which Gemini needs.
    """
    if not (isinstance(part, dict) and part.get("type") == "image_url"):
        return part
    url = str((part.get("image_url") or {}).get("url") or "")
    if url.startswith("data:") and ";base64," in url:
        header, data = url[len("data:") :].split(";base64,", 1)
        return {"inlineData": {"mimeType": header or "application/octet-stream", "data": data}}
    if url.startswith(("https://", "http://")):
        mime_type, _ = mimetypes.guess_type(url.split("?", 1)[0])
        return {"fileData": {"fileUri": url, "mimeType": mime_type or "image/jpeg"}}
    return part


# Gemini 3 refuses a function call in the history that does not carry the thought signature it was
# made with ("Function call is missing a thought_signature"). OpenBot runs each tool between two
# runs, and AG-UI's tool call has no field for a signature, so a call always comes back without
# its own. Keeping the real one would mean holding it between runs, on whichever replica answered
# the first. Google documents this value for a history whose signatures are not available.
GEMINI_UNSIGNED_CALL = b"skip_thought_signature_validator"


def _gemini_tool_call_parts(message: dict[str, Any]) -> list[Any]:
    """The assistant's tool calls as Gemini parts, each carrying the unsigned-call signature.

    CrewAI's Gemini client sends `raw_tool_call_parts` as they are, in place of the parts it would
    build from `tool_calls` without a signature.
    """
    from google.genai import types

    parts = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments else {}
            except ValueError:
                arguments = {}
        parts.append(
            types.Part(
                function_call=types.FunctionCall(
                    name=str(function.get("name") or ""),
                    args=arguments if isinstance(arguments, dict) else {},
                ),
                thought_signature=GEMINI_UNSIGNED_CALL,
            )
        )
    return parts


def _for_provider(conversation: list[dict[str, Any]], provider: str) -> list[dict[str, Any]]:
    """The conversation in the shape this provider's CrewAI client reads."""
    if provider != "google":
        return conversation
    shaped = []
    for message in _with_tool_names(conversation):
        if isinstance(message.get("content"), list):
            message = {**message, "content": [_gemini_part(part) for part in message["content"]]}
        if message.get("role") == "assistant" and message.get("tool_calls"):
            message = {**message, "raw_tool_call_parts": _gemini_tool_call_parts(message)}
        shaped.append(message)
    return shaped


def _agent(tools: list[CallerTool]) -> Agent:
    """The Bot, as a CrewAI Agent, for one turn.

    Its role, goal and backstory defer to the server's instructions, which follow CrewAI's own
    system prompt as system messages of their own: they already say who the Bot is, and a role
    invented here would only compete with them.
    """
    return Agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=_llm(),
        tools=tools,
        allow_delegation=False,
        verbose=False,
    )


class OpenBotAgentFlow(Flow[OpenBotConversationState]):
    """The Agent's conversational session, the shape CrewAI gives `Agent(conversational=True)`.

    Every turn takes the `converse` route; there is nothing to route between. The Agent is built
    for the turn from the run's own instructions and tools, so two Bots, or two runs of one, never
    share one.
    """

    _standalone_agent: ClassVar[bool] = True
    conversational: ClassVar[bool] = True
    conversational_config: ClassVar[ConversationConfig] = ConversationConfig(
        defer_trace_finalization=False
    )
    _skip_auto_memory = True

    def route_turn(self, context: dict[str, Any]) -> str:
        return "converse"

    @listen("converse")
    async def converse_turn(self):
        instructions = [
            {"role": "system", "content": text} for text in self.state.openbot_instructions
        ]
        conversation = [
            message
            for message in map(_message_dict, self.state.messages)
            if message is not None
        ]
        if not conversation:
            raise ValueError("A turn needs at least one message")
        conversation = _for_provider(
            _with_request_media(conversation), bot_settings("agent-crewai").provider
        )
        hand_back = HandBack()
        tools = _caller_tools(self.state.openbot_caller_tools, hand_back)
        agent = _agent(tools)
        with self._conversation_streaming_enabled(agent.llm), _handing_back(agent, tools):
            # CrewAI keeps every message before the newest user message as history, with its
            # role, after its own system prompt: the server's instructions lead it.
            result = await agent.kickoff_async(messages=[*instructions, *conversation])
        calls = hand_back.calls(
            result.messages, {tool.name: tool.caller_name for tool in tools}
        )
        if calls:
            # The turn's reply asks the caller to run its tools, and the run ends there.
            self._reply(None, tool_calls=calls)
            return None
        self._reply(result.raw)
        return result

    def _reply(self, content: str | None, **fields: Any) -> None:
        """The turn's reply, appended the way `append_assistant_message` does, with an id.

        One id for the run, so every snapshot the run sends after it agrees on which message it is.
        """
        object.__setattr__(self, "_assistant_reply_appended", True)
        self.state.messages.append(
            OpenBotConversationMessage(
                id=str(uuid4()), role="assistant", content=content, **fields
            )
        )
        self._emit_conversation_message_added(
            role="assistant", content=content, message_index=len(self.state.messages) - 1
        )


app = FastAPI()
# Each run's model calls carry its own gateway token, in its organization. See run_model_key.py.
app.add_middleware(RunModelKey)


@app.middleware("http")
async def refuse_without_the_server_token(request: Request, call_next):
    """Everything but `/health` carries the server's token.

    `/health` is exempt because Compose polls it before anything has a token to send, and a
    healthcheck that authenticates is a container that never reports healthy.
    """
    if request.url.path != "/health":
        expected = _expected_token()
        offered = (request.headers.get(TOKEN_HEADER) or "").strip()
        # An unset token means unconfigured, not open. A Bot that answers anybody because nobody
        # set a secret is the failure this check exists for.
        if not expected or not hmac.compare_digest(offered.encode("utf-8"), expected.encode("utf-8")):
            return JSONResponse({"error": "unauthorised"}, status_code=401)
    return await call_next(request)


@app.get("/health")
async def health():
    return {"ok": True, "harness": "crewai"}


add_crewai_flow_fastapi_endpoint(app, OpenBotAgentFlow(), "/", conversational=True)
