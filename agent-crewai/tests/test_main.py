import argparse
import ipaddress
import json
import os
import socket
import subprocess
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import main


class FakeMessage:
    def model_dump(self):
        return {"role": "assistant", "content": "probe reply"}


class FakeChoice:
    message = FakeMessage()


class FakeCompletion:
    choices = [FakeChoice()]


def run_input(messages):
    return {
        "threadId": "thread-1",
        "runId": "run-1",
        "state": {},
        "messages": messages,
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        (None, None, "openai/gpt-5.5"),
        ("", "", "openai/gpt-5.5"),
        ("   ", "gpt-4o", "openai/gpt-4o"),
        ("openai", "   ", "openai/gpt-5.5"),
        ("anthropic", "claude-3-5-sonnet-latest", "anthropic/claude-3-5-sonnet-latest"),
        ("custom-provider", "custom-model", "custom-provider/custom-model"),
        ("   ", "azure/gpt-4o", "openai/azure/gpt-4o"),
        ("anthropic", "openai/gpt-4o", "anthropic/openai/gpt-4o"),
        ("openai", "qwen/qwen3-8b", "openai/qwen/qwen3-8b"),
        ("", "openai/gpt-5.6-terra", "openai/openai/gpt-5.6-terra"),
    ],
)
def test_model_normalizes_blank_provider_and_model_before_defaults(
    monkeypatch, provider, model, expected
):
    if provider is None:
        monkeypatch.delenv("BOT_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("BOT_PROVIDER", provider)
    if model is None:
        monkeypatch.delenv("BOT_MODEL", raising=False)
    else:
        monkeypatch.setenv("BOT_MODEL", model)

    assert main._model() == expected


def test_session_messages_are_projected_to_provider_fields_without_mutating_state():
    """What the Agent is given is what a strict provider accepts; the session keeps the rest."""
    original_messages = [
        {
            "id": "system-1",
            "role": "system",
            "content": "System instructions",
            "metadata": {"transport": "ag-ui"},
        },
        {
            "id": "user-1",
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
            "name": "user_name",
            "subagent_run_id": "run-user",
        },
        {
            "id": "assistant-1",
            "role": "assistant",
            "name": "assistant_name",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"},
                    "metadata": {"ui": True},
                }
            ],
            "metadata": {"transport": "ag-ui"},
        },
        {
            "id": "tool-1",
            "role": "tool",
            "content": "Tool result",
            "tool_call_id": "call-1",
            "error": None,
            "metadata": {"transport": "ag-ui"},
        },
    ]
    state_messages = deepcopy(original_messages)

    projected = [main._message_dict(message) for message in state_messages]

    assert projected == [
        None,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
            "name": "user_name",
        },
        {
            "role": "assistant",
            "content": None,
            "name": "assistant_name",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": "Tool result",
            "tool_call_id": "call-1",
        },
    ]
    assert state_messages == original_messages


def test_every_system_message_of_the_run_is_kept_for_the_agent():
    """ag-ui-crewai's conversational path drops system messages; the Bot's role is among them."""
    from ag_ui.core import SystemMessage, UserMessage

    inputs = main.crewai_endpoint.crewai_prepare_inputs(
        state={},
        messages=[
            SystemMessage(id="standing-role:bot", role="system", content="You are Ada."),
            SystemMessage(id="granted-tools:bot", role="system", content="You hold: notes."),
            UserMessage(id="u1", role="user", content="Hello"),
        ],
        tools=[],
        context=[],
        forwarded_props={},
    )

    assert inputs[main.INSTRUCTIONS_KEY] == ["You are Ada.", "You hold: notes."]
    assert inputs[main.CALLER_TOOLS_KEY] == []


def test_messages_appended_after_validation_serialize_without_a_warning():
    """A live run appends the user message as a dict and CrewAI appends its base message class.

    CrewAI serializes the state on every step, and each of those used to log pydantic's
    "serialized value may not be as expected" on every run.
    """
    import warnings

    from crewai.flow.conversational import ConversationMessage

    state = main.OpenBotConversationState()
    # Appended the way a run appends them: in place, past validation.
    session: list = state.messages
    session.append({"role": "user", "content": "Hello", "metadata": {}, "id": "u1"})
    session.append(ConversationMessage(role="assistant", content="Hi"))
    session.append({"role": "assistant", "tool_calls": [], "id": "a2"})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dumped = state.model_dump(mode="json")
        copied = state.model_copy(deep=True).model_dump()

    assert [(m["role"], m["content"], m["id"]) for m in dumped["messages"]] == [
        ("user", "Hello", "u1"),
        ("assistant", "Hi", None),
        ("assistant", None, "a2"),
    ]
    assert copied["messages"] == dumped["messages"]
    # Serializing does not rewrite the session's own list.
    assert isinstance(state.messages[0], dict)


def isolated_environment(directory):
    directory.mkdir(parents=True, exist_ok=True)
    inherited = (
        "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "CODEX_HOME",
        "PATH", "LANG", "SYSTEMROOT", "WINDIR",
    )
    environment = {name: os.environ[name] for name in inherited if name in os.environ}
    environment.update(
        MANAGED_AGENT_TOKEN="synthetic-openbot-token",
        OPENAI_API_KEY="sk-synthetic-openbot-key",
        OTEL_SDK_DISABLED="true",
        CREWAI_DISABLE_TELEMETRY="true",
        CREWAI_DISABLE_TRACKING="true",
        CREWAI_TELEMETRY_DISABLED="true",
        CREWAI_STORAGE_DIR=str(directory / "crewai"),
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        PYTHONPYCACHEPREFIX=str(directory / "pycache"),
        TMPDIR=str(directory),
    )
    return environment


def prohibit_external_connections(event, arguments):
    if event == "socket.getaddrinfo":
        host = arguments[0]
    elif event in ("socket.connect", "socket.sendto"):
        address = arguments[1]
        if not isinstance(address, tuple):
            raise RuntimeError("Only loopback TCP/IP is allowed in this proof")
        host = address[0]
    else:
        return
    if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
        raise RuntimeError("External network access is prohibited in this proof")


def free_loopback_socket():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    return listener


def start_loopback_app(app):
    listener = free_loopback_socket()
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started, "loopback server did not start"
    return server, thread, f"http://127.0.0.1:{port}"


def stop_loopback_server(server, thread):
    server.should_exit = True
    thread.join(timeout=10)
    assert not thread.is_alive(), "loopback server did not stop"


def as_stream(completion):
    """A Chat Completions answer as the SSE stream a streamed request gets."""
    message = completion["choices"][0]["message"]
    delta = {key: value for key, value in message.items() if value is not None}
    for index, call in enumerate(delta.get("tool_calls") or []):
        call["index"] = index
    base = {
        "id": completion["id"],
        "object": "chat.completion.chunk",
        "created": completion["created"],
        "model": completion["model"],
    }
    chunks = [
        {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {
            **base,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": completion["choices"][0]["finish_reason"]}
            ],
        },
        {**base, "choices": [], "usage": completion["usage"]},
    ]

    async def stream():
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


def loopback_openai_receiver(records, strict_messages=False):
    app = FastAPI()

    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        completion = answer(body, request.headers)
        if isinstance(completion, dict) and body.get("stream"):
            return as_stream(completion)
        return completion

    def answer(body, headers):
        records.append(
            {
                "model": body.get("model"),
                "messages": body.get("messages"),
                "tools": body.get("tools"),
                "authorization": headers.get("authorization"),
            }
        )
        if not body.get("model"):
            return JSONResponse({"error": {"message": "empty model rejected"}}, status_code=400)
        if strict_messages:
            allowed = {
                "system": {"role", "content", "name"},
                "user": {"role", "content", "name"},
                "assistant": {
                    "role",
                    "audio",
                    "content",
                    "function_call",
                    "name",
                    "refusal",
                    "tool_calls",
                },
                "tool": {"role", "content", "tool_call_id"},
            }
            for index, message in enumerate(body.get("messages") or []):
                role = message.get("role")
                extra = sorted(set(message) - allowed.get(role, set()))
                if extra:
                    return JSONResponse(
                        {
                            "error": {
                                "message": f"message {index} role {role} had unsupported fields: {extra}"
                            }
                        },
                        status_code=400,
                    )
                for tool_call in message.get("tool_calls") or []:
                    extra_tool_call = sorted(set(tool_call) - {"id", "type", "function"})
                    if extra_tool_call:
                        return JSONResponse(
                            {
                                "error": {
                                    "message": (
                                        f"message {index} tool call had unsupported fields: "
                                        f"{extra_tool_call}"
                                    )
                                }
                            },
                            status_code=400,
                        )
        if provider_tool_shapes(body.get("tools")) == [show_note_shape()] and not any(
            message.get("role") == "tool" for message in body.get("messages") or []
        ):
            return {
                "id": "chatcmpl-openbot-loopback-tool-call",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_show_note_1",
                                    "type": "function",
                                    "function": {
                                        "name": "show_note",
                                        "arguments": "{\"title\":\"Quarterly plan\"}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        if any(message.get("role") == "tool" for message in body.get("messages") or []):
            return {
                "id": "chatcmpl-openbot-loopback-tool-result",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Saved note Quarterly plan.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        return {
            "id": "chatcmpl-openbot-loopback",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "loopback response",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

    return app


def show_note_agui_tool():
    return {
        "name": "show_note",
        "description": "Show a note title to the caller.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title to show.",
                }
            },
            "required": ["title"],
            "additionalProperties": False,
        },
    }


def show_note_provider_tool():
    return {
        "type": "function",
        "function": show_note_agui_tool(),
    }


def agui_events(response_text):
    events = []
    for line in response_text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line.removeprefix("data: ")))
    return events


def run_loopback_proof(proof_case, output):
    sys.addaudithook(prohibit_external_connections)
    provider, model, expected_provider_model, expected_receiver_model = {
        "blank-provider": ("   ", "gpt-4o", "openai/gpt-4o", "gpt-4o"),
        "blank-model": ("openai", "   ", "openai/gpt-5.5", "gpt-5.5"),
        "strict-projection": ("openai", "gpt-4o", "openai/gpt-4o", "gpt-4o"),
        "caller-tool": ("openai", "gpt-4o", "openai/gpt-4o", "gpt-4o"),
        "namespaced-model": (
            "openai",
            "qwen/qwen3-8b",
            "openai/qwen/qwen3-8b",
            "qwen/qwen3-8b",
        ),
        "gateway-model": (
            "openai",
            "openai/gpt-5.6-terra",
            "openai/openai/gpt-5.6-terra",
            "openai/gpt-5.6-terra",
        ),
    }[proof_case]
    os.environ["BOT_PROVIDER"] = provider
    os.environ["BOT_MODEL"] = model

    records = []
    receiver, receiver_thread, base_url = start_loopback_app(
        loopback_openai_receiver(records, strict_messages=proof_case == "strict-projection")
    )
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_BASE"] = base_url
    harness, harness_thread, harness_url = start_loopback_app(main.app)
    try:
        with httpx.Client(base_url=harness_url, timeout=20, trust_env=False) as client:
            response = client.post(
                "/",
                headers={main.TOKEN_HEADER: "synthetic-openbot-token"},
                json={
                    **run_input(
                        strict_projection_messages()
                        if proof_case == "strict-projection"
                        else basic_messages()
                    ),
                    "tools": [show_note_agui_tool()] if proof_case == "caller-tool" else [],
                },
            )
            if proof_case == "caller-tool":
                first_events = agui_events(response.text)
                first_snapshot = [
                    event.get("messages")
                    for event in first_events
                    if event.get("type") == "MESSAGES_SNAPSHOT"
                ][-1]
                tool_call = first_snapshot[-1]["toolCalls"][0]
                second_response = client.post(
                    "/",
                    headers={main.TOKEN_HEADER: "synthetic-openbot-token"},
                    json={
                        **run_input(
                            [
                                *basic_messages(),
                                {
                                    "id": first_snapshot[-1]["id"],
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": tool_call["id"],
                                            "type": "function",
                                            "function": {
                                                "name": tool_call["function"]["name"],
                                                "arguments": tool_call["function"]["arguments"],
                                            },
                                        }
                                    ],
                                },
                                {
                                    "id": "tool-result-1",
                                    "role": "tool",
                                    "content": "{\"ok\":true}",
                                    "tool_call_id": tool_call["id"],
                                },
                            ]
                        ),
                        "runId": "run-2",
                        "tools": [show_note_agui_tool()],
                    },
                )
            else:
                first_events = []
                first_snapshot = []
                tool_call = None
                second_response = None
        events = agui_events(response.text)
        second_events = agui_events(second_response.text) if second_response else []
        result = {
            "proofCase": proof_case,
            "statusCode": response.status_code,
            "secondStatusCode": second_response.status_code if second_response else None,
            "normalizedModel": main._model(),
            "receiverRecords": records,
            "eventTypes": [event.get("type") for event in events],
            "secondEventTypes": [event.get("type") for event in second_events],
            "decodedToolCall": tool_call,
            "messagesSnapshots": [
                event.get("messages")
                for event in events
                if event.get("type") == "MESSAGES_SNAPSHOT"
            ],
            "runFinished": any(event.get("type") == "RUN_FINISHED" for event in events),
            "runError": any(event.get("type") == "RUN_ERROR" for event in events),
            "teardown": "pending",
        }
    finally:
        stop_loopback_server(harness, harness_thread)
        stop_loopback_server(receiver, receiver_thread)

    result["teardown"] = "stopped"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    assert result["statusCode"] == 200, result
    assert result["normalizedModel"] == expected_provider_model, result
    assert result["runFinished"], result
    assert not result["runError"], result
    assert len(records) == (2 if proof_case == "caller-tool" else 1), result
    assert all(record["model"] == expected_receiver_model for record in records), result
    # CrewAI's own system prompt for the Agent, then the Bot's role as the server sent it.
    assert records[0]["messages"][0]["role"] == "system", result
    assert records[0]["messages"][1] == {
        "role": "system",
        "content": "You are Ada, a Bot-specific finance analyst.",
    }, result
    if proof_case == "caller-tool":
        assert provider_tool_shapes(records[0]["tools"]) == [show_note_shape()], result
        # Handed back, not run: no result on the wire and none in the thread, and the one call the
        # snapshot asks for is the model's own.
        assert "TOOL_CALL_RESULT" not in result["eventTypes"], result
        assert "TOOL_CALL_START" not in result["eventTypes"], result
        first_snapshot = result["messagesSnapshots"][-1]
        assert not [m for m in first_snapshot if m["role"] == "tool"], result
        assert [call["id"] for m in first_snapshot for call in m.get("toolCalls") or []] == [
            "call_show_note_1"
        ], result
        assert result["decodedToolCall"] == {
            "id": "call_show_note_1",
            "type": "function",
            "function": {
                "name": "show_note",
                "arguments": "{\"title\":\"Quarterly plan\"}",
            },
        }, result
        assert result["secondStatusCode"] == 200, result
        assert "RUN_FINISHED" in result["secondEventTypes"], result
        assert provider_tool_shapes(records[1]["tools"]) == [show_note_shape()], result
        assert without_empty_content(records[1]["messages"][-2:]) == [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_show_note_1",
                        "type": "function",
                        "function": {
                            "name": "show_note",
                            "arguments": "{\"title\":\"Quarterly plan\"}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "{\"ok\":true}",
                "tool_call_id": "call_show_note_1",
            },
        ], result
        assert result["messagesSnapshots"][-1][-1]["toolCalls"][0]["id"] == "call_show_note_1"
    elif proof_case == "strict-projection":
        assert_provider_messages_are_projected(records[0]["messages"])
        # The picture reaches the model, just before the question CrewAI makes the request.
        assert records[0]["messages"][2] == {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
            "name": "analyst",
        }, result
        assert records[0]["messages"][3]["role"] == "user", result
        assert "What should I review first?" in records[0]["messages"][3]["content"], result
        assert without_empty_content(records[0]["messages"][4:]) == [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{\"symbol\":\"ACME\"}"},
                    }
                ],
            },
            {"role": "tool", "content": "Synthetic result", "tool_call_id": "call-1"},
        ], result
        # Every message of the thread keeps the id it arrived under, so the snapshot replaces the
        # client's copies instead of adding the thread again under new ones. It carries no system
        # message, which leaves the client's as they are.
        snapshot = result["messagesSnapshots"][-1]
        assert [message["id"] for message in snapshot[:3]] == [
            "user-1",
            "assistant-1",
            "tool-1",
        ], result
        assert not [message for message in snapshot if message["role"] == "system"], result
        assert snapshot[1]["toolCalls"][0]["id"] == "call-1", result
        assert snapshot[2]["toolCallId"] == "call-1", result
    else:
        assert records[0]["messages"][2]["role"] == "user", result
        assert "What should I review first?" in records[0]["messages"][2]["content"], result


def without_empty_content(messages):
    """CrewAI sends `content: null` on a message that only calls tools; the old harness left it out."""
    return [
        {key: value for key, value in message.items() if not (key == "content" and value is None)}
        for message in messages
    ]


def show_note_shape():
    return {
        "name": "show_note",
        "properties": {"title"},
        "required": ["title"],
    }


def provider_tool_shapes(tools):
    """The tools the model was offered, by what matters: CrewAI writes its own schema around them."""
    return [
        {
            "name": tool["function"]["name"],
            "properties": set(tool["function"]["parameters"]["properties"]),
            "required": tool["function"]["parameters"].get("required"),
        }
        for tool in tools or []
    ]


def basic_messages():
    return [
        {
            "id": "system-1",
            "role": "system",
            "content": "You are Ada, a Bot-specific finance analyst.",
        },
        {
            "id": "user-1",
            "role": "user",
            "content": "What should I review first?",
        },
    ]


def strict_projection_messages():
    return [
        basic_messages()[0],
        {
            "id": "user-1",
            "role": "user",
            "content": [
                {"type": "text", "text": "What should I review first?"},
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "value": "data:image/png;base64,AAAA",
                        "mime_type": "image/png",
                    },
                },
            ],
            "name": "analyst",
            "metadata": {"client": "ag-ui"},
            "subagent_run_id": "run-user",
        },
        {
            "id": "assistant-1",
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"symbol\":\"ACME\"}"},
                    "metadata": {"client": "ag-ui"},
                }
            ],
            "metadata": {"client": "ag-ui"},
        },
        {
            "id": "tool-1",
            "role": "tool",
            "content": "Synthetic result",
            "tool_call_id": "call-1",
            "metadata": {"client": "ag-ui"},
            "error": None,
        },
    ]


def assert_provider_messages_are_projected(messages):
    allowed = {
        "system": {"role", "content", "name"},
        "user": {"role", "content", "name"},
        "assistant": {
            "role",
            "audio",
            "content",
            "function_call",
            "name",
            "refusal",
            "tool_calls",
        },
        "tool": {"role", "content", "tool_call_id"},
    }
    for message in messages:
        assert set(message) <= allowed[message["role"]]
        assert "id" not in message
        assert "metadata" not in message
        for tool_call in message.get("tool_calls") or []:
            assert set(tool_call) <= {"id", "type", "function"}
            assert tool_call["id"] == "call-1"


@pytest.mark.parametrize(
    "proof_case",
    ["blank-provider", "blank-model", "strict-projection", "namespaced-model", "gateway-model"],
)
def test_crewai_endpoint_uses_normalized_model_with_real_provider_loopback(
    tmp_path, proof_case
):
    output = tmp_path / f"{proof_case}.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--proof-case", proof_case,
            "--output", str(output),
        ],
        cwd=tmp_path,
        env=isolated_environment(tmp_path),
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_crewai_endpoint_forwards_caller_tools_and_accepts_tool_result_continuation(
    tmp_path,
):
    output = tmp_path / "caller-tool.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--proof-case",
            "caller-tool",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=isolated_environment(tmp_path),
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proof-case",
        choices=[
            "blank-provider",
            "blank-model",
            "strict-projection",
            "caller-tool",
            "namespaced-model",
            "gateway-model",
        ],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run_loopback_proof(arguments.proof_case, arguments.output)
