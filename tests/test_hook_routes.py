"""Claude Code HTTP hook 路由契约：输出结构、fail-open、安全校验、字段兼容。"""

import uuid

import httpx
import numpy as np
import pytest
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.routing import Route

import core.memory_store as memory_store
from core.memory_store import Memory, insert_memory, list_pending_turns
from engramtide import EngramTide
from engramtide.hooks.claude_code import (
    POST_COMPACT_PATH,
    STOP_PATH,
    USER_PROMPT_SUBMIT_PATH,
    build_hook_handlers,
    parse_payload,
)
from engramtide.service import MemoryService

LOCAL = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
    allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"],
)
BASE = "http://127.0.0.1:8765"


class FakeService:
    def __init__(self, context="<m>记忆</m>", fail=False):
        self.context = context
        self.fail = fail
        self.calls = []

    async def on_user_prompt(self, session_id, prompt, *, prompt_id=None, cwd=None):
        self.calls.append(("prompt", session_id, prompt, prompt_id, cwd))
        if self.fail:
            raise RuntimeError("boom")
        return self.context

    async def on_stop(self, session_id, message, *, prompt_id=None):
        self.calls.append(("stop", session_id, message, prompt_id))
        if self.fail:
            raise RuntimeError("boom")

    async def on_post_compact(self, session_id):
        self.calls.append(("compact", session_id))


def make_client(service, *, token=None, security=LOCAL, base=BASE):
    handlers = build_hook_handlers(lambda: service, security=security, token=token)
    app = Starlette(routes=[
        Route(path, handler, methods=["POST"]) for path, handler in handlers.items()
    ])
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base)


def prompt_body(**extra):
    body = {
        "session_id": "s1",
        "prompt_id": "p1",
        "hook_event_name": "UserPromptSubmit",
        "cwd": "/work",
        "user_prompt": "今天下雨了",
    }
    body.update(extra)
    return body


# ── payload 解析 ─────────────────────────────────────────


def test_parse_accepts_old_prompt_field_and_rejects_garbage():
    assert parse_payload({"session_id": "s", "prompt": "旧字段"}).prompt == "旧字段"
    assert parse_payload({"session_id": "s", "user_prompt": "新", "prompt": "旧"}).prompt == "新"
    assert parse_payload({"prompt": "no session"}) is None
    assert parse_payload(["not", "a", "dict"]) is None
    assert parse_payload({"session_id": 3}) is None


# ── UserPromptSubmit ─────────────────────────────────────


@pytest.mark.asyncio
async def test_user_prompt_returns_hook_specific_output():
    service = FakeService()
    async with make_client(service) as client:
        resp = await client.post(USER_PROMPT_SUBMIT_PATH, json=prompt_body())

    assert resp.status_code == 200
    assert resp.json() == {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "<m>记忆</m>",
        }
    }
    assert service.calls == [("prompt", "s1", "今天下雨了", "p1", "/work")]


@pytest.mark.asyncio
async def test_no_context_means_empty_body():
    async with make_client(FakeService(context=None)) as client:
        resp = await client.post(USER_PROMPT_SUBMIT_PATH, json=prompt_body())
    assert resp.status_code == 200
    assert resp.content == b""


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [USER_PROMPT_SUBMIT_PATH, STOP_PATH])
async def test_internal_failure_is_200_empty(path):
    async with make_client(FakeService(fail=True)) as client:
        resp = await client.post(path, json=prompt_body(last_assistant_message="x"))
    assert resp.status_code == 200
    assert resp.content == b""


@pytest.mark.asyncio
async def test_malformed_json_and_subagent_are_ignored():
    service = FakeService()
    async with make_client(service) as client:
        bad = await client.post(
            USER_PROMPT_SUBMIT_PATH,
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
        sub = await client.post(USER_PROMPT_SUBMIT_PATH, json=prompt_body(agent_id="a1"))

    assert (bad.status_code, bad.content) == (200, b"")
    assert (sub.status_code, sub.content) == (200, b"")
    assert service.calls == []


# ── Stop / PostCompact ───────────────────────────────────


@pytest.mark.asyncio
async def test_stop_records_reply_and_never_returns_decision():
    service = FakeService()
    async with make_client(service) as client:
        resp = await client.post(
            STOP_PATH,
            json={"session_id": "s1", "prompt_id": "p1", "last_assistant_message": "答"},
        )
    assert (resp.status_code, resp.content) == (200, b"")
    assert service.calls == [("stop", "s1", "答", "p1")]


@pytest.mark.asyncio
async def test_post_compact_forwards_session():
    service = FakeService()
    async with make_client(service) as client:
        resp = await client.post(POST_COMPACT_PATH, json={"session_id": "s1"})
    assert (resp.status_code, resp.content) == (200, b"")
    assert service.calls == [("compact", "s1")]


# ── 安全 ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_foreign_host_is_rejected():
    service = FakeService()
    async with make_client(service, base="http://evil.example:8765") as client:
        resp = await client.post(USER_PROMPT_SUBMIT_PATH, json=prompt_body())
    assert resp.status_code == 421
    assert service.calls == []


@pytest.mark.asyncio
async def test_non_json_content_type_is_rejected():
    async with make_client(FakeService()) as client:
        resp = await client.post(USER_PROMPT_SUBMIT_PATH, content=b"x")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_bearer_token_required_when_configured():
    service = FakeService()
    async with make_client(service, token="secret") as client:
        missing = await client.post(USER_PROMPT_SUBMIT_PATH, json=prompt_body())
        wrong = await client.post(
            USER_PROMPT_SUBMIT_PATH, json=prompt_body(),
            headers={"authorization": "Bearer nope"},
        )
        ok = await client.post(
            USER_PROMPT_SUBMIT_PATH, json=prompt_body(),
            headers={"authorization": "Bearer secret"},
        )
    assert (missing.status_code, wrong.status_code, ok.status_code) == (401, 401, 200)
    assert len(service.calls) == 1


# ── 端到端：真实 MemoryService ───────────────────────────


@pytest.mark.asyncio
async def test_end_to_end_with_real_service(monkeypatch, tmp_path):
    memory_store.close_db()
    monkeypatch.setattr(memory_store, "DB_PATH", tmp_path / "routes.db")

    async def fake_embed(_: str):
        return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr("engramtide.api.embed_text", fake_embed)
    service = MemoryService(EngramTide(validate_config=False))
    insert_memory(Memory(
        memory_id=str(uuid.uuid4()),
        type="semantic",
        content="用户喜欢在雨天散步",
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    ))

    try:
        async with make_client(service) as client:
            resp = await client.post(USER_PROMPT_SUBMIT_PATH, json=prompt_body())
            await client.post(
                STOP_PATH,
                json={"session_id": "s1", "prompt_id": "p1", "last_assistant_message": "带伞"},
            )
        context = resp.json()["hookSpecificOutput"]["additionalContext"]
        assert '<engramtide-memory session="s1">' in context
        assert "用户喜欢在雨天散步" in context
        assert [r["role"] for r in list_pending_turns("s1")] == ["user", "assistant"]
    finally:
        memory_store.close_db()
