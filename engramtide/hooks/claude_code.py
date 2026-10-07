"""Claude Code HTTP hook endpoints (UserPromptSubmit / Stop / PostCompact).

Claude Code POSTs the hook's JSON input and treats any non-2xx response as a
visible non-blocking error.  A memory-layer failure must neither block the
user's prompt nor nag on every turn, so internal failures always answer
``200`` with an empty body.  Only security rejections (Host/Origin,
Content-Type, bearer token) return 4xx — those are misconfiguration and
should be seen.
"""

from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from mcp.server.transport_security import (
    TransportSecurityMiddleware,
    TransportSecuritySettings,
)
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from engramtide.service import MemoryService

logger = logging.getLogger(__name__)

HOOK_PATH_PREFIX = "/hooks/claude-code"
USER_PROMPT_SUBMIT_PATH = f"{HOOK_PATH_PREFIX}/user-prompt-submit"
STOP_PATH = f"{HOOK_PATH_PREFIX}/stop"
POST_COMPACT_PATH = f"{HOOK_PATH_PREFIX}/post-compact"

Handler = Callable[[Request], Awaitable[Response]]


@dataclass(frozen=True)
class HookPayload:
    """The subset of Claude Code's hook input that EngramTide uses."""

    session_id: str
    prompt_id: str | None = None
    prompt: str = ""
    cwd: str | None = None
    agent_id: str | None = None
    last_assistant_message: str = ""


def _opt_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) and value else None


def parse_payload(data: Any) -> HookPayload | None:
    """Parse hook input; None when it is unusable (not an object, no session_id)."""
    if not isinstance(data, dict):
        return None
    session_id = _opt_str(data, "session_id")
    if session_id is None:
        return None
    # 当前文档字段名是 user_prompt，旧版本叫 prompt
    prompt = _opt_str(data, "user_prompt") or _opt_str(data, "prompt") or ""
    return HookPayload(
        session_id=session_id,
        prompt_id=_opt_str(data, "prompt_id"),
        prompt=prompt,
        cwd=_opt_str(data, "cwd"),
        agent_id=_opt_str(data, "agent_id"),
        last_assistant_message=_opt_str(data, "last_assistant_message") or "",
    )


def user_prompt_output(context: str | None) -> dict[str, Any] | None:
    """Wrap additionalContext in Claude Code's hookSpecificOutput shape."""
    if not context:
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }


def build_hook_handlers(
    get_service: Callable[[], MemoryService],
    *,
    security: TransportSecuritySettings | None,
    token: str | None,
) -> dict[str, Handler]:
    """Return ``{path: handler}`` for registration on the HTTP app."""
    guard = TransportSecurityMiddleware(security)

    async def _reject(request: Request) -> Response | None:
        rejected = await guard.validate_request(request, is_post=True)
        if rejected is not None:
            return rejected
        if token:
            supplied = request.headers.get("authorization", "")
            if not hmac.compare_digest(supplied, f"Bearer {token}"):
                return Response("Unauthorized", status_code=401)
        return None

    async def _payload(request: Request) -> HookPayload | None:
        try:
            payload = parse_payload(await request.json())
        except Exception:  # noqa: BLE001 — 坏请求体也不打扰用户
            logger.warning("hook 请求体不是合法 JSON，已忽略")
            return None
        # 子代理的内部对话既不注入也不记录
        if payload is None or payload.agent_id is not None:
            return None
        return payload

    async def user_prompt_submit(request: Request) -> Response:
        rejected = await _reject(request)
        if rejected is not None:
            return rejected
        payload = await _payload(request)
        if payload is None:
            return Response(status_code=200)
        try:
            context = await get_service().on_user_prompt(
                payload.session_id,
                payload.prompt,
                prompt_id=payload.prompt_id,
                cwd=payload.cwd,
            )
        except Exception:  # noqa: BLE001
            logger.exception("UserPromptSubmit hook 处理失败，本轮不注入")
            return Response(status_code=200)
        output = user_prompt_output(context)
        return JSONResponse(output) if output else Response(status_code=200)

    async def stop(request: Request) -> Response:
        # 永远返回空 body：带 decision 的响应会阻止 Claude 结束本轮
        rejected = await _reject(request)
        if rejected is not None:
            return rejected
        payload = await _payload(request)
        if payload is not None:
            try:
                await get_service().on_stop(
                    payload.session_id,
                    payload.last_assistant_message,
                    prompt_id=payload.prompt_id,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Stop hook 处理失败，本轮回复未记录")
        return Response(status_code=200)

    async def post_compact(request: Request) -> Response:
        rejected = await _reject(request)
        if rejected is not None:
            return rejected
        payload = await _payload(request)
        if payload is not None:
            try:
                await get_service().on_post_compact(payload.session_id)
            except Exception:  # noqa: BLE001
                logger.exception("PostCompact hook 处理失败")
        return Response(status_code=200)

    return {
        USER_PROMPT_SUBMIT_PATH: user_prompt_submit,
        STOP_PATH: stop,
        POST_COMPACT_PATH: post_compact,
    }
