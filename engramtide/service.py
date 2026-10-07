"""Memory service shared by the Claude Code hook routes and the MCP tools.

Read path (every user prompt, driven by the UserPromptSubmit hook): lazily
start a session keyed by Claude Code's ``session_id``, activate, retrieve,
inject.  Injected memories are acknowledged immediately — additionalContext
always reaches the model, so "injected" and "used" are the same event here.

Write path (explicit MCP calls only): dehydrate buffered conversation turns,
write single memories, consolidate, delete.

The conversation buffer is raw dialogue recorded by the hooks; it is not
memory.  Nothing in this module turns it into memories unless ``dehydrate``
is called.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import config
import core.memory_store as memory_store
from core.context_builder import ConstitutionalMemoryContext
from engramtide import api
from engramtide.api import EngramTide, EngramTideSession
from utils.time_utils import utc_now

logger = logging.getLogger(__name__)

_SECTION_TITLES = (
    ("procedural_memories", "## 偏好与规则"),
    ("semantic_memories", "## 用户事实"),
    ("episodic_memories", "## 近期经历"),
    ("emotional_memories", "## 情感沉淀"),
)
_EMPTY_SECTION = "无"


def _load_template() -> str:
    path = config.PROMPTS_DIR / "hook_context.txt"
    return path.read_text(encoding="utf-8").strip()


def fallback_prompt_id(session_id: str, prompt: str) -> str:
    """Older Claude Code builds omit prompt_id; derive one stable within a second."""
    stamp = utc_now().replace(microsecond=0).isoformat()
    digest = hashlib.sha1(f"{session_id}\0{prompt}\0{stamp}".encode("utf-8"))
    return f"derived-{digest.hexdigest()[:24]}"


def render_context(
    template: str,
    session_id: str,
    context: ConstitutionalMemoryContext | None,
    *,
    max_chars: int,
) -> str:
    """Fill the hook template, skipping empty sections and capping total length."""
    blocks = []
    if context is not None:
        for attr, title in _SECTION_TITLES:
            text = getattr(context, attr)
            if text and text != _EMPTY_SECTION:
                blocks.append(f"{title}\n{text}")
    sections = "\n\n".join(blocks)

    safe_id = html.escape(session_id, quote=True)
    rendered = template.format(session_id=safe_id, sections=sections)
    if len(rendered) <= max_chars:
        return rendered

    # 预算按 token 估算，正常到不了这里；真超了就按行截掉 sections 的尾部
    overhead = len(template.format(session_id=safe_id, sections=""))
    room = max(0, max_chars - overhead)
    cut = sections[:room]
    if "\n" in cut:
        cut = cut[: cut.rindex("\n")]
    return template.format(session_id=safe_id, sections=cut)


@dataclass
class _HookSession:
    session: EngramTideSession
    injected_ids: set[str] = field(default_factory=set)
    header_sent: bool = False


class MemoryService:
    """Own one engine and the Claude Code sessions seen by the hooks."""

    def __init__(
        self,
        engine: EngramTide,
        *,
        session_ttl_seconds: int = 86_400,
        max_sessions: int = 32,
        template: str | None = None,
    ) -> None:
        if session_ttl_seconds <= 0:
            raise ValueError("session_ttl_seconds must be > 0")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be > 0")
        self.engine = engine
        self.session_ttl_seconds = session_ttl_seconds
        self.max_sessions = max_sessions
        self.template = template if template is not None else _load_template()
        self._sessions: OrderedDict[str, _HookSession] = OrderedDict()
        self._last_used: dict[str, float] = {}
        self._lock = asyncio.Lock()

    # ── hook 读路径 ──────────────────────────────────────

    async def on_user_prompt(
        self,
        session_id: str,
        prompt: str,
        *,
        prompt_id: str | None = None,
        cwd: str | None = None,
    ) -> str | None:
        """Record the prompt and return additionalContext, or None to inject nothing."""
        text = prompt.strip()
        if not session_id or not text:
            return None
        turn_id = prompt_id or fallback_prompt_id(session_id, text)

        async with self._lock:
            # 先落缓冲：后面 embedding 失败也不能丢这条对话
            memory_store.upsert_turn(session_id, turn_id, "user", text, cwd)
            if not self._should_inject(text):
                return None
            replay = self._render_replay(session_id, turn_id)
            if replay is not None:
                return replay

        # embedding 是网络调用，放在锁外，多个会话不必排队等它
        try:
            embedding = await asyncio.wait_for(
                api.embed_text(text), timeout=config.HOOK_EMBED_TIMEOUT_SECONDS
            )
        except Exception as exc:  # noqa: BLE001 — hook 必须 fail-open
            logger.warning("hook 注入跳过：embedding 失败（%s）", exc)
            return None

        async with self._lock:
            replay = self._render_replay(session_id, turn_id)
            if replay is not None:
                return replay
            hook_session = self._get_or_create(session_id)
            session = hook_session.session
            exclude = hook_session.injected_ids if config.HOOK_DEDUP_INJECTED else set()
            prepared = await session.prepare_turn(
                text,
                turn_id=turn_id,
                query_embedding=embedding,
                exclude_ids=exclude,
                record_user_message=False,
            )
            session.acknowledge_used(turn_id)
            hook_session.injected_ids.update(prepared.included_memory_ids)

            first_block = not hook_session.header_sent
            hook_session.header_sent = True
            if not prepared.included_memory_ids and not first_block:
                return None
            # 首轮即使没有记忆也发一次头部，模型才知道 dehydrate 该用哪个 session_id
            return render_context(
                self.template,
                session_id,
                prepared.context,
                max_chars=config.HOOK_MAX_CONTEXT_CHARS,
            )

    async def on_stop(
        self,
        session_id: str,
        assistant_message: str,
        *,
        prompt_id: str | None = None,
    ) -> None:
        """Record the final assistant reply of one turn."""
        text = assistant_message.strip()
        if not session_id or not text:
            return
        async with self._lock:
            turn_id = (
                prompt_id
                or memory_store.latest_unanswered_prompt_id(session_id)
                or fallback_prompt_id(session_id, text)
            )
            memory_store.upsert_turn(session_id, turn_id, "assistant", text)

    async def on_post_compact(self, session_id: str) -> None:
        """Compaction summarised earlier injections away; allow re-injection."""
        async with self._lock:
            hook_session = self._sessions.get(session_id)
            if hook_session is not None:
                hook_session.injected_ids.clear()
                hook_session.header_sent = False

    # ── 生命周期 ─────────────────────────────────────────

    async def shutdown(self) -> None:
        async with self._lock:
            for hook_session in self._sessions.values():
                hook_session.session.close()
            self._sessions.clear()
            self._last_used.clear()
            self.engine.close()

    # ── 内部 ─────────────────────────────────────────────

    @staticmethod
    def _should_inject(text: str) -> bool:
        if not config.HOOK_ENABLED:
            return False
        if text.startswith("/"):
            return False
        return len(text) >= config.HOOK_MIN_PROMPT_CHARS

    def _render_replay(self, session_id: str, turn_id: str) -> str | None:
        """Same prompt_id again (hook retry): re-send the context, never re-activate."""
        hook_session = self._sessions.get(session_id)
        if hook_session is None:
            return None
        prepared = hook_session.session.get_prepared(turn_id)
        if prepared is None:
            return None
        self._touch(session_id)
        return render_context(
            self.template,
            session_id,
            prepared.context,
            max_chars=config.HOOK_MAX_CONTEXT_CHARS,
        )

    def _get_or_create(self, session_id: str) -> _HookSession:
        self._prune_expired()
        hook_session = self._sessions.get(session_id)
        if hook_session is None:
            # 容量满了淘汰最久没用的：hook 端不能因为容量拒绝注入。
            # 被淘汰会话的对话缓冲还在库里，dehydrate 照样能处理。
            while len(self._sessions) >= self.max_sessions:
                old_id, old = self._sessions.popitem(last=False)
                old.session.close()
                self._last_used.pop(old_id, None)
            hook_session = _HookSession(session=self.engine.start_session(session_id))
            self._sessions[session_id] = hook_session
        self._touch(session_id)
        return hook_session

    def _touch(self, session_id: str) -> None:
        self._sessions.move_to_end(session_id)
        self._last_used[session_id] = time.monotonic()

    def _prune_expired(self) -> None:
        cutoff = time.monotonic() - self.session_ttl_seconds
        expired = [sid for sid, used in self._last_used.items() if used < cutoff]
        for session_id in expired:
            self._sessions.pop(session_id).session.close()
            self._last_used.pop(session_id)
