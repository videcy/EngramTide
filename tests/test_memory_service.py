"""MemoryService 读路径：懒建会话、注入即确认、去重、重放、淘汰、fail-open。"""

import asyncio
import uuid

import numpy as np
import pytest

import core.memory_store as memory_store
from core.memory_store import Memory, insert_memory, list_pending_turns
from engramtide import EngramTide
from engramtide.service import MemoryService, render_context
from core.context_builder import ConstitutionalMemoryContext

VEC = np.array([1.0, 0.0], dtype=np.float32)
TEMPLATE = '<m session="{session_id}">\n{sections}\n</m>'


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    memory_store.close_db()
    monkeypatch.setattr(memory_store, "DB_PATH", tmp_path / "service.db")
    yield
    memory_store.close_db()


@pytest.fixture
def embed_calls(monkeypatch):
    calls: list[str] = []

    async def fake_embed(text: str) -> np.ndarray:
        calls.append(text)
        return VEC

    monkeypatch.setattr("engramtide.api.embed_text", fake_embed)
    return calls


def make_service(**kwargs) -> MemoryService:
    return MemoryService(EngramTide(validate_config=False), template=TEMPLATE, **kwargs)


def add_memory(memory_type: str = "semantic", content: str = "用户喜欢在雨天散步") -> Memory:
    memory = Memory(
        memory_id=str(uuid.uuid4()),
        type=memory_type,
        content=content,
        embedding=VEC,
    )
    insert_memory(memory)
    return memory


@pytest.mark.asyncio
async def test_inject_counts_access_and_records_buffer(embed_calls):
    service = make_service()
    memory = add_memory()

    context = await service.on_user_prompt("s1", "今天下雨了", prompt_id="p1", cwd="/w")

    assert context.startswith('<m session="s1">')
    assert "## 用户事实\n- 用户喜欢在雨天散步" in context
    assert service.engine.get_memory(memory.memory_id).access_count == 1
    [row] = list_pending_turns("s1")
    assert (row["prompt_id"], row["role"], row["content"], row["cwd"]) == (
        "p1", "user", "今天下雨了", "/w",
    )


@pytest.mark.asyncio
async def test_session_is_created_once_per_claude_session(embed_calls, monkeypatch):
    service = make_service()
    add_memory()
    starts: list[str] = []
    original = service.engine.start_session

    def counting_start(session_id=None, **kwargs):
        starts.append(session_id)
        return original(session_id, **kwargs)

    monkeypatch.setattr(service.engine, "start_session", counting_start)

    await service.on_user_prompt("s1", "今天下雨了", prompt_id="p1")
    await service.on_user_prompt("s1", "明天也下雨吗", prompt_id="p2")
    await service.on_user_prompt("s2", "今天下雨了", prompt_id="p1")

    assert starts == ["s1", "s2"]


@pytest.mark.asyncio
async def test_already_injected_memories_are_not_repeated(embed_calls):
    service = make_service()
    memory = add_memory()

    first = await service.on_user_prompt("s1", "今天下雨了", prompt_id="p1")
    second = await service.on_user_prompt("s1", "今天还在下雨", prompt_id="p2")

    assert "雨天散步" in first
    assert second is None
    assert service.engine.get_memory(memory.memory_id).access_count == 1


@pytest.mark.asyncio
async def test_dedup_can_be_disabled(embed_calls, monkeypatch):
    monkeypatch.setattr("config.HOOK_DEDUP_INJECTED", False)
    service = make_service()
    add_memory()

    await service.on_user_prompt("s1", "今天下雨了", prompt_id="p1")
    second = await service.on_user_prompt("s1", "今天还在下雨", prompt_id="p2")

    assert "雨天散步" in second


@pytest.mark.asyncio
async def test_post_compact_allows_reinjection(embed_calls):
    service = make_service()
    add_memory()

    await service.on_user_prompt("s1", "今天下雨了", prompt_id="p1")
    await service.on_post_compact("s1")
    again = await service.on_user_prompt("s1", "今天还在下雨", prompt_id="p2")

    assert "雨天散步" in again


@pytest.mark.asyncio
async def test_first_turn_sends_header_even_without_memories(embed_calls):
    service = make_service()

    first = await service.on_user_prompt("s1", "帮我写个脚本", prompt_id="p1")
    second = await service.on_user_prompt("s1", "再加个参数", prompt_id="p2")

    assert first == '<m session="s1">\n\n</m>'
    assert second is None


@pytest.mark.asyncio
async def test_replayed_prompt_id_resends_without_reactivating(embed_calls):
    service = make_service()
    memory = add_memory("episodic", "三月被反爬封了 IP")

    first = await service.on_user_prompt("s1", "重写爬虫", prompt_id="p1")
    weight = service.engine.get_memory(memory.memory_id).decay_weight
    replay = await service.on_user_prompt("s1", "重写爬虫", prompt_id="p1")

    assert replay == first
    after = service.engine.get_memory(memory.memory_id)
    assert after.decay_weight == weight
    assert after.access_count == 2  # 强激活 1 + 浮现 1，重放不再计
    assert embed_calls == ["重写爬虫"]


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", ["/compact", "好的", "   "])
async def test_short_and_slash_prompts_are_buffered_but_not_injected(embed_calls, prompt):
    service = make_service()
    add_memory()

    assert await service.on_user_prompt("s1", prompt, prompt_id="p1") is None
    assert embed_calls == []
    expected = 0 if not prompt.strip() else 1
    assert len(list_pending_turns("s1")) == expected


@pytest.mark.asyncio
async def test_hook_disabled_still_buffers(embed_calls, monkeypatch):
    monkeypatch.setattr("config.HOOK_ENABLED", False)
    service = make_service()
    add_memory()

    assert await service.on_user_prompt("s1", "今天下雨了", prompt_id="p1") is None
    assert len(list_pending_turns("s1")) == 1


@pytest.mark.asyncio
async def test_embedding_failure_fails_open_but_keeps_buffer(monkeypatch):
    async def broken(_: str):
        raise RuntimeError("provider down")

    monkeypatch.setattr("engramtide.api.embed_text", broken)
    service = make_service()
    add_memory()

    assert await service.on_user_prompt("s1", "今天下雨了", prompt_id="p1") is None
    assert len(list_pending_turns("s1")) == 1


@pytest.mark.asyncio
async def test_embedding_timeout_fails_open(monkeypatch):
    async def slow(_: str):
        await asyncio.sleep(1)
        return VEC

    monkeypatch.setattr("engramtide.api.embed_text", slow)
    monkeypatch.setattr("config.HOOK_EMBED_TIMEOUT_SECONDS", 0.01)
    service = make_service()

    assert await service.on_user_prompt("s1", "今天下雨了", prompt_id="p1") is None


@pytest.mark.asyncio
async def test_capacity_evicts_least_recently_used(embed_calls):
    service = make_service(max_sessions=2)

    await service.on_user_prompt("a", "第一个会话", prompt_id="p1")
    await service.on_user_prompt("b", "第二个会话", prompt_id="p1")
    await service.on_user_prompt("a", "回到第一个", prompt_id="p2")
    await service.on_user_prompt("c", "第三个会话", prompt_id="p1")

    assert list(service._sessions) == ["a", "c"]


@pytest.mark.asyncio
async def test_stop_pairs_reply_with_prompt(embed_calls):
    service = make_service()

    await service.on_user_prompt("s1", "问题一", prompt_id="p1")
    await service.on_stop("s1", "回答一", prompt_id="p1")
    await service.on_user_prompt("s1", "问题二", prompt_id="p2")
    await service.on_stop("s1", "回答二")  # 缺 prompt_id → 配到最近未回复的输入
    await service.on_stop("s1", "   ")

    rows = [(r["prompt_id"], r["role"], r["content"]) for r in list_pending_turns("s1")]
    assert rows == [
        ("p1", "user", "问题一"),
        ("p1", "assistant", "回答一"),
        ("p2", "user", "问题二"),
        ("p2", "assistant", "回答二"),
    ]


@pytest.mark.asyncio
async def test_missing_prompt_id_is_derived(embed_calls):
    service = make_service()

    await service.on_user_prompt("s1", "没有 prompt_id 的旧版本")

    [row] = list_pending_turns("s1")
    assert row["prompt_id"].startswith("derived-")


def test_render_skips_empty_sections_and_caps_length():
    context = ConstitutionalMemoryContext(
        procedural_memories="- 先给结论",
        episodic_memories="\n".join(f"- 经历{i}" for i in range(200)),
    )

    full = render_context(TEMPLATE, 's"1', context, max_chars=10_000)
    assert "## 偏好与规则" in full and "## 用户事实" not in full
    assert 's&quot;1' in full

    capped = render_context(TEMPLATE, "s1", context, max_chars=120)
    assert len(capped) <= 120
    assert capped.endswith("\n</m>")
    assert not capped.split("\n")[-2].endswith("经")  # 按行截断，不切半条
