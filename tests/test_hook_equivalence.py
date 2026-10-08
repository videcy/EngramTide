"""等价性验收：hook 注入路径没有改动核心数学。

同一批记忆、同一串输入，分别走
  旧路径：Python API prepare_turn + acknowledge_used
  新路径：MemoryService.on_user_prompt（HOOK_DEDUP_INJECTED=false）
逐轮逐字段比较 decay_weight / access_count / included_memory_ids，要求完全相等。
"""

import numpy as np
import pytest

import core.memory_store as memory_store
from core.memory_store import Memory, insert_memory
from engramtide import EngramTide
from engramtide.service import MemoryService


def _unit(*xs: float) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


MEMORIES = [
    ("m-sem", "semantic", "用户住在杭州", _unit(1, 0, 0), 1.0),
    ("m-proc", "procedural", "回答先给结论", _unit(0, 1, 0), 1.0),
    ("m-epi-sunk", "episodic", "三月被反爬封了 IP", _unit(0, 0, 1), 0.08),
    ("m-epi-mild", "episodic", "上周去了西湖", _unit(1, 0.9, 0), 0.6),
    ("m-emo", "emotional", "面试失败很沮丧", _unit(0.3, 0.2, 1), 0.4),
]

PROMPTS = {
    "我准备把爬虫重写一遍": _unit(0, 0.1, 1),
    "杭州最近天气怎么样": _unit(1, 0.4, 0),
    "爬虫又被封了": _unit(0.1, 0, 1),
    "周末想去西湖走走": _unit(1, 0.8, 0.1),
}


@pytest.fixture
def fake_embedding(monkeypatch):
    async def fake_embed(text: str) -> np.ndarray:
        return PROMPTS[text]

    monkeypatch.setattr("engramtide.api.embed_text", fake_embed)


def _fresh_db(monkeypatch, path):
    memory_store.close_db()
    monkeypatch.setattr(memory_store, "DB_PATH", path)
    memory_store.init_db()
    for memory_id, memory_type, content, vec, weight in MEMORIES:
        insert_memory(Memory(
            memory_id=memory_id,
            type=memory_type,
            content=content,
            embedding=vec,
            decay_weight=weight,
            arousal=0.8 if memory_type == "emotional" else 0.0,
            created_at="2026-01-01 00:00:00",
            last_accessed="2026-01-01 00:00:00",
        ))


def _snapshot(engine: EngramTide) -> dict:
    return {
        r.memory_id: (r.decay_weight, r.access_count)
        for r in engine.list_memories()
    }


@pytest.mark.asyncio
async def test_hook_path_matches_python_api_field_by_field(
    fake_embedding, monkeypatch, tmp_path
):
    monkeypatch.setattr("config.HOOK_DEDUP_INJECTED", False)

    # 旧路径
    _fresh_db(monkeypatch, tmp_path / "api.db")
    engine = EngramTide(validate_config=False)
    session = engine.start_session("conv")
    expected = []
    for i, prompt in enumerate(PROMPTS):
        prepared = await session.prepare_turn(prompt, turn_id=f"t{i}")
        session.acknowledge_used(f"t{i}")
        expected.append((sorted(prepared.included_memory_ids), _snapshot(engine)))
    engine.close()

    # 新路径
    _fresh_db(monkeypatch, tmp_path / "hook.db")
    service = MemoryService(EngramTide(validate_config=False), template="{session_id}{sections}")
    actual = []
    for i, prompt in enumerate(PROMPTS):
        await service.on_user_prompt("conv", prompt, prompt_id=f"t{i}")
        prepared = service._sessions["conv"].session.get_prepared(f"t{i}")
        actual.append((sorted(prepared.included_memory_ids), _snapshot(service.engine)))
    await service.shutdown()

    assert actual == expected
    # 确认场景确实覆盖到了激活与权重变化，而不是一路空跑
    assert expected[0][1]["m-epi-sunk"][0] > 0.08
    assert any(ids for ids, _ in expected)
