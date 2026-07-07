"""
测试上下文组装器：按类型分组、空输入、截断、debug 模式。
"""

import uuid

import numpy as np

from core.context_builder import (
    ConstitutionalMemoryContext,
    build_constitutional_memory_context,
)
from core.memory_store import Memory


def _make_result(
    content: str,
    mem_type: str = "semantic",
    score: float = 0.8,
) -> tuple[Memory, float]:
    mem = Memory(
        memory_id=str(uuid.uuid4()),
        type=mem_type,
        content=content,
        embedding=np.array([1.0], dtype=np.float32),
    )
    return (mem, score)


def test_empty_input_all_none():
    """空输入时四个片段均为 '无'。"""
    ctx = build_constitutional_memory_context([])
    assert ctx.procedural_memories == "无"
    assert ctx.semantic_memories == "无"
    assert ctx.episodic_memories == "无"
    assert ctx.emotional_memories == "无"


def test_semantic_in_correct_bucket():
    """semantic 类型记忆进入 semantic 片段。"""
    results = [_make_result("用户在做 AI 项目", "semantic")]
    ctx = build_constitutional_memory_context(results)
    assert "AI 项目" in ctx.semantic_memories
    assert ctx.procedural_memories == "无"


def test_procedural_in_correct_bucket():
    """procedural 类型记忆进入 procedural 片段，不混入其他。"""
    results = [_make_result("用户吐槽时先共情", "procedural")]
    ctx = build_constitutional_memory_context(results)
    assert "先共情" in ctx.procedural_memories
    assert ctx.semantic_memories == "无"
    assert ctx.emotional_memories == "无"


def test_episodic_in_correct_bucket():
    """episodic 类型记忆进入 episodic 片段。"""
    results = [_make_result("用户昨天调试到凌晨", "episodic")]
    ctx = build_constitutional_memory_context(results)
    assert "调试到凌晨" in ctx.episodic_memories


def test_emotional_in_correct_bucket():
    """emotional 类型记忆进入 emotional 片段。"""
    results = [_make_result("用户厌恶机械回复", "emotional")]
    ctx = build_constitutional_memory_context(results)
    assert "机械回复" in ctx.emotional_memories


def test_mixed_types_separated():
    """多种类型记忆分别进入正确片段。"""
    results = [
        _make_result("程序性规则: 先共情", "procedural", 0.9),
        _make_result("用户在做 AI 项目", "semantic", 0.8),
        _make_result("昨天熬夜了", "episodic", 0.7),
        _make_result("讨厌机械回答", "emotional", 0.6),
    ]
    ctx = build_constitutional_memory_context(results)
    assert "先共情" in ctx.procedural_memories
    assert "AI 项目" in ctx.semantic_memories
    assert "熬夜" in ctx.episodic_memories
    assert "机械回答" in ctx.emotional_memories


def test_unknown_type_downgrades_to_semantic():
    """未知类型降级到 semantic。"""
    results = [_make_result("未知类型的记忆", "unknown_type")]
    ctx = build_constitutional_memory_context(results)
    assert ctx.semantic_memories != "无"
    assert "未知类型的记忆" in ctx.semantic_memories


def test_debug_mode_shows_score():
    """debug 模式下显示分数。"""
    results = [_make_result("测试记忆", "semantic", 0.75)]
    ctx = build_constitutional_memory_context(results, debug=True)
    assert "score=0.750" in ctx.semantic_memories


def test_truncation_respects_max_tokens():
    """超过 max_tokens 时减少条目数量，不截断单条内容。"""
    results = [
        _make_result(f"这是第{i}条记忆，内容较长", "semantic", 0.9 - i * 0.1)
        for i in range(20)
    ]
    # 用很小的 max_tokens，应该只保留少量条目
    ctx = build_constitutional_memory_context(results, max_tokens=20)
    # 确保有至少一条被丢弃
    assert ctx.dropped_count > 0
    # 确保没有在句子中间截断：每条 line 都应该以 "- " 开头和完整结尾
    if ctx.semantic_memories != "无":
        for line in ctx.semantic_memories.split("\n"):
            assert line.startswith("- ")
            assert line.endswith("长")  # 内容以 "内容较长" 结尾
    # est_tokens 不超过预算
    assert ctx.est_tokens <= 20


def test_included_memory_ids_only_final_context_items():
    """included_memory_ids 只包含实际进入最终 context 的记忆。"""
    results = [
        _make_result("短记忆一", "semantic", 0.9),
        _make_result("短记忆二", "semantic", 0.8),
        _make_result("短记忆三", "semantic", 0.7),
    ]
    # 用很小的 token 预算，只能容纳前两条
    ctx = build_constitutional_memory_context(results, max_tokens=5)

    included_contents = [
        mem.content
        for mem, _ in results
        if mem.memory_id in ctx.included_memory_ids
    ]
    # 至少有一条被丢弃（具体取决于 token 估算）
    assert len(included_contents) < 3
    assert ctx.dropped_count > 0


def test_no_memory_type_still_returns_context():
    """即使没有任何匹配类型，也返回有效 Context 对象。"""
    ctx = build_constitutional_memory_context([], max_tokens=100)
    assert isinstance(ctx, ConstitutionalMemoryContext)
    assert ctx.procedural_memories == "无"


# ── Phase 2 测试：浮现记忆合并 ─────────────────────────────


def test_surfaced_memories_merged_with_retrieved():
    """浮现记忆与检索记忆合并，同时出现在 context 中。"""
    surfaced_mem = Memory(
        memory_id=str(uuid.uuid4()),
        type="procedural",
        content="浮现规则：先共情",
        embedding=np.array([1.0], dtype=np.float32),
    )
    retrieved = [_make_result("检索到的语义记忆", "semantic", 0.9)]

    ctx = build_constitutional_memory_context(
        retrieved,
        surfaced=[surfaced_mem],
    )
    # 浮现的 procedural 出现在 procedural 桶中
    assert "先共情" in ctx.procedural_memories
    # 检索的 semantic 出现在 semantic 桶中
    assert "检索到的语义记忆" in ctx.semantic_memories


def test_surfaced_priority_over_retrieved_on_same_id():
    """同一 memory_id 在浮现和检索中都存在时，浮现优先保留。"""
    mem_id = str(uuid.uuid4())
    surfaced_mem = Memory(
        memory_id=mem_id,
        type="procedural",
        content="浮现版本：先共情",
        embedding=np.array([1.0], dtype=np.float32),
    )
    # 检索中也有同一 ID 但内容不同
    retrieved_mem = Memory(
        memory_id=mem_id,
        type="procedural",
        content="检索版本：先共情",
        embedding=np.array([1.0], dtype=np.float32),
    )
    retrieved = [(retrieved_mem, 0.5)]

    ctx = build_constitutional_memory_context(
        retrieved,
        surfaced=[surfaced_mem],
    )
    # 只出现一次（去重），且是浮现版本
    assert ctx.procedural_memories.count("先共情") == 1


def test_surfaced_none_behavior_unchanged():
    """surfaced=None 时行为与 Phase 1 完全一致（无回归）。"""
    results = [_make_result("用户在做 AI 项目", "semantic")]
    ctx = build_constitutional_memory_context(results, surfaced=None)
    assert "AI 项目" in ctx.semantic_memories
    assert ctx.procedural_memories == "无"


def test_included_memory_ids_contains_surfaced():
    """浮现记忆的 ID 出现在 included_memory_ids 中。"""
    mem_id = str(uuid.uuid4())
    surfaced_mem = Memory(
        memory_id=mem_id,
        type="procedural",
        content="浮现规则",
        embedding=np.array([1.0], dtype=np.float32),
    )
    ctx = build_constitutional_memory_context(
        [],
        surfaced=[surfaced_mem],
        max_tokens=1000,
    )
    assert mem_id in ctx.included_memory_ids


def test_surfaced_empty_list_behavior_unchanged():
    """surfaced=[] 时行为不变。"""
    results = [_make_result("用户在做 AI 项目", "semantic")]
    ctx = build_constitutional_memory_context(results, surfaced=[])
    assert "AI 项目" in ctx.semantic_memories


# ── Phase 4 测试：Token 预算新字段 ─────────────────────────


def test_est_tokens_field_populated():
    """est_tokens > 0 表示实际收录 token 估算值。"""
    results = [_make_result("用户在做一个很有趣的 AI 记忆系统项目", "semantic")]
    ctx = build_constitutional_memory_context(results)
    assert ctx.est_tokens > 0
    assert ctx.dropped_count == 0


def test_dropped_count_when_over_budget():
    """超预算时 dropped_count > 0。"""
    results = [
        _make_result(f"这是第{i}条测试记忆内容比较长需要更多token", "semantic", 0.9 - i * 0.1)
        for i in range(30)
    ]
    ctx = build_constitutional_memory_context(results, max_tokens=50)
    assert ctx.dropped_count > 0
    assert ctx.est_tokens <= 50


def test_procedural_preserved_within_budget():
    """procedural 在预算内永远最先收录。"""
    proc_mem = _make_result("程序性规则：用户吐槽时先共情", "procedural", 1.0)
    sem_mems = [
        _make_result(f"语义记忆{i}", "semantic", 0.9 - i * 0.1)
        for i in range(20)
    ]
    ctx = build_constitutional_memory_context(
        [proc_mem] + sem_mems,
        max_tokens=30,  # 紧预算
    )
    # procedural 必须在场
    assert "先共情" in ctx.procedural_memories
    assert "程序性规则" in ctx.procedural_memories


def test_dropped_memories_not_in_included_ids():
    """被丢弃的记忆不出现在 included_memory_ids 中。"""
    results = [
        _make_result(f"记忆A内容较长", "semantic", 0.9),
        _make_result(f"记忆B内容较长", "semantic", 0.8),
        _make_result(f"记忆C内容较长", "semantic", 0.7),
        _make_result(f"记忆D内容较长", "semantic", 0.6),
        _make_result(f"记忆E内容较长", "semantic", 0.5),
    ]
    ctx = build_constitutional_memory_context(results, max_tokens=15)
    # 至少有一条被丢弃
    assert ctx.dropped_count > 0
    # 被丢弃的不在 included 中
    all_ids = {mem.memory_id for mem, _ in results}
    included = set(ctx.included_memory_ids)
    dropped = all_ids - included
    assert len(dropped) == ctx.dropped_count


def test_est_tokens_never_exceed_budget():
    """est_tokens ≤ max_tokens 恒成立（计划 §8.1 硬不变量，单条超预算的条目被丢弃）。"""
    results = [
        _make_result(f"条目内容{i}", "semantic", 0.9 - i * 0.1)
        for i in range(10)
    ]
    for budget in [1, 5, 20, 100, 500]:
        ctx = build_constitutional_memory_context(results, max_tokens=budget)
        assert ctx.est_tokens <= budget, (
            f"budget={budget}: est_tokens={ctx.est_tokens} 超出预算"
        )


def test_single_over_budget_item_dropped_not_included():
    """单条超预算 → 该条被丢弃（计入 dropped_count），预算不透支、后续桶不受牵连。"""
    huge = _make_result("超长记忆内容" * 100, "semantic", 0.9)
    small = _make_result("短情感", "emotional", 0.8)
    ctx = build_constitutional_memory_context([huge, small], max_tokens=20)

    # 超长条被丢弃且不破坏预算
    assert huge[0].memory_id not in ctx.included_memory_ids
    assert ctx.est_tokens <= 20
    # 预算未被透支成负值：处理顺序更靠后的 emotional 桶仍能收录短条目
    assert small[0].memory_id in ctx.included_memory_ids
    assert ctx.dropped_count == 1
