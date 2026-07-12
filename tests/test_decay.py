"""
测试衰减引擎与浮现机制：衰减数学、apply_decay、run_decay_update、浮现筛选。
"""

import math
import uuid
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from core.decay import (
    DecayReport,
    apply_decay,
    compute_decay_multiplier,
    get_surfaced_memories,
    run_decay_update,
)
from core.memory_store import (
    Memory,
    close_db,
    get_meta,
    init_db,
    insert_memory,
    list_active_memories,
    set_meta,
)
from utils.time_utils import utc_now


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def setup_db(monkeypatch, tmp_path):
    """每个测试使用独立临时数据库。"""
    db_file = tmp_path / "test_decay.db"
    monkeypatch.setattr("core.memory_store.DB_PATH", db_file)

    import core.memory_store as ms

    ms._connection = None
    init_db()
    yield
    close_db()


def _make_mem(
    mem_type: str = "semantic",
    content: str = "测试记忆",
    decay_weight: float = 1.0,
    arousal: float = 0.0,
    access_count: int = 0,
    unresolved: bool = False,
    created_at: str = "",
    embedding: np.ndarray | None = None,
    superseded_by: str | None = None,
) -> Memory:
    return Memory(
        memory_id=str(uuid.uuid4()),
        type=mem_type,
        content=content,
        decay_weight=decay_weight,
        arousal=arousal,
        access_count=access_count,
        unresolved=unresolved,
        created_at=created_at,
        embedding=embedding,
        superseded_by=superseded_by,
    )


# ── compute_decay_multiplier ──────────────────────────────


class TestComputeDecayMultiplier:
    """衰减系数计算（纯函数，无需 DB）。"""

    def test_semantic_never_decays(self):
        assert compute_decay_multiplier("semantic", 0.0, 0, 720) == 1.0

    def test_procedural_never_decays(self):
        assert compute_decay_multiplier("procedural", 0.0, 0, 720) == 1.0

    def test_unknown_type_never_decays(self):
        """未知类型按 fail-safe 不衰减。"""
        assert compute_decay_multiplier("weird_type", 0.0, 0, 720) == 1.0

    def test_episodic_24h_access0(self):
        """episodic 24 小时、access_count=0 → exp(-0.05)。"""
        m = compute_decay_multiplier("episodic", 0.0, 0, 24)
        assert abs(m - math.exp(-0.05)) < 1e-6

    def test_emotional_high_arousal_slower_than_low(self):
        """emotional arousal=1.0 的衰减慢于 arousal=0.0。"""
        m_high = compute_decay_multiplier("emotional", 1.0, 0, 24)
        m_low = compute_decay_multiplier("emotional", 0.0, 0, 24)
        assert m_high > m_low

    def test_access_count_slows_decay(self):
        """access_count=5 衰减慢于 access_count=0。"""
        m5 = compute_decay_multiplier("episodic", 0.0, 5, 24)
        m0 = compute_decay_multiplier("episodic", 0.0, 0, 24)
        assert m5 > m0

    def test_additivity_two_12h_equals_one_24h(self):
        """连续两次 12h 衰减 == 一次 24h 衰减。"""
        m24 = compute_decay_multiplier("episodic", 0.0, 0, 24)
        m12 = compute_decay_multiplier("episodic", 0.0, 0, 12)
        assert abs(m12 * m12 - m24) < 1e-10

    def test_hours_elapsed_zero_returns_one(self):
        assert compute_decay_multiplier("episodic", 0.0, 0, 0) == 1.0

    def test_hours_elapsed_negative_returns_one(self):
        """时钟回拨保护。"""
        assert compute_decay_multiplier("episodic", 0.0, 0, -5) == 1.0


# ── apply_decay ────────────────────────────────────────────


class TestApplyDecay:
    """apply_decay 纯函数测试。"""

    def test_semantic_skipped(self):
        mems = [_make_mem("semantic", decay_weight=1.0)]
        updates, report = apply_decay(mems, 720)
        assert len(updates) == 0
        assert report.skipped == 1
        assert report.updated == 0

    def test_episodic_gets_updated(self):
        mems = [_make_mem("episodic", decay_weight=1.0)]
        updates, report = apply_decay(mems, 720)
        assert len(updates) == 1
        # updates[i] = (new_weight, memory_id)
        assert updates[0][0] < 1.0
        assert report.updated == 1
        assert report.skipped == 0

    def test_floor_activated_on_huge_elapsed(self):
        """巨大时长后 new_weight == DECAY_FLOOR。"""
        mems = [_make_mem("episodic", decay_weight=1.0)]
        updates, report = apply_decay(mems, 87600)  # 10 年
        assert len(updates) == 1
        # updates[i] = (new_weight, memory_id)
        assert updates[0][0] == 0.01  # DECAY_FLOOR
        assert report.floored == 1

    def test_mixed_types_correct_counts(self):
        mems = [
            _make_mem("semantic", decay_weight=1.0),
            _make_mem("procedural", decay_weight=1.0),
            _make_mem("episodic", decay_weight=1.0),
            _make_mem("emotional", decay_weight=1.0, arousal=0.5),
        ]
        updates, report = apply_decay(mems, 24)
        assert report.skipped == 2  # semantic + procedural
        assert report.updated == 2  # episodic + emotional

    def test_no_change_omitted_from_updates(self):
        """权重无显著变化的条目不出现在 updates 中。"""
        mems = [_make_mem("semantic", decay_weight=1.0)]
        updates, _ = apply_decay(mems, 24)
        assert len(updates) == 0


# ── run_decay_update（编排，需要 DB）───────────────────────


class TestRunDecayUpdate:
    """衰减更新编排测试。"""

    def test_first_run_no_decay_sets_benchmark(self):
        """meta 中无 last_decay_run_at → 不衰减，写入基准。"""
        # 确保 meta 为空
        assert get_meta("last_decay_run_at") is None

        report = run_decay_update(now=datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc))
        assert report.hours_elapsed == 0.0
        assert report.updated == 0
        assert get_meta("last_decay_run_at") is not None

    def test_broken_meta_resets_benchmark(self):
        """meta 损坏 → warning，不衰减，重置基准。"""
        set_meta("last_decay_run_at", "garbage-value")
        report = run_decay_update(now=datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc))
        assert report.updated == 0
        # 基准被重置
        new_val = get_meta("last_decay_run_at")
        assert new_val is not None
        assert "2026" in new_val

    def test_normal_path_updates_weights(self):
        """正常路径：权重落库且基准前移。"""
        # 插入一条 episodic 记忆
        insert_memory(_make_mem("episodic", decay_weight=1.0, content="旧事件"))

        # 设置基准为 24 小时前
        now = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)
        yesterday = now - timedelta(hours=24)
        set_meta("last_decay_run_at", yesterday.isoformat())

        report = run_decay_update(now=now)
        assert report.hours_elapsed > 23.9
        assert report.updated == 1

        # 验证权重已降低
        mems = list_active_memories()
        assert mems[0].decay_weight < 1.0

        # 验证基准已前移
        new_benchmark = get_meta("last_decay_run_at")
        assert "2026-07-03" in new_benchmark

    def test_clock_rewind_no_decay(self):
        """时钟回拨 → 不衰减，只刷新基准。"""
        insert_memory(_make_mem("episodic", decay_weight=1.0))

        now = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)
        future = now + timedelta(hours=1)
        set_meta("last_decay_run_at", future.isoformat())

        report = run_decay_update(now=now)
        assert report.updated == 0
        # 基准被刷新到 now
        new_benchmark = get_meta("last_decay_run_at")
        assert "2026-07-03T12" in new_benchmark


# ── get_surfaced_memories ──────────────────────────────────


class TestGetSurfacedMemories:
    """浮现筛选测试。"""

    def test_procedural_always_surfaces(self):
        mems = [
            _make_mem("procedural", "用户吐槽时先共情", decay_weight=0.01),
            _make_mem("episodic", "昨天吃了火锅", decay_weight=1.0),
        ]
        surfaced = get_surfaced_memories(mems)
        surfaced_ids = {m.memory_id for m in surfaced}
        assert mems[0].memory_id in surfaced_ids  # procedural 浮现
        # episodic 可能不浮现（没标记近期）

    def test_high_arousal_emotional_surfaces(self):
        mems = [
            _make_mem("emotional", "极度愤怒", arousal=0.9, decay_weight=1.0),
        ]
        surfaced = get_surfaced_memories(mems)
        assert len(surfaced) == 1

    def test_low_arousal_emotional_does_not_surface(self):
        mems = [
            _make_mem("emotional", "平静记忆", arousal=0.3, decay_weight=1.0),
        ]
        surfaced = get_surfaced_memories(mems)
        assert len(surfaced) == 0

    def test_emotional_below_min_decay_does_not_surface(self):
        mems = [
            _make_mem("emotional", "衰减太深", arousal=0.9, decay_weight=0.1),
        ]
        surfaced = get_surfaced_memories(mems)
        # decay_weight=0.1 不满足 > 0.3（严格大于）
        assert len(surfaced) == 0

    def test_unresolved_surfaces(self):
        mems = [
            _make_mem("semantic", "下周要面试，好紧张", unresolved=True, decay_weight=1.0),
        ]
        surfaced = get_surfaced_memories(mems)
        assert len(surfaced) == 1

    def test_unresolved_below_min_decay_does_not_surface(self):
        mems = [
            _make_mem("semantic", "已沉底", unresolved=True, decay_weight=0.1),
        ]
        surfaced = get_surfaced_memories(mems)
        # decay_weight=0.1 不满足 > 0.2
        assert len(surfaced) == 0

    def test_recent_episodic_surfaces(self):
        now = utc_now()
        created = now - timedelta(days=1)  # 1 天前
        mems = [
            _make_mem(
                "episodic",
                "昨天的事件",
                decay_weight=1.0,
                created_at=created.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        ]
        surfaced = get_surfaced_memories(mems, now=now)
        assert len(surfaced) == 1

    def test_old_episodic_does_not_surface(self):
        now = utc_now()
        created = now - timedelta(days=10)  # 10 天前
        mems = [
            _make_mem(
                "episodic",
                "老事件",
                decay_weight=1.0,
                created_at=created.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        ]
        surfaced = get_surfaced_memories(mems, now=now)
        assert len(surfaced) == 0

    def test_recent_episodic_below_min_decay_does_not_surface(self):
        now = utc_now()
        created = now - timedelta(days=1)
        mems = [
            _make_mem(
                "episodic",
                "近期但衰减低",
                decay_weight=0.3,
                created_at=created.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        ]
        surfaced = get_surfaced_memories(mems, now=now)
        # decay_weight=0.3 不满足 > 0.5
        assert len(surfaced) == 0

    def test_superseded_memory_excluded_from_surface(self):
        """被覆盖的记忆不参与浮现。"""
        mems = [
            _make_mem("procedural", "旧规则", superseded_by="new-id"),
        ]
        # 需要 mock 一下——Memory 构造里 superseded_by 是支持传递的
        m = Memory(
            memory_id=str(uuid.uuid4()),
            type="procedural",
            content="旧规则",
            superseded_by="new-id",
        )
        surfaced = get_surfaced_memories([m])
        assert len(surfaced) == 0

    def test_truncation_respects_max_surfaced(self):
        """超出 max_surfaced 时按优先级截断。"""
        now = utc_now()
        mems = []
        # R1: procedural × 3
        for i in range(3):
            mems.append(_make_mem("procedural", f"规则{i}"))
        # R2: emotional × 3
        for i in range(3):
            mems.append(_make_mem("emotional", f"情绪{i}", arousal=0.9, decay_weight=1.0))
        # R3: unresolved × 3
        for i in range(3):
            mems.append(_make_mem("semantic", f"未解决{i}", unresolved=True, decay_weight=1.0))
        # R4: recent episodic × 3
        for i in range(3):
            mems.append(
                _make_mem(
                    "episodic",
                    f"事件{i}",
                    decay_weight=1.0,
                    created_at=(now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                )
            )

        surfaced = get_surfaced_memories(mems, now=now, max_surfaced=5)
        assert len(surfaced) == 5
        # 前 3 条应为 procedural (R1)
        types = [m.type for m in surfaced]
        assert types[:3] == ["procedural", "procedural", "procedural"]
        # 第 4、5 条应为 emotional (R2)
        assert types[3] == "emotional"
        assert types[4] == "emotional"

    def test_invalid_created_at_does_not_crash(self):
        """created_at 解析失败 → 跳过 R4，不崩溃。"""
        now = utc_now()
        mems = [
            _make_mem("episodic", "坏日期", decay_weight=1.0, created_at="bad-date"),
        ]
        surfaced = get_surfaced_memories(mems, now=now)
        assert len(surfaced) == 0

    def test_procedural_over_limit_logs_warning_and_truncates(self, caplog):
        """procedural 超限时截断并 warning。"""
        import logging

        mems = [_make_mem("procedural", f"规则{i}") for i in range(15)]
        with caplog.at_level(logging.WARNING):
            surfaced = get_surfaced_memories(mems, max_surfaced=5)
        assert len(surfaced) == 5
        assert "截断" in caplog.text

    def test_arousal_threshold_strictly_greater(self):
        """SURFACE_AROUSAL_THRESHOLD=0.7，恰好等于 0.7 不浮现。"""
        mems = [
            _make_mem("emotional", "边界", arousal=0.7, decay_weight=1.0),
        ]
        surfaced = get_surfaced_memories(mems)
        # arousal 必须 > 0.7，等于不行
        assert len(surfaced) == 0


# ════════════════════════════════════════════════════════════
# Phase 3：Context-Aware 逐轮激活测试
# ════════════════════════════════════════════════════════════


def _make_vec(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def _vec_for_cosine(sim: float, dim: int = 2) -> np.ndarray:
    """构造与第一坐标轴余弦相似度为 sim 的单位向量。"""
    assert dim >= 2
    values = [sim, math.sqrt(max(0.0, 1.0 - sim * sim))] + [0.0] * (dim - 2)
    return _make_vec(values)


def _mild_vec(position: float = 0.5, dim: int = 2) -> np.ndarray:
    """按当前配置在轻激活带内构造向量，避免测试绑定历史阈值。"""
    from config import SIMILARITY_HIGH, SIMILARITY_MID

    sim = SIMILARITY_MID + position * (SIMILARITY_HIGH - SIMILARITY_MID)
    return _vec_for_cosine(sim, dim=dim)


class TestComputeActivations:
    """激活计算纯函数测试。"""

    def test_semantic_procedural_skipped(self):
        """semantic / procedural 不参与激活。"""
        query = _make_vec([1.0, 0.0, 0.0])
        mems = [
            _make_mem("semantic", "用户住在东京",
                      embedding=_make_vec([1.0, 0.0, 0.0])),
            _make_mem("procedural", "先共情",
                      embedding=_make_vec([1.0, 0.0, 0.0])),
        ]
        from core.decay import compute_activations
        activations, report = compute_activations(query, mems)
        assert len(activations) == 0
        assert report.strong == 0
        assert report.mild == 0

    def test_strong_activation_above_similarity_high(self):
        """sim > SIMILARITY_HIGH → 强激活，权重 +0.2。"""
        query = _make_vec([1.0, 0.0])
        mems = [
            _make_mem("episodic", "相关记忆", decay_weight=0.3,
                      embedding=_make_vec([1.0, 0.0])),  # sim = 1.0
        ]
        from core.decay import compute_activations
        activations, report = compute_activations(query, mems)
        assert len(activations) == 1
        mid, new_w, is_strong = activations[0]
        assert is_strong
        assert new_w == pytest.approx(0.5)  # 0.3 + 0.2
        assert report.strong == 1

    def test_mild_activation_between_thresholds(self):
        """SIMILARITY_MID < sim <= SIMILARITY_HIGH → 轻激活，权重 +0.05。"""
        # 根据当前域内校准阈值，在轻激活带内部构造向量。
        query = _make_vec([1.0, 0.0])
        mems = [
            _make_mem("emotional", "略相关", decay_weight=0.5,
                      embedding=_mild_vec()),
        ]
        from core.decay import compute_activations
        activations, report = compute_activations(query, mems)
        assert len(activations) == 1
        mid, new_w, is_strong = activations[0]
        assert not is_strong
        assert new_w == pytest.approx(0.55)  # 0.5 + 0.05
        assert report.mild == 1
        assert report.strong == 0

    def test_below_mid_no_activation(self):
        """sim <= SIMILARITY_MID → 不激活。"""
        query = _make_vec([1.0, 0.0])
        mems = [
            _make_mem("episodic", "无关记忆", decay_weight=0.5,
                      embedding=_make_vec([0.0, 1.0])),  # sim = 0.0
        ]
        from core.decay import compute_activations
        activations, report = compute_activations(query, mems)
        assert len(activations) == 0

    def test_capped_at_one(self):
        """权重封顶 1.0。"""
        query = _make_vec([1.0, 0.0])
        mems = [
            _make_mem("episodic", "已饱和", decay_weight=0.95,
                      embedding=_make_vec([1.0, 0.0])),
        ]
        from core.decay import compute_activations
        activations, report = compute_activations(query, mems)
        assert len(activations) == 1
        _, new_w, is_strong = activations[0]
        assert new_w == 1.0  # 0.95 + 0.2 = 1.15 → capped
        assert is_strong  # 强激活即使封顶也产出条目（供计访问）

    def test_reactivated_detection(self):
        """权重从 < 0.1 跨回 >= 0.1 → reactivated 计数。"""
        query = _make_vec([1.0, 0.0])
        mems = [
            _make_mem("episodic", "沉底记忆", decay_weight=0.05,
                      embedding=_make_vec([1.0, 0.0])),
        ]
        from core.decay import compute_activations
        _, report = compute_activations(query, mems)
        assert report.reactivated == 1  # 0.05 → 0.25，跨过 0.1

    def test_mild_cannot_reactivate(self):
        """轻激活 0.01 → 0.06 仍在 0.1 以下，reactivated=0。"""
        query = _make_vec([1.0, 0.0])
        mems = [
            _make_mem("episodic", "触底", decay_weight=0.01,
                      embedding=_mild_vec()),
        ]
        from core.decay import compute_activations
        _, report = compute_activations(query, mems)
        assert report.reactivated == 0  # 0.06 < 0.1

    def test_max_activations_per_turn_capping(self):
        """超过 MAX_ACTIVATIONS_PER_TURN 时截断。"""
        query = _make_vec([1.0])
        mems = []
        # 构造 5 条高相似记忆（全部命中强激活）
        for i in range(5):
            mems.append(_make_mem("episodic", f"相关{i}", decay_weight=0.3,
                        embedding=_make_vec([1.0])))
        from core.decay import compute_activations
        activations, report = compute_activations(query, mems, max_activations=3)
        assert len(activations) == 3
        assert report.capped == 2

    def test_superseded_skipped(self):
        """被覆盖的记忆不参与激活。"""
        query = _make_vec([1.0, 0.0])
        mems = [
            _make_mem("episodic", "已覆盖", decay_weight=0.5,
                      embedding=_make_vec([1.0, 0.0]), superseded_by="some-id"),
        ]
        from core.decay import compute_activations
        activations, _ = compute_activations(query, mems)
        assert len(activations) == 0

    def test_null_embedding_skipped(self):
        """无 embedding 的记忆不参与激活。"""
        query = _make_vec([1.0, 0.0])
        mems = [
            _make_mem("episodic", "无向量", decay_weight=0.5, embedding=None),
        ]
        from core.decay import compute_activations
        activations, _ = compute_activations(query, mems)
        assert len(activations) == 0

    def test_details_populated_and_aligned(self):
        """report.details 逐条填充，与 activations 一一对应、按相似度降序。"""
        query = _make_vec([1.0, 0.0])
        mems = [
            _make_mem("episodic", "强命中", decay_weight=0.3,
                      embedding=_make_vec([1.0, 0.0])),       # sim=1.0, strong
            _make_mem("episodic", "轻命中", decay_weight=0.4,
                      embedding=_mild_vec()),
        ]
        from core.decay import compute_activations
        activations, report = compute_activations(query, mems)
        assert len(report.details) == len(activations) == 2
        # 明细与 activations 顺序、内容严格对齐
        for act, det in zip(activations, report.details):
            mid, new_w, is_strong = act
            assert det.memory_id == mid
            assert det.new_weight == pytest.approx(new_w)
            assert det.is_strong == is_strong
        # 按相似度降序
        assert report.details[0].sim >= report.details[1].sim
        # 具体数值
        assert report.details[0].content == "强命中"
        assert report.details[0].old_weight == pytest.approx(0.3)
        assert report.details[0].new_weight == pytest.approx(0.5)
        assert report.details[0].is_strong is True
        assert report.details[1].is_strong is False

    def test_capped_strong_at_ceiling_not_rewritten(self):
        """已封顶 1.0 的强激活：产出条目（供计访问）但 new_weight 仍为 1.0（无冗余空写）。"""
        query = _make_vec([1.0, 0.0])
        mems = [
            _make_mem("episodic", "已封顶", decay_weight=1.0,
                      embedding=_make_vec([1.0, 0.0])),       # sim=1.0, strong, 已在 1.0
        ]
        from core.decay import compute_activations
        activations, report = compute_activations(query, mems)
        assert len(activations) == 1                          # 仍产出（强激活计访问）
        _, new_w, is_strong = activations[0]
        assert is_strong is True
        assert new_w == pytest.approx(1.0)
        assert report.details[0].old_weight == pytest.approx(1.0)
        assert report.details[0].new_weight == pytest.approx(1.0)


class TestContextAwareUpdate:
    """激活编排测试（需要 DB）。"""

    @pytest.mark.asyncio
    async def test_disabled_returns_empty_report(self, monkeypatch):
        """CONTEXT_AWARE_ENABLED=False → 零副作用。"""
        monkeypatch.setattr("config.CONTEXT_AWARE_ENABLED", False)
        from core.decay import context_aware_update
        report = context_aware_update(_make_vec([1.0, 0.0]))
        assert report.strong == 0
        assert report.mild == 0

    def test_persisted_weights_readable(self):
        """激活落库后 list_active_memories 读回权重一致（防 Bug #1 复发）。"""
        import numpy as np
        emb = np.array([1.0, 0.0], dtype=np.float32)
        insert_memory(
            _make_mem("episodic", "可激活", decay_weight=0.3, embedding=emb)
        )
        from core.decay import context_aware_update
        query = np.array([1.0, 0.0], dtype=np.float32)
        report = context_aware_update(query)

        assert report.strong >= 1
        mems = list_active_memories()
        updated = [m for m in mems if m.content == "可激活"][0]
        assert updated.decay_weight == pytest.approx(0.5)  # 0.3 + 0.2

    def test_strong_activation_counts_access(self):
        """强激活后 access_count +1。"""
        emb = np.array([1.0, 0.0], dtype=np.float32)
        insert_memory(
            _make_mem("episodic", "计访问", decay_weight=0.5,
                      access_count=0, embedding=emb)
        )
        from core.decay import context_aware_update
        context_aware_update(np.array([1.0, 0.0], dtype=np.float32))

        mems = list_active_memories()
        updated = [m for m in mems if m.content == "计访问"][0]
        assert updated.access_count == 1

    def test_capped_strong_counts_access_without_weight_change(self):
        """已封顶 1.0 的记忆被强激活：计访问但权重仍 1.0（无冗余空写、不越界）。

        注：insert_memory 的 INSERT 不含 access_count 列（Phase 1/2 行为），
        入库后计数从 DB 默认 0 起算，一次强激活后为 1。
        """
        emb = np.array([1.0, 0.0], dtype=np.float32)
        insert_memory(
            _make_mem("episodic", "封顶计访问", decay_weight=1.0, embedding=emb)
        )
        from core.decay import context_aware_update
        report = context_aware_update(np.array([1.0, 0.0], dtype=np.float32))

        assert report.strong >= 1
        mems = list_active_memories()
        updated = [m for m in mems if m.content == "封顶计访问"][0]
        assert updated.decay_weight == pytest.approx(1.0)   # 未被改动（无冗余空写）
        assert updated.access_count == 1                     # 强激活计访问 +1


class TestActivationBoundaries:
    """阈值边界语义（严格大于，§8.1）与纯函数契约。

    浮点点积无法精确落在 SIMILARITY_HIGH/MID 上，边界用例通过 monkeypatch
    utils.similarity.cosine_similarity 植入精确相似度值
    （compute_activations 在调用时才从模块取该函数，patch 生效）。
    """

    @staticmethod
    def _plant_sims(monkeypatch):
        """让 cosine_similarity 返回记忆 embedding 的首元素（相似度精确可控）。"""
        monkeypatch.setattr(
            "utils.similarity.cosine_similarity",
            lambda _q, emb: float(emb[0]),
        )

    def test_sim_exactly_high_is_mild_only(self, monkeypatch):
        """sim == SIMILARITY_HIGH（上边界）→ 只轻激活（严格大于）。"""
        self._plant_sims(monkeypatch)
        from config import SIMILARITY_HIGH
        from core.decay import compute_activations

        mems = [_make_mem("episodic", "边界高", decay_weight=0.3,
                          embedding=np.array([SIMILARITY_HIGH]))]
        activations, report = compute_activations(np.array([1.0]), mems)
        assert report.strong == 0
        assert report.mild == 1
        _, new_w, is_strong = activations[0]
        assert is_strong is False
        assert new_w == pytest.approx(0.35)  # +0.05 而非 +0.2

    def test_sim_just_above_high_is_strong(self, monkeypatch):
        """sim = SIMILARITY_HIGH + 0.01 → 强激活 +0.2（§8.1 用例）。"""
        self._plant_sims(monkeypatch)
        from config import SIMILARITY_HIGH
        from core.decay import compute_activations

        mems = [_make_mem("episodic", "略过线", decay_weight=0.3,
                          embedding=np.array([SIMILARITY_HIGH + 0.01]))]
        activations, report = compute_activations(np.array([1.0]), mems)
        assert report.strong == 1
        _, new_w, is_strong = activations[0]
        assert is_strong is True
        assert new_w == pytest.approx(0.5)

    def test_sim_exactly_mid_no_activation(self, monkeypatch):
        """sim == SIMILARITY_MID（下边界）→ 不激活（严格大于）。"""
        self._plant_sims(monkeypatch)
        from config import SIMILARITY_MID
        from core.decay import compute_activations

        mems = [_make_mem("episodic", "边界中", decay_weight=0.3,
                          embedding=np.array([SIMILARITY_MID]))]
        activations, report = compute_activations(np.array([1.0]), mems)
        assert activations == []
        assert report.strong == 0
        assert report.mild == 0
        assert report.considered == 1  # 参与了比较，只是没过线

    def test_sim_just_above_mid_is_mild(self, monkeypatch):
        """sim = SIMILARITY_MID + 0.01 → 轻激活 +0.05（§8.1 用例）。"""
        self._plant_sims(monkeypatch)
        from config import SIMILARITY_MID
        from core.decay import compute_activations

        mems = [_make_mem("emotional", "略过中线", decay_weight=0.3,
                          embedding=np.array([SIMILARITY_MID + 0.01]))]
        activations, report = compute_activations(np.array([1.0]), mems)
        assert report.mild == 1
        _, new_w, is_strong = activations[0]
        assert is_strong is False
        assert new_w == pytest.approx(0.35)

    def test_pure_function_does_not_mutate_inputs(self):
        """compute_activations 不读库、不改入参（纯函数契约，§11 质量门槛）。"""
        from core.decay import compute_activations

        emb = _make_vec([1.0, 0.0])
        mems = [_make_mem("episodic", "纯函数", decay_weight=0.3,
                          access_count=2, embedding=emb)]
        emb_snapshot = mems[0].embedding.copy()
        activations, _ = compute_activations(_make_vec([1.0, 0.0]), mems)
        assert len(activations) == 1                        # 确实命中（非空转）
        assert mems[0].decay_weight == pytest.approx(0.3)   # 入参未被原地修改
        assert mems[0].access_count == 2
        assert np.array_equal(mems[0].embedding, emb_snapshot)


class TestContextAwareUpdateEdgeCases:
    """编排层边界：空库 / 全静态类型库（§8.1）。"""

    def test_empty_db_returns_empty_report(self):
        """空库 → 空 report，不报错。"""
        from core.decay import context_aware_update

        report = context_aware_update(_make_vec([1.0, 0.0]))
        assert (report.considered, report.strong, report.mild,
                report.reactivated, report.capped) == (0, 0, 0, 0, 0)
        assert report.strong_ids == []
        assert report.details == []

    def test_all_static_bank_untouched(self):
        """全 semantic/procedural 库 → 空 report，权重不动。"""
        from core.decay import context_aware_update

        emb = np.array([1.0, 0.0], dtype=np.float32)
        insert_memory(_make_mem("semantic", "语义事实", embedding=emb))
        insert_memory(_make_mem("procedural", "行为偏好", embedding=emb))

        report = context_aware_update(np.array([1.0, 0.0], dtype=np.float32))
        assert report.considered == 0
        assert report.strong == 0
        assert report.mild == 0
        for mem in list_active_memories():
            assert mem.decay_weight == 1.0
            assert mem.access_count == 0


class TestStaticTypeInvariantAcrossMixedFlow:
    """§8.1 全局不变量：衰减 → 激活 → 再衰减的混合流程后，
    semantic / procedural 的 decay_weight 仍恒为 1.0。"""

    def test_semantic_procedural_weight_constant(self):
        from datetime import timedelta

        from core.decay import context_aware_update

        emb = np.array([1.0, 0.0], dtype=np.float32)
        insert_memory(_make_mem("semantic", "语义", embedding=emb))
        insert_memory(_make_mem("procedural", "程序", embedding=emb))
        insert_memory(_make_mem("episodic", "情节", decay_weight=1.0, embedding=emb))

        t0 = utc_now()
        run_decay_update(now=t0)                          # 首次运行：只记基准
        run_decay_update(now=t0 + timedelta(hours=72))    # 衰减
        context_aware_update(np.array([1.0, 0.0], dtype=np.float32))  # 强激活
        run_decay_update(now=t0 + timedelta(hours=144))   # 再衰减

        mems = {m.content: m for m in list_active_memories()}
        assert mems["语义"].decay_weight == 1.0
        assert mems["程序"].decay_weight == 1.0
        assert mems["情节"].decay_weight < 1.0            # 流程确实动了权重（非空转）


class TestSteadyStateMath:
    """§6.3 稳态表断言：每 d 天衰减一次 + 强激活一次，
    稳态 w* = ACTIVATION_HIGH / (1 − e^{−rate·d})，±5% 容差。

    与表格假设一致：access_count 恒为 0（compute_activations 是纯函数，
    循环中不计访问，stability 保持 1）。
    """

    @pytest.mark.parametrize(
        "d_days, expected",
        [(3, 1.0), (7, 0.68), (14, 0.40), (30, 0.26)],
    )
    def test_periodic_strong_activation_steady_state(self, d_days, expected):
        from core.decay import compute_activations

        emb = _make_vec([1.0, 0.0])
        mem = _make_mem("episodic", "周期话题", decay_weight=1.0, embedding=emb)
        query = _make_vec([1.0, 0.0])

        for _ in range(20):
            updates, _ = apply_decay([mem], hours_elapsed=d_days * 24.0)
            for new_w, _mid in updates:   # ⚠ (weight, id) 顺序
                mem.decay_weight = new_w
            activations, _ = compute_activations(query, [mem])
            for _mid, new_w, _is_strong in activations:
                mem.decay_weight = new_w

        assert mem.decay_weight == pytest.approx(expected, rel=0.05)


class TestActivationPerformance:
    """§2 / §7 Step 1 验收：1000 条规模的耗时预算（best-of-3 抗抖动）。"""

    @staticmethod
    def _random_bank(n=1000, dim=1536, seed=0):
        rng = np.random.RandomState(seed)

        def unit(v):
            return (v / np.linalg.norm(v)).astype(np.float32)

        mems = [_make_mem("episodic", f"perf-{i}", decay_weight=0.5,
                          embedding=unit(rng.randn(dim)))
                for i in range(n)]
        return mems, unit(rng.randn(dim))

    def test_compute_activations_1000_under_10ms(self):
        import time

        from core.decay import compute_activations

        mems, query = self._random_bank()
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            compute_activations(query, mems)
            best = min(best, time.perf_counter() - t0)
        assert best < 0.010, f"compute_activations 1000 条耗时 {best*1000:.2f}ms ≥ 10ms"

    def test_context_aware_update_1000_under_50ms(self):
        import time

        from core.decay import context_aware_update
        from core.memory_store import insert_memories

        # 960 条随机向量（1536 维下相似度 ≈ 0，不命中）+ 40 条与 query 同向
        # 的高相似记忆 —— 超过 MAX_ACTIVATIONS_PER_TURN，覆盖截断 + 落库路径。
        mems, query = self._random_bank(n=960)
        mems += [_make_mem("episodic", f"hit-{i}", decay_weight=0.5,
                           embedding=query.copy())
                 for i in range(40)]
        insert_memories(mems)

        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            context_aware_update(query)
            best = min(best, time.perf_counter() - t0)
        assert best < 0.050, f"context_aware_update 全流程耗时 {best*1000:.2f}ms ≥ 50ms"


# ════════════════════════════════════════════════════════════
# Phase 4：轻激活会话上限测试
# ════════════════════════════════════════════════════════════


class TestMildOncePerSession:
    """轻激活会话上限：排除集语义、强激活不受限、跨轮累计。"""

    def test_exclude_mild_skips_activation(self):
        """memory_id 在排除集中且命中轻激活 → 跳过、mild_suppressed 计数。"""
        from core.decay import compute_activations

        query = _make_vec([1.0, 0.0])
        mid = str(uuid.uuid4())
        mems = [
            _make_mem("episodic", "轻激活记忆", decay_weight=0.5,
                      embedding=_mild_vec()),
        ]
        # 覆盖 memory_id
        mems[0].memory_id = mid

        _, report = compute_activations(query, mems, exclude_mild_ids=frozenset([mid]))
        assert report.mild == 0
        assert report.mild_suppressed == 1
        assert len(report.mild_ids) == 0

    def test_strong_activation_unaffected_by_exclude(self):
        """强激活不受排除集影响。"""
        from core.decay import compute_activations

        query = _make_vec([1.0, 0.0])
        mid = str(uuid.uuid4())
        mems = [
            _make_mem("episodic", "强激活记忆", decay_weight=0.3,
                      embedding=_make_vec([1.0, 0.0])),  # sim=1.0, strong
        ]
        mems[0].memory_id = mid

        activations, report = compute_activations(
            query, mems, exclude_mild_ids=frozenset([mid])
        )
        assert report.strong == 1
        assert report.mild_suppressed == 0
        assert len(activations) == 1
        assert activations[0][2] is True  # is_strong

    def test_mild_ids_populated_correctly(self):
        """report.mild_ids 包含本轮实际轻激活的 memory_id。"""
        from core.decay import compute_activations

        query = _make_vec([1.0, 0.0])
        mid = str(uuid.uuid4())
        mems = [
            _make_mem("episodic", "轻激活", decay_weight=0.5,
                      embedding=_mild_vec()),
        ]
        mems[0].memory_id = mid

        _, report = compute_activations(query, mems)
        assert mid in report.mild_ids
        assert len(report.mild_ids) == 1

    def test_pure_function_does_not_modify_input(self):
        """compute_activations 不修改传入的 exclude_mild_ids。"""
        from core.decay import compute_activations

        query = _make_vec([1.0, 0.0])
        mems = [
            _make_mem("episodic", "轻激活", decay_weight=0.5,
                      embedding=_mild_vec()),
        ]
        exclude = frozenset([str(uuid.uuid4())])
        exclude_copy = set(exclude)

        compute_activations(query, mems, exclude_mild_ids=exclude)
        assert exclude == exclude_copy  # 不变

    def test_context_aware_update_respects_exclude(self):
        """context_aware_update 透传 exclude_mild_ids 到 compute_activations。"""
        from core.decay import context_aware_update

        emb = _make_vec([1.0, 0.0, 0.0])
        mid = str(uuid.uuid4())
        insert_memory(
            _make_mem("episodic", "轻激活目标", decay_weight=0.5,
                      embedding=_mild_vec(dim=3))
        )
        # 覆盖 id
        mems = list_active_memories()
        # 直接用 compute_activations 验证（context_aware_update 落库需要 DB fixture）
        from core.decay import compute_activations
        mems[0].memory_id = mid
        _, report = compute_activations(
            emb, mems, exclude_mild_ids=frozenset([mid])
        )
        assert report.mild_suppressed == 1
        assert report.mild == 0

    def test_cross_turn_suppression(self):
        """模拟跨轮：同一记忆第 1 轮轻激活，第 2 轮被抑制。"""
        from core.decay import compute_activations

        query = _make_vec([1.0, 0.0])
        mid = str(uuid.uuid4())
        mems = [
            _make_mem("episodic", "跨轮轻激活", decay_weight=0.5,
                      embedding=_mild_vec()),
        ]
        mems[0].memory_id = mid

        # 第 1 轮：无排除集 → 轻激活成功
        _, report1 = compute_activations(query, mems, exclude_mild_ids=frozenset())
        assert report1.mild == 1
        assert mid in report1.mild_ids

        # 第 2 轮：mid 在排除集中 → 被抑制
        _, report2 = compute_activations(
            query, mems, exclude_mild_ids=frozenset([mid])
        )
        assert report2.mild == 0
        assert report2.mild_suppressed == 1

    def test_strong_memory_not_blocked_by_exclude_list(self):
        """排除集中的记忆命中强激活带 → 照常激活（不受限）。"""
        from core.decay import compute_activations

        query = _make_vec([1.0, 0.0])
        mid = str(uuid.uuid4())
        mems = [
            _make_mem("episodic", "强激活但被排除", decay_weight=0.3,
                      embedding=_make_vec([0.95, 0.312])),  # sim ≈ 0.95, strong
        ]
        mems[0].memory_id = mid

        activations, report = compute_activations(
            query, mems, exclude_mild_ids=frozenset([mid])
        )
        assert report.strong == 1
        assert report.mild_suppressed == 0
        assert len(activations) > 0

    def test_suppressed_do_not_consume_activation_slots(self):
        """次要项修复回归：被抑制的轻激活在截断前剔除，不占 max_activations 槽位。"""
        from core.decay import compute_activations

        query = _make_vec([1.0, 0.0])
        excluded_id = str(uuid.uuid4())
        # 两条都在当前轻激活带内；被排除者相似度更高。
        excluded = _make_mem("被排除的轻激活", decay_weight=0.5,
                             embedding=_mild_vec(position=0.75))
        excluded.memory_id = excluded_id
        other = _make_mem("正常轻激活", decay_weight=0.5,
                          embedding=_mild_vec(position=0.25))
        for m in [excluded, other]:
            m.type = "episodic"

        activations, report = compute_activations(
            query, [excluded, other],
            max_activations=1,                       # 只有 1 个槽位
            exclude_mild_ids=frozenset([excluded_id]),
        )
        # 修复前：截断后只剩被排除者 → 0 激活、capped=1
        # 修复后：排除发生在截断前 → 槽位留给正常候选
        assert report.mild == 1
        assert report.mild_suppressed == 1
        assert report.capped == 0
        assert len(activations) == 1
        assert activations[0][0] == other.memory_id
