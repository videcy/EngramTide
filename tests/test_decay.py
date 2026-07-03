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
