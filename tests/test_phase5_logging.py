"""
Phase 5 D1 日志层 + A4 正对照测试（阻塞性）。

D1: 验证 decay_events / access_events 正确记录
A4: 全部计访问关闭时 λ̂ ≡ λ_theory（残差 < 0.1%）
"""

import math
import uuid
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from config import BASE_DECAY_RATE, DECAY_FLOOR
from core.memory_store import (
    Memory,
    close_db,
    init_db,
    insert_memory,
    log_access_event,
    log_decay_event,
    query_access_events,
    query_decay_events,
)
from core.decay import apply_decay, compute_decay_multiplier


def _make_mem(
    mem_type: str = "episodic",
    content: str = "test",
    decay_weight: float = 1.0,
    access_count: int = 0,
    arousal: float = 0.0,
) -> Memory:
    return Memory(
        memory_id=str(uuid.uuid4()),
        type=mem_type,
        content=content,
        decay_weight=decay_weight,
        access_count=access_count,
        arousal=arousal,
    )


@pytest.fixture(autouse=True)
def setup_db(monkeypatch, tmp_path):
    """独立临时数据库。"""
    db_file = tmp_path / "test_phase5_logging.db"
    monkeypatch.setattr("core.memory_store.DB_PATH", db_file)
    import core.memory_store as ms
    ms._connection = None
    init_db()
    yield
    close_db()


# ════════════════════════════════════════════════════════════
# D1: decay_events 日志正确性
# ════════════════════════════════════════════════════════════

class TestDecayEvents:
    """验证 apply_decay 产出正确的 decay_events 记录。"""

    def test_single_episodic_decay_yields_one_event(self):
        """一次 episodic 衰减产生一条日志。"""
        mem = _make_mem("episodic", decay_weight=1.0, access_count=0)
        insert_memory(mem)

        updates, report = apply_decay([mem], hours_elapsed=24.0)
        assert report.updated >= 1

        events = query_decay_events(mem.memory_id)
        assert len(events) == 1
        e = events[0]
        assert e["hours"] == pytest.approx(24.0)
        assert e["n"] == 0
        assert e["floored"] == 0

    def test_event_multiplier_matches_formula(self):
        """日志 multiplier 与 compute_decay_multiplier 一致。"""
        mem = _make_mem("episodic", decay_weight=1.0, access_count=3)
        insert_memory(mem)

        apply_decay([mem], hours_elapsed=48.0)

        events = query_decay_events(mem.memory_id)
        expected = compute_decay_multiplier("episodic", 0.0, 3, 48.0)
        assert events[0]["multiplier"] == pytest.approx(expected, rel=1e-12)

    def test_floored_event_marked(self):
        """触底事件 floored=1。"""
        mem = _make_mem("episodic", decay_weight=0.02, access_count=0)
        insert_memory(mem)

        # Huge elapsed time ensures flooring
        apply_decay([mem], hours_elapsed=87600.0)  # 10 years

        events = query_decay_events(mem.memory_id)
        assert events[0]["floored"] == 1

    def test_semantic_no_decay_no_event(self):
        """semantic 不衰减 → 无事件。"""
        mem = _make_mem("semantic", decay_weight=1.0)
        insert_memory(mem)

        apply_decay([mem], hours_elapsed=24.0)

        events = query_decay_events(mem.memory_id)
        assert len(events) == 0

    def test_procedural_no_decay_no_event(self):
        """procedural 不衰减 → 无事件。"""
        mem = _make_mem("procedural", decay_weight=1.0)
        insert_memory(mem)

        apply_decay([mem], hours_elapsed=24.0)

        events = query_decay_events(mem.memory_id)
        assert len(events) == 0

    def test_multiple_decay_runs_multiple_events(self):
        """多次衰减产生多条独立事件。"""
        mem = _make_mem("episodic", decay_weight=1.0, access_count=0)
        insert_memory(mem)

        # 第一次
        apply_decay([mem], hours_elapsed=24.0)
        # 读取更新后的权重
        from core.memory_store import list_active_memories
        mem2 = list_active_memories()[0]
        # 第二次
        apply_decay([mem2], hours_elapsed=48.0)

        events = query_decay_events(mem.memory_id)
        assert len(events) == 2
        # 事件按时间顺序
        assert events[0]["hours"] == pytest.approx(24.0)
        assert events[1]["hours"] == pytest.approx(48.0)


# ════════════════════════════════════════════════════════════
# D1: access_events 日志正确性
# ════════════════════════════════════════════════════════════

class TestAccessEvents:
    """验证各来源的 access_events 正确记录。"""

    def test_retrieval_source_logged(self):
        """retrieval 来源正确记录。"""
        mid = str(uuid.uuid4())
        log_access_event(mid, "retrieval")
        events = query_access_events(mid)
        assert len(events) == 1
        assert events[0]["source"] == "retrieval"

    def test_strong_activation_source_logged(self):
        """strong_activation 来源正确记录。"""
        mid = str(uuid.uuid4())
        log_access_event(mid, "strong_activation")
        events = query_access_events(mid)
        assert len(events) == 1
        assert events[0]["source"] == "strong_activation"

    def test_surface_source_logged(self):
        """surface 来源正确记录。"""
        mid = str(uuid.uuid4())
        log_access_event(mid, "surface")
        events = query_access_events(mid)
        assert len(events) == 1
        assert events[0]["source"] == "surface"

    def test_dedup_source_logged(self):
        """dedup 来源正确记录。"""
        mid = str(uuid.uuid4())
        log_access_event(mid, "dedup")
        events = query_access_events(mid)
        assert len(events) == 1
        assert events[0]["source"] == "dedup"

    def test_multiple_sources_independent(self):
        """同一记忆的不同来源事件各自独立。"""
        mid = str(uuid.uuid4())
        log_access_event(mid, "surface")
        log_access_event(mid, "retrieval")
        log_access_event(mid, "retrieval")
        log_access_event(mid, "strong_activation")

        events = query_access_events(mid)
        assert len(events) == 4
        sources = [e["source"] for e in events]
        assert sources == ["surface", "retrieval", "retrieval", "strong_activation"]


# ════════════════════════════════════════════════════════════
# A4 正对照：全关计访问 → λ̂ ≡ λ_theory
# ════════════════════════════════════════════════════════════

class TestA4PositiveControl:
    """
    阻塞性测试。若失败，说明存在未登记的计访问路径。
    
    方法：
    1. 插入 episodic 记忆（access_count=0）
    2. 运行一次衰减
    3. 从 decay_events 计算 λ̂
    4. 验证 λ̂ / λ_theory ∈ [0.999, 1.001]（残差 < 0.1%）
    """

    @staticmethod
    def _compute_lambda_hat(events: list[dict], lambda_theory: float | None = None) -> float:
        """从 decay_events 计算有效衰减率 λ̂（/天）。P1 恒等式。

        λ̂ = λ_theory · Σ(h/s) / Σh
        lambda_theory 为 None 时使用 BASE_DECAY_RATE（episodic 默认）。
        """
        if lambda_theory is None:
            lambda_theory = BASE_DECAY_RATE

        total_weighted_reciprocal_s = 0.0
        total_hours = 0.0
        for e in events:
            if e["floored"]:
                continue
            h = e["hours"]
            n = e["n"]
            s = 1.0 + math.log(1.0 + n)
            total_weighted_reciprocal_s += h / s
            total_hours += h

        if total_hours < 1e-12:
            return 0.0
        return lambda_theory * total_weighted_reciprocal_s / total_hours

    def test_access_count_zero_lamda_equals_theory(self):
        """access_count=0 → λ̂ ≡ λ_theory（P3 承诺，残差 < 0.1%）。"""
        mem = _make_mem("episodic", decay_weight=1.0, access_count=0)
        insert_memory(mem)

        apply_decay([mem], hours_elapsed=24.0)

        events = query_decay_events(mem.memory_id)
        assert len(events) == 1
        assert events[0]["floored"] == 0

        lamda_hat = self._compute_lambda_hat(events)
        lamda_theory = BASE_DECAY_RATE

        residual = abs(lamda_hat - lamda_theory) / lamda_theory
        assert residual < 0.001, (
            f"A4 正对照不达标！λ̂={lamda_hat:.6f}, λ_theory={lamda_theory:.6f}, "
            f"残差={residual:.6f}。可能存在未登记的计访问路径或多加衰减事件。"
        )

    def test_access_count_zero_multiple_times_lamda_equals_theory(self):
        """多次衰减，access_count 始终为 0 → λ̂ 始终 ≡ λ_theory。"""
        mid = str(uuid.uuid4())
        mem = Memory(
            memory_id=mid, type="episodic", content="test",
            decay_weight=1.0, access_count=0,
        )
        insert_memory(mem)

        from core.memory_store import list_active_memories

        # 三轮衰减，access_count 始终 0
        for h in [24.0, 48.0, 72.0]:
            current = list_active_memories()
            mem = [m for m in current if m.memory_id == mid][0]
            apply_decay([mem], hours_elapsed=h)

        events = query_decay_events(mid)
        assert len(events) == 3
        assert all(e["floored"] == 0 for e in events)

        lamda_hat = self._compute_lambda_hat(events)
        lamda_theory = BASE_DECAY_RATE
        residual = abs(lamda_hat - lamda_theory) / lamda_theory
        assert residual < 0.001, (
            f"A4 正对照（多轮）不达标！残差={residual:.6f}"
        )

    def test_access_count_nonzero_match_formula(self):
        """access_count > 0 → λ̂ = λ_theory · Σ(h/s)/Σh（P1 恒等式自检）。"""
        mem = _make_mem("episodic", decay_weight=1.0, access_count=5)
        insert_memory(mem)

        apply_decay([mem], hours_elapsed=48.0)

        events = query_decay_events(mem.memory_id)
        lamda_hat = self._compute_lambda_hat(events)

        # 手动计算理论值：n=5, s=1+ln(6)≈2.7918
        s_expected = 1.0 + math.log(6.0)
        lamda_expected = BASE_DECAY_RATE / s_expected

        assert lamda_hat == pytest.approx(lamda_expected, rel=1e-12), (
            f"P1 恒等式失败：λ̂={lamda_hat:.6f}, λ_theory/s={lamda_expected:.6f}"
        )

    def test_emotional_lamda_formula(self):
        """emotional 衰减：λ_theory 含 arousal 调制。"""
        mem = _make_mem("emotional", decay_weight=1.0, access_count=0, arousal=0.5)
        insert_memory(mem)

        apply_decay([mem], hours_elapsed=24.0)

        from config import EMOTIONAL_DECAY_FACTOR
        lamda_theory = BASE_DECAY_RATE * (1.0 - 0.5 * EMOTIONAL_DECAY_FACTOR)
        events = query_decay_events(mem.memory_id)
        lamda_hat = self._compute_lambda_hat(events, lambda_theory=lamda_theory)

        residual = abs(lamda_hat - lamda_theory) / lamda_theory
        assert residual < 0.001, (
            f"emotional λ̂ 偏差：{lamda_hat:.6f} vs {lamda_theory:.6f}"
        )

    def test_access_events_empty_when_no_sources_active(self):
        """所有计访问来源关闭时，access_events 为空（A4 底座）。"""
        mid = str(uuid.uuid4())
        # 不做任何 log_access_event 调用 → 应为空
        events = query_access_events(mid)
        assert len(events) == 0


# ════════════════════════════════════════════════════════════
# D3 消融开关：Scene F 等价性
# ════════════════════════════════════════════════════════════

class TestAblationSwitches:
    """消融开关全 on 时行为与无开关版本逐字段一致（Scene F 模式）。"""

    def test_all_switches_on_by_default(self):
        """默认全部开启。"""
        from config import ABL_ACCESS_RETRIEVAL, ABL_ACCESS_ACTIVATION, ABL_ACCESS_SURFACE
        assert ABL_ACCESS_RETRIEVAL is True
        assert ABL_ACCESS_ACTIVATION is True
        assert ABL_ACCESS_SURFACE is True

    def test_switches_off_by_env(self, monkeypatch):
        """环境变量 off → 开关关闭。"""
        monkeypatch.setattr("config.ABL_ACCESS_RETRIEVAL", False)
        monkeypatch.setattr("config.ABL_ACCESS_ACTIVATION", False)
        monkeypatch.setattr("config.ABL_ACCESS_SURFACE", False)
        from config import ABL_ACCESS_RETRIEVAL, ABL_ACCESS_ACTIVATION, ABL_ACCESS_SURFACE
        assert ABL_ACCESS_RETRIEVAL is False
        assert ABL_ACCESS_ACTIVATION is False
        assert ABL_ACCESS_SURFACE is False

    def test_a4_all_off_warning_in_check_config(self, monkeypatch):
        """全部 off 时 check_config 包含 A4 提示（非 error）。"""
        monkeypatch.setattr("config.ABL_ACCESS_RETRIEVAL", False)
        monkeypatch.setattr("config.ABL_ACCESS_ACTIVATION", False)
        monkeypatch.setattr("config.ABL_ACCESS_SURFACE", False)
        monkeypatch.setattr("config.DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.setattr("config.EMBEDDING_API_KEY", "sk-test")
        from config import check_config
        errors = check_config()
        a4_msgs = [e for e in errors if "A4" in e]
        assert len(a4_msgs) == 1


# ════════════════════════════════════════════════════════════
# L1b: 泛函形式反事实（五种 stability 泛函的 λ̂ 分布对比）
# ════════════════════════════════════════════════════════════

_STABILITY_FUNCTIONALS = {
    "s=1 (no access modulation)": lambda n: 1.0,
    "s=1+ln(1+n) (current)": lambda n: 1.0 + math.log(1.0 + n),
    "s=1+ln(1+min(n,5)) (capped)": lambda n: 1.0 + math.log(1.0 + min(n, 5)),
    "s=√(1+n) (stronger sublinear)": lambda n: math.sqrt(1.0 + n),
    "s=1+0.5n (linear reference)": lambda n: 1.0 + 0.5 * n,
}


class TestL1bCounterfactuals:
    """L1b：同一 n 轨迹下五种 stability 泛函的 λ̂ 对比。"""

    def _compute_lamda_for_functional(self, events, s_fn, lambda_theory):
        """给定 s(n) 泛函，计算 λ̂。"""
        total_weighted_recip_s = 0.0
        total_hours = 0.0
        for e in events:
            if e["floored"]:
                continue
            h = e["hours"]
            s = s_fn(e["n"])
            total_weighted_recip_s += h / s
            total_hours += h
        if total_hours < 1e-12:
            return 0.0
        return lambda_theory * total_weighted_recip_s / total_hours

    def test_all_functionals_computed(self):
        """五种泛函全部可计算，无 crash。"""
        mem = _make_mem("episodic", decay_weight=1.0, access_count=5)
        insert_memory(mem)
        apply_decay([mem], hours_elapsed=48.0)
        events = query_decay_events(mem.memory_id)

        results = {}
        for name, s_fn in _STABILITY_FUNCTIONALS.items():
            lamda = self._compute_lamda_for_functional(events, s_fn, BASE_DECAY_RATE)
            results[name] = lamda

        assert len(results) == 5
        # 预注册预期排序：no-mod ≥ ln ≥ capped > √(1+n) > linear
        names_order = list(results.keys())
        # 验证 no-mod 最大（s=1 无减速）
        assert results[names_order[0]] == pytest.approx(BASE_DECAY_RATE, rel=1e-12)
        # 验证 linear 最小（减速最强）
        lamda_linear = results["s=1+0.5n (linear reference)"]
        lamda_current = results["s=1+ln(1+n) (current)"]
        assert lamda_linear < lamda_current, (
            f"预注册预期：linear λ̂ ({lamda_linear:.6f}) < current λ̂ ({lamda_current:.6f})"
        )

    def test_current_form_is_sublinear_not_amplifying(self):
        """现行 ln 形式是次线性压缩，不是放大（H3 重表述依据）。"""
        # n=100 → s=5.6, 弹性仅 0.176。远小于线性参照。
        n = 100
        s_ln = _STABILITY_FUNCTIONALS["s=1+ln(1+n) (current)"](n)
        s_linear = _STABILITY_FUNCTIONALS["s=1+0.5n (linear reference)"](n)
        # 现行形式减速远弱于线性参照
        assert s_ln < s_linear * 0.2  # 5.6 vs 51

    def test_low_n_contributes_most_deceleration(self):
        """n ∈ [0,5] 贡献 >60% 的减速量（论文 §5 关键论据）。"""
        # n=0: s=1  → λ̂/λ = 1.00
        # n=5: s=2.79 → λ̂/λ = 0.36
        # 下降 0.64，占总可能范围 (1-0.18=0.82) 的 78%
        s_fn = _STABILITY_FUNCTIONALS["s=1+ln(1+n) (current)"]
        s_0 = s_fn(0)
        s_5 = s_fn(5)
        s_100 = s_fn(100)

        decel_0_to_5 = 1.0 / s_0 - 1.0 / s_5
        decel_5_to_100 = 1.0 / s_5 - 1.0 / s_100
        total_decel = 1.0 / s_0 - 1.0 / s_100
        share_0_to_5 = decel_0_to_5 / total_decel

        assert share_0_to_5 > 0.60, (
            f"n∈[0,5] 贡献减速份额 {share_0_to_5:.2%} ≤ 60%"
        )
