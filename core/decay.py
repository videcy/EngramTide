"""
Phase 2 — 类型感知衰减引擎与浮现机制。

设计约束：
- 纯数学计算，不调用任何外部 API。
- semantic / procedural 永不衰减。
- 时间一律使用 aware UTC（utils.time_utils）。
- Phase 3 的 context_aware_update() 将挂载在本模块（见文件尾部预留注释）。
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime

from config import (
    BASE_DECAY_RATE,
    DECAY_FLOOR,
    EMOTIONAL_DECAY_FACTOR,
    MAX_SURFACED_MEMORIES,
    SURFACE_AROUSAL_THRESHOLD,
    SURFACE_EMOTIONAL_MIN_DECAY,
    SURFACE_EPISODIC_MIN_DECAY,
    SURFACE_RECENT_DAYS,
    SURFACE_UNRESOLVED_MIN_DECAY,
)
from core.memory_store import (
    Memory,
    get_meta,
    list_active_memories,
    set_meta,
    update_decay_weights,
)
from utils.time_utils import days_between, hours_between, parse_db_timestamp, utc_now

logger = logging.getLogger(__name__)

# 不衰减的类型集合
_STATIC_TYPES = {"semantic", "procedural"}


# ── 数据结构 ──────────────────────────────────────────────


@dataclass
class DecayReport:
    """一次衰减更新的摘要，供日志与测试断言。"""

    hours_elapsed: float  # 距上次运行经过的小时数
    updated: int          # 实际被更新权重的记忆条数
    skipped: int          # 免衰减跳过条数（semantic/procedural）
    floored: int          # 触底 DECAY_FLOOR 的条数


# ── 衰减数学（纯函数）─────────────────────────────────────


def compute_decay_multiplier(
    mem_type: str,
    arousal: float,
    access_count: int,
    hours_elapsed: float,
) -> float:
    """
    计算一次衰减更新中 decay_weight 应乘上的系数，范围 (0, 1]。

    - semantic / procedural（及任何未知类型）→ 返回 1.0（不衰减；
      未知类型按最保守策略处理并记一条 warning）。
    - episodic:  rate = BASE_DECAY_RATE
    - emotional: rate = BASE_DECAY_RATE * (1 - arousal * EMOTIONAL_DECAY_FACTOR)
    - stability = 1 + ln(1 + access_count)
    - time_factor = hours_elapsed / (24 * stability)
    - multiplier = exp(-rate * time_factor)
    - hours_elapsed <= 0 时返回 1.0（时钟回拨保护）。
    """
    # 时钟回拨保护
    if hours_elapsed <= 0:
        return 1.0

    # 不衰减的类型
    if mem_type in _STATIC_TYPES:
        return 1.0

    # 确定衰减率
    if mem_type == "episodic":
        rate = BASE_DECAY_RATE
    elif mem_type == "emotional":
        # 唤醒度越高衰减越慢
        rate = BASE_DECAY_RATE * (1.0 - arousal * EMOTIONAL_DECAY_FACTOR)
    else:
        # 未知类型：不衰减 + warning（fail-safe）
        logger.warning(
            "未知记忆类型 '%s'，跳过衰减（按 semantic/procedural 策略处理）",
            mem_type,
        )
        return 1.0

    # 稳定性：访问越多越稳定
    stability = 1.0 + math.log(1.0 + access_count)

    # 时间因子：归一化到天，除以稳定性
    time_factor = hours_elapsed / (24.0 * stability)

    # 指数衰减
    multiplier = math.exp(-rate * time_factor)
    return multiplier


def apply_decay(
    memories: list[Memory],
    hours_elapsed: float,
) -> tuple[list[tuple[str, float]], DecayReport]:
    """
    对内存中的记忆列表计算新权重（不落库）。

    返回 (updates, report)：
    - updates: [(memory_id, new_weight)]，只包含权重实际变化的条目；
    - new_weight = max(DECAY_FLOOR, decay_weight * multiplier)。
    """
    updates: list[tuple[str, float]] = []
    skipped = 0
    floored = 0

    for mem in memories:
        multiplier = compute_decay_multiplier(
            mem_type=mem.type,
            arousal=mem.arousal,
            access_count=mem.access_count,
            hours_elapsed=hours_elapsed,
        )

        if multiplier >= 1.0 - 1e-12:
            # 不衰减类型 (multiplier == 1.0)
            skipped += 1
            continue

        new_weight = mem.decay_weight * multiplier
        if new_weight <= DECAY_FLOOR:
            new_weight = DECAY_FLOOR
            floored += 1

        # 只记录实际变化的条目
        if abs(new_weight - mem.decay_weight) > 1e-12:
            updates.append((new_weight, mem.memory_id))

    # 实际被更新条数 = 总条数 - 跳过条数（不含触底计数，触底也算更新）
    updated = len(updates)

    report = DecayReport(
        hours_elapsed=hours_elapsed,
        updated=updated,
        skipped=skipped,
        floored=floored,
    )
    return updates, report


# ── 衰减更新编排 ──────────────────────────────────────────


def run_decay_update(now: datetime | None = None) -> DecayReport:
    """
    会话开始时调用一次。

    流程：
    1. now = now or utc_now()
    2. last = get_meta("last_decay_run_at")
       - 为 None（首次运行）→ 不衰减，set_meta 为 now，返回空 report。
       - parse_db_timestamp 失败 → log warning，不衰减，重置基准为 now
         （fail-safe：宁可漏衰减一次，不允许按错误时长猛衰减）。
    3. hours > 0 → 衰减；hours <= 0 → 只刷新基准，返回空 report。
    4. memories = list_active_memories()
    5. updates, report = apply_decay(memories, hours)
    6. update_decay_weights(updates)  ← 先落权重
    7. set_meta("last_decay_run_at", now.isoformat())  ← 再写基准
    """
    now = now or utc_now()

    last_raw = get_meta("last_decay_run_at")

    if last_raw is None:
        # 首次运行：不衰减，只记录基准
        set_meta("last_decay_run_at", now.isoformat())
        return DecayReport(hours_elapsed=0.0, updated=0, skipped=0, floored=0)

    try:
        last = parse_db_timestamp(last_raw)
    except ValueError:
        logger.warning(
            "meta 中 last_decay_run_at 损坏 (%s)，重置基准为当前时间，跳过本次衰减",
            last_raw,
        )
        set_meta("last_decay_run_at", now.isoformat())
        return DecayReport(hours_elapsed=0.0, updated=0, skipped=0, floored=0)

    hours = hours_between(last, now)

    if hours <= 0:
        # 时钟回拨：不衰减，只刷新基准
        set_meta("last_decay_run_at", now.isoformat())
        return DecayReport(hours_elapsed=0.0, updated=0, skipped=0, floored=0)

    memories = list_active_memories()
    updates, report = apply_decay(memories, hours)

    # 先落权重，再写基准（异常安全：若权重落库失败，基准不推进 → 下次重算同一时段）
    update_decay_weights(updates)
    set_meta("last_decay_run_at", now.isoformat())

    return report


# ── 浮现筛选（纯函数）─────────────────────────────────────


def get_surfaced_memories(
    memories: list[Memory],
    now: datetime | None = None,
    max_surfaced: int = MAX_SURFACED_MEMORIES,
) -> list[Memory]:
    """
    主动浮现：不依赖用户输入，会话开始时调用一次。

    入选规则（一条记忆满足任一即入选，按首个命中规则归类）：
      R1 procedural            → 永远浮现
      R2 emotional             → arousal > SURFACE_AROUSAL_THRESHOLD
                                 且 decay_weight > SURFACE_EMOTIONAL_MIN_DECAY
      R3 unresolved            → decay_weight > SURFACE_UNRESOLVED_MIN_DECAY
      R4 episodic 近期事件     → days_between(created_at, now) <= SURFACE_RECENT_DAYS
                                 且 decay_weight > SURFACE_EPISODIC_MIN_DECAY

    截断规则（超出 max_surfaced 时）：
      按 R1 > R2 > R3 > R4 的优先级保留；同规则内按 decay_weight 降序。
      procedural 原则上全保留；若 procedural 单独超限，说明写入去重失效，
      log warning 并截断（保护上下文优先于规则完整性）。

    健壮性：
      - superseded_by 非空的记忆不参与浮现（双保险）。
      - created_at 解析失败的记忆视为"非近期"，不因 R4 入选，log 一次 warning。
      - embedding 是否为空不影响浮现（浮现是元数据驱动的）。
    """
    now = now or utc_now()

    r1: list[Memory] = []  # procedural
    r2: list[Memory] = []  # high-arousal emotional
    r3: list[Memory] = []  # unresolved
    r4: list[Memory] = []  # recent episodic

    for mem in memories:
        # 双保险：superseded 不浮现
        if mem.superseded_by is not None:
            continue

        # R1: procedural 永远浮现
        if mem.type == "procedural":
            r1.append(mem)
            continue

        # R2: high-arousal emotional
        if mem.type == "emotional":
            if (
                mem.arousal > SURFACE_AROUSAL_THRESHOLD
                and mem.decay_weight > SURFACE_EMOTIONAL_MIN_DECAY
            ):
                r2.append(mem)
            continue

        # R3: unresolved
        if mem.unresolved:
            if mem.decay_weight > SURFACE_UNRESOLVED_MIN_DECAY:
                r3.append(mem)
            continue

        # R4: recent episodic
        if mem.type == "episodic":
            if mem.decay_weight > SURFACE_EPISODIC_MIN_DECAY:
                try:
                    created = parse_db_timestamp(mem.created_at)
                except ValueError:
                    logger.warning(
                        "记忆 %s 的 created_at 解析失败 (%s)，跳过 R4 浮现",
                        mem.memory_id,
                        mem.created_at,
                    )
                    continue

                if days_between(created, now) <= SURFACE_RECENT_DAYS:
                    r4.append(mem)

    # 按 decay_weight 降序排序（同规则内）
    r1.sort(key=lambda m: m.decay_weight, reverse=True)
    r2.sort(key=lambda m: m.decay_weight, reverse=True)
    r3.sort(key=lambda m: m.decay_weight, reverse=True)
    r4.sort(key=lambda m: m.decay_weight, reverse=True)

    # 按优先级截断
    result: list[Memory] = []
    remaining = max_surfaced

    # R1: procedural 全保留（超限时截断并 warning）
    if len(r1) > remaining:
        logger.warning(
            "procedural 记忆 (%d 条) 超过浮现上限 %d，截断——写入去重可能失效",
            len(r1),
            max_surfaced,
        )
        result.extend(r1[:remaining])
        return result
    result.extend(r1)
    remaining -= len(r1)

    # R2: high-arousal emotional
    if remaining <= 0:
        return result
    take = r2[:remaining]
    result.extend(take)
    remaining -= len(take)

    # R3: unresolved
    if remaining <= 0:
        return result
    take = r3[:remaining]
    result.extend(take)
    remaining -= len(take)

    # R4: recent episodic
    if remaining <= 0:
        return result
    take = r4[:remaining]
    result.extend(take)

    return result


# ── Phase 3 预留 ──────────────────────────────────────────
# def context_aware_update(current_embedding, memories) -> list[tuple[str, float]]:
#     每轮用户输入后调用；相似度 > SIMILARITY_HIGH 强激活（+ACTIVATION_HIGH），
#     > SIMILARITY_MID 轻激活（+ACTIVATION_MID）。参数届时进 config.py。
