"""
P0 + P4 — 记忆库维护：埋点保留期清理与冷归档。

定位：给「只增不减」的记忆库补上出口。当前系统里 DECAY_FLOOR=0.01 意味着权重
永不归零，只是沉底到检索不到；superseded 的旧记忆永久占位；写入速率上限是
MAX_TOPIC_SEGMENTS × DEHYDRATE_MAX_ITEMS = 64 条/会话，纯增量没有对冲。

治理后热表规模从「无上界线性增长」变成「由 ARCHIVE_IDLE_DAYS 窗口 + 访问行为
决定的稳态」。

调用时机：会话开始、衰减更新之后，按频率触发（默认每 ARCHIVE_RUN_EVERY_SESSIONS
次会话一次，计数记在 meta 里），避免每次启动都全表扫一遍。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import config
from core.memory_store import (
    archive_memories,
    collect_archive_candidates,
    get_meta,
    prune_dehydrated_turns,
    prune_event_logs,
    list_pending_links,
    purge_superseded,
    supersede_and_archive,
    update_link_status,
    get_memory,
    set_meta,
)
from utils.time_utils import days_between, parse_db_timestamp, utc_now

logger = logging.getLogger(__name__)

_SESSION_COUNTER_KEY = "maintenance_session_counter"
_LAST_RUN_KEY = "last_archive_run_at"


@dataclass
class MaintenanceReport:
    """一次维护的摘要。ran=False 表示本次会话没到触发频率。"""

    ran: bool = False
    archived: int = 0        # 移入归档表的条数
    purged: int = 0          # 硬删除的 superseded 条数
    pruned_events: int = 0   # 清理的埋点事件行数
    pruned_turns: int = 0    # 清理的已脱水对话缓冲行数
    expired_conflicts: int = 0  # 挂起过久被自动裁决的冲突关系数


def _next_session_count() -> int:
    """会话计数 +1 并返回新值。meta 值损坏时从 1 重新开始。"""
    raw = get_meta(_SESSION_COUNTER_KEY)
    try:
        count = int(raw) + 1 if raw is not None else 1
    except (TypeError, ValueError):
        count = 1
    set_meta(_SESSION_COUNTER_KEY, str(count))
    return count


def expire_pending_conflicts(now: datetime | None = None) -> int:
    """
    挂起超过 CONFLICT_PENDING_EXPIRE_DAYS 天的冲突自动裁决：created_at 较新者胜出，
    较旧者被覆盖并归档，关系标 resolved。任一条已不在热表（被归档/删除）时只标 resolved。
    CONFLICT_PENDING_EXPIRE_DAYS < 0 关闭。返回处理的关系数。
    """
    days = config.CONFLICT_PENDING_EXPIRE_DAYS
    if days < 0:
        return 0
    now = now or utc_now()
    handled = 0
    for link in list_pending_links():
        try:
            if days_between(parse_db_timestamp(link.created_at), now) < days:
                continue
        except ValueError:
            continue
        a, b = get_memory(link.src_id), get_memory(link.dst_id)
        if a is not None and b is not None and a.superseded_by is None and b.superseded_by is None:
            try:
                ta, tb = parse_db_timestamp(a.created_at), parse_db_timestamp(b.created_at)
            except ValueError:
                ta = tb = None
            if ta is not None and tb is not None:
                # 同时间时以 src（新记忆）为准
                newer, older = (b, a) if tb > ta else (a, b)
                supersede_and_archive([older.memory_id], newer.memory_id)
        update_link_status(link.id, "resolved", now=now)
        handled += 1
    return handled


def run_maintenance(
    now: datetime | None = None,
    force: bool = False,
) -> MaintenanceReport:
    """
    按频率执行维护。force=True 时无视频率立即执行（/maintenance 命令用）。

    顺序刻意如此：先清埋点（最便宜、最安全），再删过期的 supersede 链，
    最后归档沉底 episodic。任一步失败都只影响它自己，不回滚前面的成果。
    """
    now = now or utc_now()

    every = max(1, config.ARCHIVE_RUN_EVERY_SESSIONS)
    if not force:
        if _next_session_count() % every != 0:
            return MaintenanceReport(ran=False)

    report = MaintenanceReport(ran=True)

    # ① 埋点保留期清理（开着埋点时 decay_events 的行数会远超 memories 本身）
    try:
        report.pruned_events = prune_event_logs()
    except Exception as e:
        logger.warning("清理埋点事件失败: %s", e)

    # ①' 已脱水的对话原文过了保留期就删；未脱水的不动
    try:
        report.pruned_turns = prune_dehydrated_turns(now=now)
    except Exception as e:
        logger.warning("清理对话缓冲失败: %s", e)

    # ①'' 挂起过久的冲突自动裁决（较新者胜出）
    if config.CONFLICT_DETECT_ENABLED:
        try:
            report.expired_conflicts = expire_pending_conflicts(now=now)
        except Exception as e:
            logger.warning("处理过期挂起冲突失败: %s", e)

    if not config.ARCHIVE_ENABLED:
        set_meta(_LAST_RUN_KEY, now.isoformat())
        return report

    # ② supersede 链：已被新记忆完整取代，保留的唯一价值是审计追溯 → 过期硬删
    try:
        report.purged = purge_superseded(now=now)
    except Exception as e:
        logger.warning("清理 superseded 记忆失败: %s", e)

    # ③ 冷归档：只动最安全的那一类（沉底 + 从未被用过 + 长期无访问的 episodic）
    try:
        candidates = collect_archive_candidates(now=now)
        if candidates:
            report.archived = archive_memories(candidates, now=now)
    except Exception as e:
        logger.warning("归档沉底记忆失败: %s", e)

    set_meta(_LAST_RUN_KEY, now.isoformat())

    if report.archived or report.purged or report.pruned_events or report.pruned_turns or report.expired_conflicts:
        logger.info(
            "记忆库维护: 归档 %d 条, 硬删 supersede %d 条, 清理埋点 %d 行, 清理对话缓冲 %d 行, 过期冲突裁决 %d 条",
            report.archived, report.purged, report.pruned_events, report.pruned_turns,
            report.expired_conflicts,
        )
    return report


def search_archive(query_text: str, limit: int = 10):
    """
    显式翻查归档表（关键词子串匹配）。

    归档不是删除——数据仍在，只是不进热路径。这个接口让用户在需要时能把它翻出来，
    也是「归档比硬删安全」这句话的兑现方式。
    """
    from core.memory_store import _get_conn, _row_to_memory, _MEMORY_COLUMNS

    text = (query_text or "").strip()
    if not text:
        return []
    conn = _get_conn()
    rows = conn.execute(
        f"SELECT {_MEMORY_COLUMNS} FROM memories_archive "
        f"WHERE content LIKE ? ORDER BY archived_at DESC LIMIT ?",
        (f"%{text}%", limit),
    ).fetchall()
    return [_row_to_memory(r) for r in rows]
