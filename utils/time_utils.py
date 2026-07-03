"""
Phase 2 — 时间工具模块。

统一处理 SQLite UTC 时间字符串与 Python datetime 的转换。
这是 Phase 2 最容易踩坑的地方，必须集中封装。

背景：
- SQLite CURRENT_TIMESTAMP 产出 UTC，格式 "YYYY-MM-DD HH:MM:SS"（空格分隔、无时区）。
- datetime.fromisoformat() 可解析该格式（Python 3.7+），但结果是 naive datetime。
- 所有时间比较必须在 aware UTC 下进行，否则跨时区会出错。
"""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """返回当前 aware UTC 时间。全项目禁止直接调用 datetime.now()。"""
    return datetime.now(timezone.utc)


def parse_db_timestamp(ts: str) -> datetime:
    """
    解析数据库时间字符串为 aware UTC datetime。

    兼容格式：
    - "YYYY-MM-DD HH:MM:SS"（SQLite CURRENT_TIMESTAMP，空格分隔）
    - ISO 8601 带时区（如 "2026-07-03T08:15:42+00:00"）
    - ISO 8601 不带时区（退化为 naive，视为 UTC）

    naive 输入一律视为 UTC 并补上 tzinfo。
    无法解析时抛 ValueError（调用方决定降级策略）。
    """
    ts = ts.strip()
    if not ts:
        raise ValueError("时间字符串为空")

    # 尝试 fromisoformat（Python 3.7+ 支持空格分隔的 ISO-like 格式）
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        # 尝试替换空格为 T 再解析
        try:
            dt = datetime.fromisoformat(ts.replace(" ", "T"))
        except ValueError:
            raise ValueError(f"无法解析时间字符串: {ts!r}")

    # 补全时区：naive 视为 UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        # 统一转为 UTC
        dt = dt.astimezone(timezone.utc)

    return dt


def hours_between(earlier: datetime, later: datetime) -> float:
    """两个 aware datetime 之间的小时数（later - earlier），可为负。"""
    delta = later - earlier
    return delta.total_seconds() / 3600.0


def days_between(earlier: datetime, later: datetime) -> float:
    """
    两个 aware datetime 之间的天数（浮点，不做 .days 截断）。

    71 小时应返回 ≈2.958 而非 2，避免衰减计算被整数截断。
    """
    delta = later - earlier
    return delta.total_seconds() / 86400.0
