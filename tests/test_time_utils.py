"""
测试时间工具模块：parse_db_timestamp、hours_between、days_between、utc_now。
"""

from datetime import datetime, timezone, timedelta

import pytest

from utils.time_utils import (
    parse_db_timestamp,
    hours_between,
    days_between,
    utc_now,
)


class TestParseDbTimestamp:
    """解析数据库时间字符串。"""

    def test_sqlite_format_returns_aware_utc(self):
        """SQLite CURRENT_TIMESTAMP 格式 "YYYY-MM-DD HH:MM:SS" → aware UTC。"""
        dt = parse_db_timestamp("2026-07-03 08:15:42")
        assert dt.tzinfo is not None
        assert dt == datetime(2026, 7, 3, 8, 15, 42, tzinfo=timezone.utc)

    def test_iso_with_timezone_no_double_offset(self):
        """ISO 8601 带 +00:00 → 不重复偏移。"""
        dt = parse_db_timestamp("2026-07-03T08:15:42+00:00")
        assert dt == datetime(2026, 7, 3, 8, 15, 42, tzinfo=timezone.utc)

    def test_iso_with_positive_offset_converts_to_utc(self):
        """ISO 8601 带 +09:00 → 转为 UTC。"""
        dt = parse_db_timestamp("2026-07-03T08:15:42+09:00")
        expected = datetime(2026, 7, 2, 23, 15, 42, tzinfo=timezone.utc)
        assert dt == expected

    def test_iso_naive_treated_as_utc(self):
        """ISO 8601 不带时区 → 视为 UTC。"""
        dt = parse_db_timestamp("2026-07-03T08:15:42")
        assert dt.tzinfo is not None
        assert dt == datetime(2026, 7, 3, 8, 15, 42, tzinfo=timezone.utc)

    def test_space_separated_with_seconds(self):
        """空格分隔、精确到秒。"""
        dt = parse_db_timestamp("2026-01-01 00:00:00")
        assert dt == datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_invalid_string_raises_valueerror(self):
        """非法字符串抛 ValueError。"""
        with pytest.raises(ValueError, match="无法解析时间字符串"):
            parse_db_timestamp("not-a-date")

    def test_empty_string_raises(self):
        """空字符串抛 ValueError。"""
        with pytest.raises(ValueError):
            parse_db_timestamp("")

    def test_garbled_string_raises(self):
        """乱码抛 ValueError。"""
        with pytest.raises(ValueError):
            parse_db_timestamp("abc123!@#")


class TestHoursBetween:
    """小时差计算。"""

    def test_one_hour(self):
        now = utc_now()
        one_hour_ago = now - timedelta(hours=1)
        assert abs(hours_between(one_hour_ago, now) - 1.0) < 0.01

    def test_zero_hours(self):
        now = utc_now()
        assert hours_between(now, now) == 0.0

    def test_negative_when_earlier_after_later(self):
        now = utc_now()
        earlier = now - timedelta(hours=1)
        # earlier < later → negative
        assert hours_between(now, earlier) < 0


class TestDaysBetween:
    """天数差计算（浮点，不做整数截断）。"""

    def test_71_hours_not_truncated_to_2_days(self):
        """71 小时应返回 ~2.958 天，而非 2 天。"""
        d1 = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
        d2 = datetime(2026, 7, 3, 23, 0, 0, tzinfo=timezone.utc)
        days = days_between(d1, d2)
        assert days > 2.9
        assert days < 3.0

    def test_exactly_one_day(self):
        d1 = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
        d2 = datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)
        assert days_between(d1, d2) == 1.0

    def test_six_hours(self):
        d1 = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
        d2 = datetime(2026, 7, 1, 6, 0, 0, tzinfo=timezone.utc)
        assert abs(days_between(d1, d2) - 0.25) < 0.01


class TestUtcNow:
    """utc_now 基础检查。"""

    def test_returns_aware_datetime(self):
        now = utc_now()
        assert now.tzinfo is not None
        assert now.tzinfo == timezone.utc

    def test_returns_current_time(self):
        # before 必须在 utc_now() 之前取样，否则时钟前进时 before > now 造成 flaky 失败
        before = datetime.now(timezone.utc)
        now = utc_now()
        after = datetime.now(timezone.utc)
        # 在调用前后范围内
        assert before <= now <= after
