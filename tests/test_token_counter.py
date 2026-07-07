"""
Phase 4 — Token 估算器单元测试。

覆盖：空串 / 纯中文 / 纯英文 / 中英混合 / emoji / 单调性 / 异常输入。
"""

import math

import pytest

from utils.token_counter import estimate_tokens, _is_cjk
from config import TOKEN_EST_CJK, TOKEN_EST_OTHER, TOKEN_EST_SAFETY


class TestIsCjk:
    """CJK 字符判定辅助函数。"""

    def test_common_cjk(self):
        assert _is_cjk(ord("中"))
        assert _is_cjk(ord("文"))
        assert _is_cjk(ord("日"))
        assert _is_cjk(ord("本"))

    def test_common_ascii_not_cjk(self):
        assert not _is_cjk(ord("a"))
        assert not _is_cjk(ord("Z"))
        assert not _is_cjk(ord("0"))
        assert not _is_cjk(ord(" "))

    def test_cjk_punctuation(self):
        """全角标点应算 CJK。"""
        assert _is_cjk(ord("，"))
        assert _is_cjk(ord("。"))
        assert _is_cjk(ord("！"))
        assert _is_cjk(ord("？"))


class TestEstimateTokens:
    """estimate_tokens 主测试。"""

    def test_empty_string_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_pure_chinese(self):
        """纯中文：N 字 → ceil(0.6 * N * 1.1)"""
        n = 100
        expected = math.ceil(n * TOKEN_EST_CJK * TOKEN_EST_SAFETY)
        assert estimate_tokens("测" * n) == expected

    def test_pure_english(self):
        """纯 ASCII：N 字 → ceil(0.3 * N * 1.1)"""
        n = 100
        expected = math.ceil(n * TOKEN_EST_OTHER * TOKEN_EST_SAFETY)
        assert estimate_tokens("a" * n) == expected

    def test_mixed_cjk_and_english(self):
        """中英混合。"""
        text = "用户在做 AI 记忆系统项目，技术栈 Python + SQLite"
        cjk = sum(1 for ch in text if _is_cjk(ord(ch)))
        other = len(text) - cjk
        expected = math.ceil((cjk * TOKEN_EST_CJK + other * TOKEN_EST_OTHER) * TOKEN_EST_SAFETY)
        assert estimate_tokens(text) == expected

    def test_emoji_does_not_crash(self):
        """emoji 不崩溃，按 other 系数估算。"""
        result = estimate_tokens("😀🎉💻")
        assert isinstance(result, int)
        assert result >= 0

    def test_newlines_and_whitespace(self):
        """换行和空格按 other 计数。"""
        text = "hello\nworld\t  "
        result = estimate_tokens(text)
        assert result > 0

    def test_monotonic_append(self):
        """追加字符后估算值不减。"""
        base = "用户在做 AI 项目"
        base_est = estimate_tokens(base)
        longer_est = estimate_tokens(base + "，技术栈 Python")
        assert longer_est >= base_est

    def test_non_string_input(self):
        """非字符串类型不崩溃。"""
        assert estimate_tokens(12345) >= 0
        assert estimate_tokens(None) >= 0  # None → str(None) → "None" 有 4 个字符

    def test_typical_memory_entry(self):
        """典型记忆条目（中文为主 + 少量英文术语）。"""
        text = "用户在做 AI 记忆系统项目，使用 Python 和 SQLite，对 Phase 3 的检索效果满意"
        result = estimate_tokens(text)
        # 常识检查：40+ 字符中文为主，估算 token 应在 20-40 之间
        assert 15 <= result <= 50

    def test_short_phrase(self):
        """短文本。"""
        assert estimate_tokens("你好") > 0
        assert estimate_tokens("OK") > 0


# ── Phase 4 次要项修复回归：异常输入兜底（计划 §10）──────────


def test_non_string_input_warns_and_estimates(caplog):
    """非字符串输入 → warning + 转换后估算（不再静默处理）。"""
    import logging

    from utils.token_counter import estimate_tokens

    with caplog.at_level(logging.WARNING, logger="utils.token_counter"):
        result = estimate_tokens(12345)
    assert result > 0                       # "12345" 5 个 ASCII 字符
    assert any("非字符串" in r.message for r in caplog.records)


def test_unconvertible_input_falls_back_to_len():
    """str() 失败时按 len() × CJK 系数兜底（宁高勿低），而非返回 0。"""
    import math

    from config import TOKEN_EST_CJK, TOKEN_EST_SAFETY
    from utils.token_counter import estimate_tokens

    class Weird:
        def __str__(self):
            raise RuntimeError("不可转换")

        def __len__(self):
            return 10

    result = estimate_tokens(Weird())
    assert result == math.ceil(10 * TOKEN_EST_CJK * TOKEN_EST_SAFETY)


def test_no_len_no_str_returns_zero():
    """连长度都取不到才返回 0。"""
    from utils.token_counter import estimate_tokens

    class Hopeless:
        def __str__(self):
            raise RuntimeError

    assert estimate_tokens(Hopeless()) == 0
