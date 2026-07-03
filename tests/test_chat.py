"""
测试 chat 层的本地保护逻辑。
"""

import logging

from core.chat import _warn_if_prohibited_memory_phrase


def test_warn_if_prohibited_memory_phrase(caplog):
    """回复包含机械记忆表述时记录 warning。"""
    with caplog.at_level(logging.WARNING):
        _warn_if_prohibited_memory_phrase("根据我的记忆，你在做 AI 项目。")

    assert "回复包含禁用机械表述" in caplog.text
    assert "根据我的记忆" in caplog.text


def test_no_warning_for_natural_reply(caplog):
    """自然回复不记录 warning。"""
    with caplog.at_level(logging.WARNING):
        _warn_if_prohibited_memory_phrase("你是在做 AI 项目，我们继续推进。")

    assert "回复包含禁用机械表述" not in caplog.text
