"""LLM 输出上限：走配置；推理模型截断时给出明确警告。"""

import logging

import httpx
import pytest

import config
from core.dehydrator import _dehydrate_segment


class _Response:
    status_code = 200
    request = None

    def __init__(self, content, finish_reason):
        self._data = {"choices": [{
            "message": {"content": content},
            "finish_reason": finish_reason,
        }]}

    def json(self):
        return self._data


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def install(content, finish_reason="stop"):
        async def fake_post(self, url, headers=None, json=None):
            calls.append(json)
            return _Response(content, finish_reason)

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        return calls

    return install


@pytest.mark.asyncio
async def test_dehydrate_uses_configured_cap(captured, monkeypatch):
    monkeypatch.setattr(config, "DEHYDRATE_MAX_TOKENS", 12345)
    calls = captured("[]")

    assert await _dehydrate_segment([{"role": "user", "content": "晚安"}], "c") == []
    assert calls[0]["max_tokens"] == 12345


@pytest.mark.asyncio
async def test_truncated_output_warns_before_failing(captured, caplog):
    captured('[{"content": "写到一半', finish_reason="length")

    with caplog.at_level(logging.WARNING, logger="core.dehydrator"):
        with pytest.raises(ValueError):
            await _dehydrate_segment([{"role": "user", "content": "长对话"}], "c")

    assert "finish_reason=length" in caplog.text


@pytest.mark.asyncio
async def test_null_content_is_treated_as_empty(captured):
    captured(None, finish_reason="length")

    with pytest.raises(ValueError):
        await _dehydrate_segment([{"role": "user", "content": "长对话"}], "c")
