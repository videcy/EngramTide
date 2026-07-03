"""pytest 配置 — 启用 asyncio auto 模式。"""
import pytest


def pytest_configure(config):
    """注册 asyncio mark。"""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
