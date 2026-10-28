"""PROMPTS_OVERRIDE_DIR：同名文件优先，缺失时回落 prompts/。"""

import config
from core.dehydrator import _load_prompt


def test_override_file_wins_and_missing_falls_back(monkeypatch, tmp_path):
    (tmp_path / "dehydrate.txt").write_text("覆盖版", encoding="utf-8")
    monkeypatch.setattr(config, "PROMPTS_OVERRIDE_DIR", tmp_path)

    assert config.prompt_path("dehydrate.txt") == tmp_path / "dehydrate.txt"
    assert _load_prompt("dehydrate.txt") == "覆盖版"
    assert config.prompt_path("topic_split.txt") == config.PROMPTS_DIR / "topic_split.txt"


def test_no_override_uses_default(monkeypatch):
    monkeypatch.setattr(config, "PROMPTS_OVERRIDE_DIR", None)
    assert config.prompt_path("dehydrate.txt") == config.PROMPTS_DIR / "dehydrate.txt"
