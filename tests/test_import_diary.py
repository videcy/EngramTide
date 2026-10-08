"""scripts/import_diary.py：日记解析、预衰减、计划生成与按计划写库。"""

import importlib.util
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

import core.memory_store as memory_store
from core.memory_store import Memory

_SPEC = importlib.util.spec_from_file_location(
    "import_diary", Path(__file__).resolve().parent.parent / "scripts" / "import_diary.py"
)
import_diary = importlib.util.module_from_spec(_SPEC)
sys.modules["import_diary"] = import_diary  # dataclass 需要能在 sys.modules 里找到模块
_SPEC.loader.exec_module(import_diary)

TZ = ZoneInfo("Asia/Shanghai")


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    memory_store.close_db()
    monkeypatch.setattr(memory_store, "DB_PATH", tmp_path / "import.db")
    yield
    memory_store.close_db()


def write(dir_: Path, name: str, text: str) -> Path:
    path = dir_ / name
    path.write_text(text, encoding="utf-8")
    return path


# ── 解析 ─────────────────────────────────────────────────


def test_parse_sections_times_and_ids(tmp_path):
    write(tmp_path, "2026-09-20.md", (
        "开头没有标题的一段\n\n"
        "## 10:58 雨后去公园散步\n\n正文一\n第二行\n\n"
        "## 补充\n沿用上一段时间\n\n"
        "## 21:54 晚上\n\n"  # 空正文的段被丢弃
    ))
    write(tmp_path, "notes.md", "## 09:00 不是日记文件\n内容")

    entries = import_diary.load_entries(tmp_path, TZ)

    assert [(e.source_id, e.time, e.title) for e in entries] == [
        ("diary:2026-09-20Tna", None, ""),
        ("diary:2026-09-20T10:58", "10:58", "雨后去公园散步"),
        ("diary:2026-09-20T10:58#2", "10:58", "补充"),
    ]
    assert entries[1].body == "正文一\n第二行"
    # 北京时间 10:58 = UTC 02:58；无时间的段记为当天 12:00 → UTC 04:00
    assert entries[1].occurred_at == datetime(2026, 9, 20, 2, 58, tzinfo=timezone.utc)
    assert entries[0].occurred_at == datetime(2026, 9, 20, 4, 0, tzinfo=timezone.utc)


def test_entry_message_is_written_by_me(tmp_path):
    write(tmp_path, "2026-09-20.md", "## 10:58 标题\n正文")
    [entry] = import_diary.load_entries(tmp_path, TZ)

    message = import_diary.entry_message(entry)

    assert message == {"role": "assistant", "content": "【日记 2026-09-20 10:58 标题】\n正文"}


def test_prior_weight_decays_only_decaying_types():
    now = datetime(2026, 10, 8, tzinfo=timezone.utc)
    ten_days_ago = datetime(2026, 9, 28, tzinfo=timezone.utc)

    assert import_diary.prior_weight("semantic", 0.0, ten_days_ago, now) == 1.0
    assert import_diary.prior_weight("procedural", 0.0, ten_days_ago, now) == 1.0
    assert import_diary.prior_weight("episodic", 0.0, ten_days_ago, now) == pytest.approx(
        math.exp(-0.05 * 10)
    )
    calm = import_diary.prior_weight("emotional", 0.0, ten_days_ago, now)
    intense = import_diary.prior_weight("emotional", 0.9, ten_days_ago, now)
    assert intense > calm
    assert import_diary.prior_weight("episodic", 0.0, now, now) == 1.0


# ── plan ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_records_memories_and_isolates_failures(monkeypatch, tmp_path):
    write(tmp_path, "2026-09-20.md", "## 10:58 好\n正文\n## 11:00 坏\n正文")
    entries = import_diary.load_entries(tmp_path, TZ)

    async def fake_segment(messages, source_id):
        if source_id.endswith("11:00"):
            raise RuntimeError("LLM down")
        return [Memory(memory_id="x", type="episodic", content="她学会了做番茄炒蛋",
                       valence=0.3, arousal=0.2, tags=["生活"])]

    monkeypatch.setattr(import_diary, "_dehydrate_segment", fake_segment)

    records = await import_diary.build_plan(entries, concurrency=2)

    assert records[0]["memories"] == [{
        "content": "她学会了做番茄炒蛋", "type": "episodic", "valence": 0.3,
        "arousal": 0.2, "unresolved": False, "tags": ["生活"],
    }]
    assert records[1]["error"] == "LLM down"
    md = import_diary.render_plan_markdown({
        "diary_dir": str(tmp_path), "prompt": "p", "entries": records,
    })
    assert "她学会了做番茄炒蛋" in md and "提取失败：LLM down" in md


# ── apply ────────────────────────────────────────────────


def make_plan(entries):
    return {"kind": "diary-import-plan", "entries": entries}


def plan_entry(source_id, occurred_at, memories, error=None):
    return {"source_id": source_id, "date": occurred_at[:10], "time": None, "title": "",
            "occurred_at": occurred_at, "chars": 1, "memories": memories, "error": error}


def mem(content, type_="episodic", arousal=0.0, tags=()):
    return {"content": content, "type": type_, "valence": 0.0, "arousal": arousal,
            "unresolved": False, "tags": list(tags)}


@pytest.mark.asyncio
async def test_apply_backdates_predecays_and_is_idempotent(monkeypatch):
    axis = iter(range(64))

    async def fake_embed(_text):
        vec = np.zeros(64, dtype=np.float32)
        vec[next(axis)] = 1.0
        return vec

    monkeypatch.setattr(import_diary, "embed_text", fake_embed)
    now = datetime(2026, 10, 8, 4, 0, tzinfo=timezone.utc)
    plan = make_plan([
        plan_entry("diary:2026-09-28T12:00", "2026-09-28 04:00:00",
                   [mem("她在准备期末考试", tags=["学业"]), mem("她是研究生", "semantic")]),
        plan_entry("diary:2026-09-29T12:00", "2026-09-29 04:00:00", [], error="boom"),
    ])

    first = await import_diary.apply_plan(plan, now=now)
    again = await import_diary.apply_plan(plan, now=now)

    assert (first["entries"], first["inserted"], first["failed_entries"]) == (1, 2, 1)
    assert (again["entries"], again["skipped"], again["inserted"]) == (0, 1, 0)
    rows = {m.content: m for m in memory_store.list_active_memories()}
    episodic = rows["她在准备期末考试"]
    assert episodic.created_at == "2026-09-28 04:00:00"
    assert episodic.source_conv_id == "diary:2026-09-28T12:00"
    assert episodic.tags == ["学业", "diary"]
    assert episodic.decay_weight == pytest.approx(math.exp(-0.05 * 10))
    assert rows["她是研究生"].decay_weight == 1.0


@pytest.mark.asyncio
async def test_apply_clears_stale_unresolved(monkeypatch):
    async def fake_embed(text):
        vec = np.zeros(64, dtype=np.float32)
        vec[0 if "旧" in text else 1] = 1.0
        return vec

    monkeypatch.setattr(import_diary, "embed_text", fake_embed)
    now = datetime(2026, 10, 8, 4, 0, tzinfo=timezone.utc)
    stale = dict(mem("我明天要问她旧事"), unresolved=True)
    fresh = dict(mem("我答应周末提醒她"), unresolved=True)
    await import_diary.apply_plan(make_plan([
        plan_entry("diary:old", "2026-09-16 04:00:00", [stale]),
        plan_entry("diary:new", "2026-10-07 04:00:00", [fresh]),
    ]), now=now, unresolved_days=3)

    rows = {m.content: m.unresolved for m in memory_store.list_active_memories()}
    assert rows == {"我明天要问她旧事": False, "我答应周末提醒她": True}
