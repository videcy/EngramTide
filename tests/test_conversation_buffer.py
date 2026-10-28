"""Hook 对话缓冲表（conversation_turns）的存储契约。"""

from datetime import timedelta

import pytest

import core.memory_store as memory_store
from core.maintenance import run_maintenance
from core.memory_store import (
    latest_unanswered_prompt_id,
    list_pending_turns,
    mark_turns_dehydrated,
    memory_stats,
    pending_turn_summary,
    prune_dehydrated_turns,
    upsert_turn,
)
from utils.time_utils import utc_now


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    memory_store.close_db()
    monkeypatch.setattr(memory_store, "DB_PATH", tmp_path / "turns.db")
    memory_store.init_db()
    yield
    memory_store.close_db()


def test_user_and_assistant_rows_keep_conversation_order():
    upsert_turn("s1", "p1", "user", "第一句")
    upsert_turn("s1", "p1", "assistant", "第一句的回复")
    upsert_turn("s1", "p2", "user", "第二句")

    rows = list_pending_turns("s1")
    assert [(r["prompt_id"], r["role"]) for r in rows] == [
        ("p1", "user"),
        ("p1", "assistant"),
        ("p2", "user"),
    ]


def test_repeated_stop_keeps_last_reply_without_duplicating():
    upsert_turn("s1", "p1", "user", "问题")
    upsert_turn("s1", "p1", "assistant", "中间回复")
    upsert_turn("s1", "p1", "assistant", "最终回复")

    rows = list_pending_turns("s1")
    assert len(rows) == 2
    assert rows[1]["content"] == "最终回复"


def test_dehydrated_rows_are_not_rewritten():
    upsert_turn("s1", "p1", "assistant", "已脱水的回复")
    mark_turns_dehydrated([r["seq"] for r in list_pending_turns("s1")])

    upsert_turn("s1", "p1", "assistant", "迟到的改写")

    conn = memory_store._get_conn()
    content = conn.execute(
        "SELECT content FROM conversation_turns WHERE prompt_id = 'p1'"
    ).fetchone()[0]
    assert content == "已脱水的回复"
    assert list_pending_turns("s1") == []


def test_latest_unanswered_prompt_id():
    upsert_turn("s1", "p1", "user", "a")
    upsert_turn("s1", "p1", "assistant", "b")
    assert latest_unanswered_prompt_id("s1") is None

    upsert_turn("s1", "p2", "user", "c")
    assert latest_unanswered_prompt_id("s1") == "p2"
    assert latest_unanswered_prompt_id("other") is None


def test_mark_is_idempotent_and_summary_tracks_pending():
    upsert_turn("s1", "p1", "user", "a")
    upsert_turn("s2", "p1", "user", "b")
    upsert_turn("s2", "p2", "user", "c")

    summary = {row["session_id"]: row["pending"] for row in pending_turn_summary()}
    assert summary == {"s1": 1, "s2": 2}

    seqs = [r["seq"] for r in list_pending_turns("s2")]
    assert mark_turns_dehydrated(seqs) == 2
    assert mark_turns_dehydrated(seqs) == 0
    assert [r["session_id"] for r in list_pending_turns()] == ["s1"]

    stats = memory_stats()
    assert stats["buffered_turns"] == 3
    assert stats["pending_turns"] == 1


def test_prune_only_removes_old_dehydrated_rows():
    upsert_turn("s1", "p1", "user", "旧的已脱水")
    upsert_turn("s1", "p2", "user", "旧的未脱水")
    mark_turns_dehydrated([list_pending_turns("s1")[0]["seq"]])

    assert prune_dehydrated_turns(retention_days=7) == 0
    future = utc_now() + timedelta(days=8)
    assert prune_dehydrated_turns(retention_days=7, now=future) == 1

    remaining = list_pending_turns("s1")
    assert [r["prompt_id"] for r in remaining] == ["p2"]
    assert prune_dehydrated_turns(retention_days=0, now=future) == 0


def test_forced_maintenance_prunes_buffer(monkeypatch):
    upsert_turn("s1", "p1", "user", "a")
    mark_turns_dehydrated([list_pending_turns("s1")[0]["seq"]])
    monkeypatch.setattr("config.HOOK_BUFFER_RETENTION_DAYS", 1)

    report = run_maintenance(now=utc_now() + timedelta(days=2), force=True)

    assert report.pruned_turns == 1
    assert memory_stats()["buffered_turns"] == 0
