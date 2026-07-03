"""
Phase 2 手动验收 — 5 场景端到端模拟。

不依赖 LLM / embedding API，直接用确定性 fake embedding 验证核心逻辑。
"""
import sys
import uuid
import os
import tempfile
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.memory_store import (
    Memory,
    close_db,
    init_db,
    insert_memory,
    list_active_memories,
    set_meta,
)
from core.decay import run_decay_update, get_surfaced_memories
from core.memory_writer import write_memories


def _fake_embed(text: str) -> np.ndarray:
    seed = sum(ord(c) for c in text)
    rng = np.random.RandomState(seed)
    vec = rng.randn(128).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _make_mem(mem_type: str, content: str, **kw) -> Memory:
    return Memory(
        memory_id=str(uuid.uuid4()),
        type=mem_type,
        content=content,
        embedding=_fake_embed(content),
        **kw,
    )


def _reset_db(db_path: str):
    import core.memory_store as ms
    ms.DB_PATH = db_path
    ms._connection = None
    init_db()


async def main():
    db_file = os.path.join(tempfile.mkdtemp(), "acceptance.db")
    passed = 0
    total = 5

    # ════════════════════════════════════════════════════════
    # 场景 1: semantic 覆盖
    # ════════════════════════════════════════════════════════
    print("=" * 60)
    print("场景 1: semantic 覆盖检测")
    _reset_db(db_file)
    # 模拟 LLM 脱水产出的语义覆盖场景：
    # 用户搬家 → 脱水产出内容 slightly different，但真实 embedding 会给出高相似度。
    # 我们的 fake embedding 对相同内容产生相同向量（similarity=1.0），
    # 对稍微不同的文本则几乎正交。因此用完全相同内容来模拟"高相似度覆盖"场景。
    # 在实际场景中，真实 embedding 模型会对"住在东京"和"搬到大阪"产生高相似度。
    old = _make_mem("semantic", "用户的居住地是东京")
    insert_memory(old)
    # 脱水产出略有差异但语义高度重叠 → 真实 embedding 会高相似
    # 这里用完全相同内容保证 fake embedding 的相似度为 1.0
    new = _make_mem("semantic", "用户的居住地是东京")
    report = await write_memories([new])

    # 查 DB 确认 superseded_by
    import core.memory_store as ms
    row = ms._get_conn().execute(
        "SELECT superseded_by FROM memories WHERE memory_id=?",
        (old.memory_id,),
    ).fetchone()

    if row and row["superseded_by"] == new.memory_id:
        print("  ✅ 旧记忆 superseded_by → 新记忆")
        passed += 1
    else:
        val = row["superseded_by"] if row else "N/A"
        print(f"  ❌ superseded_by={val}, 预期={new.memory_id}")

    # ════════════════════════════════════════════════════════
    # 场景 2: procedural 永驻浮现
    # ════════════════════════════════════════════════════════
    print("=" * 60)
    print("场景 2: procedural 永驻浮现")
    db_file2 = os.path.join(tempfile.mkdtemp(), "acceptance2.db")
    _reset_db(db_file2)
    proc = _make_mem("procedural", "用户吐槽作业时，先共情，不要立刻讲大道理")
    insert_memory(proc)
    insert_memory(_make_mem("episodic", "今天吃了火锅"))
    memories = list_active_memories()
    surfaced = get_surfaced_memories(memories, max_surfaced=8)
    proc_surfaced = [m for m in surfaced if m.type == "procedural"]
    if len(proc_surfaced) >= 1 and proc.content in [m.content for m in proc_surfaced]:
        print("  ✅ procedural 记忆通过浮现进入上下文")
        passed += 1
    else:
        print(f"  ❌ 浮现记忆中 procedural 数量={len(proc_surfaced)}")

    # ════════════════════════════════════════════════════════
    # 场景 3: emotional 强化而非重复
    # ════════════════════════════════════════════════════════
    print("=" * 60)
    print("场景 3: emotional 强化写入")
    db_file3 = os.path.join(tempfile.mkdtemp(), "acceptance3.db")
    _reset_db(db_file3)
    old_emo = _make_mem("emotional", "组员做的图真的太丑了",
                        arousal=0.8, decay_weight=0.6, tags=["吐槽"])
    insert_memory(old_emo)
    # 相同内容 → embedding 完全一致 → 必定命中强化
    new_emo = _make_mem("emotional", "组员做的图真的太丑了",
                        arousal=0.9, tags=["愤怒"])
    report = await write_memories([new_emo])
    active = list_active_memories()
    emo_mems = [m for m in active if m.type == "emotional"]

    if len(emo_mems) == 1 and report.reinforced == 1 and report.inserted == 0:
        updated = emo_mems[0]
        if updated.decay_weight > 0.6:
            print(f"  ✅ emotional 强化: decay_weight {0.6}→{updated.decay_weight:.2f}, "
                  f"arousal={updated.arousal}, 无重复记忆")
            passed += 1
        else:
            print(f"  ❌ decay_weight 未提升: {updated.decay_weight}")
    else:
        print(f"  ❌ emotional 记忆数={len(emo_mems)}, "
              f"reinforced={report.reinforced}, inserted={report.inserted}")

    # ════════════════════════════════════════════════════════
    # 场景 4: unresolved 主动浮现
    # ════════════════════════════════════════════════════════
    print("=" * 60)
    print("场景 4: unresolved 主动浮现")
    db_file4 = os.path.join(tempfile.mkdtemp(), "acceptance4.db")
    _reset_db(db_file4)
    unresolved_mem = _make_mem("semantic", "下周要面试，好紧张",
                               unresolved=True, decay_weight=1.0)
    insert_memory(unresolved_mem)
    insert_memory(_make_mem("episodic", "今天天气不错"))
    memories = list_active_memories()
    surfaced = get_surfaced_memories(memories, max_surfaced=8)
    unres_surfaced = [m for m in surfaced if m.unresolved]
    if len(unres_surfaced) >= 1:
        print("  ✅ unresolved 记忆通过浮现进入上下文")
        passed += 1
    else:
        print("  ❌ unresolved 记忆未浮现")

    # ════════════════════════════════════════════════════════
    # 场景 5: 差异化衰减（模拟 30 天未交互）
    # ════════════════════════════════════════════════════════
    print("=" * 60)
    print("场景 5: 差异化衰减（模拟 30 天未交互）")
    db_file5 = os.path.join(tempfile.mkdtemp(), "acceptance5.db")
    _reset_db(db_file5)

    now = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    thirty_days_ago = now - timedelta(days=30)
    # 需要先 reset 再切换 DB 引用
    _reset_db(db_file5)
    set_meta("last_decay_run_at", thirty_days_ago.isoformat())

    insert_memory(_make_mem("semantic", "用户是程序员"))
    insert_memory(_make_mem("procedural", "吐槽时先共情"))
    insert_memory(_make_mem("episodic", "30天前吃了火锅"))
    insert_memory(_make_mem("emotional", "极度讨厌画大饼", arousal=0.9))

    report = run_decay_update(now=now)
    memories = list_active_memories()
    by_type = {}
    for m in memories:
        by_type[m.type] = m.decay_weight

    all_ok = True
    # semantic 不衰减
    ok1 = abs(by_type.get("semantic", 0) - 1.0) < 1e-6
    tag1 = "✅" if ok1 else "❌"
    print(f"  {tag1} semantic = {by_type.get('semantic', 0):.4f} (应 ≈1.0)")
    if not ok1:
        all_ok = False

    ok2 = abs(by_type.get("procedural", 0) - 1.0) < 1e-6
    tag2 = "✅" if ok2 else "❌"
    print(f"  {tag2} procedural = {by_type.get('procedural', 0):.4f} (应 ≈1.0)")
    if not ok2:
        all_ok = False

    epi_w = by_type.get("episodic", 1.0)
    ok3 = epi_w < 0.25
    tag3 = "✅" if ok3 else "❌"
    print(f"  {tag3} episodic = {epi_w:.4f} (应 < 0.25)")
    if not ok3:
        all_ok = False

    emo_w = by_type.get("emotional", 0)
    ok4 = emo_w > 0.5
    tag4 = "✅" if ok4 else "❌"
    print(f"  {tag4} emotional (arousal=0.9) = {emo_w:.4f} (应 > 0.5)")
    if not ok4:
        all_ok = False

    if all_ok:
        passed += 1

    # ════════════════════════════════════════════════════════
    print("=" * 60)
    print(f"\n验收结果: {passed}/{total} 通过")
    if passed == total:
        print("🎉 Phase 2 五场景手动验收全部通过！")
    else:
        print(f"⚠️ {total - passed} 个场景未通过，详见上方。")

    close_db()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
