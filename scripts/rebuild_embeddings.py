#!/usr/bin/env python3
"""
P3 迁移 — 全库 embedding 重建。

什么时候需要跑：
  - 改了 EMBEDDING_DIM（维度变了，新旧向量不可混用）
  - 换了 EMBEDDING_MODEL（向量空间变了，同理）
  - 从旧版本升级（旧向量没有 L2 归一化，矩阵乘检索需要归一化）

跑完之后**必须**再跑 scripts/calibrate_thresholds.py 重新标定
SIMILARITY_MID / SIMILARITY_HIGH。这两个阈值是空间相关的，换了维度还沿用旧值
等于把激活机制关掉了。

用法：
    python scripts/rebuild_embeddings.py            # 重建
    python scripts/rebuild_embeddings.py --dry-run  # 只统计，不调 API 不写库
    python scripts/rebuild_embeddings.py --batch 32 # 调整并发批大小

断点续传：逐批提交，中断后重跑会跳过已经是目标维度的记录（用 --force 强制全重建）。
归档表 memories_archive 同样会被重建——否则 restore_archived 会把旧维度的向量
搬回热表。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import config  # noqa: E402
from core import memory_store  # noqa: E402
from core.embedding import embed_text  # noqa: E402
from utils.time_utils import utc_now  # noqa: E402

TABLES = ("memories", "memories_archive")


def _rows_to_rebuild(conn, table: str, force: bool) -> list[tuple[str, str]]:
    """返回 [(memory_id, content)]。force=False 时跳过已是目标维度的行。"""
    target_bytes = config.EMBEDDING_DIM * 4
    sql = f"SELECT memory_id, content, length(embedding) AS n FROM {table}"
    rows = conn.execute(sql).fetchall()
    out = []
    for r in rows:
        if not force and r["n"] == target_bytes:
            continue
        if not (r["content"] or "").strip():
            continue
        out.append((r["memory_id"], r["content"]))
    return out


async def _embed_batch(contents: list[str]) -> list[np.ndarray | None]:
    """并发 embed 一批文本。单条失败返回 None，不牵连整批。"""
    async def one(text: str):
        try:
            return await embed_text(text)
        except Exception as e:  # noqa: BLE001 — 单条失败要能继续
            print(f"    ⚠ embedding 失败，跳过该条: {e}")
            return None

    return await asyncio.gather(*(one(c) for c in contents))


async def rebuild(batch_size: int, dry_run: bool, force: bool) -> int:
    memory_store.init_db()
    conn = memory_store._get_conn()

    total_done = 0
    for table in TABLES:
        try:
            targets = _rows_to_rebuild(conn, table, force)
        except Exception as e:  # 归档表可能不存在于极旧的库
            print(f"跳过 {table}: {e}")
            continue

        print(f"\n[{table}] 待重建 {len(targets)} 条 → {config.EMBEDDING_DIM} 维")
        if dry_run or not targets:
            continue

        for start in range(0, len(targets), batch_size):
            chunk = targets[start : start + batch_size]
            vectors = await _embed_batch([c for _, c in chunk])

            updates = [
                (vec.astype(np.float32).tobytes(), mid)
                for (mid, _), vec in zip(chunk, vectors)
                if vec is not None
            ]
            if updates:
                with conn:
                    conn.executemany(
                        f"UPDATE {table} SET embedding = ? WHERE memory_id = ?",
                        updates,
                    )
                total_done += len(updates)
            print(
                f"  {min(start + batch_size, len(targets))}/{len(targets)} "
                f"（本批写入 {len(updates)} 条）"
            )

    if not dry_run:
        memory_store.set_meta("embedding_dim", str(config.EMBEDDING_DIM))
        memory_store.set_meta("embedding_model", config.EMBEDDING_MODEL)
        memory_store.set_meta("embeddings_rebuilt_at", utc_now().isoformat())
        memory_store.rebuild_fts_index()
        memory_store._bump_data_version()

    return total_done


def main() -> None:
    parser = argparse.ArgumentParser(description="重建全库 embedding")
    parser.add_argument("--batch", type=int, default=16, help="每批并发条数")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不调 API")
    parser.add_argument(
        "--force", action="store_true", help="连已是目标维度的记录也重建"
    )
    args = parser.parse_args()

    errors = [e for e in config.check_config() if "embedding 维度不匹配" not in e]
    if errors:
        for e in errors:
            print(f"❌ {e}")
        sys.exit(1)

    print(
        f"模型 {config.EMBEDDING_MODEL} · 目标维度 {config.EMBEDDING_DIM} · "
        f"库 {memory_store.DB_PATH}"
    )
    done = asyncio.run(rebuild(args.batch, args.dry_run, args.force))

    if args.dry_run:
        print("\n(dry-run，未写入任何数据)")
        return

    print(f"\n✅ 重建完成，共写入 {done} 条。")
    print(
        "⚠ 下一步必须重标阈值：python scripts/calibrate_thresholds.py\n"
        "  SIMILARITY_MID / SIMILARITY_HIGH 是空间相关的，沿用旧值等于关掉激活机制。"
    )
    memory_store.close_db()


if __name__ == "__main__":
    main()
