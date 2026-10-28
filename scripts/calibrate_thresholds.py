#!/usr/bin/env python3
"""
P3 配套 — SIMILARITY_MID / SIMILARITY_HIGH 重标定。

协议（与 tests/fixtures/similarity_calibration_pairs.md 里记录的一致）：
    P95_unrelated = 无关对相似度的 95 分位
    P25_related   = 相关对相似度的 25 分位
    SIMILARITY_HIGH = P25_related              （相关对里最保守的那 25%）
    SIMILARITY_MID  = (P95_unrelated + P25_related) / 2   （间隙中点）

为什么必须重跑：这两个阈值是**空间相关**的。改了 EMBEDDING_DIM 或换了模型之后
沿用旧值，轻则轻激活带 (MID, HIGH] 变成空集，重则每条记忆都被强激活——两种都是
「激活机制静默失效」，不会报错。

样本来自 tests/fixtures/similarity_calibration_pairs.md 的两张表（相关对 / 无关对）。
需要真实 embedding API，会产生调用成本，因此不进 CI。

用法：
    python scripts/calibrate_thresholds.py
    python scripts/calibrate_thresholds.py --pairs path/to/pairs.md
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import config  # noqa: E402
from core.embedding import embed_text  # noqa: E402

DEFAULT_PAIRS = (
    Path(__file__).resolve().parent.parent
    / "tests" / "fixtures" / "similarity_calibration_pairs.md"
)


def parse_pairs(path: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    从 markdown 里解出「相关对」与「无关对」两张表。

    表格写法在文件里并不完全统一（有的单元格自带引号、分隔符前后空格不定），
    所以这里按行拆 `|`、取第 2、3 列，再剥掉包裹的引号。
    """
    related: list[tuple[str, str]] = []
    unrelated: list[tuple[str, str]] = []
    bucket: list[tuple[str, str]] | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            if "相关对" in stripped and "无关" not in stripped:
                bucket = related
            elif "无关对" in stripped:
                bucket = unrelated
            else:
                bucket = None
            continue

        if bucket is None or not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 3:
            continue
        if not re.fullmatch(r"\d+", cells[0]):   # 跳过表头与分隔行
            continue
        a, b = (c.strip().strip('"').strip("“”").strip() for c in cells[1:3])
        if a and b:
            bucket.append((a, b))

    return related, unrelated


async def similarities(pairs: list[tuple[str, str]]) -> np.ndarray:
    """算每一对的余弦相似度。embed_text 已做 L2 归一化，点积即余弦。"""
    sims: list[float] = []
    for i, (a, b) in enumerate(pairs, 1):
        va, vb = await embed_text(a), await embed_text(b)
        sims.append(float(np.dot(va, vb)))
        print(f"  {i}/{len(pairs)}  sim={sims[-1]:.4f}", end="\r")
    print(" " * 40, end="\r")
    return np.array(sims, dtype=np.float64)


async def run(pairs_path: Path) -> None:
    related_pairs, unrelated_pairs = parse_pairs(pairs_path)
    if not related_pairs or not unrelated_pairs:
        print(f"❌ 未能从 {pairs_path} 解析出样本对。")
        sys.exit(1)

    print(f"相关对 {len(related_pairs)} 组 / 无关对 {len(unrelated_pairs)} 组")
    print(f"模型 {config.EMBEDDING_MODEL} · 维度 {config.EMBEDDING_DIM}\n")

    print("计算相关对…")
    rel = await similarities(related_pairs)
    print("计算无关对…")
    unrel = await similarities(unrelated_pairs)

    p95_unrelated = float(np.percentile(unrel, 95))
    p25_related = float(np.percentile(rel, 25))
    high = p25_related
    mid = (p95_unrelated + p25_related) / 2.0

    print("\n── 分布 ──────────────────────────────")
    print(f"  相关对    中位 {np.median(rel):.4f}  P25 {p25_related:.4f}")
    print(f"  无关对    中位 {np.median(unrel):.4f}  P95 {p95_unrelated:.4f}")

    if p95_unrelated >= p25_related:
        print(
            "\n⚠ 两个分布重叠（P95_unrelated >= P25_related），阈值分层不成立。\n"
            "  说明当前 embedding 空间区分度不足——先检查维度是不是截过头了，"
            "或者扩充样本再重跑。"
        )

    print("\n── 建议写入 .env ─────────────────────")
    print(f"SIMILARITY_MID={mid:.2f}")
    print(f"SIMILARITY_HIGH={high:.2f}")
    print(
        f"\n当前生效值: MID={config.SIMILARITY_MID} / HIGH={config.SIMILARITY_HIGH}"
    )
    print("同时请把这次的分布结果同步回样本文件的头部说明。")


def main() -> None:
    parser = argparse.ArgumentParser(description="重新标定激活相似度阈值")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    args = parser.parse_args()

    errors = [e for e in config.check_config() if "embedding 维度不匹配" not in e]
    if errors:
        for e in errors:
            print(f"❌ {e}")
        sys.exit(1)

    asyncio.run(run(args.pairs))


if __name__ == "__main__":
    main()
