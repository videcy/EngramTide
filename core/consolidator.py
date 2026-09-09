"""
Phase 4 — 记忆合并/去重。

职责：
- find_merge_candidates()：纯函数，筛选可合并的近重复对。
- consolidate_memories()：执行体（LLM 融合 + 落库），手动 /consolidate 触发。
- 逻辑约束：只合并同类 episodic/emotional，保守阈值 0.92，dry-run 预览。
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from config import (
    CONSOLIDATE_MAX_PAIRS,
    CONSOLIDATE_SIMILARITY,
    CONSOLIDATED_MAX_LENGTH,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CHAT_MODEL,
    PROMPTS_DIR,
    REQUEST_TIMEOUT_SECONDS,
)
from core.embedding import embed_text
from core.memory_store import (
    Memory,
    insert_memory,
    list_active_memories,
    mark_superseded,
    set_meta,
)
from utils.similarity import cosine_similarity
from utils.time_utils import utc_now

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationReport:
    """一次合并执行的摘要。"""

    candidates: int = 0   # 候选对数
    merged: int = 0       # 成功融合条数
    skipped: int = 0      # LLM 失败/解析失败跳过的对数
    capped: int = 0       # 因 CONSOLIDATE_MAX_PAIRS 截断的对数


# ── 候选筛选（纯函数）─────────────────────────────────────


def find_merge_candidates(
    memories: list[Memory],
    threshold: float = CONSOLIDATE_SIMILARITY,
    max_pairs: int = CONSOLIDATE_MAX_PAIRS,
) -> list[tuple[Memory, Memory, float]]:
    """
    纯函数：找出可合并的近重复对，按相似度降序。

    资格规则（全部满足才入选）：
    - 两条同为 episodic 或同为 emotional；
    - 双方 superseded_by 均为 None、embedding 均非 None；
    - 双方 unresolved 均为 False；
    - cosine > threshold（严格大于）。

    贪心配对：一条记忆只出现在一个对中（按相似度降序占用）。
    """
    result, _capped = find_merge_candidates_capped(memories, threshold, max_pairs)
    return result


def find_merge_candidates_capped(
    memories: list[Memory],
    threshold: float = CONSOLIDATE_SIMILARITY,
    max_pairs: int = CONSOLIDATE_MAX_PAIRS,
) -> tuple[list[tuple[Memory, Memory, float]], int]:
    """
    find_merge_candidates 的完整版：额外返回因 max_pairs 被截断的对数。

    返回 (候选对列表, capped)。capped 只统计"若无上限本会入选"的对
    （已被贪心占用的对不计）。
    """
    # 收集合法候选
    candidates: list[Memory] = []
    for mem in memories:
        if mem.type not in ("episodic", "emotional"):
            continue
        if mem.superseded_by is not None:
            continue
        if mem.embedding is None:
            continue
        if mem.unresolved:
            continue
        candidates.append(mem)

    # 计算所有同类型对（分块矩阵乘，见 _pairs_above_threshold）
    pairs = _pairs_above_threshold(candidates, threshold)

    # 按相似度降序。_pairs_above_threshold 的产出顺序与原先的 (i, j) 双层循环一致，
    # 配合 Python 的稳定排序，同分对的相对顺序与改造前相同。
    pairs.sort(key=lambda x: x[2], reverse=True)

    # 贪心配对（截断对同样占用 id，精确模拟无上限时的贪心，避免 capped 重复计数）
    used_ids: set[str] = set()
    result: list[tuple[Memory, Memory, float]] = []
    capped = 0
    for a, b, sim in pairs:
        if a.memory_id in used_ids or b.memory_id in used_ids:
            continue
        used_ids.add(a.memory_id)
        used_ids.add(b.memory_id)
        if len(result) >= max_pairs:
            capped += 1
            continue
        result.append((a, b, sim))

    return (result, capped)


def _pairs_above_threshold(
    candidates: list[Memory],
    threshold: float,
) -> list[tuple[Memory, Memory, float]]:
    """
    在同类型记忆内找出 cosine > threshold 的所有对。

    两两比较本身是 O(N²)，改不掉——但可以把 N² 次 Python 层 cosine_similarity
    换成若干次 BLAS 矩阵乘。这是自动化合并的性能前提：N=10000 时朴素写法要跑
    5×10⁷ 次 Python 函数调用。

    分块的理由：M @ M.T 在 N=10000 时是 10⁸ 个 float32 = 400 MB。这里每次只算
    CONSOLIDATE_BLOCK_SIZE 行 × 全量，峰值内存被压到 block × N。

    只在同类型内比较（episodic 与 emotional 各自成组），这本来就是既有逻辑，
    顺带把矩阵规模又降了一档。
    """
    from config import CONSOLIDATE_BLOCK_SIZE

    by_type: dict[str, list[Memory]] = {}
    for mem in candidates:
        by_type.setdefault(mem.type, []).append(mem)

    pairs: list[tuple[Memory, Memory, float]] = []
    block = max(1, CONSOLIDATE_BLOCK_SIZE)

    for group in by_type.values():
        n = len(group)
        if n < 2:
            continue

        dims = {m.embedding.shape[0] for m in group}
        if len(dims) > 1:
            # 混维无法成矩阵；退回逐条比较并跳过维度不符的对（正经修法是重建全库）
            logger.warning(
                "合并候选中存在 %d 种 embedding 维度 %s，本次退回逐条比较。",
                len(dims), sorted(dims),
            )
            pairs.extend(_pairs_pairwise(group, threshold))
            continue

        matrix = np.vstack([m.embedding for m in group]).astype(np.float32, copy=False)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        matrix = matrix / norms

        for start in range(0, n, block):
            stop = min(start + block, n)
            sims = np.clip(matrix[start:stop] @ matrix.T, -1.0, 1.0)
            # 只取严格上三角，避免自比与重复计对
            rows, cols = np.nonzero(sims > threshold)
            for r, c in zip(rows, cols):
                i = start + int(r)
                j = int(c)
                if j <= i:
                    continue
                pairs.append((group[i], group[j], float(sims[r, j])))

    return pairs


def _pairs_pairwise(
    group: list[Memory],
    threshold: float,
) -> list[tuple[Memory, Memory, float]]:
    """混维时的逐条兜底路径。维度不符的对按不相似处理。"""
    out: list[tuple[Memory, Memory, float]] = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            try:
                sim = cosine_similarity(group[i].embedding, group[j].embedding)
            except ValueError:
                continue
            if sim > threshold:
                out.append((group[i], group[j], sim))
    return out


# ── LLM 融合 ──────────────────────────────────────────────


async def _llm_merge_pair(
    content_a: str,
    content_b: str,
    mem_type: str,
) -> str | None:
    """
    调 LLM 将两条记忆内容融合为一条。

    返回融合后的文本，失败返回 None。
    """
    prompt_path = PROMPTS_DIR / "consolidate.txt"
    if not prompt_path.exists():
        logger.error("consolidate.txt 未找到，无法执行融合")
        return None

    system_prompt = prompt_path.read_text(encoding="utf-8")
    system_prompt = system_prompt.replace("{max_length}", str(CONSOLIDATED_MAX_LENGTH))

    user_prompt = (
        f"记忆类型：{mem_type}\n\n"
        f"记忆 A：{content_a}\n\n"
        f"记忆 B：{content_b}\n\n"
        f"请输出融合后的记忆（不超过 {CONSOLIDATED_MAX_LENGTH} 字）："
    )

    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": DEEPSEEK_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 512,
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                logger.warning("融合 LLM 返回 %d: %s", response.status_code, response.text[:200])
                return None
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            if not content:
                return None
            return content
    except Exception as e:
        logger.warning("融合 LLM 调用失败: %s", e)
        return None


# ── 主入口 ────────────────────────────────────────────────


async def consolidate_memories(dry_run: bool = False) -> ConsolidationReport:
    """
    /consolidate 的执行体。

    流程（每对）：
    1. find_merge_candidates → 候选对
    2. 对每对调 LLM 融合 content
    3. 新记忆：type 同源；embedding 重算；decay_weight = max(两条)；
       access_count = 两条之和；created_at = 较早一条；tags = 并集；
       valence/arousal = |值| 较大一条的原值
    4. 落库：先 insert 新记忆，再 mark_superseded 两条旧的
    5. dry_run=True 只返回候选清单，零写入零 LLM 调用

    全部完成后写入 meta.last_consolidation_at。
    """
    memories = list_active_memories()
    pairs, capped = find_merge_candidates_capped(memories)

    if not pairs:
        return ConsolidationReport(capped=capped)

    if dry_run:
        return ConsolidationReport(candidates=len(pairs), capped=capped)

    report = ConsolidationReport(candidates=len(pairs), capped=capped)
    new_memories: list[Memory] = []
    supersede_map: list[tuple[list[str], str]] = []  # [(old_ids, new_id)]

    for a, b, sim in pairs:
        # 1. LLM 融合
        merged_content = await _llm_merge_pair(a.content, b.content, a.type)
        if merged_content is None:
            report.skipped += 1
            continue

        # 2. 计算 embedding
        try:
            merged_embedding = await embed_text(merged_content)
        except Exception as e:
            logger.warning("融合记忆 embedding 失败: %s，跳过该对", e)
            report.skipped += 1
            continue

        # 3. 构造新记忆
        # decay_weight = max
        new_decay = max(a.decay_weight, b.decay_weight)
        # access_count = 两条之和
        new_access = a.access_count + b.access_count
        # created_at = 较早一条
        new_created = a.created_at if a.created_at <= b.created_at else b.created_at
        # tags = 并集
        new_tags = list(set(a.tags + b.tags))
        # valence/arousal = |值| 较大一条的原值
        if abs(a.valence) >= abs(b.valence):
            new_valence = a.valence
            new_arousal = a.arousal
        else:
            new_valence = b.valence
            new_arousal = b.arousal

        new_id = str(uuid.uuid4())
        new_mem = Memory(
            memory_id=new_id,
            type=a.type,
            content=merged_content,
            valence=new_valence,
            arousal=new_arousal,
            created_at=new_created,
            decay_weight=new_decay,
            embedding=merged_embedding,
            tags=new_tags,
            access_count=new_access,
        )

        new_memories.append(new_mem)
        supersede_map.append(([a.memory_id, b.memory_id], new_id))
        report.merged += 1

    # 4. 落库：先 insert 新记忆，再 mark_superseded
    for new_mem in new_memories:
        insert_memory(new_mem)

    for old_ids, new_id in supersede_map:
        mark_superseded(old_ids, new_id)

    # 5. 记录时间（aware UTC——全项目禁用裸 datetime.now()）
    set_meta("last_consolidation_at", utc_now().isoformat())

    return report


# ── P4-③：自动触发 ────────────────────────────────────────


async def maybe_auto_consolidate() -> ConsolidationReport | None:
    """
    会话结束、写入完成后调用。规模越线且候选够多时自动合并一次。

    返回 None 表示本次没触发（开关关闭 / 规模未到 / 候选不足）。

    ★ 默认关闭（CONSOLIDATE_AUTO_ENABLED=false）：自动合并会在用户退出时静默
      发起若干次 LLM 调用，这个代价必须由用户显式同意。开启前请先用
      `/consolidate preview` 看看候选对长什么样。

    ★ 性能前提已就位：find_merge_candidates 的两两比较改成了分块矩阵乘
      （见 _pairs_above_threshold），否则 N=10000 时这里会直接卡死。
    """
    from config import (
        CONSOLIDATE_AUTO_ENABLED,
        CONSOLIDATE_AUTO_THRESHOLD,
        CONSOLIDATE_MIN_PAIRS,
    )

    if not CONSOLIDATE_AUTO_ENABLED:
        return None

    memories = list_active_memories()
    epi_emo = [m for m in memories if m.type in ("episodic", "emotional")]
    if len(epi_emo) <= CONSOLIDATE_AUTO_THRESHOLD:
        return None

    pairs = find_merge_candidates(memories)
    if len(pairs) < CONSOLIDATE_MIN_PAIRS:
        return None

    logger.info(
        "活跃 episodic/emotional %d 条、候选 %d 对，自动执行合并。",
        len(epi_emo), len(pairs),
    )
    return await consolidate_memories(dry_run=False)
