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

    # 计算所有同类型对
    pairs: list[tuple[Memory, Memory, float]] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            if a.type != b.type:
                continue
            sim = cosine_similarity(a.embedding, b.embedding)
            if sim > threshold:
                pairs.append((a, b, sim))

    # 按相似度降序
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
