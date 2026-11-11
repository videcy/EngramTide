"""
新旧记忆冲突处理 — 候选召回与关系判定。

职责：
- find_conflict_candidates()：纯函数，零 LLM，召回与新记忆可能相关的旧记忆。
- judge_by_rules()：规则判定（LLM 不可用时的兜底）。
- judge_relations()：整批打包成一次 LLM 调用判定关系，逐对校验、逐对回退规则。
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

import config
from core.memory_store import Memory
from utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)

RELATION_TYPES = ("duplicate", "refine", "update", "contradict", "unrelated")

_EPISODIC_UPDATE_SUBTYPES = ("correction", "followup")


@dataclass
class Candidate:
    """一对待判定的 (新记忆, 旧记忆)。"""

    new: Memory
    old: Memory
    similarity: float


@dataclass
class Relation:
    """一对记忆的判定结果。subtype ∈ correction/followup/flip/None。"""

    new_id: str
    old_id: str
    relation: str
    similarity: float
    judged_by: str  # llm / rule
    subtype: str | None = None
    reason: str | None = None
    merged: str | None = None


# ── 候选召回（纯函数）─────────────────────────────────────


def can_compare(new_type: str, old_type: str) -> bool:
    """同类型可比；semantic 与 episodic 互查；emotional、procedural 只同类。"""
    if new_type == old_type:
        return True
    return {new_type, old_type} == {"semantic", "episodic"}


def type_threshold(mem_type: str) -> float:
    """各类型的写入相似度阈值。未知类型返回 1.0（永远不过阈值）。"""
    return {
        "semantic": config.SEMANTIC_OVERRIDE_THRESHOLD,
        "emotional": config.EMOTIONAL_REINFORCE_THRESHOLD,
        "procedural": config.PROCEDURAL_DEDUP_THRESHOLD,
        "episodic": config.EPISODIC_DEDUP_THRESHOLD,
    }.get(mem_type, 1.0)


def _similarity(a, b) -> float:
    """余弦相似度；任一为 None 或计算失败返回 0.0。"""
    if a is None or b is None:
        return 0.0
    try:
        return float(cosine_similarity(a, b))
    except Exception:
        return 0.0


def find_conflict_candidates(
    new_mem: Memory,
    pool: list[Memory],
    *,
    top_k: int | None = None,
    min_sim: float | None = None,
) -> list[Candidate]:
    """
    从 pool 中召回与 new_mem 可能冲突的旧记忆，按相似度降序取 top_k。

    跳过：superseded、同 id、无 embedding、不可比类型；sim < min_sim 的不入选。
    """
    if new_mem.embedding is None:
        return []
    if top_k is None:
        top_k = config.CONFLICT_TOP_K
    if min_sim is None:
        min_sim = config.CONFLICT_CANDIDATE_MIN
    if top_k <= 0:
        return []

    result: list[Candidate] = []
    for old in pool:
        if old.memory_id == new_mem.memory_id:
            continue
        if old.superseded_by is not None or old.embedding is None:
            continue
        if not can_compare(new_mem.type, old.type):
            continue
        sim = _similarity(new_mem.embedding, old.embedding)
        if sim >= min_sim:
            result.append(Candidate(new=new_mem, old=old, similarity=sim))

    result.sort(key=lambda c: c.similarity, reverse=True)
    return result[:top_k]


# ── 规则判定 ──────────────────────────────────────────────


def _valence_flipped(new: Memory, old: Memory) -> bool:
    """valence 反号：两者都非 0 且符号相反（0 不算反号）。"""
    return new.valence * old.valence < 0


def judge_by_rules(c: Candidate) -> Relation:
    """规则兜底判定（judged_by='rule'）。"""

    def make(relation: str, subtype: str | None = None, reason: str | None = None) -> Relation:
        return Relation(
            new_id=c.new.memory_id,
            old_id=c.old.memory_id,
            relation=relation,
            similarity=c.similarity,
            judged_by="rule",
            subtype=subtype,
            reason=reason,
        )

    if c.new.type != c.old.type:
        return make("unrelated", reason="跨类型，规则不判定")
    if c.similarity < type_threshold(c.new.type):
        return make("unrelated", reason="相似度未过阈值")
    if c.new.type == "procedural":
        return make("update", reason="新指令为准")
    if c.new.type == "semantic":
        # 沿用引入冲突处理前的行为：新事实覆盖旧事实（仍受守卫约束），不判 duplicate 丢新
        return make("update", reason="相似度过阈值，新事实为准")
    if c.new.type == "emotional" and _valence_flipped(c.new, c.old):
        return make("update", subtype="flip", reason="情绪倾向反转")
    return make("duplicate", reason="相似度过阈值")


# ── LLM 判定 ──────────────────────────────────────────────


def _build_user_prompt(cands: list[Candidate]) -> str:
    """把候选对编号（从 1 开始）打包成输入 JSON。"""
    items = []
    for i, c in enumerate(cands, start=1):
        items.append({
            "pair": i,
            "type": c.new.type,
            "old": {"type": c.old.type, "text": c.old.content, "date": (c.old.created_at or "")[:10]},
            "new": {"type": c.new.type, "text": c.new.content, "date": (c.new.created_at or "")[:10]},
            "valence_flipped": _valence_flipped(c.new, c.old),
        })
    return (
        "请判定下列记忆对的关系，输出严格 JSON 数组：\n"
        + json.dumps(items, ensure_ascii=False, indent=2)
    )


async def _call_judge_llm(system_prompt: str, user_prompt: str) -> str | None:
    """调 LLM，返回原始文本；失败返回 None。"""
    url = f"{config.DEEPSEEK_BASE_URL.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": config.DEEPSEEK_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": config.CONFLICT_JUDGE_MAX_TOKENS,
    }
    try:
        async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                logger.warning("冲突判定 LLM 返回 %d: %s", response.status_code, response.text[:200])
                return None
            choice = response.json()["choices"][0]
            content = (choice["message"].get("content") or "").strip()
            if not content:
                # 推理模型的思考 token 会吃掉 max_tokens，输出被截成空串
                logger.warning(
                    "冲突判定 LLM 输出为空（finish_reason=%s），回退规则",
                    choice.get("finish_reason"),
                )
                return None
            if choice.get("finish_reason") == "length":
                logger.warning("冲突判定 LLM 输出被截断（max_tokens 不够），可能部分回退规则")
            return content
    except Exception as e:
        logger.warning("冲突判定 LLM 调用失败: %s", e)
        return None


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _parse_json_array(text: str) -> list | None:
    """解析 LLM 输出为 JSON 数组；容忍 ```json 代码块与前后多余文字。失败返回 None。"""
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            return None
    return data if isinstance(data, list) else None


def _normalize_subtype(mem_type: str, relation: str, subtype: Any) -> str | None:
    """规整 subtype：只有 update 才有；emotional 恒为 flip；episodic 限 correction/followup。"""
    if relation != "update":
        return None
    if mem_type == "emotional":
        return "flip"
    if mem_type == "episodic":
        return subtype if subtype in _EPISODIC_UPDATE_SUBTYPES else "correction"
    return None


def _relation_from_item(c: Candidate, item: Any) -> Relation | None:
    """把单个 LLM 输出项转成 Relation；非法返回 None（由调用方回退规则）。"""
    if not isinstance(item, dict):
        return None
    relation = item.get("relation")
    if not isinstance(relation, str):
        return None
    relation = relation.strip().lower()
    if relation not in RELATION_TYPES:
        return None

    reason = item.get("reason")
    reason = reason.strip() if isinstance(reason, str) and reason.strip() else None
    merged = item.get("merged")
    merged = merged.strip() if relation == "refine" and isinstance(merged, str) else None

    return Relation(
        new_id=c.new.memory_id,
        old_id=c.old.memory_id,
        relation=relation,
        similarity=c.similarity,
        judged_by="llm",
        subtype=_normalize_subtype(c.new.type, relation, item.get("subtype")),
        reason=reason,
        merged=merged or None,
    )


async def judge_relations(cands: list[Candidate]) -> list[Relation]:
    """
    批量判定关系，返回与 cands 一一对应、同顺序的结果。

    - 空列表不调 LLM；CONFLICT_LLM_JUDGE_ENABLED 关闭时全走规则。
    - 前 CONFLICT_MAX_PAIRS 对打包成一次 LLM 调用，超出部分走规则。
    - LLM 异常/非 JSON → 全部回退规则；单对缺项/编号越界/relation 非法 → 该对回退规则。
    """
    if not cands:
        return []
    results = [judge_by_rules(c) for c in cands]
    if not config.CONFLICT_LLM_JUDGE_ENABLED:
        return results

    batch = cands[:max(0, config.CONFLICT_MAX_PAIRS)]
    if not batch:
        return results

    path = config.prompt_path("conflict_judge.txt")
    if not path.exists():
        logger.error("conflict_judge.txt 未找到，冲突判定回退规则")
        return results
    system_prompt = path.read_text(encoding="utf-8").replace(
        "{max_length}", str(config.CONSOLIDATED_MAX_LENGTH)
    )

    raw = await _call_judge_llm(system_prompt, _build_user_prompt(batch))
    if raw is None:
        return results
    items = _parse_json_array(raw)
    if items is None:
        logger.warning("冲突判定输出不是 JSON 数组，全部回退规则")
        return results

    seen: set[int] = set()
    for item in items:
        pair = item.get("pair") if isinstance(item, dict) else None
        # bool 是 int 子类，排除
        if not isinstance(pair, int) or isinstance(pair, bool):
            continue
        if not 1 <= pair <= len(batch) or pair in seen:
            continue
        rel = _relation_from_item(batch[pair - 1], item)
        if rel is None:
            continue
        seen.add(pair)
        results[pair - 1] = rel
    return results
