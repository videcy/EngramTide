"""
冲突处理 — 候选召回与关系判定单元测试。

LLM 一律 mock，向量用构造的二维向量（cos = 夹角余弦）。
"""

import json
import math

import numpy as np
import pytest

import config
from core import conflict
from core.conflict import (
    Candidate,
    can_compare,
    find_conflict_candidates,
    judge_by_rules,
    judge_relations,
    type_threshold,
)
from core.memory_store import Memory


def vec(sim: float) -> np.ndarray:
    """与 [1, 0] 余弦相似度为 sim 的单位向量。"""
    return np.array([sim, math.sqrt(1 - sim * sim)], dtype=np.float32)


BASE = np.array([1.0, 0.0], dtype=np.float32)


def mem(mid, mtype="semantic", sim=None, content=None, valence=0.0, **kw) -> Memory:
    emb = BASE if sim is None else vec(sim)
    return Memory(
        memory_id=mid,
        type=mtype,
        content=content or f"内容-{mid}",
        valence=valence,
        embedding=emb,
        created_at=kw.pop("created_at", "2026-03-02T10:00:00"),
        **kw,
    )


def cand(new, old, sim) -> Candidate:
    return Candidate(new=new, old=old, similarity=sim)


# ── 召回 ──────────────────────────────────────────────────


class TestFindCandidates:
    def test_min_sim_lower_bound_inclusive(self):
        new = mem("n")
        pool = [mem("a", sim=0.60), mem("b", sim=0.50), mem("c", sim=0.9)]
        res = find_conflict_candidates(new, pool, top_k=10, min_sim=0.60 - 1e-6)
        assert [c.old.memory_id for c in res] == ["c", "a"]

    def test_default_min_from_config(self, monkeypatch):
        monkeypatch.setattr(config, "CONFLICT_CANDIDATE_MIN", 0.7)
        new = mem("n")
        res = find_conflict_candidates(new, [mem("a", sim=0.65), mem("b", sim=0.8)], top_k=5)
        assert [c.old.memory_id for c in res] == ["b"]

    def test_top_k_boundary_and_order(self):
        new = mem("n")
        pool = [mem(f"m{i}", sim=0.6 + i * 0.05) for i in range(5)]
        res = find_conflict_candidates(new, pool, top_k=3, min_sim=0.5)
        assert [c.old.memory_id for c in res] == ["m4", "m3", "m2"]
        sims = [c.similarity for c in res]
        assert sims == sorted(sims, reverse=True)
        assert len(find_conflict_candidates(new, pool, top_k=10, min_sim=0.5)) == 5

    def test_default_top_k_from_config(self, monkeypatch):
        monkeypatch.setattr(config, "CONFLICT_TOP_K", 2)
        new = mem("n")
        pool = [mem(f"m{i}", sim=0.9) for i in range(4)]
        assert len(find_conflict_candidates(new, pool, min_sim=0.5)) == 2

    def test_top_k_zero(self):
        assert find_conflict_candidates(mem("n"), [mem("a")], top_k=0, min_sim=0.0) == []

    def test_cross_type_rules(self):
        pool = [
            mem("sem", "semantic"),
            mem("epi", "episodic"),
            mem("emo", "emotional"),
            mem("pro", "procedural"),
        ]
        ids = lambda t: {  # noqa: E731
            c.old.memory_id
            for c in find_conflict_candidates(mem("n", t), pool, top_k=10, min_sim=0.5)
        }
        assert ids("semantic") == {"sem", "epi"}
        assert ids("episodic") == {"sem", "epi"}
        assert ids("emotional") == {"emo"}
        assert ids("procedural") == {"pro"}

    def test_skip_superseded_no_embedding_same_id(self):
        new = mem("n")
        no_emb = mem("noemb")
        no_emb.embedding = None
        pool = [
            mem("sup", superseded_by="x"),
            no_emb,
            mem("n"),  # 同 id
            mem("ok"),
        ]
        res = find_conflict_candidates(new, pool, top_k=10, min_sim=0.5)
        assert [c.old.memory_id for c in res] == ["ok"]

    def test_new_without_embedding(self):
        new = mem("n")
        new.embedding = None
        assert find_conflict_candidates(new, [mem("a")], top_k=3, min_sim=0.0) == []

    def test_dimension_mismatch_is_skipped(self):
        old = mem("a")
        old.embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert find_conflict_candidates(mem("n"), [old], top_k=3, min_sim=0.1) == []


class TestCanCompare:
    @pytest.mark.parametrize("a,b,ok", [
        ("semantic", "semantic", True),
        ("semantic", "episodic", True),
        ("episodic", "semantic", True),
        ("emotional", "emotional", True),
        ("emotional", "semantic", False),
        ("episodic", "emotional", False),
        ("procedural", "semantic", False),
        ("procedural", "procedural", True),
    ])
    def test_matrix(self, a, b, ok):
        assert can_compare(a, b) is ok


def test_type_threshold():
    assert type_threshold("semantic") == config.SEMANTIC_OVERRIDE_THRESHOLD
    assert type_threshold("emotional") == config.EMOTIONAL_REINFORCE_THRESHOLD
    assert type_threshold("procedural") == config.PROCEDURAL_DEDUP_THRESHOLD
    assert type_threshold("episodic") == config.EPISODIC_DEDUP_THRESHOLD


# ── 规则判定 ──────────────────────────────────────────────


class TestJudgeByRules:
    def test_cross_type_unrelated_even_if_similar(self):
        r = judge_by_rules(cand(mem("n", "semantic"), mem("o", "episodic"), 0.99))
        assert r.relation == "unrelated"
        assert r.judged_by == "rule"
        assert (r.new_id, r.old_id) == ("n", "o")

    def test_below_threshold_unrelated(self):
        r = judge_by_rules(cand(mem("n"), mem("o"), type_threshold("semantic") - 0.01))
        assert r.relation == "unrelated"

    def test_semantic_update(self):
        # semantic 过阈值按「新事实为准」，不判 duplicate 丢新
        r = judge_by_rules(cand(mem("n"), mem("o"), 0.95))
        assert r.relation == "update"
        assert r.similarity == 0.95

    def test_episodic_duplicate(self):
        r = judge_by_rules(cand(mem("n", "episodic"), mem("o", "episodic"), 0.95))
        assert r.relation == "duplicate"

    def test_procedural_update(self):
        r = judge_by_rules(cand(mem("n", "procedural"), mem("o", "procedural"), 0.95))
        assert (r.relation, r.subtype) == ("update", None)

    def test_procedural_below_threshold(self):
        r = judge_by_rules(cand(mem("n", "procedural"), mem("o", "procedural"), 0.5))
        assert r.relation == "unrelated"

    def test_emotional_valence_flip(self):
        r = judge_by_rules(
            cand(mem("n", "emotional", valence=0.6), mem("o", "emotional", valence=-0.4), 0.95)
        )
        assert (r.relation, r.subtype) == ("update", "flip")

    def test_emotional_same_sign_duplicate(self):
        r = judge_by_rules(
            cand(mem("n", "emotional", valence=-0.6), mem("o", "emotional", valence=-0.4), 0.95)
        )
        assert r.relation == "duplicate"

    @pytest.mark.parametrize("nv,ov", [(0.0, -0.5), (0.5, 0.0), (0.0, 0.0)])
    def test_emotional_zero_valence_not_flip(self, nv, ov):
        r = judge_by_rules(
            cand(mem("n", "emotional", valence=nv), mem("o", "emotional", valence=ov), 0.95)
        )
        assert r.relation == "duplicate"

    def test_emotional_flip_below_threshold(self):
        r = judge_by_rules(
            cand(mem("n", "emotional", valence=0.6), mem("o", "emotional", valence=-0.4), 0.5)
        )
        assert r.relation == "unrelated"


# ── LLM 判定 ──────────────────────────────────────────────


@pytest.fixture
def llm(monkeypatch):
    """mock _call_judge_llm；返回 state，设置 state['reply'] 控制输出，state['calls'] 记录调用。"""
    state = {"reply": None, "calls": []}

    async def fake(system_prompt, user_prompt):
        state["calls"].append((system_prompt, user_prompt))
        return state["reply"]

    monkeypatch.setattr(conflict, "_call_judge_llm", fake)
    monkeypatch.setattr(config, "CONFLICT_LLM_JUDGE_ENABLED", True)
    return state


def sem_pair(i, sim=0.95) -> Candidate:
    return cand(mem(f"n{i}", content=f"新{i}"), mem(f"o{i}", content=f"旧{i}"), sim)


def reply(*items) -> str:
    return json.dumps(list(items), ensure_ascii=False)


class TestJudgeRelations:
    async def test_empty_no_llm(self, llm):
        assert await judge_relations([]) == []
        assert llm["calls"] == []

    async def test_disabled_uses_rules(self, llm, monkeypatch):
        monkeypatch.setattr(config, "CONFLICT_LLM_JUDGE_ENABLED", False)
        res = await judge_relations([sem_pair(1)])
        assert llm["calls"] == []
        assert res[0].judged_by == "rule"
        assert res[0].relation == "update"

    async def test_normal_parse(self, llm):
        llm["reply"] = reply(
            {"pair": 1, "relation": "update", "subtype": None, "merged": None, "reason": "居住地变了"},
            {"pair": 2, "relation": "refine", "subtype": None, "merged": "合并文本", "reason": "补充"},
            {"pair": 3, "relation": "unrelated", "subtype": None, "merged": None, "reason": "无关"},
        )
        cands = [sem_pair(1), sem_pair(2), sem_pair(3)]
        res = await judge_relations(cands)
        assert [r.relation for r in res] == ["update", "refine", "unrelated"]
        assert all(r.judged_by == "llm" for r in res)
        assert [(r.new_id, r.old_id) for r in res] == [("n1", "o1"), ("n2", "o2"), ("n3", "o3")]
        assert res[0].reason == "居住地变了"
        assert res[1].merged == "合并文本"
        assert res[0].merged is None
        assert res[0].similarity == 0.95

    async def test_single_llm_call_and_prompt_content(self, llm):
        llm["reply"] = "[]"
        await judge_relations([sem_pair(1), sem_pair(2)])
        assert len(llm["calls"]) == 1
        system, user = llm["calls"][0]
        assert "{max_length}" not in system
        payload = json.loads(user[user.index("["):])
        assert [p["pair"] for p in payload] == [1, 2]
        assert payload[0]["old"]["text"] == "旧1"
        assert payload[0]["new"]["text"] == "新1"
        assert payload[0]["old"]["date"] == "2026-03-02"
        assert payload[0]["type"] == "semantic"
        assert payload[0]["valence_flipped"] is False

    async def test_valence_flipped_signal_in_prompt(self, llm):
        llm["reply"] = "[]"
        c = cand(mem("n", "emotional", valence=0.5), mem("o", "emotional", valence=-0.5), 0.9)
        await judge_relations([c])
        payload = json.loads(llm["calls"][0][1][llm["calls"][0][1].index("["):])
        assert payload[0]["valence_flipped"] is True

    async def test_code_fence_tolerated(self, llm):
        llm["reply"] = "```json\n" + reply({"pair": 1, "relation": "contradict", "reason": "x"}) + "\n```"
        res = await judge_relations([sem_pair(1)])
        assert res[0].relation == "contradict"
        assert res[0].judged_by == "llm"

    async def test_text_around_array_tolerated(self, llm):
        llm["reply"] = "结果如下：" + reply({"pair": 1, "relation": "update"}) + " 完毕"
        res = await judge_relations([sem_pair(1)])
        assert res[0].relation == "update"

    @pytest.mark.parametrize("bad", ["not json at all", '{"pair": 1}', "", "[1, 2", None])
    async def test_non_json_falls_back_all(self, llm, bad):
        llm["reply"] = bad
        res = await judge_relations([sem_pair(1), sem_pair(2)])
        assert all(r.judged_by == "rule" for r in res)
        assert [r.relation for r in res] == ["update", "update"]

    async def test_llm_failure_returns_none_falls_back(self, llm):
        # _call_judge_llm 自己吞异常并返回 None
        llm["reply"] = None
        res = await judge_relations([sem_pair(1)])
        assert res[0].judged_by == "rule"

    async def test_per_pair_fallback(self, llm):
        llm["reply"] = reply(
            {"pair": 1, "relation": "update"},
            # pair 2 缺项
            {"pair": 3, "relation": "bogus"},          # 非法 relation
            {"pair": 9, "relation": "update"},          # 越界
            {"pair": 0, "relation": "update"},          # 越界
            {"pair": "4", "relation": "update"},        # 编号类型错
            {"relation": "update"},                     # 无编号
            "garbage",
        )
        cands = [sem_pair(i) for i in range(1, 6)]
        res = await judge_relations(cands)
        assert res[0].judged_by == "llm" and res[0].relation == "update"
        for r in res[1:]:
            assert r.judged_by == "rule"
            assert r.relation == "update"

    async def test_duplicate_pair_first_wins(self, llm):
        llm["reply"] = reply(
            {"pair": 1, "relation": "update"},
            {"pair": 1, "relation": "contradict"},
        )
        res = await judge_relations([sem_pair(1)])
        assert res[0].relation == "update"

    async def test_over_max_pairs_use_rules(self, llm, monkeypatch):
        monkeypatch.setattr(config, "CONFLICT_MAX_PAIRS", 2)
        llm["reply"] = reply(*[{"pair": i, "relation": "refine"} for i in range(1, 6)])
        res = await judge_relations([sem_pair(i) for i in range(1, 5)])
        assert [r.judged_by for r in res] == ["llm", "llm", "rule", "rule"]
        assert [r.relation for r in res] == ["refine", "refine", "update", "update"]
        assert len(llm["calls"]) == 1
        payload = json.loads(llm["calls"][0][1][llm["calls"][0][1].index("["):])
        assert len(payload) == 2

    async def test_max_pairs_zero_no_llm(self, llm, monkeypatch):
        monkeypatch.setattr(config, "CONFLICT_MAX_PAIRS", 0)
        res = await judge_relations([sem_pair(1)])
        assert llm["calls"] == []
        assert res[0].judged_by == "rule"

    async def test_refine_without_merged(self, llm):
        llm["reply"] = reply({"pair": 1, "relation": "refine", "merged": None})
        res = await judge_relations([sem_pair(1)])
        assert res[0].relation == "refine" and res[0].merged is None

    async def test_merged_ignored_for_non_refine(self, llm):
        llm["reply"] = reply({"pair": 1, "relation": "update", "merged": "不该有"})
        res = await judge_relations([sem_pair(1)])
        assert res[0].merged is None

    async def test_relation_case_insensitive(self, llm):
        llm["reply"] = reply({"pair": 1, "relation": " Update "})
        res = await judge_relations([sem_pair(1)])
        assert res[0].relation == "update"

    async def test_emotional_update_subtype_forced_flip(self, llm):
        llm["reply"] = reply(
            {"pair": 1, "relation": "update", "subtype": "correction"},
            {"pair": 2, "relation": "update", "subtype": None},
        )
        cands = [
            cand(mem("n1", "emotional"), mem("o1", "emotional"), 0.9),
            cand(mem("n2", "emotional"), mem("o2", "emotional"), 0.9),
        ]
        res = await judge_relations(cands)
        assert [r.subtype for r in res] == ["flip", "flip"]

    async def test_episodic_update_subtype(self, llm):
        llm["reply"] = reply(
            {"pair": 1, "relation": "update", "subtype": "followup"},
            {"pair": 2, "relation": "update", "subtype": "correction"},
            {"pair": 3, "relation": "update", "subtype": "flip"},
            {"pair": 4, "relation": "update", "subtype": None},
        )
        cands = [cand(mem(f"n{i}", "episodic"), mem(f"o{i}", "episodic"), 0.9) for i in range(4)]
        res = await judge_relations(cands)
        assert [r.subtype for r in res] == ["followup", "correction", "correction", "correction"]

    async def test_subtype_dropped_for_other_cases(self, llm):
        llm["reply"] = reply(
            {"pair": 1, "relation": "update", "subtype": "followup"},   # semantic update
            {"pair": 2, "relation": "contradict", "subtype": "flip"},   # 非 update
        )
        cands = [
            cand(mem("n1", "semantic"), mem("o1", "semantic"), 0.9),
            cand(mem("n2", "episodic"), mem("o2", "episodic"), 0.9),
        ]
        res = await judge_relations(cands)
        assert [r.subtype for r in res] == [None, None]

    async def test_cross_type_pair_sent_to_llm(self, llm):
        llm["reply"] = reply({"pair": 1, "relation": "update", "subtype": "followup"})
        c = cand(mem("n", "episodic"), mem("o", "semantic"), 0.7)
        res = await judge_relations([c])
        assert res[0].relation == "update" and res[0].judged_by == "llm"
        assert res[0].subtype == "followup"

    async def test_missing_prompt_falls_back(self, llm, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "prompt_path", lambda name: tmp_path / "nope.txt")
        res = await judge_relations([sem_pair(1)])
        assert llm["calls"] == []
        assert res[0].judged_by == "rule"


def test_prompt_file_exists_and_mentions_all_relations():
    text = config.prompt_path("conflict_judge.txt").read_text(encoding="utf-8")
    for r in conflict.RELATION_TYPES:
        assert r in text
    assert "correction" in text and "followup" in text
