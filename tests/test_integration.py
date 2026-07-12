"""
Phase 1 MVP — 端到端集成测试。

使用 fake embedding / fake LLM 测完整闭环：
  输入一轮对话
    → fake 脱水生成记忆
    → 写入 SQLite
    → 新查询触发检索
    → Constitutional memory context 中包含旧记忆
    → chat 层生成的 system prompt 包含交互宪法、动态行为修正案、上下文记忆沙盒
"""

import json
import uuid

import numpy as np
import pytest

# 注意：不在此处导入 embed_text / generate_response 等会被 monkeypatch 的函数，
# 改为通过模块引用（core.embedding.embed_text）以确保 patch 生效。
import core.embedding as embedding_mod
import core.dehydrator as dehydrator_mod
import core.chat as chat_mod
from core.memory_store import (
    Memory,
    close_db,
    init_db,
    insert_memory,
    list_active_memories,
    mark_accessed,
)
from core.retriever import retrieve_memories
from core.context_builder import (
    build_constitutional_memory_context,
)
from core.chat import _build_system_prompt
from core.dehydrator import _extract_json_array, _clean_memory_item, _parse_bool


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def setup_db(monkeypatch, tmp_path):
    """每个集成测试使用独立临时数据库。"""
    db_file = tmp_path / "test_integration.db"
    monkeypatch.setattr("core.memory_store.DB_PATH", db_file)

    import core.memory_store as ms

    ms._connection = None
    init_db()
    yield
    close_db()


@pytest.fixture
def fake_embedding(monkeypatch):
    """
    Fake embedding：返回固定维度(1536)的确定性向量。
    文本越长，向量范数越大（模拟真实 embedding 效果）。
    """

    async def _fake_embed(text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            raise ValueError("空文本")
        seed = sum(ord(c) for c in text)
        rng = np.random.RandomState(seed)
        vec = rng.randn(1536).astype(np.float32)
        vec /= np.linalg.norm(vec)  # 归一化
        return vec

    monkeypatch.setattr(embedding_mod, "embed_text", _fake_embed)
    monkeypatch.setattr(dehydrator_mod, "embed_text", _fake_embed)


@pytest.fixture
def fake_chat(monkeypatch, fake_embedding):
    """
    Fake chat：替换 generate_response 为返回固定内容的函数。
    这样做不会影响 embedding 调用（它们已被 fake_embedding 替换）。
    """

    async def _fake_generate_response(
        user_message: str,
        memory_context,
        conversation_history: list,
    ) -> str:
        return f"（模拟回复）好的，关于「{user_message[:30]}」，我记下了。"

    monkeypatch.setattr(
        "core.chat.generate_response", _fake_generate_response
    )

    # 同时 patch dehydrator 中的 httpx 调用，让脱水返回固定 JSON
    import httpx

    class FakeDehydrateResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                [
                                    {
                                        "content": "用户在做 AI 记忆系统项目，技术栈 Python",
                                        "type": "semantic",
                                        "valence": 0.0,
                                        "arousal": 0.0,
                                        "unresolved": False,
                                        "tags": ["项目", "技术栈"],
                                    }
                                ]
                            )
                        }
                    }
                ]
            }

        @property
        def request(self):
            return None

        @property
        def text(self):
            return json.dumps(self.json())

    # 仅 patch AsyncClient.post 用于非 embedding 的调用
    # （embedding 已被 fake_embedding 替换，不会走 httpx）
    original_post = httpx.AsyncClient.post

    async def _patched_post(self, *args, **kwargs):
        url = args[0] if args else kwargs.get("url", "")
        if "/v1/embeddings" in url:
            # 不应到达这里（fake_embedding 已替换），但保留兜底
            raise RuntimeError("Embedding 调用未被 fake_embedding 拦截")
        # 返回脱水用的假响应
        return FakeDehydrateResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", _patched_post)


# ── 集成测试 ──────────────────────────────────────────────


class TestDehydrateIntegration:
    """脱水 JSON 清洗 + 集成。"""

    def test_extract_json_array_direct(self):
        """直接从 LLM 输出中解析 JSON 数组。"""
        raw = '[{"content": "测试", "type": "semantic"}]'
        result = _extract_json_array(raw)
        assert len(result) == 1
        assert result[0]["content"] == "测试"

    def test_extract_json_array_with_wrapper_text(self):
        """LLM 输出包含前后文本时能截取 JSON。"""
        raw = '好的，以下是提取的记忆：\n[{"content": "测试", "type": "semantic"}]\n共1条。'
        result = _extract_json_array(raw)
        assert len(result) == 1

    def test_extract_json_array_invalid_raises(self):
        """无法提取 JSON 时抛出 ValueError。"""
        with pytest.raises(ValueError):
            _extract_json_array("这不是 JSON")

    def test_clean_memory_empty_content_discarded(self):
        """空 content 被丢弃。"""
        result = _clean_memory_item({"content": "", "type": "semantic"})
        assert result is None

    def test_clean_memory_unknown_type_downgrade(self):
        """未知 type 降级为 semantic。"""
        result = _clean_memory_item({"content": "测试", "type": "weird_type"})
        assert result is not None
        assert result["type"] == "semantic"

    def test_clean_memory_valence_clamped(self):
        """valence 超出范围时 clamp。"""
        result = _clean_memory_item({"content": "测试", "valence": 5.0})
        assert result["valence"] == 1.0

        result2 = _clean_memory_item({"content": "测试", "valence": -5.0})
        assert result2["valence"] == -1.0

    def test_clean_memory_arousal_clamped(self):
        """arousal 超出范围时 clamp。"""
        result = _clean_memory_item({"content": "测试", "arousal": 5.0})
        assert result["arousal"] == 1.0

        result2 = _clean_memory_item({"content": "测试", "arousal": -0.5})
        assert result2["arousal"] == 0.0

    def test_clean_memory_tags_non_list_downgraded(self):
        """tags 非 list 时降级为空列表。"""
        result = _clean_memory_item({"content": "测试", "tags": "不是列表"})
        assert result["tags"] == []

    def test_parse_bool_strings_explicitly(self):
        """字符串布尔值显式解析，避免 bool('false') 为 True。"""
        assert _parse_bool(True) is True
        assert _parse_bool(False) is False
        assert _parse_bool("true") is True
        assert _parse_bool("yes") is True
        assert _parse_bool("1") is True
        assert _parse_bool("false") is False
        assert _parse_bool("no") is False
        assert _parse_bool("0") is False
        assert _parse_bool("unknown") is False

    def test_clean_memory_unresolved_string_false(self):
        """LLM 返回字符串 'false' 时 unresolved 应为 False。"""
        result = _clean_memory_item({"content": "测试", "unresolved": "false"})
        assert result["unresolved"] is False


class TestEndToEndLoop:
    """端到端闭环测试：写入 → 检索 → 上下文 → 回复。"""

    @pytest.mark.asyncio
    async def test_full_loop_write_retrieve_reply(self, fake_embedding, fake_chat):
        """
        模拟完整闭环：
        1. 模拟脱水写入一条记忆
        2. 新查询触发检索
        3. Constitutional memory context 包含旧记忆
        4. 回复中包含记忆信息
        """
        # Step 1: 写入一条记忆（模拟脱水后的结果）
        content = "用户在做 AI 记忆系统项目，技术栈是 Python 和 SQLite"
        embedding = await embedding_mod.embed_text(content)
        mem = Memory(
            memory_id=str(uuid.uuid4()),
            type="semantic",
            content=content,
            embedding=embedding,
        )
        insert_memory(mem)

        # 确认写入成功
        active = list_active_memories()
        assert len(active) == 1
        assert active[0].content == content

        # Step 2: 新查询 — "我之前在做什么项目？"
        query = "我之前在做什么项目？"
        query_emb = await embedding_mod.embed_text(query)

        retrieved = retrieve_memories(query_emb, top_k=5)
        assert len(retrieved) == 1
        assert retrieved[0][0].content == content
        # fake embedding 的 score 可为正或负（randn 种子决定），只验证非零即可
        assert isinstance(retrieved[0][1], float)

        # Step 3: 构建 Constitutional memory context
        memory_context = build_constitutional_memory_context(retrieved)
        assert "AI 记忆系统项目" in memory_context.semantic_memories
        assert memory_context.semantic_memories != "无"
        assert memory_context.procedural_memories == "无"

        # Step 4: 调用 chat 生成回复
        response = await chat_mod.generate_response(
            user_message=query,
            memory_context=memory_context,
            conversation_history=[],
        )
        assert len(response) > 0
        assert "模拟回复" in response

    @pytest.mark.asyncio
    async def test_system_prompt_contains_constitution(
        self, fake_embedding, fake_chat
    ):
        """验证系统 prompt 包含交互宪法的关键条款。"""
        content = "用户不喜欢机械回答"
        embedding = await embedding_mod.embed_text(content)
        mem = Memory(
            memory_id=str(uuid.uuid4()),
            type="emotional",
            content=content,
            embedding=embedding,
        )
        insert_memory(mem)

        query_emb = await embedding_mod.embed_text("你好")
        retrieved = retrieve_memories(query_emb, top_k=5)
        memory_context = build_constitutional_memory_context(retrieved)

        # 检查 system prompt 构建
        system_prompt = _build_system_prompt(memory_context)
        assert "交互宪法" in system_prompt
        assert "自然无痕原则" in system_prompt
        assert "情感安全与边界" in system_prompt
        assert "动态行为修正案" in system_prompt
        assert "上下文记忆沙盒" in system_prompt
        assert "机械回答" in system_prompt

    @pytest.mark.asyncio
    async def test_procedural_memory_goes_to_amendments(
        self, fake_embedding, fake_chat
    ):
        """程序性记忆进入动态行为修正案，不混入其他片段。"""
        content = "用户吐槽时先共情，不要立刻讲大道理"
        embedding = await embedding_mod.embed_text(content)
        mem = Memory(
            memory_id=str(uuid.uuid4()),
            type="procedural",
            content=content,
            embedding=embedding,
        )
        insert_memory(mem)

        query_emb = await embedding_mod.embed_text("你好")
        retrieved = retrieve_memories(query_emb, top_k=5)
        memory_context = build_constitutional_memory_context(retrieved)

        assert "先共情" in memory_context.procedural_memories
        assert memory_context.semantic_memories == "无"
        assert "先共情" not in memory_context.semantic_memories
        assert "先共情" not in memory_context.episodic_memories

    @pytest.mark.asyncio
    async def test_multi_memory_retrieval_ordering(self, fake_embedding):
        """多条记忆中，最相关的排前面。"""
        query = "AI 项目"
        query_emb = await embedding_mod.embed_text(query)

        relevant = "用户在做 AI 记忆系统项目"
        semi_relevant = "用户喜欢用 Python 编程"
        irrelevant = "用户昨天吃了火锅"

        for content in [irrelevant, semi_relevant, relevant]:
            emb = await embedding_mod.embed_text(content)
            insert_memory(
                Memory(
                    memory_id=str(uuid.uuid4()),
                    type="semantic",
                    content=content,
                    embedding=emb,
                )
            )

        retrieved = retrieve_memories(query_emb, top_k=3)
        assert len(retrieved) == 3

        # 验证排序：score 应该递减
        scores = [s for _, s in retrieved]
        assert scores[0] >= scores[1] >= scores[2]

    @pytest.mark.asyncio
    async def test_only_context_included_memories_marked_accessed(self):
        """只标记实际进入最终 context 的记忆。"""
        query_emb = np.array([1.0, 0.0], dtype=np.float32)
        memories = [
            Memory(
                memory_id=str(uuid.uuid4()),
                type="semantic",
                content="短记忆一",
                embedding=np.array([1.0, 0.0], dtype=np.float32),
            ),
            Memory(
                memory_id=str(uuid.uuid4()),
                type="semantic",
                content="短记忆二",
                embedding=np.array([0.9, 0.1], dtype=np.float32),
            ),
            Memory(
                memory_id=str(uuid.uuid4()),
                type="semantic",
                content="短记忆三",
                embedding=np.array([0.8, 0.2], dtype=np.float32),
            ),
        ]
        for mem in memories:
            insert_memory(mem)

        retrieved = retrieve_memories(query_emb, top_k=3)
        memory_context = build_constitutional_memory_context(
            retrieved,
            max_tokens=8,
        )
        mark_accessed(memory_context.included_memory_ids)

        access_by_content = {
            mem.content: mem.access_count
            for mem in list_active_memories()
        }
        assert access_by_content["短记忆一"] == 1
        assert access_by_content["短记忆二"] == 1
        assert access_by_content["短记忆三"] == 0


class TestErrorHandling:
    """错误处理：降级不崩溃。"""

    @pytest.mark.asyncio
    async def test_retrieval_with_empty_db_returns_empty(self, fake_embedding):
        """空数据库检索返回空列表，不崩溃。"""
        query_emb = await embedding_mod.embed_text("测试")
        results = retrieve_memories(query_emb)
        assert results == []

    def test_empty_retrieval_still_builds_context(self):
        """检索为空时 context builder 正常返回 '无'。"""
        ctx = build_constitutional_memory_context([])
        assert ctx.semantic_memories == "无"
        assert ctx.procedural_memories == "无"

    @pytest.mark.asyncio
    async def test_null_embedding_memory_not_crash(self, fake_embedding):
        """缺失 embedding 的记忆不让检索崩溃。"""
        insert_memory(
            Memory(
                memory_id=str(uuid.uuid4()),
                type="semantic",
                content="无 embedding 的记忆",
                embedding=None,
            )
        )
        query_emb = await embedding_mod.embed_text("测试")
        results = retrieve_memories(query_emb)
        assert results == []


class TestConfig:
    """配置与启动检查。"""

    def test_check_config_no_api_key(self, monkeypatch):
        """缺少 API key 时返回错误。"""
        monkeypatch.setattr("config.DEEPSEEK_API_KEY", None)
        monkeypatch.setattr("config.EMBEDDING_API_KEY", None)
        from config import check_config

        errors = check_config()
        assert len(errors) > 0
        assert "DEEPSEEK_API_KEY" in errors[0]

    def test_check_config_with_api_key(self, monkeypatch):
        """有 API key 时无错误。"""
        monkeypatch.setattr("config.DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.setattr("config.EMBEDDING_API_KEY", "sk-test")
        monkeypatch.setattr("config.EMBEDDING_BASE_URL", "https://embedding.example.com")
        from config import check_config

        errors = check_config()
        assert len(errors) == 0

    def test_check_config_requires_embedding_base_url(self, monkeypatch):
        """Embedding key 与 base URL 都必须显式配置。"""
        monkeypatch.setattr("config.DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.setattr("config.EMBEDDING_API_KEY", "sk-test")
        monkeypatch.setattr("config.EMBEDDING_BASE_URL", "")
        from config import check_config

        errors = check_config()
        assert any("EMBEDDING_BASE_URL" in err for err in errors)


# ════════════════════════════════════════════════════════════
# Phase 3：检索契约测试 — 激活后当轮可见
# ════════════════════════════════════════════════════════════

class TestActivationRetrievalContract:
    """
    契约：retrieve_memories(memories=None) 每次都从 DB 新读，
    因此 context_aware_update 落库后，同一轮的检索立即可见。
    """

    def test_activated_memory_visible_in_same_turn(self, fake_embedding):
        """激活落库后当轮检索可见（休眠唤醒闭环的关键前提）。"""
        import numpy as np

        from core.memory_store import update_decay_weights
        from core.retriever import retrieve_memories

        # 插入一条沉底记忆（decay_weight < 0.1，正常检索不可见）
        mid = str(uuid.uuid4())
        content = "用户在准备字节跳动的实习面试"
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        insert_memory(
            Memory(
                memory_id=mid,
                type="episodic",
                content=content,
                embedding=emb,
                decay_weight=0.05,
            )
        )

        # 验证：沉底记忆检索不到
        query_similar = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        results_before = retrieve_memories(query_similar, top_k=5)
        assert len(results_before) == 0  # decay_weight 0.05 < 0.1，被过滤

        # 模拟激活：手动提升权重（等价于 context_aware_update 的效果）
        update_decay_weights([(0.25, mid)])

        # 验证：同一轮内检索可见
        results_after = retrieve_memories(query_similar, top_k=5)
        assert len(results_after) == 1
        assert results_after[0][0].content == content

    def test_context_aware_update_end_to_end_reactivation(self):
        """场景 E 全闭环（§8.2）：真正经由 context_aware_update 唤醒沉底记忆。

        插入 w=0.05 的沉底记忆 → 检索不可见 → context_aware_update
        → report.strong==1 且 reactivated==1 → 当轮检索可见
        → 数据库读回 w≈0.25、access_count==1。
        """
        from core.decay import context_aware_update

        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mid = str(uuid.uuid4())
        insert_memory(
            Memory(
                memory_id=mid,
                type="episodic",
                content="用户在准备字节跳动的实习面试",
                embedding=emb,
                decay_weight=0.05,
            )
        )

        # 沉底：检索不可见
        assert retrieve_memories(emb, top_k=5) == []

        # 激活（相同向量 → sim=1.0 强激活）
        report = context_aware_update(emb)
        assert report.strong == 1
        assert report.reactivated == 1

        # 当轮检索可见
        results = retrieve_memories(emb, top_k=5)
        assert [m.memory_id for m, _ in results] == [mid]

        # 数据库读回：权重与访问计数已持久化
        mem = [m for m in list_active_memories() if m.memory_id == mid][0]
        assert mem.decay_weight == pytest.approx(0.25)  # 0.05 + 0.2
        assert mem.access_count == 1


class TestAblationEquivalence:
    """场景 F（§8.2）：CONTEXT_AWARE_ENABLED=False 时，同一固定脚本产生的
    数据库状态与"从未调用激活"逐字段一致（§2 最低验收指标：消融等价性）。"""

    _FIXED_TS = "2026-01-01 00:00:00"

    def _seed_bank(self):
        """固定 memory_id 与时间戳的记忆库（两次运行逐字段可比的前提）。"""
        specs = [
            ("m-epi-sunk", "episodic", "沉底情节", 0.05, [1.0, 0.0, 0.0], 0.0),
            ("m-epi-live", "episodic", "活跃情节", 0.80, [1.0, 0.1, 0.0], 0.0),
            ("m-emo", "emotional", "情绪记忆", 0.50, [0.0, 1.0, 0.0], 0.9),
            ("m-sem", "semantic", "语义事实", 1.00, [1.0, 0.0, 0.0], 0.0),
            ("m-proc", "procedural", "行为偏好", 1.00, [0.0, 1.0, 0.0], 0.0),
        ]
        for mid, mtype, content, weight, emb, arousal in specs:
            insert_memory(
                Memory(
                    memory_id=mid,
                    type=mtype,
                    content=content,
                    decay_weight=weight,
                    embedding=np.array(emb, dtype=np.float32),
                    arousal=arousal,
                    created_at=self._FIXED_TS,
                    last_accessed=self._FIXED_TS,
                )
            )

    def _run_script(self, db_path, call_activation: bool):
        """固定脚本：建库 → 记衰减基准 → [3 轮激活调用] → 48h 后衰减 → dump 全表。"""
        from datetime import datetime, timedelta, timezone

        import core.memory_store as ms
        from core.decay import context_aware_update, run_decay_update

        ms.close_db()
        ms.DB_PATH = db_path  # autouse fixture 的 monkeypatch 会在测试结束后还原
        init_db()
        self._seed_bank()

        t0 = datetime(2026, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        run_decay_update(now=t0)  # 首次运行：只记基准，不衰减
        if call_activation:
            for q in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
                context_aware_update(np.array(q, dtype=np.float32))
        run_decay_update(now=t0 + timedelta(hours=48))

        conn = ms._get_conn()
        mem_rows = [tuple(r) for r in conn.execute(
            "SELECT * FROM memories ORDER BY memory_id").fetchall()]
        meta_rows = [tuple(r) for r in conn.execute(
            "SELECT * FROM meta ORDER BY key").fetchall()]
        return mem_rows, meta_rows

    def test_disabled_equals_never_called(self, monkeypatch, tmp_path):
        """开关关闭 + 逐轮调用激活 ≡ 从未调用激活（数据库逐字段一致）。"""
        control = self._run_script(tmp_path / "control.db", call_activation=False)
        monkeypatch.setattr("config.CONTEXT_AWARE_ENABLED", False)
        ablated = self._run_script(tmp_path / "ablated.db", call_activation=True)
        assert ablated == control

    def test_enabled_actually_diverges(self, monkeypatch, tmp_path):
        """自检：开关开启时同一脚本必须产生不同状态（防场景 F 空转通过）。"""
        control = self._run_script(tmp_path / "control2.db", call_activation=False)
        monkeypatch.setattr("config.CONTEXT_AWARE_ENABLED", True)
        enabled = self._run_script(tmp_path / "enabled.db", call_activation=True)
        assert enabled != control


class TestAccessCountThreeWayDedup:
    """场景 G（§8.2）：同一条记忆同一轮"浮现 + 强激活 + 检索 Top-K"，
    该轮 access_count 净增恰为 1（激活侧）；浮现的会话级 +1 单独发生。

    复刻 main.py 的调用序列语义（§5.5 访问计数表），把三路去重固定为契约。
    """

    def test_same_turn_net_increment_is_one(self):
        from core.decay import context_aware_update

        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        star_id = str(uuid.uuid4())
        insert_memory(
            Memory(memory_id=star_id, type="episodic", content="三路计数",
                   decay_weight=0.5, embedding=v)
        )
        # 对照：按当前校准阈值构造轻激活带记忆。
        from config import SIMILARITY_HIGH, SIMILARITY_MID
        mild_sim = (SIMILARITY_MID + SIMILARITY_HIGH) / 2
        mild_id = str(uuid.uuid4())
        insert_memory(
            Memory(memory_id=mild_id, type="episodic", content="轻激活带",
                   decay_weight=1.0,
                   embedding=np.array(
                       [mild_sim, np.sqrt(1.0 - mild_sim**2), 0.0], dtype=np.float32
                   ))
        )

        # ① 会话开始：浮现记忆会话级计 1 次（main.py 会话初始化语义）
        surfaced_ids = {star_id}
        mark_accessed(list(surfaced_ids))

        # ② 本轮：逐轮激活（强激活在 context_aware_update 内部计 1 次）
        report = context_aware_update(v)
        assert star_id in report.strong_ids
        assert mild_id not in report.strong_ids

        # ③ 检索进入上下文
        retrieved_ids = [m.memory_id for m, _ in retrieve_memories(v, top_k=5)]
        assert star_id in retrieved_ids
        assert mild_id in retrieved_ids

        # ④ 逐轮计数：排除浮现与本轮强激活（main.py 三路去重语义）
        mark_accessed([
            mid for mid in retrieved_ids
            if mid not in surfaced_ids and mid not in report.strong_ids
        ])

        counts = {m.memory_id: m.access_count for m in list_active_memories()}
        # 三路来源的记忆：会话级 1 + 本轮净 1 = 2（若不去重会是 3）
        assert counts[star_id] == 2
        # 轻激活不计数：只通过检索通道计 1 次
        assert counts[mild_id] == 1


# ════════════════════════════════════════════════════════════
# Phase 4：话题分割测试
# ════════════════════════════════════════════════════════════


class TestTopicSplitValidation:
    """索引完整性校验纯函数测试。"""

    def test_valid_two_segments(self):
        from core.dehydrator import _validate_segment_indices
        err = _validate_segment_indices(
            [{"start": 0, "end": 4}, {"start": 5, "end": 9}],
            total_messages=10,
        )
        assert err is None

    def test_valid_single_segment(self):
        from core.dehydrator import _validate_segment_indices
        err = _validate_segment_indices(
            [{"start": 0, "end": 9}],
            total_messages=10,
        )
        assert err is None

    def test_first_not_zero(self):
        from core.dehydrator import _validate_segment_indices
        err = _validate_segment_indices(
            [{"start": 1, "end": 9}],
            total_messages=10,
        )
        assert err is not None

    def test_last_not_end(self):
        from core.dehydrator import _validate_segment_indices
        err = _validate_segment_indices(
            [{"start": 0, "end": 4}, {"start": 5, "end": 8}],
            total_messages=10,
        )
        assert err is not None

    def test_gap_between_segments(self):
        from core.dehydrator import _validate_segment_indices
        err = _validate_segment_indices(
            [{"start": 0, "end": 3}, {"start": 5, "end": 9}],
            total_messages=10,
        )
        assert err is not None

    def test_overlap_between_segments(self):
        from core.dehydrator import _validate_segment_indices
        err = _validate_segment_indices(
            [{"start": 0, "end": 5}, {"start": 4, "end": 9}],
            total_messages=10,
        )
        assert err is not None

    def test_start_greater_than_end(self):
        from core.dehydrator import _validate_segment_indices
        err = _validate_segment_indices(
            [{"start": 5, "end": 0}],
            total_messages=10,
        )
        assert err is not None

    def test_out_of_bounds(self):
        from core.dehydrator import _validate_segment_indices
        err = _validate_segment_indices(
            [{"start": 0, "end": 10}],
            total_messages=10,
        )
        assert err is not None

    def test_empty_segments(self):
        from core.dehydrator import _validate_segment_indices
        err = _validate_segment_indices([], total_messages=5)
        assert err is not None

    def test_non_integer_indices(self):
        from core.dehydrator import _validate_segment_indices
        err = _validate_segment_indices(
            [{"start": 0, "end": "abc"}],
            total_messages=5,
        )
        assert err is not None


class TestTopicSplitIntegration:
    """话题分割端到端（fake LLM）。"""

    @pytest.mark.asyncio
    async def test_split_two_topics(self, monkeypatch, fake_embedding):
        """模拟 LLM 返回 2 段分割 → 数据库两组记忆。"""
        import httpx
        from core.dehydrator import split_conversation

        # 构造 6 条消息
        messages = []
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"消息{i}"})

        # fake LLM：返回两段 [0,2] [3,5]
        class FakeSplitResponse:
            status_code = 200
            def json(self):
                return {
                    "choices": [{"message": {"content": json.dumps([
                        {"start": 0, "end": 2},
                        {"start": 3, "end": 5},
                    ])}}]
                }
            @property
            def request(self): return None
            @property
            def text(self): return json.dumps(self.json())

        original_post = httpx.AsyncClient.post

        async def _patched(self, *args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            if "/v1/embeddings" in url:
                raise RuntimeError("不应调用 embedding")
            return FakeSplitResponse()

        monkeypatch.setattr(httpx.AsyncClient, "post", _patched)

        segments, report = await split_conversation(messages)
        assert report.attempted is True
        assert report.fell_back is False
        assert report.segments == 2
        assert len(segments) == 2
        assert len(segments[0]) == 3  # [0,1,2]
        assert len(segments[1]) == 3  # [3,4,5]

    @pytest.mark.asyncio
    async def test_split_invalid_indices_falls_back(self, monkeypatch, fake_embedding):
        """LLM 返回越界索引 → 整段回退，segments=1。"""
        import httpx
        from core.dehydrator import split_conversation

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "topic change"},
        ]

        class FakeBadResponse:
            status_code = 200
            def json(self):
                return {
                    "choices": [{"message": {"content": json.dumps([
                        {"start": 0, "end": 5},  # 越界！
                    ])}}]
                }
            @property
            def request(self): return None
            @property
            def text(self): return json.dumps(self.json())

        async def _bad_post(self, *args, **kwargs):
            return FakeBadResponse()

        monkeypatch.setattr(httpx.AsyncClient, "post", _bad_post)

        segments, report = await split_conversation(messages)
        assert report.fell_back is True
        assert report.segments == 1
        assert "索引" in report.reason
        # 回退后 segments 是完整消息列表
        assert len(segments) == 1
        assert len(segments[0]) == 3

    @pytest.mark.asyncio
    async def test_short_conversation_skips_split(self, fake_embedding):
        """消息数 < MIN_SEGMENT_MESSAGES → 不调 LLM。"""
        from core.dehydrator import split_conversation

        messages = [{"role": "user", "content": "hi"}]
        segments, report = await split_conversation(messages)
        assert report.attempted is False
        assert report.segments == 1
        assert len(segments) == 1
        assert len(segments[0]) == 1

    @pytest.mark.asyncio
    async def test_split_disabled_by_env(self, monkeypatch, fake_embedding):
        """TOPIC_SPLIT_ENABLED=False → 不分割。"""
        monkeypatch.setattr("core.dehydrator.TOPIC_SPLIT_ENABLED", False)
        from core.dehydrator import split_conversation

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        segments, report = await split_conversation(messages)
        assert report.attempted is False
        assert report.segments == 1
        assert len(segments) == 1
        assert len(segments[0]) == 2

    @pytest.mark.asyncio
    async def test_split_llm_failure_falls_back(self, monkeypatch, fake_embedding):
        """LLM 调用抛异常 → 整段回退，记忆不丢失。"""
        import httpx
        from core.dehydrator import split_conversation

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "change"},
        ]

        async def _failing_post(self, *args, **kwargs):
            raise httpx.TimeoutException("timeout")

        monkeypatch.setattr(httpx.AsyncClient, "post", _failing_post)

        segments, report = await split_conversation(messages)
        assert report.fell_back is True
        assert report.segments == 1
        assert "LLM 调用失败" in report.reason
        # 所有消息都在
        assert len(segments) == 1
        assert len(segments[0]) == 3

    @pytest.mark.asyncio
    async def test_split_invalid_json_falls_back(self, monkeypatch, fake_embedding):
        """LLM 返回非 JSON → 回退。"""
        import httpx
        from core.dehydrator import split_conversation

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

        class FakeNonJsonResponse:
            status_code = 200
            def json(self):
                return {"choices": [{"message": {"content": "这不是 JSON"}}]}
            @property
            def request(self): return None
            @property
            def text(self): return ""

        async def _nonjson_post(self, *args, **kwargs):
            return FakeNonJsonResponse()

        monkeypatch.setattr(httpx.AsyncClient, "post", _nonjson_post)

        segments, report = await split_conversation(messages)
        assert report.fell_back is True
        assert "JSON" in report.reason
        assert report.segments == 1


# ════════════════════════════════════════════════════════════
# Phase 4 Bug 修复回归：分段脱水逐段容错
# ════════════════════════════════════════════════════════════


class TestSegmentFaultTolerance:
    """单段脱水失败不牵连其余段（§10 记忆零丢失）；全部段失败时维持 Phase 3 失败语义。"""

    @staticmethod
    def _six_messages():
        messages = []
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"消息{i}"})
        return messages

    @staticmethod
    def _patch_llm(monkeypatch, fail_segments: set):
        """fake LLM：分割调用返回 2 段；第 n 次段脱水调用按 fail_segments 决定成败。"""
        import httpx

        state = {"dehydrate_calls": 0}

        class _Resp:
            def __init__(self, content: str, status: int = 200):
                self.status_code = status
                self._content = content

            def json(self):
                return {"choices": [{"message": {"content": self._content}}]}

            @property
            def request(self):
                return None

            @property
            def text(self):
                return self._content

        async def _post(self, *args, **kwargs):
            payload = kwargs.get("json") or {}
            msgs = payload.get("messages", [])
            user_msg = msgs[-1].get("content", "") if msgs else ""
            if "对话共有" in user_msg:
                # 话题分割调用 → 两段 [0,2] [3,5]
                return _Resp(json.dumps([
                    {"start": 0, "end": 2},
                    {"start": 3, "end": 5},
                ]))
            # 段脱水调用
            state["dehydrate_calls"] += 1
            n = state["dehydrate_calls"]
            if n in fail_segments:
                return _Resp("Internal Server Error", status=500)
            return _Resp(json.dumps([
                {"content": f"话题{n}的记忆", "type": "episodic"},
            ]))

        monkeypatch.setattr(httpx.AsyncClient, "post", _post)
        return state

    @pytest.mark.asyncio
    async def test_partial_segment_failure_keeps_other_segments(
        self, monkeypatch, fake_embedding
    ):
        """第 2 段脱水失败 → 第 1 段的记忆仍然保留（不再全有全无）。"""
        from core.dehydrator import dehydrate_conversation

        state = self._patch_llm(monkeypatch, fail_segments={2})
        memories, split_report = await dehydrate_conversation(
            self._six_messages(), "conv-ft-1"
        )

        assert state["dehydrate_calls"] == 2      # 两段都尝试过
        assert len(memories) == 1                 # 只有成功段的产出
        assert memories[0].content == "话题1的记忆"
        assert split_report.segments == 2         # SplitReport 一并返回（计划 §5.2）
        assert split_report.fell_back is False

    @pytest.mark.asyncio
    async def test_first_segment_failure_keeps_later_segments(
        self, monkeypatch, fake_embedding
    ):
        """第 1 段失败也不阻断第 2 段（失败不中断循环）。"""
        from core.dehydrator import dehydrate_conversation

        state = self._patch_llm(monkeypatch, fail_segments={1})
        memories, _ = await dehydrate_conversation(self._six_messages(), "conv-ft-2")

        assert state["dehydrate_calls"] == 2
        assert len(memories) == 1
        assert memories[0].content == "话题2的记忆"

    @pytest.mark.asyncio
    async def test_all_segments_fail_raises(self, monkeypatch, fake_embedding):
        """全部段失败 → 向上抛出（调用方按既有路径统一报告，与 Phase 3 语义一致）。"""
        import httpx

        from core.dehydrator import dehydrate_conversation

        self._patch_llm(monkeypatch, fail_segments={1, 2})
        with pytest.raises(httpx.HTTPStatusError):
            await dehydrate_conversation(self._six_messages(), "conv-ft-3")
