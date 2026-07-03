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
            max_chars=12,
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
        from config import check_config

        errors = check_config()
        assert len(errors) == 0
