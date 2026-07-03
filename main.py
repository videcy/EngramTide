"""
Phase 1 MVP — CLI 主入口。

职责：
- 启动配置检查。
- 初始化数据库。
- 管理会话 ID 和对话历史。
- 运行命令行对话循环。
- 退出时触发脱水写入。

运行方式：python main.py
"""

import asyncio
import logging
import sys
import uuid
from datetime import datetime, timezone

from config import (
    TOP_K_RETRIEVE,
    check_config,
    ensure_data_dir,
)
from core.memory_store import (
    close_db,
    init_db,
    list_active_memories,
    list_recent_memories,
    mark_accessed,
)
from core.decay import run_decay_update, get_surfaced_memories
from core.memory_writer import write_memories
from core.embedding import embed_text
from core.retriever import retrieve_memories
from core.context_builder import (
    ConstitutionalMemoryContext,
    build_constitutional_memory_context,
)
from core.chat import generate_response
from core.dehydrator import dehydrate_conversation

# ── 日志配置 ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("memory-engine")

# 控制 debug 输出的开关（运行时切换）
_debug_enabled: bool = False


def _debug(msg: str) -> None:
    """仅在 debug 模式下输出。"""
    if _debug_enabled:
        logger.info("[DEBUG] %s", msg)


# ── 欢迎信息 ──────────────────────────────────────────────

WELCOME = r"""
╔══════════════════════════════════════╗
║        🧠 Memory Engine MVP         ║
║    AI 长期记忆系统 · Phase 2         ║
╚══════════════════════════════════════╝

命令:
  /exit, /quit    退出并保存记忆
  /memories       查看最近 10 条记忆
  /debug on/off   切换调试模式
  /debug decay    查看衰减摘要
  /debug surface  查看浮现记忆
"""


def _print_recent_memories() -> None:
    """打印最近 10 条记忆。"""
    memories = list_recent_memories(limit=10)
    if not memories:
        print("（数据库中暂无记忆）")
        return

    print(f"\n--- 最近记忆（共 {len(memories)} 条）---")
    for i, m in enumerate(memories, 1):
        tags_str = ", ".join(m.tags) if m.tags else "无标签"
        print(f"  [{i}] [{m.type}] {m.content}")
        print(f"      标签: {tags_str}  |  访问: {m.access_count} 次")
    print()


# ── 主循环 ────────────────────────────────────────────────


async def main_loop() -> None:
    """命令行对话主循环。"""
    global _debug_enabled

    # 1. 配置检查
    errors = check_config()
    if errors:
        for err in errors:
            print(f"❌ {err}")
        sys.exit(1)

    ensure_data_dir()

    # 2. 初始化数据库
    init_db()
    logger.info("数据库已就绪。")

    # 3. 创建会话 ID
    source_conv_id = str(uuid.uuid4())
    conv_start = datetime.now(timezone.utc).isoformat()
    logger.info("会话 ID: %s", source_conv_id[:8])

    print(WELCOME)

    # 3.5 Phase 2：会话开始时跑衰减 + 提取浮现记忆
    decay_report = run_decay_update()
    logger.info(
        "衰减更新: 距上次 %.1f 小时, 更新 %d 条, 跳过 %d 条, 触底 %d 条",
        decay_report.hours_elapsed,
        decay_report.updated,
        decay_report.skipped,
        decay_report.floored,
    )
    surfaced_memories = get_surfaced_memories(list_active_memories())
    if _debug_enabled:
        _debug(f"浮现记忆: {len(surfaced_memories)} 条")
        for sm in surfaced_memories:
            _debug(f"  [{sm.type}] {sm.content[:60]}")

    # 4. 对话历史
    conversation_history: list[dict[str, str]] = []

    # 5. 主循环
    try:
        while True:
            # 读取用户输入
            try:
                user_input = input("👤 你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n")
                await _handle_exit(conversation_history, source_conv_id)
                return

            if not user_input:
                continue

            # ── 特殊命令 ──────────────────────────────────

            if user_input == "/exit" or user_input == "/quit":
                await _handle_exit(conversation_history, source_conv_id)
                return

            if user_input == "/memories":
                _print_recent_memories()
                continue

            if user_input == "/debug decay":
                print(f"📊 衰减摘要: 距上次 {decay_report.hours_elapsed:.1f}h, "
                      f"更新 {decay_report.updated} 条, "
                      f"跳过 {decay_report.skipped} 条, "
                      f"触底 {decay_report.floored} 条")
                continue

            if user_input == "/debug surface":
                if surfaced_memories:
                    print(f"--- 浮现记忆（共 {len(surfaced_memories)} 条）---")
                    for i, sm in enumerate(surfaced_memories, 1):
                        print(f"  [{i}] [{sm.type}] {sm.content}")
                else:
                    print("（当前无浮现记忆）")
                continue

            if user_input == "/debug on":
                _debug_enabled = True
                print("🔍 调试模式已开启")
                continue

            if user_input == "/debug off":
                _debug_enabled = False
                print("🔍 调试模式已关闭")
                continue

            # ── 正常对话流程 ───────────────────────────────

            try:
                # Step A: 用户输入做 embedding
                print("  ⏳ 检索相关记忆...", end="\r")
                query_embedding = await embed_text(user_input)
                _debug(f"Query embedding 维度: {query_embedding.shape[0]}")

                # Step B: 检索 Top-K 记忆
                retrieved = retrieve_memories(
                    query_embedding,
                    top_k=TOP_K_RETRIEVE,
                )

                if _debug_enabled and retrieved:
                    _debug(f"检索到 {len(retrieved)} 条记忆:")
                    for mem, score in retrieved:
                        _debug(f"  [{score:.3f}] [{mem.type}] {mem.content[:60]}")

                # Step C: 构建 Constitutional memory context（含浮现记忆）
                memory_context = build_constitutional_memory_context(
                    retrieved,
                    surfaced=surfaced_memories,
                    debug=_debug_enabled,
                )
                mark_accessed(memory_context.included_memory_ids)

                if _debug_enabled:
                    _debug("--- 动态行为修正案 ---")
                    _debug(memory_context.procedural_memories)
                    _debug("--- 基础用户信息 ---")
                    _debug(memory_context.semantic_memories)
                    _debug("--- 近期生活事件 ---")
                    _debug(memory_context.episodic_memories)
                    _debug("--- 历史情感沉淀 ---")
                    _debug(memory_context.emotional_memories)

                # Step D: 调用 LLM 生成回复
                print("  ⏳ 生成回复...", end="\r")
                response = await generate_response(
                    user_message=user_input,
                    memory_context=memory_context,
                    conversation_history=conversation_history,
                )

                # Step E: 输出并记录
                print(f"🤖 助手: {response}")

                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": response})

            except Exception as e:
                logger.error("❌ 本轮对话失败: %s", e)
                print(f"❌ 本轮 API 调用失败，请检查网络或 API 配置。")
                print(f"   错误详情: {e}")
                # 继续主循环，不中断

    except KeyboardInterrupt:
        print("\n")
        await _handle_exit(conversation_history, source_conv_id)


# ── 退出处理 ──────────────────────────────────────────────


async def _handle_exit(
    conversation_history: list[dict[str, str]],
    source_conv_id: str,
) -> None:
    """退出时触发脱水写入。"""
    # 过滤掉命令消息（以 / 开头），只保留真实对话
    real_messages = [
        msg
        for msg in conversation_history
        if not msg["content"].startswith("/")
    ]

    if not real_messages:
        print("本次会话无有效对话，跳过脱水。再见！👋")
        close_db()
        return

    print(f"\n📝 正在将会话压缩为记忆... ({len(real_messages)} 条消息)")

    try:
        memories = await dehydrate_conversation(real_messages, source_conv_id)
        if memories:
            write_report = await write_memories(memories)
            parts = []
            if write_report.inserted:
                parts.append(f"插入 {write_report.inserted}")
            if write_report.superseded:
                parts.append(f"覆盖 {write_report.superseded}")
            if write_report.reinforced:
                parts.append(f"强化 {write_report.reinforced}")
            if write_report.deduped:
                parts.append(f"去重 {write_report.deduped}")
            if write_report.failed:
                parts.append(f"失败 {write_report.failed}")
            detail = "、".join(parts) if parts else "无变化"
            logger.info(
                "写入管线: 插入 %d, 覆盖 %d, 强化 %d, 去重 %d, 失败 %d",
                write_report.inserted,
                write_report.superseded,
                write_report.reinforced,
                write_report.deduped,
                write_report.failed,
            )
            print(f"✅ 写入完成（{detail}）。")
        else:
            print("（未生成新记忆）")
    except Exception as e:
        logger.error("❌ 脱水写入失败: %s", e)
        print(f"⚠️ 本轮会话未写入记忆（脱水失败: {e}），但旧数据未受影响。")

    close_db()
    print("再见！👋")


# ── 入口 ──────────────────────────────────────────────────


def main() -> None:
    """程序入口。"""
    asyncio.run(main_loop())


if __name__ == "__main__":
    main()
