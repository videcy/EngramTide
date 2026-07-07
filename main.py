"""
Phase 3 — CLI 主入口。

职责：
- 启动配置检查 + 数据库初始化 + 会话衰减与浮现。
- Phase 3：逐轮 Context-Aware 激活（检索前）。
- 运行命令行对话循环。
- 退出时触发脱水 + 写入管线。

运行方式：python main.py
"""

import asyncio
import logging
import sys
import uuid
from datetime import datetime, timezone

from config import (
    CONSOLIDATE_SUGGEST_COUNT,
    CONTEXT_AWARE_ENABLED,
    MAX_CONTEXT_TOKENS,
    MILD_ONCE_PER_SESSION,
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
from core.decay import run_decay_update, get_surfaced_memories, context_aware_update
from core.memory_writer import write_memories
from core.embedding import embed_text
from core.retriever import retrieve_memories_detailed
from core.context_builder import (
    ConstitutionalMemoryContext,
    build_constitutional_memory_context,
)
from core.chat import generate_response
from core.consolidator import consolidate_memories, find_merge_candidates
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
║        🧠 Memory Engine             ║
║    AI 长期记忆系统 · Phase 4         ║
╚══════════════════════════════════════╝

命令:
  /exit, /quit        退出并保存记忆
  /memories           查看最近 10 条记忆
  /consolidate        合并近重复记忆
  /consolidate preview预览合并候选（不执行）
  /debug on/off       切换调试模式
  /debug decay        查看衰减摘要
  /debug surface      查看浮现记忆
  /debug activation   查看激活统计
  /debug retrieval    查看检索分数分解
  /debug context      查看 token 预算用量
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
    surfaced_ids = {sm.memory_id for sm in surfaced_memories}

    # Phase 3 修复：浮现记忆每会话计 1 次访问
    if surfaced_ids:
        mark_accessed(list(surfaced_ids))

    if not CONTEXT_AWARE_ENABLED:
        logger.info("⚠ Context-Aware 激活已通过环境变量关闭（Phase 2 等价模式）")

    # Phase 4：启动时检查记忆库规模
    all_mems = list_active_memories()
    epi_emo_count = sum(1 for m in all_mems if m.type in ("episodic", "emotional"))
    if epi_emo_count > CONSOLIDATE_SUGGEST_COUNT:
        print(f"💡 活跃 episodic/emotional 记忆已达 {epi_emo_count} 条，"
              f"建议运行 /consolidate preview 检查可合并项。")

    if _debug_enabled:
        _debug(f"浮现记忆: {len(surfaced_memories)} 条")
        for sm in surfaced_memories:
            _debug(f"  [{sm.type}] {sm.content[:60]}")

    # Phase 3：激活累计追踪
    _session_activation_total: dict[str, int] = {"strong": 0, "mild": 0, "reactivated": 0, "suppressed": 0}
    _last_activation_details: list[str] = []

    # Phase 4：会话内轻激活上限（同一记忆至多轻激活 1 次/会话）
    session_mild_ids: set[str] = set()

    # Phase 4：debug 追踪
    _last_retrieval_details: list[str] = []
    _last_context_info: dict | None = None

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

            if user_input == "/debug activation":
                total = _session_activation_total
                print(f"📊 本会话累计激活: 强 {total['strong']} / 轻 {total['mild']} "
                      f"/ 唤醒 {total['reactivated']} / 抑制 {total['suppressed']}")
                if session_mild_ids:
                    print(f"  已轻激活 id 数（会话上限）: {len(session_mild_ids)}")
                if _last_activation_details:
                    print("  最近一轮明细:")
                    for detail in _last_activation_details:
                        print(f"    {detail}")
                else:
                    print("  （本会话尚无激活记录）")
                continue

            if user_input == "/debug on":
                _debug_enabled = True
                print("🔍 调试模式已开启")
                continue

            if user_input == "/debug off":
                _debug_enabled = False
                print("🔍 调试模式已关闭")
                continue

            if user_input == "/debug retrieval":
                if _last_retrieval_details:
                    print("--- 最近一轮检索分数分解 ---")
                    for detail in _last_retrieval_details:
                        print(f"  {detail}")
                else:
                    print("（本会话尚无检索记录）")
                continue

            if user_input == "/debug context":
                if _last_context_info:
                    info = _last_context_info
                    print(f"📊 Token 预算: {info['est_tokens']}/{info['budget']} tokens, "
                          f"收录 {info['included']} 条, 丢弃 {info['dropped']} 条")
                else:
                    print("（本会话尚无上下文构建记录）")
                continue

            if user_input == "/consolidate preview":
                all_mems = list_active_memories()
                pairs = find_merge_candidates(all_mems)
                if not pairs:
                    print("（无合并候选对）")
                else:
                    print(f"--- 合并候选（共 {len(pairs)} 对）---")
                    for i, (a, b, sim) in enumerate(pairs, 1):
                        print(f"  [{i}] sim={sim:.4f}")
                        print(f"      A: {a.content[:40]}")
                        print(f"      B: {b.content[:40]}")
                        print()
                continue

            if user_input == "/consolidate":
                print("  ⏳ 正在查找合并候选...", end="\r")
                all_mems = list_active_memories()
                pairs = find_merge_candidates(all_mems)
                if not pairs:
                    print("（无合并候选对）")
                    continue

                print(f"  发现 {len(pairs)} 对候选，正在执行合并...")
                report = await consolidate_memories(dry_run=False)
                print(f"✅ 合并完成: 候选 {report.candidates} 对, "
                      f"融合 {report.merged} 条, 跳过 {report.skipped} 条"
                      + (f", 截断 {report.capped} 对" if report.capped else ""))
                continue

            # ── 正常对话流程 ───────────────────────────────

            try:
                # Step A: 用户输入做 embedding
                print("  ⏳ 检索相关记忆...", end="\r")
                query_embedding = await embed_text(user_input)
                _debug(f"Query embedding 维度: {query_embedding.shape[0]}")

                # Step A.5 Phase 3+4：Context-Aware 逐轮激活（含轻激活会话上限）
                activation_report = context_aware_update(
                    query_embedding,
                    exclude_mild_ids=frozenset(session_mild_ids)
                        if MILD_ONCE_PER_SESSION else frozenset(),
                )
                # 记录本轮轻激活 id 到会话集合
                if MILD_ONCE_PER_SESSION and activation_report.mild_ids:
                    session_mild_ids.update(activation_report.mild_ids)

                if activation_report.strong or activation_report.mild or activation_report.mild_suppressed:
                    _session_activation_total["strong"] += activation_report.strong
                    _session_activation_total["mild"] += activation_report.mild
                    _session_activation_total["reactivated"] += activation_report.reactivated
                    _session_activation_total["suppressed"] += activation_report.mild_suppressed
                    # 用本轮明细刷新 /debug activation 的"最近一轮"缓存（按相似度降序）
                    _last_activation_details = [
                        f"[{'strong' if d.is_strong else 'mild'}|sim={d.sim:.2f}|"
                        f"w {d.old_weight:.2f}→{d.new_weight:.2f}] {d.content[:60]}"
                        for d in activation_report.details
                    ]
                    _debug(
                        f"激活: 强 {activation_report.strong} / 轻 {activation_report.mild} "
                        f"/ 唤醒 {activation_report.reactivated}"
                    )

                # Step B: 检索 Top-K 记忆（detailed 版，供分数分解）
                retrieval_details = retrieve_memories_detailed(
                    query_embedding,
                    top_k=TOP_K_RETRIEVE,
                    query_text=user_input,
                )
                retrieved = [(d.memory, d.score) for d in retrieval_details]

                # Phase 4：记录检索分数分解供 /debug retrieval（计划 §9.2 格式）
                _last_retrieval_details = [
                    f"[vec={d.sim:.2f}|"
                    f"kw={f'{d.kw:.2f}' if d.kw is not None else '--'}|"
                    f"w={d.memory.decay_weight:.2f}|→{d.score:.3f}] "
                    f"[{d.memory.type}] {d.memory.content[:40]}"
                    for d in retrieval_details
                ]

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
                # Phase 4：记录上下文信息供 /debug context
                _last_context_info = {
                    "est_tokens": memory_context.est_tokens,
                    "budget": MAX_CONTEXT_TOKENS,
                    "included": len(memory_context.included_memory_ids),
                    "dropped": memory_context.dropped_count,
                }
                # Phase 3：三路去重 — 排除浮现（已计）和本轮强激活（context_aware_update 已计）
                mark_accessed([
                    mid for mid in memory_context.included_memory_ids
                    if mid not in surfaced_ids
                    and mid not in activation_report.strong_ids
                ])

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
        memories, split_report = await dehydrate_conversation(real_messages, source_conv_id)
        # Phase 4：打印分割信息（计划 §5.2）
        if split_report.attempted and not split_report.fell_back and split_report.segments > 1:
            print(f"🔀 检测到 {split_report.segments} 个话题，已分段压缩。")
        elif split_report.fell_back:
            print(f"（话题分割回退：{split_report.reason}，整段压缩）")
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
