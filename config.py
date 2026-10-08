"""
Phase 1 MVP 配置模块。

从环境变量读取配置，提供默认值。
启动时检查必需配置（API key）。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件
load_dotenv()


def _get_env(key: str, default: str | None = None) -> str | None:
    """读取环境变量，去除首尾空白。"""
    value = os.getenv(key)
    if value is not None:
        value = value.strip()
        if value == "":
            return default
    return value if value is not None else default


# ── 路径配置 ──────────────────────────────────────────────

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 数据库路径
DB_PATH: Path = PROJECT_ROOT / _get_env("MEMORY_DB_PATH", "data/memories.db")

# Prompts 目录
PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"

# 部署专用的提示词覆盖目录（相对路径按项目根解析）。目录里有同名文件就用它，
# 没有就回落到 prompts/。用来给特定 Agent 换口吻，而不改仓库里的通用版本。
_prompts_override = _get_env("PROMPTS_OVERRIDE_DIR")
PROMPTS_OVERRIDE_DIR: Path | None = (
    PROJECT_ROOT / _prompts_override if _prompts_override else None
)


def prompt_path(filename: str) -> Path:
    """解析提示词文件：覆盖目录优先，其次 prompts/。"""
    if PROMPTS_OVERRIDE_DIR is not None:
        candidate = PROMPTS_OVERRIDE_DIR / filename
        if candidate.is_file():
            return candidate
    return PROMPTS_DIR / filename


# ── API 配置 ──────────────────────────────────────────────

DEEPSEEK_API_KEY: str | None = _get_env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL: str = _get_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Chat 模型
DEEPSEEK_CHAT_MODEL: str = _get_env("DEEPSEEK_CHAT_MODEL", "deepseek-v4-flash")

# Embedding provider 独立配置。DeepSeek 聊天 API 不提供此项目所需的 embedding 接口，
# 因此不再隐式复用聊天 key/base_url，避免首次请求才暴露配置错误。
EMBEDDING_API_KEY: str | None = _get_env("EMBEDDING_API_KEY")
EMBEDDING_BASE_URL: str = _get_env("EMBEDDING_BASE_URL", "") or ""
EMBEDDING_MODEL: str = _get_env("EMBEDDING_MODEL", "text-embedding-3-small")

# 目标 embedding 维度（P3）。text-embedding-3-* 系列基于 Matryoshka 表征学习，
# 前 k 维本身即有效低维表示，截断后检索质量损失很小（1536→512 约降 1~2 个 MTEB 点）。
# 请求体带 `dimensions`；provider 不支持时自动回退为「全维 + 本地截断 + 重新 L2 归一化」。
# ⚠ 改动此值后必须：① 跑 scripts/rebuild_embeddings.py 重建全库；
#                   ② 跑 scripts/calibrate_thresholds.py 重新标定 SIMILARITY_MID/HIGH。
EMBEDDING_DIM: int = int(_get_env("EMBEDDING_DIM", "512") or "512")

# 相似度矩阵乘的行分块大小（P1）。
# 不是为了省内存，是为了绕开 BLAS 的多线程派发开销：OpenBLAS 对超过约 500 行的
# gemv 会切到多线程，而这点计算量根本喂不饱线程池，同步开销反而吃掉十几毫秒
# （实测 1000×1536：整块 16ms / 256 行分块 0.2ms）。分块后始终走单线程内核。
VECTOR_MATMUL_BLOCK: int = int(_get_env("VECTOR_MATMUL_BLOCK", "256") or "256")

# active 记忆超过此数时提示切换 sqlite-vec；在此之前保持归一化矩阵暴力检索（P2 §切换阈值）
ANN_BACKEND_THRESHOLD: int = int(_get_env("ANN_BACKEND_THRESHOLD", "20000") or "20000")


# ── 检索与记忆参数 ────────────────────────────────────────

TOP_K_RETRIEVE: int = 5
MAX_HISTORY_MESSAGES: int = 20
DEHYDRATE_MAX_ITEMS: int = 8
REQUEST_TIMEOUT_SECONDS: int = 60

# ── 默认字段值 ────────────────────────────────────────────

DEFAULT_MEMORY_TYPE: str = "semantic"
DEFAULT_DECAY_WEIGHT: float = 1.0


# ── 衰减参数（Phase 2）──────────────────────────────────────

BASE_DECAY_RATE: float = 0.05           # episodic 基础衰减率
EMOTIONAL_DECAY_FACTOR: float = 0.7     # emotional 衰减折扣：rate = BASE × (1 - arousal × 0.7)
DECAY_FLOOR: float = 0.01               # 衰减下限，不完全遗忘
RETRIEVAL_MIN_DECAY: float = 0.1        # 低于此权重的记忆不参与检索

# ── 浮现参数（Phase 2）──────────────────────────────────────

SURFACE_AROUSAL_THRESHOLD: float = 0.7      # emotional 浮现的唤醒度阈值
SURFACE_EMOTIONAL_MIN_DECAY: float = float(
    _get_env("SURFACE_EMOTIONAL_MIN_DECAY", "0.3") or "0.3"
)  # emotional 浮现的最低衰减权重
SURFACE_UNRESOLVED_MIN_DECAY: float = float(
    _get_env("SURFACE_UNRESOLVED_MIN_DECAY", "0.2") or "0.2"
)  # unresolved 浮现的最低衰减权重
SURFACE_RECENT_DAYS: int = 3                # episodic "近期事件"窗口（天）
SURFACE_EPISODIC_MIN_DECAY: float = float(
    _get_env("SURFACE_EPISODIC_MIN_DECAY", "0.5") or "0.5"
)  # 近期 episodic 浮现的最低衰减权重
MAX_SURFACED_MEMORIES: int = 8              # 浮现总数上限，防止上下文淹没

# ── 写入管线参数（Phase 2）──────────────────────────────────

SEMANTIC_OVERRIDE_THRESHOLD: float = 0.85    # semantic 覆盖检测相似度阈值
EMOTIONAL_REINFORCE_THRESHOLD: float = 0.85  # emotional 强化匹配相似度阈值
EMOTIONAL_REINFORCE_BOOST: float = 0.2       # 强化时 decay_weight 提升量
PROCEDURAL_DEDUP_THRESHOLD: float = 0.90     # procedural 写入去重相似度阈值

# episodic 近重复门槛（P4-①）。定得比 semantic 的 0.85 高，只拦「同一件事被反复脱水」
# 的真重复；相似但不同时刻的事件仍各自成条。
EPISODIC_DEDUP_THRESHOLD: float = float(
    _get_env("EPISODIC_DEDUP_THRESHOLD", "0.88") or "0.88"
)
EPISODIC_DEDUP_ENABLED: bool = (
    _get_env("EPISODIC_DEDUP_ENABLED", "true").lower() != "false"
)
EPISODIC_REINFORCE_BOOST: float = 0.1        # episodic 近重复命中时的 decay_weight 提升量


# ── Context-Aware 激活参数（Phase 3）────────────────────────

CONTEXT_AWARE_ENABLED: bool = (
    _get_env("CONTEXT_AWARE_ENABLED", "true").lower() != "false"
)  # ★ 总开关：false 时逐轮激活完全关闭（消融/回退用）

# 默认阈值按 MiniLM 英文域内样本定标（experiment/calibration_report.json）：
# P95_unrelated=0.099、P25_related=0.558；MID 取间隙中点，HIGH 取 related P25。
SIMILARITY_HIGH: float = float(
    _get_env("SIMILARITY_HIGH", "0.56") or "0.56"
)  # MiniLM English cal: P25_related=0.56
SIMILARITY_MID: float = float(
    _get_env("SIMILARITY_MID", "0.33") or "0.33"
)  # MiniLM English cal: (P95_unrel+P25_rel)/2=0.33
ACTIVATION_HIGH: float = 0.2            # 强激活时 decay_weight 回升量
ACTIVATION_MID: float = 0.05            # 轻激活时 decay_weight 回升量
MAX_ACTIVATIONS_PER_TURN: int = 30      # ★ 单轮最多激活条数（按相似度降序保留）


# ── 话题分割参数（Phase 4）──────────────────────────────────

TOPIC_SPLIT_ENABLED: bool = (
    _get_env("TOPIC_SPLIT_ENABLED", "true").lower() != "false"
)                                       # ★ false 时整段脱水（Phase 3 等价）
MAX_TOPIC_SEGMENTS: int = 8             # 单会话最多分割段数（超出合并尾段，防 LLM 过度切分）
MIN_SEGMENT_MESSAGES: int = 2           # 少于该消息数的会话不分割（省 1 次 LLM 调用）

# ── 双通道检索参数（Phase 4）────────────────────────────────

KEYWORD_CHANNEL_ENABLED: bool = (
    _get_env("KEYWORD_CHANNEL_ENABLED", "true").lower() != "false"
)                                       # ★ false 时纯向量检索（Phase 3 等价）
RETRIEVAL_VECTOR_WEIGHT: float = 0.7    # 向量通道权重（总文档 §4.3.1）
RETRIEVAL_KEYWORD_WEIGHT: float = 0.3   # 关键词通道权重

# ── Token 预算参数（Phase 4，取代 MAX_MEMORY_CONTEXT_CHARS）──

MAX_CONTEXT_TOKENS: int = 1500          # 记忆上下文的 token 预算（总文档 §七）
TOKEN_EST_CJK: float = 0.6              # 每 CJK 字符的估算 token（DeepSeek 官方口径）
TOKEN_EST_OTHER: float = 0.3            # 每非 CJK 字符的估算 token
TOKEN_EST_SAFETY: float = 1.1           # 安全系数（估算宁高勿低）

# ── 激活会话上限（Phase 4）──────────────────────────────────

MILD_ONCE_PER_SESSION: bool = (
    _get_env("MILD_ONCE_PER_SESSION", "true").lower() != "false"
)                                       # ★ false 时轻激活无会话上限（Phase 3 等价/消融）

# ── 记忆合并参数（Phase 4）──────────────────────────────────

CONSOLIDATE_SIMILARITY: float = 0.92    # 合并候选相似度阈值（严格大于）
CONSOLIDATE_MAX_PAIRS: int = 20         # 单次 /consolidate 最多处理的候选对（控 LLM 成本）
CONSOLIDATE_SUGGEST_COUNT: int = 500    # 活跃 episodic+emotional 超此数时启动提示建议清理
CONSOLIDATED_MAX_LENGTH: int = 120      # 融合记忆内容的最大字数


# ── Phase 5 消融开关（环境变量可控）────────────────────────

ABL_ACCESS_RETRIEVAL: bool = (
    _get_env("ABL_ACCESS_RETRIEVAL", "on").lower() != "off"
)                                       # ★ off 时检索入上下文不计访问
ABL_ACCESS_ACTIVATION: bool = (
    _get_env("ABL_ACCESS_ACTIVATION", "on").lower() != "off"
)                                       # ★ off 时强激活不计访问
ABL_ACCESS_SURFACE: bool = (
    _get_env("ABL_ACCESS_SURFACE", "on").lower() != "off"
)                                       # ★ off 时浮现不计访问


# ── Phase 5 埋点日志开关（P0：生产默认关闭）─────────────
#
# 与上面的 ABL_ACCESS_* 是**正交**的两件事：
#   ABL_ACCESS_*      控制「是否计访问」（改变记忆语义，消融实验用）
#   *_LOG_ENABLED     控制「是否记录事件」（只影响观测，不改语义）
# 生产运行不需要埋点；打开时请配合 EVENT_LOG_RETENTION_DAYS 做保留期清理。

DECAY_LOG_ENABLED: bool = (
    _get_env("DECAY_LOG_ENABLED", "false").lower() == "true"
)
ACCESS_LOG_ENABLED: bool = (
    _get_env("ACCESS_LOG_ENABLED", "false").lower() == "true"
)
EVENT_LOG_RETENTION_DAYS: int = int(
    _get_env("EVENT_LOG_RETENTION_DAYS", "30") or "30"
)


# ── 关键词通道后端（P2）────────────────────────────────────

# FTS5（trigram 分词，对 CJK 有效）倒排索引。建表失败或本机 SQLite 未编译 FTS5 时
# 自动降级到 rapidfuzz 全表模糊匹配，再降级到纯向量。
FTS_ENABLED: bool = (
    _get_env("FTS_ENABLED", "true").lower() != "false"
)
FTS_CANDIDATE_LIMIT: int = int(_get_env("FTS_CANDIDATE_LIMIT", "50") or "50")


# ── 记忆增长治理（P4）──────────────────────────────────────

ARCHIVE_ENABLED: bool = (
    _get_env("ARCHIVE_ENABLED", "true").lower() != "false"
)
ARCHIVE_IDLE_DAYS: int = int(_get_env("ARCHIVE_IDLE_DAYS", "90") or "90")
SUPERSEDED_RETENTION_DAYS: int = int(
    _get_env("SUPERSEDED_RETENTION_DAYS", "30") or "30"
)
ARCHIVE_RUN_EVERY_SESSIONS: int = int(
    _get_env("ARCHIVE_RUN_EVERY_SESSIONS", "20") or "20"
)

CONSOLIDATE_AUTO_ENABLED: bool = (
    _get_env("CONSOLIDATE_AUTO_ENABLED", "false").lower() == "true"
)                                       # ★ 自动合并会调用 LLM，默认关闭需显式开启
CONSOLIDATE_AUTO_THRESHOLD: int = int(
    _get_env("CONSOLIDATE_AUTO_THRESHOLD", "300") or "300"
)
CONSOLIDATE_MIN_PAIRS: int = int(_get_env("CONSOLIDATE_MIN_PAIRS", "5") or "5")
CONSOLIDATE_BLOCK_SIZE: int = int(
    _get_env("CONSOLIDATE_BLOCK_SIZE", "1000") or "1000"
)                                       # 候选对矩阵分块行数，控 O(N²) 峰值内存


# ── Claude Code hook 注入 ─────────────────────────────────

HOOK_ENABLED: bool = (
    _get_env("HOOK_ENABLED", "true").lower() != "false"
)                                       # ★ false 时 hook 路由一律返回空 body（对照实验）
HOOK_DEDUP_INJECTED: bool = (
    _get_env("HOOK_DEDUP_INJECTED", "true").lower() != "false"
)                                       # ★ false 时每轮完整注入，不排除本会话已注入的记忆
HOOK_MIN_PROMPT_CHARS: int = int(_get_env("HOOK_MIN_PROMPT_CHARS", "4") or "4")
HOOK_MAX_CONTEXT_CHARS: int = int(
    _get_env("HOOK_MAX_CONTEXT_CHARS", "9000") or "9000"
)                                       # Claude Code 对 additionalContext 的上限是 10000 字符
HOOK_EMBED_TIMEOUT_SECONDS: float = float(
    _get_env("HOOK_EMBED_TIMEOUT_SECONDS", "4") or "4"
)                                       # 必须小于 settings 里 hook 的 timeout
HOOK_BUFFER_RETENTION_DAYS: int = int(
    _get_env("HOOK_BUFFER_RETENTION_DAYS", "7") or "7"
)                                       # 已脱水对话原文的保留期；未脱水的永不自动清理


# ── 启动检查 ──────────────────────────────────────────────

def check_config() -> list[str]:
    """
    检查配置完整性，返回错误列表。

    在 main.py 启动时调用；如果有错误，打印并退出。
    """
    errors: list[str] = []

    if not DEEPSEEK_API_KEY:
        errors.append(
            "缺少 DEEPSEEK_API_KEY 环境变量。\n"
            "  请创建 .env 文件并设置：DEEPSEEK_API_KEY=sk-your-key\n"
            "  或参考 .env.example 模板。"
        )

    if not EMBEDDING_API_KEY:
        errors.append(
            "缺少 EMBEDDING_API_KEY 环境变量。\n"
            "  请配置提供 OpenAI-compatible /v1/embeddings 接口的服务凭据。"
        )

    if not EMBEDDING_BASE_URL:
        errors.append(
            "缺少 EMBEDDING_BASE_URL 环境变量。\n"
            "  请填写 embedding 服务的基础 URL（以 /v1 结尾或不带都可以，代码会补全为 /v1/embeddings）。"
        )

    dim_error = check_embedding_dim()
    if dim_error:
        errors.append(dim_error)

    if not (0.0 < SIMILARITY_MID < SIMILARITY_HIGH <= 1.0):
        errors.append(
            f"激活阈值分层非法：要求 0 < SIMILARITY_MID < SIMILARITY_HIGH <= 1，"
            f"当前 MID={SIMILARITY_MID}、HIGH={SIMILARITY_HIGH}。\n"
            f"  MID >= HIGH 会使轻激活带 (MID, HIGH] 为空集，轻激活机制失效。"
        )

    if not (0 < HOOK_MAX_CONTEXT_CHARS <= 10_000):
        errors.append(
            f"HOOK_MAX_CONTEXT_CHARS={HOOK_MAX_CONTEXT_CHARS} 非法：要求 0 < 值 <= 10000。\n"
            f"  超过 10000 字符时 Claude Code 会把注入内容转存成文件，只留预览。"
        )

    # Phase 5 消融开关互斥检查（warning 而非 error——全部 off 本身就是 A4 正对照）
    off_count = sum([
        not ABL_ACCESS_RETRIEVAL,
        not ABL_ACCESS_ACTIVATION,
        not ABL_ACCESS_SURFACE,
    ])
    if off_count == 3:
        errors.append(
            "⚠ A4 正对照模式：全部消融开关关闭（ABL_ACCESS_*=off）。"
            "此模式下 λ̂ 应 ≡ λ_theory。"
        )

    return errors


def stored_embedding_dim() -> int | None:
    """
    读取库里记录的 embedding 维度（meta.embedding_dim）。

    直接开只读连接查 meta，不经 core.memory_store —— 后者 import config，
    走它会形成循环导入。库不存在 / meta 表未建 / 值损坏一律返回 None。
    """
    import sqlite3

    if not Path(DB_PATH).exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'embedding_dim'"
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    if not row:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


def check_embedding_dim() -> str | None:
    """
    校验库内 embedding 维度与当前 EMBEDDING_DIM 一致，不一致返回错误文案。

    这条校验很重要：维度不匹配不会抛异常，只会让相似度全变成垃圾值——
    静默失败，且表现为「记忆突然全都想不起来了」，极难定位。
    """
    stored = stored_embedding_dim()
    if stored is None or stored == EMBEDDING_DIM:
        return None
    return (
        f"embedding 维度不匹配：库内记录 {stored} 维，当前配置 EMBEDDING_DIM={EMBEDDING_DIM}。\n"
        f"  两种向量不可混用（相似度会静默退化为垃圾值）。请二选一：\n"
        f"    ① 重建全库：python scripts/rebuild_embeddings.py\n"
        f"       （随后必须跑 scripts/calibrate_thresholds.py 重标 SIMILARITY_MID/HIGH）\n"
        f"    ② 改回原维度：在 .env 设 EMBEDDING_DIM={stored}"
    )


def ensure_data_dir() -> None:
    """确保 data/ 目录存在。"""
    data_dir = DB_PATH.parent
    data_dir.mkdir(parents=True, exist_ok=True)
