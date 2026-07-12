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
            "  请填写 embedding 服务的基础 URL（代码会追加 /v1/embeddings）。"
        )

    if not (0.0 < SIMILARITY_MID < SIMILARITY_HIGH <= 1.0):
        errors.append(
            f"激活阈值分层非法：要求 0 < SIMILARITY_MID < SIMILARITY_HIGH <= 1，"
            f"当前 MID={SIMILARITY_MID}、HIGH={SIMILARITY_HIGH}。\n"
            f"  MID >= HIGH 会使轻激活带 (MID, HIGH] 为空集，轻激活机制失效。"
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


def ensure_data_dir() -> None:
    """确保 data/ 目录存在。"""
    data_dir = DB_PATH.parent
    data_dir.mkdir(parents=True, exist_ok=True)
