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
DEEPSEEK_CHAT_MODEL: str = _get_env("DEEPSEEK_CHAT_MODEL", "deepseek-chat")

# Embedding provider（独立于聊天模型，默认沿用 DeepSeek key/base_url 以兼容旧配置）
EMBEDDING_API_KEY: str | None = _get_env("EMBEDDING_API_KEY", DEEPSEEK_API_KEY)
EMBEDDING_BASE_URL: str = _get_env("EMBEDDING_BASE_URL", DEEPSEEK_BASE_URL)
EMBEDDING_MODEL: str = _get_env("EMBEDDING_MODEL", "text-embedding-3-small")


# ── 检索与记忆参数 ────────────────────────────────────────

TOP_K_RETRIEVE: int = 5
MAX_MEMORY_CONTEXT_CHARS: int = 4000
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
SURFACE_EMOTIONAL_MIN_DECAY: float = 0.3    # emotional 浮现的最低衰减权重
SURFACE_UNRESOLVED_MIN_DECAY: float = 0.2   # unresolved 浮现的最低衰减权重
SURFACE_RECENT_DAYS: int = 3                # episodic "近期事件"窗口（天）
SURFACE_EPISODIC_MIN_DECAY: float = 0.5     # 近期 episodic 浮现的最低衰减权重
MAX_SURFACED_MEMORIES: int = 8              # 浮现总数上限，防止上下文淹没

# ── 写入管线参数（Phase 2）──────────────────────────────────

SEMANTIC_OVERRIDE_THRESHOLD: float = 0.85    # semantic 覆盖检测相似度阈值
EMOTIONAL_REINFORCE_THRESHOLD: float = 0.85  # emotional 强化匹配相似度阈值
EMOTIONAL_REINFORCE_BOOST: float = 0.2       # 强化时 decay_weight 提升量
PROCEDURAL_DEDUP_THRESHOLD: float = 0.90     # procedural 写入去重相似度阈值


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
            "  如果 embedding 与聊天使用同一个 provider，可不单独设置 EMBEDDING_API_KEY，"
            "但必须提供 DEEPSEEK_API_KEY。\n"
            "  如果使用独立 embedding provider，请设置 EMBEDDING_API_KEY。"
        )

    return errors


def ensure_data_dir() -> None:
    """确保 data/ 目录存在。"""
    data_dir = DB_PATH.parent
    data_dir.mkdir(parents=True, exist_ok=True)
