"""
Phase 4 — Token 估算器。

保守估算文本的 token 数（不引入外部 tokenizer 依赖）。
系数来源：DeepSeek 官方口径（1 汉字 ≈ 0.6 token，1 英文字符 ≈ 0.3 token），
乘安全系数后估算值系统性偏高 ~10%，满足预算控制"宁高勿低"的唯一硬性要求。
"""

import logging
import math

from config import TOKEN_EST_CJK, TOKEN_EST_OTHER, TOKEN_EST_SAFETY

logger = logging.getLogger(__name__)

# CJK 字符的 Unicode 范围
_CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F), # CJK Unified Ideographs Extension C
    (0x2B740, 0x2B81F), # CJK Unified Ideographs Extension D
    (0x2B820, 0x2CEAF), # CJK Unified Ideographs Extension E
    (0x2CEB0, 0x2EBEF), # CJK Unified Ideographs Extension F
    (0x30000, 0x3134F), # CJK Unified Ideographs Extension G
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F), # CJK Compatibility Ideographs Supplement
    # 常用 CJK 标点／全角字符（按 CJK 系数估算更安全）
    (0x3000, 0x303F),   # CJK Symbols and Punctuation
    (0xFF00, 0xFFEF),   # Halfwidth and Fullwidth Forms
    (0xFE30, 0xFE4F),   # CJK Compatibility Forms
]


def _is_cjk(codepoint: int) -> bool:
    """判断一个 Unicode 码点是否属于 CJK 范围。"""
    for lo, hi in _CJK_RANGES:
        if lo <= codepoint <= hi:
            return True
    return False


def estimate_tokens(text: str) -> int:
    """
    保守估算文本的 token 数。

    est = ceil((cjk_chars · TOKEN_EST_CJK + other_chars · TOKEN_EST_OTHER)
               · TOKEN_EST_SAFETY)

    空字符串返回 0。非字符串类型 warning 后尝试转为字符串；转换失败时
    按 len() 以 CJK 系数兜底估算（宁高勿低，计划 §10）；连长度都取不到才返回 0。
    """
    if not isinstance(text, str):
        logger.warning(
            "estimate_tokens 收到非字符串类型 %s，转换后估算", type(text).__name__
        )
        try:
            text = str(text)
        except Exception:
            # 兜底：按对象长度 × CJK 系数（最保守的单字符成本）估算
            try:
                return math.ceil(len(text) * TOKEN_EST_CJK * TOKEN_EST_SAFETY)
            except Exception:
                return 0

    if not text:
        return 0

    cjk_count = 0
    other_count = 0

    for ch in text:
        if _is_cjk(ord(ch)):
            cjk_count += 1
        else:
            other_count += 1

    raw_estimate = cjk_count * TOKEN_EST_CJK + other_count * TOKEN_EST_OTHER
    safe_estimate = raw_estimate * TOKEN_EST_SAFETY

    return math.ceil(safe_estimate)
