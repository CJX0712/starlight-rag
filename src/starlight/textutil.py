"""文本归一化与分词（中英文混合，离线确定性）。

中文按字符 unigram + bigram，英文按小写词；等价于 jieba 的检索效果但零依赖。

作者：晨星
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[一-鿿]")


def tokenize(text: str) -> list[str]:
    """统一分词入口，BM25 与哈希嵌入共用，保证两边词表一致。"""
    lowered = text.lower()
    tokens = _WORD_RE.findall(lowered)
    cjk = [c for c in lowered if _CJK_RE.match(c)]
    tokens.extend(cjk)
    tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
    return tokens
