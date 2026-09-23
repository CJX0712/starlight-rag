"""启发式重排：查询词覆盖率信号，权重有界。

工程教训（见 docs/decisions/ADR-004）：弱重排器若直接决定最终顺序，
会在查询语言上力不从心时无人制衡地压掉正确答案。因此重排分只占
weight 权重，与归一化 RRF 分线性混合，检索主信号始终占主导。

作者：晨星
"""
from __future__ import annotations

from .textutil import tokenize


def heuristic_rerank(
    query: str,
    candidates: list[tuple[str, float, str]],
    weight: float = 0.3,
) -> list[tuple[str, float]]:
    """candidates: (chunk_id, rrf_score, text)，返回按最终分降序的 (chunk_id, final_score)。"""
    if not candidates:
        return []
    weight = max(0.0, min(weight, 0.5))
    q_tokens = set(tokenize(query))
    max_rrf = max(score for _, score, _ in candidates) or 1.0
    out: list[tuple[str, float]] = []
    for cid, rrf_score, text in candidates:
        doc_tokens = set(tokenize(text))
        coverage = len(q_tokens & doc_tokens) / (len(q_tokens) or 1)
        final = (1 - weight) * (rrf_score / max_rrf) + weight * coverage
        out.append((cid, final))
    out.sort(key=lambda x: x[1], reverse=True)
    return out
