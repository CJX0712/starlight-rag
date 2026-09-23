"""RRF（Reciprocal Rank Fusion）多路召回融合。

作者：晨星
"""
from __future__ import annotations


def rrf_fuse(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """把多路 (id, score) 排名按 RRF 公式融合：score = Σ 1/(k + rank + 1)。

    只用名次不用原始分，天然免疫两路分数量纲不一致的问题。
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (item_id, _) in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
