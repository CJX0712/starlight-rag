"""BM25 稀疏检索：rank-bm25 封装，支持按文档删除（重建索引）。

作者：晨星
"""
from __future__ import annotations

from .textutil import tokenize


class BM25Index:
    """分词与哈希嵌入共用 tokenize，保证稀疏/稠密两路词表一致。"""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._doc_ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._bm25 = None

    def _rebuild(self) -> None:
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None

    def add(self, ids: list[str], texts: list[str], doc_id: str) -> None:
        for cid, text in zip(ids, texts):
            self._ids.append(cid)
            self._doc_ids.append(doc_id)
            self._tokens.append(tokenize(text))
        self._rebuild()

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._ids, (float(s) for s in scores)), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def delete_doc(self, doc_id: str) -> None:
        keep = [i for i, d in enumerate(self._doc_ids) if d != doc_id]
        self._ids = [self._ids[i] for i in keep]
        self._doc_ids = [self._doc_ids[i] for i in keep]
        self._tokens = [self._tokens[i] for i in keep]
        self._rebuild()
