"""向量库实现：内存余弦（默认离线）+ FAISS IndexIDMap（生产）。

作者：晨星
"""
from __future__ import annotations

import hashlib
import math


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class MemoryVectorStore:
    """进程内字典 + 暴力余弦，零依赖离线实现。数据规模 <= 10k 块时足够。"""

    def __init__(self) -> None:
        self._items: dict[str, tuple[list[float], dict]] = {}

    def add(self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        for cid, vec, meta in zip(ids, vectors, metadatas):
            self._items[cid] = (list(vec), dict(meta))

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        scored = [(cid, _cosine(vector, vec)) for cid, (vec, _) in self._items.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def delete_doc(self, doc_id: str) -> None:
        self._items = {cid: vm for cid, vm in self._items.items() if vm[1].get("doc_id") != doc_id}

    def count(self) -> int:
        return len(self._items)


class FaissVectorStore:
    """FAISS 生产实现：IndexFlatIP + 归一化向量（等价余弦），IDMap 支持删除。"""

    def __init__(self, dim: int) -> None:
        import faiss

        self._dim = dim
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self._meta: dict[str, dict] = {}
        self._int_to_id: dict[int, str] = {}

    @staticmethod
    def _to_int_id(chunk_id: str) -> int:
        digest = hashlib.blake2b(chunk_id.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF

    def add(self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        import numpy as np

        arr = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        int_ids = np.asarray([self._to_int_id(cid) for cid in ids], dtype="int64")
        self._index.add_with_ids(arr, int_ids)
        for cid, iid, meta in zip(ids, int_ids.tolist(), metadatas):
            self._meta[cid] = dict(meta)
            self._int_to_id[iid] = cid

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        import numpy as np

        if self._index.ntotal == 0:
            return []
        arr = np.asarray([vector], dtype="float32")
        norm = np.linalg.norm(arr) or 1.0
        scores, int_ids = self._index.search(arr / norm, min(k, self._index.ntotal))
        out: list[tuple[str, float]] = []
        for iid, score in zip(int_ids[0].tolist(), scores[0].tolist()):
            cid = self._int_to_id.get(iid)
            if cid is not None:
                out.append((cid, float(score)))
        return out

    def delete_doc(self, doc_id: str) -> None:
        import numpy as np

        victims = [cid for cid, meta in self._meta.items() if meta.get("doc_id") == doc_id]
        if not victims:
            return
        int_ids = np.asarray([self._to_int_id(cid) for cid in victims], dtype="int64")
        self._index.remove_ids(int_ids)
        for cid in victims:
            self._meta.pop(cid, None)
            self._int_to_id.pop(self._to_int_id(cid), None)

    def count(self) -> int:
        return int(self._index.ntotal)
