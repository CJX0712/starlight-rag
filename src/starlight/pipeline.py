"""RAG 管道服务：ingest / query / delete 编排，唯一业务逻辑汇聚点。

依赖全部通过构造函数注入（Embedder/VectorStore/LexicalIndex/LLM），
本模块不关心具体实现，可整体用 fake 依赖独立验证。

作者：晨星
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from .chunking import chunk_text
from .config import Settings
from .fusion import rrf_fuse
from .generation import Answer, Source, build_prompt
from .protocols import Chunk, Embedder, LexicalIndex, LLM, VectorStore
from .rerank import heuristic_rerank


class RAGPipeline:
    """检索增强生成主链路：分块 -> 双路召回 -> RRF 融合 -> 有界重排 -> 生成。"""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        lexical: LexicalIndex,
        llm: LLM,
        settings: Settings | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._lexical = lexical
        self._llm = llm
        self._settings = settings or Settings()
        self._chunks: dict[str, Chunk] = {}
        self._docs: dict[str, dict[str, Any]] = {}

    def ingest_document(
        self,
        title: str,
        content: str,
        doc_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        if not content or not content.strip():
            raise ValueError("content 不能为空")
        doc_id = doc_id or uuid.uuid4().hex[:12]
        chunks = chunk_text(doc_id, content, self._settings.chunk_size, self._settings.chunk_overlap)
        if not chunks:
            raise ValueError("文档无可索引内容")
        vectors = self._embedder.embed([c.text for c in chunks])
        metas = [{"doc_id": doc_id} for _ in chunks]
        self._store.add([c.id for c in chunks], vectors, metas)
        self._lexical.add([c.id for c in chunks], [c.text for c in chunks], doc_id)
        for c in chunks:
            self._chunks[c.id] = c
        self._docs[doc_id] = {
            "title": title,
            "metadata": metadata or {},
            "chunk_ids": [c.id for c in chunks],
        }
        return {"doc_id": doc_id, "title": title, "chunks": len(chunks)}

    def query(self, question: str, top_k: int | None = None) -> Answer:
        if not question or not question.strip():
            raise ValueError("question 不能为空")
        t0 = time.perf_counter()
        k = top_k or self._settings.top_k

        qvec = self._embedder.embed([question])[0]
        dense_rank = self._store.search(qvec, k * 4)
        sparse_rank = self._lexical.search(question, k * 4)
        fused = rrf_fuse([dense_rank, sparse_rank], self._settings.rrf_k)[: k * 2]

        candidates = [
            (cid, score, self._chunks[cid].text) for cid, score in fused if cid in self._chunks
        ]
        ranked = heuristic_rerank(question, candidates, self._settings.rerank_weight)[:k]
        score_map = dict(ranked)
        picked = [self._chunks[cid] for cid, _ in ranked]

        prompt = build_prompt(question, [c.text for c in picked])
        text = self._llm.generate(prompt)
        sources = [
            Source(
                chunk_id=c.id,
                doc_id=c.doc_id,
                title=str(self._docs[c.doc_id]["title"]),
                snippet=c.text[:160],
                score=round(score_map[c.id], 4),
            )
            for c in picked
        ]
        return Answer(text=text, sources=sources, latency_ms=round((time.perf_counter() - t0) * 1000, 1))

    def list_documents(self) -> list[dict[str, Any]]:
        return [
            {"doc_id": did, "title": d["title"], "chunks": len(d["chunk_ids"]), "metadata": d["metadata"]}
            for did, d in self._docs.items()
        ]

    def delete_document(self, doc_id: str) -> None:
        if doc_id not in self._docs:
            raise KeyError(doc_id)
        entry = self._docs.pop(doc_id)
        self._store.delete_doc(doc_id)
        self._lexical.delete_doc(doc_id)
        for cid in entry["chunk_ids"]:
            self._chunks.pop(cid, None)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)
