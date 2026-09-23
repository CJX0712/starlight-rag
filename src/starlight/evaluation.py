"""检索质量评估：recall@k 与 MRR。

铁律：评估必须运行在全新构建的管道上（由调用方传入工厂），
复用运行期已写入数据的单例管道会造成 doc_id 重复、指标失真。

作者：晨星
"""
from __future__ import annotations

from typing import Any, Callable

from .pipeline import RAGPipeline


def evaluate(
    pipeline_factory: Callable[[], RAGPipeline],
    corpus: list[dict[str, Any]],
    cases: list[dict[str, str]],
    k: int = 5,
) -> dict[str, Any]:
    """在全新管道上注入语料并逐案评估。

    corpus: [{"doc_id", "title", "content"}...]
    cases:  [{"question", "expect_doc_id"}...]
    """
    pipe = pipeline_factory()
    for doc in corpus:
        pipe.ingest_document(title=doc["title"], content=doc["content"], doc_id=doc.get("doc_id"))

    hits = 0
    rr_sum = 0.0
    details: list[dict[str, Any]] = []
    for case in cases:
        answer = pipe.query(case["question"], top_k=k)
        ranked_doc_ids = [s.doc_id for s in answer.sources]
        expect = case["expect_doc_id"]
        hit = expect in ranked_doc_ids
        hits += int(hit)
        rr = 1.0 / (ranked_doc_ids.index(expect) + 1) if hit else 0.0
        rr_sum += rr
        details.append({"question": case["question"], "expect_doc_id": expect, "hit": hit, "rr": round(rr, 4)})

    total = len(cases) or 1
    return {
        "cases": len(cases),
        "recall_at_k": round(hits / total, 4),
        "mrr": round(rr_sum / total, 4),
        "details": details,
    }
