"""RAG 管道单测（全离线 fake 依赖）。作者：晨星"""
import pytest

from starlight.bm25_index import BM25Index
from starlight.config import Settings
from starlight.embeddings import HashEmbedder
from starlight.llms import MockLLM
from starlight.pipeline import RAGPipeline
from starlight.vectorstores import MemoryVectorStore

LONG_DOC = (
    "星环引擎是第三代聚变推进装置，额定输出功率 42 兆瓦，"
    "采用氦-3 燃料循环，散热系统使用液态金属回路。"
    "引擎在满载工况下的比冲达到 9000 秒，可为深空探测器提供持续推力。"
) * 16  # 约 1500 字符，显式超过默认 800 分块阈值，覆盖多块路径


@pytest.fixture
def pipeline():
    return RAGPipeline(HashEmbedder(), MemoryVectorStore(), BM25Index(), MockLLM(), Settings())


class TestIngest:
    def test_ingest_multi_chunk(self, pipeline):
        result = pipeline.ingest_document(title="星环引擎白皮书", content=LONG_DOC, doc_id="doc-engine")
        assert result["doc_id"] == "doc-engine"
        assert result["chunks"] > 1  # 文档超过阈值，必须真的分块

    def test_ingest_empty_content_rejected(self, pipeline):
        with pytest.raises(ValueError, match="不能为空"):
            pipeline.ingest_document(title="空文档", content="  ")

    def test_list_documents(self, pipeline):
        pipeline.ingest_document(title="白皮书", content=LONG_DOC, doc_id="doc-engine")
        docs = pipeline.list_documents()
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "doc-engine"
        assert docs[0]["chunks"] > 1


class TestQuery:
    def test_query_returns_answer_with_sources(self, pipeline):
        pipeline.ingest_document(title="星环引擎白皮书", content=LONG_DOC, doc_id="doc-engine")
        answer = pipeline.query("星环引擎的额定功率是多少？")
        assert answer.text  # 断言对外语义：答案非空
        assert len(answer.sources) > 0
        assert all(s.doc_id == "doc-engine" for s in answer.sources)
        assert answer.latency_ms >= 0

    def test_query_empty_question_rejected(self, pipeline):
        with pytest.raises(ValueError, match="不能为空"):
            pipeline.query("")

    def test_query_on_empty_knowledge_base(self, pipeline):
        answer = pipeline.query("任意问题")
        assert "未" in answer.text or answer.sources == []

    def test_irrelevant_query_ranks_relevant_doc_first(self, pipeline):
        pipeline.ingest_document(title="星环引擎白皮书", content=LONG_DOC, doc_id="doc-engine")
        pipeline.ingest_document(title="菜谱", content="红烧肉的做法：" * 100, doc_id="doc-cook")
        answer = pipeline.query("星环引擎的燃料循环")
        assert answer.sources[0].doc_id == "doc-engine"


class TestDelete:
    def test_delete_removes_from_all_indexes(self, pipeline):
        pipeline.ingest_document(title="白皮书", content=LONG_DOC, doc_id="doc-engine")
        pipeline.delete_document("doc-engine")
        assert pipeline.chunk_count == 0
        assert pipeline.list_documents() == []
        answer = pipeline.query("星环引擎")
        assert answer.sources == []

    def test_delete_missing_raises(self, pipeline):
        with pytest.raises(KeyError):
            pipeline.delete_document("no-such-doc")
