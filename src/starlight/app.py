"""服务装配层：按配置组装依赖 + FastAPI 路由。零业务规则，业务全在 pipeline。

作者：晨星
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from . import __version__
from .bm25_index import BM25Index
from .config import Settings
from .embeddings import FastEmbedder, HashEmbedder
from .evaluation import evaluate
from .llms import LlamaCppLLM, MockLLM
from .pipeline import RAGPipeline
from .schemas import (
    DocumentInfo,
    EvalRequest,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceModel,
)
from .vectorstores import FaissVectorStore, MemoryVectorStore

_WEBUI = Path(__file__).resolve().parent.parent.parent / "webui" / "index.html"


def build_embedder(settings: Settings):
    if settings.embedder == "fastembed":
        return FastEmbedder()
    return HashEmbedder(settings.embedding_dim)


def build_store(settings: Settings, dim: int):
    if settings.vectorstore == "faiss":
        return FaissVectorStore(dim)
    return MemoryVectorStore()


def build_llm(settings: Settings):
    if settings.llm == "llamacpp":
        return LlamaCppLLM(settings.llm_model_path, settings.llm_threads, settings.llm_ctx)
    return MockLLM()


def build_pipeline(settings: Settings) -> RAGPipeline:
    """管道工厂：每次调用构建全新实例（评估端点依赖此保证确定性）。"""
    embedder = build_embedder(settings)
    store = build_store(settings, embedder.dim)
    return RAGPipeline(embedder, store, BM25Index(), build_llm(settings), settings)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    pipeline = build_pipeline(settings)
    app = FastAPI(title="StarLight-RAG", version=__version__)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "chunks": pipeline.chunk_count}

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_WEBUI)

    @app.post("/api/v1/documents", response_model=IngestResponse, status_code=201)
    def ingest(req: IngestRequest) -> IngestResponse:
        try:
            result = pipeline.ingest_document(
                title=req.title, content=req.content, doc_id=req.doc_id, metadata=req.metadata
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return IngestResponse(**result)

    @app.get("/api/v1/documents", response_model=list[DocumentInfo])
    def list_docs() -> list[DocumentInfo]:
        return [DocumentInfo(**d) for d in pipeline.list_documents()]

    @app.delete("/api/v1/documents/{doc_id}", status_code=204)
    def delete_doc(doc_id: str) -> None:
        try:
            pipeline.delete_document(doc_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}") from exc

    @app.post("/api/v1/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        try:
            answer = pipeline.query(req.question, top_k=req.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return QueryResponse(
            answer=answer.text,
            sources=[SourceModel(**vars(s)) for s in answer.sources],
            latency_ms=answer.latency_ms,
        )

    @app.post("/api/v1/evaluate")
    def evaluate_endpoint(req: EvalRequest) -> dict:
        corpus = [doc.model_dump() for doc in req.corpus]
        cases = [case.model_dump() for case in req.cases]
        return evaluate(lambda: build_pipeline(settings), corpus, cases, k=req.k)

    return app
