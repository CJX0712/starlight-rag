"""API 请求/响应模型（pydantic v2）。

作者：晨星
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    doc_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    title: str
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceModel(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceModel]
    latency_ms: float


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    chunks: int
    metadata: dict[str, Any]


class CorpusDoc(BaseModel):
    doc_id: str | None = None
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)


class EvalCase(BaseModel):
    question: str = Field(min_length=1)
    expect_doc_id: str = Field(min_length=1)


class EvalRequest(BaseModel):
    corpus: list[CorpusDoc]
    cases: list[EvalCase]
    k: int = Field(default=5, ge=1, le=20)


class ErrorResponse(BaseModel):
    detail: str
