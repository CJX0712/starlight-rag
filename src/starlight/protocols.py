"""核心抽象：Protocol 接口 + 数据模型。

所有外部依赖（嵌入/向量库/LLM）都定义为 Protocol，运行时注入实现，
使每个模块可用 fake 实现独立单测。

作者：晨星
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Chunk:
    """文档分块，检索的最小单元。"""

    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Embedder(Protocol):
    """文本向量化。"""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    """向量存取与近邻检索。"""

    def add(self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None: ...

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]: ...

    def delete_doc(self, doc_id: str) -> None: ...

    def count(self) -> int: ...


class LexicalIndex(Protocol):
    """稀疏检索（BM25）。"""

    def add(self, ids: list[str], texts: list[str], doc_id: str) -> None: ...

    def search(self, query: str, k: int) -> list[tuple[str, float]]: ...

    def delete_doc(self, doc_id: str) -> None: ...


class LLM(Protocol):
    """文本生成。"""

    def generate(self, prompt: str, max_tokens: int = 512) -> str: ...
