"""全局配置：环境变量驱动，默认值保证离线可运行。

作者：晨星
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """系统配置。所有外部依赖均可通过环境变量切换生产/离线实现。"""

    embedder: str = "hash"        # hash(离线) | fastembed(生产 ONNX)
    vectorstore: str = "memory"   # memory(离线) | faiss(生产)
    llm: str = "mock"             # mock(离线) | llamacpp(本地 GGUF)
    embedding_dim: int = 384
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    rrf_k: int = 60
    rerank_weight: float = 0.3    # 重排权重有界，防止弱重排器独断排序
    llm_model_path: str = ""
    llm_threads: int = 4          # CPU 推理线程锁 2-4，瓶颈在内存带宽
    llm_ctx: int = 4096
    host: str = "127.0.0.1"
    port: int = 8000

    @staticmethod
    def from_env() -> "Settings":
        e = os.environ
        return Settings(
            embedder=e.get("STARLIGHT_EMBEDDER", "hash"),
            vectorstore=e.get("STARLIGHT_VECTORSTORE", "memory"),
            llm=e.get("STARLIGHT_LLM", "mock"),
            embedding_dim=int(e.get("STARLIGHT_EMBEDDING_DIM", "384")),
            chunk_size=int(e.get("STARLIGHT_CHUNK_SIZE", "800")),
            chunk_overlap=int(e.get("STARLIGHT_CHUNK_OVERLAP", "120")),
            top_k=int(e.get("STARLIGHT_TOP_K", "5")),
            rrf_k=int(e.get("STARLIGHT_RRF_K", "60")),
            rerank_weight=float(e.get("STARLIGHT_RERANK_WEIGHT", "0.3")),
            llm_model_path=e.get("STARLIGHT_LLM_MODEL_PATH", ""),
            llm_threads=int(e.get("STARLIGHT_LLM_THREADS", "4")),
            llm_ctx=int(e.get("STARLIGHT_LLM_CTX", "4096")),
            host=e.get("STARLIGHT_HOST", "127.0.0.1"),
            port=int(e.get("STARLIGHT_PORT", "8000")),
        )
