"""向量化实现：离线哈希嵌入（默认）+ fastembed ONNX 生产嵌入。

作者：晨星
"""
from __future__ import annotations

import hashlib
import math

from .textutil import tokenize


class HashEmbedder:
    """确定性哈希嵌入：token 经 blake2b 哈希映射到固定维度计数，再 L2 归一化。

    零依赖、零网络、完全离线，供单测/E2E/无 Key 环境使用；
    语义能力有限，生产环境切换 FastEmbedder。
    """

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正数")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            for tok in tokenize(text):
                digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                vec[int.from_bytes(digest, "big") % self._dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class FastEmbedder:
    """fastembed(ONNX) 生产实现，惰性加载模型，默认中文优化模型。"""

    def __init__(self, model: str = "BAAI/bge-small-zh-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)
        probe = list(self._model.embed(["探"]))[0]
        self._dim = int(probe.shape[0])

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._model.embed(list(texts))]
