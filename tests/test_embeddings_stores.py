"""嵌入与向量库单测。作者：晨星"""
import math

import pytest

from starlight.embeddings import HashEmbedder
from starlight.vectorstores import FaissVectorStore, MemoryVectorStore


class TestHashEmbedder:
    def test_dim_and_normalization(self):
        emb = HashEmbedder(dim=128)
        assert emb.dim == 128
        vec = emb.embed(["星环引擎能耗测试"])[0]
        assert len(vec) == 128
        norm = math.sqrt(sum(v * v for v in vec))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_deterministic(self):
        emb = HashEmbedder()
        assert emb.embed(["确定性测试"]) == emb.embed(["确定性测试"])

    def test_similar_texts_closer_than_dissimilar(self):
        emb = HashEmbedder()
        a = emb.embed(["星环引擎的能耗参数"])[0]
        b = emb.embed(["星环引擎的能耗指标"])[0]
        c = emb.embed(["quantum computing research"])[0]

        def cos(x, y):
            return sum(p * q for p, q in zip(x, y))

        assert cos(a, b) > cos(a, c)


@pytest.fixture(params=[MemoryVectorStore, lambda: FaissVectorStore(8)])
def store(request):
    return request.param()


class TestVectorStore:
    def _seed(self, store):
        vecs = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        ids = ["c1", "c2", "c3"]
        metas = [{"doc_id": "d1"}, {"doc_id": "d1"}, {"doc_id": "d2"}]
        store.add(ids, vecs, metas)

    def test_add_and_count(self, store):
        self._seed(store)
        assert store.count() == 3

    def test_search_nearest_first(self, store):
        self._seed(store)
        results = store.search([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2)
        assert results[0][0] == "c1"
        assert results[1][0] == "c3"  # 次近邻是同向的 c3

    def test_search_empty_store(self, store):
        assert store.search([1.0] * 8, 5) == []

    def test_delete_doc(self, store):
        self._seed(store)
        store.delete_doc("d1")
        assert store.count() == 1
        results = store.search([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5)
        assert all(cid == "c3" for cid, _ in results)
