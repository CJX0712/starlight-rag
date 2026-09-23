"""BM25 / RRF 融合 / 有界重排单测。作者：晨星"""
from starlight.bm25_index import BM25Index
from starlight.fusion import rrf_fuse
from starlight.rerank import heuristic_rerank


class TestBM25Index:
    def test_search_relevant_first(self):
        idx = BM25Index()
        idx.add(["c1", "c2"], ["星环引擎采用聚变供能", "今天天气晴朗适合出行"], "d1")
        results = idx.search("星环引擎供能方式", 2)
        assert results[0][0] == "c1"

    def test_empty_index(self):
        assert BM25Index().search("任意查询", 3) == []

    def test_delete_doc_rebuilds(self):
        idx = BM25Index()
        idx.add(["c1"], ["星环引擎采用聚变供能"], "d1")
        idx.add(["c2"], ["星环引擎的散热设计"], "d2")
        idx.delete_doc("d1")
        results = idx.search("星环引擎", 5)
        assert all(cid == "c2" for cid, _ in results)


class TestRRFFuse:
    def test_fusion_prefers_consensus(self):
        dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        sparse = [("b", 5.0), ("a", 4.0), ("d", 3.0)]
        fused = rrf_fuse([dense, sparse])
        # a、b 都在两路前列，应排最前；d 只在一路，排最后
        assert {cid for cid, _ in fused[:2]} == {"a", "b"}
        assert fused[-1][0] in {"c", "d"}

    def test_empty_rankings(self):
        assert rrf_fuse([[], []]) == []


class TestHeuristicRerank:
    def test_empty_candidates(self):
        assert heuristic_rerank("查询", []) == []

    def test_weight_bounded(self):
        candidates = [("c1", 1.0, "星环引擎"), ("c2", 0.5, "完全无关内容")]
        # weight 超过上限应被钳制到 0.5，不抛错
        ranked = heuristic_rerank("星环", candidates, weight=0.9)
        assert ranked[0][0] == "c1"

    def test_rerank_cannot_override_strong_retrieval_signal(self):
        """重排权重有界：检索分显著领先但覆盖略低的块，不应被微弱覆盖优势翻盘。"""
        candidates = [
            ("strong", 1.0, "星环引擎的能耗参数与散热设计"),
            ("weak", 0.1, "星环"),
        ]
        ranked = heuristic_rerank("星环", candidates, weight=0.3)
        assert ranked[0][0] == "strong"
