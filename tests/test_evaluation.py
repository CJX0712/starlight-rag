"""评估模块单测。作者：晨星"""
from starlight.app import build_pipeline
from starlight.config import Settings
from starlight.evaluation import evaluate

DOC_A = "星环引擎采用氦-3 燃料循环，额定功率 42 兆瓦，比冲 9000 秒。" * 30
DOC_B = "红烧肉需要五花肉、冰糖、生抽，小火慢炖两小时。" * 30


def _corpus():
    return [
        {"doc_id": "a", "title": "引擎白皮书", "content": DOC_A},
        {"doc_id": "b", "title": "菜谱", "content": DOC_B},
    ]


def test_perfect_recall_and_mrr():
    settings = Settings()
    cases = [
        {"question": "星环引擎的燃料循环", "expect_doc_id": "a"},
        {"question": "红烧肉要炖多久", "expect_doc_id": "b"},
    ]
    metrics = evaluate(lambda: build_pipeline(settings), _corpus(), cases, k=3)
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
    assert len(metrics["details"]) == 2


def test_missing_expectation_counts_as_miss():
    settings = Settings()
    cases = [{"question": "不存在的文档内容 xyz", "expect_doc_id": "ghost"}]
    metrics = evaluate(lambda: build_pipeline(settings), _corpus(), cases, k=3)
    assert metrics["recall_at_k"] == 0.0
    assert metrics["mrr"] == 0.0


def test_each_run_uses_fresh_pipeline():
    """连续两次评估互不影响（全新管道，无状态泄漏）。"""
    settings = Settings()
    cases = [{"question": "星环引擎", "expect_doc_id": "a"}]
    first = evaluate(lambda: build_pipeline(settings), _corpus(), cases)
    second = evaluate(lambda: build_pipeline(settings), _corpus(), cases)
    assert first == second
