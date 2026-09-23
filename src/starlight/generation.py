"""RAG 生成：prompt 组装 + 引用来源结构。

作者：晨星
"""
from __future__ import annotations

from dataclasses import dataclass, field

PROMPT_TEMPLATE = (
    "你是一个严谨的知识库问答助手。仅依据下列编号上下文回答问题，"
    "回答时用 [编号] 标注引用来源；上下文不足时明确说明不知道，禁止编造。\n\n"
    "{contexts}\n\n问题：{question}\n回答："
)


@dataclass
class Source:
    chunk_id: str
    doc_id: str
    title: str
    snippet: str
    score: float


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    latency_ms: float = 0.0


def build_prompt(question: str, chunk_texts: list[str]) -> str:
    contexts = "\n".join(f"[{i + 1}] {text}" for i, text in enumerate(chunk_texts))
    if not contexts:
        contexts = "（无相关上下文）"
    return PROMPT_TEMPLATE.format(contexts=contexts, question=question)
