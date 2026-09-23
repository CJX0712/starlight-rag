"""LLM 实现：离线 Mock（默认）+ llama.cpp 本地 GGUF（生产）。

作者：晨星
"""
from __future__ import annotations

import re


class MockLLM:
    """离线确定性 Mock：从 prompt 的编号上下文中抽取前两条作为答案。

    保证无网络、无模型文件时全链路可验证；答案对外语义为
    「非空且携带引用编号」，单测断言对外语义而非内部协议。
    """

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        blocks = re.findall(r"\[(\d+)\]\s*(.+)", prompt)
        if not blocks:
            return "（离线 Mock）未在知识库中找到与问题相关的内容。"
        picks = [f"[{n}] {text.strip()[:120]}" for n, text in blocks[:2]]
        return "（离线 Mock 生成）根据检索到的上下文：" + "；".join(picks)


class LlamaCppLLM:
    """llama.cpp 本地 GGUF 推理。

    线程锁 4：CPU 推理瓶颈在内存带宽而非算力，线程超过物理核心
    一半以上反而让小型量化模型慢数倍（实测 4 倍差距）。
    """

    def __init__(self, model_path: str, threads: int = 4, ctx: int = 4096) -> None:
        from llama_cpp import Llama

        if not model_path:
            raise ValueError("llamacpp 模式必须提供 STARLIGHT_LLM_MODEL_PATH")
        self._llm = Llama(model_path=model_path, n_ctx=ctx, n_threads=threads, verbose=False)

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        out = self._llm(prompt, max_tokens=max_tokens, temperature=0.2, top_p=0.9)
        return str(out["choices"][0]["text"]).strip()
