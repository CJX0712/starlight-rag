# ADR-001: 采用免 torch 的 CPU 友好 AI 栈

## Status: Accepted (2026-09-24)

## Background

目标环境为无 GPU 的通用服务器/PC（含 Windows）。主流 RAG 栈默认依赖
torch + sentence-transformers + 云端 LLM API：安装体积大（torch 超 2GB）、
Windows 轮子不全、私有数据需出域。

## Decision

选定全 CPU 栈，全部组件有 win_amd64 轮子、免 torch：

- 嵌入：fastembed 0.8.0（ONNX Runtime，BAAI/bge-small-zh-v1.5 中文优化）
- 向量库：faiss-cpu 1.15.0（IndexFlatIP + IndexIDMap）
- 稀疏检索：rank-bm25 0.2.2（纯 Python）
- 生成：llama-cpp-python 0.3.19（本地 GGUF，线程锁 4）
- PDF：pypdf 6.18.1（纯 Python）

## Consequences

- 正面：安装体积小、Windows/Linux 均可装、数据不出域、离线可验证
- 负面：嵌入/生成质量弱于 GPU 大模型方案；faiss-cpu 无 GPU 加速
- 缓解：Protocol 抽象层允许后续无缝替换为更强实现

## Related ADRs

ADR-002（离线可验证架构）、ADR-003（中文分词选型）
