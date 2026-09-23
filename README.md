# StarLight-RAG

端到端可运行的企业级 RAG（检索增强生成）知识库问答系统。

作者：晨星 · 许可：MIT

## 特性

- **混合检索**：BM25 稀疏召回 + 向量稠密召回，RRF（Reciprocal Rank Fusion）融合
- **有界重排**：查询覆盖率重排信号权重封顶 50%，检索主信号始终占主导，防止弱重排器压掉正确答案
- **CPU 友好、免 GPU**：fastembed(ONNX) 嵌入 + faiss-cpu 向量库 + llama.cpp 本地 GGUF 生成，全程无 torch
- **离线可全量验证**：哈希嵌入 / 内存向量库 / Mock LLM 三个零依赖实现，`verify` 一键命令在无网络、无 API Key 下全绿
- **可插拔架构**：嵌入 / 向量库 / 词法索引 / LLM 全部定义为 Protocol，环境变量一键切换离线/生产实现
- **内置 Web UI**：单文件 HTML 对话界面（内联 CSS/JS，零外部依赖），支持文档导入、问答、引用溯源
- **质量门禁**：44 项单测 + 16 项端到端断言 + 全仓 emoji 扫描，接入 `scripts/verify.py`

## 系统架构

```
文档(txt/md/pdf) --> [loading 加载] --> [chunking 递归分块]
                                            |
                    +-----------------------+-----------------------+
                    |                                               |
            [embeddings 向量化]                             [bm25_index 词法索引]
                    |                                               |
            [vectorstores 向量库]                                   |
                    +-----------------------+-----------------------+
                                            |
查询 --> [双路召回 k*4] --> [fusion RRF 融合] --> [rerank 有界重排] --> [generation prompt 组装]
                                                                        |
                                                                  [llms 生成] --> 答案 + 引用来源
```

模块单一职责，依赖全部构造注入；`pipeline.py` 是唯一业务编排点，`app.py`/`main.py` 仅装配零业务。

| 模块 | 职责 | 离线实现 | 生产实现 |
|------|------|----------|----------|
| `loading` | 文档加载 | txt/md | pypdf 解析 pdf |
| `chunking` | 递归分块（段落边界 + 重叠） | - | - |
| `embeddings` | 文本向量化 | HashEmbedder | FastEmbedder (BAAI/bge-small-zh-v1.5) |
| `vectorstores` | 向量存取检索 | MemoryVectorStore | FaissVectorStore (IndexIDMap) |
| `bm25_index` | 稀疏检索 | rank-bm25 | 同左 |
| `fusion` | RRF 多路融合 | - | - |
| `rerank` | 有界启发式重排 | - | - |
| `llms` | 文本生成 | MockLLM | LlamaCppLLM (本地 GGUF) |
| `pipeline` | 业务编排 | - | - |
| `app` / `main` | 服务装配 | - | - |

## 快速开始

### 环境要求

- Python >= 3.10（已在 3.13 验证）

### 安装

```bash
pip install -r requirements.lock.txt
```

### 一键验证（离线，无需任何 Key 或模型文件）

```bash
python scripts/verify.py
```

依次执行：44 项单元测试 -> 16 项端到端断言（真实拉起服务进程）-> P0 emoji 门禁。

### 启动服务

```bash
# 离线模式（默认，零外部依赖）
set PYTHONPATH=src && python -m starlight.main

# 生产模式（ONNX 嵌入 + FAISS + 本地 GGUF 生成）
set PYTHONPATH=src
set STARLIGHT_EMBEDDER=fastembed
set STARLIGHT_VECTORSTORE=faiss
set STARLIGHT_LLM=llamacpp
set STARLIGHT_LLM_MODEL_PATH=models/qwen2.5-3b-instruct-q4_k_m.gguf
python -m starlight.main
```

启动后访问 `http://127.0.0.1:8000/` 使用 Web UI，`/docs` 查看交互式 API 文档。

## API 一览

| Method | Path | 功能 |
|--------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/v1/documents` | 导入文档（自动分块索引） |
| GET | `/api/v1/documents` | 文档列表 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档（三索引同步清理） |
| POST | `/api/v1/query` | 知识库问答（答案 + 引用来源） |
| POST | `/api/v1/evaluate` | 检索评估（recall@k / MRR，全新管道运行） |

完整契约见 `docs/openapi.yaml`。

### 示例

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d "{\"title\": \"引擎白皮书\", \"content\": \"星环引擎采用氦-3 燃料循环，额定功率 42 兆瓦……\"}"

curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"星环引擎的额定功率是多少？\"}"
```

## 配置项（环境变量）

| 变量 | 默认 | 说明 |
|------|------|------|
| `STARLIGHT_EMBEDDER` | `hash` | `hash`(离线) / `fastembed`(生产) |
| `STARLIGHT_VECTORSTORE` | `memory` | `memory`(离线) / `faiss`(生产) |
| `STARLIGHT_LLM` | `mock` | `mock`(离线) / `llamacpp`(本地 GGUF) |
| `STARLIGHT_LLM_MODEL_PATH` | - | GGUF 模型文件路径（llamacpp 必填） |
| `STARLIGHT_LLM_THREADS` | `4` | CPU 推理线程数，锁 2-4（瓶颈在内存带宽） |
| `STARLIGHT_CHUNK_SIZE` | `800` | 分块字符数 |
| `STARLIGHT_CHUNK_OVERLAP` | `120` | 分块重叠字符数 |
| `STARLIGHT_TOP_K` | `5` | 检索返回块数 |
| `STARLIGHT_RRF_K` | `60` | RRF 融合常数 |
| `STARLIGHT_RERANK_WEIGHT` | `0.3` | 重排权重（封顶 0.5） |
| `STARLIGHT_HOST` / `STARLIGHT_PORT` | `127.0.0.1` / `8000` | 监听地址 |

## 文档

- `ARCHITECTURE.md` —— 架构设计与模块契约
- `docs/SPEC.md` —— 规格契约（范围/API/验收标准）
- `docs/openapi.yaml` —— OpenAPI 3.0 契约
- `docs/decisions/` —— ADR 架构决策记录

## 验证标准

| 层级 | 命令 | 覆盖 |
|------|------|------|
| 单测 | `python -m pytest -q` | 44 项：分块/嵌入/向量库/BM25/融合/重排/管道/API/评估 |
| E2E | `python scripts/e2e.py` | 16 项：真实服务进程 + 成功流 + 4 条错误流 |
| 门禁 | `python tools/scan_emoji.py` | 全仓 emoji 扫描 |
| 一键 | `python scripts/verify.py` | 以上全部 |
