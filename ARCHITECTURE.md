# StarLight-RAG 架构设计

作者：晨星

## 设计原则

1. **单一职责**：每个模块只做一件事，单文件不超过 300 行
2. **依赖倒置**：所有外部能力（嵌入/向量库/词法索引/LLM）定义为 `protocols.py` 中的 Protocol，业务层只依赖抽象
3. **装配零业务**：`app.py` / `main.py` 只做依赖组装与路由转发，业务规则全部集中在 `pipeline.py`
4. **离线可验证**：每个 Protocol 都有零依赖离线实现，任何环境下 `verify` 可证明系统真实可用
5. **评估确定性**：评估永远在全新构建的管道上运行，杜绝运行期数据污染指标

## 模块契约

### protocols.py —— 核心抽象

```python
class Embedder(Protocol):
    dim: int
    def embed(texts: list[str]) -> list[list[float]]  # 返回 L2 归一化向量

class VectorStore(Protocol):
    def add(ids, vectors, metadatas)
    def search(vector, k) -> list[(chunk_id, score)]  # 按相关度降序
    def delete_doc(doc_id)
    def count() -> int

class LexicalIndex(Protocol):
    def add(ids, texts, doc_id)
    def search(query, k) -> list[(chunk_id, score)]
    def delete_doc(doc_id)

class LLM(Protocol):
    def generate(prompt, max_tokens=512) -> str
```

任何满足 Protocol 的实现都可注入，无需改业务代码。

### 调用关系（依赖只向下）

```
main -> app(装配) -> pipeline(编排) -> chunking / fusion / rerank / generation
                                  -> Protocol 注入: embeddings / vectorstores / bm25_index / llms
evaluation -> pipeline 工厂（每次全新实例）
```

- `textutil.tokenize` 被 `embeddings` 与 `bm25_index` 共用，保证稀疏/稠密两路词表一致
- 模块间无环向依赖；`pipeline` 不 import 任何具体实现（由 `app` 注入）

## 检索链路

```
query
  |-- embed(query) --> vectorstore.search(k*4)   --+
  |-- bm25.search(query, k*4) ---------------------+--> rrf_fuse(k=60) --> top k*2
                                                              |
                                              heuristic_rerank(weight<=0.5) --> top k
                                                              |
                                              build_prompt(编号上下文) --> llm.generate
                                                              |
                                              Answer(text, sources[{chunk_id, doc_id, title, snippet, score}], latency_ms)
```

关键工程决策：

- **RRF 只用名次不用原始分**：天然免疫稠密/稀疏两路分数量纲不一致
- **重排权重封顶 0.5**：弱重排信号不能独断最终顺序（见 ADR-004）
- **分块重叠双向保留**：硬切点与块间边界都保留 overlap，跨块语义不断裂
- **FAISS 用 IndexFlatIP + 归一化**：内积等价余弦；IndexIDMap 支持按块删除

## 部署拓扑

单机单进程即可运行（uvicorn 内嵌）。生产建议：

- 前置反向代理（Nginx）做 TLS 与限流
- `STARLIGHT_EMBEDDER=fastembed` + `STARLIGHT_VECTORSTORE=faiss` + `STARLIGHT_LLM=llamacpp`
- GGUF 模型建议 Q4_K_M 量化，线程锁 4（CPU 推理瓶颈在内存带宽，线程过多反而拖慢数倍）
- 数据默认进程内存态，重启即清空；持久化属 v2 范围（见 SPEC「明确不做」）

## 可复现性

- `requirements.lock.txt`：全量锁定版本（含传递依赖），由已验证环境 freeze 生成并剔除 Windows-only 包
- `pyproject.toml`：构建配置 + pytest 配置
- 干净环境复现：`pip install -r requirements.lock.txt` -> `python scripts/verify.py` 全绿即证明一致
