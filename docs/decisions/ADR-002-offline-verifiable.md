# ADR-002: 离线可全量验证作为硬性架构约束

## Status: Accepted (2026-09-24)

## Background

多数 AI Demo 的「能跑」依赖外部条件：API Key、模型下载、网络、数据库。
一旦条件缺失，测试全红，无法区分「代码错了」还是「环境缺了」。
单元测试全绿但系统从未真实启动过，是最高发的交付事故。

## Decision

1. 每个外部依赖定义为 Protocol（Embedder/VectorStore/LexicalIndex/LLM），
   各配一个零依赖离线实现：HashEmbedder / MemoryVectorStore / BM25(本就离线) / MockLLM
2. 默认配置（无环境变量）即离线模式，`scripts/verify.py` 在无网络、无 Key、
   无模型文件下必须全绿
3. E2E 必须真实拉起服务进程（非 TestClient），轮询 /health 后跑成功流 + 全部错误流
4. 评估永远在全新构建的管道上运行（工厂注入），杜绝单例状态污染指标

## Consequences

- 正面：任何干净环境一条命令证明系统真实可用；CI 无需任何密钥
- 负面：需维护双实现；MockLLM 的「答案」无真实生成质量
- 缓解：生产实现通过环境变量切换，同一代码路径被两种实现共享

## Related ADRs

ADR-001（CPU 栈）
