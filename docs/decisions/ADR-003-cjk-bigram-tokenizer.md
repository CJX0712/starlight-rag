# ADR-003: 中文分词采用字符 bigram 而非 jieba

## Status: Accepted (2026-09-24)

## Background

BM25 与哈希嵌入都需要分词。候选方案：jieba（词典分词）、
字符 unigram+bigram。

## Decision

采用字符 unigram + bigram（`textutil.tokenize`），英文按小写词。
BM25 与哈希嵌入共用同一分词函数，保证稀疏/稠密两路词表一致。

理由：

1. 检索场景下 bigram 召回等价甚至优于词典分词（无 OOV 问题）
2. 零依赖、纯确定性，符合离线可验证约束（ADR-002）
3. 两路召回共用词表避免「同词不同 token」导致的融合损耗

## Consequences

- 正面：零依赖、确定性、无 OOV、两路词表一致
- 负面：索引体积略大于词典分词；无词性/实体信息
- 缓解：10k 分块规模内体积差异可忽略

## Related ADRs

ADR-002（离线可验证）
