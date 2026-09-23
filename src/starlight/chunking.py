"""递归字符分块器：按自然段落边界切分，超长段落硬切并保留重叠。

作者：晨星
"""
from __future__ import annotations

from .protocols import Chunk


def chunk_text(doc_id: str, text: str, size: int = 800, overlap: int = 120) -> list[Chunk]:
    """把文档切成不超过 size 字符的分块，相邻块间保留 overlap 字符上下文。

    策略：
    1. 按换行拆成段落；
    2. 段落贪心合并进缓冲区，超过 size 即落块；
    3. 单段超过 size 时硬切，硬切点保留 overlap 重叠；
    4. 落块时向后一块预置前一块尾部 overlap 字符，保证跨块语义连续。
    """
    text = text.strip()
    if not text:
        return []
    if size <= 0:
        raise ValueError("size 必须为正数")
    overlap = max(0, min(overlap, size // 2))

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    raw: list[str] = []
    buf = ""
    for para in paragraphs:
        while len(para) > size:
            if buf:
                raw.append(buf)
                buf = ""
            raw.append(para[:size])
            para = para[size - overlap:]
        if buf and len(buf) + len(para) + 1 > size:
            raw.append(buf)
            buf = para
        else:
            buf = f"{buf}\n{para}".strip() if buf else para
    if buf:
        raw.append(buf)

    if overlap > 0 and len(raw) > 1:
        with_overlap = [raw[0]]
        for chunk in raw[1:]:
            tail = with_overlap[-1][-overlap:]
            with_overlap.append((tail + chunk)[: size + overlap])
        raw = with_overlap

    return [Chunk(id=f"{doc_id}#c{i}", doc_id=doc_id, text=c) for i, c in enumerate(raw)]
