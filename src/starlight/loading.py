"""文档加载：txt / md / pdf 统一入口。

作者：晨星
"""
from __future__ import annotations

from pathlib import Path


def load_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_pdf(path: str | Path) -> str:
    """pypdf 逐页抽取文本，页间用换行拼接。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_document(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".txt", ".md", ".markdown"}:
        return load_text_file(path)
    raise ValueError(f"不支持的文档类型: {suffix}（支持 .txt/.md/.pdf）")
