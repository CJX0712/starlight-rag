"""分块器单测。作者：晨星"""
from starlight.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("d1", "") == []
    assert chunk_text("d1", "   \n  ") == []


def test_short_text_single_chunk():
    chunks = chunk_text("d1", "这是一段很短的文本。")
    assert len(chunks) == 1
    assert chunks[0].id == "d1#c0"
    assert chunks[0].doc_id == "d1"


def test_long_text_splits_within_size_limit():
    text = "星环引擎" * 400  # 1600 字符，超过 800 阈值，必须分多块
    chunks = chunk_text("d1", text, size=800, overlap=120)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 800 + 120


def test_paragraph_boundaries_respected():
    text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
    chunks = chunk_text("d1", text, size=800, overlap=120)
    assert len(chunks) == 1
    assert "第一段" in chunks[0].text and "第三段" in chunks[0].text


def test_overlap_preserves_context_across_chunks():
    text = "甲乙丙丁" * 300  # 1200 字符
    chunks = chunk_text("d1", text, size=500, overlap=50)
    assert len(chunks) >= 2
    # 后一块开头包含前一块尾部内容（重叠区）
    assert chunks[1].text[:10] == chunks[0].text[-50:-40]
