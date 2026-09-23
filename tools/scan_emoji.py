"""P0 门禁：全仓 emoji 扫描，禁止 emoji 作为功能图标/装饰进入交付物。

用法: python tools/scan_emoji.py
发现 emoji 即列出 文件:行号 并以退出码 1 失败。作者：晨星
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 全部使用 unicode 转义书写，避免本文件自身含 emoji 字面量导致自检出
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF\U00002600-\U000026FF\U00002700-\U000027BF"
    "\U0000FE00-\U0000FE0F\U0001F000-\U0001F02F\U0001F0A0-\U0001F0FF"
    "\U0001F100-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U0000200D\U000020E3]"
)
SCAN_SUFFIXES = {".py", ".md", ".html", ".css", ".js", ".yaml", ".yml", ".json", ".txt", ".toml"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv", "dist", "build"}


def scan() -> list[str]:
    hits: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SCAN_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if EMOJI_RE.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{lineno}")
    return hits


def main() -> int:
    hits = scan()
    if hits:
        print(f"P0 门禁失败：发现 {len(hits)} 处 emoji：")
        for h in hits:
            print(f"  {h}")
        return 1
    print("P0 门禁通过：全仓无 emoji。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
