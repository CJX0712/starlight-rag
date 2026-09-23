"""一键验证：单测 -> E2E -> P0 emoji 门禁。任一失败即非 0 退出。

用法: python scripts/verify.py
作者：晨星
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(name: str, cmd: list[str]) -> bool:
    print(f"\n===== {name} =====")
    result = subprocess.run(cmd, cwd=ROOT)
    print(f"----- {name}: {'通过' if result.returncode == 0 else '失败'} -----")
    return result.returncode == 0


def main() -> int:
    ok = True
    ok &= run("单元测试", [sys.executable, "-m", "pytest", "-q"])
    ok &= run("端到端验证", [sys.executable, str(ROOT / "scripts" / "e2e.py")])
    ok &= run("P0 emoji 门禁", [sys.executable, str(ROOT / "tools" / "scan_emoji.py")])
    print("\n===== 总结:", "全部通过" if ok else "存在失败项", "=====")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
