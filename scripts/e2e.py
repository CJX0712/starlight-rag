"""自包含 E2E：真实拉起服务进程，验证核心成功流 + 全部错误流。

用法: python scripts/e2e.py
退出码 0 = 全部通过。作者：晨星
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
PORT = 18123
BASE = f"http://127.0.0.1:{PORT}"

# 显式超过默认 800 字符分块阈值，保证 E2E 覆盖多块检索路径
LONG_DOC = (
    "星环引擎是第三代聚变推进装置，额定输出功率 42 兆瓦，采用氦-3 燃料循环，"
    "散热系统使用液态金属回路，满载工况下比冲达到 9000 秒，可为深空探测器提供持续推力。"
    "引擎控制单元内置三重冗余飞控计算机，支持在轨自主故障诊断与重构。\n"
) * 10

passed = 0
failed = 0
failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        failures.append(f"{name}: {detail}")
        print(f"  [FAIL] {name} {detail}")


def wait_ready(proc: subprocess.Popen) -> bool:
    for _ in range(60):
        try:
            if httpx.get(f"{BASE}/health", timeout=2).status_code == 200:
                return True
        except httpx.TransportError:
            pass
        if proc.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def main() -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["STARLIGHT_HOST"] = "127.0.0.1"
    env["STARLIGHT_PORT"] = str(PORT)
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        [sys.executable, "-m", "starlight.main"],
        cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
    )
    try:
        print("1. 启动服务进程…")
        if not wait_ready(proc):
            out = proc.stdout.read() if proc.stdout else ""
            check("服务启动", False, out[-2000:])
            return 1
        check("服务启动 /health 200", True)
        client = httpx.Client(base_url=BASE, timeout=15)

        print("2. 成功流：导入 -> 查询 -> 列表 -> 删除")
        r = client.post("/api/v1/documents", json={"title": "星环引擎白皮书", "content": LONG_DOC, "doc_id": "e2e-engine"})
        check("导入返回 201", r.status_code == 201, r.text)
        check("文档显式超过阈值 -> 多分块", r.json().get("chunks", 0) > 1, r.text)

        r = client.post("/api/v1/query", json={"question": "星环引擎的额定功率与燃料循环"})
        body = r.json()
        check("查询返回 200", r.status_code == 200, r.text)
        check("答案非空（对外语义）", bool(body.get("answer")), str(body)[:300])
        check("引用来源命中目标文档", any(s["doc_id"] == "e2e-engine" for s in body.get("sources", [])), str(body)[:300])

        r = client.get("/api/v1/documents")
        check("文档列表包含已导入文档", any(d["doc_id"] == "e2e-engine" for d in r.json()), r.text)

        r = client.delete("/api/v1/documents/e2e-engine")
        check("删除返回 204", r.status_code == 204, r.text)
        r = client.post("/api/v1/query", json={"question": "星环引擎"})
        check("删除后查询无来源", r.json().get("sources") == [], r.text[:300])

        print("3. 评估端点（全新管道，不污染主服务）")
        r = client.post("/api/v1/evaluate", json={
            "corpus": [{"doc_id": "ev-a", "title": "白皮书", "content": LONG_DOC}],
            "cases": [{"question": "星环引擎的燃料", "expect_doc_id": "ev-a"}],
            "k": 3,
        })
        check("评估返回 200", r.status_code == 200, r.text)
        check("recall@3 == 1.0", r.json().get("recall_at_k") == 1.0, r.text)
        r = client.get("/health")
        check("评估后主服务 chunks 仍为 0", r.json().get("chunks") == 0, r.text)

        print("4. 错误流")
        r = client.post("/api/v1/query", json={"question": ""})
        check("空问题 -> 422", r.status_code == 422, r.text)
        r = client.post("/api/v1/documents", json={"title": "空", "content": ""})
        check("空内容导入 -> 422", r.status_code == 422, r.text)
        r = client.delete("/api/v1/documents/ghost-doc")
        check("删除不存在文档 -> 404", r.status_code == 404, r.text)
        r = client.post("/api/v1/query", json={"question": "功率", "top_k": 99})
        check("top_k 越界 -> 422", r.status_code == 422, r.text)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/pid", str(proc.pid), "/t", "/f"], capture_output=True)

    print(f"\n结果: 通过 {passed} / 失败 {failed}")
    for f in failures:
        print(f"  失败项: {f}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
