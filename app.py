"""RAG 服务：FastAPI 暴露三个能力 —— 异步入库、任务轮询、流式问答。

为什么入库要异步（面试必问）：
embedding 是慢操作，几十个块要几秒到几十秒。如果放在 HTTP 请求里同步跑，
请求会一直挂着，前端超时、网关断开。所以改成：
    提交任务 → 立刻返回 job_id → 后台跑 → 前端轮询进度
这就是"异步任务"的最小形态。

为什么不用 Celery：
单进程 demo 用 BackgroundTasks 够用。它的局限要能讲清楚：
任务在进程内存里，进程重启就丢；多副本部署时，提交和查询可能落到不同实例，
状态查不到。要上生产就得换真队列（Celery / RQ / 云队列）。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import asyncio
import json
import time
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

import kb
from config import CHAT_API_KEY, CHAT_BASE_URL, CHAT_MODEL

MIN_SIM = 0.30
TOP_K = 4
MAX_CTX = 3000

app = FastAPI(title="RAG Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # demo 阶段放开；上线要改成具体域名
    allow_methods=["*"],
    allow_headers=["*"],
)

_chat = AsyncOpenAI(api_key=CHAT_API_KEY, base_url=CHAT_BASE_URL)

# job_id -> 任务状态。内存版，进程重启即丢（这是已知限制，要主动说）
JOBS: dict[str, dict] = {}

SYSTEM = """你是知识库问答助手，只能依据给出的资料回答。
规则：
1. 资料里没有的内容，直接说「资料里没有相关内容」，不要编。
2. 回答后用 [1] [2] 标注引用的资料序号。
3. 简洁，不超过 200 字。"""


# ---------------- 请求模型 ----------------
class IngestReq(BaseModel):
    source: str = Field(..., min_length=1, description="文档名，用于引用展示")
    text: str = Field(..., min_length=1, description="文档正文")
    strategy: str = Field("heading", description="heading | fixed")


class AskReq(BaseModel):
    question: str = Field(..., min_length=1)
    strategy: str | None = None
    k: int = TOP_K


# ---------------- 基础接口 ----------------
@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}


@app.get("/api/kb/stats")
def kb_stats():
    return {"ok": True, **kb.stats()}


@app.get("/api/kb/sources")
def kb_sources():
    return {"ok": True, "items": kb.sources()}


@app.get("/")
def index():
    """前端页面直接由 FastAPI 提供。

    为什么不做成独立的 Next 项目：
    Vercel 跑不了常驻的 Python 服务，页面和接口分两个域名部署，
    线上就会出现"页面能开、接口 404"。内嵌成一个进程，演示时不会有这个问题。
    """
    page = Path(__file__).parent / "static" / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="static/index.html 不存在")
    return FileResponse(page, media_type="text/html; charset=utf-8")


# ---------------- 异步入库 ----------------
def _run_ingest(job_id: str, source: str, text: str, strategy: str) -> None:
    """后台任务：更新进度 → 入库 → 落结果。"""
    job = JOBS[job_id]
    try:
        job.update(status="running", progress=10, message="切分中")
        chunks = kb.ingest_text(source, text, strategy)
        job.update(status="done", progress=100, chunks=chunks, message="入库完成")
    except Exception as e:  # 后台任务必须自己兜住异常，否则任务静默消失
        job.update(status="failed", progress=100, message=f"{type(e).__name__}: {e}")


@app.post("/api/ingest")
def ingest(req: IngestReq, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "source": req.source,
        "strategy": req.strategy,
        "chunks": 0,
        "message": "已入队",
        "created_at": int(time.time()),
    }
    bg.add_task(_run_ingest, job_id, req.source, req.text, req.strategy)
    return {"ok": True, "job_id": job_id, "poll": f"/api/jobs/{job_id}"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在（内存版，进程重启会丢失）")
    return job


# ---------------- 流式问答 ----------------
def sse(event: str, data: dict) -> str:
    """SSE 帧：event 行 + data 行 + 空行。

    ensure_ascii=False 很重要：否则中文会被转成 \\uXXXX，前端显示乱码。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def build_ctx(hits: list[dict]) -> str:
    parts, used = [], 0
    for i, h in enumerate(hits, start=1):
        block = f"[{i}] 来源：{h['meta']['source']}\n{h['text']}"
        if used + len(block) > MAX_CTX:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


@app.post("/api/ask")
async def ask(req: AskReq):
    hits = await asyncio.to_thread(kb.retrieve, req.question, req.strategy, req.k)
    best = max((h["sim"] for h in hits), default=0.0)

    async def gen():
        # 先把引用发出去：前端可以立刻渲染引用卡片，不用等生成完。
        # 这是流式 RAG 的一个体验细节，面试可以讲。
        citations = [
            {
                "index": i,
                "source": h["meta"]["source"],
                "chunk": h["meta"]["index"],
                "sim": h["sim"],
                "preview": h["text"][:80],
            }
            for i, h in enumerate(hits, start=1)
        ]
        yield sse("citations", {"citations": citations, "best": best})

        if best < MIN_SIM:
            yield sse("delta", {"text": f"资料里没有相关内容（最高相似度 {best:.2f}，低于阈值 {MIN_SIM}）。"})
            yield sse("done", {"reason": "low_similarity"})
            return

        ctx = build_ctx(hits)
        try:
            stream = await _chat.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"资料：\n{ctx}\n\n问题：{req.question}"},
                ],
                temperature=0.2,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield sse("delta", {"text": delta})
            yield sse("done", {"reason": "finish"})
        except Exception as e:
            # 流已经开始，不能再改状态码，只能用事件告诉前端出错了
            yield sse("error", {"message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
