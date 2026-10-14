# -*- coding: utf-8 -*-
"""企业知识库问答 Web 服务（FastAPI）。

启动:
  python app.py                 # 默认 http://127.0.0.1:8000
  python app.py --port 8765

接口:
  GET  /                聊天页面
  GET  /api/info        知识库信息（文档列表 / 索引 / 模型）
  POST /api/chat        提问，SSE 流式返回（sources / token / done / error）
  POST /api/clear       清空指定会话历史
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_engine import Config, RAGEngine, logger

STATIC_DIR = Path(__file__).resolve().parent / "static"

# 全局引擎：启动时初始化（加载/构建向量库），失败时置空并由接口返回 503
engine: Optional[RAGEngine] = None


def sse(data: dict) -> str:
    """SSE 数据帧：统一 data: {json}\n\n 格式。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    try:
        engine = RAGEngine(Config())
        logger.info("RAG 引擎初始化完成")
    except Exception as e:  # noqa: BLE001 - 启动失败不影响服务起得来，接口会返回 503
        logger.error("RAG 引擎初始化失败: %s", e)
        engine = None
    yield


app = FastAPI(title="企业知识库问答", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class ClearRequest(BaseModel):
    session_id: str


def require_engine() -> RAGEngine:
    if engine is None:
        raise HTTPException(503, "RAG 引擎未初始化，请检查 .env 中的 DASHSCOPE_API_KEY 后重启服务")
    return engine


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/info")
def info():
    eng = require_engine()
    return eng.kb_info()


@app.post("/api/chat")
async def chat(req: ChatRequest):
    eng = require_engine()
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "问题不能为空")
    session_id = req.session_id or f"web_{uuid.uuid4().hex[:8]}"

    def gen():
        try:
            for kind, data in eng.stream_answer(q, session_id):
                if kind == "sources":
                    yield sse({"type": "sources", "data": data})
                elif kind == "token":
                    yield sse({"type": "token", "delta": data})
                elif kind == "done":
                    yield sse({"type": "done", "answer": data["answer"], "session_id": session_id})
        except Exception as e:  # noqa: BLE001 - 流式过程中异常也要通知前端
            logger.error("回答生成失败: %s", e)
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/clear")
def clear(req: ClearRequest):
    eng = require_engine()
    eng.clear_session(req.session_id)
    return {"ok": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动企业知识库问答 Web 服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    args = parser.parse_args()
    uvicorn.run("app:app", host=args.host, port=args.port)
