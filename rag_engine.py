# -*- coding: utf-8 -*-
"""企业知识库 RAG 引擎（核心逻辑，供 CLI 与 Web 服务共用）。

- 文档解析 / 切分 / 向量库构建与自愈（维度校验、自动重建）
- 检索 + LLM 问答（流式 / 非流式）
- 多会话历史管理（内存，自动裁剪最近 N 轮）
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Optional

import dashscope
import pdfplumber
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FAISS: 官方未发布独立包，实际使用 langchain-community 的 FAISS；
# try 仅为未来官方拆包后自动适配，当前始终走 except 分支
try:
    from langchain_faiss import FAISS  # type: ignore
except ImportError:
    from langchain_community.vectorstores import FAISS

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ===================== 配置 =====================
@dataclass
class Config:
    """全局配置，可用命令行参数覆盖。"""

    docs_dir: Path = Path("./docs")
    index_dir: Path = Path("./faiss_index")

    chunk_size: int = 350
    chunk_overlap: int = 70

    retrieve_top_k: int = 5
    max_history_turns: int = 10  # 会话保留的最大轮数，防止上下文无限膨胀

    # 注意：原代码的 qwen3.8-max 不是有效模型名。已实测当前账号可用的模型为
    # qwen-max / qwen-plus / qwen-turbo / qwen-long（qwen3 系列均返回 404），
    # 默认使用其中效果最好的 qwen-max，可在此按需更换。
    llm_model: str = "qwen-max"
    # text-embedding-v1 为旧版模型，推荐 v3/v4（效果更好）；账号不支持时可改回 v1。
    embedding_model: str = "text-embedding-v3"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    temperature: float = 0.1
    llm_timeout: int = 120

    session_id: str = "session_001"

    def apply_args(self, args) -> "Config":
        """用命令行参数覆盖默认配置。"""
        if args.docs:
            self.docs_dir = args.docs
        if args.index:
            self.index_dir = args.index
        if args.top_k and args.top_k > 0:
            self.retrieve_top_k = args.top_k
        if args.session:
            self.session_id = args.session
        return self


SYSTEM_PROMPT = """你是「企业知识库助手」。回答规则：
1. 只能依据【参考内容】回答，严禁编造或凭常识脑补；
2. 回答须引用原文依据，并在引用后标注来源文件名；
3. 参考内容不足以回答时，直接回复【知识库暂无相关信息】，不要强行作答；
4. 数字、日期、专有名词必须与原文一致；
5. 用与用户相同的语言回答，简洁专业。"""


# ===================== 会话历史管理 =====================
class SessionHistoryManager:
    """会话历史管理器：隔离多会话，并自动裁剪历史长度。

    历史用 HumanMessage / AIMessage 列表保存，直接作为 MessagesPlaceholder
    的输入，不再依赖已弃用的 InMemoryChatMessageHistory。
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._store: Dict[str, List] = {}

    def get(self, session_id: str) -> List:
        history = self._store.get(session_id)
        if history is None:
            history = []
            self._store[session_id] = history
        return history

    def add(self, session_id: str, user_text: str, ai_text: str) -> None:
        history = self.get(session_id)
        history.append(HumanMessage(content=user_text))
        history.append(AIMessage(content=ai_text))
        # 只保留最近 N 轮（每轮 = 1 条用户消息 + 1 条回答），控制 token 成本与延迟
        while len(history) > self.max_turns * 2:
            history.pop(0)

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)
        logger.info("已清空会话 %s 的历史", session_id)


# ===================== 模块1：PDF 文档解析 =====================
def extract_text_from_pdf(pdf_path: str) -> str:
    """逐页提取 PDF 文本，跳过空白页。"""
    parts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                parts.append(page_text)
    return "\n".join(parts)


def load_all_pdf_from_folder(folder_path: Path) -> List[Document]:
    """批量读取文件夹内全部 PDF，返回 Document 列表。"""
    folder_path.mkdir(exist_ok=True)
    pdf_files = sorted(folder_path.glob("*.pdf"))  # 排序，保证构建结果稳定可复现
    if not pdf_files:
        logger.warning("目录 %s 下没有找到 PDF 文件", folder_path)
        return []

    raw_docs: List[Document] = []
    for pdf_file in pdf_files:
        logger.info("正在加载文件: %s", pdf_file.name)
        try:
            content = extract_text_from_pdf(str(pdf_file))
            if not content.strip():
                logger.warning("%s 提取文本为空，跳过", pdf_file.name)
                continue
            raw_docs.append(
                Document(
                    page_content=content,
                    metadata={"source": pdf_file.name, "file_path": str(pdf_file)},
                )
            )
        except Exception as e:  # noqa: BLE001 - 单文件失败不影响整体
            logger.error("解析文件 %s 失败: %s", pdf_file.name, e)
    logger.info("成功加载 %d 个 PDF 文件", len(raw_docs))
    return raw_docs


# ===================== 向量化 =====================
class DashScopeEmbedding(Embeddings):
    """DashScope 文本向量化。

    - 修复原生 SDK 要求 input 必须为数组的问题；
    - 增加响应状态码校验与失败重试（指数退避）。
    """

    def __init__(self, model: str, api_key: str, max_retries: int = 3):
        dashscope.api_key = api_key
        self.model = model
        self.max_retries = max_retries

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, text_type="document")

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], text_type="query")[0]

    def _embed(self, texts: List[str], text_type: str) -> List[List[float]]:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = dashscope.TextEmbedding.call(
                    model=self.model, input=texts, text_type=text_type
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Embedding API 返回 {resp.status_code}: "
                        f"{getattr(resp, 'code', '')} {getattr(resp, 'message', '')}"
                    )
                return [item["embedding"] for item in resp.output["embeddings"]]
            except Exception as e:  # noqa: BLE001 - 统一重试兜底
                last_err = e
                logger.warning("Embedding 调用失败(第%d/%d次): %s", attempt, self.max_retries, e)
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Embedding 调用连续失败: {last_err}") from last_err


# ===================== 模块2：RAG 知识库构建与检索 =====================
def split_docs(raw_docs: List[Document], cfg: Config) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", "。", "，"],
    )
    chunks = splitter.split_documents(raw_docs)
    logger.info("文档切分完成，分片数量: %d", len(chunks))
    return chunks


def build_index(cfg: Config, embeddings: Embeddings) -> FAISS:
    """读取 PDF 并构建向量库。"""
    raw_docs = load_all_pdf_from_folder(cfg.docs_dir)
    if not raw_docs:
        raise RuntimeError(f"目录 {cfg.docs_dir} 没有可用 PDF，无法构建向量库")
    chunks = split_docs(raw_docs, cfg)
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(str(cfg.index_dir))
    logger.info("FAISS 向量库已保存至 %s", cfg.index_dir)
    return vector_db


def load_index(cfg: Config, embeddings: Embeddings) -> FAISS:
    # allow_dangerous_deserialization 仅在索引来自可信环境时可开启
    # （FAISS 索引本质是 pickle，加载不可信来源的索引有反序列化风险）
    vector_db = FAISS.load_local(
        str(cfg.index_dir), embeddings, allow_dangerous_deserialization=True
    )
    # 维度校验：更换 embedding 模型（如 v1→v3）后，索引维度与查询维度不一致，
    # FAISS 不会在加载时报错，而是首次查询时报 "assert d == self.d"。
    # 这里提前探测一次，发现不一致就抛错，由 load_or_build_index 自动触发重建。
    probe_dim = len(embeddings.embed_query("dimension-probe"))
    index_dim = vector_db.index.d
    if probe_dim != index_dim:
        raise ValueError(
            f"Embedding 维度不匹配：当前模型输出 {probe_dim} 维，"
            f"已存在索引为 {index_dim} 维（多半是换过 embedding 模型），需要重建索引"
        )
    logger.info("加载本地 FAISS 向量库成功: %s", cfg.index_dir)
    return vector_db


def load_or_build_index(cfg: Config, embeddings: Embeddings, rebuild: bool = False) -> FAISS:
    """加载索引；不存在、损坏（如更换了 embedding 模型导致维度不匹配）或强制时自动重建。"""
    if rebuild:
        logger.info("检测到 --rebuild，强制重建向量库")
        return build_index(cfg, embeddings)
    if not cfg.index_dir.exists():
        logger.info("未检测到 FAISS 索引，开始读取 PDF 并构建向量库")
        return build_index(cfg, embeddings)
    try:
        return load_index(cfg, embeddings)
    except Exception as e:  # noqa: BLE001 - 索引损坏时自动重建兜底
        logger.warning("加载索引失败(%s)，尝试自动重建...", e)
        return build_index(cfg, embeddings)


def build_retriever(vector_db: FAISS, cfg: Config):
    # 基础向量检索，去掉 Rerank 重排；如需更高精度可自行接入 bge-reranker
    return vector_db.as_retriever(search_kwargs={"k": cfg.retrieve_top_k})


def format_docs(docs: List[Document]) -> str:
    blocks = [
        f"【片段{i + 1}，来源：{doc.metadata.get('source', '未知来源')}】\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n".join(blocks)


def build_rag_chain(retriever, cfg: Config):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "参考内容：\n{context}\n\n用户问题：{question}"),
        ]
    )
    llm = ChatOpenAI(
        model=cfg.llm_model,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        timeout=cfg.llm_timeout,
    )
    # 用 RunnablePassthrough.assign 保留输入里的 question / history 字段，只追加 context；
    # 若改用 RunnableParallel 的 dict 组合，未声明的键（history）会被丢弃，
    # 导致 prompt 缺少 {history} 报错。
    # context 分支必须先 itemgetter("question") 取出字符串再进 retriever，
    # 否则 retriever 拿到 dict 去向量化，Embedding API 会报
    # "contents is neither str nor list of str"。
    rag_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# ===================== RAG 引擎（CLI 与 Web 共用） =====================
class RAGEngine:
    """封装索引加载/构建、检索、问答与会话管理。"""

    def __init__(self, cfg: Config, rebuild: bool = False, api_key: Optional[str] = None):
        self.cfg = cfg
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("未配置 DASHSCOPE_API_KEY，请在 .env 中填写 DASHSCOPE_API_KEY=sk-xxx")
        self.embeddings = DashScopeEmbedding(cfg.embedding_model, api_key)
        self.vector_db = load_or_build_index(cfg, self.embeddings, rebuild=rebuild)
        self.retriever = build_retriever(self.vector_db, cfg)
        self.history_mgr = SessionHistoryManager(max_turns=cfg.max_history_turns)
        self.chain = build_rag_chain(self.retriever, cfg)

    # ---- 检索 ----
    def retrieve(self, question: str, top_k: Optional[int] = None):
        """返回 [(Document, score)]，score 为 L2 距离（越小越相似）。"""
        k = top_k or self.cfg.retrieve_top_k
        return self.vector_db.similarity_search_with_score(question, k=k)

    def retrieve_payload(self, question: str, top_k: Optional[int] = None) -> List[dict]:
        """检索结果转成 Web 可序列化的 dict 列表。"""
        return [
            {
                "index": i + 1,
                "distance": round(float(score), 4),
                "source": doc.metadata.get("source", "未知来源"),
                "content": doc.page_content,
            }
            for i, (doc, score) in enumerate(self.retrieve(question, top_k))
        ]

    # ---- 问答（非流式，CLI 用） ----
    def answer(self, question: str, session_id: str) -> str:
        history = self.history_mgr.get(session_id)
        ans = self.chain.invoke({"question": question, "history": history})
        self.history_mgr.add(session_id, question, ans)
        return ans

    # ---- 问答（流式，Web 用） ----
    def stream_answer(self, question: str, session_id: str):
        """生成器：yield (kind, data)，kind ∈ {sources, token, done}；结束后写入历史。"""
        yield ("sources", self.retrieve_payload(question))
        history = self.history_mgr.get(session_id)
        parts: List[str] = []
        for chunk in self.chain.stream({"question": question, "history": history}):
            parts.append(chunk)
            yield ("token", chunk)
        answer = "".join(parts)
        self.history_mgr.add(session_id, question, answer)
        yield ("done", {"answer": answer})

    # ---- 会话 ----
    def clear_session(self, session_id: str) -> None:
        self.history_mgr.clear(session_id)

    # ---- 知识库信息（Web 侧边栏用） ----
    def kb_info(self) -> dict:
        docs = sorted(self.cfg.docs_dir.glob("*.pdf"))
        return {
            "docs": [{"name": p.name} for p in docs],
            "index": {
                "exists": self.cfg.index_dir.exists(),
                "dim": getattr(self.vector_db.index, "d", None),
                "vectors": getattr(self.vector_db.index, "ntotal", None),
            },
            "model": {
                "llm": self.cfg.llm_model,
                "embedding": self.cfg.embedding_model,
            },
        }
