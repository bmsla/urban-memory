# -*- coding: utf-8 -*-
"""企业知识库多轮问答（命令行版）。

用法:
  1. 在 .env 中配置 DASHSCOPE_API_KEY
  2. python main.py              # 问答模式
     python main.py --rebuild     # 强制重建向量库
     python main.py --docs ./other --session my_sess

对话命令:
  exit / quit : 退出
  /clear      : 清空当前会话历史
  /help       : 查看帮助

Web 版请运行: python app.py
"""

import argparse
from pathlib import Path

from rag_engine import Config, RAGEngine, logger


def run_chat_loop(engine: RAGEngine, cfg: Config) -> None:
    print("===== 企业知识库多轮问答 =====")
    print("命令: exit/quit 退出 · /clear 清空当前会话 · /help 帮助")
    while True:
        try:
            raw = input("\n请输入问题：")
        except (KeyboardInterrupt, EOFError):
            print("\n对话结束")
            break
        q = raw.strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("对话结束")
            break
        if q == "/clear":
            engine.clear_session(cfg.session_id)
            continue
        if q == "/help":
            print("可用命令: exit/quit 退出, /clear 清空当前会话历史")
            continue

        # 1) 召回展示（向量距离越小越相似；开启 rerank 后按精排相关度排序）
        hits = engine.retrieve(q)
        print("\n【召回片段】")
        for idx, hit in enumerate(hits, 1):
            doc = hit["doc"]
            src = doc.metadata.get("source", "未知")
            if hit["rerank_score"] is not None:
                print(f"--- 片段{idx} 精排相关度:{hit['rerank_score']:.4f} (向量距离:{hit['distance']}) 来源:{src}")
            else:
                print(f"--- 片段{idx} 向量距离:{hit['distance']:.4f} 来源:{src}")
            print(doc.page_content)
            print()

        # 2) 生成回答（异常时提示而非直接崩溃，便于排查模型名/配额问题）
        try:
            ans = engine.answer(q, cfg.session_id)
        except Exception as e:  # noqa: BLE001 - 交互模式下友好降级
            logger.error("回答生成失败: %s", e)
            print(f"[错误] 生成回答失败：{e}\n请检查模型名称 / API Key / 配额后重试。")
            continue
        print(f"回答：{ans}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="企业知识库多轮问答 (RAG, 命令行版)")
    parser.add_argument("--docs", type=Path, help="PDF 文档目录（默认 ./docs）")
    parser.add_argument("--index", type=Path, help="FAISS 索引目录（默认 ./faiss_index）")
    parser.add_argument("--rebuild", action="store_true", help="强制重建向量库")
    parser.add_argument("--session", help="会话 ID（默认 session_001）")
    parser.add_argument("--top-k", type=int, help="检索返回片段数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config().apply_args(args)
    engine = RAGEngine(cfg, rebuild=args.rebuild)
    run_chat_loop(engine, cfg)


if __name__ == "__main__":
    main()
