# -*- coding: utf-8 -*-
"""RAG 离线评估：检索召回率@k + 答案质量（LLM-as-judge）。

用法：
    python eval.py

对比两档检索配置，量化 RAG 效果：
  - baseline: 纯向量检索（双塔 Embedding）
  - rerank:   向量召回 candidate_k 条 + gte-rerank-v2 Cross-Encoder 精排

指标：
  - 来源召回率@5：正确的来源文档是否进入 Top-5
  - 答案平均分：qwen-max 作为 judge 按参考答案打 1-5 分
  - 正确率：评分 >= 4 的样本占比

结果写入 eval_report.md。
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag_engine import Config, RAGEngine

load_dotenv()

TEST_SET = Path("test_set.json")
REPORT_MD = Path("eval_report.md")

JUDGE_PROMPT = """你是严格的 RAG 评分员。只根据【参考答案要点】判断【模型回答】是否准确，不要自行发挥。
评分规则：5=完全正确覆盖要点；4=基本正确有小瑕疵；3=部分正确；2=大部分错误；1=完全错误或编造。
只输出一个整数分数。

问题：{q}
参考答案要点：{ref}
模型回答：{ans}
分数："""


def judge_answer(question: str, reference: str, answer: str, cfg: Config) -> int:
    """用 qwen-max 作为裁判模型给答案打 1-5 分。"""
    llm = ChatOpenAI(
        model=cfg.llm_model,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=cfg.base_url,
        temperature=0,
    )
    out = llm.invoke(JUDGE_PROMPT.format(q=question, ref=reference, ans=answer)).content
    m = re.search(r"[1-5]", out)
    return int(m.group()) if m else 0


def run_mode(use_rerank: bool, samples: list) -> dict:
    cfg = Config()
    cfg.use_rerank = use_rerank
    engine = RAGEngine(cfg)

    rows = []
    for i, s in enumerate(samples):
        hits = engine.retrieve(s["question"], top_k=cfg.retrieve_top_k)
        sources = [h["doc"].metadata["source"] for h in hits]
        hit_ok = s["expected_source"] in sources
        ans = engine.answer(s["question"], session_id=f"eval_{i}")
        score = judge_answer(s["question"], s["reference_answer"], ans, cfg)
        rows.append(
            {
                "question": s["question"],
                "recall": hit_ok,
                "top_source": sources[0] if sources else "-",
                "judge": score,
                "answer": ans,
            }
        )
        print(f"  [{i + 1}/{len(samples)}] 召回={'✓' if hit_ok else '✗'} 评分={score}  {s['question']}")

    n = len(rows)
    return {
        "rows": rows,
        "recall": sum(r["recall"] for r in rows) / n,
        "avg_score": sum(r["judge"] for r in rows) / n,
        "pass_rate": sum(1 for r in rows if r["judge"] >= 4) / n,
    }


def write_report(results: dict) -> None:
    lines = ["# RAG 评估报告\n"]
    lines.append("| 检索配置 | 来源召回率@5 | 答案平均分(1-5) | 正确率(评分>=4) |")
    lines.append("|---|---|---|---|")
    for label, m in results.items():
        lines.append(
            f"| {label} | {m['recall']:.0%} | {m['avg_score']:.2f} | {m['pass_rate']:.0%} |"
        )
    for label, m in results.items():
        lines.append(f"\n## {label} 明细\n")
        for r in m["rows"]:
            lines.append(
                f"- **{r['question']}** —— 召回:{'✓' if r['recall'] else '✗'} "
                f"评分:{r['judge']} 首选来源:{r['top_source']}\n  > {r['answer'][:120]}\n"
            )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    samples = json.loads(TEST_SET.read_text(encoding="utf-8"))
    results = {}
    for use_rerank, label in [(False, "baseline（纯向量检索）"), (True, "rerank（向量召回 + gte-rerank-v2 精排）")]:
        print(f"\n===== {label} =====")
        results[label] = run_mode(use_rerank, samples)
        m = results[label]
        print(f"汇总: 召回率@5={m['recall']:.0%}  答案平均分={m['avg_score']:.2f}  正确率={m['pass_rate']:.0%}")

    write_report(results)
    print(f"\n评估完成，报告已写入 {REPORT_MD}")


if __name__ == "__main__":
    main()
