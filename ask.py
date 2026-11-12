"""检索 + 生成：RAG 的在线链路。

用法：
    python ask.py "流式中断为什么需要三层"                 # 只检索，看命中
    python ask.py "流式中断为什么需要三层" --answer        # 检索 + 模型回答
    python ask.py "..." --strategy fixed                  # 指定用哪套切分

三个关键设计：
1. 召回后必须看分数。相似度低就承认"资料里没有"，不要让模型硬编。
   幻觉最容易出现在"检索到不相关片段，但模型仍然努力回答"的时候。
2. 引用要带来源和分数，前端才能做成可点击的卡片。
3. 上下文要标注序号，模型才能正确地引用 [1] [2]。
"""
import sys

# Windows 控制台默认 GBK，直接 print 中文/符号会 UnicodeEncodeError。
# 在脚本入口把 stdout 改成 UTF-8，比每次设环境变量省事。
sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI

from config import CHAT_API_KEY, CHAT_BASE_URL, CHAT_MODEL
from kb import retrieve

TOP_K = 4
MIN_SIM = 0.30   # 低于这个分数视为没召回到，直接兜底
MAX_CTX = 3000   # 拼进 prompt 的上下文字符上限，防止挤爆

_chat = OpenAI(api_key=CHAT_API_KEY, base_url=CHAT_BASE_URL)

SYSTEM = """你是知识库问答助手，只能依据给出的资料回答。
规则：
1. 资料里没有的内容，直接说「资料里没有相关内容」，不要编。
2. 回答后标注引用的资料序号，格式为 [1] [2]。
3. 回答简洁，不超过 200 字。"""


def build_context(hits) -> str:
    parts, used = [], 0
    for i, h in enumerate(hits, start=1):
        block = f"[{i}] 来源：{h['meta']['source']}\n{h['text']}"
        if used + len(block) > MAX_CTX:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def answer(question: str, hits) -> str:
    ctx = build_context(hits)
    resp = _chat.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"资料：\n{ctx}\n\n问题：{question}"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('用法：python ask.py "问题" [--answer] [--strategy fixed|heading]')
    question = sys.argv[1]
    want_answer = "--answer" in sys.argv
    strategy = None
    if "--strategy" in sys.argv:
        strategy = sys.argv[sys.argv.index("--strategy") + 1]

    hits = retrieve(question, strategy)
    print(f"\n【问题】{question}   （策略={strategy or '全部'}）")
    print("【召回】")
    for i, h in enumerate(hits, start=1):
        flag = "✓" if h["sim"] >= MIN_SIM else "✗"
        print(f"  {flag} [{i}] sim={h['sim']:.4f}  {h['meta']['source']} #chunk{h['meta']['index']}")
        print(f"       {h['text'][:60].replace(chr(10), ' ')}…")

    best = max(h["sim"] for h in hits) if hits else 0
    if best < MIN_SIM:
        print(f"\n【兜底】最高相似度 {best:.4f} < 阈值 {MIN_SIM}，判定资料里没有")
        return

    if want_answer:
        print("\n【回答】")
        print(answer(question, hits))


if __name__ == "__main__":
    main()
