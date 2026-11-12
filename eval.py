"""评测 v2：从"文档级命中"改成"块级命中"。

为什么要改：
v1 只判断 Top-1 来自哪个文档，三篇文档主题差异极大，两种策略都 10/10，
得出"两种切分一样好"的错误结论。

真正的差异在块粒度：召回的块里**到底有没有答案**。
所以每条问题加一个关键词，检查 Top-1 块是否包含它。
关键词必须是"只有答案块才会有"的词，不能是通用词。

三个指标：
- hit@1_doc : Top-1 来自正确文档（粗）
- hit@1_kw  : Top-1 块里包含答案关键词（细，最有意义）
- avg_sim   : Top-1 平均相似度（看召回的"确信程度"）
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from ask import retrieve

# (问题, 期望文档, 答案关键词)
CASES = [
    ("流式中断为什么要传到最底层", "01-流式输出", "AbortController"),
    ("为什么不用 EventSource", "01-流式输出", "不支持 POST"),
    ("TextDecoder 为什么要传 stream 参数", "01-流式输出", "乱码"),
    ("done 和 error 事件为什么要分开", "01-流式输出", "无法区分"),
    ("tool_calls 里的 arguments 是什么类型", "02-工具调用", "JSON 字符串"),
    ("为什么要设置 MAX_ROUNDS", "02-工具调用", "循环"),
    ("并行调用和串行调用有什么区别", "02-工具调用", "串行依赖"),
    ("缓存命中的输入价格是标准价的多少", "03-成本核算", "五十分之一"),
    ("为什么配额要按金额而不是按次数", "03-成本核算", "四倍"),
    ("首字延迟的实测值是多少", "03-成本核算", "三百到四百"),
]


def run(strategy: str):
    hit_doc = hit_kw = hit_kw4 = 0
    sims = []
    rows = []
    for q, expect, kw in CASES:
        hits = retrieve(q, strategy, k=4)
        top = hits[0]
        h_doc = top["meta"]["source"] == expect
        h_kw = kw in top["text"]
        # top-4 里任意一块包含关键词：这才是拼上下文后"答案在不在"的真实情况，
        # 因为生成阶段用的是 top-4，不是只有 top-1。
        h_kw4 = any(kw in h["text"] for h in hits)
        hit_doc += h_doc
        hit_kw += h_kw
        hit_kw4 += h_kw4
        sims.append(top["sim"])
        rows.append((q, h_doc, h_kw, top["sim"]))
    return hit_doc, hit_kw, hit_kw4, sum(sims) / len(sims), rows


def main() -> None:
    summary = {}
    n = len(CASES)
    for strategy in ("heading", "fixed"):
        hit_doc, hit_kw, hit_kw4, avg, rows = run(strategy)
        summary[strategy] = (hit_kw, hit_kw4, avg)
        print(f"\n===== {strategy} =====")
        for q, h_doc, h_kw, sim in rows:
            mark = "KW " if h_kw else ("doc" if h_doc else "MISS")
            print(f"  [{mark}] sim={sim:.4f}  {q[:30]}")
        print(f"  → top1答案={hit_kw}/{n}  top4答案={hit_kw4}/{n}  avg_sim={avg:.4f}")

    print("\n===== 汇总 =====")
    for s, (kw, kw4, avg) in summary.items():
        print(f"  {s:<8} top1 {kw}/{n} ({kw/n:.0%})  top4 {kw4}/{n} ({kw4/n:.0%})  平均相似度 {avg:.4f}")


if __name__ == "__main__":
    main()
