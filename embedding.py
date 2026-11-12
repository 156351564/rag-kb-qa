"""Embedding 封装：把文本批量转向量。

两个必须知道的细节：
1. 通义的 OpenAI 兼容端点不支持无限批量，v3 单次最多 10 条 —— 所以要分批。
   不分批会直接报错，而且报错信息不会告诉你"是批量太大"。
2. 入库和查询必须用同一个 model。用 v3 入库、v2 查询，向量空间不一致，
   相似度会算成垃圾值，但不会报错 —— 这是最隐蔽的坑。
"""
from openai import OpenAI

from config import EMBED_API_KEY, EMBED_BASE_URL, EMBED_MODEL

import time

_client = OpenAI(api_key=EMBED_API_KEY, base_url=EMBED_BASE_URL, timeout=60.0)

BATCH = 10   # 通义 text-embedding-v3 单请求上限
RETRIES = 3  # 网络抖动重试次数


def _embed_batch(batch: list[str]) -> list[list[float]]:
    """单批请求，带指数退避重试。

    为什么要重试：embedding 是外部依赖，超时和 5xx 是常态。
    入库脚本跑到一半失败，重跑整个文档代价很大（前面的 token 白花），
    所以在批级别重试是最划算的粒度。
    """
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            resp = _client.embeddings.create(model=EMBED_MODEL, input=batch)
            return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        except Exception as e:
            last = e
            if attempt < RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))  # 1.5s、3s 递增
    raise RuntimeError(f"embedding 连续失败 {RETRIES} 次：{last}") from last


def embed(texts: list[str]) -> list[list[float]]:
    """把一批文本转成向量，顺序与输入一致。"""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        vectors.extend(_embed_batch(texts[i : i + BATCH]))
    return vectors


if __name__ == "__main__":
    v = embed(["流式输出怎么防止续费", "Function Calling 的回填"])
    print("条数:", len(v), "维度:", len(v[0]))
