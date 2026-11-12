"""Embedding 封装：把文本批量转向量。

两个必须知道的细节：
1. 通义的 OpenAI 兼容端点不支持无限批量，v3 单次最多 10 条 —— 所以要分批。
   不分批会直接报错，而且报错信息不会告诉你"是批量太大"。
2. 入库和查询必须用同一个 model。用 v3 入库、v2 查询，向量空间不一致，
   相似度会算成垃圾值，但不会报错 —— 这是最隐蔽的坑。
"""
from openai import OpenAI

from config import EMBED_API_KEY, EMBED_BASE_URL, EMBED_MODEL

_client = OpenAI(api_key=EMBED_API_KEY, base_url=EMBED_BASE_URL)

BATCH = 10  # 通义 text-embedding-v3 单请求上限


def embed(texts: list[str]) -> list[list[float]]:
    """把一批文本转成向量，顺序与输入一致。"""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        resp = _client.embeddings.create(model=EMBED_MODEL, input=batch)
        # resp.data 的顺序和输入顺序一致，但保险起见按 index 排序
        vectors.extend([d.embedding for d in sorted(resp.data, key=lambda d: d.index)])
    return vectors


if __name__ == "__main__":
    v = embed(["流式输出怎么防止续费", "Function Calling 的回填"])
    print("条数:", len(v), "维度:", len(v[0]))
