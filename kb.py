"""知识库核心层：入库与检索。

把 Chroma 的读写集中在这里，有两个直接好处：
1. CLI（ask.py / ingest.py）和 HTTP 服务（app.py）共用同一套逻辑，不会出现
   "命令行好用、接口不好用"的偏差。
2. 面试讲分层时，可以说清楚"业务层不直接碰向量库"。

为什么单独抽这一层而不是写在 router 里：
入库是慢操作（embedding 要几秒到几十秒），需要被后台任务复用；
如果逻辑写在请求处理函数里，后台任务就没法调用它。
"""
from pathlib import Path

import chromadb

from chunker import STRATEGIES
from config import CHROMA_DIR, COLLECTION
from embedding import embed


def get_collection():
    """每次拿一个新句柄。Chroma 的 PersistentClient 轻量，不用做全局单例。"""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_text(source: str, text: str, strategy: str = "heading") -> int:
    """把一段文本切块、向量化、写入。返回块数。

    id 用 {source}:{strategy}:{i} 保证幂等 —— 同一份文档重复入库不会产生重复块。
    这一点很关键：否则检索 top-k 里会出现多份相同内容，看起来能用但效果很差。
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"未知策略：{strategy}")
    chunks = STRATEGIES[strategy](text)
    if not chunks:
        return 0

    ids = [f"{source}:{strategy}:{i}" for i in range(len(chunks))]
    metas = [{"source": source, "strategy": strategy, "index": i} for i in range(len(chunks))]
    vectors = embed(chunks)

    col = get_collection()
    col.upsert(ids=ids, documents=chunks, embeddings=vectors, metadatas=metas)
    return len(chunks)


def ingest_file(path: str | Path, strategy: str = "heading") -> int:
    p = Path(path)
    return ingest_text(p.stem, p.read_text(encoding="utf-8"), strategy)


def retrieve(question: str, strategy: str | None = None, k: int = 4) -> list[dict]:
    """检索 top-k，返回 [{text, meta, sim}]，sim 是余弦相似度（1 - 距离）。"""
    col = get_collection()
    qv = embed([question])[0]
    where = {"strategy": strategy} if strategy else None
    res = col.query(query_embeddings=[qv], n_results=k, where=where)
    return [
        {"text": doc, "meta": meta, "sim": round(1 - dist, 4)}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def stats() -> dict:
    col = get_collection()
    return {"collection": col.name, "chunks": col.count()}
