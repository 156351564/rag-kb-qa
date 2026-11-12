"""入库脚本：文档 → 切分 → 向量 → Chroma。

用法：
    python ingest.py                  # 默认 heading 策略
    python ingest.py fixed            # 固定窗策略
    python ingest.py heading --reset  # 清库重建

为什么 id 要自己拼：
Chroma 的 upsert 靠 id 去重。如果 id 用随机值，同一个文档重复入库会产生
重复块，检索时 top-k 里全是同一段话 —— 这是"看起来能用但效果差"的典型。
所以 id 用 {文件名}:{策略}:{序号}，天然幂等。
"""
import sys
from pathlib import Path

import chromadb

from chunker import STRATEGIES
from config import CHROMA_DIR, COLLECTION
from embedding import embed

DOCS_DIR = Path(__file__).parent / "docs"


def load_docs() -> list[tuple[str, str]]:
    files = sorted(DOCS_DIR.glob("*.md"))
    if not files:
        raise SystemExit(f"docs 目录没有 md 文件：{DOCS_DIR}")
    return [(f.stem, f.read_text(encoding="utf-8")) for f in files]


def main() -> None:
    strategy = sys.argv[1] if len(sys.argv) > 1 else "heading"
    reset = "--reset" in sys.argv

    if strategy not in STRATEGIES:
        raise SystemExit(f"未知策略：{strategy}，可选 {list(STRATEGIES)}")

    splitter = STRATEGIES[strategy]
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if reset:
        try:
            client.delete_collection(COLLECTION)
            print(f"[reset] 已删除集合 {COLLECTION}")
        except Exception:
            pass

    col = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},  # 用余弦距离，文本语义相似度更合适
    )

    docs = load_docs()
    ids, texts, metas = [], [], []

    for name, content in docs:
        chunks = splitter(content)
        for i, chunk in enumerate(chunks):
            ids.append(f"{name}:{strategy}:{i}")
            texts.append(chunk)
            metas.append({"source": name, "strategy": strategy, "index": i})

    print(f"[load] {len(docs)} 个文档 → {len(texts)} 个块（策略={strategy}）")

    vectors = embed(texts)
    print(f"[embed] 完成，维度 {len(vectors[0])}")

    col.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
    print(f"[store] 写入 {col.name}，当前总数 {col.count()}")


if __name__ == "__main__":
    main()
