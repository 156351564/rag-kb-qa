"""切分策略：今天要做对比实验的两套方案。

为什么切分是 RAG 里最值得讲的一环：
它决定了"模型能看到什么"。切错了，后面检索再准、模型再强都救不回来，
因为**被切断的信息根本不在上下文里**。

两套策略：
- fixed : 固定窗 + 重叠。简单、可控，但会把语义单元切断。
- heading: 按 Markdown 标题切。语义完整，但块长不可控。
"""
import re

from config import CHUNK_OVERLAP, CHUNK_SIZE


def fixed_window(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """固定窗切分：每次走 size-overlap 步长。

    重叠的作用：如果一句关键条件正好卡在边界，前后两块各带一半。
    没有重叠时两块都不完整，检索命中哪一块都答不对。
    """
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        piece = text[start : start + size]
        if len(piece.strip()) < 20:  # 末尾残片丢掉，没有信息量
            break
        chunks.append(piece.strip())
        if start + size >= len(text):
            break
    return chunks


def by_heading(text: str, min_len: int = 80, max_len: int = 900) -> list[str]:
    """按 Markdown 标题切分，并做两件收尾：
    - 太短的块合并到上一块（避免"## 注意"这种孤零零的标题成块）
    - 太长的块再按固定窗切（避免一个超长章节挤爆上下文）
    """
    lines = text.splitlines()
    blocks: list[str] = []
    buf: list[str] = []

    for line in lines:
        if re.match(r"^#{1,4}\s+", line) and buf:
            blocks.append("\n".join(buf).strip())
            buf = [line]
        else:
            buf.append(line)
    if buf:
        blocks.append("\n".join(buf).strip())

    merged: list[str] = []
    for b in blocks:
        if not b:
            continue
        if merged and len(b) < min_len:
            merged[-1] = merged[-1] + "\n" + b
        else:
            merged.append(b)

    out: list[str] = []
    for b in merged:
        if len(b) <= max_len:
            out.append(b)
        else:
            out.extend(fixed_window(b, size=max_len, overlap=CHUNK_OVERLAP))
    return out


STRATEGIES = {
    "fixed": fixed_window,
    "heading": by_heading,
}
