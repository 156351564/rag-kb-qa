"""配置集中在一处：所有环境变量只在这里读一次。

为什么单独一个文件：
后面 FastAPI、入库脚本、检索脚本都要用这些值。散在各处改起来容易漏，
而且面试时"配置从哪来"是常被问的点——集中在一个文件最好讲。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

# ---- 对话模型（DeepSeek，OpenAI 兼容接口）----
CHAT_BASE_URL = os.getenv("OPENAI_BASE_URL")
CHAT_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")

# ---- Embedding（通义 DashScope，同样走 OpenAI 兼容接口）----
# 关键决策：DeepSeek 没有 embedding 接口，必须换供应商。
# 换的代价是"两套 key、两套限流"——这是 RAG 项目第一个真实坑。
EMBED_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_API_KEY = os.getenv("DASHSCOPE_API_KEY")
EMBED_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")

# ---- 向量库（Chroma 本地持久化）----
CHROMA_DIR = str(ROOT / "chroma_db")
# 注意：Chroma 1.x 要求集合名 3-512 字符，且只能含 [a-zA-Z0-9._-]。
# 用 "kb" 会直接报 Validation error —— 这个坑排查起来不快，记一下。
COLLECTION = "kb_chunks"

# ---- 切分参数 ----
CHUNK_SIZE = 500     # 固定窗大小（字符数，中文按字算）
CHUNK_OVERLAP = 80   # 相邻块重叠，避免一句话被切断后两边都不完整
