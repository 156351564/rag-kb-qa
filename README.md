# 企业知识库问答（RAG）

> 一个可以跑起来的最小 RAG 系统：文档入库 → 向量检索 → 带引用的流式回答。
> 重点不是"调通 API"，而是把**切分策略对比、召回兜底、异步入库**这三个工程问题做完整。

---

## 它解决什么问题

把文档变成可以提问的知识库。和直接粘贴全文给模型的区别：

| | 直接塞全文 | RAG |
|---|---|---|
| 上下文 | 受窗口限制，长文档塞不下 | 只取最相关的 4 个块 |
| 成本 | 每次全量输入 | 输入量降一个数量级 |
| 可解释 | 答错不知道哪来的 | 每条结论带引用，可回原文核对 |
| 更新 | 改一处要重新贴 | 增量入库即可 |

---

## 实测数据

**召回命中率**（10 条评测问题，见 `eval.py`）

| 切分策略 | top-1 含答案 | top-4 含答案 | 平均相似度 |
|---|---|---|---|
| **heading（按标题切，采用）** | 9/10 (90%) | **10/10 (100%)** | **0.6991** |
| fixed（固定窗 500/80） | 10/10 (100%) | 10/10 (100%) | 0.6006 |

**为什么选 heading，而不是 top-1 满分的那一个**：

两者在 top-4 下都能覆盖答案，最终都能答对。差别在**排序精度**——heading 的平均相似度高 16%，
意味着同样拼 4 个块进上下文，它的 4 个块更相关、噪音更少。

fixed 的 top-1 命中率反而更高，原因是**块更大，装的东西更多**，关键词更容易被包含进来。
这是大块的假象，不是精度更高。

**服务端实测**（本地，DeepSeek 官方接口）

| 指标 | 数值 |
|---|---|
| 首字延迟（含检索） | 1161 ms |
| 单次总耗时 | 1269 ms |
| 兜底阈值命中 | 资料外问题 best=0.297 < 0.30，正确拒答 |

> 首字比纯对话（385ms）高 3 倍，因为多了一次 query embedding + 向量检索，占 200-400ms。

---

## 架构

```
离线：docs/*.md
        │
        ├─ chunker.py ── heading（按标题切） / fixed（固定窗 500/80）
        │
        ├─ embedding.py ── 通义 text-embedding-v3（1024 维，批量 10 条）
        │
        └─ kb.py ── Chroma（cosine 距离，本地持久化）

在线：question
        │
        ├─ embed → 向量检索 top-4
        ├─ 相似度 < 0.30 → 直接兜底「资料里没有」，不让模型硬编
        └─ 拼上下文 → 流式生成 → 带 [1][2] 引用
```

```
rag-service/
├── config.py        # 所有环境变量集中一处
├── embedding.py     # 通义 embedding 封装（分批 + 重试）
├── chunker.py       # 两套切分策略
├── kb.py            # 知识库核心层：入库 / 检索 / 统计 / 文档列表
├── ingest.py        # CLI：批量入库 docs/
├── ask.py           # CLI：单次问答（可只看检索）
├── eval.py          # 10 条评测，对比两种切分
├── app.py           # FastAPI：异步入库 + job 轮询 + 流式问答 + 提供页面
├── static/
│   └── index.html   # 单页界面（原生 JS 解 SSE 帧）
└── docs/            # 测试文档（3 篇，取自项目一的技术笔记）
```

---

## 启动

```powershell
# 1. 环境（用 uv 建 3.12；3.14 上 chromadb 还没有对应 wheel）
uv venv --python 3.12
uv pip install -r requirements.txt

# 2. 配置
# 复制 .env.example 为 .env，填入 DASHSCOPE_API_KEY 与 OPENAI_* 三件套

# 3. 入库（两种策略各跑一次）
python ingest.py heading --reset
python ingest.py fixed

# 4. 命令行问答
python ask.py "中断为什么要传到最底层" --strategy heading --answer

# 5. 评测
python eval.py

# 6. 起服务（同时提供页面和接口）
python -m uvicorn app:app --host 127.0.0.1 --port 8123
# 浏览器打开 http://127.0.0.1:8123
```

## 页面

单页界面由 FastAPI 直接提供（`static/index.html`），一个进程搞定：

- 左侧：入库（选切分策略）+ 进度条轮询 + 已入库文档列表
- 右侧：提问 → 引用卡片先出现 → 逐字生成 → 首字/总耗时打点

**为什么不做成独立的 Next 项目**：Vercel 跑不了常驻的 Python 服务，
页面和接口分两个域名部署，线上就会出现「页面能开、接口 404」。
内嵌成一个进程，演示时不会有这个问题。

前端解 SSE 帧用的是 `fetch + ReadableStream`：累积 buffer，按双换行切分，
剩余的留到下次读取。**不能假设一次 read 拿到完整帧** —— 网络分包会把一帧切开。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 单页界面 |
| GET | `/health` | 存活检查 |
| GET | `/api/kb/stats` | 当前块数 |
| GET | `/api/kb/sources` | 已入库文档列表（按文档 + 策略聚合块数） |
| POST | `/api/ingest` | **异步入库**，返回 `job_id` |
| GET | `/api/jobs/{id}` | 轮询进度（queued / running / done / failed） |
| POST | `/api/ask` | **SSE 流式问答**，先推引用再推正文 |

SSE 事件协议：

```
event: citations  →  { citations: [{index, source, chunk, sim, preview}], best }
event: delta      →  { text }
event: done       →  { reason: "finish" | "low_similarity" }
event: error      →  { message }
```

---

## 三个关键设计

### 1. 入库必须异步

embedding 是慢操作，几十个块要几秒到几十秒。放在 HTTP 请求里同步跑，请求会挂着，
前端超时、网关先断开。

所以拆成两步：提交任务立刻返回 `job_id`，后台用 `BackgroundTasks` 跑，前端轮询进度。

**已知限制（主动说）**：`BackgroundTasks` 把任务放进程内存，重启就丢；
多副本部署时提交和查询可能落到不同实例，状态查不到。上生产要换 Celery / RQ / 云队列。

### 2. 相似度阈值 0.30 是测出来的

不是拍的。做法是问一批资料内和资料外的问题，看相似度分布：

- 资料内问题：0.50 ~ 0.80
- 资料外问题：0.24 ~ 0.30

0.30 正好卡在两者之间。**兜底比答错更重要**——检索不到却让模型硬答，是幻觉最集中的场景。

### 3. 引用先发，正文后发

引用在检索阶段就已经确定了，不需要等生成。所以第一个 SSE 事件就推 `citations`，
前端可以立刻把引用卡片渲染出来，用户不用干等。

---

## 踩坑记录

### ① Chroma 1.x 的集合名限制

```
chromadb.errors.InvalidArgumentError: Expected a name containing 3-512 characters
from [a-zA-Z0-9._-], starting and ending with a character in [a-zA-Z0-9]. Got: kb
```

集合名用 `kb` 会直接失败——**2 个字符不满足 3 字符下限**。改成 `kb_chunks`。
这个报错信息其实写得很清楚，但第一眼容易以为是连接问题。

### ② DeepSeek 没有 embedding 接口

对话和向量必须分两个供应商：对话走 DeepSeek，向量走通义 `text-embedding-v3`。
代价是两套 key、两套限流、两套错误码。这是 RAG 项目第一个真实成本。

另外通义 v3 单请求最多 10 条文本，必须分批，否则报错但不会告诉你"是批量太大"。

### ③ Windows 控制台 GBK

脚本里 `print` 中文或 `✓` 这类符号会 `UnicodeEncodeError`。在入口加一行：

```python
sys.stdout.reconfigure(encoding="utf-8")
```

比每次设环境变量省事。

### ④ 评测指标设计错了，结论就是错的

第一版评测只判断「Top-1 来自哪个文档」，结果两种切分策略都是 10/10，得出「没有差别」的结论。

问题在于：3 篇文档主题差异极大（流式 / 工具调用 / 成本），**闭着眼都能命中文档**。

改成块级判断——检查召回的块里到底有没有答案关键词——才看出真实差异在相似度和排序精度上。

**教训：指标选错，实验等于白做。** 而"指标选错"往往发生在指标看起来合理的时候。

### ⑤ 入库到一半超时，前面的钱白花

第一次跑 `ingest.py heading`，14 个块分 2 批请求，第二批超时直接抛 `APITimeoutError`，
整个入库失败。

加了批级重试（3 次，1.5s / 3s 退避）后解决。**为什么在批级别重试而不是文档级别**：
已经成功的批次不必重跑，重试粒度越细，浪费越少。

同时入库用的 `upsert` + 稳定 id 是幂等的，所以重试不会产生重复块。

---

## 已知限制

- `jobs` 是内存字典，进程重启丢失
- 无鉴权，无配额，不适合直接暴露公网
- 只支持 Markdown，没有 PDF / Word 解析
- 无增量更新检测，改文档要整份重新入库
- 没有 rerank，top-4 里仍会混入语义相近但无关的块
- 评测集只有 10 条，规模不足以下统计结论

---

## 下一步

- [ ] 加 rerank（通义 gte-rerank），先召回 20 条再精排到 4 条
- [ ] 支持 PDF 解析（pypdf）+ 异步任务进度细化到"已处理 3/10 篇"
- [ ] 前端知识库页：上传、文档列表、问答、引用卡片定位
- [ ] 把 jobs 换成 Redis，支持多副本
