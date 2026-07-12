# EngramTide

EngramTide 是一个面向 AI Agent 的长期记忆系统，实现四类记忆
（semantic / episodic / emotional / procedural）的差异化管理。

**核心闭环：** 对话 → 话题分割 + 脱水写入记忆（类型感知管线） → 衰减与浮现 → 逐轮 Context-Aware 激活 → 双通道检索 + 浮现合并 → Token 预算截断 → Constitutional AI 上下文 → 带记忆回复。

## Context-Aware 激活与衰减

- **逐轮激活引擎**：每轮用户输入后、检索之前，按余弦相似度对 episodic/emotional 记忆做激活——强激活（sim > `SIMILARITY_HIGH`，+0.2 并计访问）/ 轻激活（sim > `SIMILARITY_MID`，+0.05 不计访问）。纯向量计算，零 LLM 调用。
- **休眠唤醒**：沉底记忆（decay_weight < 0.1，检索不到但未删除）被相关输入强激活后当轮立即重回检索候选池。
- **可配置激活阈值**：源码默认 `SIMILARITY_MID=0.33`、`SIMILARITY_HIGH=0.56`，均可通过同名环境变量覆盖；`check_config()` 强制检查 `0 < MID < HIGH ≤ 1`。
- **总开关**：`CONTEXT_AWARE_ENABLED=false` 时完全关闭逐轮激活。

## 上下文、检索与记忆维护

- **话题分割**：退出脱水前先调 LLM 按话题切分会话，每段独立脱水（embedding 语义更纯）；索引校验失败/LLM 失败整段回退，单段失败不牵连其余段（记忆零丢失）。
- **双通道检索**：`(0.7·向量 + 0.3·rapidfuzz 关键词) × decay_weight`，关键词通道兜住专有名词；`KEYWORD_CHANNEL_ENABLED=false` 或 rapidfuzz 缺失时降级为纯向量检索。
- **Token 预算**：`MAX_CONTEXT_TOKENS=1500` 替代字符数近似——DeepSeek 系数保守估算器（宁高勿低），按条截断、procedural 保底、被丢弃记忆不计访问。
- **轻激活会话上限**：`MILD_ONCE_PER_SESSION`——同一会话内同一记忆至多轻激活 1 次（强激活不设限），抑制发生在截断之前不占槽位。
- **记忆合并**：`/consolidate` 手动触发——同类型、相似度 > 0.92 的近重复对由 LLM 融合为一条，旧记忆 `superseded_by` 指向新记忆；`preview` 模式零成本预览。

## 快速开始

### 1. 环境准备

```powershell
# 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

```powershell
# 复制环境变量模板
copy .env.example .env

# 编辑 .env，填入你的 API Key
# DEEPSEEK_API_KEY=sk-your-actual-key
```

Embedding 服务需要单独配置，并提供 OpenAI-compatible `/v1/embeddings` 接口：

```powershell
EMBEDDING_API_KEY=sk-your-embedding-key
EMBEDDING_BASE_URL=https://your-embedding-provider.example.com
EMBEDDING_MODEL=text-embedding-3-small
```

### 3. 启动

```powershell
python main.py
```

启动时会自动运行衰减更新并提取浮现记忆；活跃 episodic/emotional 超过 500 条时提示运行 `/consolidate preview`。

### 4. 功能开关（环境变量）

| 变量 | 默认 | 关闭后行为 |
| --- | --- | --- |
| `CONTEXT_AWARE_ENABLED` | true | 关闭逐轮语义激活 |
| `TOPIC_SPLIT_ENABLED` | true | 不分割话题，整段会话直接脱水 |
| `KEYWORD_CHANNEL_ENABLED` | true | 只使用向量相似度检索 |
| `MILD_ONCE_PER_SESSION` | true | 轻激活不设会话内次数上限 |

## 对话命令

| 命令 | 说明 |
| --- | --- |
| `/exit` `/quit` | 结束会话，触发话题分割 + 脱水 + 写入管线，退出程序 |
| `/memories` | 打印数据库中最近 10 条记忆 |
| `/consolidate` | 合并近重复记忆（LLM 融合 + supersede 落库） |
| `/consolidate preview` | 预览合并候选对（零成本，不执行） |
| `/debug on` / `off` | 开关调试日志 |
| `/debug decay` | 查看最近一次衰减摘要 |
| `/debug surface` | 查看当前浮现记忆列表 |
| `/debug activation` | 查看本会话激活统计（强/轻/唤醒/抑制）与最近一轮明细 |
| `/debug retrieval` | 最近一轮 Top-K 分数分解 `[vec=\|kw=\|w=\|→]` |
| `/debug context` | Token 预算用量（估算/预算/收录/丢弃） |

## 项目结构

```
EngramTide/
├── README.md
├── requirements.txt
├── config.py              # 全部阈值、功能开关与服务配置
├── main.py                # CLI 主循环（衰减→浮现→逐轮激活→检索→上下文→写入）
├── core/
│   ├── memory_store.py    # SQLite 存储层（哑 CRUD + meta 表）
│   ├── embedding.py       # Embedding API 客户端
│   ├── dehydrator.py      # 话题分割 + 分段脱水
│   ├── retriever.py       # 向量 + 关键词双通道检索
│   ├── context_builder.py # Constitutional AI 上下文 + Token 预算
│   ├── chat.py            # LLM 聊天调用
│   ├── decay.py           # 衰减引擎 + 浮现 + 逐轮激活
│   ├── memory_writer.py   # 类型感知写入管线
│   └── consolidator.py    # 记忆合并与去重
├── prompts/
│   ├── system_constitution.txt  # 交互宪法模板
│   ├── dehydrate.txt            # 脱水 prompt
│   ├── topic_split.txt          # 话题分割 prompt
│   └── consolidate.txt          # 记忆融合 prompt
├── utils/
│   ├── similarity.py      # 余弦相似度
│   ├── time_utils.py      # UTC 时间工具（全项目禁用裸 datetime.now()）
│   └── token_counter.py   # Token 保守估算器
├── tests/                 # 单元 + 集成测试（277 项）
│   └── fixtures/          # 真实 API 校准样本对等归档（不进 CI）
└── data/                  # SQLite 数据库存放目录
```

## 核心架构

```
会话启动
  ├─ run_decay_update()       → 衰减引擎（纯数学，无 LLM）
  ├─ get_surfaced_memories()  → 主动浮现（R1>R2>R3>R4 优先级）
  └─ 库规模检查 → /consolidate 提示

每轮对话
  ├─ embed_text(user_input)
  ├─ context_aware_update()   → 逐轮激活（强/轻/唤醒，轻激活受会话上限）
  ├─ retrieve_memories_detailed() → 双通道检索（vec+kw，含分数分解）
  ├─ build_constitutional_memory_context(retrieved, surfaced=...)
  │     └─ 去重合并（浮现优先）→ 四类分组 → Token 预算按条截断
  └─ generate_response()      → Constitutional AI 模板 → LLM

会话退出
  ├─ split_conversation()     → LLM 话题分割（失败整段回退）
  ├─ _dehydrate_segment() × N → 逐段脱水（单段失败不牵连其余段）
  └─ write_memories()         → 写入管线
        ├─ semantic   → 覆盖检测 → mark_superseded
        ├─ emotional  → 强化检测 → reinforce_memory（不 insert）
        ├─ procedural → 去重检测 → mark_accessed（不 insert）
        └─ episodic   → 直接 insert
```

## 衰减与激活公式

| 类型 | 衰减率 | 说明 |
|------|--------|------|
| semantic | 0 | 永不衰减 |
| procedural | 0 | 永不衰减 |
| episodic | `0.05` | 标准艾宾浩斯 |
| emotional | `0.05 × (1 - arousal × 0.7)` | 唤醒度越高衰减越慢 |

```
multiplier = exp(-rate × hours_elapsed / (24 × (1 + ln(1 + access_count))))
new_weight = max(0.01, old_weight × multiplier)
```

逐轮激活（衰减是乘法通道，激活是加法通道）：

```
sim > SIMILARITY_HIGH                         → weight = min(1.0, weight + 0.2)，access_count + 1
SIMILARITY_MID < sim ≤ SIMILARITY_HIGH       → weight = min(1.0, weight + 0.05)，同会话每记忆至多 1 次
单轮激活上限 30 条（按相似度降序截断）
```

## 浮现规则

| 优先级 | 条件 | 阈值 |
|--------|------|------|
| R1 | procedural | 永远浮现 |
| R2 | emotional | `arousal > 0.7` 且 `decay_weight > 0.3` |
| R3 | unresolved | `decay_weight > 0.2` |
| R4 | episodic 近期 | `≤ 3 天` 且 `decay_weight > 0.5` |

超出上限（8 条）时按 R1 > R2 > R3 > R4 截断，同规则内按 decay_weight 降序。

## 运行测试

```powershell
pytest -p asyncio -o asyncio_mode=auto
```

当前 **277 项测试全部通过**，覆盖：衰减数学、激活阈值边界、浮现筛选、写入管线、话题分割回退、双通道打分、Token 预算截断、会话上限、合并候选与端到端、功能开关回退等价性和核心流程回归。

## 技术栈

- **语言：** Python 3.10+
- **数据库：** SQLite（含 WAL 模式）
- **向量存储：** numpy float32 BLOB 存入 SQLite
- **LLM：** DeepSeek（OpenAI-compatible 接口）
- **Embedding：** 独立 OpenAI-compatible `/v1/embeddings` provider
- **关键词匹配：** rapidfuzz（缺失时自动降级纯向量）

## 开发过程中的后续研究备注

本项目首先作为长期记忆系统完成设计和实现，后续论文问题来自开发验收过程中的意外观察，
并非为了证明论文结论而预先搭建系统。

系统功能完成后的收尾验收记录中包含一项“**S3 消融实验归因修正**”：早期曾把 S3 中的
虚假保鲜主要归因于会话内重复轻激活，
但加入 `MILD_ONCE_PER_SESSION` 后，轻激活次数虽然显著下降，虚假保鲜率却没有变化。
进一步拆解发现，小记忆库中的邻居几乎每轮都会进入 Top-5，检索计访问使
`access_count` 持续升高，并通过 `1 + ln(1 + access_count)` 减缓衰减。原归因因此被消融证伪。

为分开度量不同访问来源，后续在更大记忆库上加入了检索、语义激活和系统自主浮现的
访问计数消融。这个过程中又观察到：自主浮现不仅把记忆带入上下文，还会计作一次访问，
从而改变后续衰减速度；也就是说，记忆系统自身的浮现行为会干扰对“自然遗忘”的测量。

随后引入 Qwen3-Embedding-0.6B 作为第二个 embedding 空间进行对照，并分别校准语义激活阈值。
两个空间给出的通道份额排序发生变化，而作为固定参照的自主浮现事件数不会因为其他通道变强
或变弱而被动改写。由此形成论文的核心结论：在阈值门控的 Agent 记忆系统中，仅用相对份额做
通道归因并不稳健；应同时报告绝对事件数、embedding 模型、阈值与完整校准协议。
