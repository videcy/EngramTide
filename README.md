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

## 接入现有 Python Agent

除自带 CLI 外，EngramTide 提供本地 Python API。调用方在仓库根目录运行，或将仓库根目录
加入其 Python 环境后，即可导入：

```python
from engramtide import EngramTide
```

最小会话流程：

```python
from engramtide import EngramTide


async def run_conversation(agent, user_inputs: list[str]) -> list[str]:
    engine = EngramTide(db_path="data/memories.db")
    session = engine.start_session()
    responses = []

    for user_input in user_inputs:
        prepared = await session.prepare_turn(user_input)

        # 四类记忆片段，可按宿主 Agent 的 prompt 格式自行组装。
        sections = prepared.prompt_sections()
        response = await agent.generate(
            user_input=user_input,
            memory_context=sections,
        )

        # 只有真正交给 Agent 使用的记忆才计访问；同一 turn 重试不会重复计数。
        session.acknowledge_used(prepared.turn_id)
        session.add_message("assistant", response)
        responses.append(response)

    # 整段对话结束后统一提取并写入新记忆。
    await session.end()
    engine.close()
    return responses
```

主要接口：

| 接口 | 作用 |
| --- | --- |
| `EngramTide.start_session()` | 执行衰减并收集本会话的自主浮现候选 |
| `session.prepare_turn()` | 语义激活、检索并按 Token 预算生成四类记忆上下文 |
| `session.acknowledge_used()` | 对宿主实际采用的检索/浮现记忆计访问，支持幂等重试 |
| `session.add_message()` | 记录宿主 Agent 的回复，供会话结束时提取记忆 |
| `session.end()` | 话题分割、脱水并执行类型感知写入 |
| `list_memories()` / `get_memory()` | 查看当前记忆 |
| `export_memories()` | 导出不含 embedding 的 JSON-compatible 数据 |
| `delete_memories()` | 硬删除记忆及其访问/衰减日志 |

Python API 当前采用单进程、单 SQLite 数据库模型；不要在同一进程中并行创建指向不同
数据库的多个 `EngramTide` 实例。`prepare_turn()` 会记录用户消息，宿主生成回复后应调用
`add_message("assistant", response)`。若宿主最终没有采用某些记忆，可以向
`acknowledge_used(turn_id, memory_ids=[...])` 只传实际使用的 ID。

## Streamable HTTP MCP Server

EngramTide 可以作为本机或可信局域网中的单用户 MCP Server 运行：

```powershell
python -m engramtide.mcp
```

默认端点为：

```text
http://127.0.0.1:8765/mcp
```

在支持 Streamable HTTP 的 MCP 客户端中添加以下服务器配置即可：

```json
{
  "mcpServers": {
    "engramtide": {
      "type": "streamable-http",
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

不同客户端对 `type` 字段的名称可能略有区别，连接 URL 保持不变。也可以使用 MCP
Inspector 连接该 URL 检查工具列表和调用结果。

推荐的 Agent 调用顺序：

1. 对话开始调用 `start_session`，保存返回的显式 `session_id`。
2. 每轮生成回复前调用 `prepare_turn(session_id, user_input, turn_id)`，将返回的四类
   `memory_context` 注入 Agent prompt。
3. 回复生成成功后调用 `commit_turn(session_id, turn_id, assistant_message)`；该调用确认
   实际使用的记忆并记录回复，重复提交同一 `turn_id` 不会重复计数或写入对话。
4. 正常结束时调用 `end_session`，执行话题分割、脱水和记忆写入；放弃会话时可调用
   `close_session`，后者不会提取新记忆。

MCP Server 还提供 `list_memories`、`export_memories` 和 `delete_memories` 管理工具。
`delete_memories` 是不可恢复的硬删除，宿主 Agent 应在调用前获得用户明确确认。

可选环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ENGRAMTIDE_MCP_HOST` | `127.0.0.1` | 监听地址；可信局域网可改为 `0.0.0.0` |
| `ENGRAMTIDE_MCP_PORT` | `8765` | 监听端口 |
| `ENGRAMTIDE_MCP_PATH` | `/mcp` | Streamable HTTP 端点路径 |
| `ENGRAMTIDE_MCP_ALLOWED_HOSTS` | 本机 Host | 非本机监听时必填，多个值用逗号分隔 |
| `ENGRAMTIDE_MCP_ALLOWED_ORIGINS` | 本机 Origin | 浏览器客户端跨域访问时按需配置 |
| `ENGRAMTIDE_DB_PATH` | 继承 `MEMORY_DB_PATH` | MCP 实例使用的 SQLite 文件 |
| `ENGRAMTIDE_SESSION_TTL_SECONDS` | `86400` | 空闲会话过期时间 |
| `ENGRAMTIDE_MAX_SESSIONS` | `32` | 同时保留的显式会话上限 |

此版本不提供 OAuth 或 Token 鉴权。默认仅绑定回环地址；只有在网络内所有设备均可信、
且有防火墙隔离时才应监听 `0.0.0.0`。此时必须将实际服务器地址（含端口）写入
`ENGRAMTIDE_MCP_ALLOWED_HOSTS`；浏览器型客户端还需配置 Origin。服务始终启用官方 SDK
的 DNS rebinding 防护。显式会话保存在 MCP Server 进程内，服务重启后应重新调用
`start_session`；已写入 SQLite 的长期记忆不受影响。不要直接将该端口暴露到公网。

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
├── engramtide/
│   ├── __init__.py        # 公共 API 导出
│   ├── api.py             # EngramTide / EngramTideSession Facade
│   └── mcp/               # Streamable HTTP MCP Server 与会话适配层
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
├── tests/                 # 单元 + 集成测试（286 项）
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

当前 **286 项测试全部通过**，覆盖：衰减数学、激活阈值边界、浮现筛选、写入管线、话题分割回退、双通道打分、Token 预算截断、会话上限、合并候选与端到端、Python API 与 MCP 会话/幂等/工具发现契约、功能开关回退等价性和核心流程回归。

## 技术栈

- **语言：** Python 3.10+
- **数据库：** SQLite（含 WAL 模式）
- **向量存储：** numpy float32 BLOB 存入 SQLite
- **LLM：** DeepSeek（OpenAI-compatible 接口）
- **Embedding：** 独立 OpenAI-compatible `/v1/embeddings` provider
- **关键词匹配：** rapidfuzz（缺失时自动降级纯向量）
- **Agent 协议：** MCP Streamable HTTP（官方 Python SDK）

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
