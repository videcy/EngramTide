# Memory Engine — Phase 2

AI 长期记忆系统，实现四类记忆（semantic / episodic / emotional / procedural）的差异化管理。

**核心闭环：** 对话 → 脱水写入记忆（类型感知管线） → 衰减与浮现 → 向量检索 + 浮现合并 → Constitutional AI 上下文 → 带记忆回复。

## Phase 2 新特性

- **类型感知衰减引擎**：semantic/procedural 永不衰减；episodic 艾宾浩斯衰减；emotional 唤醒度越高衰减越慢；访问越频繁越稳定。
- **主动浮现机制**：procedural 永驻浮现、高唤醒 emotional 浮现、unresolved 事项浮现、近期 episodic 浮现——不依赖用户输入，会话启动时自动激活。
- **写入管线**：semantic 覆盖检测（`superseded_by`）、emotional 强化写入（同情绪提升权重）、procedural 去重（防重复堆积）、episodic 直写。
- **会话间隔追踪**：`meta` 表持久化上次衰减时间，重启时自动计算距上次会话的小时数。
- **脱水质量升级**：分类判定规则、情感数值锚点、few-shot 示例、procedural 保守化约束。

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

Embedding provider 可以和聊天 provider 分离：

```powershell
# 不设置 EMBEDDING_API_KEY / EMBEDDING_BASE_URL 时，默认沿用 DeepSeek 配置
EMBEDDING_API_KEY=sk-your-embedding-key
EMBEDDING_BASE_URL=https://your-embedding-provider.example.com
EMBEDDING_MODEL=text-embedding-3-small
```

### 3. 启动

```powershell
python main.py
```

启动时会自动运行衰减更新并提取浮现记忆，输出一行衰减摘要日志。

### 4. 向后兼容

Phase 1 的旧数据库文件可直接使用。首次以 Phase 2 代码启动时，会自动补齐 `meta` 表和新索引，旧数据不丢失。首次启动不触发衰减（无基准时间）。

## 对话命令

| 命令 | 说明 |
| --- | --- |
| `/exit` `/quit` | 结束会话，触发脱水 + 写入管线，退出程序 |
| `/memories` | 打印数据库中最近 10 条记忆 |
| `/debug on` | 开启调试日志（检索分数、上下文注入、浮现记忆） |
| `/debug off` | 关闭调试日志 |
| `/debug decay` | 查看最近一次衰减摘要（更新/跳过/触底条数） |
| `/debug surface` | 查看当前浮现记忆列表 |

## 项目结构

```
memory-engine/
├── README.md
├── requirements.txt
├── config.py              # 配置与环境变量（含 Phase 2 衰减/浮现/写入参数）
├── main.py                # CLI 主循环（启动衰减→浮现→对话→写入管线）
├── core/
│   ├── memory_store.py    # SQLite 存储层（含 meta 表、衰减/强化/覆盖写入）
│   ├── embedding.py       # Embedding API 客户端
│   ├── dehydrator.py      # 会话脱水 → 记忆
│   ├── retriever.py       # 向量检索（decay_weight < 0.1 过滤）
│   ├── context_builder.py # Constitutional AI 上下文组装（检索+浮现合并）
│   ├── chat.py            # LLM 聊天调用
│   ├── decay.py           # ★ 衰减引擎 + 浮现机制（Phase 2 新增）
│   └── memory_writer.py   # ★ 类型感知写入管线（Phase 2 新增）
├── prompts/
│   ├── system_constitution.txt  # 交互宪法模板
│   └── dehydrate.txt            # 脱水 prompt（v2：分类规则 + 锚点 + few-shot）
├── utils/
│   ├── similarity.py      # 余弦相似度
│   └── time_utils.py      # ★ UTC 时间工具（Phase 2 新增）
├── tests/                 # 单元测试与集成测试（138 项）
└── data/                  # SQLite 数据库存放目录
```

## 核心架构

```
会话启动
  ├─ run_decay_update()      → 衰减引擎（纯数学，无 LLM）
  ├─ get_surfaced_memories()  → 主动浮现（R1>R2>R3>R4 优先级）
  └─ 输出衰减摘要日志

每轮对话
  ├─ embed_text(user_input)
  ├─ retrieve_memories()      → 向量检索（decay_weight < 0.1 过滤）
  ├─ build_constitutional_memory_context(retrieved, surfaced=...)
  │     └─ 检索 + 浮现 → 去重合并（浮现优先）→ 四类分组
  └─ generate_response()      → Constitutional AI 模板 → LLM

会话退出
  ├─ dehydrate_conversation() → LLM 脱水
  └─ write_memories()         → 写入管线
        ├─ semantic   → 覆盖检测 → mark_superseded
        ├─ emotional  → 强化检测 → reinforce_memory（不 insert）
        ├─ procedural → 去重检测 → mark_accessed（不 insert）
        └─ episodic   → 直接 insert
```

## 衰减公式

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
pytest -v
```

当前 **138 项测试全部通过**，覆盖：衰减数学、浮现筛选、写入管线、存储层扩展、上下文合并、检索过滤、时间工具、Phase 1 回归。

## 手动验收

```powershell
python tests/acceptance_phase2.py
```

覆盖 Phase 2 五大场景：semantic 覆盖、procedural 永驻浮现、emotional 强化、unresolved 浮现、差异化衰减。

## 技术栈

- **语言：** Python 3.10+
- **数据库：** SQLite（含 WAL 模式）
- **向量存储：** numpy float32 BLOB 存入 SQLite
- **LLM：** DeepSeek（OpenAI-compatible 接口）
- **Embedding：** 独立 OpenAI-compatible `/v1/embeddings` provider
