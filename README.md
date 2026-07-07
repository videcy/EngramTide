# Memory Engine — Phase 4

AI 长期记忆系统，实现四类记忆（semantic / episodic / emotional / procedural）的差异化管理。

**核心闭环：** 对话 → 话题分割 + 脱水写入记忆（类型感知管线） → 衰减与浮现 → 逐轮 Context-Aware 激活 → 双通道检索 + 浮现合并 → Token 预算截断 → Constitutional AI 上下文 → 带记忆回复。

## Phase 3 特性（Context-Aware 逐轮激活）

- **逐轮激活引擎**：每轮用户输入后、检索之前，按余弦相似度对 episodic/emotional 记忆做激活——强激活（sim > `SIMILARITY_HIGH`，+0.2 并计访问）/ 轻激活（sim > `SIMILARITY_MID`，+0.05 不计访问）。纯向量计算，零 LLM 调用。
- **休眠唤醒**：沉底记忆（decay_weight < 0.1，检索不到但未删除）被相关输入强激活后当轮立即重回检索候选池。
- **阈值真实校准**：`SIMILARITY_MID=0.72`、`SIMILARITY_HIGH=0.85`——由真实 embedding 分布定标（无关对 P95=0.588 < 0.72 < 相关对 P25=0.851），`check_config()` 强制 `MID < HIGH` 分层合法性。
- **总开关**：`CONTEXT_AWARE_ENABLED=false` 时行为与 Phase 2 完全一致（消融/回退）。

## Phase 4 特性

- **话题分割**：退出脱水前先调 LLM 按话题切分会话，每段独立脱水（embedding 语义更纯）；索引校验失败/LLM 失败整段回退，单段失败不牵连其余段（记忆零丢失）。
- **双通道检索**：`(0.7·向量 + 0.3·rapidfuzz 关键词) × decay_weight`，关键词通道兜住专有名词；`KEYWORD_CHANNEL_ENABLED=false` 或 rapidfuzz 缺失时降级纯向量（与 Phase 3 逐位一致）。
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

启动时会自动运行衰减更新并提取浮现记忆；活跃 episodic/emotional 超过 500 条时提示运行 `/consolidate preview`。

### 4. 消融开关（环境变量）

| 变量 | 默认 | 关闭后行为 |
| --- | --- | --- |
| `CONTEXT_AWARE_ENABLED` | true | 无逐轮激活（Phase 2 等价） |
| `TOPIC_SPLIT_ENABLED` | true | 整段脱水（Phase 3 等价） |
| `KEYWORD_CHANNEL_ENABLED` | true | 纯向量检索（Phase 3 等价） |
| `MILD_ONCE_PER_SESSION` | true | 轻激活无会话上限（Phase 3 等价） |

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
memory-engine/
├── README.md
├── requirements.txt
├── config.py              # 全部阈值与开关集中于此（Phase 2/3/4 参数分块）
├── main.py                # CLI 主循环（衰减→浮现→逐轮激活→检索→上下文→写入）
├── core/
│   ├── memory_store.py    # SQLite 存储层（哑 CRUD + meta 表）
│   ├── embedding.py       # Embedding API 客户端
│   ├── dehydrator.py      # ★ 话题分割 + 分段脱水（Phase 4 升级）
│   ├── retriever.py       # ★ 向量 + 关键词双通道检索（Phase 4 升级）
│   ├── context_builder.py # ★ Constitutional AI 上下文（Token 预算版，Phase 4）
│   ├── chat.py            # LLM 聊天调用
│   ├── decay.py           # 衰减引擎 + 浮现 + 逐轮激活（Phase 2/3）
│   ├── memory_writer.py   # 类型感知写入管线（Phase 2）
│   └── consolidator.py    # ★ 记忆合并/去重（Phase 4 新增）
├── prompts/
│   ├── system_constitution.txt  # 交互宪法模板
│   ├── dehydrate.txt            # 脱水 prompt
│   ├── topic_split.txt          # ★ 话题分割 prompt（Phase 4）
│   └── consolidate.txt          # ★ 记忆融合 prompt（Phase 4）
├── utils/
│   ├── similarity.py      # 余弦相似度
│   ├── time_utils.py      # UTC 时间工具（全项目禁用裸 datetime.now()）
│   └── token_counter.py   # ★ Token 保守估算器（Phase 4 新增）
├── tests/                 # 单元 + 集成测试（254 项）
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
sim > 0.85          → weight = min(1.0, weight + 0.2)，access_count + 1
0.72 < sim ≤ 0.85   → weight = min(1.0, weight + 0.05)，同会话每记忆至多 1 次
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

当前 **254 项测试全部通过**，覆盖：衰减数学、激活阈值边界、浮现筛选、写入管线、话题分割回退、双通道打分、Token 预算截断、会话上限、合并候选与端到端、消融等价性、Phase 1-3 回归。

对比实验（S1-S4 + 轻激活上限消融）见 `phase3_experiment_design.ipynb`；结论回填于 `plan/phase3|4/*_technical_plan.md` 附录。

## 技术栈

- **语言：** Python 3.10+
- **数据库：** SQLite（含 WAL 模式）
- **向量存储：** numpy float32 BLOB 存入 SQLite
- **LLM：** DeepSeek（OpenAI-compatible 接口）
- **Embedding：** 独立 OpenAI-compatible `/v1/embeddings` provider
- **关键词匹配：** rapidfuzz（缺失时自动降级纯向量）
