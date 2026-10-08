#!/usr/bin/env python3
"""
从 Markdown 日记导入记忆：先出计划给人审，再按计划写库。

日记格式：每天一个 ``YYYY-MM-DD.md``，用 ``## HH:MM 标题`` 分段（时间可省略）。
每段当作「我」写下的一段记录，交给脱水提示词提取记忆（会走 PROMPTS_OVERRIDE_DIR，
所以口吻跟随部署配置）。

两步走，写进库的内容与审过的计划完全一致：

    # 1. 出计划：调 LLM 提取，不写库。生成 plan.json 和便于阅读的 plan.md
    python scripts/import_diary.py plan --diary-dir ~/diary --out diary-plan.json

    # 2. 审完后写库：只做 embedding 和写入，不再调 LLM
    python scripts/import_diary.py apply diary-plan.json

写库时：
- created_at / last_accessed 设为日记当时的时间，近期浮现（R4）只认真正近期的记录；
- decay_weight 按「从那时到现在」预先衰减。运行中的衰减只按「距上次衰减多久」统一
  计算，不看每条记忆的年龄，不预先衰减的话旧事会以满权重进库；
- 超过 ``--unresolved-days``（默认 3 天）的段，unresolved 一律清掉：日记里的旧待办多半
  早已了结，留着会让它们按浮现规则 R3 每次会话都冒出来；
- source_conv_id = ``diary:YYYY-MM-DDTHH:MM``，tags 追加 ``diary``，导错了能整批定位；
- 已导入的段记在 meta 里，重复 apply 会跳过。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core import memory_store  # noqa: E402
from core.decay import compute_decay_multiplier  # noqa: E402
from core.dehydrator import _dehydrate_segment  # noqa: E402
from core.embedding import embed_text  # noqa: E402
from core.memory_store import Memory  # noqa: E402
from core.memory_writer import write_memories  # noqa: E402
from utils.time_utils import utc_now  # noqa: E402

_FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.md$")
_HEADING_RE = re.compile(r"^##\s+(?:(\d{1,2}:\d{2})\s*)?(.*)$")
_META_PREFIX = "import:"
_DB_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"  # 与 SQLite CURRENT_TIMESTAMP 一致（UTC）
DEFAULT_UNRESOLVED_DAYS = 3


@dataclass(frozen=True)
class DiaryEntry:
    source_id: str
    date: str
    time: str | None
    title: str
    body: str
    occurred_at: datetime  # aware UTC


def parse_diary_file(path: Path, tz: ZoneInfo) -> list[DiaryEntry]:
    """把一天的日记拆成段。无时间的段沿用上一段的时间，开头无时间则记为当天 12:00。"""
    match = _FILE_RE.match(path.name)
    if not match:
        return []
    date = match.group(1)

    sections: list[tuple[str | None, str, list[str]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            sections.append((heading.group(1), heading.group(2).strip(), []))
        elif sections:
            sections[-1][2].append(line)
        elif line.strip():
            sections.append((None, "", [line]))

    entries: list[DiaryEntry] = []
    last_time: str | None = None
    seen: dict[str, int] = {}
    for time, title, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        time = time or last_time
        last_time = time
        clock = time or "12:00"
        hour, minute = (int(x) for x in clock.split(":"))
        local = datetime.fromisoformat(date).replace(hour=hour, minute=minute, tzinfo=tz)
        base_id = f"diary:{date}T{clock if time else 'na'}"
        seen[base_id] = seen.get(base_id, 0) + 1
        source_id = base_id if seen[base_id] == 1 else f"{base_id}#{seen[base_id]}"
        entries.append(DiaryEntry(
            source_id=source_id,
            date=date,
            time=time,
            title=title,
            body=body,
            occurred_at=local.astimezone(timezone.utc),
        ))
    return entries


def load_entries(diary_dir: Path, tz: ZoneInfo) -> list[DiaryEntry]:
    entries: list[DiaryEntry] = []
    for path in sorted(diary_dir.iterdir()):
        if path.is_file():
            entries.extend(parse_diary_file(path, tz))
    return entries


def entry_message(entry: DiaryEntry) -> dict[str, str]:
    """日记是「我」写的，作为助手一侧的一条消息交给脱水。"""
    stamp = f"{entry.date} {entry.time or ''}".strip()
    header = f"【日记 {stamp}{' ' + entry.title if entry.title else ''}】"
    return {"role": "assistant", "content": f"{header}\n{entry.body}"}


def prior_weight(memory_type: str, arousal: float, occurred_at: datetime, now: datetime) -> float:
    hours = max(0.0, (now - occurred_at).total_seconds() / 3600)
    multiplier = compute_decay_multiplier(
        mem_type=memory_type, arousal=arousal, access_count=0, hours_elapsed=hours,
    )
    return max(config.DECAY_FLOOR, multiplier)


def keeps_unresolved(occurred_at: datetime, now: datetime, days: int) -> bool:
    """只有足够近的日记段才保留 unresolved。"""
    return now - occurred_at <= timedelta(days=days)


# ── plan ─────────────────────────────────────────────────


async def build_plan(entries: list[DiaryEntry], concurrency: int) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(entry: DiaryEntry) -> dict:
        record = {
            "source_id": entry.source_id,
            "date": entry.date,
            "time": entry.time,
            "title": entry.title,
            "occurred_at": entry.occurred_at.strftime(_DB_TIME_FORMAT),
            "chars": len(entry.body),
            "memories": [],
            "error": None,
        }
        async with semaphore:
            try:
                memories = await _dehydrate_segment([entry_message(entry)], entry.source_id)
            except Exception as exc:  # noqa: BLE001 — 单段失败不影响其余段
                record["error"] = str(exc)
                print(f"  ✗ {entry.source_id}: {exc}", file=sys.stderr)
                return record
        record["memories"] = [
            {
                "content": m.content,
                "type": m.type,
                "valence": m.valence,
                "arousal": m.arousal,
                "unresolved": m.unresolved,
                "tags": list(m.tags),
            }
            for m in memories
        ]
        print(f"  ✓ {entry.source_id}: {len(memories)} 条", file=sys.stderr)
        return record

    return await asyncio.gather(*(one(e) for e in entries))


def render_plan_markdown(plan: dict, unresolved_days: int = DEFAULT_UNRESOLVED_DAYS) -> str:
    now = utc_now()
    lines = [
        "# 日记导入计划（dry-run，尚未写库）",
        "",
        f"- 来源：`{plan['diary_dir']}`",
        f"- 脱水提示词：`{plan['prompt']}`",
        f"- 段数：{len(plan['entries'])}，提取记忆："
        f"{sum(len(e['memories']) for e in plan['entries'])} 条，"
        f"失败段：{sum(1 for e in plan['entries'] if e['error'])}",
        "- 「导入权重」= 按日记时间预先衰减后的 decay_weight；semantic / procedural 不衰减",
        f"- 超过 {unresolved_days} 天的段，「未解决」标记在写库时清除（显示为 ~~未解决~~）",
        "",
    ]
    by_type: dict[str, int] = {}
    for entry in plan["entries"]:
        for m in entry["memories"]:
            by_type[m["type"]] = by_type.get(m["type"], 0) + 1
    lines.append("类型分布：" + "，".join(f"{k} {v}" for k, v in sorted(by_type.items())))
    lines.append("")

    for entry in plan["entries"]:
        title = f"{entry['date']} {entry['time'] or ''} {entry['title']}".strip()
        lines.append(f"## {title}")
        lines.append(f"`{entry['source_id']}` · 原文 {entry['chars']} 字")
        lines.append("")
        if entry["error"]:
            lines.append(f"> ✗ 提取失败：{entry['error']}")
        elif not entry["memories"]:
            lines.append("> （没有提取到记忆）")
        occurred = datetime.strptime(entry["occurred_at"], _DB_TIME_FORMAT).replace(
            tzinfo=timezone.utc
        )
        for m in entry["memories"]:
            weight = prior_weight(m["type"], m["arousal"], occurred, now)
            flags = ""
            if m["unresolved"]:
                kept = keeps_unresolved(occurred, now, unresolved_days)
                flags = " · 未解决" if kept else " · ~~未解决~~"
            lines.append(
                f"- **{m['type']}** {m['content']}  "
                f"<sub>v={m['valence']:+.1f} a={m['arousal']:.1f} 导入权重 {weight:.2f}{flags}"
                f"{' · ' + '/'.join(m['tags']) if m['tags'] else ''}</sub>"
            )
        lines.append("")
    return "\n".join(lines)


async def cmd_plan(args: argparse.Namespace) -> int:
    tz = ZoneInfo(args.tz)
    diary_dir = Path(args.diary_dir).expanduser().resolve()
    entries = load_entries(diary_dir, tz)
    if args.since:
        entries = [e for e in entries if e.date >= args.since]
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        print("没有找到日记段落", file=sys.stderr)
        return 1

    print(f"共 {len(entries)} 段，开始提取（并发 {args.concurrency}）…", file=sys.stderr)
    records = await build_plan(entries, args.concurrency)
    plan = {
        "kind": "diary-import-plan",
        "generated_at": utc_now().strftime(_DB_TIME_FORMAT),
        "diary_dir": str(diary_dir),
        "timezone": args.tz,
        "prompt": str(config.prompt_path("dehydrate.txt")),
        "entries": records,
    }
    out = Path(args.out).expanduser()
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(render_plan_markdown(plan, args.unresolved_days), encoding="utf-8")
    total = sum(len(r["memories"]) for r in records)
    failed = sum(1 for r in records if r["error"])
    print(f"完成：{total} 条记忆，{failed} 段失败。计划 → {out}，可读版 → {md}", file=sys.stderr)
    return 0 if not failed else 2


# ── apply ────────────────────────────────────────────────


async def apply_plan(
    plan: dict,
    *,
    now: datetime | None = None,
    unresolved_days: int = DEFAULT_UNRESOLVED_DAYS,
) -> dict:
    """按计划写库。返回汇总；已导入的段跳过。"""
    now = now or utc_now()
    memory_store.init_db()
    summary = {"entries": 0, "skipped": 0, "failed_entries": 0,
               "inserted": 0, "superseded": 0, "reinforced": 0, "deduped": 0, "failed": 0}

    # 按时间顺序写：semantic 覆盖、emotional 强化都依赖「后来的事实更新」
    entries = sorted(plan["entries"], key=lambda e: (e["occurred_at"], e["source_id"]))
    for entry in entries:
        key = _META_PREFIX + entry["source_id"]
        if entry["error"]:
            summary["failed_entries"] += 1
            continue
        if memory_store.get_meta(key) is not None:
            summary["skipped"] += 1
            continue
        occurred = datetime.strptime(entry["occurred_at"], _DB_TIME_FORMAT).replace(
            tzinfo=timezone.utc
        )
        recent = keeps_unresolved(occurred, now, unresolved_days)
        memories = []
        for m in entry["memories"]:
            tags = list(dict.fromkeys([*m["tags"], "diary"]))
            memories.append(Memory(
                memory_id=str(uuid.uuid4()),
                type=m["type"],
                content=m["content"],
                valence=m["valence"],
                arousal=m["arousal"],
                created_at=entry["occurred_at"],
                last_accessed=entry["occurred_at"],
                decay_weight=prior_weight(m["type"], m["arousal"], occurred, now),
                embedding=await embed_text(m["content"]),
                source_conv_id=entry["source_id"],
                unresolved=m["unresolved"] and recent,
                tags=tags,
            ))
        report = await write_memories(memories)
        for field in ("inserted", "superseded", "reinforced", "deduped", "failed"):
            summary[field] += getattr(report, field)
        memory_store.set_meta(key, now.strftime(_DB_TIME_FORMAT))
        summary["entries"] += 1
    return summary


async def cmd_apply(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.plan).expanduser().read_text(encoding="utf-8"))
    if plan.get("kind") != "diary-import-plan":
        print("不是 import_diary.py 生成的计划文件", file=sys.stderr)
        return 1
    errors = config.check_config()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    summary = await apply_plan(plan, unresolved_days=args.unresolved_days)
    memory_store.close_db()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="调 LLM 提取记忆，生成待审计划，不写库")
    p_plan.add_argument("--diary-dir", required=True)
    p_plan.add_argument("--out", default="diary-import-plan.json")
    p_plan.add_argument("--tz", default="Asia/Shanghai", help="日记时间所在时区")
    p_plan.add_argument("--since", help="只处理该日期（YYYY-MM-DD）及之后的日记")
    p_plan.add_argument("--limit", type=int, help="只处理前 N 段（试跑用）")
    p_plan.add_argument("--concurrency", type=int, default=4)
    p_plan.add_argument("--unresolved-days", type=int, default=DEFAULT_UNRESOLVED_DAYS,
                        help="只有这么多天以内的段保留「未解决」标记（仅影响可读版展示）")

    p_apply = sub.add_parser("apply", help="把审过的计划写进记忆库")
    p_apply.add_argument("plan")
    p_apply.add_argument("--unresolved-days", type=int, default=DEFAULT_UNRESOLVED_DAYS,
                         help="只有这么多天以内的段保留「未解决」标记")

    args = parser.parse_args(argv)
    handler = cmd_plan if args.command == "plan" else cmd_apply
    return asyncio.run(handler(args))


if __name__ == "__main__":
    sys.exit(main())
