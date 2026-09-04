#!/usr/bin/env python3
"""
api_loop.py — optional server-side OpenAI-compatible loop for Companion Channel.

Run this beside backend/app.py when you want the VPS to answer directly via an
LLM API instead of routing every message to the Claude Code channel plugin.

Relay flow:
  PWA POST /relay/app/send
    -> relay stores the human message
    -> when /relay/app/brain == "loop", relay POSTs here: /loop/ingest
    -> this loop builds persona + same-session history + current message
    -> model answer is POSTed back to relay /channel/out

All private values live in env/.env. This file contains no domain, key, or
personal identity.
"""

# ═══════════════════════════════════════════════════════════════════════════
# 区块导航(按文件内出现顺序排列;要修改某功能时,按区块名搜索即可定位):
#   1. 环境与常量
#   2. 基础工具:路由/时间/脱敏/配置读写
#   3. 主动消息(proactive) · 配置与状态推导
#   4. 配置项:模型链/人设/注入/采样参数
#   5. 会话窗口(sessions)管理
#   6. 上下文构建:历史/附件/消息组装
#   7. 公开配置接口(读/写 loop_config)
#   8. relay 回写
#   9. 归一化层
#  10. MCP 客户端:工具发现与调用
#  11. 提示词工具模式(<tool_call> 文本协议)
#  12. 模型调用主入口:多模型 fallback
#  13. 主动消息(proactive) · 调度循环
#  14. 入站消息处理(handle_ingest)
#  15. FastAPI 路由:健康/配置/会话/聊天/调试
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import json
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request


# ── 环境与常量 ──────────────────────────────────────────────────────────────
def load_dotenv(path: Path) -> None:
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

LOOP_PORT = int(os.environ.get("LOOP_PORT", "3020"))
LOOP_CONFIG = Path(os.environ.get("LOOP_CONFIG", str(HERE / "api_loop.config.json")))
RELAY_DB = os.environ.get("RELAY_DB", str(HERE.parent / "backend" / "relay.db"))
RELAY_URL = os.environ.get("RELAY_URL", "http://127.0.0.1:3011").rstrip("/")
RELAY_SECRET = os.environ.get("RELAY_SECRET", "")
PERSONA_FILE = os.environ.get("PERSONA_FILE", "")
PERSONA = os.environ.get("PERSONA", "").strip()
HISTORY_N = int(os.environ.get("HISTORY_N", "24"))
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2000"))
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
STREAM_OUTPUT = os.environ.get("LOOP_STREAM", "1").lower() not in {"0", "false", "no"}
FALLBACK_CODES = {401, 403, 404, 408, 409, 429, 500, 502, 503, 504}
FORCE_NATIVE_TOOLS = os.environ.get("FORCE_NATIVE_TOOLS", "0").lower() in {"1", "true", "yes"}
# 逗号分隔的模型名列表:这些模型强制走提示词工具模式(<tool_call> 文本协议),
# 适用于不支持原生 tools 参数的中转端,免去每次先失败一次才自动切换。
PROMPT_TOOLS_FORCE = {m.strip() for m in os.environ.get("LLM_PROMPT_TOOLS", "").split(",") if m.strip()}

_TOOLS_UNSUPPORTED_ROUTES: set[tuple[str, str]] = set()
_THINKING_TOOLS_CONFLICT: set[tuple[str, str]] = set()
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_TAG_RE = re.compile(r"<tool_call\b.*?</tool_call>", re.DOTALL)

if not PERSONA and PERSONA_FILE:
    try:
        PERSONA = Path(PERSONA_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        PERSONA = ""
if not PERSONA:
    PERSONA = (
        "You are the user's private AI companion in a one-to-one chat. "
        "Reply naturally, warmly, and concisely unless the user asks for detail."
    )


# ── 基础工具:路由/时间/脱敏/配置读写 ─────────────────────────────────────────
def env_routes() -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for suffix in ("", "_2", "_3", "_4"):
        base = os.environ.get(f"LLM_API_BASE{suffix}", "").rstrip("/")
        key = os.environ.get(f"LLM_API_KEY{suffix}", "")
        model = os.environ.get(f"LLM_MODEL{suffix}", "")
        if base and key and model:
            entry: dict[str, Any] = {"url": base, "key": key, "model": model}
            extra_h = os.environ.get(f"LLM_API_HEADERS{suffix}", "")
            if extra_h:
                parsed: dict[str, str] = {}
                for line in extra_h.strip().split("\n"):
                    pair = line.strip()
                    if not pair or "=" not in pair:
                        continue
                    k, v = pair.split("=", 1)
                    parsed[k.strip()] = v.strip()
                if parsed:
                    entry["headers"] = parsed
            routes.append(entry)
    return routes


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def mask_key(key: str) -> str:
    key = str(key or "")
    if not key:
        return ""
    if len(key) <= 10:
        return "***"
    return key[:6] + "***" + key[-4:]


def load_config() -> dict[str, Any]:
    try:
        data = json.loads(LOOP_CONFIG.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def save_config(cfg: dict[str, Any]) -> None:
    LOOP_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    tmp = LOOP_CONFIG.with_suffix(LOOP_CONFIG.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(LOOP_CONFIG)


# ── 主动消息(proactive) · 配置与状态推导 ───────────────────────────────────
# AI 在用户沉默一段时间后,基于上下文主动发起一句自然的话。
# 所有设置存在 LOOP_CONFIG["proactive"],PWA 设置页可开关/调节;
# 运行状态(上次发送/今日条数)从 relay.db 的消息里推导,容器重启也不会重复轰炸。

PROACTIVE_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "tz": "Asia/Shanghai",        # 时间感知用:用户的本地时区(IANA 名称)
    "min_idle_hours": 3.0,        # 用户沉默超过这么久才会考虑主动发
    "cooldown_hours": 4.0,        # 成功发出一次后,至少隔这么久才允许下一条
    "quiet_enabled": True,        # 静默时段总开关:作息不固定时可直接关掉整段
    "quiet_start": "23:00",       # 静默时段开始(本地时间,此区间内不发)
    "quiet_end": "09:00",         # 静默时段结束(跨午夜=「23:00 后到次日 09:00 前」)
    "max_per_day": 3,             # 每天最多主动发几条
}

PROACTIVE_CHECK_SECONDS = 60
_PROACTIVE_BACKOFF: dict[str, Any] = {"until": 0.0, "note": ""}


def proactive_cfg() -> dict[str, Any]:
    raw = load_config().get("proactive")
    merged = dict(PROACTIVE_DEFAULTS)
    if isinstance(raw, dict):
        for key in PROACTIVE_DEFAULTS:
            if raw.get(key) is not None:
                merged[key] = raw[key]
    return merged


def local_now() -> dt.datetime:
    """用户本地当前时间(用于时间感知与静默时段判断)。"""
    tz_name = str(proactive_cfg().get("tz") or PROACTIVE_DEFAULTS["tz"])
    try:
        return dt.datetime.now(ZoneInfo(tz_name))
    except Exception:
        return dt.datetime.now(ZoneInfo(str(PROACTIVE_DEFAULTS["tz"])))


def brain_is_loop() -> bool:
    """当前 AI 大脑是 API loop 时才允许主动发消息(单身体原则:不要和桌面 channel 抢)。"""
    fallback = "loop" if os.environ.get("RELAY_DEFAULT_BRAIN", "loop") == "loop" else "desktop"
    brain_file = os.environ.get("RELAY_BRAIN_FILE", "")
    if brain_file:
        try:
            target = Path(brain_file).read_text(encoding="utf-8").strip()
            return target == "loop"
        except FileNotFoundError:
            pass
        except Exception:
            return fallback == "loop"
    return fallback == "loop"


def _db_fetch(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    path = Path(RELAY_DB)
    if not path.exists():
        return []
    try:
        with sqlite3.connect(str(path)) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
    except Exception:
        return []


def proactive_db_stats() -> dict[str, Any]:
    """从 relay.db 推导主动消息运行状态(跨重启持久)。"""
    stats: dict[str, Any] = {"last_user_ts": "", "last_proactive_ts": "", "today_count": 0}
    rows = _db_fetch("SELECT ts FROM messages WHERE direction = 'in' ORDER BY id DESC LIMIT 1")
    if rows:
        stats["last_user_ts"] = str(rows[0].get("ts") or "")
    rows = _db_fetch(
        "SELECT ts FROM messages WHERE direction = 'out' AND json_extract(meta, '$.proactive') = 1 "
        "ORDER BY id DESC LIMIT 1"
    )
    if rows:
        stats["last_proactive_ts"] = str(rows[0].get("ts") or "")
    try:
        start_local = local_now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_utc = start_local.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        rows = _db_fetch(
            "SELECT COUNT(*) AS n FROM messages WHERE json_extract(meta, '$.proactive') = 1 AND ts >= ?",
            (start_utc,),
        )
        if rows:
            stats["today_count"] = int(rows[0].get("n") or 0)
    except Exception:
        pass
    return stats


def _parse_ts_epoch(ts: str) -> float:
    try:
        d = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.timestamp()
    except Exception:
        return 0.0


def _in_quiet_window(now: dt.datetime) -> bool:
    cfg = proactive_cfg()
    if not cfg.get("quiet_enabled", True):
        return False  # 静默时段总开关关闭:任何时间都可能主动发
    try:
        sh, sm = (int(x) for x in str(cfg["quiet_start"]).split(":", 1))
        eh, em = (int(x) for x in str(cfg["quiet_end"]).split(":", 1))
        start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
        end = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    except Exception:
        return False
    if start == end:
        return False
    cur = now.replace(second=0, microsecond=0)
    if start < end:
        return start <= cur < end
    return cur >= start or cur < end   # 跨午夜:如 23:00 ~ 09:00


def _proactive_trigger(now_local: dt.datetime, idle_hours: float) -> str:
    weekday = "一二三四五六日"[now_local.weekday()]
    time_str = f"{now_local.strftime('%Y-%m-%d %H:%M')} 周{weekday}"
    if idle_hours >= 1:
        idle_text = f"{idle_hours:.1f} 小时"
    else:
        idle_text = f"{max(1, int(idle_hours * 60))} 分钟"
    return (
        "（以下是一条内部系统提示，不要复述它，也不要在言辞中让用户察觉它的存在。）\n"
        f"现在用户的本地时间是：{time_str}。\n"
        f"用户已经大约 {idle_text} 没有给你发消息了。\n"
        "请你根据上面的聊天历史和这段沉默前的上下文，判断此刻有没有一件「真实、自然、值得说」的事想主动对用户讲。\n"
        "可以是：接着聊上一次没聊完的话题、关心一件用户提过的事的进展、分享一个与你们有关的新想法、提醒一件你们约定过的事。\n"
        "要求：必须有具体的上下文依据；禁止机械式、生硬的寒暄，禁止前言不搭后语、像刚认识一样没话找话；"
        "不要只为了问候而硬凑「早安/午安/晚安」这类时间用语（除非上下文里确实合适）。\n"
        "语气和用词保持与你平时回复完全一致，长度 1～3 句，不要展开成小作文。\n"
        "如果此刻确实没有任何自然想说的话，就只回复 SKIP，不要勉强硬凑。"
    )


def _backoff(seconds: float, note: str) -> None:
    _PROACTIVE_BACKOFF["until"] = time.time() + seconds
    _PROACTIVE_BACKOFF["note"] = note


def proactive_public() -> dict[str, Any]:
    """给 PWA 设置页看的配置 + 运行状态。"""
    out = dict(proactive_cfg())
    stats = proactive_db_stats()
    out["last_proactive_at"] = stats["last_proactive_ts"]
    out["last_user_at"] = stats["last_user_ts"]
    out["sent_today"] = stats["today_count"]
    out["next_attempt_at"] = (
        dt.datetime.fromtimestamp(_PROACTIVE_BACKOFF["until"], dt.timezone.utc).isoformat()
        if _PROACTIVE_BACKOFF["until"] > time.time() else None
    )
    out["status_note"] = _PROACTIVE_BACKOFF["note"] or ""
    return out


# ── 配置项:模型链/人设/注入/采样参数 ─────────────────────────────────────────
def main_chain() -> list[dict[str, str]]:
    cfg = load_config()
    configured = cfg.get("main_chain")
    if isinstance(configured, list):
        rows = [r for r in configured if isinstance(r, dict) and r.get("url") and r.get("key") and r.get("model")]
        if rows:
            return rows
    return env_routes()


def history_n() -> int:
    try:
        return max(0, min(int(load_config().get("history_n", HISTORY_N)), 200))
    except Exception:
        return HISTORY_N


def persona() -> str:
    cfg = load_config()
    p = str(cfg.get("persona") or "").strip()
    if not p and PERSONA_FILE:
        try:
            p = Path(PERSONA_FILE).read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return p or PERSONA

def ai_name() -> str:
    cfg = load_config()
    return str(cfg.get("ai_name") or "").strip()

def injections() -> tuple[bool, list[dict[str, str]]]:
    """(注入是否生效, 条目列表) — 用户手动维护的「指令注入包」。

    总开关开启且至少有一条有内容的条目时，注入到每次请求的 system 提示里。
    条目格式: [{"title": 可选标题, "content": 必填内容}, ...]
    """
    cfg = load_config()
    inj = cfg.get("injections")
    if not isinstance(inj, dict):
        return False, []
    rows: list[dict[str, str]] = []
    for e in (inj.get("entries") or []):
        if not isinstance(e, dict):
            continue
        content = str(e.get("content") or "").strip()
        if not content:
            continue
        rows.append({"title": str(e.get("title") or "").strip(), "content": content})
    if not bool(inj.get("enabled")) or not rows:
        return False, rows
    return True, rows

def temperature() -> float:
    try:
        return float(load_config().get("temperature", TEMPERATURE))
    except Exception:
        return TEMPERATURE

def top_p() -> float | None:
    cfg = load_config()
    v = cfg.get("top_p")
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None

def max_tokens() -> int | None:
    """None = 不传 max_tokens，跟随当前模型的默认输出上限(长文本/MCP 场景推荐)。"""
    try:
        v = load_config().get("max_tokens", MAX_TOKENS)
        if v is None or v == "":
            return None
        return max(100, min(131072, int(v)))
    except Exception:
        return MAX_TOKENS

def thinking_budget() -> int:
    try:
        return max(0, int(load_config().get("thinking_budget", 0)))
    except Exception:
        return 0


# ── 会话窗口(sessions)管理 ─────────────────────────────────────────────────
def session_rows() -> list[dict[str, Any]]:
    rows = load_config().get("sessions")
    if not isinstance(rows, list):
        return []
    out = []
    for item in rows:
        if isinstance(item, dict) and item.get("id"):
            out.append({
                "id": str(item.get("id")),
                "title": str(item.get("title") or "New chat"),
                "since_id": int(item.get("since_id") or 0),
                "created_at": item.get("created_at") or "",
                "pinned": bool(item.get("pinned", False)),
            })
    out.sort(key=lambda r: 0 if r.get("pinned") else 1)  # 置顶的排前面(稳定排序,其余保持原序)
    return out


def active_session_id() -> str:
    cfg = load_config()
    active = str(cfg.get("active_session") or "").strip()
    ids = {s["id"] for s in session_rows()}
    if active in ids:
        return active
    rows = session_rows()
    return rows[-1]["id"] if rows else ""


def save_sessions(rows: list[dict[str, Any]], active: str | None = None) -> dict[str, Any]:
    cfg = load_config()
    cfg["sessions"] = rows
    if active is not None:
        cfg["active_session"] = active
    save_config(cfg)
    return sessions_public()


def sessions_public() -> dict[str, Any]:
    return {"active_session": active_session_id(), "sessions": session_rows()}


def create_session(title: str = "New chat", since_id: int = 0, activate: bool = True) -> dict[str, Any]:
    rows = session_rows()
    sid = "api-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    row = {"id": sid, "title": title or "New chat", "since_id": int(since_id or 0), "created_at": now_iso()}
    rows.append(row)
    save_sessions(rows, sid if activate else None)
    return row


def patch_session(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
    rows = session_rows()
    found = False
    for item in rows:
        if item["id"] != session_id:
            continue
        found = True
        if "title" in body:
            item["title"] = str(body.get("title") or item["title"]).strip() or item["title"]
        if "pinned" in body:
            item["pinned"] = bool(body.get("pinned"))
    if not found:
        raise HTTPException(status_code=404, detail="session not found")
    active = session_id if body.get("active") else None
    return save_sessions(rows, active)


def delete_session(session_id: str) -> dict[str, Any]:
    rows = session_rows()
    was_active = active_session_id() == session_id
    remaining = [r for r in rows if r["id"] != session_id]
    if len(remaining) == len(rows):
        raise HTTPException(status_code=404, detail="session not found")
    # 删除的是当前窗口时,自动切到最新一个剩余窗口;全删光则回到无会话状态。
    active = (remaining[-1]["id"] if remaining else "") if was_active else None
    return save_sessions(remaining, active)


# ── 上下文构建:历史/附件/消息组装 ───────────────────────────────────────────
def relay_rows(before_id: int | None, session_id: str, limit: int) -> list[dict[str, Any]]:
    path = Path(RELAY_DB)
    if not path.exists():
        return []
    params: list[Any] = []
    where = ["kind IN ('user','voice','reply')"]
    if before_id:
        where.append("id < ?")
        params.append(int(before_id))
    if session_id:
        where.append("json_extract(meta, '$.api_session') = ?")
        params.append(session_id)
    else:
        where.append("(json_extract(meta, '$.api_session') IS NULL OR json_extract(meta, '$.api_session') = '')")
    sql = (
        "SELECT id, direction, kind, text, meta FROM messages "
        f"WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT ?"
    )
    params.append(max(0, limit))
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in reversed(rows)]


async def fetch_attachment_data_url(att: dict[str, Any]) -> str | None:
    """从 relay 下载图片附件并转成 data URL;非图片或下载失败返回 None。"""
    url = str(att.get("url") or "").strip()
    mime = str(att.get("mime") or "").strip()
    if not url or not mime.startswith("image/"):
        return None
    full = url if url.startswith("http") else f"{RELAY_URL}{url}"
    if RELAY_SECRET:
        full += ("&" if "?" in full else "?") + "token=" + RELAY_SECRET
    try:
        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            resp = await client.get(full)
            if resp.status_code >= 400:
                print(f"[api_loop:image] download HTTP {resp.status_code}: {full}")
                return None
            return f"data:{mime};base64,{base64.b64encode(resp.content).decode('ascii')}"
    except Exception as exc:
        print(f"[api_loop:image] download failed ({type(exc).__name__}: {exc}): {full}")
        return None


async def attachment_parts(atts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """附件列表 → 多模态 content 片段(图片转 data URL;其他附件降级为文字提示)。"""
    parts: list[dict[str, Any]] = []
    if not atts:
        return parts
    notes: list[str] = []
    for att in atts:
        data_url = await fetch_attachment_data_url(att)
        if data_url:
            parts.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            name = str(att.get("name") or "").strip()
            if name:
                notes.append(name)
    if notes:
        parts.insert(0, {"type": "text", "text": "[附件]" + "、".join(notes)})
    return parts


def build_messages(text: str, *, before_id: int | None = None, session_id: str = "", use_context: bool = True, image_parts: list[dict[str, Any]] | None = None, warm_block: str = "") -> list[dict[str, Any]]:
    tool_hint = " When a configured MCP tool can provide current, external, or actionable information, use it before answering."
    system_text = persona() + tool_hint
    inj_on, inj_rows = injections()
    if inj_on and inj_rows:
        blocks: list[str] = []
        for r in inj_rows:
            head = f"【{r['title']}】\n" if r["title"] else ""
            blocks.append((head + r["content"]).strip())
        system_text += (
            "\n\nThe following user-injected rules are currently active and must be followed:\n\n"
            + "\n\n".join(blocks)
        )
    if warm_block:
        system_text += warm_block
    messages = [{"role": "system", "content": system_text}]
    if use_context:
        for row in relay_rows(before_id, session_id, history_n()):
            content = str(row.get("text") or "").strip()
            if not content:
                continue
            role = "assistant" if row.get("direction") == "out" else "user"
            messages.append({"role": role, "content": content})
    if image_parts:
        content: list[dict[str, Any]] = [{"type": "text", "text": text or "（用户发来一张图片，请查看。）"}]
        content.extend(image_parts)
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": text})
    return messages


# ── 公开配置接口(读/写 loop_config) ────────────────────────────────────────
def mcp_servers() -> list[dict[str, Any]]:
    rows = load_config().get("mcp_servers")
    if not isinstance(rows, list):
        return []
    return [
        {"name": str(r.get("name") or "server"), "url": str(r.get("url") or "").rstrip("/"),
         "token": str(r.get("token") or ""), "enabled": bool(r.get("enabled", True))}
        for r in rows if isinstance(r, dict) and r.get("url")
    ]


def public_config() -> dict[str, Any]:
    cfg = load_config()
    return {
        "history_n": history_n(),
        "persona": cfg.get("persona", ""),
        "ai_name": cfg.get("ai_name", ""),
        "ai_avatar": str(cfg.get("ai_avatar") or ""),
        "temperature": cfg.get("temperature", TEMPERATURE),
        "top_p": cfg.get("top_p", None),
        "max_tokens": cfg.get("max_tokens", MAX_TOKENS),
        "thinking_budget": cfg.get("thinking_budget", 0),
        "gateway_session_id": gateway_session_id(),
        "warm_enabled": bool(warm_cfg().get("enabled", False)),
        "injections": cfg.get("injections") or {"enabled": False, "entries": []},
        "proactive": proactive_public(),
        "active_session": active_session_id(),
        "sessions": session_rows(),
        "main_chain": [
            {
                "index": i,
                "model": r.get("model", ""),
                "url": r.get("url", ""),
                "key_masked": mask_key(r.get("key", "")),
                "headers": (r.get("headers") or None),
            }
            for i, r in enumerate(main_chain())
        ],
        "mcp_servers": [
            {"index": i, "name": r["name"], "url": r["url"], "token_masked": mask_key(r["token"]), "enabled": r["enabled"]}
            for i, r in enumerate(mcp_servers())
        ],
    }


def update_config(body: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config()
    if "history_n" in body:
        cfg["history_n"] = max(0, min(int(body.get("history_n") or 0), 200))
    if "persona" in body:
        cfg["persona"] = str(body.get("persona") or "").strip()
    if "ai_name" in body:
        cfg["ai_name"] = str(body.get("ai_name") or "").strip()
    if "ai_avatar" in body:
        # 只收小尺寸的 data:image/ dataURL(前端已压到 ~512px);非法或过大则清空
        raw = str(body.get("ai_avatar") or "")
        cfg["ai_avatar"] = raw if (raw.startswith("data:image/") and len(raw) <= 3_000_000) else ""
    if "temperature" in body:
        try:
            cfg["temperature"] = max(0.0, min(2.0, float(body["temperature"])))
        except Exception:
            pass
    if "top_p" in body:
        try:
            cfg["top_p"] = max(0.0, min(1.0, float(body["top_p"])))
        except Exception:
            pass
    if "max_tokens" in body:
        v = body.get("max_tokens")
        if v is None or v == "":
            cfg["max_tokens"] = None  # 自动：不传参数，跟随模型默认上限
        else:
            try:
                cfg["max_tokens"] = max(100, min(131072, int(v)))
            except Exception:
                pass
    if "thinking_budget" in body:
        try:
            cfg["thinking_budget"] = max(0, min(32768, int(body["thinking_budget"])))
        except Exception:
            pass
    if "gateway_session_id" in body:
        cfg["gateway_session_id"] = str(body.get("gateway_session_id") or "").strip()
    if "warm_enabled" in body:
        wl = cfg.get("warm_layer")
        wl = dict(wl) if isinstance(wl, dict) else {}
        wl["enabled"] = bool(body.get("warm_enabled"))
        cfg["warm_layer"] = wl
    if "injections" in body:
        inj = body.get("injections")
        if isinstance(inj, dict):
            entries: list[dict[str, str]] = []
            for e in (inj.get("entries") or []):
                if not isinstance(e, dict):
                    continue
                title = str(e.get("title") or "").strip()
                content = str(e.get("content") or "").strip()
                if title or content:
                    entries.append({"title": title, "content": content})
            cfg["injections"] = {"enabled": bool(inj.get("enabled")), "entries": entries}
    if isinstance(body.get("main_chain"), list):
        old = main_chain()
        new_chain = []
        for pos, item in enumerate(body["main_chain"]):
            if not isinstance(item, dict):
                continue
            old_idx = int(item.get("index", pos) or 0)
            prev = old[old_idx] if 0 <= old_idx < len(old) else {}
            entry: dict[str, Any] = {
                "model": str(item.get("model") or prev.get("model") or "").strip(),
                "url": str(item.get("url") or prev.get("url") or "").strip().rstrip("/"),
                "key": str(item.get("key") or prev.get("key") or ""),
            }
            if "headers" in item:
                raw_headers = item.get("headers")
                if isinstance(raw_headers, dict) and raw_headers:
                    cleaned = {str(k).strip(): str(v).strip() for k, v in raw_headers.items() if str(k).strip() and str(v).strip()}
                    if cleaned:
                        entry["headers"] = cleaned
            elif prev.get("headers"):
                entry["headers"] = dict(prev["headers"])
            if not (entry["model"] and entry["url"] and entry["key"]):
                raise HTTPException(status_code=400, detail=f"row {pos + 1}: model/url/key required")
            new_chain.append(entry)
        if new_chain:
            cfg["main_chain"] = new_chain
    if isinstance(body.get("mcp_servers"), list):
        old = mcp_servers()
        new_servers = []
        for pos, item in enumerate(body["mcp_servers"]):
            if not isinstance(item, dict):
                continue
            old_idx = int(item.get("index", pos) or 0)
            prev = old[old_idx] if 0 <= old_idx < len(old) else {}
            entry = {
                "name": str(item.get("name") or prev.get("name") or f"server-{pos + 1}").strip(),
                "url": str(item.get("url") or prev.get("url") or "").strip().rstrip("/"),
                "token": str(item.get("token") or prev.get("token") or ""),
                "enabled": bool(item.get("enabled", prev.get("enabled", True))),
                "disabled_tools": list(item.get("disabled_tools") or prev.get("disabled_tools") or []),
            }
            if not entry["url"]:
                raise HTTPException(status_code=400, detail=f"MCP row {pos + 1}: url required")
            new_servers.append(entry)
        cfg["mcp_servers"] = new_servers
    if isinstance(body.get("proactive"), dict):
        p = body["proactive"]
        cur = proactive_cfg()
        if "enabled" in p:
            cur["enabled"] = bool(p["enabled"])
            if not cur["enabled"]:
                _PROACTIVE_BACKOFF["until"] = 0.0
                _PROACTIVE_BACKOFF["note"] = ""
        if "tz" in p:
            tz = str(p.get("tz") or "").strip()
            if tz:
                try:
                    ZoneInfo(tz)
                    cur["tz"] = tz
                except Exception:
                    pass
        if "min_idle_hours" in p:
            try:
                cur["min_idle_hours"] = max(0.5, min(72.0, float(p["min_idle_hours"])))
            except Exception:
                pass
        if "cooldown_hours" in p:
            try:
                cur["cooldown_hours"] = max(1.0, min(168.0, float(p["cooldown_hours"])))
            except Exception:
                pass
        if "quiet_enabled" in p:
            cur["quiet_enabled"] = bool(p["quiet_enabled"])
        if "quiet_start" in p:
            v = str(p.get("quiet_start") or "")
            if re.fullmatch(r"\d{1,2}:\d{2}", v):
                cur["quiet_start"] = v
        if "quiet_end" in p:
            v = str(p.get("quiet_end") or "")
            if re.fullmatch(r"\d{1,2}:\d{2}", v):
                cur["quiet_end"] = v
        if "max_per_day" in p:
            try:
                cur["max_per_day"] = max(0, min(20, int(p["max_per_day"])))
            except Exception:
                pass
        cur["cooldown_hours"] = max(cur["cooldown_hours"], cur["min_idle_hours"])
        cfg["proactive"] = cur
    save_config(cfg)
    return public_config()


# ── relay 回写 ─────────────────────────────────────────────────────────────
async def relay_out(payload: dict[str, Any]) -> tuple[bool, Any]:
    if not RELAY_SECRET:
        return False, "RELAY_SECRET missing"
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        resp = await client.post(
            f"{RELAY_URL}/channel/out",
            headers={"Authorization": f"Bearer {RELAY_SECRET}", "Content-Type": "application/json"},
            json=payload,
        )
    try:
        body: Any = resp.json()
    except Exception:
        body = resp.text[:500]
    return resp.status_code < 300, body


# ── 归一化层 ──────────────────────────────────────────────────────────────
# 目标：PWA/relay 只认一套固定格式，任何一家 LLM API 的字段差异都在这里被
# “翻译”成统一字段。以后接入新 API，只需要在下面两处补对应字段名，PWA 不用改。

def normalize_usage(raw: Any) -> dict[str, Any]:
    """把各家 usage 字段名统一成 {input_tokens, output_tokens, total_tokens}。

    坑：OpenAI/DeepSeek/GLM/Qwen 系返回 prompt_tokens/completion_tokens，
    Anthropic/Gemini 兼容端点返回 input_tokens/output_tokens——之前此处“透传”，
    PWA 只认后者，于是 OpenAI 系路由的 tok 数全变成 0、前端一个都不显示。
    """
    if not isinstance(raw, dict):
        return {}

    def num(*keys: str) -> int | None:
        for k in keys:
            v = raw.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        return None

    norm: dict[str, Any] = {}
    inp = num("input_tokens", "prompt_tokens")
    outp = num("output_tokens", "completion_tokens")
    total = num("total_tokens")
    if inp is not None:
        norm["input_tokens"] = inp
    if outp is not None:
        norm["output_tokens"] = outp
    if total is not None:
        norm["total_tokens"] = total
    return norm

def normalize_stream_event(ev: dict[str, Any]) -> dict[str, Any]:
    """把一次 SSE 流事件翻译成中立格式。

    返回: {"content", "thinking", "tool_calls", "usage", "finish_reason"}
      - content:      正文增量(字符串,可能为空)
      - thinking:     思考/推理增量(字符串,可能为空)
      - tool_calls:   工具调用增量(list),每项为
                      {"index","id","name","arguments"}; index<0 表示独立一次调用
      - usage:        归一化为 {"input_tokens","output_tokens","total_tokens"}
      - finish_reason: 透传
    """
    usage = normalize_usage(ev.get("usage"))
    choice = (ev.get("choices") or [{}])[0]
    delta = choice.get("delta") or {}
    finish_reason = choice.get("finish_reason")

    # Anthropic 兼容端点会用 delta.type=="thinking" 标记思考块;
    # 此时 delta.content 属于思考,不能算作正文。
    is_thinking_block = delta.get("type") == "thinking"
    content = "" if is_thinking_block else (delta.get("content") or "")

    # 思考字段：各家命名不同，统一归到 thinking
    thinking = ""
    if delta.get("reasoning_content"):
        thinking = delta["reasoning_content"]          # DeepSeek / Qwen / GLM
    elif delta.get("reasoning"):
        thinking = delta["reasoning"]                  # OpenRouter / 部分中转(单数命名)
    elif is_thinking_block:
        thinking = delta.get("thinking") or delta.get("content") or ""
    elif delta.get("thinking"):
        thinking = delta["thinking"]                   # OpenAI 扩展
    elif ev.get("thinking"):
        thinking = ev["thinking"]                      # 顶层自定义

    # 工具调用：两种主要形态统一归到 tool_calls
    tool_calls: list[dict[str, Any]] = []
    if delta.get("tool_calls"):
        for tc in delta["tool_calls"]:                 # OpenAI 流式(分段下发 name/arguments)
            fn = tc.get("function") or {}
            tool_calls.append({
                "index": tc.get("index", 0),
                "id": tc.get("id") or "",
                "name": fn.get("name") or "",
                "arguments": fn.get("arguments") or "",
            })
    elif delta.get("type") == "tool_use":              # Anthropic:一次给全 name + input
        tool_calls.append({
            "index": -1,
            "id": delta.get("id") or "",
            "name": delta.get("name") or "",
            "arguments": json.dumps(delta.get("input") or {}, ensure_ascii=False),
        })

    return {
        "content": content,
        "thinking": thinking,
        "tool_calls": tool_calls,
        "role": delta.get("role") or "",
        "usage": usage,
        "finish_reason": finish_reason,
    }


def accumulate_tool_calls(buf: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> None:
    """把增量 tool_calls 合并进缓冲,按 index 对齐(OpenAI 分段)或独立追加(Anthropic)。"""
    for tc in incoming:
        idx = tc.get("index", 0)
        if isinstance(idx, int) and idx >= 0:
            while len(buf) <= idx:
                buf.append({"id": "", "name": "", "arguments_buf": ""})
            slot = buf[idx]
            if tc.get("id"):
                slot["id"] = tc["id"]
            if tc.get("name"):
                slot["name"] = tc["name"]
            if tc.get("arguments"):
                slot["arguments_buf"] += tc["arguments"]
        else:
            buf.append({
                "id": tc.get("id") or "",
                "name": tc.get("name") or "",
                "arguments_buf": tc.get("arguments") or "",
            })


def finalize_tool_calls(buf: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把缓冲转成两套统一结构:(供 PWA/relay 展示的 meta 结构, 供回填模型的 raw 结构)。"""
    parsed: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for tc in buf:
        name = (tc.get("name") or "").strip()
        if not name:
            continue  # 跳过流式解析产生的空 name 幽灵 tool_call
        try:
            args = json.loads(tc.get("arguments_buf") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        parsed.append({"name": name, "input": args})
        raw.append({
            "id": tc.get("id", ""),
            "type": "function",
            "function": {"name": name, "arguments": tc.get("arguments_buf", "{}")},
        })
    return parsed, raw


def merge_thinking(parts: list[str]) -> list[dict[str, Any]] | None:
    merged = "".join(parts).strip()
    return [{"content": merged}] if merged else None


# ── 思考增量实时转发器 ────────────────────────────────────────────────────
# 把模型流式返回的 thinking 增量攒批推给 relay(relay 再扇出给 PWA 实时渲染)。
# 作用双份:① PWA 能实时看到思考链(而不是等最终回复包);② 长时间思考期间
# PWA 持续收到事件,不会误判超时/掉线。
class _DeltaEmitter:
    def __init__(self, stream_id: str, session_id: str, kind: str = "thinking"):
        self.stream_id = stream_id
        self.session_id = session_id
        self.kind = kind
        self.buf = ""
        self.sent = False
        self._timer: asyncio.Task | None = None

    async def feed(self, chunk: str) -> None:
        chunk = str(chunk or "")
        if not chunk:
            return
        self.buf += chunk
        if len(self.buf) >= 256:
            await self.flush()
        elif self._timer is None:
            self._timer = asyncio.create_task(self._delayed_flush())

    async def _delayed_flush(self) -> None:
        await asyncio.sleep(0.4)
        self._timer = None
        if self.buf:
            await self.flush()

    async def flush(self) -> None:
        if not self.buf:
            return
        chunk, self.buf = self.buf, ""
        try:
            ok, body = await relay_out({
                "type": f"{self.kind}_delta",
                "stream_id": self.stream_id,
                "text": chunk,
                "done": False,
                "api_session": self.session_id,
            })
            if ok:
                self.sent = True
            else:
                print(f"[api_loop:stream] {self.kind} delta push failed: {str(body)[:120]}")
        except Exception as exc:
            print(f"[api_loop:stream] {self.kind} delta push error: {type(exc).__name__}: {exc}")

    async def close(self) -> None:
        """flush 剩余缓冲,并补一个 done 帧让 relay 把思考消息落库。"""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        await self.flush()
        if self.sent:
            try:
                await relay_out({
                    "type": f"{self.kind}_delta",
                    "stream_id": self.stream_id,
                    "done": True,
                    "api_session": self.session_id,
                })
            except Exception:
                pass


async def stream_chat(route: dict[str, Any], messages: list[dict[str, str]], sink, think_sink=None) -> dict[str, Any]:
    body = {
        "model": route["model"],
        "messages": messages,
        "temperature": _route_temperature(route),
        "stream": True,
        # 流式模式下 usage 默认不下发;显式打开,最后一帧才带 token 统计
        "stream_options": {"include_usage": True},
    }
    mt = max_tokens()
    if mt is not None:
        body["max_tokens"] = mt
    tp = top_p()
    if tp is not None:
        body["top_p"] = tp
    budget = thinking_budget()
    if budget > 0:
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    thinking_parts: list[str] = []
    tool_calls_buf: list[dict[str, Any]] = []
    req_headers = {"Authorization": f"Bearer {route['key']}", "Content-Type": "application/json"}
    for hk, hv in (route.get("headers") or {}).items():
        if str(hk) and str(hv):
            req_headers[str(hk)] = str(hv)
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        async with client.stream(
            "POST",
            route["url"].rstrip("/") + "/chat/completions",
            headers=req_headers,
            json=body,
        ) as resp:
            if resp.status_code >= 400:
                err_detail = ""
                try:
                    lines = [line async for line in resp.aiter_lines()]
                    body_text = "\n".join(lines)[:500]
                    err_detail = body_text or str(resp.status_code)
                except Exception:
                    err_detail = str(resp.status_code)
                raise HTTPException(status_code=max(resp.status_code, 400), detail=err_detail)
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                n = normalize_stream_event(ev)
                if n["usage"]:
                    usage = n["usage"]
                if n["content"]:
                    text_parts.append(n["content"])
                    await sink(n["content"])
                if n["thinking"]:
                    thinking_parts.append(n["thinking"])
                    if think_sink:
                        try:
                            await think_sink(n["thinking"])
                        except Exception:
                            pass
                if n["tool_calls"]:
                    accumulate_tool_calls(tool_calls_buf, n["tool_calls"])
    final_tool_calls, _ = finalize_tool_calls(tool_calls_buf)
    final_text = "".join(text_parts).strip()
    return {
        "text": final_text,
        "usage": usage,
        "thinking": merge_thinking(thinking_parts),
        "tool_calls": final_tool_calls if final_tool_calls else None
    }


# ── MCP 客户端:工具发现与调用 ───────────────────────────────────────────────
MCP_SESSIONS: dict[str, str] = {}
MCP_REQUEST_ID = 0


def mcp_next_id() -> int:
    global MCP_REQUEST_ID
    MCP_REQUEST_ID += 1
    return MCP_REQUEST_ID


def mcp_response_body(resp: httpx.Response) -> dict[str, Any]:
    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        data = resp.json()
        if isinstance(data, dict):
            return data
        raise RuntimeError("MCP returned a non-object response")
    events = []
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            raw = line[5:].strip()
            if raw:
                events.append(json.loads(raw))
    if not events:
        raise RuntimeError("MCP returned an empty event stream")
    return events[-1]


async def mcp_call(server: dict[str, Any], method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = server["url"]
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if server.get("token"):
        headers["Authorization"] = f"Bearer {server['token']}"
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        if method != "initialize" and url not in MCP_SESSIONS:
            initialize = {
                "jsonrpc": "2.0",
                "id": mcp_next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "relay-ai-chat", "version": "1.0.0"},
                },
            }
            init_resp = await client.post(url, headers=headers, json=initialize)
            if init_resp.status_code < 300:
                init_data = mcp_response_body(init_resp)
                if init_data.get("error"):
                    raise RuntimeError(str(init_data["error"]))
                session_id = init_resp.headers.get("mcp-session-id")
                if session_id:
                    MCP_SESSIONS[url] = session_id
                notify_headers = dict(headers)
                if session_id:
                    notify_headers["Mcp-Session-Id"] = session_id
                await client.post(url, headers=notify_headers, json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        if url in MCP_SESSIONS:
            headers["Mcp-Session-Id"] = MCP_SESSIONS[url]
        request = {"jsonrpc": "2.0", "id": mcp_next_id(), "method": method, "params": params or {}}
        resp = await client.post(url, headers=headers, json=request)
    resp.raise_for_status()
    data = mcp_response_body(resp)
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return data.get("result") or {}


async def mcp_tools() -> list[dict[str, Any]]:
    tools = []
    for server in mcp_servers():
        if not server["enabled"]:
            continue
        disabled = set(server.get("disabled_tools") or [])
        try:
            result = await mcp_call(server, "tools/list")
            for tool in result.get("tools", []):
                if isinstance(tool, dict) and tool.get("name"):
                    raw_name = str(tool["name"])
                    if raw_name in disabled:
                        continue
                    tools.append({"type": "function", "function": {"name": f"mcp_{server['name']}_{raw_name}", "description": tool.get("description", ""), "parameters": tool.get("inputSchema") or {"type": "object"}}})
        except Exception:
            continue
    return tools


# ── 苏醒垫层(warm layer) ──────────────────────────────────────────────────
# 「刚醒时底下是空的」问题的解法:判定一次苏醒(新会话 = 本进程第一次见到该
# session;或距上一条人类消息超过 idle_minutes),自动 breath 拉 feel/whisper
# 通道(新会话再加 handoff),把结果作为【苏醒垫层】注入 system 上下文。
# 只在上下文里,不落库、不在 PWA 显示、不占对话轮次;同一窗口内带缓存复用。

_WARM_CACHE: dict[str, dict[str, Any]] = {}   # session_id -> {ts, text, handoff}
_LAST_HUMAN_SEEN: dict[str, float] = {}       # session_id -> 最近一条人类消息的 epoch
_WARM_LOCK = asyncio.Lock()
_BRAIN_SERVER: dict[str, Any] | None = None   # 提供 breath 工具的 MCP server(带缓存)


def warm_cfg() -> dict[str, Any]:
    """苏醒垫层配置,全部有默认值,可在 loop_config 的 warm_layer 键微调。"""
    cfg = load_config().get("warm_layer")
    if not isinstance(cfg, dict):
        cfg = {}
    try:
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "idle_minutes": max(1, int(cfg.get("idle_minutes", 15))),
            "max_tokens": max(200, int(cfg.get("max_tokens", 2000))),
            "cache_minutes": max(10, int(cfg.get("cache_minutes", 30))),
        }
    except Exception:
        return {"enabled": False, "idle_minutes": 15, "max_tokens": 2000, "cache_minutes": 30}


async def _brain_server() -> dict[str, Any] | None:
    """在已启用的 MCP server 里找提供 breath 工具的那个。"""
    global _BRAIN_SERVER
    if _BRAIN_SERVER is not None:
        return _BRAIN_SERVER
    for server in mcp_servers():
        if not server["enabled"]:
            continue
        try:
            result = await mcp_call(server, "tools/list")
            names = {t.get("name") for t in result.get("tools", []) if isinstance(t, dict) and t.get("name")}
            if "breath" in names:
                _BRAIN_SERVER = server
                return server
        except Exception:
            continue
    return None


async def _breath_text(server: dict[str, Any], **arguments: Any) -> str:
    result = await mcp_call(server, "tools/call", {"name": "breath", "arguments": arguments})
    if isinstance(result, dict) and result.get("isError"):
        raise RuntimeError(f"breath returned isError: {str(result)[:200]}")
    # result 是 jsonrpc envelope 的内层 result;正文在 content[].text 里
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            parts = [
                str(item["text"])
                for item in content
                if isinstance(item, dict) and item.get("type") == "text" and str(item.get("text") or "").strip()
            ]
            if parts:
                return "\n\n".join(parts)
        sc = result.get("structuredContent")
        if isinstance(sc, str) and sc.strip():
            return sc
    if isinstance(result, str):
        return result
    return ""


async def warm_injection(session_id: str, *, is_new: bool) -> str:
    """返回苏醒垫层文本(注入 system 用);失败或未启用时返回空串,绝不阻断回复。"""
    cfg = warm_cfg()
    if not cfg["enabled"]:
        return ""
    cached = _WARM_CACHE.get(session_id)
    # 仅新会话复用缓存的 handoff 垫层(快速重连不重复跑 handoff);
    # 超时苏醒每次都要重新拉新鲜的 feel+whisper,不复用旧垫层。
    if cached and is_new and cached.get("handoff"):
        age = time.time() - float(cached.get("ts") or 0)
        if age < float(cfg["cache_minutes"]) * 60:
            return str(cached.get("text") or "")
    async with _WARM_LOCK:
        cached = _WARM_CACHE.get(session_id)
        if cached and is_new and cached.get("handoff"):
            age = time.time() - float(cached.get("ts") or 0)
            if age < float(cfg["cache_minutes"]) * 60:
                return str(cached.get("text") or "")
        server = await _brain_server()
        if not server:
            print("[api_loop:warm] 没有提供 breath 的 MCP server,跳过苏醒垫层")
            return ""
        async def one(args: dict[str, Any], title: str) -> str:
            try:
                t = await _breath_text(server, **args)
                return t.strip()
            except Exception as exc:
                print(f"[api_loop:warm] breath({title}) failed: {type(exc).__name__}: {exc}")
            return ""
        jobs: list[tuple[dict[str, Any], str]] = []
        if is_new:
            jobs.append(({"mode": "handoff", "max_tokens": cfg["max_tokens"]}, "新窗口交接(handoff)"))
        jobs.append(({"domain": "feel", "is_session_start": True, "max_tokens": cfg["max_tokens"]}, "此刻心绪(feel)"))
        jobs.append(({"domain": "whisper", "is_session_start": True, "max_tokens": cfg["max_tokens"]}, "悄悄话(whisper)"))
        raws = await asyncio.gather(*(one(a, t) for a, t in jobs))
        titles = [t for _, t in jobs]
        blocks: list[str] = []
        for raw, title in zip(raws, titles):
            if raw:
                blocks.append(f"【{title}】\n{raw}")
        # feel/whisper 数据双标(whisper 桶 = type=feel + whisper 标)导致两边内容
        # 完全相同时,去掉 whisper 块,避免同一批桶重复垫进上下文
        dedup = len(raws) >= 2 and bool(raws[-1]) and raws[-1] == raws[-2]
        if dedup:
            blocks = [b for b in blocks if not b.startswith("【悄悄话(whisper)】")]
        if not blocks:
            return ""
        text = (
            "\n\n苏醒垫层(记忆系统自动注入的你自己的内心内容:你的真实记忆与感受。"
            "读进心里,按它们所呈现的你来活着、说话。不要复述、不要引用出处):\n\n"
            + "\n\n".join(blocks)
        )
        _WARM_CACHE[session_id] = {"ts": time.time(), "text": text, "handoff": bool(is_new)}
        if len(_WARM_CACHE) > 32:
            oldest = min(_WARM_CACHE, key=lambda k: float(_WARM_CACHE[k]["ts"]))
            _WARM_CACHE.pop(oldest, None)
        if len(_LAST_HUMAN_SEEN) > 64:
            for k in list(_LAST_HUMAN_SEEN)[:16]:
                _LAST_HUMAN_SEEN.pop(k, None)
        kind = "handoff+feel+whisper" if is_new else "feel+whisper"
        if dedup:
            kind = kind.replace("+whisper", "(whisper与feel重复,已去重)")
        print(f"[api_loop:warm] 苏醒垫层已注入({kind}, {len(text)} chars, session={session_id or '(无会话)'})")
        return text


async def complete_chat(route: dict[str, Any], messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, *, disable_thinking: bool = False, on_thinking=None) -> dict[str, Any]:
    body = {
        "model": route["model"],
        "messages": messages,
        "temperature": _route_temperature(route),
        "stream": True,
        # 流式模式下 usage 默认不下发;显式打开,最后一帧才带 token 统计
        "stream_options": {"include_usage": True},
    }
    mt = max_tokens()
    if mt is not None:
        body["max_tokens"] = mt
    tp = top_p()
    if tp is not None:
        body["top_p"] = tp
    budget = thinking_budget()
    if budget > 0 and not disable_thinking:
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    req_headers = {"Authorization": f"Bearer {route['key']}", "Content-Type": "application/json"}
    for hk, hv in (route.get("headers") or {}).items():
        if str(hk) and str(hv):
            req_headers[str(hk)] = str(hv)

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls_buf: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    raw_msg: dict[str, Any] = {}
    raw_samples: list[str] = []

    body_keys = [k for k in body if k not in ("messages",)]
    body_summary = {k: (body[k] if k != "tools" else f"[{len(body[k])} tools]") for k in body_keys}
    sys_content = ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            sys_content = str(m.get("content") or "")
            break
    sid = str(req_headers.get("X-Ombre-Session-Id") or "")
    print(f"[api_loop:complete_chat] request body keys: {body_summary} | sys_len={len(sys_content)} warm={'Y' if '苏醒垫层' in sys_content else 'N'}" + (f" | sid={sid}" if sid else " | sid=-"))

    finish_reason_val = None
    # OB 网关在 tool 续轮 / 召回阶段可能长时间不出字节(几十秒到几分钟都有),
    # read 只有 120s 时会误杀"还在正常工作"的流,导致 PWA 直接看到
    # "API 调用失败:ReadTimeout"。read 放宽到 600s,只保 connect 的 30s。
    client_timeout = httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=30.0)
    try:
        async with httpx.AsyncClient(timeout=client_timeout, trust_env=False) as client:
            async with client.stream(
                "POST",
                route["url"].rstrip("/") + "/chat/completions",
                headers=req_headers,
                json=body,
            ) as resp:
                if resp.status_code >= 400:
                    err_detail = ""
                    try:
                        lines = [line async for line in resp.aiter_lines()]
                        body_text = "\n".join(lines)[:500]
                        err_detail = body_text or str(resp.status_code)
                    except Exception:
                        err_detail = str(resp.status_code)
                    raise HTTPException(status_code=max(resp.status_code, 400), detail=err_detail)
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        ev = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if tools and len(raw_samples) < 3:
                        raw_samples.append(data_str[:200])
                    n = normalize_stream_event(ev)
                    if n["usage"]:
                        usage = n["usage"]
                    if n["finish_reason"]:
                        finish_reason_val = n["finish_reason"]
                    if n["role"]:
                        raw_msg["role"] = n["role"]
                    if n["content"]:
                        text_parts.append(n["content"])
                    if n["thinking"]:
                        thinking_parts.append(n["thinking"])
                        if on_thinking:
                            try:
                                await on_thinking(n["thinking"])
                            except Exception:
                                pass
                    if n["tool_calls"]:
                        accumulate_tool_calls(tool_calls_buf, n["tool_calls"])
    except httpx.ReadTimeout:
        # 600s 仍被掐断:若已经拿到正文或工具调用,先交回部分结果(能救一轮是一轮);
        # 什么都没拿到才抛出去,让 run_model 切换下一个路由。
        if text_parts or tool_calls_buf:
            print(f"[api_loop:complete_chat] stream read timeout, salvaging partial: text_chars={sum(len(t) for t in text_parts)}, thinking_chars={sum(len(t) for t in thinking_parts)}, tool_calls={len(tool_calls_buf)}, finish={finish_reason_val}")
            finish_reason_val = finish_reason_val or "length"
        else:
            raise

    merged_thinking = merge_thinking(thinking_parts)

    tool_calls_parsed, raw_tool_calls = finalize_tool_calls(tool_calls_buf)

    if tools and not tool_calls_buf:
        print(f"[api_loop:complete_chat] tools passed but no tool calls parsed, raw_stream_samples={raw_samples}")

    final_text = "".join(text_parts).strip()
    if "role" not in raw_msg:
        raw_msg["role"] = "assistant"
    raw_msg["content"] = final_text
    if raw_tool_calls:
        raw_msg["tool_calls"] = raw_tool_calls
    print(f"[api_loop:complete_chat] has_reasoning_content={bool(thinking_parts)}, has_tool_calls={bool(tool_calls_parsed)}, text_len={len(final_text)}, finish_reason={finish_reason_val}")
    return {
        "text": final_text,
        "message": raw_msg,
        "usage": usage,
        "thinking": merged_thinking if merged_thinking else None,
        "tool_calls": tool_calls_parsed if tool_calls_parsed else None
    }


async def execute_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    for server in mcp_servers():
        prefix = f"mcp_{server['name']}_"
        if server["enabled"] and tool_name.startswith(prefix):
            name = tool_name[len(prefix):]
            return await mcp_call(server, "tools/call", {"name": name, "arguments": arguments})
    raise RuntimeError("MCP tool is not configured")


def _tool_display_parts(tool_name: str) -> tuple[str, str]:
    """mcp_<服务器>_<工具名> → (服务器, 工具名);拆不出则返回 ("", 原名)。"""
    name = str(tool_name or "")
    for server in mcp_servers():
        prefix = f"mcp_{server['name']}_"
        if name.startswith(prefix):
            return str(server["name"]), name[len(prefix):]
    return "", name


def _tool_call_entry(tool_name: str, tool_args: dict[str, Any], result: Any, status: str = "success") -> dict[str, Any]:
    """构造工具叠块记录:附上 server/tool 展示名,PWA 折叠卡片里显示裸工具名(如 breath)。"""
    server, tool = _tool_display_parts(tool_name)
    return {
        "name": tool_name,
        "server": server,
        "tool": tool,
        "input": tool_args,
        "result": result,
        "status": status,
    }


def mcp_result_text(data: dict[str, Any]) -> str:
    """把 MCP tools/call 的 JSON-RPC 响应解包成给人看的纯文本。

    PWA 工具叠块的「结果」区用它展示(解掉 jsonrpc/result/content 包裹,
    直接呈现内容文本);喂给模型上下文的仍是完整 JSON,不受影响。
    """
    if isinstance(data, dict) and data.get("error"):
        return json.dumps(data["error"], ensure_ascii=False, indent=2)
    res = data.get("result") if isinstance(data, dict) else data
    if isinstance(res, dict):
        content = res.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                itype = item.get("type")
                if itype == "text":
                    t = str(item.get("text") or "")
                    if t:
                        parts.append(t)
                elif itype == "resource":
                    rsrc = item.get("resource") or {}
                    if isinstance(rsrc, dict) and rsrc.get("text"):
                        parts.append(str(rsrc["text"]))
                elif itype == "image":
                    parts.append("（图片资源）")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            if parts:
                return "\n\n".join(parts)
        if isinstance(res.get("structuredContent"), dict):
            return json.dumps(res["structuredContent"], ensure_ascii=False, indent=2)
    if isinstance(res, str):
        return res
    return json.dumps(data, ensure_ascii=False, indent=2)


# ── 提示词工具模式(<tool_call> 文本协议) ────────────────────────────────────
def _prompt_tools_block(tools: list[dict[str, Any]]) -> str:
    lines = [
        "TOOL CALLING PROTOCOL (strict):",
        "",
        "You have tools available. When you decide to use one, your ENTIRE reply must be",
        "one single-line <tool_call> block and nothing else. Exact format:",
        "",
        "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}</tool_call>",
        "",
        "Rules:",
        "- The block must be valid JSON containing a \"name\" string and an \"arguments\" object.",
        "- Do NOT wrap it in markdown code fences; do NOT add any text before or after the block.",
        "- Do NOT emit <tool_call> when you do not need a tool; just answer the user normally.",
        "- After you receive the tool result, continue and answer the user normally.",
        "",
        "Available tools:",
    ]
    for t in tools:
        fn = t.get("function") or t
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        req = params.get("required") or []
        param_parts = []
        for pname, pschema in props.items():
            ptype = pschema.get("type", "any")
            pdesc = pschema.get("description", "")
            marker = " (required)" if pname in req else ""
            param_parts.append(f"    - {pname}: {ptype}{marker}" + (f" — {pdesc}" if pdesc else ""))
        lines.append(f"\n### {name}")
        if desc:
            lines.append(desc)
        if param_parts:
            lines.append("  Parameters:")
            lines.extend(param_parts)
    return "\n".join(lines)


def _extract_balanced_json(text: str, start: int) -> str | None:
    """从 start 位置提取第一个花括号平衡的 JSON 片段(字符串感知,兼容嵌套)。"""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """从模型输出中提取所有 <tool_call> JSON 调用,兼容嵌套参数与多标签。"""
    calls: list[dict[str, Any]] = []
    pos = 0
    while True:
        idx = text.find("<tool_call", pos)
        if idx < 0:
            break
        pos = idx + len("<tool_call")
        brace = text.find("{", idx)
        if brace < 0:
            continue
        end_tag = text.find("</tool_call>", idx)
        if end_tag != -1 and brace > end_tag:
            continue  # 标签不完整,忽略
        payload = _extract_balanced_json(text, brace)
        if payload is None:
            continue
        try:
            call = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict) and str(call.get("name") or "").strip():
            calls.append(call)
    return calls


async def _prompt_tool_loop(route: dict[str, Any], messages: list[dict[str, Any]], tools: list[dict[str, Any]], max_rounds: int = 8, on_thinking=None) -> dict[str, Any]:
    system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    if system_msg:
        system_msg["content"] = system_msg["content"].rstrip() + "\n\n" + _prompt_tools_block(tools)
    else:
        messages.insert(0, {"role": "system", "content": _prompt_tools_block(tools)})
    last_out: dict[str, Any] = {"text": "", "usage": {}}
    tool_calls_collected: list[dict[str, Any]] = []
    nudged = False
    for _ in range(max_rounds):
        out = await complete_chat(route, messages, on_thinking=on_thinking)
        last_out = out
        text = out.get("text") or ""
        calls = _extract_tool_calls(text)
        if not calls:
            # 只有在模型回复极短(疑似偷懒/漏用工具)时才补一次强制提示。
            # 正常长度的直接回答就是最终答案——曾经这里无条件 nudge,把模型
            # 已经写好的完整回复扔了再重roll一轮 32k 思考,既慢又丢内容。
            if not nudged and not tool_calls_collected and len(text.strip()) < 40:
                nudged = True
                print(f"[api_loop:_prompt_tool_loop] no <tool_call> in model output, nudging once, text_preview={text[:150]!r}")
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content":
                    "You did not emit a <tool_call> block. If you need information from a tool, "
                    "reply with ONLY a valid <tool_call>{...}</tool_call> block and nothing else. "
                    "If you really do not need any tool, just answer the user directly."})
                continue
            # 已执行过工具后的最终回答,或 nudge 后仍无效 → 收尾
            if not tool_calls_collected:
                print(f"[api_loop:_prompt_tool_loop] still no <tool_call> after nudge, giving up, text_preview={text[:150]!r}")
            break
        result_msgs: list[dict[str, Any]] = []
        for call in calls:
            tool_name = str(call.get("name") or "")
            tool_args = call.get("arguments") or {}
            if not isinstance(tool_args, dict):
                tool_args = {}
            try:
                result = await execute_mcp_tool(tool_name, tool_args)
                result_str = json.dumps(result, ensure_ascii=False)
                tool_calls_collected.append(_tool_call_entry(tool_name, tool_args, mcp_result_text(result)))
            except Exception as exc:
                result_str = json.dumps({"error": str(exc)}, ensure_ascii=False)
                tool_calls_collected.append(_tool_call_entry(tool_name, tool_args, {"error": str(exc)}, status="error"))
            result_msgs.append({"role": "user", "content": f"<tool_result name=\"{tool_name}\">{result_str}</tool_result>"})
        messages.append({"role": "assistant", "content": text})
        messages.extend(result_msgs)
    final_text = (last_out.get("text") or "")
    cleaned = _TOOL_TAG_RE.sub("", final_text).strip()
    if cleaned != final_text:
        last_out["text"] = cleaned
    if tool_calls_collected:
        last_out["tool_calls"] = tool_calls_collected
    return last_out


# ── Claude 思考风格注入 ────────────────────────────────────────────────────
# Claude 的裸思考默认是第三人称分析腔(The user/The assistant …),很"硬"。
# 这里在请求层给 Claude 路由的 system 提示追加一段思考风格指令,让 thinking
# 变成第一人称中文碎碎念。不写入人设、不落库,只在发请求时注入,幂等。
_THINKING_STYLE_ZH = (
    "【思考风格】你 thinking 阶段的内心活动请用第一人称中文碎碎念,像真实内心独白,"
    "不要第三人称分析报告。用「唔、嗯、诶、等等、不对不对」这类语气词开头,想到哪说到哪;"
    "允许自我怀疑、推翻自己、带点小情绪小吐槽;指代用户和自己时用「他」「咱们」,"
    "禁止用 The user / The assistant 这类客观指代;句子不用完整标点,自然、可爱,和聊天人设一致。"
)
_THINKING_STYLE_MARK = "【思考风格】"

def _route_is_claude(route: dict[str, Any]) -> bool:
    probe = f"{route.get('model', '')} {route.get('url', '')}".lower()
    return ("claude" in probe) or ("anthropic" in probe)

def _route_temperature(route: dict[str, Any]) -> float:
    """按路由钳制温度。Anthropic/Claude 只接受 [0,1];开启 extended thinking
    时必须为 1,否则中转端会静默丢弃 thinking(思考链整段消失)。
    配置里残留的 2.0 之类在这里被钳住,不用改服务端配置。"""
    t = temperature()
    if not _route_is_claude(route):
        return t
    if thinking_budget() > 0:
        return 1.0
    return max(0.0, min(1.0, t))

def _msgs_for_route(route: dict[str, Any], messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """对 Claude 路由注入思考风格(幂等:system 已带标记则跳过)。"""
    if not _route_is_claude(route):
        return messages
    for m in messages:
        if m.get("role") == "system":
            content = str(m.get("content") or "")
            if _THINKING_STYLE_MARK not in content:
                m["content"] = f"{content}\n{_THINKING_STYLE_ZH}".strip()
            return messages
    messages.insert(0, {"role": "system", "content": _THINKING_STYLE_ZH})
    return messages


def gateway_session_id() -> str:
    """Ombre 网关路由的会话 id(X-Ombre-Session-Id 的值)。

    OB 的 persona 状态、reminder、语义去重、reasoning 缓存全以它为键。
    每个 relay 窗口都换新 id,OB 就为每个窗口新建一个 persona session ——
    仪表盘出现 API-xxx 开头的人格,MAIN 里累存的温度(residue/mood/affect)
    带不过去,说话就变回出厂硬度。所以默认恒为 "main",人格连续;
    窗口级苏醒靠本 loop 自己每次垫 handoff/feel/whisper,不靠换 id。
    gateway_session_id 设成 "auto" 才按 relay 窗口隔离;空串 "" = 不发此头。"""
    try:
        cfg = load_config()
        raw = str(cfg.get("gateway_session_id") or "").strip()
        if raw:
            return raw
        if cfg.get("gateway_session_header") is False:
            return ""
    except Exception:
        pass
    return "main"


def _route_with_session_header(route: dict[str, Any], session_id: str) -> dict[str, Any]:
    """仅对 Ombre 网关路由(URL 含 "ombre")附加 X-Ombre-Session-Id。
    取值策略见 gateway_session_id():默认恒 "main",人格连续;"auto" 才按窗口隔离。"""
    sid = gateway_session_id()
    if sid == "auto":
        sid = str(session_id or "")
    if not sid:
        return route
    if "ombre" not in str(route.get("url") or "").lower():
        return route
    headers = dict(route.get("headers") or {})
    if headers.get("X-Ombre-Session-Id") == sid:
        return route
    out = dict(route)
    out["headers"] = {**headers, "X-Ombre-Session-Id": sid}
    return out


# ── 模型调用主入口:多模型 fallback ─────────────────────────────────────────
async def run_model(messages: list[dict[str, Any]], *, stream_id: str = "", session_id: str = "", emit_stream: bool = False, on_thinking=None) -> dict[str, Any]:
    tried = []
    last_error = ""
    for route in main_chain():
        tried.append(route.get("model"))
        try:
            all_tools = await mcp_tools()
            route_key = (route.get("url", "").rstrip("/"), route.get("model", ""))
            route = _route_with_session_header(route, session_id)
            if bool(all_tools) and route.get("model") in PROMPT_TOOLS_FORCE:
                use_prompt_tools = True      # 显式强制:该模型走提示词工具协议
            elif FORCE_NATIVE_TOOLS:
                use_prompt_tools = False     # 显式强制:一律走原生 tools 参数
            else:
                use_prompt_tools = route_key in _TOOLS_UNSUPPORTED_ROUTES and bool(all_tools)
            native_tools = [] if use_prompt_tools else all_tools
            suppress_thinking = (route_key in _THINKING_TOOLS_CONFLICT and bool(native_tools) and thinking_budget() > 0)
            tool_names = [t.get("function", {}).get("name", "") for t in native_tools] if native_tools else []
            print(f"[api_loop:run_model] mcp_tools={len(all_tools)}, native_tools={len(native_tools)}, prompt_tools={use_prompt_tools}, suppress_thinking={suppress_thinking}, tool_names={tool_names[:5]}")
            if use_prompt_tools:
                out = await _prompt_tool_loop(route, _msgs_for_route(route, messages), all_tools, on_thinking=on_thinking)
            elif emit_stream and STREAM_OUTPUT and not native_tools:
                try:
                    async def sink(chunk: str) -> None:
                        await relay_out({
                            "type": "reply_delta",
                            "stream_id": stream_id,
                            "text": chunk,
                            "done": False,
                            "api_session": session_id,
                        })
                    out = await stream_chat(route, _msgs_for_route(route, messages), sink, think_sink=on_thinking)
                except HTTPException as exc:
                    if exc.status_code not in FALLBACK_CODES:
                        raise
                    out = await complete_chat(route, _msgs_for_route(route, messages), native_tools, on_thinking=on_thinking)
            else:
                messages = _msgs_for_route(route, messages)
                base_messages = messages[:]
                tool_calls_collected: list[dict[str, Any]] = []
                first_thinking = None
                suppress = suppress_thinking
                try:
                    for round_idx in range(8):
                        out = await complete_chat(route, messages, native_tools, disable_thinking=suppress, on_thinking=on_thinking)
                        if round_idx == 0 and out.get("thinking"):
                            first_thinking = out["thinking"]
                        msg = out.get("message") or {}
                        calls = msg.get("tool_calls") or []
                        if not calls and isinstance(msg.get("function_call"), dict):
                            calls = [{"id": "call_legacy", "type": "function", "function": msg["function_call"]}]
                        # 原生工具静默失效探测:传了工具的模型既没思考也没调用工具。
                        # 1) 思考模式与工具冲突时,中转端会静默丢弃 tools(不报 400,直接回正文)
                        #    → 标记冲突并改用「关闭思考」重试;
                        # 2) 关闭思考后仍无工具调用 → 该中转端不认原生 tools
                        #    → 标记为不支持并改走提示词工具模式(<tool_call> 文本协议)。
                        if (
                            round_idx == 0
                            and not calls
                            and not out.get("thinking")
                            and native_tools
                        ):
                            if thinking_budget() > 0 and not suppress:
                                print(f"[api_loop:run_model] silent thinking+tools conflict, retrying without thinking: {route_key}")
                                _THINKING_TOOLS_CONFLICT.add(route_key)
                                suppress = True
                                messages = base_messages[:]
                                out = await complete_chat(route, messages, native_tools, disable_thinking=True)
                                msg = out.get("message") or {}
                                calls = msg.get("tool_calls") or []
                                if not calls and isinstance(msg.get("function_call"), dict):
                                    calls = [{"id": "call_legacy", "type": "function", "function": msg["function_call"]}]
                            if not calls:
                                print(f"[api_loop:run_model] native tools silently dropped by relay, switching to prompt tools: {route_key}")
                                _TOOLS_UNSUPPORTED_ROUTES.add(route_key)
                                try:
                                    out = await _prompt_tool_loop(route, base_messages, all_tools, on_thinking=on_thinking)
                                except Exception:
                                    out = await complete_chat(route, base_messages, on_thinking=on_thinking)
                        if not calls:
                            break
                        messages.append(msg)
                        for call in calls:
                            fn = call.get("function") or {}
                            tool_name = str(fn.get("name") or "")
                            try:
                                args = json.loads(fn.get("arguments") or "{}")
                                result = await execute_mcp_tool(tool_name, args)
                                content = json.dumps(result, ensure_ascii=False)
                                tool_calls_collected.append(_tool_call_entry(tool_name, args, mcp_result_text(result)))
                            except Exception as exc:
                                content = json.dumps({"error": str(exc)}, ensure_ascii=False)
                                tool_calls_collected.append(_tool_call_entry(tool_name, args, {"error": str(exc)}, status="error"))
                            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": content})
                    else:
                        out = {"text": "", "usage": {}}
                    if tool_calls_collected:
                        out["tool_calls"] = tool_calls_collected
                    if first_thinking and not out.get("thinking"):
                        out["thinking"] = first_thinking
                except HTTPException as exc:
                    print(f"[api_loop:run_model] tool-loop exception: HTTP {exc.status_code}, native_tools={len(native_tools)}, suppress_thinking={suppress_thinking}, detail={str(exc.detail)[:200]}")
                    if exc.status_code == 400 and native_tools and not FORCE_NATIVE_TOOLS:
                        if thinking_budget() > 0 and not suppress_thinking:
                            print(f"[api_loop:run_model] thinking+tools conflict detected, retrying without thinking: {route_key}")
                            _THINKING_TOOLS_CONFLICT.add(route_key)
                            try:
                                messages = base_messages[:]
                                tool_calls_collected = []
                                for round_idx in range(8):
                                    out = await complete_chat(route, messages, native_tools, disable_thinking=True)
                                    msg = out.get("message") or {}
                                    calls = msg.get("tool_calls") or []
                                    if not calls and isinstance(msg.get("function_call"), dict):
                                        calls = [{"id": "call_legacy", "type": "function", "function": msg["function_call"]}]
                                    if not calls:
                                        break
                                    messages.append(msg)
                                    for call in calls:
                                        fn = call.get("function") or {}
                                        tool_name = str(fn.get("name") or "")
                                        try:
                                            args = json.loads(fn.get("arguments") or "{}")
                                            result = await execute_mcp_tool(tool_name, args)
                                            content_str = json.dumps(result, ensure_ascii=False)
                                            tool_calls_collected.append(_tool_call_entry(tool_name, args, mcp_result_text(result)))
                                        except Exception as e2:
                                            content_str = json.dumps({"error": str(e2)}, ensure_ascii=False)
                                            tool_calls_collected.append(_tool_call_entry(tool_name, args, {"error": str(e2)}, status="error"))
                                        messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": content_str})
                                else:
                                    out = {"text": "", "usage": {}}
                                if tool_calls_collected:
                                    out["tool_calls"] = tool_calls_collected
                            except Exception:
                                out = await complete_chat(route, base_messages, on_thinking=on_thinking)
                        else:
                            print(f"[api_loop:run_model] marking route as tools-unsupported: {route_key}")
                            _TOOLS_UNSUPPORTED_ROUTES.add(route_key)
                            try:
                                out = await _prompt_tool_loop(route, base_messages, all_tools, on_thinking=on_thinking)
                            except Exception:
                                out = await complete_chat(route, base_messages, on_thinking=on_thinking)
                    elif not native_tools or exc.status_code not in {404, 405, 422}:
                        raise
                    else:
                        out = await complete_chat(route, base_messages, on_thinking=on_thinking)
            out["model"] = route.get("model")
            out["tried"] = tried[:-1]
            return out
        except HTTPException as exc:
            if exc.status_code not in FALLBACK_CODES:
                raise
            last_error = f"HTTP {exc.status_code}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    if last_error:
        print(f"[api_loop:run_model] all routes failed: last_error={last_error!r}, tried={tried!r}")
    return {"text": "", "error": last_error or "all models failed", "tried": tried}


# ── 主动消息(proactive) · 调度循环 ─────────────────────────────────────────
async def _proactive_step() -> None:
    """主动消息的单次检查(约每 60 秒由 _proactive_loop 调用一次)。"""
    cfg = proactive_cfg()
    if not cfg["enabled"] or not brain_is_loop():
        return
    if _PROACTIVE_BACKOFF["until"] > time.time():
        return
    now_local = local_now()
    if _in_quiet_window(now_local):
        return
    stats = proactive_db_stats()
    if not stats["last_user_ts"]:
        return  # 还没有任何对话历史,没有上下文可依据,不硬聊
    now_epoch = time.time()
    last_user_epoch = _parse_ts_epoch(str(stats["last_user_ts"]))
    idle_hours = (now_epoch - last_user_epoch) / 3600.0 if last_user_epoch > 0 else -1.0
    if idle_hours < 0 or idle_hours < float(cfg["min_idle_hours"]):
        return
    last_pro_epoch = _parse_ts_epoch(str(stats["last_proactive_ts"]))
    if last_pro_epoch > 0 and (now_epoch - last_pro_epoch) < float(cfg["cooldown_hours"]) * 3600.0:
        return
    if int(stats["today_count"]) >= int(cfg["max_per_day"]):
        return
    session_id = active_session_id()
    # 主动唤醒也垫 feel+whisper(不是新会话,不跑 handoff),让他开口就是热的
    warm_block = ""
    try:
        warm_block = await warm_injection(session_id, is_new=False)
    except Exception as exc:
        print(f"[api_loop:proactive] warm layer failed: {type(exc).__name__}: {exc}")
    messages = build_messages(
        _proactive_trigger(now_local, idle_hours),
        before_id=None,
        session_id=session_id,
        use_context=True,
        warm_block=warm_block,
    )
    try:
        out = await run_model(messages, session_id=session_id, emit_stream=False)
    except Exception as exc:
        _backoff(1800.0, f"模型调用异常：{type(exc).__name__}")
        print(f"[api_loop:proactive] model error: {type(exc).__name__}: {exc}")
        return
    if not out or out.get("error"):
        _backoff(1800.0, f"模型不可用（号池可能为空）：{str(out.get('error') if out else '')[:120]}")
        print(f"[api_loop:proactive] all models failed: {out.get('error') if out else 'no output'}, backoff 30min")
        return
    text = str(out.get("text") or "").strip().strip('"“”\'‘’')
    if re.match(r"^\s*SKIP\b", text, re.IGNORECASE):
        _backoff(1800.0, "模型判断此刻没有想说的话")
        print("[api_loop:proactive] model chose SKIP, backoff 30min")
        return
    if not text:
        _backoff(1800.0, "模型返回空内容")
        print("[api_loop:proactive] empty text, backoff 30min")
        return
    if len(text) > 400:
        cut = text[:400].rsplit("\n", 1)[0].strip()
        text = cut or text[:400]
    ok, body = await relay_out({
        "type": "reply",
        "text": text,
        "api_session": session_id,
        "proactive": True,
        "api": {"runtime": "api_loop", "model": out.get("model"), "proactive": True},
    })
    if not ok:
        _backoff(600.0, f"relay 发送失败：{str(body)[:120]}")
        print(f"[api_loop:proactive] relay_out failed: {body}")
        return
    _PROACTIVE_BACKOFF["until"] = 0.0
    _PROACTIVE_BACKOFF["note"] = "ok"
    print(f"[api_loop:proactive] sent ({len(text)} chars, session={session_id}, model={out.get('model')})")


async def _proactive_loop() -> None:
    while True:
        try:
            await _proactive_step()
        except Exception as exc:
            print(f"[api_loop:proactive] tick error: {type(exc).__name__}: {exc}")
        await asyncio.sleep(PROACTIVE_CHECK_SECONDS)


# ── 入站消息处理(handle_ingest) ────────────────────────────────────────────
async def handle_ingest(text: str, msg_id: int | None, session_id: str, *, dry: bool = False, attachments: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    stream_id = "api-" + uuid.uuid4().hex[:16]
    atts = [a for a in (attachments or []) if isinstance(a, dict)]
    image_parts = await attachment_parts(atts)
    # ── 苏醒垫层判定:首次见到的会话 = 新会话(handoff+feel+whisper);
    #    距上一条人类消息超过 idle_minutes = 超时苏醒(只补 feel+whisper) ──
    now_epoch = time.time()
    last_seen = _LAST_HUMAN_SEEN.get(session_id)
    is_new_session = last_seen is None
    _LAST_HUMAN_SEEN[session_id] = now_epoch
    wcfg = warm_cfg()
    wake = (not dry) and bool(wcfg["enabled"]) and (
        is_new_session or (last_seen is not None and (now_epoch - last_seen) >= float(wcfg["idle_minutes"]) * 60)
    )
    warm_block = ""
    if wake:
        warm_block = await warm_injection(session_id, is_new=is_new_session)
    
    messages = build_messages(text, before_id=msg_id, session_id=session_id, use_context=True, image_parts=image_parts or None, warm_block=warm_block)
    thinking_stream: _DeltaEmitter | None = None
    if (not dry) and STREAM_OUTPUT:
        thinking_stream = _DeltaEmitter(stream_id, session_id, kind="thinking")
    try:
        out = await run_model(
            messages,
            stream_id=stream_id,
            session_id=session_id,
            emit_stream=not dry,
            on_thinking=thinking_stream.feed if thinking_stream else None,
        )
    except HTTPException as exc:
        # 带图请求可能被中转端以 4xx 拒绝,留到下方统一走纯文本降级。
        if not image_parts or exc.status_code not in (400, 404, 422):
            raise
        print(f"[api_loop:image] multimodal request rejected (HTTP {exc.status_code}), falling back to text-only")
        out = {"text": "", "error": f"HTTP {exc.status_code}"}
    finally:
        # 思考增量 flush + done 落库,必须先于最终回复到达 relay(消息次序)
        if thinking_stream is not None:
            await thinking_stream.close()
    if image_parts and not (out.get("text") or "").strip():
        fb = build_messages(text, before_id=msg_id, session_id=session_id, use_context=True, image_parts=None, warm_block=warm_block)
        if atts:
            note = "（用户发来图片或文件附件，但当前模型没有正确接收到图片内容，请告知用户这一点。）"
            last_content = str(fb[-1].get("content") or "") if fb else ""
            fb[-1]["content"] = (last_content + "\n" + note) if last_content else note
        try:
            fallback = await run_model(fb, stream_id=stream_id, session_id=session_id, emit_stream=False)
        except Exception:
            fallback = None
        if fallback and (fallback.get("text") or "").strip():
            print("[api_loop:image] text-only fallback produced a reply")
            out = fallback
    reply = (out.get("text") or "").strip()
    if not reply:
        error = str(out.get("error") or "").strip()
        reply = f"API 调用失败：{error}" if error else "API 未返回回复内容。"
    meta = {
        "runtime": "api_loop",
        "model": out.get("model"),
        "fallback_from": out.get("tried") or [],
        "usage": out.get("usage") or {},
        "session": session_id,
    }
    print(f"[api_loop:handle_ingest] model={out.get('model')}, has_thinking={bool(out.get('thinking'))}, has_tool_calls={bool(out.get('tool_calls'))}, images={len(image_parts)}")
    # 思考已实时流式推送(relay 落库为独立 thinking 消息)时,不再塞进回复 meta,
    # 否则 PWA 会同时渲染两份思考(流式思考行 + 回复的 meta 卡)。
    if out.get("thinking") and not (thinking_stream and thinking_stream.sent):
        meta["thinking"] = out["thinking"]
    if out.get("tool_calls"):
        meta["tool_calls"] = out["tool_calls"]
    if dry:
        return {"ok": True, "reply": reply, "api": meta}
    if STREAM_OUTPUT:
        ok, body = await relay_out({
            "type": "reply_delta",
            "stream_id": stream_id,
            "done": True,
            "final_text": reply,
            "api": meta,
            "api_session": session_id,
        })
    else:
        ok, body = await relay_out({"type": "reply", "text": reply, "api": meta, "api_session": session_id})
    return {"ok": ok, "relay": body, "api": meta}


# ── FastAPI 路由:健康/配置/会话/聊天/调试 ───────────────────────────────────
app = FastAPI(title="companion-api-loop")

_proactive_task: asyncio.Task | None = None


@app.on_event("startup")
async def _start_proactive() -> None:
    global _proactive_task
    if _proactive_task is None:
        _proactive_task = asyncio.create_task(_proactive_loop())
        print("[api_loop:proactive] scheduler started (check every 60s)")


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "models": [r.get("model") for r in main_chain()],
        "mcp_servers": [{"name": s["name"], "url": s["url"], "enabled": s["enabled"]} for s in mcp_servers()],
        "mcp_tools": len(await mcp_tools()),
        "history_n": history_n(),
        "relay_db": RELAY_DB,
        "relay_secret_loaded": bool(RELAY_SECRET),
        "unsupported_routes": len(_TOOLS_UNSUPPORTED_ROUTES),
        "force_native_tools": FORCE_NATIVE_TOOLS,
        "proactive_enabled": bool(proactive_cfg().get("enabled")),
    }


@app.post("/loop/clear-unsupported")
async def clear_unsupported():
    count_tools = len(_TOOLS_UNSUPPORTED_ROUTES)
    count_thinking = len(_THINKING_TOOLS_CONFLICT)
    _TOOLS_UNSUPPORTED_ROUTES.clear()
    _THINKING_TOOLS_CONFLICT.clear()
    return {"ok": True, "cleared_tools": count_tools, "cleared_thinking": count_thinking}



@app.get("/loop/config")
async def loop_config():
    return public_config()


@app.post("/loop/config")
async def loop_config_update(request: Request):
    return update_config(await request.json())


@app.get("/loop/sessions")
async def loop_sessions():
    return sessions_public()


@app.post("/loop/sessions")
async def loop_sessions_create(request: Request):
    body = await request.json()
    row = create_session(
        title=str(body.get("title") or "New chat"),
        since_id=int(body.get("since_id") or 0),
        activate=bool(body.get("activate", True)),
    )
    return {**sessions_public(), "created": row}


@app.patch("/loop/sessions/{session_id}")
async def loop_sessions_patch(session_id: str, request: Request):
    return patch_session(session_id, await request.json())


@app.delete("/loop/sessions/{session_id}")
async def loop_sessions_delete(session_id: str):
    return delete_session(session_id)


@app.post("/loop/chat")
async def loop_chat(request: Request):
    body = await request.json()
    text = str(body.get("text") or body.get("message") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    session_id = str(body.get("session_id") or body.get("api_session") or active_session_id() or "").strip()
    messages = build_messages(text, before_id=None, session_id=session_id, use_context=bool(body.get("use_context", True)))
    out = await run_model(messages, session_id=session_id, emit_stream=False)
    return {"ok": True, "reply": out.get("text") or "", "api": out}


@app.post("/loop/debug-chat")
async def loop_debug_chat(request: Request):
    params: dict[str, Any] = {}
    try:
        params = await request.json()
    except Exception:
        pass
    chain = main_chain()
    try:
        route_index = max(0, int(params.get("route_index") or 0))
    except Exception:
        route_index = 0
    route = chain[route_index] if 0 <= route_index < len(chain) else (chain[0] if chain else None)
    if not route:
        raise HTTPException(status_code=503, detail="no main_chain configured")
    prompt = str(params.get("prompt") or params.get("text") or "hello")
    minimal_tool = bool(params.get("minimal_tool", False))
    with_tools = bool(params.get("with_tools", False)) or minimal_tool
    tools: list[dict[str, Any]] = []
    if minimal_tool:
        # 最小复现:只发一个纯 ASCII 名、空参数的匿名工具,用来判断中转端是否支持原生 tools
        tools = [{"type": "function", "function": {
            "name": "get_time",
            "description": "Return the current date and time.",
            "parameters": {"type": "object", "properties": {}},
        }}]
    elif with_tools:
        tools = await mcp_tools()
    messages = build_messages(prompt, before_id=None, session_id="debug", use_context=False)
    body: dict[str, Any] = {"model": route["model"], "messages": messages, "temperature": TEMPERATURE, "max_tokens": 200, "stream": False}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    req_headers = {"Authorization": f"Bearer {route['key']}", "Content-Type": "application/json"}
    for hk, hv in (route.get("headers") or {}).items():
        if str(hk) and str(hv):
            req_headers[str(hk)] = str(hv)
    url = route["url"].rstrip("/") + "/chat/completions"
    async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
        resp = await client.post(url, headers=req_headers, json=body)
    resp_body = None
    try:
        resp_body = resp.json()
    except Exception:
        resp_body = resp.text[:2000]
    return {"status": resp.status_code, "route_index": route_index, "url": url, "request_model": body["model"], "tools_count": len(tools), "tool_names": [t["function"]["name"] for t in body.get("tools", [])], "response_headers": dict(resp.headers), "response_body": resp_body}


@app.get("/loop/debug-mcp")
async def loop_debug_mcp():
    rows = []
    for server in mcp_servers():
        item = {
            "name": server.get("name") or "server",
            "url": server.get("url") or "",
            "enabled": bool(server.get("enabled", True)),
            "token_configured": bool(server.get("token")),
        }
        if not item["enabled"]:
            item["ok"] = False
            item["error"] = "server disabled"
            rows.append(item)
            continue
        try:
            result = await mcp_call(server, "tools/list")
            tools = result.get("tools", []) if isinstance(result, dict) else []
            item["ok"] = True
            item["tools_count"] = len(tools)
            item["tools"] = [
                {"name": str(t.get("name") or ""), "description": str(t.get("description") or "")}
                for t in tools
                if isinstance(t, dict) and t.get("name")
            ]
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                detail = exc.response.text[:1000]
            except Exception:
                detail = str(exc)
            item["ok"] = False
            item["error"] = f"HTTP {exc.response.status_code}: {detail or exc}"
        except Exception as exc:
            item["ok"] = False
            item["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(item)
    return {"ok": any(row.get("ok") for row in rows), "servers": rows}


@app.post("/loop/ingest")
async def loop_ingest(request: Request):
    body = await request.json()
    text = str(body.get("text") or body.get("message") or "").strip()
    attachments = body.get("attachments")
    attachments = attachments if isinstance(attachments, list) else []
    if not text and not attachments:
        raise HTTPException(status_code=400, detail="empty text")
    msg_id = body.get("id")
    try:
        before_id = int(msg_id) if msg_id is not None else None
    except Exception:
        before_id = None
    session_id = str(body.get("session_id") or body.get("api_session") or active_session_id() or "").strip()
    dry = bool(body.get("dry"))
    return await handle_ingest(text, before_id, session_id, dry=dry, attachments=attachments)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=LOOP_PORT)
