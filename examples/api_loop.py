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

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request


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

_TOOLS_UNSUPPORTED_ROUTES: set[tuple[str, str]] = set()
_THINKING_TOOLS_CONFLICT: set[tuple[str, str]] = set()
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

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

def max_tokens() -> int:
    try:
        return max(100, int(load_config().get("max_tokens", MAX_TOKENS)))
    except Exception:
        return MAX_TOKENS

def thinking_budget() -> int:
    try:
        return max(0, int(load_config().get("thinking_budget", 0)))
    except Exception:
        return 0


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


def build_messages(text: str, *, before_id: int | None = None, session_id: str = "", use_context: bool = True) -> list[dict[str, str]]:
    tool_hint = " When a configured MCP tool can provide current, external, or actionable information, use it before answering."
    messages = [{"role": "system", "content": persona() + tool_hint}]
    if use_context:
        for row in relay_rows(before_id, session_id, history_n()):
            content = str(row.get("text") or "").strip()
            if not content:
                continue
            role = "assistant" if row.get("direction") == "out" else "user"
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": text})
    return messages


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
        "temperature": cfg.get("temperature", TEMPERATURE),
        "top_p": cfg.get("top_p", None),
        "max_tokens": cfg.get("max_tokens", MAX_TOKENS),
        "thinking_budget": cfg.get("thinking_budget", 0),
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
        try:
            cfg["max_tokens"] = max(100, min(32000, int(body["max_tokens"])))
        except Exception:
            pass
    if "thinking_budget" in body:
        try:
            cfg["thinking_budget"] = max(0, min(32000, int(body["thinking_budget"])))
        except Exception:
            pass
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
    save_config(cfg)
    return public_config()


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


async def stream_chat(route: dict[str, Any], messages: list[dict[str, str]], sink) -> dict[str, Any]:
    body = {
        "model": route["model"],
        "messages": messages,
        "temperature": temperature(),
        "max_tokens": max_tokens(),
        "stream": True,
    }
    tp = top_p()
    if tp is not None:
        body["top_p"] = tp
    budget = thinking_budget()
    if budget > 0:
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    thinking_blocks: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
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
                if isinstance(ev.get("usage"), dict):
                    usage = ev["usage"]
                delta = (((ev.get("choices") or [{}])[0]).get("delta") or {})
                chunk = delta.get("content") or ""
                if chunk:
                    text_parts.append(chunk)
                    await sink(chunk)
                
                # DeepSeek/OpenAI reasoning_content field (KEY FIELD!)
                if delta.get("reasoning_content"):
                    thinking_blocks.append({"content": delta.get("reasoning_content") or ""})
                # Anthropic thinking format (delta.type == "thinking")
                elif delta.get("type") == "thinking":
                    thinking_blocks.append({"content": delta.get("thinking") or delta.get("content") or ""})
                # OpenAI extended thinking format (delta.thinking exists)
                elif delta.get("thinking"):
                    thinking_blocks.append({"content": delta.get("thinking") or ""})
                # Custom format: top-level thinking field
                elif ev.get("thinking"):
                    thinking_blocks.append({"content": ev.get("thinking") or ""})
                
                # OpenAI tool_calls format (streamed: name/arguments arrive in chunks)
                if delta.get("tool_calls"):
                    for tc in delta.get("tool_calls", []):
                        idx = tc.get("index", 0)
                        func = tc.get("function") or {}
                        while len(tool_calls) <= idx:
                            tool_calls.append({"name": "", "arguments_buf": ""})
                        if func.get("name"):
                            tool_calls[idx]["name"] = func["name"]
                        if func.get("arguments"):
                            tool_calls[idx]["arguments_buf"] += func["arguments"]
                # Anthropic tool_use format
                elif delta.get("type") == "tool_use":
                    tool_calls.append({"name": delta.get("name") or "", "arguments_buf": json.dumps(delta.get("input") or {})})
    final_tool_calls = []
    for tc in tool_calls:
        try:
            args = json.loads(tc.get("arguments_buf") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        final_tool_calls.append({"name": tc.get("name", ""), "input": args})
    final_text = "".join(text_parts).strip()
    return {
        "text": final_text,
        "usage": usage,
        "thinking": thinking_blocks if thinking_blocks else None,
        "tool_calls": final_tool_calls if final_tool_calls else None
    }


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
    async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
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


async def complete_chat(route: dict[str, Any], messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, *, disable_thinking: bool = False) -> dict[str, Any]:
    body = {
        "model": route["model"],
        "messages": messages,
        "temperature": temperature(),
        "max_tokens": max_tokens(),
        "stream": True,
    }
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
    thinking_blocks: list[dict[str, Any]] = []
    tool_calls_buf: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    raw_msg: dict[str, Any] = {}

    body_keys = [k for k in body if k not in ("messages",)]
    body_summary = {k: (body[k] if k != "tools" else f"[{len(body[k])} tools]") for k in body_keys}
    print(f"[api_loop:complete_chat] request body keys: {body_summary}")

    finish_reason_val = None
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
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
                if isinstance(ev.get("usage"), dict):
                    usage = ev["usage"]
                choice0 = (ev.get("choices") or [{}])[0]
                if choice0.get("finish_reason"):
                    finish_reason_val = choice0["finish_reason"]
                delta = (choice0.get("delta") or {})
                if delta.get("role"):
                    raw_msg["role"] = delta["role"]
                chunk = delta.get("content") or ""
                if chunk:
                    text_parts.append(chunk)
                if delta.get("reasoning_content"):
                    thinking_blocks.append({"content": delta["reasoning_content"]})
                elif delta.get("type") == "thinking":
                    thinking_blocks.append({"content": delta.get("thinking") or delta.get("content") or ""})
                elif delta.get("thinking"):
                    thinking_blocks.append({"content": delta["thinking"]})
                elif ev.get("thinking"):
                    thinking_blocks.append({"content": ev["thinking"]})
                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        func = tc.get("function") or {}
                        while len(tool_calls_buf) <= idx:
                            tool_calls_buf.append({"id": "", "name": "", "arguments_buf": ""})
                        if tc.get("id"):
                            tool_calls_buf[idx]["id"] = tc["id"]
                        if func.get("name"):
                            tool_calls_buf[idx]["name"] = func["name"]
                        if func.get("arguments"):
                            tool_calls_buf[idx]["arguments_buf"] += func["arguments"]

    merged_thinking = []
    if thinking_blocks:
        merged = "".join(b.get("content", "") for b in thinking_blocks)
        if merged.strip():
            merged_thinking.append({"content": merged})

    tool_calls_parsed: list[dict[str, Any]] = []
    raw_tool_calls = []
    for tc in tool_calls_buf:
        name = (tc.get("name") or "").strip()
        if not name:
            continue  # 跳过流式解析产生的空 name 幽灵 tool_call，避免下发 "MCP tool is not configured"
        try:
            args = json.loads(tc.get("arguments_buf") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        tool_calls_parsed.append({"name": name, "input": args})
        raw_tool_calls.append({
            "id": tc.get("id", ""),
            "type": "function",
            "function": {"name": name, "arguments": tc.get("arguments_buf", "{}")}
        })

    final_text = "".join(text_parts).strip()
    if "role" not in raw_msg:
        raw_msg["role"] = "assistant"
    raw_msg["content"] = final_text
    if raw_tool_calls:
        raw_msg["tool_calls"] = raw_tool_calls
    print(f"[api_loop:complete_chat] has_reasoning_content={bool(thinking_blocks)}, has_tool_calls={bool(tool_calls_parsed)}, text_len={len(final_text)}, finish_reason={finish_reason_val}")
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


def _prompt_tools_block(tools: list[dict[str, Any]]) -> str:
    lines = ["You have the following tools available. To call a tool, output EXACTLY this format (no markdown, no extra text around it):",
             "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}</tool_call>",
             "",
             "After you receive the tool result, continue your response to the user.",
             "You may call multiple tools in sequence if needed. Only call a tool when it is genuinely useful.",
             "",
             "Available tools:"]
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


async def _prompt_tool_loop(route: dict[str, Any], messages: list[dict[str, Any]], tools: list[dict[str, Any]], max_rounds: int = 8) -> dict[str, Any]:
    system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    if system_msg:
        system_msg["content"] = system_msg["content"].rstrip() + "\n\n" + _prompt_tools_block(tools)
    else:
        messages.insert(0, {"role": "system", "content": _prompt_tools_block(tools)})
    last_out: dict[str, Any] = {"text": "", "usage": {}}
    tool_calls_collected: list[dict[str, Any]] = []
    for _ in range(max_rounds):
        out = await complete_chat(route, messages)
        text = out.get("text") or ""
        last_out = out
        matches = list(_TOOL_CALL_RE.finditer(text))
        if not matches:
            break
        for m in matches:
            try:
                call = json.loads(m.group(1))
                tool_name = str(call.get("name") or "")
                tool_args = call.get("arguments") or {}
                if not isinstance(tool_args, dict):
                    tool_args = {}
                result = await execute_mcp_tool(tool_name, tool_args)
                result_str = json.dumps(result, ensure_ascii=False)
                tool_calls_collected.append({"name": tool_name, "input": tool_args, "result": result})
            except Exception as exc:
                result_str = json.dumps({"error": str(exc)}, ensure_ascii=False)
                tool_calls_collected.append({"name": tool_name, "input": tool_args, "result": {"error": str(exc)}})
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"<tool_result name=\"{tool_name}\">{result_str}</tool_result>"})
            text = ""
        if not text:
            continue
    final_text = (last_out.get("text") or "")
    final_text = _TOOL_CALL_RE.sub("", final_text).strip()
    if final_text != last_out.get("text"):
        last_out["text"] = final_text
    if tool_calls_collected:
        last_out["tool_calls"] = tool_calls_collected
    return last_out


async def run_model(messages: list[dict[str, Any]], *, stream_id: str = "", session_id: str = "", emit_stream: bool = False) -> dict[str, Any]:
    tried = []
    last_error = ""
    for route in main_chain():
        tried.append(route.get("model"))
        try:
            all_tools = await mcp_tools()
            route_key = (route.get("url", "").rstrip("/"), route.get("model", ""))
            use_prompt_tools = (route_key in _TOOLS_UNSUPPORTED_ROUTES and bool(all_tools)) if not FORCE_NATIVE_TOOLS else False
            native_tools = [] if use_prompt_tools else all_tools
            suppress_thinking = (route_key in _THINKING_TOOLS_CONFLICT and bool(native_tools) and thinking_budget() > 0)
            tool_names = [t.get("function", {}).get("name", "") for t in native_tools] if native_tools else []
            print(f"[api_loop:run_model] mcp_tools={len(all_tools)}, native_tools={len(native_tools)}, prompt_tools={use_prompt_tools}, suppress_thinking={suppress_thinking}, tool_names={tool_names[:5]}")
            if use_prompt_tools:
                out = await _prompt_tool_loop(route, messages, all_tools)
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
                    out = await stream_chat(route, messages, sink)
                except HTTPException as exc:
                    if exc.status_code not in FALLBACK_CODES:
                        raise
                    out = await complete_chat(route, messages, native_tools)
            else:
                base_messages = messages[:]
                tool_calls_collected: list[dict[str, Any]] = []
                first_thinking = None
                try:
                    for round_idx in range(8):
                        out = await complete_chat(route, messages, native_tools, disable_thinking=suppress_thinking)
                        if round_idx == 0 and out.get("thinking"):
                            first_thinking = out["thinking"]
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
                                content = json.dumps(result, ensure_ascii=False)
                                tool_calls_collected.append({"name": tool_name, "input": args, "result": result, "status": "success"})
                            except Exception as exc:
                                content = json.dumps({"error": str(exc)}, ensure_ascii=False)
                                tool_calls_collected.append({"name": tool_name, "input": args, "result": {"error": str(exc)}, "status": "error"})
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
                                            tool_calls_collected.append({"name": tool_name, "input": args, "result": result, "status": "success"})
                                        except Exception as e2:
                                            content_str = json.dumps({"error": str(e2)}, ensure_ascii=False)
                                            tool_calls_collected.append({"name": tool_name, "input": args, "result": {"error": str(e2)}, "status": "error"})
                                        messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": content_str})
                                else:
                                    out = {"text": "", "usage": {}}
                                if tool_calls_collected:
                                    out["tool_calls"] = tool_calls_collected
                            except Exception:
                                out = await complete_chat(route, base_messages)
                        else:
                            print(f"[api_loop:run_model] marking route as tools-unsupported: {route_key}")
                            _TOOLS_UNSUPPORTED_ROUTES.add(route_key)
                            try:
                                out = await _prompt_tool_loop(route, base_messages, all_tools)
                            except Exception:
                                out = await complete_chat(route, base_messages)
                    elif not native_tools or exc.status_code not in {404, 405, 422}:
                        raise
                    else:
                        out = await complete_chat(route, base_messages)
            out["model"] = route.get("model")
            out["tried"] = tried[:-1]
            return out
        except HTTPException as exc:
            if exc.status_code not in FALLBACK_CODES:
                raise
            last_error = f"HTTP {exc.status_code}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return {"text": "", "error": last_error or "all models failed", "tried": tried}


async def handle_ingest(text: str, msg_id: int | None, session_id: str, *, dry: bool = False) -> dict[str, Any]:
    stream_id = "api-" + uuid.uuid4().hex[:16]
    messages = build_messages(text, before_id=msg_id, session_id=session_id, use_context=True)
    out = await run_model(messages, stream_id=stream_id, session_id=session_id, emit_stream=not dry)
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
    print(f"[api_loop:handle_ingest] model={out.get('model')}, has_thinking={bool(out.get('thinking'))}, has_tool_calls={bool(out.get('tool_calls'))}")
    if out.get("thinking"):
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


app = FastAPI(title="companion-api-loop")


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


@app.post("/loop/chat")
async def loop_chat(request: Request):
    body = await request.json()
    text = str(body.get("text") or body.get("message") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    session_id = str(body.get("session_id") or body.get("api_session") or active_session_id() or "").strip()
    messages = build_messages(text, before_id=None, session_id=session_id, use_context=bool(body.get("use_context", True)))
    out = await run_model(messages, emit_stream=False)
    return {"ok": True, "reply": out.get("text") or "", "api": out}


@app.post("/loop/debug-chat")
async def loop_debug_chat(request: Request):
    params: dict[str, Any] = {}
    try:
        params = await request.json()
    except Exception:
        pass
    route = main_chain()[0] if main_chain() else None
    if not route:
        raise HTTPException(status_code=503, detail="no main_chain configured")
    prompt = str(params.get("prompt") or params.get("text") or "hello")
    with_tools = bool(params.get("with_tools", False))
    tools: list[dict[str, Any]] = []
    if with_tools:
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
    return {"status": resp.status_code, "url": url, "request_model": body["model"], "tools_count": len(tools), "tool_names": [t["function"]["name"] for t in body.get("tools", [])], "response_headers": dict(resp.headers), "response_body": resp_body}


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
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    msg_id = body.get("id")
    try:
        before_id = int(msg_id) if msg_id is not None else None
    except Exception:
        before_id = None
    session_id = str(body.get("session_id") or body.get("api_session") or active_session_id() or "").strip()
    dry = bool(body.get("dry"))
    return await handle_ingest(text, before_id, session_id, dry=dry)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=LOOP_PORT)
