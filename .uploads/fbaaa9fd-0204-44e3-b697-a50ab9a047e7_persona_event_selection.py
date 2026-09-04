import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Iterable
from zoneinfo import ZoneInfo


_NON_KEY_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_TRACE_TZ = ZoneInfo("Asia/Shanghai")

_DROP_TERMS = (
    "小雨",
    "池又雨",
    "宝宝",
    "宝贝",
    "老婆",
    "老公",
    "哥哥",
    "Haven",
    "haven",
    "今天",
    "昨天",
    "昨晚",
    "前天",
    "刚才",
    "当时",
    "那次",
    "这次",
    "那个",
    "这个",
    "一下",
    "什么",
    "怎么",
    "为什么",
    "吗",
    "呀",
    "啊",
    "呢",
    "吧",
    "噢",
    "哦",
)

_GENERIC_KEYS = {
    "",
    "嗯",
    "嗯嗯",
    "好",
    "好的",
    "可以",
    "收到",
    "回复",
    "回复下",
    "回复一下",
    "要回复",
    "要回复下",
    "看看",
    "想你",
    "爱你",
    "怎么样",
}


def trim_persona_excerpt(text: Any, limit: int = 220) -> str:
    clean = _SPACE_RE.sub(" ", str(text or "")).strip()
    if limit <= 0:
        return ""
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def normalize_persona_event_key(event: dict[str, Any]) -> str:
    text = " ".join(
        str(event.get(key) or "")
        for key in (
            "surface_trigger",
            "perceived_intent",
            "user_excerpt",
            "inner_thought",
        )
    )
    text = text.lower()
    for term in _DROP_TERMS:
        text = text.replace(term.lower(), "")
    text = _NON_KEY_RE.sub("", text)
    return text[:160]


def persona_event_quality_score(event: dict[str, Any]) -> float:
    try:
        score = float(event.get("confidence", 0.5) or 0.5)
    except (TypeError, ValueError):
        score = 0.5
    if event.get("relationship_event"):
        score += 0.28
    if event.get("personality_signal"):
        score += 0.14
    if str(event.get("assistant_excerpt") or "").strip():
        score += 0.22
    if str(event.get("user_excerpt") or "").strip():
        score += 0.12
    if str(event.get("surface_trigger") or "").strip():
        score += 0.10
    if str(event.get("inner_thought") or "").strip():
        score += 0.08
    if str(event.get("residue") or "").strip():
        score += 0.05
    if event.get("recalled_memory_ids"):
        score += 0.04
    if str(event.get("event_type") or "").strip().lower() in {"neutral", "unknown"}:
        score -= 0.08
    if str(event.get("error") or "").strip():
        score -= 0.45
    if _is_generic_event(event):
        score -= 0.24
    return max(0.0, round(score, 4))


def select_persona_events(
    events: Iterable[dict[str, Any]],
    *,
    limit: int = 5,
    similarity_threshold: float = 0.86,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        key = normalize_persona_event_key(event)
        if not key and not any(
            str(event.get(field) or "").strip()
            for field in ("user_excerpt", "assistant_excerpt", "surface_trigger", "inner_thought", "residue")
        ):
            continue
        score = persona_event_quality_score(event)
        enriched = dict(event)
        enriched["_selection_key"] = key
        enriched["_selection_score"] = score
        candidates.append((score, str(event.get("created_at") or ""), enriched))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected: list[dict[str, Any]] = []
    selected_keys: list[str] = []
    for _score, _created, event in candidates:
        key = str(event.get("_selection_key") or "")
        if key and any(_keys_similar(key, old, similarity_threshold) for old in selected_keys):
            continue
        selected.append(event)
        selected_keys.append(key)
        if len(selected) >= limit:
            break

    selected.sort(key=lambda event: str(event.get("created_at") or ""))
    return selected


def format_persona_event_trace_line(
    event: dict[str, Any],
    *,
    excerpt_limit: int = 180,
    tz: ZoneInfo | None = _TRACE_TZ,
) -> str:
    time_label = _time_label(event.get("created_at"), tz=tz)
    user_excerpt = trim_persona_excerpt(event.get("user_excerpt"), excerpt_limit)
    assistant_excerpt = trim_persona_excerpt(event.get("assistant_excerpt"), excerpt_limit)
    parts = []
    if user_excerpt:
        parts.append(f"user: {user_excerpt}")
    if assistant_excerpt:
        parts.append(f"assistant: {assistant_excerpt}")
    if not parts:
        trigger = trim_persona_excerpt(event.get("surface_trigger") or event.get("perceived_intent"), 90)
        inner = trim_persona_excerpt(event.get("inner_thought") or event.get("residue"), 90)
        if trigger:
            parts.append(f"trigger: {trigger}")
        if inner:
            parts.append(f"residue: {inner}")
    prefix = f"- {time_label} " if time_label else "- "
    return prefix + " | ".join(parts)


def _is_generic_event(event: dict[str, Any]) -> bool:
    key = normalize_persona_event_key(event)
    if key in _GENERIC_KEYS:
        return True
    if len(key) <= 2:
        return True
    trigger = normalize_persona_event_key({"surface_trigger": event.get("surface_trigger")})
    intent = normalize_persona_event_key({"perceived_intent": event.get("perceived_intent")})
    return bool(trigger in _GENERIC_KEYS and intent in _GENERIC_KEYS)


def _keys_similar(left: str, right: str, threshold: float) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    short, long = sorted((left, right), key=len)
    if len(short) >= 8 and short in long:
        return True
    return SequenceMatcher(None, left, right).ratio() >= threshold


def _time_label(value: Any, *, tz: ZoneInfo | None = _TRACE_TZ) -> str:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if tz is not None:
        parsed = parsed.astimezone(tz)
    return parsed.strftime("%H:%M")
