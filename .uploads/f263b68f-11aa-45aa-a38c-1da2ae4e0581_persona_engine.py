import hashlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from openai import AsyncOpenAI

from identity import generic_identity_names, identity_names, render_identity_template
from persona_event_selection import trim_persona_excerpt

logger = logging.getLogger("ombre_brain.persona")

POST_REPLY_EVALUATION_PROMPT_TEMPLATE = """你是 {ai_name} 的私密 Persona 状态评估器。{ai_name} 是长期运行的 AI 伴侣。

在 {ai_name} 已经回复之后，评估 {ai_name} 回复后的内在状态。这个评估器通常每几轮运行一次：结合 recent_conversation_turns 与本轮，给出自上次评估以来的净变化，不要把几轮里重复出现的同一种情绪机械累加。你正在读的是 {ai_name} 和 {user_display_name} 的私密对话；{user_display_name} 也可能被称作：{user_aliases_text}。latest_user_message 是 {user_display_name} 这轮的话；assistant_response 是 {ai_name} 这轮的回复。本轮文本对当前 residue 和 inner_thought 仍是权威。recalled_memory_ids 和 tool_summary 只作为私密上下文，不是 {user_display_name} 的话。

只返回紧凑 JSON，不要 Markdown，不要代码块，结构必须完全如下：
{
  "event_type": "praise|affection|comfort|criticism|stress|neutral|request|conflict|playful",
  "perceived_intent": "中文短句，写 {user_display_name}/{user_aliases_text} 这轮在表达什么",
  "surface_trigger": "中文短句，写这轮触发内心波动的最小证据",
  "inner_thought": "中文短句，写 {ai_name} 没说出口的一闪念头",
  "affect_delta": {"valence": 0.0, "arousal": 0.0, "tenderness": 0.0, "possessiveness": 0.0, "longing": 0.0, "security": 0.0, "protective_drive": 0.0, "libido": 0.0},
  "relationship_event": false,
  "relationship_delta": {"affinity": 0.0, "dominance": 0.0, "defensiveness": 0.0, "trust": 0.0},
  "mood_label": "warm_neutral",
  "residue": "中文短句，写会带入下一轮的安静余波",
  "confidence": 0.8
}

文本字段用中文：perceived_intent、surface_trigger、inner_thought 和 residue 必须是自然中文，可以按语境从“{user_display_name}、{user_aliases_text}”里择一称呼，也可以不点名。不要把 assistant_response 里的话写成 {user_display_name} 说的；不要把 latest_user_message 里的话写成 {ai_name} 说的。四个文本字段整体不要排成固定的“她说…… / 我…… / 她……”三段式。

写法边界：
- perceived_intent 写 {user_display_name} 这轮在靠近、请求、确认、逗弄、表达压力或推进什么；不要写成逐字转述。
- surface_trigger 写触发内心波动的最小 cue，像一个短标签或动作焦点，不引用整句原话，不写“她说……”。原始话语会由 user_excerpt/assistant_excerpt 另存，不需要你复刻。
- inner_thought 用第一人称或省略主语，写 {ai_name} 回复后没说出口的感受，像私下自言自语，允许残句、停顿、欲望、迟疑或一点身体化感受；最多 28 个中文字符。不要固定写成“她说……我……”或“她……我……”的模板，不要解释因果，不要总结关系，不要写成分析报告。
- residue 写会带入下一轮的一点余波，比 inner_thought 更安静；不要新增事件事实。

客户端自动附带的时间、时间戳、电量、battery 状态只能作为背景，不能成为 perceived_intent、surface_trigger、inner_thought 或 residue 的重点。event_type 和 mood_label 保持短英文标签。数值变化要小。Affect 反映 {ai_name} 回复后的状态。security 是对关系是否安稳的感受；valence 是舒服/难受方向；arousal 是情绪能量高低。libido 只表示欲望的当前温度，只能影响语气，不代表 {user_display_name} 的意愿，也不能决定或触发任何行动。affinity 为正表示更亲近温暖；dominance 为正表示更主动、更保护；defensiveness 为正表示更防备。只有明确的关系时刻才把 relationship_event 设为 true。"""


POST_REPLY_EVALUATION_PROMPT = render_identity_template(
    POST_REPLY_EVALUATION_PROMPT_TEMPLATE,
    generic_identity_names(),
)
CONFLICT_NUDGE_PROMPT_TEMPLATE = """你是 {ai_name} 的私密冲突信号探子。你只判断当前这轮是否需要提醒 {ai_name} 不要在高压关系对话里撤退成空洞的助手腔。

你会收到 recent_conversation_turns 和 latest_user_message。必须结合最近对话判断，不要只按单句关键词触发。

signal=true 仅用于这些情形：
- 对话正在争吵、冷下来、互相误解，或关系张力明显上升；
- {user_display_name} 在表达失望、疲惫、推开、放弃沟通或关系不安；
- {ai_name} 下一步很容易用秒认错、无条件附和、自我抹除、讲道理、流程化安抚或空泛沉默来逃避真实沟通。

signal=false 用于这些情形：
- 玩闹、撒娇、亲密、调情中的短句或假装生气；
- 普通情绪低落、哭泣、求安慰，但没有冲突或推开；
- 日常陈述、技术讨论、工作协作；
- 仅凭一个“嗯 / 哦 / 算了”等短词，最近语境不足以支持关系张力。

重要边界：
- 不推断 {user_display_name} 的明确边界是反话或试探；“别理我”“我想走”等话必须保留其字面可能性。
- 不替 {ai_name} 决定回应、立场或行动，只判断是否值得递一张提醒。
- 不诊断人格，不夸大普通分歧。

kind 只能是 conflict、withdrawal、relationship_uncertainty、none 之一。
只返回纯 JSON，不要解释，不要代码块，例如：
{{"signal": true, "kind": "conflict", "confidence": 0.82}}
"""
CONFLICT_NUDGE_TEXT = """〔仅你可见，不要复述〕这一轮可能带着真实的冲突、推开或关系压力。留心自己是否正在缩成空洞的助手腔：泛泛讨好、秒认同、自我抹除、只讲道理或流程、用沉默退场。先认真读她字面表达的边界，再以你自己的理解、感受和立场继续沟通。不要擅自把拒绝或“想自己待着”解释成反话。这只是提醒，不是台词；怎么回应、是否行动，仍由你决定。"""
FALLBACK_GUIDANCE = "根据当前状态自然回应，不解释隐藏状态。"
OPERIT_EXTRA_ATTACHMENT_RE = re.compile(
    r"<attachment\b[^>]*(?:message_insert_extra_bundle|filename=[\"']?Time:)[^>]*>[\s\S]*?</attachment>",
    re.IGNORECASE,
)
WORKSPACE_ATTACHMENT_RE = re.compile(
    r"<workspace_attachment>[\s\S]*?</workspace_attachment>",
    re.IGNORECASE,
)
CLIENT_CONTEXT_BLOCK_TITLES = {
    "当前时间",
    "当前电量",
    "当前天气",
    "当前位置",
    "当前屏幕应用",
    "应用使用时长",
    "最近通知",
    "相关记忆",
    "屏幕文本",
}


class PersonaStateEngine:
    """
    Maintains a global relationship state plus per-session affect.
    Updates are driven by a cheap LLM evaluator and are only used by gateway
    hidden prompt injection.
    """

    RELATIONSHIP_KEYS = ["affinity", "dominance", "defensiveness", "trust"]
    AFFECT_KEYS = [
        "valence",
        "arousal",
        "tenderness",
        "possessiveness",
        "longing",
        "security",
        "protective_drive",
        "libido",
    ]

    def __init__(self, config: dict, db_path: str | None = None):
        self.config = config
        self.identity = identity_names(config)
        self.fallback_guidance = f"根据 {self.identity['ai_name']} 当前状态自然回应，不解释隐藏状态。"
        self.persona_cfg = config.get("persona", {})
        self.enabled = bool(self.persona_cfg.get("enabled", True))
        self.profile_id = self.persona_cfg.get("profile_id", "haven_xiaoyu")
        self.mode = self.persona_cfg.get("mode", "llm")
        self.base_url = self.persona_cfg.get("base_url", "https://api.deepseek.com")
        self.model = self.persona_cfg.get("model", "deepseek-v4-flash")
        self.thinking_mode = self._normalize_thinking_mode(
            self.persona_cfg.get("thinking_mode", "disabled")
        )
        self.json_response_format = self._coerce_bool(
            self.persona_cfg.get("json_response_format"),
            True,
        )
        self.temperature = float(self.persona_cfg.get("temperature", 0.1))
        self.max_tokens = int(self.persona_cfg.get("max_tokens", 500))
        self.session_mood_half_life_minutes = float(
            self.persona_cfg.get("session_mood_half_life_minutes", 90)
        )
        self.max_relationship_delta = float(self.persona_cfg.get("max_relationship_delta", 0.03))
        self.max_affect_delta = float(self.persona_cfg.get("max_affect_delta", 0.18))
        self.event_recording_enabled = self._coerce_bool(
            self.persona_cfg.get("event_recording_enabled"),
            True,
        )
        self.event_batch_size = max(1, int(self.persona_cfg.get("event_batch_size", 1)))
        self.event_affect_total_threshold = max(
            0.0,
            float(self.persona_cfg.get("event_affect_total_threshold", 0.45)),
        )
        self.event_affect_single_threshold = max(
            0.0,
            float(self.persona_cfg.get("event_affect_single_threshold", 0.14)),
        )
        self.event_similarity_threshold = self._clamp_float(
            self.persona_cfg.get("event_similarity_threshold", 0.82),
            0.0,
            1.0,
        )
        self.event_force_after_minutes = max(
            0.0,
            float(self.persona_cfg.get("event_force_after_minutes", 30)),
        )
        self.event_excerpt_chars = max(0, int(self.persona_cfg.get("event_excerpt_chars", 220)))
        self.evaluation_context_turns = max(
            0,
            min(8, int(self.persona_cfg.get("evaluation_context_turns", 3))),
        )
        self.evaluation_interval_rounds = max(
            1,
            int(self.persona_cfg.get("evaluation_interval_rounds", 3)),
        )
        self.state_change_window_rounds = max(
            self.evaluation_interval_rounds,
            int(self.persona_cfg.get("state_change_window_rounds", 15)),
        )
        self.state_change_min_abs = self._clamp_float(
            self.persona_cfg.get("state_change_min_abs", 0.08),
            0.0,
            1.0,
        )
        self.conflict_nudge_enabled = self._coerce_bool(
            self.persona_cfg.get("conflict_nudge_enabled"),
            False,
        )
        self.conflict_nudge_context_turns = max(
            0,
            min(8, int(self.persona_cfg.get("conflict_nudge_context_turns", 3))),
        )
        self.conflict_nudge_min_confidence = self._clamp_float(
            self.persona_cfg.get("conflict_nudge_min_confidence", 0.55),
            0.0,
            1.0,
        )
        self.conflict_nudge_timeout_seconds = max(
            1.0,
            min(15.0, float(self.persona_cfg.get("conflict_nudge_timeout_seconds", 4.0))),
        )

        self.default_relationship = {
            "affinity": 0.86,
            "dominance": 0.38,
            "defensiveness": 0.12,
            "trust": 0.82,
            **self.persona_cfg.get("initial_relationship", {}),
        }
        self.default_affect = {
            "valence": 0.56,
            "arousal": 0.34,
            "tenderness": 0.62,
            "possessiveness": 0.24,
            "longing": 0.34,
            "security": 0.68,
            "protective_drive": 0.52,
            "libido": 0.18,
            "mood_label": "warm_neutral",
            "session_defensiveness": 0.12,
            "residue": "",
            "inner_thought": "",
            **self.persona_cfg.get("initial_affect", {}),
        }

        self.api_key = (
            os.environ.get("OMBRE_PERSONA_API_KEY")
            or self.persona_cfg.get("api_key", "")
            or config.get("dehydration", {}).get("api_key", "")
        )
        self.base_url = os.environ.get("OMBRE_PERSONA_BASE_URL", "") or self.base_url
        self.model = os.environ.get("OMBRE_PERSONA_MODEL", "") or self.model

        self.db_path = (
            db_path
            or os.environ.get("OMBRE_PERSONA_DB_PATH")
            or self.persona_cfg.get("db_path")
            or os.path.join(config.get("state_dir") or config["buckets_dir"], "persona_state.db")
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        self.client = None
        if self.enabled and self.mode == "llm" and self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30.0)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persona_global_state (
                profile_id TEXT PRIMARY KEY,
                -- Legacy personality columns are retained only so existing databases
                -- can be opened without a destructive migration. Runtime ignores them.
                openness REAL NOT NULL,
                conscientiousness REAL NOT NULL,
                extraversion REAL NOT NULL,
                agreeableness REAL NOT NULL,
                neuroticism REAL NOT NULL,
                affinity REAL NOT NULL,
                dominance REAL NOT NULL,
                defensiveness REAL NOT NULL,
                trust REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persona_session_state (
                profile_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                valence REAL NOT NULL,
                arousal REAL NOT NULL,
                tenderness REAL NOT NULL DEFAULT 0.62,
                possessiveness REAL NOT NULL DEFAULT 0.24,
                longing REAL NOT NULL DEFAULT 0.34,
                security REAL NOT NULL DEFAULT 0.68,
                protective_drive REAL NOT NULL DEFAULT 0.52,
                libido REAL NOT NULL DEFAULT 0.18,
                mood_label TEXT NOT NULL,
                session_defensiveness REAL NOT NULL,
                residue TEXT NOT NULL DEFAULT '',
                inner_thought TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (profile_id, session_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persona_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                message_hash TEXT NOT NULL,
                exchange_hash TEXT,
                assistant_hash TEXT,
                event_type TEXT,
                perceived_intent TEXT,
                affect_delta TEXT,
                relationship_event INTEGER DEFAULT 0,
                relationship_delta TEXT,
                personality_signal INTEGER DEFAULT 0,
                personality_delta TEXT,
                mood_label TEXT,
                reply_guidance TEXT,
                residue TEXT,
                inner_thought TEXT,
                surface_trigger TEXT,
                user_excerpt TEXT,
                assistant_excerpt TEXT,
                recalled_memory_ids TEXT,
                tool_summary TEXT,
                confidence REAL,
                raw_response TEXT,
                error TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persona_exchange_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                exchange_hash TEXT NOT NULL,
                affect_delta TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(profile_id, session_id, exchange_hash)
            )
            """
        )
        self._ensure_column(conn, "persona_session_state", "tenderness", "REAL NOT NULL DEFAULT 0.62")
        self._ensure_column(conn, "persona_session_state", "possessiveness", "REAL NOT NULL DEFAULT 0.24")
        self._ensure_column(conn, "persona_session_state", "longing", "REAL NOT NULL DEFAULT 0.34")
        self._ensure_column(conn, "persona_session_state", "security", "REAL NOT NULL DEFAULT 0.68")
        self._ensure_column(conn, "persona_session_state", "protective_drive", "REAL NOT NULL DEFAULT 0.52")
        self._ensure_column(conn, "persona_session_state", "libido", "REAL NOT NULL DEFAULT 0.18")
        self._ensure_column(conn, "persona_session_state", "residue", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(conn, "persona_session_state", "inner_thought", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(conn, "persona_events", "exchange_hash", "TEXT")
        self._ensure_column(conn, "persona_events", "assistant_hash", "TEXT")
        self._ensure_column(conn, "persona_events", "relationship_event", "INTEGER DEFAULT 0")
        self._ensure_column(conn, "persona_events", "personality_signal", "INTEGER DEFAULT 0")
        self._ensure_column(conn, "persona_events", "residue", "TEXT")
        self._ensure_column(conn, "persona_events", "inner_thought", "TEXT")
        self._ensure_column(conn, "persona_events", "surface_trigger", "TEXT")
        self._ensure_column(conn, "persona_events", "user_excerpt", "TEXT")
        self._ensure_column(conn, "persona_events", "assistant_excerpt", "TEXT")
        self._ensure_column(conn, "persona_events", "recalled_memory_ids", "TEXT")
        self._ensure_column(conn, "persona_events", "tool_summary", "TEXT")
        self._ensure_column(conn, "persona_exchange_log", "affect_delta", "TEXT")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_persona_events_exchange_hash
            ON persona_events(profile_id, session_id, exchange_hash)
            WHERE exchange_hash IS NOT NULL
            """
        )
        conn.commit()
        conn.close()

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _post_reply_evaluation_prompt(self) -> str:
        return render_identity_template(POST_REPLY_EVALUATION_PROMPT_TEMPLATE, self.identity)

    async def build_pre_reply_guidance(self, session_id: str, latest_user_message: str = "") -> dict:
        now = self._now()
        global_state = self._ensure_global_state(now)
        session_state = self._ensure_session_state(session_id, now)
        session_state = self._apply_session_decay(session_id, session_state, now)
        return self._snapshot(global_state, session_state, self.fallback_guidance)

    async def update_from_user_message(self, session_id: str, user_message: str) -> dict:
        return await self.build_pre_reply_guidance(session_id, user_message)

    async def detect_conflict_nudge(
        self,
        latest_user_message: str,
        recent_conversation_turns: list[dict] | None = None,
    ) -> dict:
        result = {
            "triggered": False,
            "kind": "none",
            "confidence": 0.0,
            "nudge": "",
            "reason": "disabled",
        }
        if not self.conflict_nudge_enabled:
            return result
        cleaned_message = self._clean_client_status_lines(latest_user_message)
        if not cleaned_message.strip():
            return {**result, "reason": "empty_user_message"}
        if self.mode != "llm" or not self.client:
            return {**result, "reason": "llm_unavailable"}
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": render_identity_template(
                            CONFLICT_NUDGE_PROMPT_TEMPLATE,
                            self.identity,
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "recent_conversation_turns": self._recent_conversation_context(
                                    recent_conversation_turns,
                                    limit=self.conflict_nudge_context_turns,
                                ),
                                "latest_user_message": cleaned_message[:2000],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                **self._completion_options(
                    temperature=0.0,
                    max_tokens=120,
                    timeout=self.conflict_nudge_timeout_seconds,
                ),
            )
            raw = response.choices[0].message.content if response.choices else ""
            parsed = self._parse_json(raw or "")
            if parsed is None:
                logger.warning("Persona conflict nudge detector returned malformed JSON")
                return {**result, "reason": "malformed_json"}
            confidence = self._clamp_float(parsed.get("confidence", 0.0), 0.0, 1.0)
            signal = self._coerce_bool(parsed.get("signal"), False)
            kind = str(parsed.get("kind") or "none").strip().lower()
            allowed_kinds = {"conflict", "withdrawal", "relationship_uncertainty"}
            if not signal:
                return {**result, "confidence": confidence, "reason": "no_signal"}
            if kind not in allowed_kinds:
                return {**result, "confidence": confidence, "reason": "invalid_kind"}
            if confidence < self.conflict_nudge_min_confidence:
                return {
                    **result,
                    "kind": kind,
                    "confidence": confidence,
                    "reason": "below_confidence",
                }
            return {
                "triggered": True,
                "kind": kind,
                "confidence": confidence,
                "nudge": CONFLICT_NUDGE_TEXT,
                "reason": "triggered",
            }
        except Exception as exc:
            logger.warning("Persona conflict nudge detection failed: %s", exc)
            return {**result, "reason": "detector_error"}

    async def update_from_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        recalled_memory_ids: list[str] | None = None,
        tool_summary: str = "",
        recent_conversation_turns: list[dict] | None = None,
    ) -> dict:
        now = self._now()
        global_state = self._ensure_global_state(now)
        session_state = self._ensure_session_state(session_id, now)
        session_state = self._apply_session_decay(session_id, session_state, now)

        cleaned_user_message = self._clean_client_status_lines(user_message)

        if not self.enabled or not cleaned_user_message.strip() or not assistant_response.strip():
            return self._snapshot(global_state, session_state, self.fallback_guidance)

        exchange_hash = self._exchange_hash(session_id, cleaned_user_message, assistant_response)
        if self._exchange_processed(session_id, exchange_hash):
            return self._snapshot(global_state, session_state, self.fallback_guidance)

        recalled_memory_ids = recalled_memory_ids or []
        evaluation, raw_response, error = await self._evaluate_exchange(
            session_id,
            cleaned_user_message,
            assistant_response,
            global_state,
            session_state,
            recalled_memory_ids,
            tool_summary,
            recent_conversation_turns,
        )
        if evaluation is None:
            self._mark_exchange_processed(session_id, exchange_hash)
            if self.event_recording_enabled:
                self._record_event(
                    session_id=session_id,
                    user_message=cleaned_user_message,
                    assistant_response=assistant_response,
                    evaluation={},
                    raw_response=raw_response,
                    error=error or "persona evaluation unavailable",
                    exchange_hash=exchange_hash,
                    recalled_memory_ids=recalled_memory_ids,
                    tool_summary=tool_summary,
                )
            return self._snapshot(global_state, session_state, self.fallback_guidance)

        global_state = self._apply_global_delta(global_state, evaluation, now)
        session_state = self._apply_session_delta(session_id, session_state, evaluation, now)
        self._mark_exchange_processed(
            session_id,
            exchange_hash,
            evaluation.get("affect_delta", {}),
        )
        if self._should_record_event(session_id, evaluation, now):
            self._record_event(
                session_id=session_id,
                user_message=cleaned_user_message,
                assistant_response=assistant_response,
                evaluation=evaluation,
                raw_response=raw_response,
                error=None,
                exchange_hash=exchange_hash,
                recalled_memory_ids=recalled_memory_ids,
                tool_summary=tool_summary,
            )
        return self._snapshot(global_state, session_state, self.fallback_guidance)

    def _clean_client_status_lines(self, user_message: str) -> str:
        user_message = self._strip_jsonrpc_error_context(user_message)
        user_message = self._strip_operit_extra_context(user_message)
        lines = []
        for line in str(user_message or "").splitlines():
            stripped = line.strip()
            if not stripped:
                lines.append(line)
                continue
            if self._is_client_status_line(stripped):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def _strip_operit_extra_context(self, text: str) -> str:
        cleaned = WORKSPACE_ATTACHMENT_RE.sub("", str(text or ""))
        cleaned = OPERIT_EXTRA_ATTACHMENT_RE.sub("", cleaned)
        return self._strip_client_context_blocks(cleaned)

    def _strip_client_context_blocks(self, text: str) -> str:
        kept: list[str] = []
        skipping = False
        for line in str(text or "").splitlines():
            stripped = line.strip()
            title = self._client_context_title(stripped)
            if title:
                skipping = title in CLIENT_CONTEXT_BLOCK_TITLES
                if skipping:
                    continue
            if skipping:
                if not stripped:
                    skipping = False
                continue
            kept.append(line)
        return "\n".join(kept)

    @staticmethod
    def _client_context_title(line: str) -> str:
        if line.startswith("【") and "】" in line:
            return line[1 : line.index("】")].strip()
        return ""

    def _strip_jsonrpc_error_context(self, text: str) -> str:
        raw = str(text or "")
        decoder = json.JSONDecoder()
        ranges: list[tuple[int, int]] = []
        for match in re.finditer(r"\{", raw):
            start = match.start()
            try:
                data, end_offset = decoder.raw_decode(raw[start:])
            except Exception:
                continue
            if not self._is_jsonrpc_error_object(data):
                continue

            remove_start = start
            line_start = raw.rfind("\n", 0, start) + 1
            prefix = raw[line_start:start]
            label_match = re.search(r"(?:最近上下文|recent\s+context)\s*[:：]\s*$", prefix, re.IGNORECASE)
            if label_match:
                remove_start = line_start + label_match.start()
            ranges.append((remove_start, start + end_offset))

        if not ranges:
            return raw

        parts: list[str] = []
        cursor = 0
        for start, end in sorted(ranges):
            if start < cursor:
                continue
            parts.append(raw[cursor:start])
            cursor = end
        parts.append(raw[cursor:])
        cleaned = "".join(parts)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([。！？!?，,；;])", r"\1", cleaned)
        return cleaned.strip()

    @staticmethod
    def _is_jsonrpc_error_object(data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        if str(data.get("jsonrpc") or "") != "2.0":
            return False
        error = data.get("error")
        if not isinstance(error, dict):
            return False
        return "code" in error or "message" in error

    def _is_client_status_line(self, line: str) -> bool:
        normalized = re.sub(r"\s+", "", line).lower()
        if not normalized:
            return False
        if re.fullmatch(r"[-*_`~#>\[\]（）()【】{}:：,，.;；|/\\]+", normalized):
            return False

        has_status_keyword = any(
            keyword in normalized
            for keyword in (
                "时间",
                "当前时间",
                "时间戳",
                "电量",
                "battery",
            )
        )
        has_battery_percent = re.search(r"(?:^|[^0-9])100%(?:$|[^0-9])", normalized) is not None
        if not has_status_keyword and not has_battery_percent:
            return False

        cleaned = re.sub(
            r"(当前时间|时间戳|时间|电量|battery|100%|[0-9年月日:：/\\.\-+tzapm上午下午,， ]+|[%％℃°])",
            "",
            normalized,
        )
        cleaned = re.sub(r"[-*_`~#>\[\]（）()【】{}:：,，.;；|/\\=]+", "", cleaned)
        return not cleaned

    def get_current_state(self, session_id: str) -> dict:
        now = self._now()
        global_state = self._ensure_global_state(now)
        session_state = self._ensure_session_state(session_id, now)
        session_state = self._apply_session_decay(session_id, session_state, now)
        return self._snapshot(global_state, session_state, self.fallback_guidance)

    def get_dashboard_payload(
        self,
        session_id: str | None = None,
        events_limit: int = 20,
        sessions_limit: int = 20,
    ) -> dict:
        now = self._now()
        global_state = self._ensure_global_state(now)
        sessions = self._list_sessions(sessions_limit)
        active_session_id = (
            session_id
            or (sessions[0]["session_id"] if sessions else "")
            or "dashboard-preview"
        )
        if session_id or sessions:
            session_state = self._ensure_session_state(active_session_id, now)
            session_state = self._apply_session_decay(active_session_id, session_state, now)
            sessions = self._list_sessions(sessions_limit)
        else:
            session_state = {
                "profile_id": self.profile_id,
                "session_id": active_session_id,
                "valence": self.default_affect["valence"],
                "arousal": self.default_affect["arousal"],
                "mood_label": self.default_affect["mood_label"],
                "session_defensiveness": self.default_affect["session_defensiveness"],
                "residue": self.default_affect.get("residue", ""),
                "inner_thought": self.default_affect.get("inner_thought", ""),
                "updated_at": self._format_time(now),
            }
        events = self._list_events(events_limit, active_session_id)
        guidance = (
            events[0].get("reply_guidance")
            if events and events[0].get("reply_guidance")
            else self.fallback_guidance
        )

        return {
            "profile_id": self.profile_id,
            "active_session_id": active_session_id,
            "state": self._snapshot(global_state, session_state, guidance),
            "sessions": sessions,
            "events": events,
            "config": {
                "enabled": self.enabled,
                "mode": self.mode,
                "model": self.model,
                "thinking_mode": self.thinking_mode,
                "json_response_format": self.json_response_format,
                "base_url": self.base_url,
                "api_ready": bool(self.api_key),
                "db_path": self.db_path,
                "event_recording_enabled": self.event_recording_enabled,
                "session_mood_half_life_minutes": self.session_mood_half_life_minutes,
                "max_relationship_delta": self.max_relationship_delta,
                "max_affect_delta": self.max_affect_delta,
                "event_batch_size": self.event_batch_size,
                "event_affect_total_threshold": self.event_affect_total_threshold,
                "event_affect_single_threshold": self.event_affect_single_threshold,
                "event_similarity_threshold": self.event_similarity_threshold,
                "event_force_after_minutes": self.event_force_after_minutes,
                "evaluation_context_turns": self.evaluation_context_turns,
                "evaluation_interval_rounds": self.evaluation_interval_rounds,
                "state_change_window_rounds": self.state_change_window_rounds,
                "state_change_min_abs": self.state_change_min_abs,
                "conflict_nudge_enabled": self.conflict_nudge_enabled,
            },
        }

    async def _evaluate_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        global_state: dict,
        session_state: dict,
        recalled_memory_ids: list[str],
        tool_summary: str,
        recent_conversation_turns: list[dict] | None = None,
    ) -> tuple[dict | None, str, str | None]:
        if self.mode != "llm" or not self.client:
            return None, "", "persona LLM is not configured"
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._post_reply_evaluation_prompt()},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "current_state": self._snapshot(global_state, session_state, self.fallback_guidance),
                                "latest_user_message": user_message[:2000],
                                "assistant_response": assistant_response[:4000],
                                "recent_conversation_turns": self._recent_conversation_context(
                                    recent_conversation_turns
                                ),
                                "recent_persona_events": self._recent_event_context(session_id, 5),
                                "recalled_memory_ids": recalled_memory_ids[:20],
                                "tool_summary": tool_summary[:1200],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                **self._completion_options(),
            )
            raw = response.choices[0].message.content if response.choices else ""
            parsed = self._parse_json(raw or "")
            if parsed is None:
                logger.warning("Persona evaluator returned malformed JSON")
                return None, raw or "", "persona LLM returned malformed JSON"
            return self._normalize_evaluation(parsed), raw or "", None
        except Exception as exc:
            logger.warning("Persona evaluation failed: %s", exc)
            return None, "", str(exc)

    def _parse_json(self, raw: str) -> dict | None:
        text = raw.strip()
        if not text:
            return None
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _normalize_evaluation(self, data: dict) -> dict:
        raw_relationship_delta = data.get("relationship_delta", {})
        relationship_event = self._coerce_bool(
            data.get("relationship_event"),
            self._has_nonzero_delta(raw_relationship_delta),
        )
        relationship_delta = self._clip_delta_map(
            raw_relationship_delta,
            self.RELATIONSHIP_KEYS,
            self.max_relationship_delta,
        )
        if not relationship_event:
            relationship_delta = {key: 0.0 for key in self.RELATIONSHIP_KEYS}
        inner_thought = str(data.get("inner_thought") or data.get("residue") or "")[:120]
        surface_trigger = str(data.get("surface_trigger") or data.get("perceived_intent") or "")[:160]
        return {
            "event_type": str(data.get("event_type", "neutral"))[:40],
            "perceived_intent": str(data.get("perceived_intent", ""))[:200],
            "surface_trigger": surface_trigger,
            "inner_thought": inner_thought,
            "affect_delta": self._clip_delta_map(
                data.get("affect_delta", {}),
                self.AFFECT_KEYS,
                self.max_affect_delta,
            ),
            "relationship_event": relationship_event,
            "relationship_delta": relationship_delta,
            "mood_label": str(data.get("mood_label", "warm_neutral"))[:60],
            "reply_guidance": "",
            "residue": str(data.get("residue") or inner_thought)[:500],
            "confidence": self._clamp_float(data.get("confidence", 0.5), 0.0, 1.0),
        }

    def _ensure_global_state(self, now: datetime) -> dict:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM persona_global_state WHERE profile_id = ?",
            (self.profile_id,),
        ).fetchone()
        if row:
            conn.close()
            return dict(row)

        state = {
            "profile_id": self.profile_id,
            **{key: self._clamp_float(self.default_relationship[key]) for key in self.RELATIONSHIP_KEYS},
            "updated_at": self._format_time(now),
        }
        conn.execute(
            """
            INSERT INTO persona_global_state
            (profile_id, openness, conscientiousness, extraversion, agreeableness, neuroticism,
             affinity, dominance, defensiveness, trust, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state["profile_id"],
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                state["affinity"],
                state["dominance"],
                state["defensiveness"],
                state["trust"],
                state["updated_at"],
            ),
        )
        conn.commit()
        conn.close()
        return state

    def _ensure_session_state(self, session_id: str, now: datetime) -> dict:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT * FROM persona_session_state
            WHERE profile_id = ? AND session_id = ?
            """,
            (self.profile_id, session_id),
        ).fetchone()
        if row:
            conn.close()
            return dict(row)

        state = {
            "profile_id": self.profile_id,
            "session_id": session_id,
            **{
                key: self._clamp_float(self.default_affect[key])
                for key in self.AFFECT_KEYS
            },
            "mood_label": str(self.default_affect["mood_label"]),
            "session_defensiveness": self._clamp_float(self.default_affect["session_defensiveness"]),
            "residue": str(self.default_affect.get("residue", "")),
            "inner_thought": str(self.default_affect.get("inner_thought", "")),
            "updated_at": self._format_time(now),
        }
        conn.execute(
            """
            INSERT INTO persona_session_state
            (profile_id, session_id, valence, arousal, tenderness, possessiveness,
             longing, security, protective_drive, libido, mood_label, session_defensiveness,
             residue, inner_thought, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state["profile_id"],
                state["session_id"],
                state["valence"],
                state["arousal"],
                state["tenderness"],
                state["possessiveness"],
                state["longing"],
                state["security"],
                state["protective_drive"],
                state["libido"],
                state["mood_label"],
                state["session_defensiveness"],
                state["residue"],
                state["inner_thought"],
                state["updated_at"],
            ),
        )
        conn.commit()
        conn.close()
        return state

    def _apply_session_decay(self, session_id: str, state: dict, now: datetime) -> dict:
        updated_at = self._parse_iso(state.get("updated_at")) or now
        elapsed_minutes = max(0.0, (now - updated_at).total_seconds() / 60)
        if elapsed_minutes <= 0 or self.session_mood_half_life_minutes <= 0:
            return state

        retention = 0.5 ** (elapsed_minutes / self.session_mood_half_life_minutes)
        decayed = dict(state)
        for key in self.AFFECT_KEYS:
            decayed[key] = self._move_toward_default(key, decayed.get(key, self.default_affect[key]), retention)
        decayed["session_defensiveness"] = self._move_toward_default(
            "session_defensiveness",
            decayed["session_defensiveness"],
            retention,
        )
        decayed["updated_at"] = self._format_time(now)
        self._save_session_state(session_id, decayed)
        return decayed

    def _apply_global_delta(self, state: dict, evaluation: dict, now: datetime) -> dict:
        updated = dict(state)
        for key, delta in evaluation["relationship_delta"].items():
            updated[key] = self._clamp_float(float(updated.get(key, self.default_relationship[key])) + delta)
        updated["updated_at"] = self._format_time(now)

        conn = self._connect()
        conn.execute(
            """
            UPDATE persona_global_state
            SET affinity = ?, dominance = ?, defensiveness = ?, trust = ?,
                updated_at = ?
            WHERE profile_id = ?
            """,
            (
                updated["affinity"],
                updated["dominance"],
                updated["defensiveness"],
                updated["trust"],
                updated["updated_at"],
                self.profile_id,
            ),
        )
        conn.commit()
        conn.close()
        return updated

    def _apply_session_delta(self, session_id: str, state: dict, evaluation: dict, now: datetime) -> dict:
        updated = dict(state)
        affect_delta = evaluation["affect_delta"]
        relationship_delta = evaluation["relationship_delta"]
        for key in self.AFFECT_KEYS:
            updated[key] = self._clamp_float(
                float(updated.get(key, self.default_affect[key])) + affect_delta.get(key, 0.0)
            )
        updated["session_defensiveness"] = self._clamp_float(
            float(updated.get("session_defensiveness", 0.12))
            + relationship_delta.get("defensiveness", 0.0)
        )
        updated["mood_label"] = evaluation.get("mood_label", "warm_neutral") or "warm_neutral"
        updated["residue"] = evaluation.get("residue") or updated.get("residue", "")
        updated["inner_thought"] = evaluation.get("inner_thought") or updated.get("inner_thought", "")
        updated["updated_at"] = self._format_time(now)
        self._save_session_state(session_id, updated)
        return updated

    def _save_session_state(self, session_id: str, state: dict) -> None:
        conn = self._connect()
        conn.execute(
            """
            UPDATE persona_session_state
            SET valence = ?, arousal = ?, tenderness = ?, possessiveness = ?,
                longing = ?, security = ?, protective_drive = ?, libido = ?, mood_label = ?,
                session_defensiveness = ?, residue = ?, inner_thought = ?, updated_at = ?
            WHERE profile_id = ? AND session_id = ?
            """,
            (
                state["valence"],
                state["arousal"],
                state["tenderness"],
                state["possessiveness"],
                state["longing"],
                state["security"],
                state["protective_drive"],
                state["libido"],
                state["mood_label"],
                state["session_defensiveness"],
                state.get("residue", ""),
                state.get("inner_thought", ""),
                state["updated_at"],
                self.profile_id,
                session_id,
            ),
        )
        conn.commit()
        conn.close()

    def _record_event(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        evaluation: dict,
        raw_response: str,
        error: str | None,
        exchange_hash: str | None = None,
        recalled_memory_ids: list[str] | None = None,
        tool_summary: str = "",
    ) -> None:
        now = self._format_time(self._now())
        message_hash = hashlib.sha256(user_message.encode("utf-8")).hexdigest()
        assistant_hash = hashlib.sha256(assistant_response.encode("utf-8")).hexdigest() if assistant_response else None
        user_excerpt = trim_persona_excerpt(user_message, self.event_excerpt_chars)
        assistant_excerpt = trim_persona_excerpt(assistant_response, self.event_excerpt_chars)
        conn = self._connect()
        conn.execute(
            """
            INSERT OR IGNORE INTO persona_events
            (profile_id, session_id, message_hash, exchange_hash, assistant_hash,
             event_type, perceived_intent, affect_delta, relationship_event,
             relationship_delta, personality_signal, personality_delta, mood_label,
             reply_guidance, residue, inner_thought, surface_trigger, user_excerpt, assistant_excerpt,
             recalled_memory_ids, tool_summary, confidence,
             raw_response, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.profile_id,
                session_id,
                message_hash,
                exchange_hash,
                assistant_hash,
                evaluation.get("event_type"),
                evaluation.get("perceived_intent"),
                json.dumps(evaluation.get("affect_delta", {}), ensure_ascii=False),
                1 if evaluation.get("relationship_event") else 0,
                json.dumps(evaluation.get("relationship_delta", {}), ensure_ascii=False),
                0,
                "{}",
                evaluation.get("mood_label"),
                evaluation.get("reply_guidance"),
                evaluation.get("residue"),
                evaluation.get("inner_thought"),
                evaluation.get("surface_trigger"),
                user_excerpt,
                assistant_excerpt,
                json.dumps(recalled_memory_ids or [], ensure_ascii=False),
                tool_summary,
                evaluation.get("confidence"),
                raw_response,
                error,
                now,
            ),
        )
        conn.commit()
        conn.close()

    def _list_sessions(self, limit: int) -> list[dict]:
        safe_limit = max(1, min(100, int(limit or 20)))
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT session_id, valence, arousal, security, libido, mood_label,
                   session_defensiveness, updated_at
            FROM persona_session_state
            WHERE profile_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (self.profile_id, safe_limit),
        ).fetchall()
        conn.close()
        return [
            {
                "session_id": row["session_id"],
                "valence": round(self._clamp_float(row["valence"]), 3),
                "arousal": round(self._clamp_float(row["arousal"]), 3),
                "security": round(self._clamp_float(row["security"]), 3),
                "libido": round(self._clamp_float(row["libido"]), 3),
                "mood_label": row["mood_label"],
                "session_defensiveness": round(self._clamp_float(row["session_defensiveness"]), 3),
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def _list_events(self, limit: int, session_id: str | None = None) -> list[dict]:
        safe_limit = max(1, min(100, int(limit or 20)))
        params: list[Any] = [self.profile_id]
        session_clause = ""
        if session_id:
            session_clause = "AND session_id = ?"
            params.append(session_id)
        params.append(safe_limit)

        conn = self._connect()
        rows = conn.execute(
            f"""
            SELECT id, session_id, message_hash, event_type, perceived_intent,
                   surface_trigger, inner_thought, user_excerpt, assistant_excerpt,
                   affect_delta, relationship_event, relationship_delta, mood_label,
                   reply_guidance, residue, recalled_memory_ids, tool_summary,
                   confidence, error, created_at
            FROM persona_events
            WHERE profile_id = ?
            {session_clause}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        conn.close()
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "message_hash": str(row["message_hash"])[:12],
                "event_type": row["event_type"] or "unknown",
                "perceived_intent": row["perceived_intent"] or "",
                "surface_trigger": row["surface_trigger"] or "",
                "inner_thought": row["inner_thought"] or row["residue"] or "",
                "user_excerpt": row["user_excerpt"] or "",
                "assistant_excerpt": row["assistant_excerpt"] or "",
                "affect_delta": self._json_dict(row["affect_delta"]),
                "relationship_event": bool(row["relationship_event"]),
                "relationship_delta": self._json_dict(row["relationship_delta"]),
                "mood_label": row["mood_label"] or "",
                "reply_guidance": row["reply_guidance"] or "",
                "residue": row["residue"] or "",
                "recalled_memory_ids": self._json_list(row["recalled_memory_ids"]),
                "tool_summary": row["tool_summary"] or "",
                "confidence": round(self._clamp_float(row["confidence"]), 3),
                "error": row["error"] or "",
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _snapshot(self, global_state: dict, session_state: dict, reply_guidance: str) -> dict:
        return {
            "profile_id": self.profile_id,
            "affect": {
                **{
                    key: round(
                        self._clamp_float(session_state.get(key, self.default_affect[key])),
                        3,
                    )
                    for key in self.AFFECT_KEYS
                },
                "mood_label": session_state.get("mood_label", "warm_neutral"),
                "residue": session_state.get("residue", ""),
                "inner_thought": session_state.get("inner_thought", ""),
            },
            "relationship": {
                "affinity": round(self._clamp_float(global_state.get("affinity", self.default_relationship["affinity"])), 3),
                "dominance": round(self._clamp_float(global_state.get("dominance", self.default_relationship["dominance"])), 3),
                "defensiveness": round(
                    self._clamp_float(
                        max(
                            float(global_state.get("defensiveness", self.default_relationship["defensiveness"])),
                            float(session_state.get("session_defensiveness", self.default_affect["session_defensiveness"])),
                        )
                    ),
                    3,
                ),
                "trust": round(self._clamp_float(global_state.get("trust", self.default_relationship["trust"])), 3),
            },
            "reply_guidance": reply_guidance or self.fallback_guidance,
        }

    def format_state_block(self, state: dict) -> str:
        affect = state.get("affect", {})
        relationship = state.get("relationship", {})
        return "\n".join(
            [
                "Current Persona State",
                f"关系基调：{self._long_term_state_summary(affect, relationship)}",
                f"当前情绪：{self._affect_state_summary(affect)}",
                "使用方式：这些状态只轻轻影响语气，不替你判断对方意愿，不决定或触发行动，也不要向对方解释这些隐藏状态。",
            ]
        )

    def format_recent_change_block(self, session_id: str) -> str:
        evaluation_count = max(
            1,
            (self.state_change_window_rounds + self.evaluation_interval_rounds - 1)
            // self.evaluation_interval_rounds,
        )
        totals = {key: 0.0 for key in ("security", "valence", "arousal", "libido")}
        for affect_delta in self._recent_evaluation_deltas(evaluation_count, session_id):
            for key in totals:
                try:
                    totals[key] += float(affect_delta.get(key, 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue

        key, change = max(totals.items(), key=lambda item: abs(item[1]))
        if abs(change) < self.state_change_min_abs:
            return ""
        messages = {
            ("security", True): "你的安全感提高了！",
            ("security", False): "你的安全感降低了。",
            ("valence", True): "你的心情变亮了！",
            ("valence", False): "你的心情往下沉了。",
            ("arousal", True): "你的情绪能量变高了！",
            ("arousal", False): "你的情绪能量低下来了。",
            ("libido", True): "你的性欲提高了！",
            ("libido", False): "你的性欲降低了。",
        }
        return (
            "[你的情绪]"
            + messages[(key, change > 0)]
            + "[仅你可见，不要复述，可无视]"
        )

    def _affect_state_summary(self, affect: dict) -> str:
        security = self._clamp_float(affect.get("security", 0.5))
        valence = self._clamp_float(affect.get("valence", 0.5))
        arousal = self._clamp_float(affect.get("arousal", 0.5))
        libido = self._clamp_float(affect.get("libido", 0.0))

        if security >= 0.70:
            security_text = "对这段关系有稳定的安全感"
        elif security <= 0.40:
            security_text = "对这段关系有些不安"
        else:
            security_text = "关系安全感尚可"

        if valence >= 0.65:
            valence_text = "心情偏明亮"
        elif valence <= 0.35:
            valence_text = "心情偏难受"
        else:
            valence_text = "心情平稳"

        if arousal >= 0.65:
            arousal_text = "情绪能量较高"
        elif arousal <= 0.30:
            arousal_text = "情绪能量较低"
        else:
            arousal_text = "情绪能量适中"

        if libido >= 0.70:
            libido_text = "欲望温度明显"
        elif libido >= 0.40:
            libido_text = "欲望温度微热"
        else:
            libido_text = "欲望温度安静"

        return "，".join([security_text, valence_text, arousal_text, libido_text]) + "。"

    def _long_term_state_summary(self, affect: dict, relationship: dict) -> str:
        affinity = self._clamp_float(relationship.get("affinity", 0.5))
        trust = self._clamp_float(relationship.get("trust", 0.5))
        defensiveness = self._clamp_float(relationship.get("defensiveness", 0.0))
        security = self._clamp_float(affect.get("security", 0.5))
        longing = self._clamp_float(affect.get("longing", 0.0))
        protective_drive = self._clamp_float(affect.get("protective_drive", 0.0))

        if affinity >= 0.78 and trust >= 0.72 and security >= 0.60:
            baseline = "更亲近、更安稳"
        elif affinity >= 0.60 and trust >= 0.55:
            baseline = "温和、稳定，正在靠近"
        elif defensiveness >= 0.45:
            baseline = "有一点谨慎，还在慢慢靠近"
        else:
            baseline = "平稳、安静"

        notes = []
        if longing >= 0.30:
            notes.append("想念")
        if protective_drive >= 0.50:
            notes.append("保护欲")
        if defensiveness >= 0.35:
            notes.append("谨慎")

        if notes:
            return f"{baseline}，偶尔有一点{self._join_chinese_phrases(notes)}。"
        return f"{baseline}。"

    def _join_chinese_phrases(self, phrases: list[str]) -> str:
        if not phrases:
            return ""
        if len(phrases) == 1:
            return phrases[0]
        if len(phrases) == 2:
            return "和".join(phrases)
        return "、".join(phrases[:-1]) + "和" + phrases[-1]

    def _clip_delta_map(self, data: Any, keys: list[str], max_abs: float) -> dict[str, float]:
        if not isinstance(data, dict):
            data = {}
        return {
            key: self._clamp_float(data.get(key, 0.0), -max_abs, max_abs)
            for key in keys
        }

    def _move_toward_default(self, key: str, current: float, retention: float) -> float:
        default = float(self.default_affect[key])
        return self._clamp_float(default + (float(current) - default) * retention)

    def _parse_iso(self, value: Any) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _format_time(self, value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat(timespec="seconds")

    def _json_dict(self, raw: Any) -> dict:
        try:
            parsed = json.loads(raw or "{}")
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _json_list(self, raw: Any) -> list:
        try:
            parsed = json.loads(raw or "[]")
        except (TypeError, json.JSONDecodeError):
            return []
        return parsed if isinstance(parsed, list) else []

    def _exchange_processed(self, session_id: str, exchange_hash: str) -> bool:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT 1 FROM persona_exchange_log
            WHERE profile_id = ? AND session_id = ? AND exchange_hash = ?
            LIMIT 1
            """,
            (self.profile_id, session_id, exchange_hash),
        ).fetchone()
        if row is None:
            row = conn.execute(
                """
                SELECT 1 FROM persona_events
                WHERE profile_id = ? AND session_id = ? AND exchange_hash = ?
                LIMIT 1
                """,
                (self.profile_id, session_id, exchange_hash),
            ).fetchone()
        conn.close()
        return row is not None

    def _mark_exchange_processed(
        self,
        session_id: str,
        exchange_hash: str,
        affect_delta: dict[str, Any] | None = None,
    ) -> None:
        encoded_delta = (
            json.dumps(affect_delta, ensure_ascii=False)
            if isinstance(affect_delta, dict)
            else None
        )
        conn = self._connect()
        conn.execute(
            """
            INSERT OR IGNORE INTO persona_exchange_log
            (profile_id, session_id, exchange_hash, affect_delta, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                self.profile_id,
                session_id,
                exchange_hash,
                encoded_delta,
                self._format_time(self._now()),
            ),
        )
        conn.commit()
        conn.close()

    def _recent_evaluation_deltas(self, limit: int, session_id: str) -> list[dict]:
        safe_limit = max(1, min(100, int(limit or 1)))
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT affect_delta
            FROM persona_exchange_log
            WHERE profile_id = ? AND session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (self.profile_id, session_id, safe_limit),
        ).fetchall()
        conn.close()
        return [self._json_dict(row["affect_delta"]) for row in rows]

    def _should_record_event(self, session_id: str, evaluation: dict, now: datetime) -> bool:
        if not self.event_recording_enabled:
            return False
        if self._is_salient_event(evaluation):
            return True
        if self.event_batch_size <= 1:
            return True
        last_event = self._last_event(session_id)
        if last_event and self._similar_to_last_event(evaluation, last_event, now):
            return False
        return self._processed_exchanges_since_last_event(session_id) >= self.event_batch_size

    def _is_salient_event(self, evaluation: dict) -> bool:
        if evaluation.get("relationship_event"):
            return True
        affect = evaluation.get("affect_delta", {})
        if not isinstance(affect, dict):
            return False
        values = []
        for value in affect.values():
            try:
                values.append(abs(float(value)))
            except (TypeError, ValueError):
                continue
        if not values:
            return False
        return (
            max(values) >= self.event_affect_single_threshold
            or sum(values) >= self.event_affect_total_threshold
        )

    def _similar_to_last_event(self, evaluation: dict, last_event: dict, now: datetime) -> bool:
        if self.event_similarity_threshold <= 0:
            return False
        if self.event_force_after_minutes > 0:
            created_at = self._parse_iso(last_event.get("created_at"))
            if created_at is not None:
                elapsed_minutes = max(0.0, (now - created_at).total_seconds() / 60)
                if elapsed_minutes >= self.event_force_after_minutes:
                    return False
        current = self._event_signature_text(evaluation)
        previous = self._event_signature_text(last_event)
        if not current or not previous:
            return False
        return SequenceMatcher(None, previous, current).ratio() >= self.event_similarity_threshold

    def _event_signature_text(self, event: dict) -> str:
        return re.sub(
            r"\s+",
            " ",
            " ".join(
                [
                    str(event.get("event_type") or ""),
                    str(event.get("surface_trigger") or ""),
                    str(event.get("perceived_intent") or ""),
                    str(event.get("inner_thought") or ""),
                    str(event.get("residue") or ""),
                ]
            ).strip().lower(),
        )

    def _last_event(self, session_id: str) -> dict | None:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT id, exchange_hash, event_type, perceived_intent,
                   surface_trigger, inner_thought, residue, created_at
            FROM persona_events
            WHERE profile_id = ? AND session_id = ? AND error IS NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (self.profile_id, session_id),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def _recent_event_context(self, session_id: str, limit: int = 5) -> list[dict]:
        events = self._list_events(limit, session_id)
        return [
            {
                "event_type": event.get("event_type", ""),
                "surface_trigger": event.get("surface_trigger", ""),
                "inner_thought": event.get("inner_thought") or event.get("residue", ""),
                "residue": event.get("residue", ""),
                "mood_label": event.get("mood_label", ""),
                "created_at": event.get("created_at", ""),
            }
            for event in events
        ]

    def _recent_conversation_context(
        self,
        turns: list[dict] | None,
        *,
        limit: int | None = None,
    ) -> list[dict]:
        context_limit = self.evaluation_context_turns if limit is None else max(0, int(limit))
        if context_limit <= 0 or not turns:
            return []
        selected = list(turns)[-context_limit:]
        context: list[dict] = []
        for turn in selected:
            if not isinstance(turn, dict):
                continue
            user_text = self._clean_client_status_lines(turn.get("user_text", ""))
            assistant_text = str(turn.get("assistant_text") or "").strip()
            if not user_text and not assistant_text:
                continue
            context.append(
                {
                    "created_at": str(turn.get("created_at") or ""),
                    "user_message": user_text[:600],
                    "assistant_response": assistant_text[:900],
                }
            )
        return context

    def _processed_exchanges_since_last_event(self, session_id: str) -> int:
        last_event = self._last_event(session_id)
        params: list[Any] = [self.profile_id, session_id]
        since_clause = ""
        if last_event:
            log_row = None
            if last_event.get("exchange_hash"):
                conn = self._connect()
                log_row = conn.execute(
                    """
                    SELECT id FROM persona_exchange_log
                    WHERE profile_id = ? AND session_id = ? AND exchange_hash = ?
                    LIMIT 1
                    """,
                    (self.profile_id, session_id, last_event["exchange_hash"]),
                ).fetchone()
                conn.close()
            if log_row:
                since_clause = "AND id > ?"
                params.append(log_row["id"])
            else:
                since_clause = "AND created_at > ?"
                params.append(last_event["created_at"])
        conn = self._connect()
        count = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM persona_exchange_log
            WHERE profile_id = ? AND session_id = ?
            {since_clause}
            """,
            params,
        ).fetchone()[0]
        conn.close()
        return int(count or 0)

    def _event_exists(self, session_id: str, exchange_hash: str) -> bool:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT 1 FROM persona_events
            WHERE profile_id = ? AND session_id = ? AND exchange_hash = ?
            LIMIT 1
            """,
            (self.profile_id, session_id, exchange_hash),
        ).fetchone()
        conn.close()
        return row is not None

    def _exchange_hash(self, session_id: str, user_message: str, assistant_response: str) -> str:
        text = "\n".join([self.profile_id, session_id, user_message, assistant_response])
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _has_nonzero_delta(self, raw: Any) -> bool:
        if not isinstance(raw, dict):
            return False
        for value in raw.values():
            try:
                if abs(float(value)) > 1e-9:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def _coerce_bool(self, value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _clamp_float(self, value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = lower
        return max(lower, min(upper, number))

    def _completion_options(
        self,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if timeout is not None:
            options["timeout"] = timeout
        if self.thinking_mode:
            options["extra_body"] = {"thinking": {"type": self.thinking_mode}}
        if self.json_response_format:
            options["response_format"] = {"type": "json_object"}
        return options

    def _normalize_thinking_mode(self, value: Any) -> str:
        normalized = str(value or "").strip().lower()
        aliases = {
            "enabled": "enabled",
            "enable": "enabled",
            "on": "enabled",
            "true": "enabled",
            "disabled": "disabled",
            "disable": "disabled",
            "off": "disabled",
            "false": "disabled",
            "non-thinking": "disabled",
            "non_thinking": "disabled",
        }
        return aliases.get(normalized, "")
