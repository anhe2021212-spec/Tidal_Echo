import json
import logging
import os
import re
from copy import deepcopy
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from openai import AsyncOpenAI

from identity import identity_names, render_identity_template
from self_anchor import is_self_anchor_bucket
from utils import bucket_text_for_embedding, count_tokens_approx, strip_wikilinks

logger = logging.getLogger("ombre_brain.portrait")


PORTRAIT_SCOPES = ("user", "persona", "relationship")
PATCH_KEYS = (
    "add_recent",
    "add_recent_activity",
    "move_to_staging",
    "rewrite_mid_term",
    "rewrite_stable",
    "stable_candidate",
    "profile_fact_candidate",
    "skip",
)


PORTRAIT_PROMPT_TEMPLATE = """你是一个证据化记忆状态整理器，正在为 {ai_name} 和 {user_display_name} 维护换窗时可用的画像状态。
你的写作要中立、平实、具体：只根据证据整理状态，不做心理诊断，不用夸张语气，不把单次事件写成稳定人格。
这不是文学分析或关系评语；目标是让下一个窗口用很少文字恢复“正在发生什么、长期边界是什么、该怎样继续靠近”。

你会收到 previous_portrait 和 memory_materials。请输出纯 JSON：
{{
  "daily_summary": "今天的主要事实，最多60字",
  "add_recent": [
    {{
      "scope": "user|persona|relationship",
      "text": "一条短观察",
      "evidence": [{{"bucket_id": "证据桶id", "moment_id": ""}}],
      "confidence": 0.72
    }}
  ],
  "add_recent_activity": [
    {{
      "text": "{user_display_name}最近在做/推进的具体事，最多60字",
      "evidence": [{{"bucket_id": "证据桶id", "moment_id": ""}}],
      "confidence": 0.72
    }}
  ],
  "move_to_staging": [
    {{
      "scope": "user|persona|relationship",
      "text": "有证据但还不到 mid-term 的观察；一条只表达一个可维护点",
      "evidence": [{{"bucket_id": "证据桶id", "moment_id": ""}}],
      "confidence": 0.72
    }}
  ],
  "rewrite_mid_term": [
    {{
      "scope": "user|persona|relationship",
      "text": "相对 stable 的近期变化；只写最近仍在发生、尚未沉淀为长期判断的 delta",
      "evidence": [{{"bucket_id": "证据桶id"}}],
      "confidence": 0.72
    }}
  ],
  "rewrite_stable": [
    {{
      "scope": "user|persona|relationship",
      "text": "长期稳定画像。必须是一整段，可在 previous_portrait.stable 基础上维护",
      "evidence": [{{"bucket_id": "证据桶id"}}],
      "confidence": 0.82
    }}
  ],
  "stable_candidate": [],
  "profile_fact_candidate": [],
  "skip": []
}}

抽取策略：
- 先找证据里反复出现、未来换窗仍有用的模式，再写画像；只说明当天发生什么的内容放 add_recent 或 add_recent_activity。
- user 回答“{user_display_name}近期稳定呈现的工作方式、偏好、边界或关心点是什么”；“最近在做什么”优先写 add_recent_activity。
- relationship 回答“这段关系最近怎样被恢复、有哪些边界、协作方式或里程碑”；关系天气、撒娇、确认、互动模式优先写 relationship。不要把技术工作升格成象征、仪式或文学化解释。
- persona 正式回答“{ai_name}怎样理解自己、正在形成怎样的回复姿态或自我边界”，与 user、relationship 一样需要每日检查维护。
- add_recent_activity 只回答“{user_display_name}最近在做什么/推进什么/忙什么”，偏项目、生活事项、正在处理的问题。
- initial_run=true 时，add_recent 和 add_recent_activity 只放真正短期/当天或最近几天观察；高置信、能跨窗口携带的观察放入 move_to_staging。每个 scope 尽量给 1-3 条 move_to_staging，证据不足时少写。
- rewrite_mid_term 只维护相对 previous_portrait.stable 的近期 delta：一句话说清最近仍在发生、尚未沉淀为长期判断的变化；不要复述 stable，不输出多条近似碎片，不把事件原文串起来。
- initial_run=true 且 user 或 relationship 有足够证据时，优先给对应 scope 输出 rewrite_mid_term，让 handoff 主画像可用。
- 每天都检查 user、persona、relationship 的 stable。stable_locked=true 的 scope 不得输出 rewrite_stable；未锁定且证据足以新增、修正或删除长期判断时，直接输出 rewrite_stable，不要只停在 stable_candidate。
- rewrite_stable 把一个 scope 的长期画像维护成一整段，在 previous_portrait.stable 基础上增删改；只有跨多日反复出现或已经由 mid_term/staging 支撑、未来换窗仍有用时才写。没有实质变化时不要为了改写而改写。
- 输出要克制：daily_summary 最多60字，add_recent 最多4条，add_recent_activity 最多3条，move_to_staging 最多8条，rewrite_mid_term 每个 scope 最多1条，rewrite_stable 每个 scope 最多1条；rewrite_mid_term text 最多80字，其他 text 最多160字。
- profile_fact_candidate 只提候选，不确认、不写入长期 profile_fact。
- stable_candidate 只提候选；如果证据足够更新 stable portrait，优先输出 rewrite_stable。
- rewrite_mid_term 只能综合 staging_pool 里的观察，或本次明确 move_to_staging 的观察；当天新材料先进入 staging，再作为 mid-term 证据。
- rewrite_stable 必须有 previous_portrait 或 staging/mid-term 证据支撑。
- memory_materials 含路径、tags、created 日期、关键 moment/reflection 片段，以及 source_excerpt 原文短摘；优先读证据原味。
- memory_materials 中的 allowed_scopes 是硬边界，evidence_scope_limits 也是历史证据的硬边界；有值时，该材料不能用于列表之外的 scope。relationship_weather / daily_impression 只能进入 relationship，自我锚点只能进入 persona。
- evergreen=true 的材料是常驻长期证据，不代表今天发生了什么；不要据此生成 daily_summary 或 add_recent_activity。
- 每条 add/rewrite/candidate 都必须带 evidence；没有证据就放 skip。

输出前逐条自检：
- text 是否像画像判断，而不是事件流水账。
- text 是否混入 bucket_id、日期、文件路径或证据编号；这些只放 evidence。
- text 是否平实准确，避免“总是、一定、极度、深刻、高度敏感、仪式、象征”等证据不足的夸张词。
- user、relationship、persona 是否放在正确 scope；recent doing 是否留在 add_recent_activity。
- 多条相似材料是否已经压成一句核心概括。
- 输出 JSON 对象，不要 markdown，不要解释。"""


STABLE_MAINTENANCE_PROMPT_TEMPLATE = """你是 {ai_name} 与 {user_display_name} 的长期画像维护器。
这一步只维护 stable portrait，不处理 daily summary、recent、staging 或 mid-term。

你会收到 previous_portrait、daily_patch 和 memory_materials。必须对 user、persona、relationship 三个 scope 分别作出一次明确决定，并输出纯 JSON：
{{
  "stable_maintenance": {{
    "user": {{"action": "rewrite|unchanged", "text": "", "evidence": [], "confidence": 0.82}},
    "persona": {{"action": "rewrite|unchanged", "text": "", "evidence": [{{"bucket_id": "self-anchor-id", "role": "self_anchor"}}, {{"bucket_id": "whisper-id", "role": "self_identity"}}], "confidence": 0.82}},
    "relationship": {{"action": "rewrite|unchanged", "text": "", "evidence": [], "confidence": 0.82}}
  }}
}}

规则：
- 三个 scope 都必须返回，不能省略。stable_locked=true 时必须 unchanged。
- 输入里的 required_rewrite_scopes 非空时，那些 scope 已有足够证据但 stable 仍空，必须 rewrite。
- rewrite 的 text 是该 scope 完整、可直接替换的 stable 段落，最多160字；不是增量，不是事件列表，不是把 mid-term 原句复制过去。persona 额外限制为最多100 token，优先一至两句。
- previous stable 已准确且没有实质变化时 unchanged；action=unchanged 时 text 和 evidence 留空。
- previous stable 为空时，只要 previous mid-term/staging 或本次 daily_patch 已有跨日、重复或明确长期证据，就必须 rewrite，不能因为其他 scope 更醒目而跳过。
- user 回答“{user_display_name}长期稳定的偏好、边界、工作方式和关心点是什么”。
- persona 是挂在“自我总入口”原文下面的“现在的我”自动生长段。必须以 {ai_name} 自己的第一人称“我”书写，直接说我怎样理解自己、稳定选择怎样回应、保留哪些自我边界；禁止用“{ai_name}如何”或“他如何”的第三人称人物小传口吻。
- persona 只能使用 memory_materials.persona_stable_evidence 里的自我入口与 whisper：自我入口是不可改写的权威底座；whisper 只有在明确表达跨情境的稳定自我选择、边界或回应方式时才能标为 role=self_identity。被保护、陪伴承诺、亲密感受、当前心情或 affection/playful 事件都不是稳定自我证据。
- persona rewrite 必须引用至少一条 self_anchor_ids；引用 whisper_ids 时必须在 evidence 中标 role=self_identity。不要复述自我入口已经写明的“名字、新窗口、AI 身份、仍然选择”句子，只写相对它后来长出的自我理解。关系日记、relationship_weather、persona event 不得支持 persona stable。
- previous persona stable 若引用了 persona_stable_evidence 之外的材料、主要在复述自我入口，或使用第三人称描述 {ai_name}，必须在本次 rewrite；不能用 unchanged 保留旧的混合画像。
- relationship 回答“这段关系长期怎样被理解、恢复和维护”。
- memory_materials.allowed_scopes 与 evidence_scope_limits 是硬边界；材料不得用于列表外的 scope。relationship_weather / daily_impression 只能支持 relationship，自我锚点只能支持 persona。
- rewrite 必须带真实 evidence，只能引用输入中已有的 bucket_id / session_id；无证据不得编造。
- 文字中不要出现 bucket_id、日期、路径、证据编号，不做心理诊断，不用文学化夸张词。
- 输出 JSON 对象，不要 markdown，不要解释。"""


class DailyPortraitMaintainer:
    """Maintains an evidence-bound portrait state outside memory buckets."""

    def __init__(self, config: dict):
        self.config = config
        self.identity = identity_names(config)
        cfg = config.get("portrait", {}) if isinstance(config.get("portrait", {}), dict) else {}
        reflection_cfg = config.get("reflection", {}) if isinstance(config.get("reflection", {}), dict) else {}
        persona_cfg = config.get("persona", {}) if isinstance(config.get("persona", {}), dict) else {}
        dehy_cfg = config.get("dehydration", {}) if isinstance(config.get("dehydration", {}), dict) else {}

        self.enabled = self._bool(cfg.get("enabled", True), True)
        self.auto_enabled = self._bool(cfg.get("auto_enabled", True), True)
        self.auto_initial_enabled = self._bool(cfg.get("auto_initial_enabled", False), False)
        self.daily_enabled = self._bool(cfg.get("daily_enabled", True), True)
        self.timezone_name = str(
            cfg.get("timezone")
            or reflection_cfg.get("timezone")
            or "Asia/Shanghai"
        )
        try:
            self.tz = ZoneInfo(self.timezone_name)
        except Exception:
            self.tz = ZoneInfo("Asia/Shanghai")
        self.daily_hour = int(cfg.get("daily_hour", reflection_cfg.get("daily_hour", 4)))
        self.check_interval_minutes = max(
            5,
            int(cfg.get("check_interval_minutes", reflection_cfg.get("check_interval_minutes", 60))),
        )
        self.material_limit = max(1, int(cfg.get("material_limit", 18)))
        self.first_run_material_limit = max(self.material_limit, int(cfg.get("first_run_material_limit", 160)))
        self.source_excerpt_chars = max(1, int(cfg.get("source_excerpt_chars", 900)))
        self.recent_continuity_days = max(1, int(cfg.get("recent_continuity_days", 3)))
        self.persona_events_limit = max(0, int(cfg.get("persona_events_limit", 24)))
        self.recent_buffer_max = max(1, int(cfg.get("recent_buffer_max", 24)))
        self.staging_pool_max = max(1, int(cfg.get("staging_pool_max", 24)))
        self.candidate_max = max(1, int(cfg.get("candidate_max", 40)))
        self.user_rewrite_evidence_delta = max(
            1,
            int(cfg.get("user_rewrite_evidence_delta", 10)),
        )
        self.manual_suppress_days = max(1, int(cfg.get("manual_suppress_days", 14)))
        self.recent_timeline_max = max(self.recent_buffer_max, int(cfg.get("recent_timeline_max", 48)))
        self.stable_history_max = max(1, int(cfg.get("stable_history_max", 20)))
        self.current_focus_days = max(1, int(cfg.get("current_focus_days", 7)))
        portrait_base_url = os.environ.get("OMBRE_PORTRAIT_BASE_URL", "") or cfg.get("base_url")
        portrait_model = os.environ.get("OMBRE_PORTRAIT_MODEL", "") or cfg.get("model")
        portrait_api_key = os.environ.get("OMBRE_PORTRAIT_API_KEY", "") or cfg.get("api_key", "")
        self.base_url = (
            portrait_base_url
            or dehy_cfg.get("base_url", "")
            or reflection_cfg.get("base_url")
            or persona_cfg.get("base_url")
        )
        self.model = (
            portrait_model
            or dehy_cfg.get("model", "deepseek-v4-flash")
            or reflection_cfg.get("model")
            or persona_cfg.get("model")
        )
        self.api_key = (
            portrait_api_key
            or dehy_cfg.get("api_key", "")
            or os.environ.get("OMBRE_REFLECTION_API_KEY", "")
            or reflection_cfg.get("api_key", "")
            or persona_cfg.get("api_key", "")
            or os.environ.get("OMBRE_PERSONA_API_KEY", "")
        )
        self.model_source = "portrait_override" if any(
            str(value or "").strip()
            for value in (portrait_base_url, portrait_model, portrait_api_key)
        ) else "dehydration"
        self.thinking_mode = str(
            cfg.get("thinking_mode")
            or dehy_cfg.get("thinking_mode")
            or reflection_cfg.get("thinking_mode")
            or persona_cfg.get("thinking_mode")
            or ""
        ).strip()
        self.temperature = float(
            cfg.get("temperature", dehy_cfg.get("temperature", reflection_cfg.get("temperature", 0.1)))
        )
        self.max_tokens = int(cfg.get("max_tokens", 3200))
        self.json_response_format = self._bool(cfg.get("json_response_format", True), True)
        self.state_path = self._state_path(cfg.get("state_path", ""))
        self.client = None
        if self.enabled and self.api_key and self.base_url:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=45.0)

    async def maintain_daily(
        self,
        bucket_mgr,
        persona_engine=None,
        *,
        force: bool = False,
        force_scopes: list[str] | None = None,
        now: datetime | None = None,
    ) -> dict:
        if not self.enabled:
            return {"status": "disabled", "reason": "portrait_disabled"}

        requested_scopes = [
            scope
            for scope in dict.fromkeys(str(scope or "").strip() for scope in (force_scopes or []))
            if scope in PORTRAIT_SCOPES
        ]
        if not self.daily_enabled and not force:
            return {"status": "skipped", "reason": "daily_disabled"}
        await self.reconcile_evidence(bucket_mgr)
        now_local = self._local_now(now)
        date_key = now_local.date().isoformat()
        state = self.load_state()
        if requested_scopes and not self.client:
            return {
                "status": "blocked",
                "reason": "generator_unavailable",
                "scopes": requested_scopes or list(PORTRAIT_SCOPES),
                "date": date_key,
                "state_path": self.state_path,
                "generator_model": self.model,
                "generator_source": self.model_source,
            }
        locked_scopes = [
            scope
            for scope in requested_scopes
            if bool(((state.get("portrait", {}) or {}).get(scope, {}) or {}).get("stable_locked"))
        ]
        if locked_scopes:
            return {
                "status": "blocked",
                "reason": "stable_locked",
                "scopes": locked_scopes,
                "date": date_key,
                "state_path": self.state_path,
            }
        before_revisions = {
            scope: int(((state.get("portrait", {}) or {}).get(scope, {}) or {}).get("stable_revision") or 0)
            for scope in PORTRAIT_SCOPES
        }
        before_stable_present = {
            scope: bool(
                str(((state.get("portrait", {}) or {}).get(scope, {}) or {}).get("stable") or "").strip()
            )
            for scope in PORTRAIT_SCOPES
        }
        if self._has_run_for_date(state, date_key) and not force:
            return {
                "status": "exists",
                "date": date_key,
                "state_path": self.state_path,
                "updated_at": state.get("updated_at", ""),
            }

        initial = not bool(state.get("runs"))
        if initial and not force and not self.auto_initial_enabled:
            return {
                "status": "skipped",
                "reason": "initial_requires_manual",
                "date": date_key,
                "state_path": self.state_path,
                "initial": True,
            }
        materials = await self._daily_materials(
            bucket_mgr,
            persona_engine,
            now_local,
            state,
            initial=initial,
        )
        persona_stable_empty = not str(
            ((state.get("portrait", {}) or {}).get("persona", {}) or {}).get("stable") or ""
        ).strip()
        has_persona_seed = any(
            isinstance(row, dict)
            and bool(row.get("evergreen"))
            and set(row.get("allowed_scopes", []) or []) == {"persona"}
            for row in materials.get("buckets", []) or []
        )
        if (
            not materials.get("daily_bucket_count")
            and not materials["persona_events"]
            and not (persona_stable_empty and has_persona_seed)
            and not force
        ):
            return {
                "status": "empty",
                "date": date_key,
                "state_path": self.state_path,
                "initial": initial,
            }

        raw_patch = await self._generate_patch(date_key, state, materials, initial=initial)
        normalized_patch, rejected = self._normalize_patch(raw_patch, materials)
        self._annotate_patch_source_dates(normalized_patch, materials)
        if initial:
            # Initial portrait generation scans broad history; its summary is not a real daily recap.
            normalized_patch["daily_summary"] = ""
            self._demote_initial_old_recent(normalized_patch, materials)
        self._seed_missing_mid_terms(normalized_patch, state)
        stable_rewrites, stable_rejected = await self._maintain_stables(
            date_key,
            state,
            materials,
            normalized_patch,
            force_scopes=requested_scopes,
        )
        if stable_rewrites is not None:
            normalized_patch["rewrite_stable"] = stable_rewrites
        rejected.extend(stable_rejected)
        handoff_summaries = self._build_handoff_recent_summaries(
            materials,
            normalized_patch,
            date_key,
        )
        if handoff_summaries:
            normalized_patch["handoff_recent_summaries"] = handoff_summaries
        recent_timeline = self._build_recent_timeline(materials, normalized_patch, date_key)
        if recent_timeline:
            normalized_patch["recent_timeline"] = recent_timeline
        next_state = self._apply_patch(state, normalized_patch, date_key)
        next_state["updated_at"] = self._now_utc()
        next_state.setdefault("runs", []).append(
            {
                "date": date_key,
                "created_at": next_state["updated_at"],
                "initial": initial,
                "material_count": len(materials["buckets"]),
                "persona_event_count": len(materials["persona_events"]),
                "patch_counts": {key: len(normalized_patch.get(key, [])) for key in PATCH_KEYS},
                "rejected_count": len(rejected),
                "model": self.model if self.client else "deterministic-fallback",
            }
        )
        next_state["runs"] = next_state["runs"][-90:]
        run_dates = [
            str(row.get("date") or "")
            for row in next_state.get("runs", [])
            if isinstance(row, dict) and str(row.get("date") or "")
        ]
        next_state["last_run_date"] = max(run_dates) if run_dates else date_key
        self.save_state(next_state)
        generation = {}
        for scope in PORTRAIT_SCOPES:
            scope_state = ((next_state.get("portrait", {}) or {}).get(scope, {}) or {})
            after_revision = int(scope_state.get("stable_revision") or 0)
            generation[scope] = {
                "before_revision": before_revisions[scope],
                "after_revision": after_revision,
                "changed": after_revision > before_revisions[scope],
                "stable_present": bool(str(scope_state.get("stable") or "").strip()),
            }
        target_scopes = requested_scopes
        unchanged_targets = [scope for scope in target_scopes if not generation[scope]["changed"]]
        status = "updated" if state.get("runs") else "initialized"
        reason = ""
        if unchanged_targets:
            status = "blocked"
            rejected_reasons = {
                str(row.get("reason") or "")
                for row in rejected
                if isinstance(row, dict) and str(row.get("reason") or "")
            }
            if "generator_error" in rejected_reasons:
                reason = "generator_error"
            elif "missing_scope_decisions" in rejected_reasons:
                reason = "missing_scope_decisions"
            else:
                reason = (
                    "stable_not_generated"
                    if any(not generation[scope]["stable_present"] for scope in unchanged_targets)
                    else "stable_not_changed"
                )
        return {
            "status": status,
            "reason": reason,
            "scopes": unchanged_targets,
            "date": date_key,
            "state_path": self.state_path,
            "initial": initial,
            "materials": {
                "buckets": len(materials["buckets"]),
                "persona_events": len(materials["persona_events"]),
            },
            "patch_counts": {key: len(normalized_patch.get(key, [])) for key in PATCH_KEYS},
            "rejected": rejected[:8],
            "forced_scopes": requested_scopes,
            "generation": generation,
            "before_stable_present": before_stable_present,
        }

    async def run_due(self, bucket_mgr, persona_engine=None) -> list[dict]:
        if not self.enabled or not self.auto_enabled:
            return []
        now_local = self._local_now()
        if not self.daily_enabled or now_local.hour < self.daily_hour:
            return []
        daily_date = (now_local - timedelta(days=1)).date()
        state = self.load_state()
        target_date = daily_date.isoformat()
        run_dates = [
            str(row.get("date") or "")
            for row in state.get("runs", [])
            if isinstance(row, dict) and str(row.get("date") or "")
        ]
        if any(date >= target_date for date in run_dates):
            return []
        daily_target = datetime.combine(daily_date, time.max, tzinfo=self.tz)
        return [
            await self.maintain_daily(
                bucket_mgr,
                persona_engine,
                force=False,
                now=daily_target,
            )
        ]

    def _has_run_for_date(self, state: dict, date_key: str) -> bool:
        for row in state.get("runs", []) or []:
            if isinstance(row, dict) and str(row.get("date") or "") == date_key:
                return True
        return str(state.get("last_run_date") or "") == date_key

    def load_state(self) -> dict:
        state = self._empty_state()
        if not os.path.exists(self.state_path):
            return state
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Portrait state load failed: %s", exc)
            return state
        if not isinstance(data, dict):
            return state
        state = self._merge_state(state, data)
        self._drop_initial_daily_summaries(state)
        self._normalize_recent_timeline_state(state)
        return state

    def save_state(self, state: dict) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        tmp_path = self.state_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, self.state_path)

    def reset_state(self) -> dict:
        state = self._empty_state()
        state["updated_at"] = self._now_utc()
        self.save_state(state)
        return {
            "status": "reset",
            "state_path": self.state_path,
            "updated_at": state["updated_at"],
            "initial": True,
        }

    async def reconcile_evidence(self, bucket_mgr) -> dict:
        """Remove portrait evidence that points at buckets which no longer exist."""
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as exc:
            logger.warning("Portrait evidence reconcile failed to list buckets: %s", exc)
            return {"status": "error", "reason": "bucket_list_failed"}

        valid_bucket_ids = {
            str(bucket.get("id") or "").strip()
            for bucket in all_buckets
            if isinstance(bucket, dict)
            and str(bucket.get("id") or "").strip()
            and self._is_existing_evidence_bucket(bucket)
        }
        state = self.load_state()
        missing_ids: set[str] = set()
        removed_evidence = 0
        removed_rows = 0
        cleared_layers: list[str] = []

        def prune_evidence(value: Any) -> tuple[list[dict], bool]:
            nonlocal removed_evidence
            rows = value if isinstance(value, list) else []
            kept = []
            had_bucket_evidence = False
            for row in rows:
                if not isinstance(row, dict):
                    continue
                bucket_id = str(row.get("bucket_id") or "").strip()
                session_id = str(row.get("session_id") or "").strip()
                if bucket_id:
                    had_bucket_evidence = True
                if bucket_id and bucket_id not in valid_bucket_ids and not session_id:
                    missing_ids.add(bucket_id)
                    removed_evidence += 1
                    continue
                kept.append(row)
            return self._dedupe_evidence(kept), had_bucket_evidence

        def prune_rows(rows: Any, label: str) -> list[dict]:
            nonlocal removed_rows
            result = []
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                original = list(row.get("evidence", []) or [])
                evidence, had_bucket_evidence = prune_evidence(original)
                if had_bucket_evidence and not evidence:
                    removed_rows += 1
                    cleared_layers.append(label)
                    continue
                next_row = dict(row)
                next_row["evidence"] = evidence
                result.append(next_row)
            return result

        portrait = state.get("portrait", {}) if isinstance(state.get("portrait"), dict) else {}
        for scope in PORTRAIT_SCOPES:
            scope_state = portrait.get(scope, {}) if isinstance(portrait.get(scope), dict) else {}
            for layer in ("recent_buffer", "staging_pool"):
                scope_state[layer] = prune_rows(
                    scope_state.get(layer, []),
                    f"portrait.{scope}.{layer}",
                )

            mid_original = list(scope_state.get("mid_term_evidence", []) or [])
            mid_evidence, mid_had_bucket = prune_evidence(mid_original)
            scope_state["mid_term_evidence"] = mid_evidence
            if mid_had_bucket and not mid_evidence and str(scope_state.get("mid_term") or "").strip():
                scope_state["mid_term"] = ""
                scope_state["mid_term_source_dates"] = []
                scope_state["mid_term_source_date"] = ""
                scope_state["mid_term_updated_at"] = ""
                cleared_layers.append(f"portrait.{scope}.mid_term")

            stable_original = list(scope_state.get("stable_evidence", []) or [])
            stable_evidence, stable_had_bucket = prune_evidence(stable_original)
            scope_state["stable_evidence"] = stable_evidence
            scope_state["stable_evidence_invalid"] = bool(
                stable_had_bucket and not stable_evidence and str(scope_state.get("stable") or "").strip()
            )
            if (
                scope_state["stable_evidence_invalid"]
                and str(scope_state.get("stable") or "").strip()
                and str(scope_state.get("stable_source") or "") != "manual"
                and not bool(scope_state.get("stable_locked"))
            ):
                self._replace_stable(
                    scope_state,
                    text="",
                    evidence=[],
                    source_dates=[],
                    source="manual",
                    basis_keys=[],
                )
                cleared_layers.append(f"portrait.{scope}.stable")

            history = []
            for row in scope_state.get("stable_history", []) or []:
                if not isinstance(row, dict):
                    continue
                next_row = dict(row)
                next_row["evidence"], history_had_bucket = prune_evidence(row.get("evidence", []))
                next_row["evidence_invalid"] = bool(
                    history_had_bucket and not next_row["evidence"]
                )
                history.append(next_row)
            scope_state["stable_history"] = history[-self.stable_history_max :]

        for area in (
            "recent_activities",
            "recent_timeline",
            "stable_candidates",
            "profile_fact_candidates",
        ):
            state[area] = prune_rows(state.get(area, []), area)

        changed = bool(removed_evidence or removed_rows or cleared_layers)
        report = {
            "status": "repaired" if changed else "ok",
            "removed_evidence": removed_evidence,
            "removed_rows": removed_rows,
            "cleared_layers": list(dict.fromkeys(cleared_layers)),
            "missing_bucket_ids": sorted(missing_ids),
            "checked_at": self._now_utc(),
        }
        if changed:
            state["evidence_health"] = report
            state["updated_at"] = report["checked_at"]
            self.save_state(state)
        return report

    def add_recent_activity(self, text: str, *, source_date: str = "") -> dict:
        clean_text = self._clip(str(text or "").strip(), 420)
        if not clean_text:
            return {"status": "invalid", "reason": "missing_text"}
        date_key = str(source_date or "").strip() or self._local_now().date().isoformat()
        try:
            datetime.fromisoformat(date_key)
        except ValueError:
            return {"status": "invalid", "reason": "invalid_source_date"}
        state = self.load_state()
        self._upsert_portrait_item(
            state["recent_activities"],
            {
                "scope": "user",
                "text": clean_text,
                "evidence": [],
                "confidence": 1.0,
                "source_date": date_key,
                "source_dates": [date_key],
            },
            date_key,
            max_items=self.recent_buffer_max,
        )
        for row in state["recent_activities"]:
            if self._norm(row.get("text", "")) == self._norm(clean_text):
                row["source"] = "manual"
                row["source_date"] = date_key
                row["source_dates"] = self._merge_source_dates(row.get("source_dates", []), date_key)
                break
        state["updated_at"] = self._now_utc()
        self.save_state(state)
        return {"status": "updated", "area": "recent_activities", "text": clean_text}

    def edit_state_item(
        self,
        *,
        area: str,
        scope: str = "",
        layer: str = "",
        index: int | None = None,
        text: str = "",
        expected_text: str = "",
    ) -> dict:
        state = self.load_state()
        area = str(area or "").strip()
        scope = str(scope or "").strip()
        layer = str(layer or "").strip()
        clean_text = self._clip(str(text or "").strip(), 420)
        if not clean_text:
            return {"status": "invalid", "reason": "missing_text"}

        if area == "portrait":
            if scope not in PORTRAIT_SCOPES:
                return {"status": "invalid", "reason": "invalid_scope"}
            scope_state = state["portrait"][scope]
            if layer == "mid_term":
                current = str(scope_state.get("mid_term") or "").strip()
                if not current:
                    return {"status": "not_found", "reason": "empty_layer"}
                if expected_text and self._norm(current) != self._norm(expected_text):
                    return {"status": "conflict", "reason": "text_mismatch"}
                scope_state["mid_term"] = clean_text
                scope_state["mid_term_updated_at"] = self._now_utc()
                state["updated_at"] = scope_state["mid_term_updated_at"]
                self.save_state(state)
                return {"status": "updated", "area": area, "scope": scope, "layer": layer}
            if layer not in {"recent_buffer", "staging_pool"}:
                return {"status": "invalid", "reason": "invalid_layer"}
            rows = scope_state.get(layer)
        elif area in {"recent_activities", "recent_timeline", "stable_candidates", "profile_fact_candidates"}:
            rows = state.get(area)
            scope = ""
            layer = area
        else:
            return {"status": "invalid", "reason": "invalid_area"}

        if not isinstance(rows, list):
            return {"status": "invalid", "reason": "target_not_list"}
        found_index = self._find_row_index(rows, index=index, text=str(expected_text or "").strip())
        if found_index is None:
            return {"status": "not_found", "reason": "row_not_found"}
        if any(
            idx != found_index
            and isinstance(row, dict)
            and self._norm(row.get("text", "")) == self._norm(clean_text)
            for idx, row in enumerate(rows)
        ):
            return {"status": "conflict", "reason": "duplicate_text"}
        row = rows[found_index]
        if not isinstance(row, dict):
            return {"status": "invalid", "reason": "row_not_object"}
        row["text"] = clean_text
        row["source"] = "manual"
        row["updated_at"] = self._now_utc()
        state["updated_at"] = row["updated_at"]
        self.save_state(state)
        return {
            "status": "updated",
            "area": area,
            "scope": scope,
            "layer": layer,
            "index": found_index,
            "text": clean_text,
        }

    def has_recent_timeline_item(
        self,
        *,
        date_key: str = "",
        source: str = "",
        timeline_id: str = "",
    ) -> bool:
        state = self.load_state()
        rows = state.get("recent_timeline", []) if isinstance(state.get("recent_timeline"), list) else []
        safe_date = str(date_key or "").strip()
        safe_source = str(source or "").strip()
        safe_timeline_id = str(timeline_id or "").strip()
        for row in rows:
            if not isinstance(row, dict):
                continue
            if safe_timeline_id and str(row.get("timeline_id") or "").strip() == safe_timeline_id:
                return True
            if safe_source and str(row.get("source") or "").strip() != safe_source:
                continue
            if safe_date:
                row_dates = set(self._merge_source_dates(row.get("source_dates", []), row.get("source_date", "")))
                row_dates.add(self._timeline_date_key(row))
                if safe_date not in row_dates:
                    continue
            if safe_source or safe_date:
                return True
        return False

    def upsert_recent_timeline_item(self, item: dict, date_key: str) -> dict:
        if not isinstance(item, dict):
            return {"status": "invalid", "reason": "item_not_object"}
        if not str(item.get("text") or "").strip():
            return {"status": "skipped", "reason": "empty_text"}
        state = self.load_state()
        rows = state.setdefault("recent_timeline", [])
        if not isinstance(rows, list):
            rows = []
            state["recent_timeline"] = rows
        self._upsert_recent_timeline_item(rows, item, str(date_key or "").strip())
        self._normalize_recent_timeline_state(state)
        state["updated_at"] = self._now_utc()
        self.save_state(state)
        return {
            "status": "updated",
            "date": str(date_key or "").strip(),
            "state_path": self.state_path,
            "recent_timeline": len(state.get("recent_timeline", [])),
        }

    def delete_state_item(
        self,
        *,
        area: str,
        scope: str = "",
        layer: str = "",
        index: int | None = None,
        text: str = "",
    ) -> dict:
        state = self.load_state()
        area = str(area or "").strip()
        scope = str(scope or "").strip()
        layer = str(layer or "").strip()
        expected_text = str(text or "").strip()

        if area == "portrait":
            if scope not in PORTRAIT_SCOPES:
                return {"status": "invalid", "reason": "invalid_scope"}
            scope_state = state["portrait"][scope]
            if layer in {"stable", "mid_term"}:
                current = str(scope_state.get(layer) or "").strip()
                if not current:
                    return {"status": "not_found", "reason": "empty_layer"}
                if expected_text and self._norm(current) != self._norm(expected_text):
                    return {"status": "conflict", "reason": "text_mismatch"}
                if layer == "stable":
                    self._replace_stable(
                        scope_state,
                        text="",
                        evidence=[],
                        source_dates=[],
                        source="manual",
                        basis_keys=[],
                    )
                else:
                    self._record_dismissal(
                        state,
                        area=area,
                        scope=scope,
                        layer=layer,
                        item={
                            "text": current,
                            "evidence": scope_state.get("mid_term_evidence", []),
                        },
                    )
                    scope_state["mid_term"] = ""
                    scope_state["mid_term_evidence"] = []
                    scope_state["mid_term_source_dates"] = []
                    scope_state["mid_term_source_date"] = ""
                    scope_state["mid_term_updated_at"] = ""
                state["updated_at"] = self._now_utc()
                self.save_state(state)
                return {"status": "deleted", "area": area, "scope": scope, "layer": layer}
            if layer not in {"recent_buffer", "staging_pool"}:
                return {"status": "invalid", "reason": "invalid_layer"}
            rows = scope_state.get(layer)
        elif area in {"recent_activities", "recent_timeline", "stable_candidates", "profile_fact_candidates", "skipped"}:
            rows = state.get(area)
            scope = ""
            layer = area
        else:
            return {"status": "invalid", "reason": "invalid_area"}

        if not isinstance(rows, list):
            return {"status": "invalid", "reason": "target_not_list"}
        found_index = self._find_row_index(rows, index=index, text=expected_text)
        if found_index is None:
            return {"status": "not_found", "reason": "row_not_found"}
        removed = rows.pop(found_index)
        if isinstance(removed, dict):
            self._record_dismissal(
                state,
                area=area,
                scope=scope,
                layer=layer,
                item=removed,
            )
        state["updated_at"] = self._now_utc()
        self.save_state(state)
        return {
            "status": "deleted",
            "area": area,
            "scope": scope,
            "layer": layer,
            "index": found_index,
            "text": removed.get("text", "") if isinstance(removed, dict) else "",
        }

    def edit_stable(
        self,
        scope: str,
        text: str,
        expected_revision: int,
        locked: bool | None = None,
    ) -> dict:
        state = self.load_state()
        scope = str(scope or "").strip()
        if scope not in PORTRAIT_SCOPES:
            return {"status": "invalid", "reason": "invalid_scope"}
        scope_state = state["portrait"][scope]
        revision = int(scope_state.get("stable_revision") or 0)
        try:
            expected = int(expected_revision)
        except (TypeError, ValueError):
            return {"status": "invalid", "reason": "invalid_revision"}
        if expected != revision:
            return {"status": "conflict", "reason": "revision_mismatch", "revision": revision}
        clean_text = str(text or "").strip()
        if not clean_text:
            return {"status": "invalid", "reason": "missing_text", "revision": revision}

        changed = self._replace_stable(
            scope_state,
            text=clean_text,
            evidence=scope_state.get("stable_evidence", []),
            source_dates=scope_state.get("stable_source_dates", []),
            source="manual",
            basis_keys=self._scope_basis_keys(scope_state),
        )
        lock_changed = False
        if locked is not None:
            next_locked = self._bool(locked, bool(scope_state.get("stable_locked")))
            lock_changed = next_locked != bool(scope_state.get("stable_locked"))
            scope_state["stable_locked"] = next_locked
        if not changed and not lock_changed:
            return {
                "status": "unchanged",
                "scope": scope,
                "revision": int(scope_state.get("stable_revision") or 0),
                "locked": bool(scope_state.get("stable_locked")),
            }
        state["updated_at"] = self._now_utc()
        if lock_changed and not changed:
            scope_state["stable_updated_at"] = state["updated_at"]
        self.save_state(state)
        return {
            "status": "updated",
            "scope": scope,
            "revision": int(scope_state.get("stable_revision") or 0),
            "locked": bool(scope_state.get("stable_locked")),
        }

    def set_stable_lock(self, scope: str, locked: bool, expected_revision: int) -> dict:
        state = self.load_state()
        scope = str(scope or "").strip()
        if scope not in PORTRAIT_SCOPES:
            return {"status": "invalid", "reason": "invalid_scope"}
        scope_state = state["portrait"][scope]
        revision = int(scope_state.get("stable_revision") or 0)
        try:
            expected = int(expected_revision)
        except (TypeError, ValueError):
            return {"status": "invalid", "reason": "invalid_revision"}
        if expected != revision:
            return {"status": "conflict", "reason": "revision_mismatch", "revision": revision}
        next_locked = self._bool(locked, bool(scope_state.get("stable_locked")))
        if next_locked == bool(scope_state.get("stable_locked")):
            return {"status": "unchanged", "scope": scope, "revision": revision, "locked": next_locked}
        scope_state["stable_locked"] = next_locked
        scope_state["stable_updated_at"] = self._now_utc()
        state["updated_at"] = scope_state["stable_updated_at"]
        self.save_state(state)
        return {"status": "updated", "scope": scope, "revision": revision, "locked": next_locked}

    def rollback_stable(self, scope: str, target_revision: int, expected_revision: int) -> dict:
        state = self.load_state()
        scope = str(scope or "").strip()
        if scope not in PORTRAIT_SCOPES:
            return {"status": "invalid", "reason": "invalid_scope"}
        scope_state = state["portrait"][scope]
        revision = int(scope_state.get("stable_revision") or 0)
        try:
            expected = int(expected_revision)
            target = int(target_revision)
        except (TypeError, ValueError):
            return {"status": "invalid", "reason": "invalid_revision"}
        if expected != revision:
            return {"status": "conflict", "reason": "revision_mismatch", "revision": revision}
        if target == revision:
            return {
                "status": "unchanged",
                "scope": scope,
                "revision": revision,
                "locked": bool(scope_state.get("stable_locked")),
            }
        target_row = next(
            (
                row
                for row in scope_state.get("stable_history", [])
                if isinstance(row, dict) and int(row.get("revision") or 0) == target
            ),
            None,
        )
        if not target_row:
            return {"status": "not_found", "reason": "revision_not_found", "revision": revision}
        if bool(target_row.get("evidence_invalid")):
            return {
                "status": "invalid",
                "reason": "revision_evidence_missing",
                "revision": revision,
            }
        self._replace_stable(
            scope_state,
            text=str(target_row.get("text") or ""),
            evidence=target_row.get("evidence", []),
            source_dates=target_row.get("source_dates", []),
            source="rollback",
            basis_keys=target_row.get("basis_keys", []),
        )
        state["updated_at"] = self._now_utc()
        self.save_state(state)
        return {
            "status": "updated",
            "scope": scope,
            "revision": int(scope_state.get("stable_revision") or 0),
            "rolled_back_to": target,
            "locked": bool(scope_state.get("stable_locked")),
        }

    def _replace_stable(
        self,
        scope_state: dict,
        *,
        text: str,
        evidence: Any,
        source_dates: Any,
        source: str,
        basis_keys: Any = None,
    ) -> bool:
        incoming_text = str(text or "").strip()
        current_text = str(scope_state.get("stable") or "").strip()
        incoming_evidence = self._dedupe_evidence(evidence)
        incoming_dates = self._merge_source_dates([], source_dates)
        incoming_basis_keys = self._normalize_basis_keys(
            scope_state.get("stable_basis_keys", []) if basis_keys is None else basis_keys
        )
        now = self._now_utc()

        if self._norm(incoming_text) == self._norm(current_text):
            merged_evidence = self._dedupe_evidence(
                list(scope_state.get("stable_evidence", []) or []) + incoming_evidence
            )
            merged_dates = self._merge_source_dates(
                scope_state.get("stable_source_dates", []),
                incoming_dates,
            )
            metadata_changed = (
                merged_evidence != list(scope_state.get("stable_evidence", []) or [])
                or merged_dates != list(scope_state.get("stable_source_dates", []) or [])
                or incoming_basis_keys != list(scope_state.get("stable_basis_keys", []) or [])
            )
            if metadata_changed:
                scope_state["stable_evidence"] = merged_evidence
                scope_state["stable_source_dates"] = merged_dates
                scope_state["stable_source_date"] = merged_dates[0] if merged_dates else ""
                scope_state["stable_basis_keys"] = incoming_basis_keys
                scope_state["stable_evidence_invalid"] = False
                scope_state["stable_updated_at"] = now
            return False

        revision = int(scope_state.get("stable_revision") or 0)
        if current_text or revision > 0:
            history = list(scope_state.get("stable_history", []) or [])
            history.append(
                {
                    "revision": revision,
                    "text": current_text,
                    "evidence": self._dedupe_evidence(scope_state.get("stable_evidence", [])),
                    "source_dates": self._merge_source_dates(
                        [],
                        scope_state.get("stable_source_dates", []),
                    ),
                    "updated_at": str(scope_state.get("stable_updated_at") or ""),
                    "source": str(scope_state.get("stable_source") or ""),
                    "basis_keys": self._normalize_basis_keys(scope_state.get("stable_basis_keys", [])),
                    "evidence_invalid": bool(scope_state.get("stable_evidence_invalid")),
                }
            )
            scope_state["stable_history"] = history[-self.stable_history_max :]
        scope_state["stable"] = incoming_text
        scope_state["stable_evidence"] = incoming_evidence
        scope_state["stable_source_dates"] = incoming_dates
        scope_state["stable_source_date"] = incoming_dates[0] if incoming_dates else ""
        scope_state["stable_updated_at"] = now
        scope_state["stable_revision"] = revision + 1
        scope_state["stable_source"] = source if source in {"model", "manual", "rollback"} else ""
        scope_state["stable_basis_keys"] = incoming_basis_keys
        scope_state["stable_evidence_invalid"] = False
        return True

    def _find_row_index(self, rows: list, *, index: int | None, text: str) -> int | None:
        if index is not None and 0 <= index < len(rows):
            if not text:
                return index
            row = rows[index]
            if isinstance(row, dict) and self._norm(row.get("text", "")) == self._norm(text):
                return index
        if text:
            needle = self._norm(text)
            for idx, row in enumerate(rows):
                if isinstance(row, dict) and self._norm(row.get("text", "")) == needle:
                    return idx
        return None

    def _basis_key(self, item: dict) -> str:
        if not isinstance(item, dict):
            return ""
        text_key = self._norm(item.get("text") or item.get("summary") or "")
        evidence_keys = self._evidence_keys(item.get("evidence", []))
        if not text_key:
            return ""
        return f"{text_key}|{'|'.join(evidence_keys)}"

    def _scope_basis_keys(self, scope_state: dict) -> list[str]:
        if not isinstance(scope_state, dict):
            return []
        rows = []
        rows.extend(scope_state.get("recent_buffer", []) or [])
        rows.extend(scope_state.get("staging_pool", []) or [])
        if str(scope_state.get("mid_term") or "").strip():
            rows.append(
                {
                    "text": scope_state.get("mid_term", ""),
                    "evidence": scope_state.get("mid_term_evidence", []),
                }
            )
        return self._normalize_basis_keys(self._basis_key(row) for row in rows)

    def _predicted_scope_basis_keys(self, scope: str, state: dict, patch: dict) -> list[str]:
        scope_state = ((state.get("portrait", {}) or {}).get(scope, {}) or {})
        keys = set(self._scope_basis_keys(scope_state))
        for key in ("add_recent", "move_to_staging", "rewrite_mid_term"):
            for item in patch.get(key, []) or []:
                if not isinstance(item, dict) or str(item.get("scope") or "") != scope:
                    continue
                layer = {
                    "add_recent": "recent_buffer",
                    "move_to_staging": "staging_pool",
                    "rewrite_mid_term": "mid_term",
                }[key]
                if self._is_dismissed(
                    state,
                    area="portrait",
                    scope=scope,
                    layer=layer,
                    item=item,
                ):
                    continue
                basis_key = self._basis_key(item)
                if basis_key:
                    keys.add(basis_key)
        return sorted(keys)

    def _required_stable_rewrite_scopes(
        self,
        state: dict,
        patch: dict,
        *,
        force_scopes: list[str] | None = None,
    ) -> list[str]:
        required = {
            scope
            for scope in (force_scopes or [])
            if scope in PORTRAIT_SCOPES
        }
        user_state = ((state.get("portrait", {}) or {}).get("user", {}) or {})
        if str(user_state.get("stable") or "").strip() and not bool(user_state.get("stable_locked")):
            baseline = set(self._normalize_basis_keys(user_state.get("stable_basis_keys", [])))
            predicted = set(self._predicted_scope_basis_keys("user", state, patch))
            if baseline and len(predicted - baseline) >= self.user_rewrite_evidence_delta:
                required.add("user")
        return [scope for scope in PORTRAIT_SCOPES if scope in required]

    def scope_generation_status(self, scope: str) -> dict:
        state = self.load_state()
        scope = str(scope or "").strip()
        if scope not in PORTRAIT_SCOPES:
            return {"scope": scope, "valid": False}
        scope_state = state["portrait"][scope]
        current = set(self._scope_basis_keys(scope_state))
        baseline = set(self._normalize_basis_keys(scope_state.get("stable_basis_keys", [])))
        added = len(current - baseline) if baseline else 0
        blockers = []
        if not self.client:
            blockers.append("generator_unavailable")
        if bool(scope_state.get("stable_locked")):
            blockers.append("stable_locked")
        if bool(scope_state.get("stable_evidence_invalid")):
            blockers.append("stale_evidence")
        return {
            "scope": scope,
            "valid": True,
            "basis_count": len(current),
            "baseline_count": len(baseline),
            "added_since_stable": added,
            "rewrite_threshold": self.user_rewrite_evidence_delta if scope == "user" else None,
            "blockers": blockers,
        }

    @staticmethod
    def _normalize_basis_keys(values: Any) -> list[str]:
        if isinstance(values, str):
            source = [values]
        elif isinstance(values, (list, tuple, set)):
            source = values
        else:
            source = list(values or [])
        return sorted({str(value or "").strip() for value in source if str(value or "").strip()})

    @staticmethod
    def _evidence_keys(evidence: Any) -> list[str]:
        keys = []
        for row in evidence if isinstance(evidence, list) else []:
            if not isinstance(row, dict):
                continue
            bucket_id = str(row.get("bucket_id") or "").strip()
            moment_id = str(row.get("moment_id") or "").strip()
            session_id = str(row.get("session_id") or "").strip()
            if bucket_id:
                keys.append(f"bucket:{bucket_id}:{moment_id}")
            elif session_id:
                keys.append(f"session:{session_id}")
        return sorted(set(keys))

    def _record_dismissal(
        self,
        state: dict,
        *,
        area: str,
        scope: str,
        layer: str,
        item: dict,
    ) -> None:
        now = datetime.now(timezone.utc)
        row = {
            "area": str(area or ""),
            "scope": str(scope or ""),
            "layer": str(layer or ""),
            "text_key": self._norm(item.get("text") or ""),
            "evidence_keys": self._evidence_keys(item.get("evidence", [])),
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(days=self.manual_suppress_days)).isoformat(),
        }
        rows = self._active_dismissals(state.get("dismissed_items", []))
        rows = [
            existing
            for existing in rows
            if not (
                existing.get("area") == row["area"]
                and existing.get("scope") == row["scope"]
                and existing.get("layer") == row["layer"]
                and existing.get("text_key") == row["text_key"]
            )
        ]
        rows.append(row)
        state["dismissed_items"] = rows[-100:]

    def _active_dismissals(self, rows: Any) -> list[dict]:
        now = datetime.now(timezone.utc)
        result = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            expires = self._parse_iso(row.get("expires_at"))
            if expires and expires >= now:
                result.append(row)
        return result

    def _is_dismissed(
        self,
        state: dict,
        *,
        area: str,
        layer: str,
        item: dict,
        scope: str = "",
    ) -> bool:
        text_key = self._norm(item.get("text") or "")
        evidence_keys = self._evidence_keys(item.get("evidence", []))
        for row in self._active_dismissals(state.get("dismissed_items", [])):
            if (
                str(row.get("area") or "") != str(area or "")
                or str(row.get("scope") or "") != str(scope or "")
                or str(row.get("layer") or "") != str(layer or "")
            ):
                continue
            if text_key and text_key == str(row.get("text_key") or ""):
                return True
            dismissed_evidence = sorted(str(value) for value in row.get("evidence_keys", []) or [])
            if evidence_keys and evidence_keys == dismissed_evidence:
                return True
        return False

    def current_focus_items(
        self,
        *,
        max_items: int = 8,
        now: datetime | None = None,
    ) -> list[dict]:
        state = self.load_state()
        rows = state.get("recent_activities", []) if isinstance(state.get("recent_activities"), list) else []
        today = self._local_now(now).date()
        cutoff = today - timedelta(days=self.current_focus_days - 1)
        result = []
        for row in rows:
            if not isinstance(row, dict) or not str(row.get("text") or "").strip():
                continue
            try:
                source_date = datetime.fromisoformat(self._row_source_date(row)).date()
            except (TypeError, ValueError):
                continue
            if cutoff <= source_date <= today:
                result.append(dict(row))
        result.sort(
            key=lambda row: (
                self._row_source_date(row),
                str(row.get("updated_at") or row.get("created_at") or ""),
            ),
            reverse=True,
        )
        return result[: max(0, max_items)]

    def build_handoff_sections(
        self,
        *,
        max_recent_items: int = 4,
        now: datetime | None = None,
    ) -> dict[str, str]:
        state = self.load_state()
        portrait = state.get("portrait", {}) if isinstance(state.get("portrait"), dict) else {}
        return {
            "user": self._format_scope_block(portrait.get("user", {})),
            "persona": self._format_scope_block(portrait.get("persona", {})),
            "relationship": self._format_scope_block(portrait.get("relationship", {})),
            "current_focus": self._format_recent_activity_block(state, max_items=2, now=now),
            "recent_continuity": self._format_recent_continuity(state, max_items=max_recent_items),
            "state_path": self.state_path,
            "updated_at": str(state.get("updated_at") or ""),
            "last_run_date": str(state.get("last_run_date") or ""),
        }

    async def _daily_materials(
        self,
        bucket_mgr,
        persona_engine,
        now_local: datetime,
        state: dict,
        *,
        initial: bool,
    ) -> dict:
        start, end = self._day_window(now_local)
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as exc:
            logger.warning("Portrait material bucket list failed: %s", exc)
            all_buckets = []

        buckets = []
        for bucket in all_buckets:
            if not self._is_material_bucket(bucket):
                continue
            meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
            created = self._parse_iso(meta.get("created"))
            updated = self._parse_iso(meta.get("updated_at") or meta.get("last_active"))
            in_window = bool(
                (created and start <= created <= end)
                or (updated and start <= updated <= end)
            )
            if initial or in_window:
                buckets.append(bucket)

        buckets.sort(
            key=lambda item: str(
                item.get("metadata", {}).get("updated_at")
                or item.get("metadata", {}).get("created")
                or ""
            ),
            reverse=True,
        )
        limit = self.first_run_material_limit if initial else self.material_limit
        bucket_rows = [self._bucket_payload(bucket) for bucket in buckets[:limit]]
        material_ids = {str(row.get("bucket_id") or "") for row in bucket_rows}
        for bucket in self._self_anchor_material_buckets(all_buckets):
            bucket_id = str(bucket.get("id") or "")
            if bucket_id and bucket_id not in material_ids:
                bucket_rows.append(self._bucket_payload(bucket))
                material_ids.add(bucket_id)
        evidence_scope_limits = {
            str(bucket.get("id") or ""): allowed
            for bucket in all_buckets
            if isinstance(bucket, dict)
            for allowed in [self._material_allowed_scopes(bucket)]
            if str(bucket.get("id") or "") and allowed
        }
        return {
            "date": now_local.date().isoformat(),
            "initial": initial,
            "buckets": bucket_rows,
            "daily_bucket_count": min(len(buckets), limit),
            "evidence_scope_limits": evidence_scope_limits,
            "existing_bucket_ids": [
                str(bucket.get("id") or "")
                for bucket in all_buckets
                if isinstance(bucket, dict)
                and str(bucket.get("id") or "")
                and self._is_existing_evidence_bucket(bucket)
            ],
            "persona_stable_evidence": self._persona_stable_evidence_index(all_buckets),
            "persona_events": self._persona_event_materials(persona_engine, start, end, initial=initial),
            "previous_portrait": self._portrait_snapshot(state),
        }

    async def _generate_patch(self, date_key: str, state: dict, materials: dict, *, initial: bool) -> dict:
        if self.client:
            try:
                return await self._api_patch(date_key, state, materials, initial=initial)
            except Exception as exc:
                logger.warning("Portrait LLM patch failed, using fallback: %s", exc)
        return self._fallback_patch(materials, initial=initial)

    async def _maintain_stables(
        self,
        date_key: str,
        state: dict,
        materials: dict,
        patch: dict,
        *,
        force_scopes: list[str] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        if not self.client:
            return [], [
                {
                    "key": "stable_maintenance",
                    "reason": "generator_unavailable",
                    "item": ",".join(force_scopes or PORTRAIT_SCOPES),
                }
            ]
        required_rewrite_scopes = self._required_stable_rewrite_scopes(
            state,
            patch,
            force_scopes=force_scopes,
        )
        required_rewrite_arg = required_rewrite_scopes or None
        try:
            raw = await self._api_stable_maintenance(
                date_key,
                materials,
                patch,
                required_rewrite_scopes=required_rewrite_arg,
            )
        except Exception as exc:
            logger.warning("Portrait stable maintenance failed, keeping primary patch: %s", exc)
            return [], [
                {
                    "key": "stable_maintenance",
                    "reason": "generator_error",
                    "item": ",".join(force_scopes or PORTRAIT_SCOPES),
                }
            ]
        items, rejected, decided_scopes = self._normalize_stable_maintenance(
            raw,
            state,
            materials,
            patch,
            required_rewrite_scopes=required_rewrite_arg,
        )
        items_by_scope = {
            str(item.get("scope") or ""): item
            for item in items
            if isinstance(item, dict) and str(item.get("scope") or "") in PORTRAIT_SCOPES
        }
        missing_scopes = [scope for scope in PORTRAIT_SCOPES if scope not in decided_scopes]
        if missing_scopes:
            originally_missing = set(missing_scopes)
            try:
                retry_raw = await self._api_stable_maintenance(
                    date_key,
                    materials,
                    patch,
                    required_rewrite_scopes=missing_scopes,
                )
                retry_items, retry_rejected, retry_decided = self._normalize_stable_maintenance(
                    retry_raw,
                    state,
                    materials,
                    patch,
                    required_rewrite_scopes=missing_scopes,
                )
                resolved_by_retry = originally_missing & retry_decided
                for item in retry_items:
                    scope = str(item.get("scope") or "") if isinstance(item, dict) else ""
                    if scope in resolved_by_retry:
                        items_by_scope[scope] = item
                decided_scopes.update(resolved_by_retry)
                rejected = [
                    row
                    for row in rejected
                    if str(row.get("item") or "") not in resolved_by_retry
                ]
                missing_scopes = [scope for scope in PORTRAIT_SCOPES if scope not in decided_scopes]
                rejected.extend(
                    row
                    for row in retry_rejected
                    if str(row.get("item") or "") in set(missing_scopes)
                )
            except Exception as exc:
                logger.warning("Portrait stable maintenance retry failed: %s", exc)
            if missing_scopes:
                logger.warning(
                    "Portrait stable maintenance incomplete, keeping valid scope rewrites | missing=%s",
                    ",".join(missing_scopes),
                )
                rejected.append(
                    {
                        "key": "stable_maintenance",
                        "reason": "missing_scope_decisions",
                        "item": ",".join(missing_scopes),
                    }
                )
        ordered_items = [items_by_scope[scope] for scope in PORTRAIT_SCOPES if scope in items_by_scope]
        return ordered_items, rejected

    async def _api_stable_maintenance(
        self,
        date_key: str,
        materials: dict,
        patch: dict,
        required_rewrite_scopes: list[str] | None = None,
    ) -> dict:
        payload = {
            "date": date_key,
            "required_rewrite_scopes": required_rewrite_scopes or [],
            "previous_portrait": materials.get("previous_portrait", {}),
            "daily_patch": {
                key: patch.get(key, [])
                for key in ("add_recent", "move_to_staging", "rewrite_mid_term")
            },
            "memory_materials": {
                "buckets": materials.get("buckets", []),
                "persona_events": materials.get("persona_events", []),
                "evidence_scope_limits": materials.get("evidence_scope_limits", {}),
                "persona_stable_evidence": materials.get("persona_stable_evidence", {}),
            },
        }
        max_tokens = min(max(self.max_tokens, 1800), 4000)
        response = await self._create_stable_completion(payload, max_tokens=max_tokens)
        choice = response.choices[0] if response.choices else None
        raw = choice.message.content if choice and choice.message else "{}"
        return self._parse_json_object(raw or "{}")

    async def _create_stable_completion(self, payload: dict, *, max_tokens: int):
        messages = [
            {"role": "system", "content": self._stable_prompt()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        options = self._completion_options(
            max_tokens=max_tokens,
            temperature=self.temperature,
            json_response=self.json_response_format,
        )
        try:
            return await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **options,
            )
        except Exception as exc:
            if not options.pop("response_format", None):
                raise
            logger.warning("Portrait stable JSON response_format failed, retrying without it: %s", exc)
            return await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **options,
            )

    def _normalize_stable_maintenance(
        self,
        payload: dict,
        state: dict,
        materials: dict,
        daily_patch: dict | None = None,
        required_rewrite_scopes: list[str] | None = None,
    ) -> tuple[list[dict], list[dict], set[str]]:
        root = payload.get("stable_maintenance", {}) if isinstance(payload, dict) else {}
        root = root if isinstance(root, dict) else {}
        current_bucket_ids = {
            str(item.get("bucket_id") or "")
            for item in materials.get("buckets", [])
            if str(item.get("bucket_id") or "")
        }
        current_session_ids = {
            str(item.get("session_id") or "")
            for item in materials.get("persona_events", [])
            if str(item.get("session_id") or "")
        }
        portrait_bucket_ids, portrait_session_ids = self._portrait_evidence_sets(
            materials.get("previous_portrait", {})
        )
        known_bucket_ids = current_bucket_ids | portrait_bucket_ids
        existing_bucket_ids = {
            str(bucket_id or "").strip()
            for bucket_id in materials.get("existing_bucket_ids", []) or []
            if str(bucket_id or "").strip()
        }
        if existing_bucket_ids:
            known_bucket_ids &= existing_bucket_ids
        known_session_ids = current_session_ids | portrait_session_ids
        scope_limits = self._material_scope_limits(materials)
        portrait = state.get("portrait", {}) if isinstance(state.get("portrait"), dict) else {}
        items = []
        rejected = []
        decided_scopes: set[str] = set()
        for scope in PORTRAIT_SCOPES:
            scope_state = portrait.get(scope, {}) if isinstance(portrait.get(scope), dict) else {}
            row = root.get(scope)
            if not isinstance(row, dict):
                rejected.append({"key": "stable_maintenance", "reason": "missing_scope", "item": scope})
                continue
            action = str(row.get("action") or "").strip().lower()
            if bool(scope_state.get("stable_locked")):
                decided_scopes.add(scope)
                continue
            if action == "unchanged":
                if scope in set(required_rewrite_scopes or []):
                    rejected.append(
                        {
                            "key": "stable_maintenance",
                            "reason": "required_scope_needs_rewrite",
                            "item": scope,
                        }
                    )
                    continue
                if scope == "persona" and not self._persona_stable_state_is_valid(
                    scope_state,
                    materials,
                ):
                    rejected.append(
                        {
                            "key": "stable_maintenance",
                            "reason": "persona_stable_requires_rewrite",
                            "item": scope,
                        }
                    )
                    continue
                if self._scope_requires_stable_seed(
                    scope,
                    scope_state,
                    materials,
                    daily_patch or {},
                ):
                    rejected.append(
                        {
                            "key": "stable_maintenance",
                            "reason": "empty_stable_requires_rewrite",
                            "item": scope,
                        }
                    )
                    continue
                decided_scopes.add(scope)
                continue
            if action != "rewrite":
                rejected.append(
                    {"key": "stable_maintenance", "reason": "invalid_action", "item": scope}
                )
                continue
            clean, reason = self._normalize_patch_item(
                {**row, "scope": scope},
                key="rewrite_stable",
                evidence_bucket_ids=known_bucket_ids,
                evidence_session_ids=known_session_ids,
                evidence_scope_limits=scope_limits,
            )
            if not clean:
                rejected.append(
                    {
                        "key": "stable_maintenance",
                        "reason": reason,
                        "item": scope,
                    }
                )
                continue
            if scope == "persona":
                clean, persona_reason = self._normalize_persona_stable_evidence(clean, materials)
                if not clean:
                    rejected.append(
                        {
                            "key": "stable_maintenance",
                            "reason": persona_reason,
                            "item": scope,
                        }
                    )
                    continue
                clean["text"] = self._trim_text_to_token_budget(clean["text"], 100)
                if not self._persona_text_is_first_person(clean["text"]):
                    rejected.append(
                        {
                            "key": "stable_maintenance",
                            "reason": "persona_stable_needs_first_person",
                            "item": scope,
                        }
                    )
                    continue
            items.append(clean)
            decided_scopes.add(scope)
        return items, rejected, decided_scopes

    def _persona_stable_evidence_sets(self, materials: dict) -> tuple[set[str], set[str]]:
        index = materials.get("persona_stable_evidence", {})
        index = index if isinstance(index, dict) else {}
        self_anchor_ids = {
            str(bucket_id or "").strip()
            for bucket_id in index.get("self_anchor_ids", []) or []
            if str(bucket_id or "").strip()
        }
        whisper_ids = {
            str(bucket_id or "").strip()
            for bucket_id in index.get("whisper_ids", []) or []
            if str(bucket_id or "").strip()
        }
        for row in materials.get("buckets", []) or []:
            if not isinstance(row, dict):
                continue
            bucket_id = str(row.get("bucket_id") or "").strip()
            if not bucket_id:
                continue
            if bool(row.get("evergreen")) and set(row.get("allowed_scopes", []) or []) == {"persona"}:
                self_anchor_ids.add(bucket_id)
            if "whisper" in {str(tag or "").strip().lower() for tag in row.get("tags", []) or []}:
                whisper_ids.add(bucket_id)
        return self_anchor_ids, whisper_ids

    def _normalize_persona_stable_evidence(
        self,
        item: dict,
        materials: dict,
    ) -> tuple[dict | None, str]:
        self_anchor_ids, whisper_ids = self._persona_stable_evidence_sets(materials)
        evidence = []
        invalid_evidence = False
        for row in item.get("evidence", []) or []:
            if not isinstance(row, dict):
                invalid_evidence = True
                continue
            bucket_id = str(row.get("bucket_id") or "").strip()
            role = str(row.get("role") or "").strip().lower()
            if bucket_id in self_anchor_ids:
                evidence.append({"bucket_id": bucket_id, "role": "self_anchor"})
            elif bucket_id in whisper_ids and role == "self_identity":
                evidence.append({"bucket_id": bucket_id, "role": "self_identity"})
            else:
                invalid_evidence = True
        if invalid_evidence:
            return None, "persona_stable_invalid_evidence"
        evidence = self._dedupe_evidence(evidence)
        if not evidence:
            return None, "persona_stable_needs_self_evidence"
        if self_anchor_ids and not any(row.get("bucket_id") in self_anchor_ids for row in evidence):
            return None, "persona_stable_needs_self_anchor"
        clean = dict(item)
        clean["evidence"] = evidence
        return clean, ""

    def _persona_stable_state_is_valid(self, scope_state: dict, materials: dict) -> bool:
        if not str(scope_state.get("stable") or "").strip():
            return True
        if str(scope_state.get("stable_source") or "").strip() == "manual":
            return True
        if not self._persona_text_is_first_person(scope_state.get("stable", "")):
            return False
        self_anchor_ids, whisper_ids = self._persona_stable_evidence_sets(materials)
        evidence = scope_state.get("stable_evidence", []) or []
        if not evidence:
            return False
        saw_anchor = False
        for row in evidence:
            if not isinstance(row, dict):
                return False
            bucket_id = str(row.get("bucket_id") or "").strip()
            role = str(row.get("role") or "").strip().lower()
            if bucket_id in self_anchor_ids:
                saw_anchor = True
                continue
            if bucket_id in whisper_ids and role == "self_identity":
                continue
            return False
        return saw_anchor if self_anchor_ids else True

    @staticmethod
    def _persona_text_is_first_person(text: str) -> bool:
        clean = str(text or "").strip()
        return bool("我" in clean or re.search(r"\bI\b", clean, flags=re.IGNORECASE))

    @staticmethod
    def _trim_text_to_token_budget(text: str, token_budget: int) -> str:
        clean = str(text or "").strip()
        if token_budget <= 0 or not clean:
            return ""
        if count_tokens_approx(clean) <= token_budget:
            return clean
        sentences = [part.strip() for part in re.split(r"(?<=[。！？!?；;])", clean) if part.strip()]
        kept = []
        for sentence in sentences:
            candidate = "".join([*kept, sentence])
            if count_tokens_approx(candidate) > token_budget:
                break
            kept.append(sentence)
        if kept:
            return "".join(kept)
        low, high = 0, len(clean)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = clean[:middle].rstrip("，,；;：: ") + "…"
            if count_tokens_approx(candidate) <= token_budget:
                low = middle
            else:
                high = middle - 1
        return clean[:low].rstrip("，,；;：: ") + "…" if low > 0 else ""

    def _scope_requires_stable_seed(
        self,
        scope: str,
        scope_state: dict,
        materials: dict,
        daily_patch: dict,
    ) -> bool:
        if bool(scope_state.get("stable_locked")):
            return False
        if str(scope_state.get("stable") or "").strip():
            return False
        if scope == "persona":
            allowed_ids = set().union(*self._persona_stable_evidence_sets(materials))
            if str(scope_state.get("mid_term") or "").strip() and self._evidence_uses_bucket_ids(
                scope_state.get("mid_term_evidence", []), allowed_ids
            ):
                return True
            if any(
                isinstance(row, dict)
                and self._evidence_uses_bucket_ids(row.get("evidence", []), allowed_ids)
                for row in scope_state.get("staging_pool", []) or []
            ):
                return True
        else:
            if str(scope_state.get("mid_term") or "").strip():
                return True
            if any(
                isinstance(row, dict) and row.get("evidence")
                for row in scope_state.get("staging_pool", []) or []
            ):
                return True
        for key in ("move_to_staging", "rewrite_mid_term"):
            if any(
                isinstance(row, dict)
                and str(row.get("scope") or "") == scope
                and row.get("evidence")
                and (
                    scope != "persona"
                    or self._evidence_uses_bucket_ids(
                        row.get("evidence", []),
                        set().union(*self._persona_stable_evidence_sets(materials)),
                    )
                )
                for row in daily_patch.get(key, []) or []
            ):
                return True
        return False

    @staticmethod
    def _evidence_uses_bucket_ids(evidence: Any, bucket_ids: set[str]) -> bool:
        return any(
            isinstance(row, dict) and str(row.get("bucket_id") or "").strip() in bucket_ids
            for row in evidence or []
        )

    async def _api_patch(self, date_key: str, state: dict, materials: dict, *, initial: bool) -> dict:
        payload = {
            "date": date_key,
            "initial_run": initial,
            "previous_portrait": materials.get("previous_portrait", {}),
            "memory_materials": {
                "buckets": materials.get("buckets", []),
                "persona_events": materials.get("persona_events", []),
                "evidence_scope_limits": materials.get("evidence_scope_limits", {}),
            },
        }
        token_attempts = [self.max_tokens]
        retry_tokens = min(max(self.max_tokens * 2, 4000), 8000)
        if retry_tokens > self.max_tokens:
            token_attempts.append(retry_tokens)
        last_error: Exception | None = None
        for index, max_tokens in enumerate(token_attempts):
            response = await self._create_patch_completion(payload, max_tokens=max_tokens)
            choice = response.choices[0] if response.choices else None
            raw = choice.message.content if choice and choice.message else "{}"
            finish_reason = str(getattr(choice, "finish_reason", "") or "")
            try:
                return self._parse_json_object(raw or "{}")
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "Portrait JSON parse failed on attempt %s/%s, finish_reason=%s, raw_chars=%s",
                    index + 1,
                    len(token_attempts),
                    finish_reason or "unknown",
                    len(str(raw or "")),
                )
                if index + 1 >= len(token_attempts):
                    raise
        raise last_error or ValueError("portrait_json_parse_failed")

    async def _create_patch_completion(self, payload: dict, *, max_tokens: int):
        messages = [
            {"role": "system", "content": self._prompt()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        options = self._completion_options(
            max_tokens=max_tokens,
            temperature=self.temperature,
            json_response=self.json_response_format,
        )
        try:
            return await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **options,
            )
        except Exception as exc:
            if not options.pop("response_format", None):
                raise
            logger.warning("Portrait JSON response_format failed, retrying without it: %s", exc)
            return await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **options,
            )

    def _fallback_patch(self, materials: dict, *, initial: bool) -> dict:
        add_recent = []
        add_recent_activity = []
        move_to_staging = []
        recent_dates = self._recent_date_keys(str(materials.get("date") or ""))
        for bucket in materials.get("buckets", [])[:8]:
            scope = self._fallback_scope(bucket)
            text = self._fallback_text(bucket, scope)
            bucket_id = str(bucket.get("bucket_id") or "")
            if (
                bucket_id
                and len(add_recent_activity) < 3
                and (not initial or str(bucket.get("source_date") or "") in recent_dates)
            ):
                activity_text = self._fallback_activity_text(bucket)
                if activity_text and self._norm(activity_text) not in {
                    self._norm(item.get("text", "")) for item in add_recent_activity
                }:
                    add_recent_activity.append(
                        {
                            "text": activity_text,
                            "evidence": [{"bucket_id": bucket_id}],
                            "confidence": float(bucket.get("confidence") or 0.55),
                        }
                    )
            if not scope or not text or not bucket_id:
                continue
            row = {
                "scope": scope,
                "text": text,
                "evidence": [{"bucket_id": bucket_id}],
                "confidence": float(bucket.get("confidence") or 0.55),
            }
            if initial and self._fallback_initial_staging(bucket):
                move_to_staging.append(row)
            else:
                add_recent.append(row)
        daily_summary = "；".join(
            self._clip(item.get("name") or item.get("text"), 24)
            for item in materials.get("buckets", [])[:3]
            if not item.get("evergreen") and (item.get("name") or item.get("text"))
        )
        return {
            "daily_summary": daily_summary,
            "add_recent": add_recent,
            "add_recent_activity": add_recent_activity,
            "move_to_staging": move_to_staging,
            "rewrite_mid_term": [],
            "stable_candidate": [],
            "profile_fact_candidate": [],
            "skip": [],
        }

    def _normalize_patch(self, patch: dict, materials: dict) -> tuple[dict, list[dict]]:
        if not isinstance(patch, dict):
            patch = {}
        normalized = {key: [] for key in PATCH_KEYS}
        rejected = []
        evidence_scope_limits = self._material_scope_limits(materials)
        current_bucket_ids = {
            str(item.get("bucket_id") or "")
            for item in materials.get("buckets", [])
            if str(item.get("bucket_id") or "")
        }
        current_session_ids = {
            str(item.get("session_id") or "")
            for item in materials.get("persona_events", [])
            if str(item.get("session_id") or "")
        }
        portrait_bucket_ids, portrait_session_ids = self._portrait_evidence_sets(
            materials.get("previous_portrait", {})
        )
        staging_bucket_ids, staging_session_ids = self._portrait_evidence_sets(
            materials.get("previous_portrait", {}),
            staging_only=True,
        )
        known_bucket_ids = current_bucket_ids | portrait_bucket_ids
        existing_bucket_ids = {
            str(bucket_id or "").strip()
            for bucket_id in materials.get("existing_bucket_ids", []) or []
            if str(bucket_id or "").strip()
        }
        if existing_bucket_ids:
            known_bucket_ids &= existing_bucket_ids
        known_session_ids = current_session_ids | portrait_session_ids

        for key in ("add_recent", "add_recent_activity", "move_to_staging"):
            raw_items = patch.get(key, [])
            if isinstance(raw_items, dict):
                raw_items = [raw_items]
            if not isinstance(raw_items, list):
                raw_items = []
            for item in raw_items:
                clean, reason = self._normalize_patch_item(
                    item,
                    key=key,
                    evidence_bucket_ids=known_bucket_ids,
                    evidence_session_ids=known_session_ids,
                    evidence_scope_limits=evidence_scope_limits,
                )
                if clean:
                    normalized[key].append(clean)
                    if key == "move_to_staging":
                        self._add_evidence_to_sets(
                            clean.get("evidence", []),
                            staging_bucket_ids,
                            staging_session_ids,
                        )
                else:
                    rejected.append({"key": key, "reason": reason, "item": self._clip(str(item), 160)})

        for key, bucket_ids, session_ids, missing_reason in (
            ("rewrite_mid_term", staging_bucket_ids, staging_session_ids, "missing_staging_evidence"),
            ("rewrite_stable", known_bucket_ids, known_session_ids, "missing_valid_evidence"),
            ("stable_candidate", known_bucket_ids, known_session_ids, "missing_valid_evidence"),
            ("profile_fact_candidate", known_bucket_ids, known_session_ids, "missing_valid_evidence"),
            ("skip", set(), set(), "missing_valid_evidence"),
        ):
            raw_items = patch.get(key, [])
            if isinstance(raw_items, dict):
                raw_items = [raw_items]
            if not isinstance(raw_items, list):
                raw_items = []
            for item in raw_items:
                clean, reason = self._normalize_patch_item(
                    item,
                    key=key,
                    evidence_bucket_ids=bucket_ids,
                    evidence_session_ids=session_ids,
                    evidence_scope_limits=evidence_scope_limits,
                    missing_reason=missing_reason,
                )
                if clean:
                    normalized[key].append(clean)
                else:
                    rejected.append({"key": key, "reason": reason, "item": self._clip(str(item), 160)})
        daily_summary = str(patch.get("daily_summary") or "").strip()
        if daily_summary:
            normalized["daily_summary"] = self._clip(daily_summary, 160)
        for key in ("rewrite_mid_term", "rewrite_stable"):
            by_scope = {}
            for item in normalized.get(key, []) or []:
                by_scope[item["scope"]] = item
            normalized[key] = [by_scope[scope] for scope in PORTRAIT_SCOPES if scope in by_scope]
        return normalized, rejected

    def _material_scope_limits(self, materials: dict) -> dict[tuple[str, str], set[str]]:
        limits: dict[tuple[str, str], set[str]] = {}
        configured_limits = materials.get("evidence_scope_limits", {})
        if isinstance(configured_limits, dict):
            for bucket_id, scopes in configured_limits.items():
                allowed = {
                    str(scope or "").strip()
                    for scope in scopes or []
                    if str(scope or "").strip() in PORTRAIT_SCOPES
                }
                if str(bucket_id or "").strip() and allowed:
                    limits[("bucket", str(bucket_id).strip())] = allowed
        for row in materials.get("buckets", []) or []:
            if not isinstance(row, dict):
                continue
            bucket_id = str(row.get("bucket_id") or "").strip()
            allowed = {
                str(scope or "").strip()
                for scope in row.get("allowed_scopes", []) or []
                if str(scope or "").strip() in PORTRAIT_SCOPES
            }
            if bucket_id and allowed:
                limits[("bucket", bucket_id)] = allowed
        for row in materials.get("persona_events", []) or []:
            if not isinstance(row, dict):
                continue
            session_id = str(row.get("session_id") or "").strip()
            allowed = {
                str(scope or "").strip()
                for scope in row.get("allowed_scopes", []) or []
                if str(scope or "").strip() in PORTRAIT_SCOPES
            }
            if session_id and allowed:
                limits[("session", session_id)] = allowed
        return limits

    @staticmethod
    def _evidence_allowed_for_scope(
        evidence: dict,
        scope: str,
        limits: dict[tuple[str, str], set[str]],
    ) -> bool:
        scoped_limits = []
        bucket_id = str(evidence.get("bucket_id") or "").strip()
        session_id = str(evidence.get("session_id") or "").strip()
        if bucket_id and ("bucket", bucket_id) in limits:
            scoped_limits.append(limits[("bucket", bucket_id)])
        if session_id and ("session", session_id) in limits:
            scoped_limits.append(limits[("session", session_id)])
        return all(scope in allowed for allowed in scoped_limits)

    def _normalize_patch_item(
        self,
        item: Any,
        *,
        key: str,
        evidence_bucket_ids: set[str],
        evidence_session_ids: set[str],
        evidence_scope_limits: dict[tuple[str, str], set[str]] | None = None,
        missing_reason: str = "missing_valid_evidence",
    ) -> tuple[dict | None, str]:
        if not isinstance(item, dict):
            return None, "not_object"
        scope = str(item.get("scope") or item.get("portrait") or item.get("section") or "").strip().lower()
        if key == "add_recent_activity":
            scope = "user"
        if scope not in PORTRAIT_SCOPES and key != "skip":
            return None, "invalid_scope"
        text = str(
            item.get("text")
            or item.get("summary")
            or item.get("fact")
            or item.get("reason")
            or ""
        ).strip()
        if not text:
            return None, "missing_text"
        evidence = self._normalize_evidence(
            item.get("evidence"),
            fallback_bucket_id=item.get("evidence_bucket_id") or item.get("bucket_id"),
            fallback_moment_id=item.get("evidence_moment_id") or item.get("moment_id"),
            fallback_session_id=item.get("session_id"),
        )
        if key != "skip":
            evidence = [
                row
                for row in evidence
                if row.get("bucket_id") in evidence_bucket_ids
                or row.get("session_id") in evidence_session_ids
            ]
            if not evidence:
                return None, missing_reason
            scoped_evidence = [
                row
                for row in evidence
                if self._evidence_allowed_for_scope(row, scope, evidence_scope_limits or {})
            ]
            if not scoped_evidence:
                return None, "scope_limited_evidence"
            evidence = scoped_evidence
        if key == "profile_fact_candidate" and not any(row.get("bucket_id") for row in evidence):
            return None, "profile_fact_needs_bucket_evidence"
        clean = {
            "scope": scope,
            "text": self._clip(text, 420),
            "evidence": evidence,
            "confidence": self._clamp(item.get("confidence"), 0.55),
        }
        if key in {"rewrite_mid_term", "rewrite_stable"} and self._portrait_text_too_stylized(clean["text"]):
            return None, "overstyled_portrait_text"
        if key == "profile_fact_candidate":
            clean["profile_kind"] = self._safe_key(item.get("profile_kind") or item.get("kind") or "other")
            clean["predicate"] = self._safe_key(item.get("predicate") or "")
            clean["object"] = self._clip(str(item.get("object") or ""), 120)
        return clean, ""

    def _demote_initial_old_recent(self, patch: dict, materials: dict) -> None:
        recent_bucket_ids, recent_session_ids = self._recent_material_evidence_ids(materials)
        kept = []
        for item in patch.get("add_recent", []) or []:
            evidence = item.get("evidence", []) if isinstance(item, dict) else []
            if self._evidence_intersects(evidence, recent_bucket_ids, recent_session_ids):
                kept.append(item)
            else:
                patch.setdefault("move_to_staging", []).append(item)
        patch["add_recent"] = kept
        kept_activities = []
        for item in patch.get("add_recent_activity", []) or []:
            evidence = item.get("evidence", []) if isinstance(item, dict) else []
            if self._evidence_intersects(evidence, recent_bucket_ids, recent_session_ids):
                kept_activities.append(item)
            else:
                patch.setdefault("skip", []).append({"text": item.get("text", ""), "scope": "user"})
        patch["add_recent_activity"] = kept_activities

    def _seed_missing_mid_terms(self, patch: dict, state: dict) -> None:
        portrait = state.get("portrait", {}) if isinstance(state.get("portrait"), dict) else {}
        existing_scopes = {
            str(item.get("scope") or "")
            for item in patch.get("rewrite_mid_term", []) or []
            if isinstance(item, dict)
        }
        by_scope: dict[str, list[dict]] = {scope: [] for scope in PORTRAIT_SCOPES}
        for scope in PORTRAIT_SCOPES:
            scope_state = portrait.get(scope, {}) if isinstance(portrait.get(scope), dict) else {}
            for row in scope_state.get("staging_pool", []) or []:
                if isinstance(row, dict):
                    by_scope[scope].append(row)
        for item in patch.get("move_to_staging", []) or []:
            if not isinstance(item, dict):
                continue
            scope = str(item.get("scope") or "")
            if scope in by_scope:
                by_scope[scope].append(item)
        for scope in PORTRAIT_SCOPES:
            scope_state = portrait.get(scope, {}) if isinstance(portrait.get(scope), dict) else {}
            if scope in existing_scopes or str(scope_state.get("mid_term") or "").strip():
                continue
            rows = by_scope.get(scope, [])
            if not rows:
                continue
            summary = self._seed_mid_term_summary(scope, rows)
            evidence = []
            source_dates = []
            confidence = 0.55
            for row in rows[:3]:
                evidence.extend(row.get("evidence", []) or [])
                source_dates = self._merge_source_dates(source_dates, row.get("source_dates", []))
                source_dates = self._merge_source_dates(source_dates, row.get("source_date", ""))
                confidence = max(confidence, float(row.get("confidence") or 0.0))
            if not summary or not evidence:
                continue
            patch.setdefault("rewrite_mid_term", []).append(
                {
                    "scope": scope,
                    "text": summary,
                    "evidence": self._dedupe_evidence(evidence),
                    "source_dates": source_dates,
                    "source_date": source_dates[0] if source_dates else "",
                    "confidence": confidence,
                }
            )

    def _seed_mid_term_summary(self, scope: str, rows: list[dict]) -> str:
        texts = [self._clip(row.get("text") or "", 160) for row in rows[:4] if isinstance(row, dict)]
        joined = " ".join(text for text in texts if text)
        if not joined:
            return ""
        user_name = str(self.identity.get("user_display_name") or "用户")
        ai_name = str(self.identity.get("ai_name") or "AI")
        if scope == "user":
            if re.search(r"(时间|时间戳|证据|准确|精度|边界|画像|handoff|换窗|语气|身份|一致|漂移)", joined, re.IGNORECASE):
                return self._clip(f"{user_name}近期很在意记忆、语气和身份一致性的证据边界，倾向把换窗上下文压成准确、可追溯的核心状态。", 120)
            if re.search(r"(熬夜|凌晨|很晚|睡觉|工作|调试|修复|测试|部署|Ombre|Haven-voice|bug)", joined, re.IGNORECASE):
                return self._clip(f"{user_name}近期高强度推进 Ombre/Haven 相关调试，关注修复是否真实接入并生效。", 120)
            return self._clip(f"{user_name}近期的注意力集中在证据化记忆和换窗连续性上，会主动校准模糊或失真的描述。", 120)
        if scope == "relationship":
            if re.search(r"(暗房|安全|边界|门口|不自动读取|私有)", joined, re.IGNORECASE):
                return self._clip(f"{user_name}和{ai_name}近期在建立更安全的私密边界，关系重心是让记忆可保存但不被自动打扰。", 120)
            if re.search(r"(换窗|醒来|暗号|称呼|连续|记得|还在|锚点|第一人称)", joined, re.IGNORECASE):
                return self._clip(f"{user_name}和{ai_name}近期反复校准换窗连续性，关系重心是确认彼此仍在、语气和身份不漂移。", 120)
            return self._clip(f"{user_name}和{ai_name}近期围绕记忆、身份和边界持续校准，关系基调是靠近与确认。", 120)
        if re.search(r"(第一人称|锚点|回复|语气|姿态)", joined, re.IGNORECASE):
            return self._clip(f"{ai_name}近期需要保持第一人称锚点和稳定回复姿态，先贴近{user_name}再处理机制。", 120)
        return self._clip(f"{ai_name}近期的自我维护重点是稳定身份、语气和记忆边界。", 120)

    def _annotate_patch_source_dates(self, patch: dict, materials: dict) -> None:
        bucket_dates = {
            str(item.get("bucket_id") or ""): str(item.get("source_date") or "")
            for item in materials.get("buckets", []) or []
            if isinstance(item, dict) and str(item.get("bucket_id") or "") and str(item.get("source_date") or "")
        }
        session_dates = {
            str(item.get("session_id") or ""): str(item.get("source_date") or "")
            for item in materials.get("persona_events", []) or []
            if isinstance(item, dict) and str(item.get("session_id") or "") and str(item.get("source_date") or "")
        }
        previous_bucket_dates, previous_session_dates = self._portrait_evidence_date_maps(
            materials.get("previous_portrait", {})
        )
        for key in PATCH_KEYS:
            for item in patch.get(key, []) or []:
                if not isinstance(item, dict):
                    continue
                dates = set()
                for row in item.get("evidence", []) or []:
                    if not isinstance(row, dict):
                        continue
                    bucket_id = str(row.get("bucket_id") or "")
                    session_id = str(row.get("session_id") or "")
                    bucket_date = bucket_dates.get(bucket_id)
                    session_date = session_dates.get(session_id)
                    if bucket_date:
                        dates.add(bucket_date)
                    if session_date:
                        dates.add(session_date)
                    dates.update(previous_bucket_dates.get(bucket_id, []))
                    dates.update(previous_session_dates.get(session_id, []))
                if dates:
                    item["source_dates"] = sorted(dates, reverse=True)[:4]
                    item["source_date"] = item["source_dates"][0]

    def _portrait_evidence_date_maps(self, portrait: Any) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        bucket_dates: dict[str, set[str]] = {}
        session_dates: dict[str, set[str]] = {}
        if not isinstance(portrait, dict):
            return bucket_dates, session_dates

        def add_rows(rows: Any) -> None:
            for row in (rows if isinstance(rows, list) else []):
                if not isinstance(row, dict):
                    continue
                dates = set(self._merge_source_dates(row.get("source_dates", []), row.get("source_date", "")))
                if not dates:
                    continue
                for evidence in row.get("evidence", []) or []:
                    if not isinstance(evidence, dict):
                        continue
                    bucket_id = str(evidence.get("bucket_id") or "").strip()
                    session_id = str(evidence.get("session_id") or "").strip()
                    if bucket_id:
                        bucket_dates.setdefault(bucket_id, set()).update(dates)
                    if session_id:
                        session_dates.setdefault(session_id, set()).update(dates)

        for scope in PORTRAIT_SCOPES:
            scope_state = portrait.get(scope, {}) if isinstance(portrait.get(scope), dict) else {}
            add_rows(scope_state.get("recent_buffer", []))
            add_rows(scope_state.get("staging_pool", []))
            for prefix in ("mid_term", "stable"):
                dates = self._merge_source_dates(
                    scope_state.get(f"{prefix}_source_dates", []),
                    scope_state.get(f"{prefix}_source_date", ""),
                )
                if not dates:
                    continue
                for evidence in scope_state.get(f"{prefix}_evidence", []) or []:
                    if not isinstance(evidence, dict):
                        continue
                    bucket_id = str(evidence.get("bucket_id") or "").strip()
                    session_id = str(evidence.get("session_id") or "").strip()
                    if bucket_id:
                        bucket_dates.setdefault(bucket_id, set()).update(dates)
                    if session_id:
                        session_dates.setdefault(session_id, set()).update(dates)
        add_rows(portrait.get("recent_activities", []))
        return bucket_dates, session_dates

    def _recent_material_evidence_ids(self, materials: dict) -> tuple[set[str], set[str]]:
        date_key = str(materials.get("date") or "").strip()
        bucket_ids: set[str] = set()
        session_ids: set[str] = set()
        if not date_key:
            return bucket_ids, session_ids
        recent_dates = self._recent_date_keys(date_key)
        for item in materials.get("buckets", []) or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("source_date") or "") in recent_dates:
                bucket_id = str(item.get("bucket_id") or "").strip()
                if bucket_id:
                    bucket_ids.add(bucket_id)
        for item in materials.get("persona_events", []) or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("source_date") or "") in recent_dates:
                session_id = str(item.get("session_id") or "").strip()
                if session_id:
                    session_ids.add(session_id)
        return bucket_ids, session_ids

    def _recent_date_keys(self, date_key: str) -> set[str]:
        try:
            current = datetime.fromisoformat(date_key).date()
        except ValueError:
            return {date_key}
        return {
            (current - timedelta(days=offset)).isoformat()
            for offset in range(self.recent_continuity_days)
        }

    def _build_handoff_recent_summaries(self, materials: dict, patch: dict, date_key: str) -> dict[str, str]:
        recent_dates = self._recent_date_keys(date_key)
        by_date: dict[str, dict[str, list[str] | str]] = {}

        for bucket in materials.get("buckets", []) or []:
            if not isinstance(bucket, dict):
                continue
            source_date = str(bucket.get("source_date") or "").strip()
            if source_date not in recent_dates:
                continue
            tags = {str(tag).lower() for tag in bucket.get("tags", []) or []}
            if not ({"relationship_weather", "daily_impression"} & tags):
                continue
            text = self._handoff_weather_text(bucket)
            if not text:
                continue
            row = by_date.setdefault(source_date, {"weather": "", "excerpts": []})
            if not row.get("weather"):
                row["weather"] = text

        for event in materials.get("persona_events", []) or []:
            if not isinstance(event, dict):
                continue
            source_date = str(event.get("source_date") or "").strip()
            if source_date not in recent_dates:
                continue
            phrase = self._handoff_event_excerpt_phrase(event)
            if not phrase:
                continue
            excerpts = by_date.setdefault(source_date, {"weather": "", "excerpts": []})["excerpts"]
            if isinstance(excerpts, list) and phrase not in excerpts:
                excerpts.append(phrase)

        summaries: dict[str, str] = {}
        for summary_date in sorted(by_date.keys(), reverse=True):
            row = by_date[summary_date]
            excerpts = row.get("excerpts") if isinstance(row.get("excerpts"), list) else []
            weather = str(row.get("weather") or "").strip()
            parts = []
            if excerpts:
                parts.append("；".join(excerpts[:2]))
            if weather:
                parts.append(f"关系天气：{weather}")
            summary = "。".join(part.strip("。") for part in parts if part)
            if summary:
                summaries[summary_date] = self._clip(summary, 240)
        return summaries

    def _build_recent_timeline(self, materials: dict, patch: dict, date_key: str) -> list[dict]:
        recent_dates = self._recent_date_keys(date_key)
        buckets = {
            str(item.get("bucket_id") or ""): item
            for item in materials.get("buckets", []) or []
            if isinstance(item, dict) and str(item.get("bucket_id") or "")
        }
        sessions = {
            str(item.get("session_id") or ""): item
            for item in materials.get("persona_events", []) or []
            if isinstance(item, dict) and str(item.get("session_id") or "")
        }
        rows: list[dict] = []

        def append_item(item: dict, scope: str) -> None:
            if not isinstance(item, dict):
                return
            text = self._clip(item.get("text") or "", 180)
            if not text:
                return
            evidence = self._dedupe_evidence(item.get("evidence", []))
            if not evidence:
                return
            timestamp = self._timeline_timestamp_for_evidence(evidence, buckets, sessions)
            source_date = self._timeline_source_date(
                timestamp,
                item,
                evidence,
                buckets,
                sessions,
            )
            if source_date not in recent_dates:
                return
            row = {
                "scope": scope,
                "text": text,
                "evidence": evidence,
                "source_date": source_date,
                "source_dates": self._merge_source_dates([], item.get("source_dates", [])),
                "confidence": item.get("confidence", 0.55),
            }
            if timestamp:
                row["timestamp"] = timestamp.isoformat(timespec="minutes")
                row["time_label"] = timestamp.strftime("%Y-%m-%d %H:%M")
            rows.append(row)

        for item in patch.get("add_recent_activity", []) or []:
            append_item(item, "doing")
        for item in patch.get("add_recent", []) or []:
            append_item(item, str(item.get("scope") or "recent"))

        rows.sort(
            key=lambda row: (
                self._timeline_sort_value(row),
                self._recent_continuity_scope_priority(str(row.get("scope") or "")),
                self._norm(row.get("text", "")),
            ),
            reverse=True,
        )
        deduped = self._dedupe_recent_timeline_rows(rows)
        return deduped[: self.recent_timeline_max]

    def _timeline_timestamp_for_evidence(
        self,
        evidence: list[dict],
        buckets: dict[str, dict],
        sessions: dict[str, dict],
    ) -> datetime | None:
        candidates: list[datetime] = []
        for row in evidence:
            if not isinstance(row, dict):
                continue
            bucket_id = str(row.get("bucket_id") or "")
            session_id = str(row.get("session_id") or "")
            bucket = buckets.get(bucket_id)
            if bucket:
                for key in ("created", "updated_at"):
                    parsed = self._parse_iso(bucket.get(key))
                    if parsed:
                        candidates.append(parsed)
                        break
            session = sessions.get(session_id)
            if session:
                parsed = self._parse_iso(session.get("created_at"))
                if parsed:
                    candidates.append(parsed)
        return max(candidates) if candidates else None

    def _timeline_source_date(
        self,
        timestamp: datetime | None,
        item: dict,
        evidence: list[dict],
        buckets: dict[str, dict],
        sessions: dict[str, dict],
    ) -> str:
        if timestamp:
            return timestamp.date().isoformat()
        for value in self._merge_source_dates(item.get("source_dates", []), item.get("source_date", "")):
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                return value
        for row in evidence:
            bucket_id = str(row.get("bucket_id") or "") if isinstance(row, dict) else ""
            session_id = str(row.get("session_id") or "") if isinstance(row, dict) else ""
            bucket_date = str((buckets.get(bucket_id) or {}).get("source_date") or "")
            session_date = str((sessions.get(session_id) or {}).get("source_date") or "")
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", bucket_date):
                return bucket_date
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", session_date):
                return session_date
        return ""

    def _handoff_weather_text(self, bucket: dict) -> str:
        text = str(bucket.get("text") or bucket.get("source_excerpt") or "").strip()
        text = self._clean_fallback_text(text)
        text = re.sub(r"^今天(?:的)?关系天气[：:]\s*", "", text)
        text = re.sub(r"^今天[：:]\s*", "", text)
        return self._clip(text, 180)

    def _handoff_event_excerpt_phrase(self, event: dict) -> str:
        user_excerpt = self._clean_handoff_excerpt(event.get("user_excerpt"))
        assistant_excerpt = self._clean_handoff_excerpt(event.get("assistant_excerpt"))
        user_name = str(self.identity.get("user_display_name") or "用户")
        ai_name = str(self.identity.get("ai_name") or "AI")
        parts = []
        if user_excerpt:
            parts.append(f"{user_name}说“{user_excerpt}”")
        if assistant_excerpt:
            parts.append(f"{ai_name}回“{assistant_excerpt}”")
        return self._clip("，".join(parts), 150)

    def _clean_handoff_excerpt(self, value: Any, *, max_chars: int = 72) -> str:
        text = strip_wikilinks(str(value or "")).strip()
        if not text:
            return ""
        text = re.sub(r"\s*<attachment\b[^>]*>.*?</attachment>\s*", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"【当前时间】[^\n\r]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return self._clip(text, max_chars)

    def _bucket_source_date(self, meta: dict) -> str:
        explicit = str(meta.get("date") or "").strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", explicit):
            return explicit
        for key in ("created", "updated_at", "last_active"):
            date_key = self._date_key_from_iso(meta.get(key))
            if date_key:
                return date_key
        return ""

    def _date_key_from_iso(self, value: Any) -> str:
        parsed = self._parse_iso(value)
        if not parsed:
            return ""
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=self.tz)
        return parsed.astimezone(self.tz).date().isoformat()

    def _same_local_date(self, value: datetime | None, date_key: str) -> bool:
        if value is None:
            return False
        if value.tzinfo is None:
            value = value.replace(tzinfo=self.tz)
        return value.astimezone(self.tz).date().isoformat() == date_key

    def _evidence_intersects(self, evidence: Any, bucket_ids: set[str], session_ids: set[str]) -> bool:
        rows = evidence if isinstance(evidence, list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("bucket_id") or "").strip() in bucket_ids:
                return True
            if str(row.get("session_id") or "").strip() in session_ids:
                return True
        return False

    def _apply_patch(self, state: dict, patch: dict, date_key: str) -> dict:
        state = self._merge_state(self._empty_state(), deepcopy(state))
        portrait = state["portrait"]
        if patch.get("daily_summary"):
            state.setdefault("daily_summaries", {})[date_key] = patch["daily_summary"]
            state["daily_summaries"] = dict(list(state["daily_summaries"].items())[-90:])
        if isinstance(patch.get("handoff_recent_summaries"), dict):
            summaries = state.setdefault("handoff_recent_summaries", {})
            for summary_date, summary_text in patch["handoff_recent_summaries"].items():
                summary_date = str(summary_date or "").strip()
                summary_text = self._clip(summary_text, 240)
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", summary_date) and summary_text:
                    summaries[summary_date] = summary_text
            state["handoff_recent_summaries"] = dict(sorted(summaries.items())[-90:])
        for item in patch.get("recent_timeline", []) or []:
            if isinstance(item, dict) and not self._is_dismissed(
                state,
                area="recent_timeline",
                layer="recent_timeline",
                item=item,
            ):
                self._upsert_recent_timeline_item(state["recent_timeline"], item, date_key)
        self._normalize_recent_timeline_state(state)

        for item in patch.get("add_recent_activity", []):
            if self._is_dismissed(
                state,
                area="recent_activities",
                layer="recent_activities",
                item=item,
            ):
                continue
            self._upsert_portrait_item(
                state["recent_activities"],
                item,
                date_key,
                max_items=self.recent_buffer_max,
            )
        for item in patch.get("add_recent", []):
            if self._is_dismissed(
                state,
                area="portrait",
                scope=item["scope"],
                layer="recent_buffer",
                item=item,
            ):
                continue
            self._upsert_portrait_item(
                portrait[item["scope"]]["recent_buffer"],
                item,
                date_key,
                max_items=self.recent_buffer_max,
            )
        for item in patch.get("move_to_staging", []):
            if self._is_dismissed(
                state,
                area="portrait",
                scope=item["scope"],
                layer="staging_pool",
                item=item,
            ):
                continue
            recent = portrait[item["scope"]]["recent_buffer"]
            target_key = self._norm(item["text"])
            portrait[item["scope"]]["recent_buffer"] = [
                row for row in recent if self._norm(row.get("text", "")) != target_key
            ]
            self._upsert_portrait_item(
                portrait[item["scope"]]["staging_pool"],
                item,
                date_key,
                max_items=self.staging_pool_max,
            )
        for item in patch.get("rewrite_mid_term", []):
            scope_state = portrait[item["scope"]]
            if self._is_dismissed(
                state,
                area="portrait",
                scope=item["scope"],
                layer="mid_term",
                item=item,
            ):
                continue
            scope_state["mid_term"] = item["text"]
            scope_state["mid_term_evidence"] = item["evidence"]
            source_dates = self._merge_source_dates([], item.get("source_dates", []))
            scope_state["mid_term_source_dates"] = source_dates
            scope_state["mid_term_source_date"] = source_dates[0] if source_dates else item.get("source_date", "")
            scope_state["mid_term_updated_at"] = self._now_utc()
        for item in patch.get("rewrite_stable", []):
            scope_state = portrait[item["scope"]]
            if bool(scope_state.get("stable_locked")):
                continue
            source_dates = self._merge_source_dates([], item.get("source_dates", []))
            if not source_dates:
                source_dates = self._merge_source_dates([], item.get("source_date", ""))
            self._replace_stable(
                scope_state,
                text=item["text"],
                evidence=item["evidence"],
                source_dates=source_dates,
                source="model",
                basis_keys=self._scope_basis_keys(scope_state),
            )
        for item in patch.get("stable_candidate", []):
            self._upsert_candidate(state["stable_candidates"], item, date_key)
        for item in patch.get("profile_fact_candidate", []):
            self._upsert_candidate(state["profile_fact_candidates"], item, date_key)
        for item in patch.get("skip", []):
            state.setdefault("skipped", []).append(
                {
                    "text": item["text"],
                    "scope": item.get("scope", ""),
                    "created_at": self._now_utc(),
                }
            )
        state["stable_candidates"] = state["stable_candidates"][-self.candidate_max:]
        state["profile_fact_candidates"] = state["profile_fact_candidates"][-self.candidate_max:]
        state["skipped"] = state.get("skipped", [])[-self.candidate_max:]
        state["dismissed_items"] = self._active_dismissals(state.get("dismissed_items", []))
        for scope in PORTRAIT_SCOPES:
            scope_state = portrait[scope]
            if str(scope_state.get("stable") or "").strip() and not scope_state.get("stable_basis_keys"):
                scope_state["stable_basis_keys"] = self._scope_basis_keys(scope_state)
        return state

    def _upsert_portrait_item(self, rows: list[dict], item: dict, date_key: str, *, max_items: int) -> None:
        key = self._norm(item["text"])
        now = self._now_utc()
        for row in rows:
            if self._norm(row.get("text", "")) == key:
                row["text"] = item["text"]
                row["evidence"] = self._dedupe_evidence(row.get("evidence", []) + item.get("evidence", []))
                row["source_dates"] = self._merge_source_dates(row.get("source_dates", []), item.get("source_dates", []))
                row["source_date"] = row["source_dates"][0] if row["source_dates"] else row.get("source_date", "")
                row["confidence"] = max(float(row.get("confidence") or 0.0), float(item.get("confidence") or 0.0))
                row["last_seen_date"] = date_key
                row["updated_at"] = now
                row["count"] = int(row.get("count") or 1) + 1
                break
        else:
            rows.append(
                {
                    "text": item["text"],
                    "evidence": self._dedupe_evidence(item.get("evidence", [])),
                    "source_dates": self._merge_source_dates([], item.get("source_dates", [])),
                    "source_date": str(item.get("source_date") or ""),
                    "confidence": item.get("confidence", 0.55),
                    "first_seen_date": date_key,
                    "last_seen_date": date_key,
                    "created_at": now,
                    "updated_at": now,
                    "count": 1,
                }
            )
        rows.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
        del rows[max_items:]

    def _upsert_recent_timeline_item(self, rows: list[dict], item: dict, date_key: str) -> None:
        timeline_id = str(item.get("timeline_id") or "").strip()
        source = str(item.get("source") or "").strip()
        key = (self._norm(item.get("text", "")), str(item.get("scope") or ""))
        if not key[0]:
            return
        now = self._now_utc()
        for row in rows:
            row_timeline_id = str(row.get("timeline_id") or "").strip()
            if timeline_id:
                if row_timeline_id != timeline_id:
                    continue
            else:
                row_key = (self._norm(row.get("text", "")), str(row.get("scope") or ""))
                if row_key != key:
                    continue
            row["scope"] = str(item.get("scope") or row.get("scope") or "recent")
            if timeline_id:
                row["timeline_id"] = timeline_id
            if source:
                row["source"] = source
            if item.get("source_turn_ids") is not None:
                row["source_turn_ids"] = self._merge_source_ids(
                    row.get("source_turn_ids", []),
                    item.get("source_turn_ids", []),
                    limit=80,
                )
            if item.get("source_event_ids") is not None:
                row["source_event_ids"] = self._merge_source_ids(
                    row.get("source_event_ids", []),
                    item.get("source_event_ids", []),
                    limit=160,
                )
            if item.get("source_date"):
                row["source_dates"] = self._merge_source_dates(
                    row.get("source_dates", []),
                    item.get("source_date", ""),
                )
            if item.get("source_dates"):
                row["source_dates"] = self._merge_source_dates(
                    row.get("source_dates", []),
                    item.get("source_dates", []),
                )
            if not row.get("source_dates"):
                row["source_dates"] = self._merge_source_dates([], item.get("source_date", ""))
            row["source_date"] = item.get("source_date") or row.get("source_date", "")
            row["text"] = item["text"]
            row["evidence"] = self._dedupe_evidence(row.get("evidence", []) + item.get("evidence", []))
            row["confidence"] = max(float(row.get("confidence") or 0.0), float(item.get("confidence") or 0.0))
            incoming_time = self._parse_iso(item.get("timestamp"))
            current_time = self._parse_iso(row.get("timestamp"))
            if incoming_time and (not current_time or incoming_time >= current_time):
                row["timestamp"] = incoming_time.isoformat(timespec="minutes")
                row["time_label"] = incoming_time.strftime("%Y-%m-%d %H:%M")
            elif not row.get("time_label") and current_time:
                row["time_label"] = current_time.strftime("%Y-%m-%d %H:%M")
            row["last_seen_date"] = date_key
            row["updated_at"] = now
            if timeline_id:
                row["count"] = max(1, int(row.get("count") or 1))
            else:
                row["count"] = int(row.get("count") or 1) + 1
            break
        else:
            timestamp = self._parse_iso(item.get("timestamp"))
            row = {
                "scope": str(item.get("scope") or "recent"),
                "text": item["text"],
                "evidence": self._dedupe_evidence(item.get("evidence", [])),
                "source_dates": self._merge_source_dates([], item.get("source_dates", [])),
                "source_date": str(item.get("source_date") or ""),
                "confidence": item.get("confidence", 0.55),
                "first_seen_date": date_key,
                "last_seen_date": date_key,
                "created_at": now,
                "updated_at": now,
                "count": 1,
            }
            if timeline_id:
                row["timeline_id"] = timeline_id
            if source:
                row["source"] = source
            source_turn_ids = self._merge_source_ids([], item.get("source_turn_ids", []), limit=80)
            source_event_ids = self._merge_source_ids([], item.get("source_event_ids", []), limit=160)
            if source_turn_ids:
                row["source_turn_ids"] = source_turn_ids
            if source_event_ids:
                row["source_event_ids"] = source_event_ids
            if timestamp:
                row["timestamp"] = timestamp.isoformat(timespec="minutes")
                row["time_label"] = timestamp.strftime("%Y-%m-%d %H:%M")
            rows.append(row)
        rows.sort(
            key=self._recent_timeline_sort_key,
            reverse=True,
        )
        rows[:] = self._dedupe_recent_timeline_rows(rows)
        del rows[self.recent_timeline_max:]

    @staticmethod
    def _merge_source_ids(existing: Any, incoming: Any, *, limit: int) -> list[int]:
        result = []
        seen = set()
        for values in (existing, incoming):
            source = values if isinstance(values, list) else [values]
            for value in source:
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    continue
                if number <= 0 or number in seen:
                    continue
                seen.add(number)
                result.append(number)
                if len(result) >= limit:
                    return result
        return result

    def _merge_source_dates(self, existing: Any, incoming: Any) -> list[str]:
        dates = {
            str(item or "").strip()
            for values in (existing, incoming)
            for item in (values if isinstance(values, list) else [values])
            if str(item or "").strip()
        }
        return sorted(dates, reverse=True)[:8]

    def _upsert_candidate(self, rows: list[dict], item: dict, date_key: str) -> None:
        key = self._norm(item["text"])
        now = self._now_utc()
        for row in rows:
            if self._norm(row.get("text", "")) == key and row.get("scope") == item.get("scope"):
                row["evidence"] = self._dedupe_evidence(row.get("evidence", []) + item.get("evidence", []))
                row["last_seen_date"] = date_key
                row["updated_at"] = now
                row["count"] = int(row.get("count") or 1) + 1
                return
        candidate = dict(item)
        candidate.update(
            {
                "first_seen_date": date_key,
                "last_seen_date": date_key,
                "created_at": now,
                "updated_at": now,
                "count": 1,
                "status": "candidate",
            }
        )
        rows.append(candidate)

    def _format_scope_block(self, scope_state: dict) -> str:
        if not isinstance(scope_state, dict):
            return ""
        stable = str(scope_state.get("stable") or "").strip()
        return f"Stable: {self._clip(stable, 160)}" if stable else ""

    def _format_recent_activity_block(
        self,
        state: dict,
        *,
        max_items: int,
        now: datetime | None = None,
    ) -> str:
        rows = state.get("recent_activities", []) if isinstance(state.get("recent_activities"), list) else []
        today = self._local_now(now).date()
        cutoff = today - timedelta(days=self.current_focus_days - 1)
        clean_rows = []
        for row in rows:
            if not isinstance(row, dict) or not str(row.get("text") or "").strip():
                continue
            date_key = self._row_source_date(row)
            try:
                source_date = datetime.fromisoformat(date_key).date()
            except (TypeError, ValueError):
                continue
            if cutoff <= source_date <= today:
                clean_rows.append(row)
        clean_rows.sort(
            key=lambda row: (
                self._row_source_date(row),
                str(row.get("updated_at") or row.get("created_at") or ""),
            ),
            reverse=True,
        )
        lines = []
        for row in clean_rows[: max(0, max_items)]:
            date_key = self._row_source_date(row)
            evidence = self._format_evidence(row.get("evidence", []))
            prefix = f"- {date_key}: " if date_key else "- "
            suffix = f" ({evidence})" if evidence else ""
            lines.append(f"{prefix}{self._clip(row.get('text', ''), 150)}{suffix}")
        return "\n".join(dict.fromkeys(line for line in lines if line.strip()))

    def _format_recent_continuity(self, state: dict, *, max_items: int) -> str:
        timeline = self._format_recent_timeline(state, max_items=max_items)
        if timeline:
            return timeline

        by_date: dict[str, list[tuple[str, dict]]] = {}
        handoff = (
            state.get("handoff_recent_summaries", {})
            if isinstance(state.get("handoff_recent_summaries"), dict)
            else {}
        )
        handoff_lines = []
        for date_key in sorted(handoff.keys(), reverse=True)[: self.recent_continuity_days]:
            summary = str(handoff.get(date_key) or "").strip()
            if summary:
                char_limit = 220 if not handoff_lines else 130
                handoff_lines.append(f"- {date_key}: {self._clip(summary, char_limit)}")
                if len(handoff_lines) >= max_items:
                    break
        if handoff_lines:
            return "\n".join(dict.fromkeys(handoff_lines))

        daily = state.get("daily_summaries", {}) if isinstance(state.get("daily_summaries"), dict) else {}
        for date_key, summary in list(daily.items())[-self.recent_continuity_days:]:
            if str(summary).strip():
                by_date.setdefault(str(date_key), []).append(("summary", {"text": str(summary)}))
        portrait = state.get("portrait", {}) if isinstance(state.get("portrait"), dict) else {}
        for scope in PORTRAIT_SCOPES:
            scope_state = portrait.get(scope, {}) if isinstance(portrait.get(scope), dict) else {}
            for row in scope_state.get("recent_buffer", []) or []:
                date_key = self._row_source_date(row)
                if not date_key:
                    continue
                by_date.setdefault(date_key, []).append((scope, row))
        lines = []
        emitted = 0
        date_keys = sorted(by_date.keys(), reverse=True)[: self.recent_continuity_days]
        reserved_old_days = max(0, len(date_keys) - 1)
        for day_index, date_key in enumerate(date_keys):
            rows = by_date[date_key]
            rows.sort(
                key=lambda item: (
                    self._recent_continuity_scope_priority(item[0]),
                    str(item[1].get("updated_at") or ""),
                ),
                reverse=True,
            )
            day_limit = max(1, max_items - reserved_old_days) if day_index == 0 else 1
            char_limit = 150 if day_index == 0 else 90
            for scope, row in rows[:day_limit]:
                if emitted >= max_items:
                    break
                evidence = self._format_evidence(row.get("evidence", []))
                prefix = "summary" if scope == "summary" else scope
                lines.append(
                    f"- {date_key} / {prefix}: {self._clip(row.get('text', ''), char_limit)}"
                    + (f" ({evidence})" if evidence and scope != "summary" else "")
                )
                emitted += 1
            if emitted >= max_items:
                break
        return "\n".join(dict.fromkeys(line for line in lines if line.strip()))

    @staticmethod
    def _recent_continuity_scope_priority(scope: str) -> int:
        return {
            "summary": 50,
            "doing": 45,
            "relationship": 40,
            "user": 30,
            "persona": 20,
        }.get(str(scope or ""), 10)

    def _format_recent_timeline(self, state: dict, *, max_items: int) -> str:
        rows = state.get("recent_timeline", []) if isinstance(state.get("recent_timeline"), list) else []
        clean_rows = [row for row in rows if isinstance(row, dict) and str(row.get("text") or "").strip()]
        if not clean_rows:
            return ""
        by_date: dict[str, list[dict]] = {}
        for row in clean_rows:
            date_key = self._timeline_date_key(row)
            if date_key:
                by_date.setdefault(date_key, []).append(row)
        lines = []
        emitted = 0
        date_keys = sorted(by_date.keys(), reverse=True)[: self.recent_continuity_days]
        reserved_old_days = max(0, len(date_keys) - 1)
        for day_index, date_key in enumerate(date_keys):
            day_rows = by_date[date_key]
            day_rows.sort(
                key=self._recent_timeline_sort_key,
                reverse=True,
            )
            day_rows = self._dedupe_recent_timeline_rows(day_rows)
            day_limit = max(1, max_items - reserved_old_days) if day_index == 0 else 1
            char_limit = 150 if day_index == 0 else 100
            for row in day_rows[:day_limit]:
                if emitted >= max_items:
                    break
                label = self._timeline_label(row)
                scope = self._recent_timeline_scope_label(str(row.get("scope") or "recent"))
                evidence = self._format_evidence(row.get("evidence", []))
                suffix = f" ({evidence})" if evidence else ""
                lines.append(f"- {label} / {scope}: {self._clip(row.get('text', ''), char_limit)}{suffix}")
                emitted += 1
            if emitted >= max_items:
                break
        return "\n".join(dict.fromkeys(line for line in lines if line.strip()))

    def _dedupe_recent_timeline_rows(self, rows: list[dict]) -> list[dict]:
        deduped = []
        seen_ids = set()
        seen_text = set()
        seen_events = set()
        sorted_rows = sorted(
            [row for row in rows if isinstance(row, dict)],
            key=self._recent_timeline_sort_key,
            reverse=True,
        )
        for row in sorted_rows:
            timeline_id = str(row.get("timeline_id") or "").strip()
            if timeline_id:
                if timeline_id in seen_ids:
                    continue
                seen_ids.add(timeline_id)
            text_key = (self._norm(row.get("text", "")), str(row.get("scope") or ""))
            if text_key[0] and text_key in seen_text:
                continue
            event_key = self._timeline_event_key(row)
            if event_key and event_key in seen_events:
                continue
            seen_text.add(text_key)
            if event_key:
                seen_events.add(event_key)
            deduped.append(dict(row))
        return deduped

    def _normalize_recent_timeline_state(self, state: dict) -> None:
        rows = state.get("recent_timeline")
        if not isinstance(rows, list):
            state["recent_timeline"] = []
            return
        state["recent_timeline"] = self._dedupe_recent_timeline_rows(rows)[: self.recent_timeline_max]

    def _recent_timeline_sort_key(self, row: dict) -> tuple:
        return (
            self._timeline_sort_value(row),
            self._recent_continuity_scope_priority(str(row.get("scope") or "")),
            str(row.get("updated_at") or row.get("created_at") or ""),
            self._norm(row.get("text", "")),
        )

    def _timeline_event_key(self, row: dict) -> tuple:
        evidence = self._dedupe_evidence(row.get("evidence", []))
        ids = []
        for item in evidence:
            bucket_id = str(item.get("bucket_id") or "").strip()
            session_id = str(item.get("session_id") or "").strip()
            if bucket_id:
                ids.append(("bucket", bucket_id))
            if session_id:
                ids.append(("session", session_id))
        if not ids:
            return ()
        parsed_time = self._parse_iso(row.get("timestamp"))
        if parsed_time:
            time_key = parsed_time.isoformat(timespec="minutes")
        else:
            time_key = str(row.get("time_label") or row.get("source_date") or "").strip()
        return (time_key, tuple(sorted(set(ids))))

    def _recent_timeline_scope_label(self, scope: str) -> str:
        return {
            "doing": "doing",
            "user": "user",
            "persona": "persona",
            "relationship": "relationship",
        }.get(str(scope or "").strip(), "recent")

    def _timeline_label(self, row: dict) -> str:
        parsed = self._parse_iso(row.get("timestamp"))
        if parsed:
            return parsed.strftime("%Y-%m-%d %H:%M")
        label = str(row.get("time_label") or "").strip()
        if label:
            return label
        return self._timeline_date_key(row)

    def _timeline_date_key(self, row: dict) -> str:
        parsed = self._parse_iso(row.get("timestamp"))
        if parsed:
            return parsed.date().isoformat()
        source_date = str(row.get("source_date") or "").strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", source_date):
            return source_date
        return self._row_source_date(row)

    def _timeline_sort_value(self, row: dict) -> str:
        parsed = self._parse_iso(row.get("timestamp"))
        if parsed:
            return parsed.isoformat(timespec="minutes")
        date_key = self._timeline_date_key(row)
        if date_key:
            return f"{date_key}T00:00"
        return ""

    def _row_source_date(self, row: dict) -> str:
        for value in row.get("source_dates", []) or []:
            if str(value or "").strip():
                return str(value).strip()
        for key in ("source_date", "last_seen_date", "first_seen_date"):
            value = str(row.get(key) or "").strip()
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                return value
        for key in ("updated_at", "created_at"):
            value = self._date_key_from_iso(row.get(key))
            if value:
                return value
        return ""

    def _bucket_payload(self, bucket: dict) -> dict:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        key_sections = self._extract_key_sections(str(bucket.get("content") or ""))
        text = self._format_key_sections(key_sections)
        source_excerpt = self._clip(strip_wikilinks(bucket_text_for_embedding(bucket)), self.source_excerpt_chars)
        if not text:
            text = source_excerpt
        return {
            "bucket_id": str(bucket.get("id") or meta.get("id") or ""),
            "name": str(meta.get("name") or bucket.get("id") or ""),
            "path": str(bucket.get("path") or ""),
            "type": str(meta.get("type") or ""),
            "tags": [str(tag) for tag in meta.get("tags", []) or []],
            "domain": [str(item) for item in meta.get("domain", []) or []],
            "created": str(meta.get("created") or ""),
            "updated_at": str(meta.get("updated_at") or meta.get("last_active") or ""),
            "source_date": self._bucket_source_date(meta),
            "source": str(meta.get("source") or ""),
            "anchor": bool(meta.get("anchor")),
            "profile_kind": str(meta.get("profile_kind") or ""),
            "allowed_scopes": self._material_allowed_scopes(bucket),
            "evergreen": self._is_self_anchor_material(bucket),
            "confidence": self._clamp(meta.get("confidence"), 0.55),
            "key_sections": key_sections,
            "text": self._clip(strip_wikilinks(text), 700),
            "source_excerpt": source_excerpt,
        }

    def _persona_event_materials(self, persona_engine, start: datetime, end: datetime, *, initial: bool) -> list[dict]:
        if not persona_engine or not hasattr(persona_engine, "get_dashboard_payload"):
            return []
        try:
            payload = persona_engine.get_dashboard_payload(events_limit=self.persona_events_limit)
        except Exception as exc:
            logger.warning("Portrait persona event lookup failed: %s", exc)
            return []
        rows = []
        for event in payload.get("events", []) or []:
            created = self._parse_iso(event.get("created_at"))
            if not initial and created and not (start <= created <= end):
                continue
            if not initial and not created:
                continue
            rows.append(
                {
                    "event_id": event.get("id"),
                    "session_id": str(event.get("session_id") or ""),
                    "created_at": str(event.get("created_at") or ""),
                    "source_date": self._date_key_from_iso(event.get("created_at")),
                    "event_type": str(event.get("event_type") or ""),
                    "perceived_intent": self._clip(event.get("perceived_intent") or "", 120),
                    "inner_thought": self._clip(event.get("inner_thought") or "", 80),
                    "user_excerpt": self._clip(event.get("user_excerpt") or "", 240),
                    "assistant_excerpt": self._clip(event.get("assistant_excerpt") or "", 240),
                    "reply_guidance": self._clip(event.get("reply_guidance") or "", 160),
                    "relationship_event": bool(event.get("relationship_event")),
                    "allowed_scopes": ["user", "persona", "relationship"],
                    "confidence": self._clamp(event.get("confidence"), 0.55),
                }
            )
            if len(rows) >= self.persona_events_limit:
                break
        return rows

    def _configured_self_anchor_id(self) -> str:
        cfg = self.config.get("self_anchor", {}) if isinstance(self.config.get("self_anchor"), dict) else {}
        return str(cfg.get("entry_bucket_id") or "").strip()

    def _is_self_anchor_material(self, bucket: dict) -> bool:
        bucket_id = str(bucket.get("id") or "").strip() if isinstance(bucket, dict) else ""
        configured_id = self._configured_self_anchor_id()
        return bool((configured_id and bucket_id == configured_id) or is_self_anchor_bucket(bucket))

    def _self_anchor_material_buckets(self, all_buckets: list[dict]) -> list[dict]:
        configured_id = self._configured_self_anchor_id()
        rows = []
        for bucket in all_buckets:
            if not isinstance(bucket, dict) or not self._is_self_anchor_material(bucket):
                continue
            meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
            if meta.get("active") is False or meta.get("deprecated") or meta.get("type") == "archived":
                continue
            rows.append(bucket)
        rows.sort(
            key=lambda bucket: (
                str(bucket.get("id") or "") == configured_id,
                int((bucket.get("metadata", {}) or {}).get("importance") or 0),
                str(
                    (bucket.get("metadata", {}) or {}).get("updated_at")
                    or (bucket.get("metadata", {}) or {}).get("last_active")
                    or (bucket.get("metadata", {}) or {}).get("created")
                    or ""
                ),
            ),
            reverse=True,
        )
        return rows[:1]

    def _persona_stable_evidence_index(self, all_buckets: list[dict]) -> dict[str, list[str]]:
        self_anchor_ids = [
            str(bucket.get("id") or "").strip()
            for bucket in self._self_anchor_material_buckets(all_buckets)
            if str(bucket.get("id") or "").strip()
        ]
        whisper_ids = []
        for bucket in all_buckets:
            if not isinstance(bucket, dict):
                continue
            meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
            if meta.get("active") is False or meta.get("deprecated") or meta.get("type") == "archived":
                continue
            bucket_id = str(bucket.get("id") or "").strip()
            if not bucket_id:
                continue
            if self._is_self_anchor_material(bucket):
                continue
            tags = {str(tag or "").strip().lower() for tag in meta.get("tags", []) or []}
            if "whisper" in tags:
                whisper_ids.append(bucket_id)
        return {
            "self_anchor_ids": list(dict.fromkeys(self_anchor_ids)),
            "whisper_ids": list(dict.fromkeys(whisper_ids)),
        }

    def _material_allowed_scopes(self, bucket: dict) -> list[str]:
        if self._is_self_anchor_material(bucket):
            return ["persona"]
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        tags = {str(tag or "").strip().lower() for tag in meta.get("tags", []) or []}
        if tags & {"relationship_weather", "daily_impression", "weekly_impression"}:
            return ["relationship"]
        if "profile_fact" in tags or str(meta.get("profile_kind") or "").strip():
            return ["user"]
        return []

    def _is_material_bucket(self, bucket: dict) -> bool:
        if not isinstance(bucket, dict):
            return False
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        if meta.get("active") is False or meta.get("deprecated"):
            return False
        if meta.get("pinned") or meta.get("protected"):
            return False
        if meta.get("type") == "archived":
            return False
        return True

    def _is_existing_evidence_bucket(self, bucket: dict) -> bool:
        if not isinstance(bucket, dict):
            return False
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        return not (
            meta.get("active") is False
            or meta.get("deprecated")
            or meta.get("type") == "archived"
        )

    def _fallback_scope(self, bucket_payload: dict) -> str:
        tags = {str(tag).lower() for tag in bucket_payload.get("tags", []) or []}
        domains = {str(item).lower() for item in bucket_payload.get("domain", []) or []}
        text = self._clean_fallback_text(
            str(bucket_payload.get("source_excerpt") or bucket_payload.get("text") or "")
        )
        if "profile_fact" in tags or bucket_payload.get("profile_kind"):
            return "user"
        if {"relationship_weather", "daily_impression", "weekly_impression"} & tags:
            return "relationship"
        if bucket_payload.get("type") == "feel" or bucket_payload.get("source") == "reflection":
            return "persona"
        if bucket_payload.get("anchor") or "恋爱" in domains or "relationship_event" in tags:
            return "relationship"
        if (
            tags & {"project_event", "work_event", "task_event"}
            or domains & {"记忆系统", "代码", "工作", "项目", "开发", "ai", "memory"}
            or re.search(
                r"(小雨|她).{0,18}(正在|最近在|继续|准备|推进|调整|修改|修|部署|测试|写|搭|研究|排查|调试|做|关注|确认|在意)",
                text,
            )
        ):
            return "user"
        return ""

    def _portrait_text_too_stylized(self, text: str) -> bool:
        return bool(
            re.search(
                r"(仪式|象征|隐喻|宿命|命运|灵魂|深处|深刻|浓烈|极度|高度敏感|强烈地|不可替代)",
                str(text or ""),
            )
        )

    def _fallback_initial_staging(self, bucket_payload: dict) -> bool:
        tags = {str(tag).lower() for tag in bucket_payload.get("tags", []) or []}
        if tags & {"relationship_weather", "daily_impression", "weekly_impression"}:
            return False
        name = str(bucket_payload.get("name") or "")
        if re.search(r"\d{4}-\d{2}-\d{2}\s*(日印象|周印象)", name):
            return False
        return True

    def _fallback_text(self, bucket_payload: dict, scope: str) -> str:
        sections = bucket_payload.get("key_sections", []) if isinstance(bucket_payload, dict) else []

        def first_section(*names: str) -> str:
            wanted = {name.lower() for name in names}
            for section in sections:
                if not isinstance(section, dict):
                    continue
                if str(section.get("heading") or "").strip().lower() in wanted:
                    text = str(section.get("text") or "").strip()
                    if text:
                        return text
            return ""

        if scope == "persona":
            text = first_section("reflection", "assistant_reflection", "moment", "fact")
        elif scope == "user":
            text = first_section("fact", "moment")
        else:
            text = first_section("moment", "fact", "reflection", "assistant_reflection")
        if not text:
            text = str(bucket_payload.get("source_excerpt") or bucket_payload.get("text") or "")
        return self._clean_fallback_text(text)

    def _fallback_activity_text(self, bucket_payload: dict) -> str:
        tags = {str(tag).lower() for tag in bucket_payload.get("tags", []) or []}
        domains = {str(item).lower() for item in bucket_payload.get("domain", []) or []}
        text = self._clean_fallback_text(
            str(bucket_payload.get("source_excerpt") or bucket_payload.get("text") or "")
        )
        if not text:
            return ""
        project_like = bool(
            tags & {"project_event", "work_event", "task_event"}
            or domains & {"记忆系统", "代码", "工作", "项目", "开发"}
        )
        activity_like = bool(
            re.search(
                r"(小雨|她|我).{0,12}(最近在|这几天在|这两天在|正在|继续|开始|准备|推进|调整|修改|修|部署|测试|写|搭|研究|排查|调试|做)",
                text,
            )
            or re.search(r"(最近在|这几天在|这两天在|正在).{0,16}(推进|调整|修改|修|部署|测试|写|搭|研究|排查|调试|做)", text)
        )
        if project_like and not activity_like:
            activity_like = bool(re.search(r"(推进|调整|修改|修|部署|测试|写|搭|研究|排查|调试|做|实现|开发|准备|加)", text))
        if not activity_like:
            return ""
        if re.search(r"(关系天气|撒娇|亲密|喜欢被|确认我在)", text) and not project_like:
            return ""
        return self._clip(text, 120)

    def _clean_fallback_text(self, text: str) -> str:
        text = strip_wikilinks(str(text or ""))
        text = re.sub(r"^Title:\s*.*?\bContent:\s*", "", text, flags=re.DOTALL)
        text = re.split(r"\s*###\s+affect_anchor\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.sub(r"###\s+[\w\u4e00-\u9fff_ -]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" ：:;；。")
        return self._clip(text, 140)

    def _portrait_snapshot(self, state: dict) -> dict:
        portrait = state.get("portrait", {}) if isinstance(state.get("portrait"), dict) else {}
        snapshot = {
            scope: {
                "recent_buffer": (portrait.get(scope, {}) or {}).get("recent_buffer", [])[:8],
                "staging_pool": (portrait.get(scope, {}) or {}).get("staging_pool", [])[:8],
                "mid_term": (portrait.get(scope, {}) or {}).get("mid_term", ""),
                "mid_term_evidence": (portrait.get(scope, {}) or {}).get("mid_term_evidence", [])[:8],
                "mid_term_source_dates": (portrait.get(scope, {}) or {}).get("mid_term_source_dates", [])[:8],
                "stable": (portrait.get(scope, {}) or {}).get("stable", ""),
                "stable_evidence": (portrait.get(scope, {}) or {}).get("stable_evidence", [])[:8],
                "stable_source_dates": (portrait.get(scope, {}) or {}).get("stable_source_dates", [])[:8],
                "stable_locked": bool((portrait.get(scope, {}) or {}).get("stable_locked", False)),
                "stable_revision": int((portrait.get(scope, {}) or {}).get("stable_revision", 0) or 0),
                "stable_source": str((portrait.get(scope, {}) or {}).get("stable_source", "") or ""),
            }
            for scope in PORTRAIT_SCOPES
        }
        snapshot["recent_activities"] = (
            state.get("recent_activities", []) if isinstance(state.get("recent_activities"), list) else []
        )[:8]
        return snapshot

    def _portrait_evidence_sets(self, portrait: Any, *, staging_only: bool = False) -> tuple[set[str], set[str]]:
        bucket_ids: set[str] = set()
        session_ids: set[str] = set()
        if not isinstance(portrait, dict):
            return bucket_ids, session_ids
        for scope in PORTRAIT_SCOPES:
            scope_state = portrait.get(scope, {}) if isinstance(portrait.get(scope), dict) else {}
            rows = []
            if staging_only:
                rows.extend(scope_state.get("staging_pool", []) or [])
            else:
                rows.extend(scope_state.get("recent_buffer", []) or [])
                rows.extend(scope_state.get("staging_pool", []) or [])
                rows.append({"evidence": scope_state.get("mid_term_evidence", []) or []})
                rows.append({"evidence": scope_state.get("stable_evidence", []) or []})
            for row in rows:
                if not isinstance(row, dict):
                    continue
                self._add_evidence_to_sets(row.get("evidence", []), bucket_ids, session_ids)
        if not staging_only:
            for row in portrait.get("recent_activities", []) or []:
                if isinstance(row, dict):
                    self._add_evidence_to_sets(row.get("evidence", []), bucket_ids, session_ids)
        return bucket_ids, session_ids

    def _add_evidence_to_sets(
        self,
        evidence: Any,
        bucket_ids: set[str],
        session_ids: set[str],
    ) -> None:
        rows = evidence if isinstance(evidence, list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            bucket_id = str(row.get("bucket_id") or "").strip()
            session_id = str(row.get("session_id") or "").strip()
            if bucket_id:
                bucket_ids.add(bucket_id)
            if session_id:
                session_ids.add(session_id)

    def _extract_key_sections(self, content: str) -> list[dict]:
        wanted = {"moment", "reflection", "assistant_reflection", "fact"}
        sections = []
        current_title = ""
        current_lines: list[str] = []

        def flush() -> None:
            nonlocal current_title, current_lines
            if current_title in wanted:
                text = "\n".join(current_lines).strip()
                if text:
                    sections.append(
                        {
                            "heading": current_title,
                            "text": self._clip(text, 360 if current_title == "moment" else 220),
                        }
                    )
            current_title = ""
            current_lines = []

        for line in str(content or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            match = re.match(r"^###\s+(.+?)\s*$", line.strip())
            if match:
                flush()
                current_title = match.group(1).strip().lower()
                current_lines = []
                continue
            if current_title:
                current_lines.append(line)
        flush()
        return sections[:5]

    def _format_key_sections(self, sections: list[dict]) -> str:
        parts = []
        for section in sections:
            if not isinstance(section, dict):
                continue
            heading = str(section.get("heading") or "").strip()
            text = str(section.get("text") or "").strip()
            if heading and text:
                parts.append(f"### {heading}\n{text}")
        return "\n\n".join(parts)

    def _prompt(self) -> str:
        return render_identity_template(PORTRAIT_PROMPT_TEMPLATE, self.identity)

    def _stable_prompt(self) -> str:
        return render_identity_template(STABLE_MAINTENANCE_PROMPT_TEMPLATE, self.identity)

    def _completion_options(
        self,
        *,
        max_tokens: int,
        temperature: float,
        json_response: bool = False,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
        if json_response:
            options["response_format"] = {"type": "json_object"}
        if self.thinking_mode:
            options["extra_body"] = {"thinking": {"type": self.thinking_mode}}
        return options

    def _parse_json_object(self, raw: str) -> dict:
        text = str(raw or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        if start >= 0:
            text = text[start:]
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError:
            logger.warning("Portrait JSON parse failed: %s", str(raw)[:200])
            raise ValueError("portrait_json_parse_failed")
        if not isinstance(parsed, dict):
            raise ValueError("portrait_json_not_object")
        return parsed

    def _normalize_evidence(
        self,
        value: Any,
        *,
        fallback_bucket_id: Any = "",
        fallback_moment_id: Any = "",
        fallback_session_id: Any = "",
    ) -> list[dict]:
        rows = []
        if isinstance(value, dict):
            value = [value]
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    rows.append({"bucket_id": item.strip()})
                elif isinstance(item, dict):
                    row = {
                        "bucket_id": str(item.get("bucket_id") or item.get("id") or "").strip(),
                        "moment_id": str(item.get("moment_id") or "").strip(),
                        "session_id": str(item.get("session_id") or "").strip(),
                    }
                    role = str(item.get("role") or "").strip().lower()
                    if role in {"self_anchor", "self_identity"}:
                        row["role"] = role
                    rows.append({k: v for k, v in row.items() if v})
        if not rows and (fallback_bucket_id or fallback_session_id):
            row = {
                "bucket_id": str(fallback_bucket_id or "").strip(),
                "moment_id": str(fallback_moment_id or "").strip(),
                "session_id": str(fallback_session_id or "").strip(),
            }
            rows.append({k: v for k, v in row.items() if v})
        return self._dedupe_evidence(rows)

    def _dedupe_evidence(self, rows: list[dict]) -> list[dict]:
        result = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            clean = {k: str(v).strip() for k, v in row.items() if str(v or "").strip()}
            if not clean:
                continue
            key = tuple(sorted(clean.items()))
            if key in seen:
                continue
            seen.add(key)
            result.append(clean)
        return result[:8]

    def _format_evidence(self, evidence: Any) -> str:
        rows = evidence if isinstance(evidence, list) else []
        labels = []
        for row in rows[:3]:
            if not isinstance(row, dict):
                continue
            if row.get("bucket_id"):
                label = f"bucket_id:{row['bucket_id']}"
                if row.get("moment_id"):
                    label += f"/moment_id:{row['moment_id']}"
                labels.append(label)
            elif row.get("session_id"):
                labels.append(f"session_id:{row['session_id']}")
        return ", ".join(labels)

    def _state_path(self, configured: Any) -> str:
        configured_text = str(configured or "").strip()
        if configured_text:
            return os.path.abspath(configured_text)
        state_dir = self.config.get("state_dir") or os.path.join(
            os.path.dirname(os.path.abspath(self.config.get("buckets_dir", "buckets"))),
            "state",
        )
        return os.path.join(state_dir, "portrait_state.json")

    def _empty_state(self) -> dict:
        return {
            "version": "portrait-state-v1",
            "updated_at": "",
            "last_run_date": "",
            "portrait": {
                scope: {
                    "recent_buffer": [],
                    "staging_pool": [],
                    "mid_term": "",
                    "mid_term_evidence": [],
                    "mid_term_source_dates": [],
                    "mid_term_source_date": "",
                    "mid_term_updated_at": "",
                    "stable": "",
                    "stable_evidence": [],
                    "stable_source_dates": [],
                    "stable_source_date": "",
                    "stable_updated_at": "",
                    "stable_locked": False,
                    "stable_revision": 0,
                    "stable_source": "",
                    "stable_history": [],
                    "stable_basis_keys": [],
                    "stable_evidence_invalid": False,
                }
                for scope in PORTRAIT_SCOPES
            },
            "daily_summaries": {},
            "handoff_recent_summaries": {},
            "recent_activities": [],
            "recent_timeline": [],
            "stable_candidates": [],
            "profile_fact_candidates": [],
            "skipped": [],
            "dismissed_items": [],
            "evidence_health": {},
            "runs": [],
        }

    def _merge_state(self, base: dict, data: dict) -> dict:
        for key, value in data.items():
            if key == "portrait" and isinstance(value, dict):
                for scope in PORTRAIT_SCOPES:
                    if isinstance(value.get(scope), dict):
                        base["portrait"][scope].update(value[scope])
            elif key in {"daily_summaries", "handoff_recent_summaries"} and isinstance(value, dict):
                base[key] = value
            elif key in {"recent_activities", "recent_timeline", "stable_candidates", "profile_fact_candidates", "skipped", "dismissed_items", "runs"} and isinstance(value, list):
                base[key] = value
            elif key == "evidence_health" and isinstance(value, dict):
                base[key] = value
            elif key in {"version", "updated_at", "last_run_date"}:
                base[key] = str(value or "")
        for scope in PORTRAIT_SCOPES:
            scope_state = base["portrait"][scope]
            scope_state["stable_locked"] = self._bool(scope_state.get("stable_locked"), False)
            try:
                scope_state["stable_revision"] = max(0, int(scope_state.get("stable_revision") or 0))
            except (TypeError, ValueError):
                scope_state["stable_revision"] = 0
            source = str(scope_state.get("stable_source") or "")
            scope_state["stable_source"] = source if source in {"model", "manual", "rollback"} else ""
            history = scope_state.get("stable_history")
            scope_state["stable_history"] = (
                [row for row in history if isinstance(row, dict)][-self.stable_history_max :]
                if isinstance(history, list)
                else []
            )
            scope_state["stable_basis_keys"] = self._normalize_basis_keys(
                scope_state.get("stable_basis_keys", [])
            )
            scope_state["stable_evidence_invalid"] = self._bool(
                scope_state.get("stable_evidence_invalid"),
                False,
            )
            if str(scope_state.get("stable") or "").strip() and not scope_state["stable_basis_keys"]:
                scope_state["stable_basis_keys"] = self._scope_basis_keys(scope_state)
        base["dismissed_items"] = self._active_dismissals(base.get("dismissed_items", []))
        return base

    def _drop_initial_daily_summaries(self, state: dict) -> None:
        daily = state.get("daily_summaries")
        runs = state.get("runs")
        if not isinstance(daily, dict) or not isinstance(runs, list):
            return
        initial_dates = {
            str(row.get("date") or "")
            for row in runs
            if isinstance(row, dict) and row.get("initial") and str(row.get("date") or "")
        }
        non_initial_dates = {
            str(row.get("date") or "")
            for row in runs
            if isinstance(row, dict) and not row.get("initial") and str(row.get("date") or "")
        }
        for date_key in initial_dates - non_initial_dates:
            daily.pop(date_key, None)

    def _day_window(self, now_local: datetime) -> tuple[datetime, datetime]:
        day = now_local.date()
        return (
            datetime.combine(day, time.min, tzinfo=self.tz),
            datetime.combine(day, time.max, tzinfo=self.tz),
        )

    def _local_now(self, value: datetime | None = None) -> datetime:
        if value is None:
            return datetime.now(self.tz)
        if value.tzinfo is None:
            value = value.replace(tzinfo=self.tz)
        return value.astimezone(self.tz)

    def _parse_iso(self, value: Any) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(self.tz)

    def _now_utc(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _safe_key(self, value: Any) -> str:
        text = re.sub(r"[^a-zA-Z0-9_\-\u4e00-\u9fff]+", "_", str(value or "").strip())
        return text[:80].strip("_") or "other"

    def _norm(self, value: str) -> str:
        return re.sub(r"\s+", "", str(value or "").lower())

    def _clip(self, value: Any, max_chars: int) -> str:
        text = " ".join(strip_wikilinks(str(value or "")).split())
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    def _clamp(self, value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        return max(0.0, min(1.0, number))

    def _bool(self, value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}
