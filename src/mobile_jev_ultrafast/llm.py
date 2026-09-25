"""Auxiliary LLM client — pre-task planning, deadlock escape, result summary.

Ports the three responsibilities of ``LlmApiClient`` in
``mobile-jev-jarvis/api/LlmApiClient.kt`` to Python:

* :func:`plan_task` — given the goal and the list of installed app labels,
  returns a short plan text plus the labels of the apps relevant to the
  goal. The labels are looked up locally against ``app_label_to_pkg`` so
  the LLM never needs to know package names (saves tokens, and stops the
  LLM from inventing package names).
* :func:`decide_next_action` — given the current screen, the recent
  actions, and a string describing the deadlock, returns one short
  suggestion like "执行 SWIPE_LEFT 切换桌面" or "执行 PRESS_BACK".
* :func:`summarize_result` — given the goal, the final page snapshot,
  and the recent log buffer, returns a 1–2 sentence Chinese summary
  that the dashboard surfaces as a "summary card".

All three helpers reuse the same OpenAI-compatible endpoint already
configured for ``TEXT_MODEL_*`` so there is no new key to provision.
Each helper has a ``scripted`` deterministic fallback that returns a
sensible answer without a network round-trip (useful for tests and
for the ``--fake`` demo path).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Iterable

from . import verbose
from .model import _goal_keywords, post_json

log = logging.getLogger(__name__)

# Reuse the same prompt-trimming / CJK-aware keyword helper the decision
# backend uses. Importing it here keeps one place to evolve the parsing.

_SCRIPTED_PLAN = (
    "1. 分析用户目标，识别可能涉及的 APP。\n"
    "2. 通过 LAUNCH_APP 启动相关 APP，或在桌面搜索。\n"
    "3. 按目标顺序执行操作。\n"
    "4. 校验结果并结束。\n"
    "DONE_TRIGGER=操作型:执行最后一步操作后即可视为完成，无需验证可见结果"
)

_PLAN_SYSTEM = (
    "你是一名 Android 手机操控专家。请阅读用户目标，结合设备上已安装的 APP 列表，"
    "给出操作计划并指出本次任务涉及的 APP。\n\n"
    "输出要求：\n"
    "- 中文。\n"
    "- 第一行起输出操作计划，每条一行。\n"
    "- 计划不超过 10 条，按时间顺序。\n"
    "- 包含判断逻辑（如「找不到 X 则尝试 Y」）。\n"
    "- 不要描述图标外观（AI 不能识图，只能读文字）。\n"
    "- 严格遵守以下桌面导航规则：\n"
    "  * **【重要】优先使用 LAUNCH_APP 直接启动已安装的目标 APP**。LAUNCH_APP 对所有\n"
    "    已安装 APP 都可用（包名见用户消息中的「已安装 APP」列表），不要让 Jev 在桌面\n"
    "    翻页查找。计划中必须写明「LAUNCH_APP 启动 <APP 名>（<包名>）」或同等表述。\n"
    "  * **【重要】每次 LAUNCH_APP 之后必须立刻插入一条 WAIT_LONGER（等 5 秒）**。\n"
    "    绝大多数 APP 冷启动后会显示开屏广告 / 启动页 / 权限弹窗，2–3 秒仍可能不够，\n"
    "    慢网络下广告视频要 4–5 秒；不等待 Jev 会在广告层上误点，或把首次亮屏当作\n"
    "    “已进入首页”去验证 DONE_TRIGGER，导致后续步骤全部错位。计划中写为\n"
    "    「WAIT_LONGER 等广告结束（~5 秒）」或「WAIT_LONGER 等启动页」。\n"
    "  * 只有当目标 APP 不在已安装列表中（无法 LAUNCH_APP）时，才让 Jev 在桌面\n"
    "    SWIPE_LEFT / SWIPE_RIGHT 翻页查找。\n"
    "  * 桌面可能有多页，但用 SWIPE 翻页是次优路径 —— 能 LAUNCH_APP 就不要 SWIPE。\n"
    "  * 系统设置 APP 的 label 是「设置」(齿轮图标)，不要误点运营商/工具类 APP。\n"
    "  * 中国联通/中国移动/中国电信 不是系统设置。\n"
    "  * 文件夹打开后显示重命名文本框时，先 PRESS_BACK 关闭。\n"
    "  * 点错 APP 后立即 PRESS_BACK 返回。\n"
    "- 可选动作补充：Jev 提供的等待动作有 WAIT（0.1 秒刷新）、WAIT_LONG（2 秒动画）\n"
    "  和 WAIT_LONGER（5 秒启动广告）。默认广告后启动的场景一律用 WAIT_LONGER；\n"
    "  只有 LAUNCH_APP 之外的动画过渡（例如弹出菜单展开）才用 WAIT_LONG。\n"
    "- 倒数第二行格式必须为：DONE_TRIGGER=<触发条件>\n"
    "  DONE_TRIGGER 用于告诉 Jev 何时选 DONE。三种取值：\n"
    "    操作型:<最后一步动作描述> — 用户目标是执行 N 次某动作（如「上滑 1 下」、"
    "「点击按钮」、「向左滑 3 次」），执行该动作后无论界面是否变化都视为完成。\n"
    "    到达型:<屏幕描述或关键词> — 用户目标是到达某屏幕/状态（如「打开设置」、"
    "「进入我的页面」），看到对应界面后才选 DONE。\n"
    "    触发词型:<字面关键词> — 等待某关键词出现在屏幕文本里即可结束。\n"
    "  选错类型会导致提前结束或死循环：判断不准时优先 到达型（保守）。\n"
    "- 最后一行格式必须为：RELEVANT_APPS=名称1,名称2,名称3（逗号分隔、可空）"
)

_DECIDE_SYSTEM = (
    "你是 Android 操控专家。AI 卡死了，请给出一个跳出循环的操作建议。\n\n"
    "要求：\n"
    "- 直接输出建议，不超过 150 字，不解释。\n"
    "- 只使用以下动作：PRESS_BACK、PRESS_HOME、PRESS_RECENTS、SWIPE_LEFT、"
    "SWIPE_RIGHT、SCROLL_UP、SCROLL_DOWN、LAUNCH_APP、WAIT_LONGER。\n"
    "- 优先 PRESS_BACK / PRESS_HOME 回到桌面。\n"
    "- 如果是刚 LAUNCH_APP 但页面还没加载完（仍处于广告 / 启动页），"
    "返回「执行 WAIT_LONGER」等 5 秒，不要重试点击。\n"
    "- 如需 LAUNCH_APP，写明包名（如 com.android.settings）。\n"
    "- 不要重复最近已经尝试过的操作。\n\n"
    "示例：\n"
    "执行 PRESS_BACK 返回上一级\n"
    "执行 SWIPE_LEFT 切换桌面\n"
    "执行 WAIT_LONGER 等启动广告结束\n"
    "执行 LAUNCH_APP com.android.settings"
)

_SUMMARY_SUCCESS_SYSTEM = (
    "任务已完成。请根据任务目标和当前屏幕内容，用 1-2 句中文总结实际执行结果。"
    "只描述结果本身，不要描述步骤，不要加客套话。"
)

_SUMMARY_FAILURE_SYSTEM = (
    "任务执行失败或被阻塞。请根据以下执行日志分析失败的真实原因，"
    "然后用 1-2 句话给出针对性的建议。建议要针对日志中的实际问题，不要泛泛而谈。"
)


def _extract_relevant_apps(text: str) -> tuple[str, list[str], str]:
    """Split the LLM reply into ``(plan_text, relevant_app_labels, done_trigger)``.

    The LLM is told to put two marker lines at the end of the reply:

    * ``DONE_TRIGGER=<...>`` — the rule Jev uses to know when the user's
      goal is satisfied (operational / destination / keyword-based). Kept
      inside ``plan_text`` so Jev reads it on every tick; also returned
      separately for verbose logging.
    * ``RELEVANT_APPS=<labels>`` — comma-separated list of app labels.

    Tolerates leading/trailing whitespace, case-insensitive prefixes, and
    missing trailing newlines.
    """
    raw = (text or "").strip()
    plan_lines: list[str] = []
    relevant: list[str] = []
    done_trigger = ""
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"(?i)^relevant_apps\s*[=:]\s*(.+)$", stripped)
        if m:
            labels = [s.strip() for s in re.split(r"[,，、]", m.group(1)) if s.strip()]
            relevant = labels
            continue
        m = re.match(r"(?i)^done_trigger\s*[=:]\s*(.+)$", stripped)
        if m:
            done_trigger = stripped  # keep the original line for plan_text
            plan_lines.append(stripped)
            continue
        plan_lines.append(stripped)
    plan_text = "\n".join(plan_lines).strip() or _SCRIPTED_PLAN
    return plan_text, relevant, done_trigger


def _call_llm(messages: list[dict], *, max_tokens: int = 1024) -> str:
    """Hit the OpenAI-compatible text-model endpoint and return the content.

    Mirrors :func:`mobile_jev_ultrafast.model._field_text_helper` so the
    auxiliary LLM reuses the same credentials, base URL, and reasoning
    flags as ``field_text``. We deliberately do *not* require JSON
    output here — the plan/decide/summarize helpers parse free-form
    replies, with deterministic fallbacks if the reply is empty or the
    call fails.
    """
    key = os.environ.get("TEXT_MODEL_API_KEY")
    if not key:
        raise RuntimeError(
            "Auxiliary LLM needs TEXT_MODEL_API_KEY; "
            "either set it in .env or disable the feature (e.g. --no-plan)."
        )
    base = os.environ.get("TEXT_MODEL_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    model = os.environ.get("TEXT_MODEL", "deepseek-chat")
    reasoning = {"thinking": {"type": "disabled"}} if "api.deepseek.com/" in base else {"reasoning": {"effort": "low"}}
    if os.environ.get("TEXT_MODEL_REASONING") == "none":
        reasoning = {"reasoning": {"enabled": False}}
    body = {
        "model": model,
        "max_tokens": max_tokens,
        **reasoning,
        "messages": messages,
    }
    verbose.emit(2, "── auxiliary LLM request ────────────────────────────────")
    verbose.emit(2, verbose.blob(body))
    response = post_json(base + "/chat/completions", key, body)
    try:
        content = response["choices"][0]["message"]["content"] or ""
    except (KeyError, TypeError, IndexError):
        return ""
    # Strip ``<think>…</think>`` blocks that some MiniMax / reasoning
    # models prefix the visible reply with; otherwise the plan / decide /
    # summarize helpers see a noisy prefix and regex parsers miss their
    # markers. Mirrors :func:`mobile_jev_ultrafast.model._strip_thinking_tags`.
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    verbose.emit(2, "── auxiliary LLM response ───────────────────────────────")
    verbose.emit(2, content or "(empty)")
    return content


# ---------------------------------------------------------------------------
# Pre-task planning
# ---------------------------------------------------------------------------

# Apps whose label is so generic ("设置") that the LLM might pick the wrong
# one. These are always suggested when present on the device; the LLM just
# confirms. Keep this short — the LLM still does the heavy lifting.
_KNOWN_RELEVANT_LABELS = {
    "设置", "Settings", "Settings ", "System Settings",
    "电话", "Phone", "联系人", "Contacts",
    "短信", "Messages", "相机", "Camera",
    "时钟", "Clock", "日历", "Calendar",
    "浏览器", "Browser",
}


def _list_installed_labels(device) -> list[str]:
    """Return labels of apps installed on the device.

    Prefers a real ``list_apps`` round-trip; falls back to an empty list
    when the device layer doesn't expose the API (FakeAutoX in
    particular) so the planner can still run with a generic plan.
    """
    list_apps = getattr(device, "list_installed_apps", None) or getattr(
        getattr(device, "mcp", None), "installed_apps", None
    )
    if list_apps is None:
        return []
    try:
        rows = list_apps() or []
    except Exception as exc:  # noqa: BLE001 — planning must never break the loop
        log.debug("installed_apps probe failed: %s", exc)
        return []
    out: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = (row.get("label") or row.get("name") or "").strip()
        if label:
            out.append(label)
    return out


def _scripted_plan(goal: str, installed: list[str], relevant_apps: list[str]) -> dict:
    """Deterministic offline plan — used when no LLM key is configured.

    Heuristics over the goal:
    * If a label from ``_KNOWN_RELEVANT_LABELS`` appears as a substring
      of the goal, append it to ``relevant_apps``.
    * Otherwise emit an empty plan and rely on the agent to find the app.
    * Pick a ``DONE_TRIGGER`` line based on the goal's shape:
        - operational verbs (上滑/下滑/点击/双击 + a count) → 操作型
        - destination verbs (打开/进入/跳转/查看/看看/找到) → 到达型
        - default → 到达型 (conservative; avoids premature DONE).
    """
    plan_lines = [
        "1. 观察当前屏幕状态。",
        "2. 如不在桌面，按需 LAUNCH_APP 或先 PRESS_HOME。",
        "3. 每次 LAUNCH_APP 后立刻执行 WAIT_LONGER 等 5 秒，跳过启动广告 / 启动页。",
        "4. 按目标顺序执行操作。",
        "5. 校验结果并结束。",
    ]
    keywords = [kw for kw in _goal_keywords(goal) if kw]
    matches: list[str] = []
    seen: set[str] = set()
    for kw in keywords:
        for label in installed:
            if label in seen:
                continue
            if kw and (kw in label or label in kw):
                matches.append(label)
                seen.add(label)
    done_trigger = _scripted_done_trigger(goal)
    if done_trigger:
        plan_lines.append(done_trigger)
    plan_text = "\n".join(plan_lines)
    return {
        "plan_text": plan_text,
        "relevant_apps": matches[:5],
        "done_trigger": done_trigger,
        "model": "scripted",
    }


_OPERATIONAL_VERBS = (
    "上滑", "向上滑", "向上滚动",
    "下滑", "向下滑", "向下滚动",
    "左滑", "向左滑", "右滑", "向右滑",
    "滑动", "滚动", "翻页",
    "点击", "点一下", "按一下", "点这个", "点那", "点一下",
    "双击", "长按",
    "返回", "回桌面", "退回",
)
_DESTINATION_VERBS = (
    "打开", "进入", "跳转", "切到", "切换",
    "查看", "看看", "检查", "确认", "核实",
    "找到", "寻找", "搜索",
    "打开后", "进入后",
)


def _scripted_done_trigger(goal: str) -> str:
    """Return a ``DONE_TRIGGER=...`` line for the offline planner.

    Used only when the auxiliary LLM is unavailable. Mirrors the
    operational/destination distinction in :data:`questions.NEXT_ACTION`
    so even ``AGENT_DECISION_BACKEND=scripted`` runs can stop on simple
    operational goals like "上滑 1 下" without looping forever.
    """
    if not goal:
        return "DONE_TRIGGER=到达型:出现与目标一致的屏幕或文本"
    lowered = goal.lower()
    has_count = bool(
        re.search(r"\d+\s*(?:下|次|个|遍|回|次|下)", goal)
        or re.search(r"(?:下|次|个|遍|回)\s*\d+", goal)
        or re.search(r"\d+\s*(?:times?|steps?|clicks?|taps?)", lowered)
        or re.search(r"\b(?:once|twice|three times)\b", lowered)
    )
    is_operational = any(verb in goal for verb in _OPERATIONAL_VERBS) and has_count
    if is_operational:
        return f"DONE_TRIGGER=操作型:执行目标中的动作 {goal} 后立即结束，无需验证屏幕"
    is_destination = any(verb in goal for verb in _DESTINATION_VERBS)
    if is_destination:
        return "DONE_TRIGGER=到达型:看到目标屏幕或目标元素后再选 DONE"
    return "DONE_TRIGGER=到达型:出现与目标一致的屏幕或文本"


def plan_task(goal: str, device, *, enabled: bool = True) -> dict:
    """Plan the task and identify relevant apps.

    Returns a dict with keys:
      * ``plan_text`` — free-form plan, injected into Jev ``state.task_plan``.
      * ``relevant_apps`` — list of labels (not packages) the planner thinks
        are involved.
      * ``done_trigger`` — the ``DONE_TRIGGER=...`` line itself, kept here
        for verbose logging / inspection; the line is also embedded in
        ``plan_text`` so Jev reads it on every tick.
      * ``model`` — ``"scripted"`` for the offline fallback, otherwise the
        LLM model name.

    If ``enabled`` is False the helper short-circuits to the scripted plan
    (useful for tests and for the ``--no-plan`` CLI flag).
    """
    installed = _list_installed_labels(device)
    if verbose.enabled(1):
        verbose.emit(1, f"  PLAN: goal={goal!r}")
        verbose.emit(1, f"  PLAN: installed apps={installed or '(none detected)'}")
    if not enabled:
        return _scripted_plan(goal, installed, [])
    user_content = (
        f"设备语言：{os.environ.get('AUTOX_DEVICE_LANG', 'zh-CN')}\n"
        f"已安装 APP：{('、'.join(installed)) or '(无法读取列表)'}\n\n"
        f"任务：{goal}"
    )
    try:
        reply = _call_llm(
            [
                {"role": "system", "content": _PLAN_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1200,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("plan_task LLM call failed; falling back to scripted plan: %s", exc)
        return _scripted_plan(goal, installed, [])
    plan_text, relevant, done_trigger = _extract_relevant_apps(reply)
    if verbose.enabled(1):
        verbose.emit(1, f"  PLAN: relevant_apps={relevant}")
        verbose.emit(1, verbose.wrap("  PLAN: ", plan_text))
    return {
        "plan_text": plan_text[:5000],
        "relevant_apps": relevant[:8],
        "done_trigger": done_trigger,
        "model": os.environ.get("TEXT_MODEL", "deepseek-chat"),
    }


# ---------------------------------------------------------------------------
# Deadlock escape
# ---------------------------------------------------------------------------

_KNOWN_PACKAGES = {
    "settings": "com.android.settings",
    "phone": "com.android.dialer",
    "contacts": "com.android.contacts",
    "messages": "com.android.messaging",
    "camera": "com.android.camera",
    "clock": "com.android.deskclock",
    "browser": "com.android.browser",
    "files": "com.android.documentsui",
    "wechat": "com.tencent.mm",
    "taobao": "com.taobao.taobao",
    "alipay": "com.eg.android.AlipayGphone",
}


def _parse_decide_suggestion(text: str) -> dict:
    """Convert a free-form LLM suggestion into a structured action.

    Accepts the canonical verbs (PRESS_BACK / PRESS_HOME / SWIPE_LEFT /
    SWIPE_RIGHT / SCROLL_UP / SCROLL_DOWN / LAUNCH_APP) plus a small
    alias map (返回 → PRESS_BACK, 桌面 → PRESS_HOME, 左滑 → SWIPE_LEFT, …).
    Unrecognised suggestions come back as ``{"kind": "passthrough",
    "text": ...}`` so the caller can inject them into recent_actions
    verbatim — the agent will see them as LLM_SUGGESTION and obey.
    """
    raw = (text or "").strip()
    if not raw:
        return {"kind": "passthrough", "text": raw}
    # NB: do NOT include ``.`` (period) here — package names like
    # ``com.taobao.idlefish`` carry dots that the LAUNCH_APP regex needs
    # to capture whole. The previous code stripped dots and the regex
    # ended up matching only ``com`` instead of the full package.
    cleaned = re.sub(r"[\s。!！,，;；]+", " ", raw).strip()
    lowered = cleaned.lower()

    verbs = (
        ("press_back", ("press_back", "back", "返回", "back键", "back 键")),
        ("press_home", ("press_home", "home", "桌面", "home键", "home 键")),
        ("press_recents", ("press_recents", "recents", "多任务", "最近任务")),
        ("swipe_left", ("swipe_left", "左滑", "向左滑")),
        ("swipe_right", ("swipe_right", "右滑", "向右滑")),
        ("scroll_up", ("scroll_up", "上滑", "向上滑")),
        ("scroll_down", ("scroll_down", "下滑", "向下滑")),
        # Wait tiers — ``wait_longer`` (5s) is what the planner now
        # recommends after LAUNCH_APP to ride out splash + ad; we
        # recognise both Latin and Chinese aliases here so the
        # deadlock-decide LLM can say "等广告结束" or
        # "WAIT_LONGER" interchangeably.
        ("wait_longer", ("wait_longer", "等 5 秒", "等5秒", "等广告", "等启动", "等启动页")),
    )
    for kind, aliases in verbs:
        for alias in aliases:
            if alias and alias in lowered:
                return {"kind": kind, "text": raw}
    m = re.search(r"launch_app\s+([a-zA-Z0-9_.]+)", cleaned, flags=re.IGNORECASE)
    if m:
        return {"kind": "launch_app", "package": m.group(1), "text": raw}
    # As a last resort, scan for known app keywords the LLM might mention.
    # Use word boundaries so the substring "taobao" doesn't false-hit when
    # the LLM actually named 闲鱼 (``com.taobao.idlefish``); otherwise the
    # agent launches the wrong app.
    for label, pkg in _KNOWN_PACKAGES.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])", lowered):
            return {"kind": "launch_app", "package": pkg, "text": raw}
    return {"kind": "passthrough", "text": raw}


def _scripted_decide(recent_actions: list[dict], page: dict) -> dict:
    """Deterministic offline deadlock breaker.

    Priority order:
    0. If the very last action was a ``launch`` and we haven't yet
       waited for the splash / ad to finish, suggest ``WAIT_LONGER``
       (3 s) — otherwise we'd repeat the click on the ad creative and
       loop forever. Cold-launch screens are the most common deadlock
       after :func:`plan_task` was disabled.
    1. If we are stuck inside an unrelated package (e.g. AutoX editor) and
       the goal likely wants us on the launcher, suggest ``PRESS_HOME``.
    2. If recent actions are all CLICK on the same element, suggest
       ``PRESS_BACK`` (we're probably in a child screen).
    3. Otherwise suggest ``SCROLL_DOWN`` so the agent looks for more
       content.
    """
    keys = [(h.get("kind") or "") + ":" + (h.get("target") or "") for h in recent_actions[-6:]]
    if recent_actions and recent_actions[-1].get("kind") == "launch":
        return {"kind": "wait_longer", "text": "执行 WAIT_LONGER 等启动广告结束", "model": "scripted"}
    if len(keys) >= 3 and all(k == keys[0] and not k.startswith("scroll") for k in keys[-3:]):
        return {"kind": "press_back", "text": "执行 PRESS_BACK 返回", "model": "scripted"}
    return {"kind": "scroll_down", "text": "执行 SCROLL_DOWN 查看更多内容", "model": "scripted"}


def decide_next_action(
    *,
    goal: str,
    page: dict,
    recent_actions: list[dict],
    deadlock_note: str,
    enabled: bool = True,
) -> dict:
    """Ask the auxiliary LLM (or its scripted fallback) how to escape.

    The return value is a structured suggestion the agent loop turns
    into either an immediate action (``PRESS_BACK`` etc.) or an
    ``LLM_SUGGESTION`` recent_actions entry the next Jev tick will obey.
    """
    if not enabled:
        return _scripted_decide(recent_actions, page)
    action_lines = [
        f"{i + 1}. {a.get('operation', '?')} → {a.get('action', a.get('target', '?'))}"
        for i, a in enumerate(recent_actions[-8:])
    ]
    recent_text = "\n".join(action_lines) or "(无)"
    page_text = (page.get("text") or "")[:300]
    elements = page.get("actions") or []
    elem_lines = []
    for a in elements[:12]:
        if a.get("kind") not in {"click", "fill"}:
            continue
        role = a.get("role") or "?"
        label = (a.get("label") or "")[:24]
        elem_lines.append(f"[{a.get('id', '?')}] {role} {label}")
    elements_text = "\n".join(elem_lines) or "(无可点击元素)"
    user_content = (
        f"目标：{goal}\n"
        f"当前包名：{page.get('package', '?')}\n"
        f"页面文本（前 300 字）：{page_text}\n"
        f"可见元素：\n{elements_text}\n\n"
        f"近期操作（最近 8 条）：\n{recent_text}\n\n"
        f"卡死原因：{deadlock_note}\n"
    )
    try:
        reply = _call_llm(
            [
                {"role": "system", "content": _DECIDE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            max_tokens=3000,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("decide_next_action LLM call failed; using scripted fallback: %s", exc)
        return _scripted_decide(recent_actions, page)
    parsed = _parse_decide_suggestion(reply)
    parsed["model"] = os.environ.get("TEXT_MODEL", "deepseek-chat")
    parsed["raw"] = reply
    if verbose.enabled(1):
        verbose.emit(1, f"  FALLBACK: deadlock={deadlock_note}")
        verbose.emit(1, f"  FALLBACK: parsed suggestion={parsed.get('kind')} {parsed.get('package', '')}")
        verbose.emit(1, f"  FALLBACK: raw reply={reply!r}")
    return parsed


# ---------------------------------------------------------------------------
# Result summary
# ---------------------------------------------------------------------------

def _trim_log(logs: Iterable[str], *, limit: int = 1200) -> str:
    """Trim a log buffer to ``limit`` characters while keeping the tail."""
    buf = "\n".join(line for line in logs if line)
    if len(buf) <= limit:
        return buf
    return buf[-limit:]


def summarize_result(
    *,
    goal: str,
    page: dict,
    logs: list[str],
    success: bool,
    enabled: bool = True,
) -> dict:
    """Generate a 1–2 sentence Chinese summary of the run.

    ``logs`` carries the recent ``[PLAN]/[ACT]/[DEC]/[ERR]`` lines the
    agent loop accumulated; we feed the tail into the prompt so the
    summary can quote real actions.
    """
    page_text = (page.get("text") or "")[:500]
    elements = page.get("actions") or []
    elem_lines = []
    for a in elements[:20]:
        role = a.get("role") or "?"
        label = (a.get("label") or "")[:20]
        elem_lines.append(f"[{role}]{label}")
    elements_text = "、".join(elem_lines) or "(无)"
    if success:
        system = _SUMMARY_SUCCESS_SYSTEM
        user = (
            f"任务目标：{goal}\n"
            f"当前屏幕文本：{page_text}\n"
            f"当前屏幕元素：{elements_text}\n\n"
            f"结果总结："
        )
    else:
        system = _SUMMARY_FAILURE_SYSTEM
        user = (
            f"任务目标：{goal}\n"
            f"执行日志（最近 {min(len(logs), 40)} 条）：\n"
            f"{_trim_log(logs)}\n\n"
            f"分析失败原因并给出建议："
        )
    if not enabled:
        return {"summary": "（未启用 LLM 总结）", "model": "scripted"}
    try:
        reply = _call_llm(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=400,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("summarize_result LLM call failed: %s", exc)
        return {"summary": "（总结失败）", "model": "scripted"}
    if verbose.enabled(1):
        verbose.emit(1, f"  SUMMARY: {reply}")
    return {"summary": reply[:600], "model": os.environ.get("TEXT_MODEL", "deepseek-chat")}