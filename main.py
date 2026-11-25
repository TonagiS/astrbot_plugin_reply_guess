"""没有 @、也没有引用回复时，判断对方是不是在接着跟 bot 说话。

三信号叠加（缺一个都不接）：
1. 对话窗口：bot 刚在群里说过话，窗口内最多跟 max_follow 条，且中间没人插话
2. 规则粗筛：长度、没有 @ 别人、不是指令、不是纯图、可选必须带"你/这/那/吗/呢"这类指代
3. 模型复判：把 bot 上一句 + 这条候选交给小模型，只让它回答 YES/NO

判定通过也不一定接：还要过 probability（默认 0.85）和每群每小时配额，
默认偏向"宁可沉默"，因为插错话比少说一句贵得多。
"""

from __future__ import annotations

import random
import time

from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Image, Plain
from astrbot.api.platform import MessageType
from astrbot.api.star import Context, Star

PLUGIN_DIR = Path(__file__).resolve().parent
LOG_PATH = PLUGIN_DIR / "data" / "reply_guess.log"


def _dbg(msg: str) -> None:
    """环形调试日志：接没接、为什么，都能回看。"""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > 1024 * 1024:
            LOG_PATH.write_text("", encoding="utf-8")
        stamp = time.strftime("%m-%d %H:%M:%S")
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] {msg}\n")
    except Exception:  # noqa: BLE001
        pass

JUDGE_PROMPT = (
    "下面是一段 QQ 群聊。A 是机器人刚说的一句话，B 是紧接着别人的发言"
    "（既没有 @ 机器人，也没有引用回复）。\n"
    "判断机器人现在**该不该接这句话**。只有三条同时成立才答 REPLY：\n"
    "1) B 是在对机器人说，或者顺着 A 的话题往下走（不是对别人说话、不是换话题）；\n"
    "2) 接一句能**补充新东西**（回答问题、给判断、接梗、表态都算；单纯附和、重复、"
    "只是「嗯」「哈哈」这种不算）；\n"
    "3) 不会打断别人之间的对话，也不像是机器人刚说过的话被重复一遍。\n"
    "只要有一条不成立就答 SILENT。\n"
    "第一行只输出 REPLY 或 SILENT，第二行用不超过 10 个字说明理由。\n\n"
    "A: {bot}\nB: {user}\n"
)


class ReplyGuess(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.config = config or {}
        self._last_bot: dict[str, tuple[float, str]] = {}
        self._follow: dict[str, int] = {}
        self._quota: dict[str, tuple[int, int]] = {}
        self._last_judge: dict[str, float] = {}

    # ------------------------------------------------------------------ config

    def _cfg(self, key: str, default=None):
        try:
            value = self.config.get(key, default)
        except Exception:  # noqa: BLE001
            value = default
        return default if value is None else value

    def _list(self, key: str) -> list[str]:
        raw = self._cfg(key, [])
        if isinstance(raw, str):
            raw = [x.strip() for x in raw.replace("，", ",").split(",")]
        return [str(x).strip() for x in (raw or []) if str(x).strip()]

    def _in_scope(self, event: AstrMessageEvent) -> bool:
        groups = self._list("groups")
        if not groups:
            return True
        gid = str(event.get_group_id() or "")
        return gid in groups or event.unified_msg_origin in groups

    # -------------------------------------------------------- 记录 bot 发言

    @filter.after_message_sent()
    async def after_sent(self, event: AstrMessageEvent):
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return
        if not self._cfg("enable", True):
            return
        umo = event.unified_msg_origin
        if not self._in_scope(event):
            return
        text = ""
        try:
            result = event.get_result()
            if result is not None:
                text = str(result.get_plain_text() or "").strip()
            if not text:
                chain = getattr(result, "chain", None) or []
                text = "".join(getattr(c, "text", "") for c in chain).strip()
        except Exception:  # noqa: BLE001
            text = ""
        if not text:
            _dbg(f"after_sent: 拿不到已发文本，窗口不开 ({umo})")
            return
        self._last_bot[umo] = (time.time(), text[:300])
        self._follow[umo] = 0
        _dbg(f"after_sent: 开窗口 ({umo}) bot={text[:40]}")

    # ---------------------------------------------------------------- 判定

    async def _judge(self, bot_text: str, user_text: str) -> bool:
        provider_id = str(self._cfg("judge_provider_id", "deepseek/deepseek-flash") or "")
        try:
            if provider_id:
                provider = self.context.get_provider_by_id(provider_id)
            else:
                provider = await self.context.get_using_provider_async()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"reply_guess: 取判定模型失败: {exc}")
            return False
        if not provider:
            return False
        prompt = JUDGE_PROMPT.format(bot=bot_text[:300], user=user_text[:300])
        try:
            resp = await provider.text_chat(
                prompt=prompt,
                session_id=f"reply-guess-{int(time.time())}",
                persist=False,
            )
            answer = str(getattr(resp, "completion_text", "") or "").strip()
        except Exception as exc:  # noqa: BLE001
            _dbg(f"判定失败: {exc}")
            return False
        first = answer.splitlines()[0].strip().upper() if answer else ""
        _dbg(f"判定结果: {answer[:40]!r}")
        return first.startswith("REPLY")

    def _budget(self, umo: str) -> bool:
        now = time.time()
        gap = float(self._cfg("judge_cooldown_seconds", 20))
        if now - self._last_judge.get(umo, 0.0) < gap:
            return False
        hour = int(now // 3600)
        mark, count = self._quota.get(umo, (hour, 0))
        if mark != hour:
            mark, count = hour, 0
        if count >= int(self._cfg("judge_max_per_hour", 60)):
            return False
        self._last_judge[umo] = now
        self._quota[umo] = (mark, count + 1)
        return True

    # ---------------------------------------------------------------- 主处理

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        if not self._cfg("enable", True):
            return
        if event.is_at_or_wake_command:
            return  # 正常 @/唤醒，交给原有链路
        if not self._in_scope(event):
            return
        umo = event.unified_msg_origin

        # 自己说的话不算
        try:
            if str(event.get_sender_id()) == str(event.get_self_id()):
                return
        except Exception:  # noqa: BLE001
            pass

        text = (event.message_str or "").strip()
        if not text or text.startswith("/") or text.startswith("#"):
            return
        if len(text) > int(self._cfg("max_chars", 80)):
            return
        if len(text) < int(self._cfg("min_chars", 1)):
            return

        messages = event.get_messages()
        if any(isinstance(c, At) for c in messages if getattr(c, "qq", None) != event.get_self_id()):
            return  # 在 @ 别人，不是在跟 bot 说
        if self._cfg("skip_images", True) and any(isinstance(c, Image) for c in messages):
            return

        signals = self._list("signal_words")
        if signals and self._cfg("require_signal", False):
            if not any(s and s in text for s in signals):
                return

        # 1) 对话窗口
        last = self._last_bot.get(umo)
        if not last:
            _dbg(f"跳过：这个群还没有开过窗口 ({umo}): {text[:30]}")
            return
        ts, bot_text = last
        if time.time() - ts > float(self._cfg("window_seconds", 120)):
            self._last_bot.pop(umo, None)
            self._follow.pop(umo, None)
            _dbg(f"跳过：窗口过期 {int(time.time() - ts)}s ({umo})")
            return
        if self._follow.get(umo, 0) >= int(self._cfg("max_follow", 2)):
            self._last_bot.pop(umo, None)
            _dbg(f"跳过：窗口内已接满 max_follow ({umo})")
            return

        # 2/3) 预算 + 模型复判
        if not self._budget(umo):
            _dbg(f"跳过：判定预算/冷却 ({umo})")
            return
        verdict = await self._judge(bot_text, text)
        if not verdict:
            _dbg(f"判定 NO ({umo}) bot={bot_text[:30]} | user={text[:30]}")
            return
        if random.random() > float(self._cfg("probability", 0.85)):
            _dbg(f"判定 YES 但概率没过 ({umo})")
            return

        _dbg(f"判定 YES，接话 ({umo}) bot={bot_text[:30]} | user={text[:30]}")
        self._follow[umo] = self._follow.get(umo, 0) + 1
        event.is_wake = True
        event.is_at_or_wake_command = True
