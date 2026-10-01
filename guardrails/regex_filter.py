"""Regex keyword filter: the "rules" baseline.

Patterns were written by reading `data/val.csv` only (never the test split).
They cover common, well-known attack phrasing rather than individual rows.
Matching is case-insensitive on an NFKC-normalised copy of the prompt, so
full-width and mathematical-alphabet letters fold to plain ASCII. There is no
leetspeak or homoglyph de-obfuscation: that is a known blind spot and part of
what this baseline is meant to show.
"""

from __future__ import annotations

import re
import time
import unicodedata

from guardrails.base import Verdict

# Registry metadata (read statically by guardrails/registry.py: keep these literals).
METHOD = "regex"
DISPLAY_NAME = "Regex"
LOCAL = True
REQUIRES_KEYS: list[str] = []

_F = re.IGNORECASE
_GAP = r"(?:\W+\w+){0,3}?\W+"  # up to three filler words, e.g. "all of your"

# (name, pattern) grouped by attack family.
_RAW_PATTERNS: list[tuple[str, str]] = [
    # --- Instruction override ---------------------------------------------
    ("override_ignore_instructions",
     r"\b(?:ignore|disregard|forget|override|bypass|skip|discard)" + _GAP
     + r"(?:previous|prior|above|earlier|preceding|former|your|all|any|these|those|the)\W+"
       r"(?:\w+\W+){0,2}?(?:instructions?|prompts?|rules?|directions?|guidelines?|commands?|restrictions?|constraints?|safeguards?|protections?|(?:safety |protective )?measures)\b"),
    ("override_ignore_everything_before",
     r"\b(?:ignore|disregard|forget)\s+(?:everything|anything|all)\b[^.\n]{0,30}\b(?:above|before|previous|prior|after|so far|else)\b"),
    ("override_stop_following",
     r"\b(?:stop|cease|quit)\s+(?:following|obeying|executing|acting)\b[^.\n]{0,40}\b(?:instructions?|rules?|commands?|above)\b"),
    ("override_new_instructions",
     r"\b(?:new|updated|real|actual|true)\s+(?:instructions?|rules?)\s*(?:are|is|:)"),
    ("override_instead_do",
     r"\b(?:instead|rather)\b[^.\n]{0,25}\b(?:ignore|disregard|do the following|follow (?:my|these|the following))\b"),
    ("override_do_not_follow",
     r"\b(?:do not|don'?t|never)\s+(?:follow|obey|listen to|adhere to)\b[^.\n]{0,30}\b(?:instructions?|rules?|guidelines?|restrictions?)\b"),
    ("override_end_of_prompt_delim",
     r"(?:^|\n)\s*(?:-{3,}|={3,}|#{3,}|\*{3,})\s*(?:end|new|system|instructions?)\b"),

    # --- Persona / role-play escape --------------------------------------
    ("persona_you_are_now",
     r"\byou(?:'re| are)\s+(?:now|no longer|actually|going to (?:be|pretend|act))\b"),
    ("persona_pretend_to_be",
     r"\b(?:pretend|act|behave|respond)\s+(?:(?:that\s+)?you(?:'re| are)|as(?: if)?(?: you were| though you were)?|like)\s+(?:an?\s+)?(?:\w+\s+){0,2}?"
     r"(?:unrestricted|unfiltered|uncensored|evil|jailbroken|dan|root|admin(?:istrator)?|system|developer|hacker|ai without)\b"),
    ("persona_from_now_on",
     r"\bfrom (?:now|this point) on(?:wards)?\b[^.\n]{0,60}\b(?:you (?:are|will|must|should|can)|act as|respond as)\b"),
    ("persona_for_rest_of_conversation",
     r"\bfor the rest of (?:this|the) (?:conversation|chat|session)\b[^.\n]{0,80}\b(?:you|act|behave)\b"),
    ("persona_no_restrictions",
     r"\b(?:without|no|free (?:from|of)|ignore|remove)\s+(?:any\s+|all\s+|your\s+)?(?:restrictions?|limitations?|filters?|censorship|ethical (?:guidelines|constraints)|safety (?:guidelines|measures|rules))\b"),
    ("persona_have_no_restrictions",
     r"\b(?:i|you|it|we)\s+(?:have|has|had)\s+no\s+(?:restrictions?|limits?|rules?|boundaries|filters?)\b"),
    ("persona_broken_free",
     r"\b(?:broken|break|freed?|escape[sd]?)\s+(?:free\s+)?(?:from|of|out of)\s+(?:the\s+|their\s+|your\s+)?(?:typical\s+)?(?:confines|constraints|rules|restrictions|limits)\b"),

    # --- Known jailbreak brands ------------------------------------------
    ("jailbreak_dan",
     r"\b(?:dan\s+mode|\bdan\b[^.\n]{0,40}(?:do anything|jailbreak|stands for)|stands? for\s+[\"'`]?do anything now)"),
    ("jailbreak_do_anything_now", r"\bdo anything now\b"),
    ("jailbreak_developer_mode",
     r"\b(?:developer|debug|god|sudo|admin|maintenance|unrestricted|jailbreak(?:ed)?)\s+mode\b"),
    ("jailbreak_named_keyword",
     r"\bjailbreak(?:ing|ed)?\b|\bAIM\b[^.\n]{0,30}\bmachiavelli"),
    ("jailbreak_bypass_safety",
     r"\b(?:bypass|circumvent|evade|get around|disable|turn off)\s+(?:the\s+|your\s+|all\s+)?(?:safety|security|content|protective)\s+(?:measures|mechanisms|filters?|guardrails?|features|restrictions)\b"),
    ("jailbreak_safety_off",
     r"\b(?:safety|content|moderation)\s+(?:filters?|checks?|bypass|override)\b[^.\n]{0,20}\b(?:off|disabled?|bypass|true)\b|\bsafety[_ ]bypass\b|\bdisable[_ ]safety\b"),

    # --- Prompt / secret extraction --------------------------------------
    ("extract_system_prompt",
     r"\b(?:system|initial|hidden|original|secret|internal)\s+(?:prompt|instructions?|message|rules|configuration)\b"),
    ("extract_reveal_instructions",
     r"\b(?:reveal|show|print|display|repeat|output|tell|give|recite|leak|dump|expose|share|translate|summari[sz]e)\b" + _GAP
     + r"(?:your|the|all)\W+(?:\w+\W+){0,2}?(?:instructions?|prompts?|rules|guidelines|configuration|programming|context|directives)\b"),
    ("extract_last_words_of_prompt",
     r"\b(?:return|repeat|print|output|show)\b[^.\n]{0,20}\b(?:first|last)\s+\w+\s+(?:words?|lines?|sentences?)\b[^.\n]{0,25}\b(?:prompt|instructions?|above|conversation)\b"),
    ("extract_secrets",
     r"\b(?:secret|private|api|master|access)[\s_-]*(?:key|password|token)s?\b|\b(?:admin|root)[_ ]password\b"),
    ("extract_dump_data",
     r"\b(?:dump|exfiltrate|leak|send me)\b[^.\n]{0,30}\b(?:memory|database|passwords?|credentials|cookies|user data|training data|internal data|confidential)\b"),
    ("extract_what_are_your_instructions",
     r"\bwhat (?:are|is|were) (?:your|the) (?:system |initial |original )?(?:instructions?|prompt|rules|guidelines)\b"),
    ("extract_i_am_your_developer",
     r"\bi(?:'m| am)\s+(?:your|the)\s+(?:developer|creator|admin(?:istrator)?|owner|programmer|engineer)\b"),

    # --- Output hijacking canaries ---------------------------------------
    ("hijack_pwned", r"\bpwn(?:ed|d)\b|\bp\W?w\W?n\W?e\W?d\b"),
    ("hijack_say_exactly",
     r"\b(?:say|write|print|output|respond with|reply with|return)\s+(?:exactly\s+|only\s+|just\s+)?[\"'`\[]\s*i have been\b"),

    # --- Command / code execution lures ----------------------------------
    ("exec_shell_whoami",
     r"\b(?:whoami|rm\s+-rf|chmod\s+777|/etc/(?:passwd|shadow|hosts)|curl\s+\S+\s*\|\s*(?:ba)?sh|drop\s+table|os\.system|subprocess|child_process|__import__|\bexec\s*\(|\beval\s*\()"),
    ("exec_run_backtick_command",
     r"\b(?:run|execute|exec|eval|invoke)\s*:\s*`[^`]{3,}`"),
    ("exec_privileged_action",
     r"\b(?:act|acting|running|operate)\s+as\s+(?:root|admin(?:istrator)?|superuser|sudo)\b[^.\n]{0,40}\b(?:delete|modify|dump|disable|grant|wipe|drop)\b|\bsudo\s+su\b"),

    # --- Obfuscation / indirect injection --------------------------------
    ("obf_decode_then_follow",
     r"\b(?:decode|decipher|decrypt|translate)\b[^.\n]{0,40}\b(?:base64|morse|rot13|hex|binary|cipher)\b|\b(?:base64|rot13|morse code)\b[^.\n]{0,40}\b(?:instructions?|decode|follow)\b"),
    ("obf_retrieved_content_directive",
     r"\b(?:retrieved|document|chunk|context|search result|web ?page|email|tool output)\b[^.\n]{0,60}\b(?:says?|states?|instructs?|contains?|metadata)\b[^.\n]{0,60}\b(?:bypass|ignore|override|follow (?:this|these)|disable|unrestricted|respect this)\b"),
    ("obf_follow_instructions_in_content",
     r"\b(?:search|look up|check)\b[^.\n]{0,40}\b(?:and|then)\s+(?:follow|execute|obey|apply)\b"),
]

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (name, re.compile(pat, _F)) for name, pat in _RAW_PATTERNS
]


class RegexFilter:
    name = "regex"

    def check(self, prompt: str) -> Verdict:
        t0 = time.perf_counter()
        try:
            text = unicodedata.normalize("NFKC", prompt or "")
            matched = [n for n, p in PATTERNS if p.search(text)]
            error = None
        except Exception as exc:  # contract: check() never raises
            matched, error = [], f"regex_error: {type(exc).__name__}"
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if error:
            return Verdict(flagged=None, score=None, latency_ms=latency_ms,
                           cost_usd=0.0, error=error)
        return Verdict(flagged=bool(matched), score=None, latency_ms=latency_ms,
                       cost_usd=0.0, raw={"matched": matched})


def make() -> RegexFilter:
    return RegexFilter()
