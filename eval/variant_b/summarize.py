"""B4 (one summary per child chunk, with context slots) and B5 (grounding check, one retry, extractive fallback)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from eval.variant_b import prompts
from eval.variant_b.llm import SlmClient
from eval.variant_b.profiles import Profile

# Chosen on the 50-chunk smoke test (eval/variant_b/GCP_GPU_PLAN.md): at the model card's temperature 1.0 and
# presence penalty 2.0, 43/50 outputs had the right format and 42/50 were grounded; at 0.2 / 0.0 it was 49/50 and 48/50.
SAMPLING = {"temperature": 0.2, "top_p": 0.9, "top_k": 20, "presence_penalty": 0.0}
RETRY_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "top_k": 20, "presence_penalty": 0.0}
MAX_WORDS = 90  # the prompt asks for 80; a little slack before a retry
PARENT_LIMIT = 1200


@dataclass(frozen=True)
class Context:
    """Everything one chunk's prompt is built from."""

    title: str
    profile: Profile
    text: str
    synopsis: str = ""
    glossary: str = "none"
    previous: tuple[str, ...] = ()
    parent_context: str = "none"
    glossary_names: tuple[
        str, ...
    ] = ()  # surface forms of the glossary entries shown in ``glossary``


@dataclass
class Check:
    ok: bool
    reasons: list[str]
    unsupported: list[str]
    words: int


@dataclass
class Result:
    summary: str
    status: str  # ok | retried | fallback
    check: Check
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    reasons_first_try: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Mode:
    """What to ask the small model for, and how to check the answer."""

    name: str
    system: str
    user: str  # template; only the slots it names are filled in
    retry: str
    max_tokens: int
    shape: str  # bullets | sentences | questions | keywords
    max_words: int
    strict_support: bool  # True: names/numbers must come from the chunk, title or glossary names (not the whole prompt)
    uses_previous: bool = (
        False  # chain the two previous summaries into the prompt (sequential per document)
    )
    allow_parent: bool = (
        False  # strict support may also come from the surrounding text of the parent chunk
    )
    clean_openings: bool = (
        False  # strip formulaic openings such as "This passage describes" from the output
    )


SUMMARY_V1 = Mode(
    "summary_v1", prompts.SUMMARY_SYSTEM, prompts.SUMMARY_USER, prompts.SUMMARY_RETRY, 300,
    "bullets", MAX_WORDS, False, True,
)  # fmt: skip
SUMMARY_V2 = Mode(
    "summary_v2", prompts.SUMMARY_V2_SYSTEM, prompts.SUMMARY_V2_USER, prompts.RETRY_SENTENCES, 120,
    "sentences", 45, True,
)  # fmt: skip
CONTEXT = Mode(
    "context", prompts.CONTEXT_SYSTEM, prompts.CONTEXT_USER, prompts.RETRY_SENTENCES, 140,
    "sentences", 55, False,
)  # fmt: skip
QUESTIONS = Mode(
    "questions", prompts.QUESTIONS_SYSTEM, prompts.QUESTIONS_USER, prompts.RETRY_QUESTIONS, 200,
    "questions", 70, True,
)  # fmt: skip
KEYWORDS = Mode(
    "keywords", prompts.KEYWORDS_SYSTEM, prompts.KEYWORDS_USER, prompts.RETRY_KEYWORDS, 120,
    "keywords", 50, True,
)  # fmt: skip
CONTEXT_V2 = Mode(
    "context_v2", prompts.CONTEXT_V2_SYSTEM, prompts.CONTEXT_V2_USER, prompts.RETRY_SENTENCES, 140,
    "sentences", 50, True, False, True, True,
)  # fmt: skip
MODES: dict[str, Mode] = {
    m.name: m for m in (SUMMARY_V1, SUMMARY_V2, CONTEXT, CONTEXT_V2, QUESTIONS, KEYWORDS)
}


# ---------------------------------------------------------------------------------------- prompt
def parent_window(parent: str, child: str, limit: int = PARENT_LIMIT) -> str:
    """The text around the chunk inside its parent (not the chunk itself), at most ``limit`` characters."""
    parent, child = parent.strip(), child.strip()
    if not parent or len(parent) <= len(child) * 1.15:
        return "none"  # the parent adds nothing beyond the chunk
    i = parent.find(child[:80])
    if i == -1:
        return parent[:limit]
    head_room = limit * 6 // 10
    head = parent[:i].strip()[-head_room:]
    tail = parent[i + len(child) :].strip()[: limit - head_room]
    if not head and not tail:
        return "none"
    parts = []
    if head:
        parts.append(f"...{head}")
    parts.append("[the SOURCE passage goes here]")
    if tail:
        parts.append(f"{tail}...")
    return "\n".join(parts)


def build_user_message(ctx: Context, mode: Mode = SUMMARY_V1) -> str:
    previous = "\n---\n".join(s[:600] for s in ctx.previous) if ctx.previous else "none"
    return mode.user.format(
        title=ctx.title,
        profile_rule=ctx.profile.rule,
        synopsis=ctx.synopsis or "none",
        glossary=ctx.glossary or "none",
        previous=previous,
        parent_context=ctx.parent_context or "none",
        text=ctx.text,
    )


# ---------------------------------------------------------------------------------------- B5 check
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-./%]*")


_NUMBER = re.compile(r"^[$€£]?[0-9][0-9.,]*%?$")


def _fold(s: str) -> str:
    return re.sub(r"[\s\-_]+", " ", s.lower())


def _supported(tok: str, ctx: str) -> bool:
    """Is the token (hyphens, case and a plural 's' ignored) in the folded context? Paths and ratios are split on '/'."""
    folded = _fold(tok)
    if folded in ctx or (folded.endswith("s") and folded[:-1] in ctx):
        return True
    if "/" in tok:
        parts = [p for p in tok.split("/") if p]
        return len(parts) > 1 and all(_supported(p, ctx) for p in parts)
    if _NUMBER.match(tok):  # a plain number: match it as a whole number anywhere in the context
        core = tok.strip("$€£%").strip(".,")
        return bool(core) and bool(re.search(r"(?<![0-9])" + re.escape(core) + r"(?![0-9])", ctx))
    return False


def unsupported_tokens(summary: str, context: str) -> list[str]:
    """Numbers and mid-sentence capitalised words of the summary that do not occur in what the model was given."""
    ctx = _fold(context)
    bad: list[str] = []
    for line in summary.splitlines():
        line = line.strip().lstrip("-").strip()
        for m in _TOKEN.finditer(line):
            tok = m.group(0).rstrip(".-/")
            if not tok:
                continue
            prefix = line[: m.start()].rstrip()
            sentence_initial = prefix == "" or prefix[-1] in ".!?:;"
            has_digit = any(ch.isdigit() for ch in tok)
            if (has_digit or (tok[0].isupper() and not sentence_initial)) and not _supported(
                tok, ctx
            ):
                bad.append(tok)
    return bad


_META = re.compile(
    r"\b(does not|doesn't|did not|is not|isn't|are not) (specify|state|mention|say|provide|describe|include)\b|"
    r"\bnot (specified|stated|mentioned|provided)\b|\bunclear\b",
    re.IGNORECASE,
)


def unsupported_terms(text: str, context: str) -> list[str]:
    """Keyword mode: every comma-separated term must occur in the context, or all of its words must."""
    ctx = _fold(context)
    bad: list[str] = []
    for term in (t.strip() for t in text.split(",")):
        folded = _fold(term)
        if not folded or folded in ctx or (folded.endswith("s") and folded[:-1] in ctx):
            continue
        parts = [w for w in re.findall(r"[a-z0-9][a-z0-9_.]*", folded) if len(w) >= 3]
        if not parts or not all(_supported(w, ctx) or w in ctx for w in parts):
            bad.append(term)
    return bad


def check_summary(summary: str, context: str, mode: Mode = SUMMARY_V1) -> Check:
    lines = [ln.strip() for ln in summary.strip().splitlines() if ln.strip()]
    bullets = [ln for ln in lines if ln.startswith("- ")]
    words = len(summary.split())
    reasons: list[str] = []
    if mode.shape == "keywords":
        unsupported = unsupported_terms(summary, context)
    else:
        unsupported = unsupported_tokens(summary, context)
    if not summary.strip():
        reasons.append("empty")
    elif mode.shape == "bullets" and not (2 <= len(bullets) <= 4 and len(bullets) == len(lines)):
        reasons.append(
            f"{len(lines)} lines, {len(bullets)} bullets (need 2 to 4 bullets and nothing else)"
        )
    elif mode.shape == "sentences":
        if bullets or len(lines) > 2:
            reasons.append(f"{len(lines)} lines, {len(bullets)} bullets (need plain sentences)")
        if len(re.findall(r"[.!?](\s|$)", summary.strip())) > 3:
            reasons.append("more than three sentences")
        if _META.search(summary):
            reasons.append("comment about what the passage leaves out")
    elif mode.shape == "questions":
        if not (len(lines) == 3 and len(bullets) == 3 and all(ln.endswith("?") for ln in lines)):
            reasons.append(
                f"{len(lines)} lines, {len(bullets)} bullets (need three '- ...?' lines)"
            )
    elif mode.shape == "keywords":
        terms = [t for t in (x.strip() for x in summary.split(",")) if t]
        if len(lines) != 1 or not 4 <= len(terms) <= 14 or any(len(t.split()) > 6 for t in terms):
            reasons.append(
                f"{len(lines)} lines, {len(terms)} terms (need one line of 6 to 12 short terms)"
            )
    if words > mode.max_words:
        reasons.append(f"{words} words")
    if "<think" in summary or "</think" in summary:
        reasons.append("reasoning text")
    if unsupported:
        reasons.append("not in the source or context: " + ", ".join(dict.fromkeys(unsupported)))
    return Check(not reasons, reasons, unsupported, words)


_OPENING = re.compile(
    r"^\s*(?:this|the)\s+(?:passage|section|excerpt|text|document|chunk|appendix|page)"
    r"(?:\s+(?:from|in|of)\s+[^,.;]{0,80}?)?\s*,?\s+"
    r"(?:primarily\s+|also\s+|mainly\s+)?(?:describes|explains|discusses|covers|outlines|details|presents|"
    r"introduces|summari[sz]es|provides|contains|is about|focuses on|addresses|lists)\s+(?:how\s+|that\s+)?",
    re.IGNORECASE,
)


def strip_opening(text: str) -> str:
    """Remove a formulaic opening ("This passage from the thesis describes ...") and re-capitalise the rest."""
    cleaned = _OPENING.sub("", text.strip(), count=1)
    if cleaned == text.strip() or not cleaned:
        return text.strip()
    return cleaned[0].upper() + cleaned[1:]


def fallback_summary(text: str, max_words: int = 60) -> str:
    """Extractive fallback: the first one or two sentences of the chunk, flagged by the caller."""
    flat = " ".join(text.split())
    out: list[str] = []
    words = 0
    for sentence in re.split(r"(?<=[.!?])\s+", flat):
        n = len(sentence.split())
        if out and words + n > max_words:
            break
        out.append(sentence)
        words += n
        if len(out) == 2:
            break
    return "- " + " ".join(out)[:600]


# ---------------------------------------------------------------------------------------- B4 + B5
def support_text(ctx: Context, user: str, mode: Mode) -> str:
    """What a name or number in the output may come from: the whole prompt, or only the chunk, title and glossary names."""
    if not mode.strict_support:
        return user
    parts = [ctx.title, ctx.text, " ".join(ctx.glossary_names)]
    if mode.allow_parent:
        parts.append(ctx.parent_context)
    return "\n".join(parts)


def _final(text: str, mode: Mode) -> str:
    return strip_opening(text) if mode.clean_openings else text.strip()


async def summarise(llm: SlmClient, ctx: Context, mode: Mode = SUMMARY_V1) -> Result:
    user = build_user_message(ctx, mode)
    support = support_text(ctx, user, mode)
    first = await llm.chat(mode.system, user, max_tokens=mode.max_tokens, **SAMPLING)
    check = check_summary(first.text, support, mode)
    pt, ct, calls = first.prompt_tokens, first.completion_tokens, 1
    if check.ok:
        return Result(_final(first.text, mode), "ok", check, pt, ct, calls)
    retry_user = user + mode.retry.format(reasons="; ".join(check.reasons))
    second = await llm.chat(mode.system, retry_user, max_tokens=mode.max_tokens, **RETRY_SAMPLING)
    pt, ct, calls = pt + second.prompt_tokens, ct + second.completion_tokens, 2
    check2 = check_summary(second.text, support, mode)
    if check2.ok:
        return Result(_final(second.text, mode), "retried", check2, pt, ct, calls, check.reasons)
    fallback = fallback_summary(ctx.text) if mode.shape == "bullets" else ""
    return Result(fallback, "fallback", check2, pt, ct, calls, check.reasons)
