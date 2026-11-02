"""B3: a short synopsis of the whole document (static profiles), or a "story so far" note per passage (causal)."""

from __future__ import annotations

import asyncio
import re

from eval.variant_b import prompts
from eval.variant_b.glossary import split_spans
from eval.variant_b.llm import SlmClient

MAX_SYNOPSIS_CHARS = 1100
_GREEDY = dict(temperature=0.0, top_p=1.0)


def clean_prose(text: str, limit: int = MAX_SYNOPSIS_CHARS) -> str:
    t = re.sub(r"^\s*(synopsis|note|story so far)\s*:\s*", "", text.strip(), flags=re.IGNORECASE)
    t = " ".join(t.split())
    if len(t) <= limit:
        return t
    cut = t.rfind(". ", 0, limit)
    return t[: cut + 1] if cut > limit // 2 else t[:limit].rstrip()


async def static_synopsis(llm: SlmClient, text: str) -> str:
    """One synopsis for the whole document; long documents are summarised in parts and then combined."""
    pieces = [p for _, p in split_spans(text, 9000)]
    if not pieces:
        return ""
    if len(pieces) == 1:
        r = await llm.chat(
            prompts.SYNOPSIS_SYSTEM,
            prompts.SYNOPSIS_PART.format(text=pieces[0]),
            max_tokens=220,
            **_GREEDY,
        )
        return clean_prose(r.text)
    parts = await asyncio.gather(
        *(
            llm.chat(
                prompts.SYNOPSIS_SYSTEM,
                prompts.SYNOPSIS_PART.format(text=p),
                max_tokens=160,
                **_GREEDY,
            )
            for p in pieces
        )
    )
    notes = [clean_prose(r.text, 700) for r in parts]
    while True:
        batches = [notes[i : i + 8] for i in range(0, len(notes), 8)]
        replies = await asyncio.gather(
            *(
                llm.chat(
                    prompts.SYNOPSIS_SYSTEM,
                    prompts.SYNOPSIS_COMBINE.format(parts="\n---\n".join(b)),
                    max_tokens=220,
                    **_GREEDY,
                )
                for b in batches
            )
        )
        notes = [clean_prose(r.text) for r in replies]
        if len(notes) == 1:
            return notes[0]


async def story_notes(llm: SlmClient, spans: list[str]) -> list[str]:
    """``notes[k]`` is the "story so far" available BEFORE span k (it covers spans 0..k-1 only)."""
    notes: list[str] = []
    note = ""
    for span in spans:
        notes.append(note)
        r = await llm.chat(
            prompts.STORY_SYSTEM,
            prompts.STORY_USER.format(note=note or "(nothing yet)", text=span),
            max_tokens=260,
            **_GREEDY,
        )
        note = clean_prose(r.text, 900)
    return notes
