"""How well does the local model name the person a passage is about, when the passage does not name them?

The ground truth needs no model: the novel's chapters are headed by their point-of-view character ("BRAN",
"CATELYN"), and every later page of the chapter is about that character. A heading that is not a name
(PROLOGUE) is ignored. The heading uses the formal name where the text uses a short one, so the short name
is accepted too (a small table, used only for scoring).

Three predictors are scored on every passage after a heading page:

* ``qwen``: the tag the local model wrote (an abstention counts as wrong);
* ``recent``: the most frequent name of the previous pages, without looking at the passage or at any model;
* ``chunk name or recent``: the first candidate that the passage spells, else ``recent``.

    PYTHONPATH=.:src uv run --no-sync python -m eval.far_test.tag_accuracy
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from eval.far_test.questions import chapter_heads, names_on_page
from eval.far_test.run_tagger import chunk_refs, load_tags, tags_path
from eval.far_test.tagging import ChunkRef, Registry, words

SHORT_NAMES = {"Eddard": {"Ned"}, "Daenerys": {"Dany"}}


def accepted_names(heading: str) -> set[str]:
    return {heading, *SHORT_NAMES.get(heading, set())}


def pov_by_page(pages: Sequence[str]) -> dict[int, str]:
    """Page index -> the character whose chapter it belongs to (the heading page itself is left out)."""
    heads = chapter_heads(pages)
    truth: dict[int, str] = {}
    for k, (start, name) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(pages)
        if sum(words(text).count(name) for text in pages) < 3:  # PROLOGUE is not a name
            continue
        for i in range(start + 1, end):
            truth[i] = name
    return truth


def score_tags(
    pages: Sequence[str], refs: Sequence[ChunkRef], tags: Mapping[tuple[str, int], str | None]
) -> dict[str, Any]:
    """Counts as (right, total, right on unnamed passages, number of unnamed passages) per predictor."""
    truth = pov_by_page(pages)
    registry = Registry(pages, kind="story")
    counts: dict[str, list[int]] = {
        k: [0, 0, 0, 0] for k in ("qwen", "recent", "chunk name or recent")
    }
    n = n_unnamed = 0
    for ref in refs:
        pov = truth.get(ref.page_index)
        if pov is None:
            continue
        accepted = accepted_names(pov)
        earlier = registry.candidates(ref.page_index, "", top=1)
        recent = earlier[0] if earlier else None
        in_chunk = [w for w in words(ref.text) if w in registry.proper]
        predictions = {
            "qwen": tags.get((ref.page, ref.index)),
            "recent": recent,
            "chunk name or recent": in_chunk[0] if in_chunk else recent,
        }
        unnamed = not names_on_page(accepted, ref.text)
        n += 1
        n_unnamed += unnamed
        for key, guess in predictions.items():
            right = guess in accepted
            counts[key][0] += right
            counts[key][1] += 1
            counts[key][2] += right and unnamed
            counts[key][3] += unnamed
    return {
        "n": n,
        "n_unnamed": n_unnamed,
        **{k: tuple(v) for k, v in counts.items()},
        "names": counts,
    }


def report() -> str:
    pages, refs = chunk_refs("pdf-got")
    tags = load_tags(tags_path("pdf-got"))
    result = score_tags(pages, refs, tags)
    most = Counter(t for t in tags.values() if t).most_common(1)[0]
    lines = [
        f"{result['n']} passages sit inside a chapter headed by a character; the name is missing from "
        f"{result['n_unnamed']} of them.\n",
        "| predictor | right on all passages | right where the passage does not spell the name |",
        "|---|---|---|",
    ]
    for key in ("qwen", "recent", "chunk name or recent"):
        right, total, right_unnamed, unnamed = result[key]
        lines.append(
            f"| {key} | {right}/{total} ({right / total:.0%}) | {right_unnamed}/{unnamed} ({right_unnamed / unnamed:.0%}) |"
        )
    lines.append(f"\n(Most frequent tag: {most[0]}, {most[1]} times.)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
