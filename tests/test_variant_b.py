"""Variant B (eval/variant_b): the parts that need no GPU, database or network."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from eval.variant_b import glossary as G
from eval.variant_b import pipeline as P
from eval.variant_b import recipes as R
from eval.variant_b import summarize as Z
from eval.variant_b.llm import Reply
from eval.variant_b.profiles import (
    DATASET_PROFILE,
    PROFILES,
    SHORT_DOC_DATASETS,
    group_and_order,
    profile_for_dataset,
)


class FakeLlm:
    """Returns scripted replies in order and records the prompts it was given."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.seen: list[tuple[str, str]] = []
        self.model = "fake"
        self.calls = 0
        self.cache_hits = 0

    async def chat(self, system: str, user: str, **kw: Any) -> Reply:
        self.seen.append((system, user))
        self.calls += 1
        text = self.replies.pop(0)
        return Reply(text, 100, 20, 0.01, "stop", False)


# ------------------------------------------------------------------------------------------ profiles
def test_every_benchmark_dataset_has_a_profile_and_short_docs_are_known() -> None:
    for ds, name in DATASET_PROFILE.items():
        assert name in PROFILES, ds
    assert profile_for_dataset("unknown-set").name == "generic"
    assert SHORT_DOC_DATASETS <= set(DATASET_PROFILE)


def test_modes_follow_the_spec() -> None:
    assert PROFILES["fiction"].mode == "causal"
    assert PROFILES["conversation"].mode == "causal"
    assert PROFILES["legal_contract"].mode == "static"
    assert profile_for_dataset("pdf-got").mode == "causal"
    assert profile_for_dataset("pdf-quantum").mode == "static"


def test_group_and_order() -> None:
    assert group_and_order("docsqa", "docs/PRD.md#012") == ("docs/PRD.md", 12)
    assert group_and_order("docsqa", "README.md#000") == ("README.md", 0)
    assert group_and_order("pdf-got", "p0017") == ("pdf-got", 17)
    assert group_and_order("scifact", "12345") == ("12345", 0)


# ------------------------------------------------------------------------------------------ glossary
def test_quote_support_exact_fuzzy_failed() -> None:
    span = "Lord Eddard Stark was beheaded at the Sept of Baelor by order of the king."
    assert G.quote_support("was beheaded at the Sept of Baelor", span) == "exact"
    assert (
        G.quote_support("was beheaded at the Sept of Baelor by order of the queen", span) == "fuzzy"
    )  # one word off
    assert G.quote_support("he lived happily ever after in the north", span) == "failed"
    assert (
        G.quote_support("too short", span) == "failed"
    )  # under 8 normalised characters is not evidence


def test_split_spans_cover_the_text_and_report_offsets() -> None:
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(10))
    spans = G.split_spans(text, 500)
    assert len(spans) > 1
    for start, piece in spans:
        assert text[start : start + len(piece)] == piece
        assert len(piece) <= 500
    assert "".join(p for _, p in spans).replace(" ", "").replace("\n", "") == text.replace(
        " ", ""
    ).replace("\n", "")


async def test_extract_span_keeps_verified_entries_and_drops_invented_ones() -> None:
    span = "The Licensee shall pay the fee within 30 days. The Licensor may terminate the agreement for late payment."
    reply = json.dumps(
        {
            "entries": [
                {"names": ["Licensee"], "kind": "party", "definition": "The party that pays the fee.",
                 "status": "", "quote": "The Licensee shall pay the fee within 30 days"},
                {"names": ["Licensor"], "kind": "party", "definition": "Can end the agreement.",
                 "status": "", "quote": "Licensor invents a quote that is not here at all"},
                {"names": ["Arbitrator"], "kind": "party", "definition": "Settles disputes.",
                 "status": "", "quote": "The Licensee shall pay the fee within 30 days"},
                {"names": ["it"], "kind": "other", "definition": "Nothing.", "status": "", "quote": "x" * 20},
            ]
        }
    )  # fmt: skip
    llm = FakeLlm([reply])
    entries, stats = await G.extract_span(llm, profile_for_dataset("docsqa"), span, 3)  # type: ignore[arg-type]
    assert [e.names for e in entries] == [["Licensee"]]
    assert entries[0].first_seen == 3 and entries[0].support == "exact"
    assert stats["raw"] == 4 and stats["kept"] == 1
    assert (
        stats["dropped_quote"] == 1 and stats["dropped_name"] == 1 and stats["dropped_empty"] == 1
    )


async def test_extract_span_retries_once_when_the_json_is_broken() -> None:
    good = json.dumps(
        {
            "entries": [
                {
                    "names": ["Alpha"], "kind": "component", "definition": "The main component.",
                    "status": "", "quote": "Alpha is the main component",
                }
            ]
        }
    )  # fmt: skip
    llm = FakeLlm(['{"entries": [{"names": ["Alpha"', good])
    entries, stats = await G.extract_span(
        llm, profile_for_dataset("docsqa"), "Alpha is the main component.", 0
    )  # type: ignore[arg-type]
    assert len(entries) == 1 and stats["retry"] == 1 and "parse_error" not in stats

    llm = FakeLlm(["not json", "still not json"])
    entries, stats = await G.extract_span(llm, profile_for_dataset("docsqa"), "text " * 10, 0)  # type: ignore[arg-type]
    assert entries == [] and stats["parse_error"] == 1 and stats["retry"] == 1


def test_merge_entries_joins_aliases_and_keeps_the_earliest_definition() -> None:
    a = G.Entry(
        ["Ned Stark", "Ned"], "person", "Lord of Winterfell.", "q1 quote text", 0, [(0, "")], 1
    )
    b = G.Entry(
        ["Eddard Stark", "Ned"],
        "person",
        "Executed ruler.",
        "q2 quote text",
        4,
        [(4, "executed")],
        1,
    )
    c = G.Entry(["Arya"], "person", "His daughter.", "q3 quote text", 1, [], 1)
    merged = G.merge_entries([b, a, c])
    assert len(merged) == 2
    ned = next(e for e in merged if "Ned" in e.names)
    assert ned.definition == "Lord of Winterfell." and ned.first_seen == 0
    assert ned.mentions == 2 and (4, "executed") in ned.timeline
    assert set(ned.names) >= {"Ned Stark", "Ned", "Eddard Stark"}


def test_merge_never_joins_entries_on_a_generic_word() -> None:
    a = G.Entry(["context", "RLM"], "feature", "How the RLM sees data.", "quote text here", 0)
    b = G.Entry(["context", "Skill"], "feature", "A packaged capability.", "quote text here", 1)
    c = G.Entry(["rlm", "kernel"], "component", "The REPL.", "quote text here", 2)
    merged = G.merge_entries([a, b, c])
    assert len(merged) == 2  # a and c share the specific name "RLM"; b shares only "context"
    assert any("Skill" in e.names and "RLM" not in e.names for e in merged)


def test_entries_with_many_aliases_are_matched_on_their_specific_names_only() -> None:
    lumped = G.Entry(
        ["context", "host", "kernel", "RLM"], "feature", "Lumped together.", "quote text here", 0
    )
    assert G.select_slice([lumped], "the host keeps the context", 0, causal=False) == []
    assert G.select_slice([lumped], "the RLM keeps it", 0, causal=False) == [lumped]
    plain = G.Entry(["kernel"], "component", "Runs generated code.", "quote text here", 0)
    assert G.select_slice([plain], "the kernel runs it", 0, causal=False) == [plain]


def test_causal_slice_hides_the_future_and_shows_the_state_at_that_point() -> None:
    ned = G.Entry(
        ["Ned Stark", "Ned"],
        "person",
        "Lord of Winterfell.",
        "quote one here",
        0,
        [(2, "executed")],
    )
    jon = G.Entry(["Jon"], "person", "A bastard son.", "quote two here", 5, [])
    text = "Ned told Jon the news."
    before = G.select_slice([ned, jon], text, 1, causal=True)
    assert before == [ned]  # Jon first appears at span 5: not known at span 1
    assert "[executed]" not in G.render_slice(before, 1, causal=True)
    after = G.select_slice([ned, jon], text, 6, causal=True)
    assert {e.names[0] for e in after} == {"Ned Stark", "Jon"}
    assert "[executed]" in G.render_slice(after, 6, causal=True)
    # static documents may use everything
    assert {e.names[0] for e in G.select_slice([ned, jon], text, 0, causal=False)} == {
        "Ned Stark",
        "Jon",
    }


def test_slice_is_capped_and_ignores_unmentioned_entries() -> None:
    entries = [G.Entry([f"Term{i}"], "term", "d" * 200, "quote text here", 0) for i in range(30)]
    text = " ".join(f"Term{i}" for i in range(30))
    assert len(G.select_slice(entries, text, 0, causal=False)) == 8
    assert len(G.render_slice(entries, 0, causal=False, max_chars=500)) <= 500
    assert G.render_slice([], 0, causal=False) == "none"
    assert G.select_slice(entries, "nothing relevant here", 0, causal=False) == []


# ------------------------------------------------------------------------------------------ summaries
def test_unsupported_tokens_fold_hyphens_and_catch_invented_numbers_and_names() -> None:
    ctx = "The study estimates the 10 year risk of heart disease in 7.89 million patients at Zaandam Medical Centre."
    ok = (
        "- The study estimates 10-year risk in 7.89 million patients at the Zaandam Medical Centre."
    )
    assert Z.unsupported_tokens(ok, ctx) == []
    bad = "- Risk fell by 25% in Rotterdam."
    assert Z.unsupported_tokens(bad, ctx) == ["25%", "Rotterdam"]
    assert (
        Z.unsupported_tokens("- Figures are given.", ctx) == []
    )  # sentence-initial capitals are ignored


def test_unsupported_tokens_accept_model_names_paths_and_ratios_that_the_source_contains() -> None:
    ctx = "Use `BAAI/bge-reranker-v2-m3` at http://localhost:8000/v1/preferences with blend 0.60/0.40 and Redis 6379/0."
    ok = "- The BAAI/bge-reranker-v2-m3 model (v2-m3) uses a 0.60/0.40 blend; Redis runs on 6379/0; see 8000/v1/preferences."
    assert Z.unsupported_tokens(ok, ctx) == []
    assert Z.unsupported_tokens("- Use the Cohere-rerank-3 model.", ctx) == ["Cohere-rerank-3"]
    assert Z.unsupported_tokens("- Blend is 0.70/0.30.", ctx) == ["0.70/0.30"]


def test_check_summary_rules() -> None:
    ctx = "alpha beta gamma"
    assert Z.check_summary("- alpha\n- beta", ctx).ok
    assert "bullets" in Z.check_summary("Just a sentence.", ctx).reasons[0]
    assert not Z.check_summary("- alpha", ctx).ok  # one bullet is not enough
    assert not Z.check_summary("", ctx).ok
    long = "- " + "alpha " * 100 + "\n- beta"
    assert any("words" in r for r in Z.check_summary(long, ctx).reasons)


def test_parent_window_excludes_the_chunk_and_respects_the_limit() -> None:
    child = "THE CHUNK TEXT " * 10
    parent = "before text " * 80 + child + "after text " * 80
    w = Z.parent_window(parent, child, limit=1200)
    assert "THE CHUNK TEXT" not in w and "[the SOURCE passage goes here]" in w
    assert len(w) <= 1200 + 60
    assert Z.parent_window(child, child) == "none"  # parent adds nothing
    assert Z.parent_window("", child) == "none"


def test_user_message_contains_every_slot() -> None:
    ctx = Z.Context(
        title="T", profile=profile_for_dataset("docsqa"), text="the source", synopsis="syn",
        glossary="- Foo (component): bar", previous=("p1", "p2"), parent_context="around",
    )  # fmt: skip
    msg = Z.build_user_message(ctx)
    for part in (
        "<document>T</document>",
        "syn",
        "- Foo (component): bar",
        "p1\n---\np2",
        "around",
        "the source",
    ):
        assert part in msg
    bare = Z.build_user_message(Z.Context(title="T", profile=ctx.profile, text="x"))
    assert "<synopsis>none</synopsis>" in bare and "<previous>none</previous>" in bare


async def test_summarise_ok_retried_and_fallback_paths() -> None:
    ctx = Z.Context(
        title="Doc",
        profile=profile_for_dataset("docsqa"),
        text="Alpha is a component. Beta is optional.",
    )
    good = "- Alpha is a component.\n- Beta is optional."
    res = await Z.summarise(FakeLlm([good]), ctx)  # type: ignore[arg-type]
    assert res.status == "ok" and res.calls == 1

    llm = FakeLlm(["- Alpha costs 99 dollars in Paris.\n- Beta.", good])
    res = await Z.summarise(llm, ctx)  # type: ignore[arg-type]
    assert res.status == "retried" and res.calls == 2 and res.summary == good
    assert "Revision note" in llm.seen[1][1] and "99" in llm.seen[1][1]
    assert res.reasons_first_try and "not in the source" in res.reasons_first_try[0]

    res = await Z.summarise(FakeLlm(["nonsense", "still nonsense"]), ctx)  # type: ignore[arg-type]
    assert res.status == "fallback" and res.summary.startswith("- Alpha is a component.")


# ------------------------------------------------------------------------------------------ v2 modes
def test_sentence_mode_rejects_bullets_meta_comments_and_long_text() -> None:
    ctx = "Alpha is a component. It is optional."
    assert Z.check_summary("Alpha is an optional component.", ctx, Z.SUMMARY_V2).ok
    assert not Z.check_summary("- Alpha is optional.\n- Beta.", ctx, Z.SUMMARY_V2).ok
    meta = Z.check_summary(
        "Alpha is optional. The document does not specify a default.", ctx, Z.SUMMARY_V2
    )
    assert any("leaves out" in r for r in meta.reasons)
    assert not Z.check_summary("One. Two. Three. Four.", ctx, Z.SUMMARY_V2).ok
    assert not Z.check_summary("alpha " * 60, ctx, Z.SUMMARY_V2).ok


def test_question_mode_needs_three_dash_question_lines() -> None:
    ctx = "Alpha is a component configured with ALPHA_MODE."
    good = "- What is Alpha?\n- How is Alpha configured?\n- Which variable sets ALPHA_MODE?"
    assert Z.check_summary(good, ctx, Z.QUESTIONS).ok
    assert not Z.check_summary("- What is Alpha?\n- How is Alpha configured?", ctx, Z.QUESTIONS).ok
    assert not Z.check_summary(good.replace("?", ".", 1), ctx, Z.QUESTIONS).ok


def test_keyword_mode_checks_every_term_against_the_context() -> None:
    ctx = "Alpha reads ALPHA_MODE and the cache TTL of 300 seconds."
    good = Z.check_summary(
        "Alpha, ALPHA_MODE, cache TTL, 300 seconds, reads, cache", ctx, Z.KEYWORDS
    )
    assert good.ok, good.reasons
    bad = Z.check_summary("Alpha, ALPHA_MODE, cache TTL, Redis cluster, Kafka", ctx, Z.KEYWORDS)
    assert not bad.ok and "Kafka" in bad.unsupported
    assert not Z.check_summary("Alpha\nALPHA_MODE", ctx, Z.KEYWORDS).ok


def test_strict_modes_do_not_accept_words_that_only_the_synopsis_or_neighbours_supply() -> None:
    ctx = Z.Context(
        title="Doc", profile=profile_for_dataset("docsqa"), text="Alpha uses the cache.",
        synopsis="Zeta Cluster overview", previous=("Zeta Cluster was configured",), glossary_names=("Alpha",),
    )  # fmt: skip
    user = Z.build_user_message(ctx, Z.SUMMARY_V1)
    assert "Zeta Cluster" in user and "Zeta Cluster" in Z.support_text(ctx, user, Z.CONTEXT)
    assert "Zeta Cluster" not in Z.support_text(ctx, user, Z.SUMMARY_V2)
    assert "Alpha" in Z.support_text(ctx, user, Z.SUMMARY_V2)
    assert "<synopsis>" not in Z.build_user_message(ctx, Z.QUESTIONS)
    assert "<synopsis>Zeta Cluster overview</synopsis>" in Z.build_user_message(ctx, Z.CONTEXT)


async def test_a_failed_v2_mode_leaves_no_text_behind_and_v1_keeps_its_extractive_fallback() -> (
    None
):
    ctx = Z.Context(
        title="Doc",
        profile=profile_for_dataset("docsqa"),
        text="Alpha is a component. Beta is optional.",
    )
    res = await Z.summarise(
        FakeLlm(["- bullet\n- bullet", "- still bullets\n- no"]), ctx, Z.SUMMARY_V2
    )  # type: ignore[arg-type]
    assert res.status == "fallback" and res.summary == ""
    assert (await Z.summarise(FakeLlm(["x", "y"]), ctx)).summary.startswith("- Alpha")  # type: ignore[arg-type]


async def test_only_the_v1_mode_chains_previous_summaries_by_default() -> None:
    assert Z.SUMMARY_V1.uses_previous and not any(
        m.uses_previous for m in (Z.SUMMARY_V2, Z.CONTEXT, Z.QUESTIONS, Z.KEYWORDS)
    )
    profile = profile_for_dataset("docsqa")
    gctx = P.GroupContext([], "", [], {})
    kids = [P.Child(f"c{i}", "d", i, f"Alpha number {i}.", "", "T", "g", 0) for i in range(2)]
    q = "- What is Alpha?\n- Why Alpha?\n- Where is Alpha?"
    llm = FakeLlm([q, q])
    await P.summarise_group(llm, profile, gctx, kids, mode=Z.QUESTIONS)  # type: ignore[arg-type]
    assert all("<previous>" not in user for _s, user in llm.seen)
    llm = FakeLlm(["- Alpha one.\n- Alpha two.", "- Alpha three.\n- Alpha four."])
    await P.summarise_group(llm, profile, gctx, kids)  # type: ignore[arg-type]
    assert "Alpha one." in llm.seen[1][1]  # v1: the second chunk's prompt carries the first summary


def test_formulaic_openings_are_stripped_from_context_lines() -> None:
    assert Z.strip_opening("This passage describes gradient ascent for photon storage.") == (
        "Gradient ascent for photon storage."
    )
    assert Z.strip_opening(
        "This passage from the Harvard PhD thesis explains how spin waves decay."
    ) == ("Spin waves decay.")
    assert (
        Z.strip_opening("This appendix in the thesis derives adjoint equations.")
        == "This appendix in the thesis derives adjoint equations."
    )
    assert (
        Z.strip_opening("Appendix F derives adjoint equations.")
        == "Appendix F derives adjoint equations."
    )
    assert (
        Z.CONTEXT_V2.clean_openings and Z.CONTEXT_V2.allow_parent and not Z.CONTEXT.clean_openings
    )


def test_context_v2_takes_no_synopsis_and_may_use_the_parent_text() -> None:
    ctx = Z.Context(
        title="Doc", profile=profile_for_dataset("docsqa"), text="Alpha uses the cache.",
        synopsis="Zeta Cluster overview", glossary_names=("Alpha",), parent_context="Beta precedes it.",
    )  # fmt: skip
    user = Z.build_user_message(ctx, Z.CONTEXT_V2)
    assert "Zeta Cluster" not in user and "Beta precedes it." in user
    assert "Beta" in Z.support_text(ctx, user, Z.CONTEXT_V2)
    assert "Beta" not in Z.support_text(ctx, user, Z.SUMMARY_V2)


# ------------------------------------------------------------------------------------------ pipeline
def test_groups_spans_and_chunk_positions_for_a_docs_corpus() -> None:
    corpus = {
        "docs/A.md#000": {
            "title": "A > intro",
            "text": "Intro text. " * 10,
            "title_in_content": "0",
        },
        "docs/A.md#001": {"title": "A > body", "text": "Body text. " * 10, "title_in_content": "0"},
        "docs/B.md#000": {"title": "B", "text": "Other file. " * 10, "title_in_content": "0"},
    }
    groups = P.build_groups("docsqa", corpus, profile_for_dataset("docsqa"))
    assert set(groups) == {"docs/A.md", "docs/B.md"}
    a = groups["docs/A.md"]
    assert a.doc_ids == ["docs/A.md#000", "docs/A.md#001"]
    assert [s.pos for s in a.spans] == [0, 1]
    assert P.span_position(a, "docs/A.md#001", "Body text. Body text. ") == 1
    assert P.span_position(a, "docs/A.md#000", "text that is not in the document at all") == 0


def test_causal_groups_use_small_passages() -> None:
    corpus = {
        f"p{n:04d}": {"title": f"p{n}", "text": ("page text. " * 400), "title_in_content": "0"}
        for n in (2, 1)
    }
    groups = P.build_groups("pdf-got", corpus, profile_for_dataset("pdf-got"))
    assert list(groups) == ["pdf-got"] and groups["pdf-got"].doc_ids == ["p0001", "p0002"]
    assert all(len(s.text) <= P.SPAN_CHARS_CAUSAL for s in groups["pdf-got"].spans)
    assert [s.pos for s in groups["pdf-got"].spans] == list(range(len(groups["pdf-got"].spans)))


def test_child_ordinals_follow_the_adapter_naming() -> None:
    ns = uuid.UUID("6b1c1f0e-6a52-4a7e-9d1e-0c0de0c0ffee")
    ids = [str(uuid.uuid5(ns, f"docs/A.md#000|child|{n}")) for n in (2, 0, 1)]
    assert P.child_ordinals("docs/A.md#000", ids, ns) == {ids[0]: 2, ids[1]: 0, ids[2]: 1}
    with pytest.raises(RuntimeError):
        P.child_ordinals("docs/A.md#000", [str(uuid.uuid4())], ns)


def test_embed_text_uses_the_summary_only_when_it_is_usable() -> None:
    assert P.embed_text("- s", "ok", "child") == "- s\n\nchild"
    assert P.embed_text("- s", "retried", "child") == "- s\n\nchild"
    assert P.embed_text("- s", "fallback", "child") == "child"
    assert P.embed_text(None, "ok", "child") == "child"
    assert P.embed_text("- s", "ok", "child", recipe="suffix") == "child\n\n- s"
    assert (
        P.embed_text("- s", "ok", "two words", min_words=5) == "two words"
    )  # too short for the rule
    assert P.embed_text("- s", "skipped", "child") == "child"


def test_doc_content_matches_the_adapter_title_rule() -> None:
    assert P.doc_content({"title": "T", "text": "body"}) == "T\n\nbody"
    assert P.doc_content({"title": "T", "text": "body", "title_in_content": "0"}) == "body"
    assert P.doc_content({"title": "", "text": "body"}) == "body"


def test_reason_kind_buckets() -> None:
    assert P.reason_kind("not in the source or context: 99") == "terms_not_in_source"
    assert P.reason_kind("1 lines, 0 bullets (need 2 to 4 bullets and nothing else)") == "format"
    assert P.reason_kind("104 words") == "too_long"


async def test_group_context_and_child_summaries_end_to_end_with_a_fake_model() -> None:
    corpus = {
        "docs/A.md#000": {
            "title": "A",
            "text": "Alpha is the main component. It is optional.",
            "title_in_content": "0",
        }
    }
    profile = profile_for_dataset("docsqa")
    group = P.build_groups("docsqa", corpus, profile)["docs/A.md"]
    extract = json.dumps(
        {"entries": [{"names": ["Alpha"], "kind": "component", "definition": "The main component.",
                      "status": "", "quote": "Alpha is the main component", }]}
    )  # fmt: skip
    llm = FakeLlm(
        [extract, "A short synopsis.", "- Alpha is optional.\n- Alpha is the main component."]
    )
    gctx = await P.build_group_context(llm, profile, group, short=False)  # type: ignore[arg-type]
    assert [e.names for e in gctx.entries] == [["Alpha"]] and gctx.synopsis == "A short synopsis."
    child = P.Child(
        "c1",
        "docs/A.md#000",
        0,
        "Alpha is the main component. It is optional.",
        "",
        "A",
        "docs/A.md",
        0,
    )
    results = await P.summarise_group(llm, profile, gctx, [child], use_previous=True)  # type: ignore[arg-type]
    assert results[0].status == "ok" and results[0].glossary_entries == 1
    assert "- Alpha (component): The main component." in llm.seen[-1][1]
    assert "<synopsis>A short synopsis.</synopsis>" in llm.seen[-1][1]


# ------------------------------------------------------------------------------------------ recipes
def test_recipe_texts_use_only_usable_summaries_and_build_controls() -> None:
    kids = [R.ChildText(f"c{i}", f"d{i}", None, f"chunk {i} " + "word " * 120) for i in range(3)]
    enrich = {
        "e": {
            "c0": ("- sum 0\n- Unclear: nothing", "ok"),
            "c1": ("- sum 1", "fallback"),
            "c2": ("- sum 2", "ok"),
        }
    }
    flags = {"c0": {"borrowed": True}, "c1": {"borrowed": False}, "c2": {"borrowed": False}}
    r = R.text_recipes(kids, enrich, flags, "e")
    assert r["fast"] == [k.content for k in kids]
    assert (
        r["e:prefix"][0].startswith("- sum 0") and r["e:prefix"][1] == kids[1].content
    )  # fallback: chunk alone
    assert r["e:suffix"][2].endswith("- sum 2")
    assert "Unclear" not in r["e:prefix_no_unclear"][0] and r["e:prefix_first_bullet"][
        0
    ].startswith("- sum 0\n\n")
    assert r["e:prefix_if_clean"][0] == kids[0].content  # borrowed: not used
    assert r["e:prefix_if_clean"][2].startswith("- sum 2")
    assert r["control_duplicate"][1] == f"{kids[1].content}\n\n{kids[1].content}"
    assert r["control_shuffled"][0] != r["e:prefix"][0]  # another chunk's text, not its own
    assert R.strip_unclear("- a\n- Unclear: b") == "- a" and R.first_bullet("- a\n- b") == "- a"


async def test_synopsis_pass_is_skipped_for_modes_whose_prompt_has_no_synopsis_slot() -> None:
    corpus = {
        "docs/A.md#000": {
            "title": "A",
            "text": "Alpha is the main component.",
            "title_in_content": "0",
        }
    }
    profile = profile_for_dataset("docsqa")
    group = P.build_groups("docsqa", corpus, profile)["docs/A.md"]
    extract = json.dumps({"entries": []})
    llm = FakeLlm([extract])
    gctx = await P.build_group_context(llm, profile, group, short=False, need_synopsis=False)  # type: ignore[arg-type]
    assert gctx.synopsis == "" and llm.calls == 1  # only the glossary pass ran
    assert "{synopsis}" in Z.CONTEXT.user and "{synopsis}" not in Z.CONTEXT_V2.user
