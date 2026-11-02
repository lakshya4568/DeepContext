"""Variant B: contextual child-chunk summaries built from a document glossary.

Spec: ``eval/variant_b/GLOSSARY_SPEC.md`` (profiles, glossary entry, B0-B6) and ``GCP_GPU_PLAN.md`` (GPU host).

* B0 profile      : ``profiles.py`` (fixed mapping for the benchmark datasets)
* B1/B2 glossary  : ``glossary.py`` (extract with evidence, verify, merge)
* B3 synopsis     : ``synopsis.py`` (whole-document, or a "story so far" for causal profiles)
* B4/B5 summaries : ``summarize.py`` (prompt with context slots, grounding check, retry, fallback)
* B6 index        : ``pipeline.py`` (derived Postgres schema; child vectors embed ``summary + child``)

Nothing here touches ``src/``; the summariser is any OpenAI-compatible server (``llm.py``).
"""
