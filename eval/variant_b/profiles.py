"""Document profiles for variant B (GLOSSARY_SPEC.md section 3) and the benchmark dataset mapping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    mode: str  # "static": the whole document may be used; "causal": a chunk only sees what precedes it
    rule: str  # one line put into every summary prompt
    kinds: tuple[str, ...]  # glossary entry kinds to look for
    capture: str  # what else to note in an entry's definition when the text states it


def _p(name: str, mode: str, rule: str, kinds: tuple[str, ...], capture: str) -> Profile:
    return Profile(name, mode, rule, kinds, capture)


PROFILES: dict[str, Profile] = {
    p.name: p
    for p in (
        _p(
            "fiction",
            "causal",
            "Fiction: say who acts or speaks (name them using the glossary), keep deaths, losses and "
            "changes of allegiance, and nothing that happens later.",
            ("person", "group", "place", "object", "event"),
            "role, relationships, alive or dead or missing, allegiance, location",
        ),
        _p(
            "narrative_nonfiction",
            "causal",
            "Narrative non-fiction: keep who did what, when, and who is the source of each claim.",
            ("person", "organisation", "place", "event", "date"),
            "role, dates, cause and effect, who claims what",
        ),
        _p(
            "science_paper",
            "static",
            "Science paper: keep symbols and abbreviations as written with their meaning, units, sample "
            "sizes, and say whether the passage states a claim, a method, a result or a limitation.",
            ("concept", "method", "dataset", "metric", "symbol", "abbreviation", "result"),
            "symbol meaning, units, sample size, baseline compared, figure or table or equation referred to",
        ),
        _p(
            "thesis_book",
            "static",
            "Thesis or book: keep notation and defined terms as written, say what a theorem or result "
            "assumes, and resolve 'this result' or 'the above' using the glossary.",
            ("concept", "definition", "theorem", "notation", "algorithm", "experiment", "chapter"),
            "notation meaning, assumptions of a theorem, cross-references",
        ),
        _p(
            "medical",
            "static",
            "Medical: keep negation and uncertainty ('no', 'ruled out', 'possible'), population, dose, "
            "comparator and time point; name both arms when two are compared.",
            (
                "condition",
                "drug",
                "procedure",
                "population",
                "outcome",
                "measure",
                "study_arm",
                "code",
            ),
            "dose, route, frequency, duration, comparator, sample size, time point, units",
        ),
        _p(
            "finance_report",
            "static",
            "Finance: whenever the passage has a number, keep the entity, fiscal period, unit or scale, "
            "currency and basis (GAAP or non-GAAP); never give a bare figure.",
            ("company", "segment", "metric", "period", "instrument", "programme"),
            "fiscal period, scale (thousands or millions), currency, GAAP or non-GAAP, restated or not",
        ),
        _p(
            "accounts",
            "static",
            "Accounts: keep account, period, currency, sign (debit or credit) and reference numbers with "
            "every amount; say what an adjustment adjusts.",
            ("account", "cost_centre", "counterparty", "document", "period", "adjustment"),
            "account code, debit or credit, period, currency, amount sign, basis, reference number",
        ),
        _p(
            "legal_contract",
            "static",
            "Legal contract: keep who must, may or must not do what, by when, under which condition and "
            "exception; name parties by their defined terms and never drop an exception.",
            ("party", "defined_term", "clause", "obligation", "condition", "exception"),
            "party role, effective and termination dates, governing law, carve-outs, amendments",
        ),
        _p(
            "legal_case_statute",
            "static",
            "Case law or statute: keep the jurisdiction, citation and version in force, and say whether "
            "the passage states a holding, an argument or a rule.",
            ("court", "party", "statute", "section", "holding", "test"),
            "jurisdiction, date or version in force, citation, repealed or amended",
        ),
        _p(
            "policy",
            "static",
            "Policy: keep who it applies to, the strength of each requirement (shall, should, may), "
            "dates, version and exceptions.",
            (
                "scope",
                "role",
                "requirement",
                "exception",
                "definition",
                "penalty",
                "related_policy",
            ),
            "applies-to, effective and sunset dates, version, strength, owner",
        ),
        _p(
            "technical_docs",
            "static",
            "Technical docs: keep component names, config keys, defaults, versions, and whether "
            "something is required, optional or deprecated.",
            ("component", "config_key", "command", "version", "prerequisite", "feature"),
            "default value, unit, required or optional, deprecated or removed in a version, environment",
        ),
        _p(
            "news",
            "static",
            "News: keep the outlet, date, who said each quote, and whether something is a reported "
            "claim or an established fact.",
            ("outlet", "person", "organisation", "event", "quote", "claim"),
            "publication date, author, who said each quote, claim versus fact, event date",
        ),
        _p(
            "qa_forum",
            "static",
            "Q&A forum: say what is being asked or advised, keep products, amounts and jurisdiction, and "
            "mark opinion versus fact.",
            ("question", "answer", "product", "term", "advice"),
            "topic, country or jurisdiction, date, opinion versus fact",
        ),
        _p(
            "encyclopedic",
            "static",
            "Encyclopedia article: always name the article's subject, and keep dates, places and "
            "relationships between entities.",
            ("subject", "related_entity", "date", "list_item"),
            "subject type, also known as, dates",
        ),
        _p(
            "conversation",
            "causal",
            "Conversation: keep who said what, decisions, and action items with owners and due dates.",
            ("speaker", "topic", "decision", "action_item"),
            "speaker, date, owner, due date",
        ),
        _p(
            "tabular",
            "static",
            "Table: summarise with the column headers and units, never with bare cell values.",
            ("table", "column", "row_key", "unit"),
            "caption, headers, units, footnotes",
        ),
        _p(
            "generic",
            "static",
            "Keep names, numbers with units, dates and defined terms exactly as written.",
            ("entity", "term", "abbreviation", "date"),
            "",
        ),
    )
}

# GLOSSARY_SPEC.md section 4.
DATASET_PROFILE: dict[str, str] = {
    "scifact": "science_paper",
    "fiqa": "qa_forum",
    "hotpotqa": "encyclopedic",
    "multihoprag": "news",
    "ragbench-covidqa": "medical",
    "docsqa": "technical_docs",
    "pdf-got": "fiction",
    "pdf-quantum": "thesis_book",
}

# One document is about one chunk here, so B2 (merge) and B3 (synopsis) are skipped and B1 runs once per document.
SHORT_DOC_DATASETS = frozenset({"scifact", "fiqa", "hotpotqa", "ragbench-covidqa"})


def profile_for_dataset(dataset: str) -> Profile:
    """B0 for the benchmark corpora: a fixed mapping (a model-based classifier is not needed here)."""
    return PROFILES[DATASET_PROFILE.get(dataset, "generic")]


def group_and_order(dataset: str, ext_doc_id: str) -> tuple[str, int]:
    """Which source document a corpus document belongs to, and its position inside it.

    Project docs: sections ``docs/X.md#003`` belong to file ``docs/X.md``. PDFs: pages ``p0017`` belong to the
    one PDF. Everything else is its own document.
    """
    if dataset == "docsqa":
        file, _, n = ext_doc_id.rpartition("#")
        return (file, int(n)) if file and n.isdigit() else (ext_doc_id, 0)
    if dataset.startswith("pdf-"):
        digits = ext_doc_id.lstrip("p")
        return dataset, int(digits) if digits.isdigit() else 0
    return ext_doc_id, 0
