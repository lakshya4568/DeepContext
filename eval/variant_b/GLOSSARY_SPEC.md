# Variant B — contextual summary: glossary spec and architecture

Status: **specification only. Nothing here is implemented or run.** Variant A (the Qwen3-style recipe: title + parent
snippet of at most 1,200 characters) is parked. Variant B is the default for the RAG evaluation. Whatever model writes the
summaries (a real SLM, Claude Haiku as a stand-in, or another LLM) must be named in every report as a separate factor;
a Haiku-written summary is "SLM simulated by Haiku", never "SLM".

## 1. Principles

1. **Text only.** Every glossary entry and every summary uses only the document being indexed. No world knowledge
   (the model may know Game of Thrones or a famous theorem; the index must not).
2. **Evidence-bound.** Each entry carries a short quote or a location (page, section, chunk id) that supports it.
   No quote, no entry.
3. **Time-scoped.** Entries record *where* in the document each fact becomes true. A chunk only sees facts at or before its
   own position when the profile is `causal` (stories, records, transcripts). `static` profiles (papers, contracts,
   manuals) may use the whole document.
4. **Capped.** At most 8 entries and about 350 tokens of glossary per chunk. Longer context is worse for small models.
5. **Never use benchmark questions, answers or labels** to build a glossary or a summary.
6. **Labelled.** Every stored summary records profile, variant, summariser model, prompt version and status.

## 2. Universal glossary entry

| Field | Meaning |
|---|---|
| `id` | Stable id inside the document (`g_017`). |
| `surface_forms` | Names, aliases, abbreviations, symbols, pronoun-resolvable descriptions ("the Hand of the King"). |
| `kind` | One of the kinds in the profile (person, term, symbol, metric, party, drug, account, component …). |
| `definition` | One sentence, taken from or directly supported by the text. |
| `attributes` | Profile-specific key/values (units, period, dose, jurisdiction …). |
| `timeline` | List of `{at, state}`; `at` is a position (page / section / chunk id), `state` a short change ("executed", "restated", "deprecated", "amended"). |
| `relations` | Links to other entries (`employs`, `defined_in`, `amends`, `refers_to`). |
| `evidence` | Quote of at most 20 words plus its location. |
| `confidence` | `high` (defined explicitly) or `inferred` (resolved from context). `inferred` entries are not allowed to introduce new facts. |

## 3. Document profiles

Each profile = the universal entry + the extra kinds and attributes below. A document gets one primary profile (and may
get a secondary one, for example a medical thesis = `thesis` + `medical`). If nothing fits, use `generic` (title,
section path, defined terms, named entities only).

| Profile | Entry kinds to track | Attributes to capture | What it prevents | Mode |
|---|---|---|---|---|
| **fiction** (novels, stories, scripts) | characters, houses/groups, places, objects, events | role, relationships, alive/dead/missing, allegiance, location, POV character per chapter | "he refused" with no idea who "he" is; talking about someone who already died | causal |
| **narrative_nonfiction** (history, biography, memoir, long-form journalism) | people, organisations, places, events, dates | role, dates, cause→effect links, source of claim | undated or unattributed "then he resigned" | causal |
| **science_paper** (research paper, preprint, abstract) | concepts, methods, datasets, metrics, symbols, abbreviations, results | symbol→meaning, units, sample size, baseline compared, figure/table/equation targets, claim vs. result | "the proposed method", "as in Eq. 3", "it improves by 4%" without the baseline | static |
| **thesis_book** (dissertation, textbook, monograph) | chapters, definitions, theorems/lemmas, notation, algorithms, experiments | notation table, theorem number and its assumptions, chapter goal, prerequisite concepts, cross-references ("see Section 2.4") | notation drift between chapters; "this result" pointing to an earlier theorem | static, with a chapter synopsis |
| **medical** (clinical trials, guidelines, case reports, patient records) | conditions, drugs, procedures, populations, outcomes, measures, arms, codes | dose, route, frequency, duration, comparator, n, inclusion/exclusion, outcome and time point (baseline, week 12), units, ICD/MeSH-style codes if present, **negation and uncertainty** ("no history of", "ruled out", "possible") | mixing up arms, negated findings read as positive, ambiguous abbreviations ("MS") | static; **causal for patient records** |
| **finance_report** (annual/quarterly reports, filings, analyst notes, earnings calls) | companies, segments, metrics, periods, instruments, programmes | entity, ticker if given, **fiscal period**, **scale** (thousands/millions), **currency**, GAAP vs. non-GAAP, restated or not, YoY/QoQ, forward-looking flag, table headers | a bare "12.4" with no entity, period, unit or basis | static |
| **accounts** (ledgers, statements, invoices, budgets, audits, reconciliations) | accounts, cost centres, counterparties, documents (invoice/PO), periods, adjustments | account code and name, debit/credit, period, currency and rate, amount sign, basis (cash/accrual), reference numbers, approver, reconciling items | numbers detached from account, period or sign; "adjusted" without what was adjusted | static (ordered by period) |
| **legal_contract** (contracts, terms, NDAs, licences) | parties, defined terms, clauses, obligations, conditions, exceptions | party roles, effective/termination dates, governing law and jurisdiction, **who must/may/must not do what, by when**, carve-outs ("notwithstanding", "subject to"), amendment or supersession, cross-references | "the Licensee shall …" with no idea who that is; an exception read as the rule | static |
| **legal_case_statute** (case law, statutes, regulations text) | courts, parties, statutes/sections, holdings, tests | jurisdiction, date and version in force, citation, holding vs. dicta, who it binds, repealed/amended | quoting a repealed section; mixing holding and argument | static, with a version/date mark |
| **policy** (government policy, internal policy, standards, SOPs, guidelines) | scope, roles, requirements, exceptions, definitions, penalties | applies-to, effective/sunset dates, version, "shall/should/may" strength, owner, exceptions, related policies | a requirement without its scope or strength; outdated version | static |
| **technical_docs** (manuals, API and architecture docs, READMEs, runbooks) | components, config keys, commands, versions, prerequisites | default values, units, deprecated/removed in version, required vs. optional, environment, links to other docs | a setting without its component; deprecated advice | static |
| **news** (articles, press releases, reports) | outlets, people, organisations, events, quotes, claims | publication date, outlet, author, speaker of each quote, claim vs. fact, event date vs. publication date | quotes with no speaker; "last week" with no anchor date | static per article; time anchor from the date |
| **qa_forum** (Q&A, community posts, finance and health forums) | question, answer, asker intent, products/terms, advice | topic, accepted/top answer flag, jurisdiction or country, date, "opinion vs. fact" | an answer read without its question | static |
| **encyclopedic** (wiki-style articles, entity pages) | the article subject, related entities, lists | subject type, disambiguation line, dates, "also known as" | a paragraph that never repeats the subject's name | static |
| **conversation** (meeting notes, transcripts, emails, chat) | speakers, topics, decisions, action items | speaker, date, thread position, decision, owner, due date | "I agree" with no speaker or topic | causal |
| **tabular** (any table or structured block inside another document) | table, columns, row keys, units | caption, column headers, units, footnotes, source page | numbers without headers | follows the host document |

### Rules that matter most per profile

- **fiction / narrative_nonfiction / conversation / patient records (causal):** the document synopsis is a *"story so
  far"* up to the end of the previous section, never the whole book. A character's status in the glossary slice is the
  status *at that point*. Later events are not allowed in earlier summaries.
- **medical:** keep negation, uncertainty, population, dose and time point inside the summary. If a chunk mentions two arms,
  the summary names both. Abbreviations are expanded only when the document itself expands them.
- **finance_report / accounts:** the summary must carry **entity + period + scale + currency (+ basis)** whenever the chunk
  contains a number. If the chunk lacks one of them, take it from the glossary and mark it `from_context`.
- **legal / policy:** keep the actor, the modal verb (shall / may / must not), the condition and the exception together. A
  summary may not drop an exception.
- **science_paper / thesis_book:** keep symbols and abbreviations as written, plus what they stand for; say whether the chunk
  states a *claim*, a *method*, a *result* or a *limitation*.
- **tabular:** summarise with the column headers and units, not the bare cell values.

## 4. Mapping to the benchmark sets (for later runs)

| Benchmark set | Profile |
|---|---|
| SciFact | `science_paper` |
| FiQA | `qa_forum` (+ `finance_report` terms) |
| HotpotQA | `encyclopedic` |
| MultiHop-RAG | `news` |
| RAGBench CovidQA | `medical` (+ `science_paper`) |
| Project docs set | `technical_docs` |
| Game of Thrones PDF | `fiction` |
| Quantum thesis PDF | `thesis_book` |

## 5. Variant B architecture (ingestion only; retrieval is unchanged)

```
document ─▶ parse + parent/child chunking (fast path, unchanged)
              │
              ▼
   B0  PROFILE   classify document type from title, headings, first ~1,500 tokens
              │       → profile + mode (causal | static). Fallback: generic.
              ▼
   B1  EXTRACT   per section (map): candidate glossary entries with evidence quotes
              │
              ▼
   B2  MERGE     whole document (reduce): merge aliases, resolve pronoun/role references,
              │       order the timeline, drop entries without evidence → glossary store
              ▼
   B3  SYNOPSIS  static: one document synopsis (≤ 250 tokens)
              │       causal: rolling "so far" synopsis per section boundary
              ▼
   B4  PER CHILD summary prompt =
              │       profile rules + "use only the text below"
              │       + title › section path
              │       + synopsis (global or so-far)
              │       + glossary slice: ≤ 8 entries whose surface forms occur in the child
              │         (or its parent), filtered by position when causal, ≤ 350 tokens
              │       + parent snippet (≤ 1,200 chars)
              │       + summaries of the previous 2 children
              │       + the child text
              │       → 2–3 sentence self-contained summary (≤ 90 tokens)
              ▼
   B5  CHECK     every name, number and symbol in the summary must appear in the prompt;
              │       one retry, then fall back to the child's first sentences and flag it
              ▼
   B6  EMBED     text = summary + "\n\n" + child text  (same format as variant A, so only the
                      summary content differs) → embedding model unchanged → store child vectors
```

Query time, unchanged: embed the query → search child vectors (+ lexical channel on the child text) → merge → rerank →
return the **parent chunk** to the LLM.

### What gets stored (new, per index schema)

- `glossary` table: one row per entry (`doc_id`, fields from section 2).
- Child chunk extras: `summary_text`, and `summary_meta` JSON = `{variant: "B", profile, mode, glossary_ids,
  summariser_model, prompt_version, status: ok | retried | fallback}`.
- New schemas in `deepcontext_eval`; the fast-ingestion indexes are never modified.

### Cost note

B is heavier than A. Per child the prompt is roughly 1,200–1,500 tokens plus the child, against about 450 for A, and B1–B3 read
each document about twice more. The earlier token estimate (about 5.8M input for the five core sets) was for A and must be
redone for B before anything is run.

### Open decisions (owner)

1. Which model writes B1–B4 (SLM, Haiku stand-in, or another LLM).
2. Whether the summary is also added to the lexical (BM25) channel. Default here: **no**, embedding only, as in the
   existing deep-ingestion pipeline.
3. Dataset scope for the first B run.

## 6. Revision after the first pilot (2026-11-01)

The pilot (`PILOT_DOCS.md`) and its analysis (`FINDINGS_WHY_LOWER.md`) changed B4 and B6. Summary of the changes, all in code:

- **B4 has modes** (`summarize.MODES`). `summary_v1` is the original bullet summary and is kept only to reproduce the pilot. New:
  `summary_v2` (one or two plain sentences, no comments about what is missing), `context` (a situating context line of at
  most 40 words that says where the passage sits and what it is about, using title, synopsis and glossary), `questions`
  (three questions the passage answers) and `keywords`.
- **No chain of previous summaries** except in `summary_v1`: the chain and the document synopsis were the main sources of
  other chunks' words inside a chunk's summary (34% of v1 summaries).
- **Stricter grounding** for `summary_v2`, `questions` and `keywords`: names and numbers must come from the chunk, the title
  or the glossary names, not from the synopsis, neighbours or glossary definitions.
- **B6 takes a recipe**: where the generated text goes (`prefix` or `suffix`) and a minimum chunk length below which the chunk
  is embedded alone. Each mode builds its own schema (`_vb`, `_vbs2`, `_vbctx`, `_vbq`, `_vbkw`) beside the fast one.
- The reranker still reads only the chunk text. A study that also gives it the context line is in `rerank_context.py`.
