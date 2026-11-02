# Coreference and entity graph retrieval

## What actually connects a far away name to a pronoun or description

### Takeaway

Prioritize a document scoped, evidence backed identity and mention index, then use it as a candidate retrieval channel. A name index alone cannot retrieve a page containing only a pronoun: the missing piece is a justified mention to identity link. Preserve raw text, page and span coordinates; add retrieval annotations beside it.

### Cited Findings

- BookNLP recognizes and clusters names, links pronouns, and exports token/mention annotations. Crucially, its default coreference mode deliberately prevents common noun phrases (the old man) from linking to named characters, because unconstrained full coreference often merges distinct entities. `pronominalCorefOnly=False` enables those links, with an explicit warning to inspect results. Its published ~79 average coreference F1 is not evidence of reliable book length resolution. English is the supported language. The project also supplies a name/pronoun prior; stricter own document use can disable this prior. — [BookNLP README, method owner source](https://github.com/booknlp/booknlp#character-name-clustering-and-coreference)

- **Strongest directly relevant 2025 source:** BookCoref evaluates whole books. Its human gold set is THREE canonical books (Animal Farm, Pride and Prejudice, Siddhartha), average 76,419 tokens; 50 silver training books average 216,626. Third party evaluation of off the shelf BookNLP gives **42.2 CoNLL F1**, versus Longdoc 46.6; Longdoc trained on the silver corpus reaches 67.0. The annotation pipeline obtains 80.5, using a supplied character list, character linking, **Qwen2 7B yes/no validation**, Maverick expansion in **1,500 word** windows and groups of 10 windows. It relies on external Wikidata/LiSCU character lists. Window encoding with global clustering was tested on one RTX4090 24GB; reported maximum memory includes BookNLP 6.9GB, Longdoc 5.8GB, Maverick XL 14.8GB. Full scope remains English characters and only three manually annotated test books; this is not retrieval evaluation. — [BookCoref ACL 2025, §§3–5, Tables1–4](https://aclanthology.org/2025.acl-long.1197/), [full text](https://arxiv.org/html/2507.12075v1)

- A separate 2025 study resolves document coreference before embedding. Appendix B tests **Qwen2.5 7B**, not just its main GPT4o-mini setup. On BELEBELE, SQuAD2, BoolQ and NanoSCIDOCS, Stella mean pooling average nDCG@5 rises .803→.818, while BGE CLS .799→.798. Other decoder embeddings have small mixed changes. Tests used an A6000 48GB; datasets are short passages, not books, Gemini was not tested, and this is author evidence. It supports targeted testing, not a universal expected gain. — [Jang et al 2025, appendixB Table4](https://arxiv.org/html/2507.07847v1#A2)

### Inferences

- Recommended adaptation: build **quoted own document names/aliases**, resolve locally where the text supports an antecedent, then stitch windows using explicit shared identities. Do not ask a 2B model to directly infer every pronoun's antecedent across an entire 750 page book. A 100 page retrieval separation often requires local coreference plus a global identity chain, rather than a single 100 page anaphoric jump.
- BookCoref is particularly well matched to the available GPU and Qwen budget, but it is not a drop in guarantee. Replacing its supplied character list with the user's unreliable glossary is a new error source. Use the pipeline as a design reference or test available pretrained long document models; do not propose fine tuning the full system as the first few dollar experiment.
- An exact quote verifies that a name or description occurs. It does **not** verify that 'the old man' and a particular name refer to the same individual. Require evidence for the relation, preserve ambiguous alternatives, and abstain when the local discourse does not justify a unique link. A Qwen 'yes' or self reported confidence is not calibrated confidence.
- Keep `(document_id, document_version, entity_id, mention_span, page_id, chunk_id, link_type, antecedent/evidence_spans, model_version, score/status)` as sidecar records. Document scoped IDs prevent unrelated people with the same name from merging. Annotation spans must refer to the original extracted text, not offsets in rewritten text.
- Runtime: identify the named entity in the query, retrieve its mention linked pages, filter/rank within that candidate set using the original question's event/property terms, then union with ordinary hybrid candidates. This adds an explicit path to the unnamed page. Avoid fetching all pages containing a main character without a query relevance budget.
- BookNLP/Maverick are supervised encoder NLP models, not note generating Google models. They satisfy the intent of local open processing if the user permits non generative coreference models; the Qwen verification component is within 2B–9B. No new provider or graph database is required.
- Cost is one local NLP/coreference pass plus selective Qwen verification, then ordinary SQL lookups. **No extra embedding API request is necessary for the lookup channel.** Index storage grows with mention count. Extra Google vectors are needed only if embedding rewritten aliases or sidecars; those are an optional separate ablation. Published timings cannot establish the user's actual GPU rental cost; measure tokens/sec and total GPU hours on a held out sample.
- Main risks: cluster collapse merging different people; character aliases that depend on narrator/time; possessive/nested entities; mistaken speakers; unanchored windows; OCR page/token mapping loss. In financial/medical/legal/policy texts, scope by company/report period, patient/encounter, contract party, defined term, section version and time. Resolve organizations, procedures and defined terms, not only human names. A universal literature character model is not proved for those domains.

### Gaps

- No inspected source establishes high all evidence page recall for 100+ page gaps with this user's Gemini model, 2B–9B extraction constraint and budget.
- No measured guarantee exists that a supplied quote backed alias registry transfers BookCoref's 80.5 score to arbitrary domains. BookCoref's gold results cover only three famous canonical books and should be described with that limitation.
- BookNLP default mode will miss exactly the user's common description case; full mode can increase false links. Treat both as test arms, not an installation fix.

## Which graph methods are justified for this use case

### Takeaway

A small entity to passage graph or relational mention table is the most plausible first structural change. If further association is needed, add bounded graph expansion or PPR over verified links while keeping raw passage retrieval. Full community summary GraphRAG is a poor first spend for precise paired pages.

### Cited Findings

- Original Microsoft GraphRAG extracts entities and relations, summarizes them, clusters graph communities, generates community summaries and uses map reduce for corpus level sensemaking. Its original experiments focus on podcasts/news, comprehensiveness/diversity and LLM preference, not exact page pair recall. Domain specific extraction examples are recommended in the original method for medical/legal/scientific material. — [Edge et al 2024, §§2–3](https://arxiv.org/html/2404.16130v1)
- LightRAG uses LLM extraction, entity/relation profiles, deduplication, low level entity and high level topic retrieval plus vector search. It does not by itself prove reliable long range pronoun identity; that remains upstream extraction/linking. — [Guo et al 2024, §§3.1–3.2](https://arxiv.org/html/2410.05779v1)
- HippoRAG uses extracted triples and personalized PageRank; HippoRAG2 retains passage nodes and adds query to triple linking and an LLM recognition filter, returning original passages rather than depending solely on generated graph summaries. Authors report average answer F1 59.8 vs strong dense NV Embed v2 57.0 over seven benchmarks, including NarrativeQA. Main extraction/reading uses Llama3.3 70B or GPT4o-mini, outside the user's enrichment limits; the encoder is swappable but Gemini/small extractor replication is unproved. On 11,656 MuSiQue passages, reported indexing input/output tokens are 9.2M/3.0M for HippoRAG2, 68.5M/18.3M LightRAG and 115.5M/36.1M GraphRAG (author comparison, model/hardware dependent). — [HippoRAG2 ICML2025 paper, §§3–5, appendixF](https://arxiv.org/html/2502.14802v1)
- **Independent method comparison with relevant domains:** GraphRAG Bench (2025, revised February2026) uses Gutenberg novels and NCCN medical guidelines. In its novel complex reasoning task, reranked RAG evidence recall=64.47 vs HippoRAG87.91 and HippoRAG269.77; relevance=82.08, 58.75 and85.75 respectively. On medical complex reasoning, reranked RAG recall86.49 exceeds HippoRAG277.00. On one ~56k token book, reported construction tokens: HippoRAG2~330k, LightRAG~474k, MS GraphRAG~655k. These are evidence element/semantic metrics and generated questions, **not exact 100 page pair recall**. AppendixG.4 3B–14B Qwen results vary the **generation model**, so do not cite them as proof that a 7B extraction pipeline matches stronger extraction. — [GraphRAG Bench, Table3 and appendixG Tables10–11](https://arxiv.org/html/2506.05690v3)
- **Independent criticism of marketing comparisons:** Zeng et al show unrelated generated questions and answer order/length/trial biases. With length alignment, order swaps, repeats and ties, LightRAG's Agriculture win rate vs naive RAG falls from author reported66.70% to39.06%; many ties remain. The revised source is August2026. This is answer preference evidence, not pronoun retrieval recall. — [Unbiased GraphRAG evaluation, §VI](https://arxiv.org/html/2506.06331v2)
- **Cheaper alternative to investigate:** LinearRAG (2025, latestv4 November2025) constructs entity/sentence/passage nodes using lightweight NER and semantic linking; no LLM relation extraction. Its two stages activate relevant entities through semantic bridging, then aggregate passage relevance using PPR. Authors report four dataset improvements, including MuSiQue and NarrativeQA; no independent reproduction found here. It needs additional entity and sentence embeddings, which can in principle be ordinary Google text vectors, increasing indexed text/calls. It does not resolve unnamed pronouns automatically. — [LinearRAG §§3–4](https://arxiv.org/html/2510.10114v4)
- SPRIG (2026 preprint) tests CPU entity/passage graph PPR; RRF seeded graph recall@10 .867 vs RRF .851 on HotpotQA and .794 vs .697 on2Wiki, but costs more query CPU. Crucially the paper explicitly **does not perform full entity linking or coreference**, tests only two Wikipedia QA sets, tunes validation subsets and lacks head to head runs against LinearRAG/HippoRAG. Good transparent design inspiration; weak transfer evidence for books. — [SPRIG §§5–9](https://arxiv.org/html/2602.23372v1)

### Inferences

- A graph cannot recover a correct entity–pronoun link absent from graph construction. Graph traversal is useful after identity extraction, not a substitute for it.
- Implement entity/passage adjacency inside PostgreSQL, optionally copy a bounded per document adjacency list to memory for PPR. Neither these conceptual algorithms nor the constraint requires Neo4j, a separate managed vector store, or token level embeddings. A wholesale library migration would be unnecessary to test the hypothesis.
- Start with exact mention lookup and one bounded expansion; add PPR only if verified relationships measurably improve candidates. Co-occurrence edges should mean 'occurs together', never a fabricated causal/identity relation. Cap high degree hubs and preserve original dense/keyword candidates.
- HippoRAG2 is the most relevant full graph design because it retrieves original passages and preserves a dense fallback. Replacing its extractor/filter with local7B is an experimental adaptation; published59.8 cannot be promised. Broad GraphRAG/LightRAG adds more generated intermediates on top of a glossary already having low precision, amplifying the error cascade.
- Coreference/identity is different from legal cross references, financial period/entity joins, or policy exceptions. Deterministic links such as an explicit section reference/defined term are often cheaper and more auditable than open relation extraction. A typed evidence graph can combine those links, but no universal extractor is established across every document type.

### Gaps

- Independent benchmark evidence is mixed and dependent on domains, generators and evaluation design. It does not establish universal graph superiority or a few dollar local small model run.
- GraphRAG Bench's own questions are generated via structured ontology/evidence packages; results can favor structured methods. Its semantic evidence measures and answer judges differ from the user's exact page coverage requirement. Do not relabel its recall figures as AllPages@10.
- No inspected source independently evaluates LinearRAG or SPRIG on the exact name/pronoun +100 page case.

## Practical first experiment and what should count as proof

### Takeaway

Isolate identity resolution quality before paying for a graph or reembedding the corpus. A small manually verified mention index is an oracle diagnostic; a production automatically extracted one is a separate test.

### Cited Findings

- The repository already embeds `summary_text + raw_content` in summary ingestion and resolves reranked children to parents; its architecture explicitly postpones entity graph work until a real entity heavy multihop use case exists. Context is expanded only after candidates are retrieved. These properties explain why parent expansion cannot summon a different page a hundred pages away that was never a candidate. — [summary pipeline](/Users/proximus/Documents/antigravity/RAG/src/deep_context/ingestion/summary_pipeline.py:151), [retrieval engine](/Users/proximus/Documents/antigravity/RAG/src/deep_context/retrieval/engine.py:216), [architecture](/Users/proximus/Documents/antigravity/RAG/docs/ARCHITECTURE.md:449)

### Inferences

1. First build an evaluation only oracle identity map for a small held out batch, based solely on document quotes and the annotator's verified discourse links. Retrieve raw pages through entity mentions, keep final page/token budget matched. If this does not raise all required page coverage, identity is not the only bottleneck; inspect event matching, candidate quota, page deduplication and reranker removal.
2. Next compare **automatic** BookNLP pronoun mode, full coref mode, and selective local Qwen7B/Maverick style links on the same questions. This is the actual feasibility test. An oracle score must be clearly separated and cannot be advertised as system performance.
3. Audit a stratified sample of links: explicit names/aliases, pronouns, nominal descriptions, coreference distance, entity frequency, domain and ambiguity. Report precision, recall and abstention/coverage. Quote support rate55–85% is not coreference accuracy and not enough for identity correctness. Select confidence thresholds from held out human labels, not Qwen self reported confidence. Prefer rejecting dubious links to infecting thousands of pages through transitivity.
4. Keep the original dense/keyword index. New entity candidates should have a quota so globally popular characters cannot fill the entire pool. Deduplicate by page before measuring@10; compare fixed final unique page budgets and fixed total prompt tokens. Measure every stage, including before/after reranking, since a text only reranker can downscore a correct raw pronoun page unless it sees the **verified identity sidecar** as retrieval metadata.
5. Only after the automatic link index improves true page pair recall, test rewriting a small sidecar as 'Resolved entity: X. Source assertion: ...', or embedding standalone facts as separate child retrieval records. Never overwrite raw text or cite the manufactured paraphrase as if it were the original; carry the evidence span and page(s). Distinguish source page of event assertion from the page used only to justify identity.
6. Escalate to bounded graph/PPR only when failure analysis shows meaningful multi entity/relationship paths beyond same entity lookup. Compare against a bounded two round retrieval baseline with query/event decomposition; additional query calls often cost less than exhaustive offline relation extraction. This is design reasoning, not a demonstrated result for the current repository.

### Gaps

- Nothing was run against live Google/model APIs, no GPU timing or price claim is established, and no code was edited. Existing worktree changes were preserved.
- This researcher read relevant repository docs/source, not all private documents or generated artifacts. Other researchers own comprehensive repository and eval coverage. Public research search queries contained no private document content or credentials.
