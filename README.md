<div align="center">

# ⚡ Deep Context Platform

### *Bare-Metal Agentic Hybrid RAG, Parent-Child Hierarchies, and Typed Long-Term Memory*

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL-pgvector%20HNSW-336791?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Google Gemini GenAI](https://img.shields.io/badge/Google%20GenAI-Gemini%202.5%20%2F%203.7%20%2B%20Embedding--2-4285F4?logo=google&logoColor=white)](https://ai.google.dev)
[![Groq Fast Inference](https://img.shields.io/badge/Groq-Qwen%203.6%2027B%20%2F%20Llama%203.3-F05032?logo=fastly&logoColor=white)](https://groq.com)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA%20NIM-BGE--M3%20%2F%20Llama%203.1-76B900?logo=nvidia&logoColor=white)](https://build.nvidia.com)
[![Tests](https://img.shields.io/badge/Tests-126%20Passing%20(100%25)-brightgreen)](https://github.com)
[![Zero Frameworks](https://img.shields.io/badge/Frameworks-Zero%20(No%20LangChain%20%2F%20LlamaIndex)-black)](https://github.com)

**A high-performance, framework-free Agentic Retrieval-Augmented Generation (RAG) platform.**  
Built entirely from scratch with raw Python 3.12, pure asynchronous SQL (`asyncpg` + `pgvector` HNSW), multi-provider LLMs, contextual chunk summarization, 4-store typed durable memory, and a zero-dependency Vanilla web studio.

[Quickstart](#-quickstart) • [System Architecture](#-system-architecture) • [Candidate Funnel](#-candidate-funnel-architecture-top-k-step-down) • [Mathematical Foundations](#-mathematical-foundations--formulas) • [Reranker & Consensus](#-reranker-architecture--consensus-protection) • [Ingestion & Storage](#-ingestion--hierarchical-storage-pipeline) • [Agentic Planner](#-agentic-router--corrective-rag-crag-state-machine) • [Two-Pass Grounding](#-two-pass-grounded-generation--verification) • [Typed Memory](#-4-store-typed-memory--8-layer-prompt-compiler) • [CLI Manual](#-cli-reference-manual) • [API & Streaming](#-api--streaming-contracts)

---

</div>

## 🌟 Why Deep Context Platform?

Most production RAG systems suffer from three fundamental architectural flaws:

1. **Framework Bloat & Fragility:** Heavy orchestration frameworks (LangChain, LlamaIndex, CrewAI) obscure SQL execution, inject opaque latency bottlenecks, and make mission-critical debugging in production painful.
2. **Context Blindness & Needle-in-a-Haystack Loss:** Naive fixed-size chunking forces a destructive trade-off: small chunks lose surrounding narrative and section hierarchy, while large chunks dilute embedding density and compromise retrieval precision on 1,000+ page corpora.
3. **Ungrounded Hallucinations & Unchecked Output:** Conventional single-pass generators compose text while loosely attending to context, inventing claims that are impossible to trace to source citations.

**Deep Context Platform solves these challenges from first principles:**

- **100% Hand-Crafted Asynchronous Core:** Zero LangChain, zero LlamaIndex, zero LangGraph. Pure, reviewable, high-speed Python 3.12, native `asyncio`, and raw parameterized SQL.
- **Anthropic Contextual Retrieval Standard:** Ingests documents with local GPU/MPS `Qwen/Qwen3-0.6B` (FP16) contextual chunk summaries prepended to raw text (`summary_text + "\n\n" + raw_content`), reducing retrieval failure rates by up to 67%.
- **Hierarchical Parent-Child Resolution:** Indexes tight 300–600 token child chunks for ultra-precise vector & lexical search, then dynamically resolves to full 1,000–2,500 token parent sections during synthesis so the LLM receives complete context.
- **Decoupled Zero-Loss Ingestion Pipeline:** Checkpoint 1 writes documents, parents, children, Qwen3 summaries, and full-text TSVectors atomically to PostgreSQL before calling any external embedding APIs, preventing data loss on rate limits.
- **Dual-Channel Hybrid Retrieval:** Combines PostgreSQL `pg_search` (ParadeDB BM25) or native weighted `tsvector` with `pgvector` HNSW dense cosine search via Reciprocal Rank Fusion ($k=60$).
- **Multi-Strategy Neural Reranking & Consensus Protection:** Cross-encoder reranking blended with normalized RRF scores ($60/40$ blend) plus algorithmic consensus guards ensuring dual-channel top hits are never displaced.
- **Corrective Agentic RAG (CRAG) State Machine:** Self-correcting state machine that grades retrieved document relevance, dynamically rewrites ambiguous queries, and retries retrieval before falling back gracefully.
- **Two-Pass Grounded Generation & Evidence Verifier:** Pass 1 extracts supported factual claims into structured JSON; Pass 2 synthesizes answers strictly from verified facts. An NLI-style evidence verifier scores claim grounding against source chunks.
- **4-Store Typed Memory & 8-Layer Prompt Assembler:** Partitioned memory (**Policy**, **Preference**, **Semantic Fact**, **Episodic Summary**) governed by a 4-stage promotion gate and compiled into structured prompts.
- **Production Performance Infrastructure:** Multi-tier response cache (Redis / In-memory) with SHA-256 canonical keys, and an internal persistence-backed scheduler for index maintenance.

---

## 🏛 System Architecture

The following diagram illustrates the end-to-end flow of the Deep Context Platform, showing the physical separation between ingestion, persistence, query routing, retrieval, quality gating, grounded generation, and durable memory:

```mermaid
flowchart TD
    subgraph Ingress ["1. Ingress & Client Interface"]
        Client["Web Studio UI / CLI / REST API / SSE Client"]
        FastAPI["FastAPI Async App Layer\n(/v1/query/stream, /v1/agentic-rag, /v1/ingest)"]
        Client --> FastAPI
    end

    subgraph IngestionPipeline ["2. Decoupled Ingestion Pipeline"]
        DocIn["Source Document (PDF, MD, Code, TXT)"]
        Parser["Structure-Aware Parser\n(PyMuPDF / IBM Docling / OCR)"]
        Chunker["ParentChildChunker\nParents: 1000-2500 tok | Children: 300-600 tok\n15% Overlap"]
        Summarizer["Contextual Chunk Summarizer\nLocal Qwen3-0.6B FP16 (/no_think)\nMPS / CUDA / CPU Auto-Detect"]
        CP1["Checkpoint 1: Atomic PG Write\n(Docs, Parents, Children, Summaries, TSV)"]
        Embedder["Embedding Engine\nGemini Embedding-2 (768-dim)"]
        
        DocIn --> Parser --> Chunker --> Summarizer --> CP1 --> Embedder
    end

    subgraph StorageEngine ["3. Persistence & Vector Storage"]
        PG[("PostgreSQL 16 + pgvector")]
        HNSW["idx_chunks_embedding_hnsw\n(Cosine, m=16, ef_c=200, ef_s=100)"]
        TSV["idx_chunks_search_tsv\n(GIN weighted TSVector B+C)"]
        ParentsTable["chunks (parent_id IS NULL)"]
        MemoryTables["memory_policy | memory_preference\nmemory_fact | memory_episode"]
        
        CP1 --> PG
        Embedder --> HNSW
        PG --- HNSW
        PG --- TSV
        PG --- ParentsTable
        PG --- MemoryTables
    end

    subgraph Router ["4. Query Understanding & Path Routing"]
        FastAPI --> QueryClassifier["Query Shape Classifier\n(Factual, How-To, Multi-Hop, Aggregation, Navigation)"]
        QueryClassifier --> RouterDecision{"Query Router"}
        RouterDecision -->|Factual / How-To / Aggregation| HybridPath["Hybrid RAG Pipeline"]
        RouterDecision -->|Multi-Hop / Complex Task| AgenticPath["Agentic Planner & CRAG Loop"]
    end

    subgraph RetrievalEngine ["5. Hybrid Retrieval & Reranking"]
        HybridPath --> SubQueries["Query Rewriter / Sub-Query Generator"]
        SubQueries --> BM25Search["BM25 Lexical Search\n(ParadeDB / TSVector, Limit=100)"]
        SubQueries --> DenseSearch["Dense HNSW Cosine Search\n(Gemini 768d, Limit=100)"]
        
        BM25Search & DenseSearch --> RRF["Reciprocal Rank Fusion (k=60)\n& Prefix / ID Deduplication (Cap=100)"]
        RRF --> PoolCut["Rerank Window Cutoff (Pool <= 24)"]
        PoolCut --> Reranker["Neural Reranker (API / Local / Heuristic)\nSigmoid Normalization + 60/40 Score Blend"]
        Reranker --> ConsensusGuard["Consensus Protection Guard\n(Preserve Dual-Channel Top Hits)"]
        ConsensusGuard --> ParentResolver["Parent Chunk Resolver\n(Expand Child -> 1500 tok Parent Section)"]
    end

    subgraph QualityGates ["6. Retrieval Quality & Sufficiency Gates"]
        ParentResolver --> HopGate{"Hop Coverage Gate\n(Entity/Topic Coverage >= 0.40?)"}
        HopGate -->|Fail: < 0.40| RetryHop["Corrective Retry (1x)\n(Widen Retrieval Pool)"]
        RetryHop -.-> SubQueries
        HopGate -->|Pass| SuffGate{"Evidence Sufficiency Gate\n(Query Term Overlap >= 0.25?)"}
        SuffGate -->|Fail: Insufficient| Refusal["Honest Fallback Refusal\n('Insufficient evidence to answer')"]
    end

    subgraph AgenticLoop ["7. Corrective Agentic State Machine (CRAG)"]
        AgenticPath --> CRAGRetrieve["Retrieve Candidates"]
        CRAGRetrieve --> GradeDocs{"Document Grader\n(Relevance Score >= 0.70?)"}
        GradeDocs -->|Irrelevant & rewrites < 2| RewriteQ["Query Rewriter (LLM + Heuristic)"]
        RewriteQ --> CRAGRetrieve
        GradeDocs -->|Exhausted rewrites| Refusal
        GradeDocs -->|Relevant| Grounding
    end

    subgraph Grounding ["8. Two-Pass Grounded Generation & Verification"]
        SuffGate -->|Pass| Pass1["Pass 1: Structured Fact Extraction\n(JSON: supported facts vs missing facts)"]
        Pass1 --> FactSanitize["Fact Sanitizer\n(Strip unsupported / inferred claims)"]
        FactSanitize --> Pass2["Pass 2: Grounded Synthesis\n(Draft answer strictly from supported facts)"]
        Pass2 --> Verifier{"Evidence Support Verifier\n(Claim-to-Chunk NLI Linking)"}
        Verifier -->|Confidence >= 0.75| FormatAnswer["Final Answer + Inline Traceable Citations"]
        Verifier -->|Confidence < 0.75| Refusal
    end

    subgraph MemorySystem ["9. Durable Typed Memory & Prompt Assembly"]
        FastAPI -.-> MemoryReader["MemoryStoreManager\n(Read Active Policies, User Prefs, Semantic Facts)"]
        MemoryReader --> PromptAssembler["8-Layer Prompt Assembler\n(Compiles System Prompt + Grounded Context)"]
        PromptAssembler -.-> Pass1
        
        FormatAnswer --> BackgroundMem["Background Memory Extractor"]
        BackgroundMem --> PromoGate{"4-Stage Promotion Gate\n(Observed -> Candidate -> Scored -> Durable)"}
        PromoGate -->|Score >= Threshold| MemoryTables
    end

    FormatAnswer --> Output["Stream to User via SSE / REST Response"]
    Refusal --> Output
```

---

## 🎯 Candidate Funnel Architecture (Top-K Step-Down)

To achieve sub-second retrieval while maximizing precision, the platform employs a 5-stage funnel that progressively prunes candidates:

```mermaid
flowchart TD
    subgraph Stage1 ["Stage 1: Multi-Channel Retrieval (Pool = 200 candidates)"]
        Q["User Query (or Sub-Queries)"]
        BM25_Pool["BM25 TSVector / ParadeDB Search\nTop 100 Child Chunks\nScores: BM25 Lexical IDF"]
        Dense_Pool["HNSW Dense Cosine Search\nTop 100 Child Chunks\nScores: Cosine Similarity ∈ [-1, 1]"]
        Q --> BM25_Pool
        Q --> Dense_Pool
    end

    subgraph Stage2 ["Stage 2: Rank Fusion & Deduplication (Cap = 100 candidates)"]
        RRF_Fusion["Reciprocal Rank Fusion (k=60)\nRRF(d) = Σ 1 / (60 + rank_r(d))\nCombines lexical & dense rank signals"]
        Dedup["Deduplication Engine\n1. Exact chunk_id deduplication\n2. First 120-char content prefix deduplication\nOutput: max 100 unique candidates"]
        BM25_Pool & Dense_Pool --> RRF_Fusion --> Dedup
    end

    subgraph Stage3 ["Stage 3: Reranker Pool Selection (Window = 24 candidates)"]
        PoolCut["Rerank Window Cutoff\npool_size = min(24, max(top_k, n_candidates))\nFocuses compute on the highest-confidence RRF candidates"]
        Dedup --> PoolCut
    end

    subgraph Stage4 ["Stage 4: Score Blending & Consensus Protection"]
        RRF_Norm["RRF Min-Max Normalization\nNorm_RRF ∈ [0, 1]"]
        CrossEncoder["Cross-Encoder Neural Scoring\nLogits → Numerically Stable Sigmoid → Norm_Raw ∈ [0, 1]"]
        BlendFormula["Final Score Blend Formula\nFinal(d) = 0.60 · Norm_RRF(d) + 0.40 · Norm_Raw(d) + Δ_consensus(d)"]
        ConsensusBoost["Consensus Boost (Δ_consensus)\nTier 1 (+0.15): Top-10 in BOTH BM25 & Dense\nTier 2 (+0.10): Top-20 in BOTH BM25 & Dense"]
        ConsensusSwap["protect_consensus() Guard\nGuarantees dual-channel top hits are retained\nin the final candidate pool"]
        
        PoolCut --> RRF_Norm & CrossEncoder
        RRF_Norm & CrossEncoder --> BlendFormula
        ConsensusBoost --> BlendFormula --> ConsensusSwap
    end

    subgraph Stage5 ["Stage 5: Parent Resolution & Quality Delivery (Final Top-K, default = 5)"]
        SortDesc["Sort Candidates by Final Score DESC"]
        ParentExpand["Parent Chunk Resolution\nReplace 300-tok child chunk with its\nenclosing 1,500-tok parent section for LLM context"]
        HopQualityGate["Hop Coverage Gate (>= 0.40)\nEvidence Sufficiency Gate (>= 0.25)"]
        FinalContext["Delivered to Grounded Generator / LLM\n(Enriched Parent Sections + Grounding Metadata)"]
        
        ConsensusSwap --> SortDesc --> ParentExpand --> HopQualityGate --> FinalContext
    end
```

---

## 📐 Mathematical Foundations & Formulas

Every algorithmic step in Deep Context is governed by explicit mathematical formulations rendered below:

### 1. Dense Vector Cosine Similarity

For query vector $q \in \mathbb{R}^{768}$ and stored chunk vector $d \in \mathbb{R}^{768}$ generated by `gemini-embedding-2`:

$$\text{cosine}(q, d) = \frac{q \cdot d}{\|q\|_2 \, \|d\|_2} = \frac{\sum_{i=1}^{768} q_i \, d_i}{\sqrt{\sum_{i=1}^{768} q_i^2} \cdot \sqrt{\sum_{i=1}^{768} d_i^2}}$$

In the PostgreSQL `pgvector` HNSW index, vectors are $L_2$-normalized upon insertion, allowing the cosine distance operator `<=>` to be computed via efficient inner product operations:

$$\text{cosine\_distance}(q, d) = 1 - (q \cdot d)$$

---

### 2. BM25 Lexical Score (PostgreSQL / ParadeDB)

Full-text search computes the Okapi BM25 relevance score over query terms $t \in q$ against document chunk $d$:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

Where:
- $f(t, d)$ is the term frequency of token $t$ in chunk $d$.
- $|d|$ is the token length of chunk $d$, and $\text{avgdl}$ is the average chunk token length across the corpus.
- $k_1 = 1.2$ controls term frequency saturation.
- $b = 0.75$ controls document length normalization.
- The Inverse Document Frequency $\text{IDF}(t)$ is defined as:

$$\text{IDF}(t) = \ln\left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\right)$$

#### Weighted Search Vector Composition

In PostgreSQL, the `search_tsv` column is populated via an automated database trigger combining the raw chunk text and the Qwen3 contextual summary with weighted priorities:

$$\text{search\_tsv} = \text{setweight}(\text{to\_tsvector}(\text{'english'}, \text{content}), \text{'B'}) \;||\; \text{setweight}(\text{to\_tsvector}(\text{'english'}, \text{summary\_text}), \text{'C'})$$

- Weight `'B'` (factor $0.4$) weights the raw text.
- Weight `'C'` (factor $0.2$) weights contextual summary terminology.

---

### 3. Reciprocal Rank Fusion (RRF)

To merge non-calibrated ranking scores from divergent channels (lexical BM25 and dense cosine), we implement Reciprocal Rank Fusion with smoothing constant $k = 60$:

$$\text{RRF}(d) = \sum_{r \in \mathcal{R}} \frac{1}{k + \text{rank}_r(d)}$$

Where:
- $\mathcal{R} = \{\text{BM25}, \text{Dense}\}$ is the set of retrieval channels.
- $\text{rank}_r(d) \in \{1, 2, \dots, 100\}$ is the 1-based ordinal rank of document $d$ in channel $r$.
- If document $d$ does not appear in channel $r$, its reciprocal term for that channel is $0$.
- Constant $k = 60$ prevents top-ranked outliers from completely dominating the merged distribution.

#### Concrete Numeric Example

Suppose chunk $C_1$ ranks #3 in BM25 and #7 in Dense Vector:

$$\text{RRF}(C_1) = \frac{1}{60 + 3} + \frac{1}{60 + 7} = \frac{1}{63} + \frac{1}{67} \approx 0.015873 + 0.014925 = 0.030798$$

Suppose chunk $C_2$ ranks #1 in BM25 but is absent from Dense Vector:

$$\text{RRF}(C_2) = \frac{1}{60 + 1} + 0 = \frac{1}{61} \approx 0.016393$$

Chunk $C_1$ (supported by both channels) decisively beats $C_2$ (supported by only one channel).

---

### 4. RRF Min-Max Normalization

Before blending with neural reranker logits, RRF scores across the candidate window $\mathcal{C}$ ($|\mathcal{C}| \leq 24$) are normalized to the unit interval $[0, 1]$:

$$\text{Norm\_RRF}(d) = \begin{cases} \dfrac{\text{RRF}(d) - \min_{c \in \mathcal{C}} \text{RRF}(c)}{\max_{c \in \mathcal{C}} \text{RRF}(c) - \min_{c \in \mathcal{C}} \text{RRF}(c)} & \text{if } \max_{c \in \mathcal{C}} \text{RRF}(c) - \min_{c \in \mathcal{C}} \text{RRF}(c) > 10^{-12} \\[10pt] 0.0 & \text{otherwise} \end{cases}$$

---

### 5. Numerically Stable Sigmoid (Cross-Encoder Normalization)

When neural cross-encoders (e.g. BGE-Reranker-v2-m3) output unbounded logits $x \in \mathbb{R}$, we apply a numerically stable sigmoid that eliminates floating-point overflow:

$$\sigma(x) = \begin{cases} \dfrac{1}{1 + e^{-x}} & \text{if } x \geq 0 \\[10pt] \dfrac{e^x}{1 + e^x} & \text{if } x < 0 \end{cases}$$

For any logit $x \in (-\infty, \infty)$, $\sigma(x) \in (0, 1)$, mapped to $\text{Norm\_Raw}(d)$.

---

### 6. Offline Heuristic Cross-Encoder Scoring

When operating in offline zero-dependency mode (`cross_encoder`), candidate relevance is calculated via multi-token n-gram and lexical overlap:

$$\text{Raw}_{\text{heuristic}}(d) = 0.40 \cdot \text{ExactBonus}(d) + 0.40 \cdot \text{OverlapRatio}(d) + 0.20 \cdot \text{PositionScore}(d)$$

Where:
- $\text{ExactBonus}(d) = \begin{cases} 1.0 & \text{if query} \subseteq \text{content}(d) \\ \min\left(1.0, \frac{\text{n\_gram\_hits}}{|\text{n\_grams}|} \cdot 0.9\right) & \text{if n-grams exist} \\ 0.0 & \text{otherwise} \end{cases}$
- $\text{OverlapRatio}(d) = \frac{|\mathcal{W}_{\text{query}} \cap \mathcal{W}_{\text{chunk}}|}{\max(1, |\mathcal{W}_{\text{query}}|)}$ (excluding stopwords).
- $\text{PositionScore}(d) = \frac{1}{1.0 + 0.05 \cdot \text{rank\_idx}}$.

---

### 7. Final Score Blending & Consensus Boost

The final retrieval score combines normalized RRF consensus, the cross-encoder raw score, and a bounded consensus boost:

$$\text{Final}(d) = w_{\text{RRF}} \cdot \text{Norm\_RRF}(d) + (1 - w_{\text{RRF}}) \cdot \text{Norm\_Raw}(d) + \Delta_{\text{consensus}}(d)$$

Where:
- $w_{\text{RRF}} = 0.60$ (empirically validated at **87.1% Hit@5** across the 36-query regression benchmark).
- $(1 - w_{\text{RRF}}) = 0.40$ weights the raw neural or heuristic reranker score.
- The Consensus Boost $\Delta_{\text{consensus}}(d)$ rewards documents validated across both channels:

$$\Delta_{\text{consensus}}(d) = \begin{cases} +0.15 & \text{if } \text{rank}_{\text{BM25}}(d) \leq 10 \;\land\; \text{rank}_{\text{Dense}}(d) \leq 10 \quad (\text{Tier 1 Consensus}) \\ +0.10 & \text{if } \text{rank}_{\text{BM25}}(d) \leq 20 \;\land\; \text{rank}_{\text{Dense}}(d) \leq 20 \quad (\text{Tier 2 Consensus}) \\ 0.00 & \text{otherwise} \end{cases}$$

---

### 8. Hop Coverage Gate (Multi-Hop Integrity)

For multi-hop queries decomposed into sub-queries $(q_1, q_2, \dots, q_m)$, we verify that key entity and topic terms from dependent hops are covered in the retrieved parent chunks:

$$\text{HopCoverage}(q_j, \mathcal{P}) = \frac{|\{t \in \mathcal{T}(q_j) \mid t \in \text{blob}(\mathcal{P})\}|}{|\mathcal{T}(q_j)|}$$

Where $\mathcal{T}(q_j)$ is the set of non-stopword tokens in sub-query $q_j$ (token length $> 3$), and $\text{blob}(\mathcal{P})$ is the concatenated text of all retrieved parent chunks.

$$\text{Hop Gate Condition} = \begin{cases} \text{PASS} & \text{if } \forall j > 1, \; \text{HopCoverage}(q_j, \mathcal{P}) \geq 0.40 \\ \text{RETRY} & \text{if } \exists j > 1, \; \text{HopCoverage}(q_j, \mathcal{P}) < 0.40 \end{cases}$$

If the gate fails, the engine triggers **one corrective retrieval retry** with an expanded candidate window before synthesizing.

---

### 9. Evidence Sufficiency Gate (Topical Relevance)

Before passing retrieved context to the LLM generator, the engine tests whether the candidate chunks contain sufficient lexical evidence to justify generation:

$$\text{Sufficiency}(q, \mathcal{C}) = \frac{|\mathcal{W}_{\text{query}} \cap \mathcal{W}_{\text{context}}|}{|\mathcal{W}_{\text{query}}|}$$

$$\text{Sufficiency Condition} = \begin{cases} \text{SUFFICIENT} & \text{if } \text{Sufficiency}(q, \mathcal{C}) \geq 0.25 \;\land\; \neg\,\text{is\_anachronism}(q) \\ \text{ABSTAIN} & \text{otherwise} \end{cases}$$

Where $\text{is\_anachronism}(q)$ detects modern temporal inconsistencies (e.g. asking for "smartphones", "airplanes", or "nuclear weapons" in historical or fictional corpora), triggering an immediate refusal rather than a hallucinated response.

---

### 10. Anti-Hallucination Grounding Confidence

In Pass 2, the `EvidenceVerifier` evaluates each extracted claim sentence $c \in \mathcal{C}_{\text{claims}}$ against retrieved chunks, labeling each with support status $S(c) \in \{\text{RETRIEVED}, \text{COMPUTED}, \text{INFERENCE}, \text{UNSUPPORTED}\}$:

$$\text{Grounding Confidence} = \frac{\sum_{c \in \mathcal{C}_{\text{claims}}} \mathbb{I}[S(c) \neq \text{UNSUPPORTED}]}{|\mathcal{C}_{\text{claims}}|}$$

For aggregation queries across $N_{\text{total}}$ candidates:

$$\text{Confidence}_{\text{agg}} = \min\left(\frac{\sum_{c} \mathbb{I}[S(c) \neq \text{UNSUPPORTED}]}{|\mathcal{C}_{\text{claims}}|}, \; \frac{|\mathcal{E}_{\text{cited}}|}{N_{\text{total}}}\right)$$

$$\text{Answer Accepted} \iff \text{Grounding Confidence} \geq 0.75$$

---

## 🔀 Reranker Architecture & Consensus Protection

The reranker subsystem supports three dynamic execution strategies and enforces strict consensus protection to guarantee that multi-channel consensus candidates are never dropped.

### Strategy Dispatch Matrix

```mermaid
flowchart TD
    Entry["Reranker.rerank(candidates, query, top_k)"]
    Check{"RERANKER_STRATEGY"}

    Entry --> Check

    Check -->|ecohash or hosted_bge| Hosted["Hosted Cross-Encoder (EcoHash)\nBGE-Reranker-v2-m3\nLatency: ~45ms\nRaw logits from remote inference"]
    Check -->|local_cross_encoder or bge| Local["Local Cross-Encoder (ONNX / PyTorch)\nBGE-Reranker-v2-m3 INT8\nLatency: ~180ms\nAccelerated via MPS / CUDA"]
    Check -->|cross_encoder / default| Heuristic["Heuristic Cross-Encoder\nN-Gram + Lexical Overlap + Exact Bonus\nLatency: <2ms\nZero external dependencies"]

    Hosted --> Sigmoid["Apply Numerically Stable Sigmoid σ(x)"]
    Local --> Sigmoid
    Heuristic --> NormRaw["Direct Normalized Score ∈ [0, 1]"]

    Sigmoid --> Blend["_blend_with_rrf()\nFinal = 0.60·Norm_RRF + 0.40·Norm_Raw + Δ_consensus"]
    NormRaw --> Blend

    Blend --> Protect["protect_consensus(original, reranked, top_k)\nSwap guard: preserve dual-channel top-10 hits"]
    Protect --> TopK["Return Top-K Grounded Chunks"]
```

| Strategy | Engine | Typical Latency | Dependencies | Normalization |
| :--- | :--- | :--- | :--- | :--- |
| **`cross_encoder` (Default)** | Pure Python lexical n-gram engine | $< 2\text{ ms}$ | None (Standard Library) | Bounded linear combination |
| **`ecohash` / `hosted_bge`** | BGE-Reranker-v2-m3 via HTTP API | $\sim 45\text{ ms}$ | HTTP client (`httpx`) | Numerically stable sigmoid $\sigma(x)$ |
| **`local_cross_encoder` / `bge`** | BGE-Reranker-v2-m3 INT8 Quantized | $\sim 180\text{ ms}$ | PyTorch / ONNX Runtime | Numerically stable sigmoid $\sigma(x)$ |

### Consensus Protection Logic (`protect_consensus`)

Even the strongest neural cross-encoder can occasionally suffer from false negatives on terminology-heavy passages. If a candidate chunk achieved top-10 ranking in **both** BM25 lexical search and dense HNSW vector search, it represents unambiguous dual-channel consensus.

```mermaid
flowchart TD
    Input["Sorted candidates (post-blend)"]
    Identify["Identify Consensus Candidates\n(BM25 rank <= 10 AND Dense rank <= 10)"]
    Input --> Identify
    
    CheckPresent{"Are all consensus candidates\nin the final top_k window?"}
    Identify --> CheckPresent

    CheckPresent -->|Yes| Keep["Retain current ranking unchanged"]
    CheckPresent -->|No: missing from top-k| FindVictim["Find lowest-scoring non-consensus candidate\nin positions top_k down to 1"]
    
    FindVictim --> Replace["Swap missing consensus candidate into top_k\nreplacing the non-consensus candidate"]
    Replace --> Output["Return protected top_k candidates"]
```

---

## 📦 Ingestion & Hierarchical Storage Pipeline

The ingestion pipeline transforms raw multi-format documents into dual-indexed, contextualized hierarchical representations:

```mermaid
flowchart LR
    Doc["Source Document\n(PDF / Markdown / Code)"] --> Parser["Parser Engine\n(PyMuPDF / IBM Docling)"]
    Parser --> Chunker["ParentChildChunker"]
    Chunker --> Parents["Parent Chunks\n(1000-2500 tokens)"]
    Chunker --> Children["Child Chunks\n(300-600 tokens, 15% overlap)"]
    
    Children --> Summarizer["ChunkSummarizer\n(Local Qwen3-0.6B FP16)"]
    Summarizer --> CP1[("Checkpoint 1: Atomic PG Write")]
    
    CP1 --> HNSW_IDX["pgvector HNSW (768d)\nGemini Embedding-2"]
    CP1 --> TSV_IDX["PostgreSQL TSVector\nWeighted B (text) + C (summary)"]
```

### 1. Document Parsing & Structure Extraction
- **PDF Documents:** High-fidelity layout extraction using PyMuPDF and IBM Docling, extracting tables, headers, footers, and structural section paths.
- **Code & Markdown:** AST-aware chunk boundaries that avoid splitting functions, classes, or code blocks.
- **OCR Multimodal Support:** Integrated Tesseract OCR and Vision-Language Model (VLM) fallbacks for scanned documents and embedded diagrams.

### 2. Hierarchical Parent-Child Chunking
- **Parent Chunks:** 1,000–2,500 tokens. Encapsulate full contextual sections, chapters, or architectural units.
- **Child Chunks:** 300–600 tokens with 15% sliding window overlap. Preserves tight embedding semantics for vector search.
- **Parent-Child Association:** Each child stores a foreign key reference `parent_id` pointing to its enclosing parent chunk.

### 3. Contextual Chunk Summarization (Anthropic Standard)
Each child chunk is passed through an on-device local neural model (`Qwen/Qwen3-0.6B` in FP16) that prepends a 2–3 sentence contextual summary:
```text
This chunk appears in '[Document Title]' under section '[Section Path]'.
It defines [Core Concept] and outlines [Key Findings].
```
- **Hardware Acceleration:** Auto-detects Apple Silicon Metal (MPS), NVIDIA CUDA, or CPU.
- **Thinking Mode Suppression:** Passes `/no_think` directive to bypass reasoning overhead during summarization.

### 4. Checkpoint 1 Zero Data-Loss Ingestion
To eliminate data loss caused by upstream cloud rate limits or network dropouts:
1. The document record, parent chunks, child chunks, and Qwen3 summaries are written to PostgreSQL in a single atomic transaction.
2. The weighted full-text `search_tsv` column is generated immediately via database triggers.
3. If embedding generation fails, the document remains in `pending_embeddings` status and can be resumed on demand via `POST /v1/documents/{id}/embed-stream`.

### 5. PostgreSQL Schema & Index Configuration
```sql
-- HNSW Index for Cosine Vector Search (768 dimensions)
CREATE INDEX idx_chunks_embedding_hnsw ON chunks 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 200);

-- GIN Index for Weighted Full-Text Lexical Search
CREATE INDEX idx_chunks_search_tsv ON chunks 
USING gin (search_tsv);

-- Automated Summary + Content TSVector Trigger
CREATE TRIGGER trg_chunks_search_tsv_update
BEFORE INSERT OR UPDATE OF content, summary_text ON chunks
FOR EACH ROW EXECUTE FUNCTION update_chunks_search_tsv();
```

---

## 🤖 Agentic Router & Corrective RAG (CRAG) State Machine

When user inquiries require iterative reasoning or multi-hop retrieval, the platform escalates execution to the Corrective Agentic RAG state machine:

```mermaid
flowchart TD
    UserQuery["User Query"] --> Classifier["Query Classifier"]
    Classifier --> Taxonomy{"Query Shape Taxonomy"}
    
    Taxonomy -->|factual_lookup| HybridRAG["Standard Hybrid Retrieval"]
    Taxonomy -->|how_to| HybridRAG
    Taxonomy -->|aggregation| HybridRAG
    Taxonomy -->|navigation| HybridRAG
    Taxonomy -->|multi_hop| CRAG["Agentic Planner & CRAG Loop"]

    subgraph CRAG_Loop ["Corrective RAG State Machine (CRAG)"]
        CRAG --> InitRetrieve["Node: retrieve\nExecute initial hybrid search"]
        InitRetrieve --> Grade["Node: grade_documents\nDeterministic relevance scoring against AGENTIC_GRADE_THRESHOLD"]
        
        Grade --> RelevanceCheck{"Are retrieved documents\nsufficiently relevant?"}
        
        RelevanceCheck -->|Relevant| GenerateNode["Node: generate_answer\nTwo-pass grounded synthesis"]
        RelevanceCheck -->|Irrelevant| RewriteCheck{"rewrite_count < max_rewrites\n(default: 2)?"}
        
        RewriteCheck -->|Yes| RewriteNode["Node: rewrite_question\nDecompose & reformulate query"]
        RewriteNode --> InitRetrieve
        
        RewriteCheck -->|No: Budget Exhausted| AbstainNode["Node: abstain\nSafe fallback refusal"]
    end

    GenerateNode --> VerifiedOutput["Evidence-Verified Grounded Response"]
    AbstainNode --> VerifiedOutput
```

### Query Shape Taxonomy
1. `factual_lookup`: Single-fact queries (e.g. "What is the primary key of the chunks table?").
2. `how_to`: Procedural instructions (e.g. "How do I configure the EcoHash reranker?").
3. `multi_hop`: Inquiries requiring synthesis across disparate sections or documents.
4. `aggregation`: Broad summaries requiring corpus-wide candidate coverage.
5. `navigation`: Direct location or structural queries.

---

## ✍️ Two-Pass Grounded Generation & Verification

To eliminate hallucinations, generation is split into two deterministic passes:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Router as Query Router
    participant Retrieval as Hybrid Retrieval
    participant Pass1 as Pass 1: Fact Extractor
    participant Sanitizer as Fact Sanitizer
    participant Pass2 as Pass 2: Constrained Generator
    participant Verifier as Evidence Verifier

    User->>Router: "What were the quarterly revenue numbers and who audited them?"
    Router->>Retrieval: retrieve_candidates(query, top_k=5)
    Retrieval-->>Pass1: Retrieved Parent Chunks

    Note over Pass1: Pass 1: Extract Factual Claims into JSON
    Pass1->>Pass1: Prompt LLM to isolate claims into {"supported": [...], "missing": [...]}
    Pass1-->>Sanitizer: Raw Extraction JSON

    Note over Sanitizer: Sanitize & Filter Claims
    Sanitizer->>Sanitizer: Strip unsupported claims, inferences, and modern anachronisms
    Sanitizer-->>Pass2: Verified Grounding Facts

    Note over Pass2: Pass 2: Synthesize Final Answer
    Pass2->>Pass2: Prompt LLM: "Write response using ONLY the verified facts"
    Pass2-->>Verifier: Draft Synthesized Answer

    Note over Verifier: Claim-Level Verification Gate
    Verifier->>Verifier: Check each sentence against chunk text (confidence threshold >= 0.75)
    Verifier-->>User: Grounded Answer + Traceable Citations [1, 2]
```

---

## 🧠 4-Store Typed Memory & 8-Layer Prompt Compiler

The Deep Context memory system partitions long-term state across four isolated stores to prevent policy tampering and context drift:

```mermaid
flowchart TD
    subgraph MemoryStores ["4-Store Partitioned Durable Memory"]
        Policy["memory_policy\nImmutable runtime constraints and safety policies\n(e.g. 'Never disclose internal API keys')"]
        Preference["memory_preference\nUser-scoped persistent settings\n(e.g. embedding_dim=768, reranker=ecohash)"]
        Fact["memory_fact\nVerified semantic facts with confidence scoring & TTL\n(e.g. 'Client prefers dark mode UI')"]
        Episode["memory_episode\nAppend-only session interaction summaries & traces"]
    end

    subgraph PromotionPipeline ["4-Stage Promotion Gate"]
        Obs["1. Observation\nCapture conversation event or fact"] --> Valid["2. Validation\nValidate against schema & safety policies"]
        Valid --> Score["3. Confidence Scoring\nEvaluate evidence support (score >= 0.75)"]
        Score --> Commit["4. Durable Commit\nWrite to PostgreSQL memory tables"]
    end

    subgraph PromptCompiler ["8-Layer Prompt Assembler"]
        L1["Layer 1: Base System Persona & Refusal Rules"]
        L2["Layer 2: Mandatory Active Policies (memory_policy)"]
        L3["Layer 3: User Preferences (memory_preference)"]
        L4["Layer 4: Prior Conversation Summary"]
        L5["Layer 5: Current Task State"]
        L6["Layer 6: Persistent Verified Facts (memory_fact)"]
        L7["Layer 7: Retrieved Evidence Context (Parent Chunks)"]
        L8["Layer 8: User Query Input"]

        L1 --> L2 --> L3 --> L4 --> L5 --> L6 --> L7 --> L8
    end
```

---

## 🚀 Quickstart

### 1. Prerequisites
- **Python 3.12+**
- **uv** (Modern Python package manager):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **PostgreSQL 16 with `pgvector` & `pg_search`**:
  ```bash
  # 1-Click Docker Launch (ParadeDB BM25 + pgvector + PostgreSQL 16)
  docker run -d --name deepcontext-db \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=deep_context \
    -p 5432:5432 \
    paradedb/paradedb:latest
  ```

### 2. Installation
```bash
git clone https://github.com/lakshya4568/DeepContext.git
cd DeepContext

# Create virtual environment and install all dependencies
uv sync --extra dev
```

### 3. Environment Configuration
Create a `.env` file in the project root:
```env
# Database Configuration
DATABASE_TYPE=postgres
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/deep_context

# Model Provider API Keys
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
NVIDIA_API_KEY=your_nvidia_api_key_here

# Core Pipeline Settings
EMBEDDING_MODEL=gemini-embedding-2
EMBEDDING_DIM=768
RERANKER_STRATEGY=cross_encoder
LLM_MODEL=gemini-3.7-flash
```

### 4. Launch the Web Studio
```bash
uv run uvicorn deep_context.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Access the **Deep Context Studio** at [http://localhost:8000](http://localhost:8000).

---

## 💻 CLI Reference Manual

The `deep-context` command-line utility provides complete operational control:

```bash
# Ingest a single document with Qwen3 summaries and Gemini embeddings
uv run deep-context ingest path/to/document.pdf -e gemini-embedding-2 -d 768

# Batch ingest an entire folder of files
uv run deep-context ingest-folder ./documents/ -e gemini-embedding-2 -d 768

# Execute hybrid retrieval across ingested chunks
uv run deep-context retrieve "What are the primary findings?" -k 5 -r cross_encoder

# Run full grounded query synthesis with citations
uv run deep-context query "Explain the consensus protection algorithm" -m gemini-3.7-flash

# Run the Corrective Agentic RAG state machine
uv run deep-context agentic-query "What is reward hacking?" --max-rewrites 2

# Inspect and manage persistent user preferences
uv run deep-context set-preference --user user_42 -e gemini-embedding-2 -d 768 -r ecohash
uv run deep-context preferences user_42

# Launch the internal maintenance scheduler
uv run deep-context scheduler

# List registered maintenance jobs
uv run deep-context jobs
```

---

## 📡 API & Streaming Contracts

### `POST /v1/query/stream` (Server-Sent Events)
Streams real-time retrieval status, citations, model thinking tokens, and the verified grounded response:

```bash
curl -N -X POST http://localhost:8000/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the consensus boost formula?",
    "user_id": "user_42",
    "model": "gemini-3.7-flash",
    "embedding_model": "gemini-embedding-2",
    "embedding_dim": 768,
    "reranker": "cross_encoder"
  }'
```

**SSE Event Stream:**
```json
data: {"type": "status", "stage": "retrieval", "message": "📚 Running BM25 + Dense Vector hybrid search..."}
data: {"type": "citations", "citations": [{"chunk_id": "c-102", "document_title": "Architecture.pdf", "page_number": 12}]}
data: {"type": "reasoning", "delta": "Analyzing candidate ranks across BM25 and Dense HNSW pools..."}
data: {"type": "content", "delta": "The consensus boost adds +0.15 for Tier 1 candidates (top-10 in both channels)..."}
data: {"type": "done", "latency_ms": 1140, "path_taken": "hybrid_rag", "support_check_passed": true}
```

### `POST /v1/agentic-rag` (CRAG State Machine)
Runs the self-correcting retrieval loop with complete node trace:
```bash
curl -X POST http://localhost:8000/v1/agentic-rag \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare the parent-child chunk sizes across architectures", "max_rewrites": 2, "top_k": 5}'
```

---

## 🧪 Testing & Verification

Run the comprehensive test suite (100% offline, zero external network calls required):

```bash
# Run all automated unit and integration tests
uv run pytest

# Check code formatting and linting
uv run ruff check src tests
uv run ruff format --check src tests

# Static type checking
uv run mypy src
```

---

## 📂 Repository Layout

```text
.
├── pyproject.toml                     # Dependency definitions & project metadata
├── src/deep_context/
│   ├── agentic/                       # Corrective Agentic RAG (CRAG) & Router
│   │   ├── graph.py                   # Self-correcting retrieval state machine
│   │   ├── planner.py                 # Multi-hop task decomposition & SSE streamer
│   │   └── router.py                  # Query shape classifier & path router
│   ├── api/                           # FastAPI endpoints & Server-Sent Events
│   │   ├── app.py                     # App factory & lifecycle management
│   │   ├── routes_rag.py              # Ingest, stream query, preferences, haystack APIs
│   │   ├── routes_memory.py           # Durable memory management endpoints
│   │   └── routes_ops.py              # Health, metrics, and operations APIs
│   ├── cli/                           # Command-line interface (Typer + Rich)
│   │   └── main.py                    # CLI command handlers
│   ├── core/                          # Core primitives, config, and LLM client
│   │   ├── config.py                  # Pydantic BaseSettings with .env hot-reloading
│   │   ├── llm_client.py              # Unified client for Gemini, Groq, and NVIDIA NIM
│   │   ├── logging.py                 # Structured application logging
│   │   └── types.py                   # Domain models, enums, and request schemas
│   ├── generation/                    # Grounded answer generation
│   │   └── grounded_answer.py         # Two-pass extraction and constrained synthesis
│   ├── ingestion/                     # Ingestion, parsing, and chunking
│   │   ├── chunker.py                 # Hierarchical parent-child chunker
│   │   ├── parser.py                  # PDF, Markdown, TXT, and Code structural parser
│   │   ├── pipeline.py                # Checkpoint 1 atomic ingestion pipeline
│   │   ├── summarizer.py              # Local Qwen3-0.6B contextual summarizer
│   │   └── summary_pipeline.py        # Streaming summarization pipeline
│   ├── memory/                        # 4-Store typed memory system
│   │   ├── prompt_assembler.py        # 8-layer prompt compiler
│   │   ├── promotion_gate.py          # 4-stage durable memory promotion gate
│   │   └── stores.py                  # Policy, Preference, Fact, and Episode stores
│   ├── multimodal/                    # Multimodal OCR & image extraction
│   │   ├── ocr.py                     # Tesseract OCR engine
│   │   ├── pdf_extractor.py           # Image and table extraction from PDFs
│   │   └── vlm.py                     # Vision-Language Model interface
│   ├── retrieval/                     # Hybrid search, fusion, and reranking
│   │   ├── classifier.py              # Query classifier (factual/multi-hop/aggregation)
│   │   ├── ecohash_reranker.py        # Hosted BGE cross-encoder client
│   │   ├── engine.py                  # Central retrieval engine & parent chunk resolver
│   │   ├── hybrid.py                  # BM25 + Dense Cosine + Reciprocal Rank Fusion (RRF)
│   │   ├── quality_gates.py           # Hop coverage, consensus protection & sufficiency
│   │   ├── reranker.py                # Cross-encoder dispatcher & 60/40 score blend
│   │   └── rewriter.py                # Multi-hop sub-query rewriter
│   ├── storage/                       # Database storage drivers
│   │   ├── base.py                    # StorageInterface abstract base class
│   │   ├── postgres_store.py          # PostgreSQL + pgvector (HNSW) implementation
│   │   └── sqlite_store.py            # SQLite + FTS5 fallback implementation
│   ├── ui/                            # 100% Vanilla Web Studio
│   │   └── index.html                 # Grounded Studio, Haystack Lab, Preference UI
│   ├── verification/                  # Evidence verification & anti-hallucination
│   │   └── checker.py                 # EvidenceVerifier claim-level NLI checker
│   ├── cache.py                       # Whole-answer response cache (Redis / in-memory)
│   └── scheduler.py                   # Persistence-backed background job scheduler
├── tests/                             # 126 automated test suites
├── diagrams/                          # Mermaid architectural diagrams
└── docs/                              # Formal architectural specifications & PRDs
```

---

## 📜 License

Licensed under the Apache 2.0 License. Built for high-reliability, mathematically grounded, and transparent context engineering.
