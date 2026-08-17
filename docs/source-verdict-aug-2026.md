<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## Short verdict — August 2026

There isn’t one universally “best RAG.” The strongest practical architecture today is:

> **Adaptive/Agentic Hybrid RAG + typed long/short-term memory + hierarchical retrieval + reranking**, with **RLM used selectively for very large, information-dense analysis**.


| Goal | Best choice |
| :-- | :-- |
| Normal chatbot/document QA | Hybrid vector RAG |
| Complex, multi-step queries | Agentic RAG |
| Long structured PDFs/legal/financial docs | Vectorless/hierarchical RAG such as PageIndex |
| Global reasoning over millions of tokens | RLM |
| Personalized assistants/agents | Typed memory system, not plain RAG |
| Highest consistency over huge static context | RLM |
| Best production balance of cost, latency, freshness, citations | Agentic hybrid RAG |


***

# 1. RAG approaches compared

## A. Traditional/naive RAG

Typical pipeline:

```text
Documents → chunk → embed → vector DB
User query → embed → top-k chunks → prompt → LLM answer
```


### Strengths

- Simple and cheap.
- Fast enough for production.
- Good for straightforward semantic lookup.
- Easy to cite sources and enforce document permissions.


### Weaknesses

- Retrieval is single-shot and static.
- Weak at multi-hop reasoning.
- Bad queries produce bad retrieval.
- Similarity is not always relevance.
- Retrieved chunks may miss document-level context.
- Hallucinations still occur if retrieval is poor.
- No durable understanding of the user or previous sessions.


### Best for

- FAQs.
- Small/medium knowledge bases.
- Basic documentation assistants.
- Low-latency systems.

Traditional RAG should now be considered a baseline, not the final architecture.

***

## B. Modern/advanced vector RAG

Modern vector RAG improves the traditional pipeline using:

- Query rewriting
- Multi-query retrieval
- HyDE or hypothetical answer retrieval
- Hybrid BM25 + dense-vector search
- Metadata filtering
- Parent-child retrieval
- Contextual retrieval
- Late chunking
- Reranking
- Corrective retrieval
- Self-reflection
- Knowledge graphs


### Important advanced techniques

#### 1. Hybrid search

Use:

- **Dense vector search** for conceptual/semantic matches.
- **BM25/full-text search** for exact names, IDs, functions, API terms, error codes, and rare keywords.
- **Reciprocal Rank Fusion/RRF** to combine rankings.

This is currently the safest default retrieval strategy. Pure vector search often misses exact technical terms, while BM25 misses paraphrases.

#### 2. Parent-child retrieval

Index small child chunks for precise matching, but send the larger parent section to the LLM.

```text
Search:
child chunk = 300–600 tokens

Generate:
parent section = 1,000–2,500 tokens
```

This gives high retrieval precision without losing context.

#### 3. Contextual retrieval

Before embedding a chunk, prepend a short generated explanation of where the chunk belongs:

```text
Document: Spring Security Guide
Section: JWT Authentication
This chunk explains refresh-token rotation after access-token expiry.

[actual chunk]
```

This improves retrieval when chunks lack context, but increases ingestion cost because an LLM must process each chunk.

#### 4. Late chunking

Instead of embedding isolated chunks:

1. Embed the full document/token sequence first.
2. Divide the contextual embeddings afterward.

Every chunk representation can therefore include document-level context. It is useful for long documents but requires an embedding model with a sufficiently large context window.

#### 5. GraphRAG

Represent entities and relationships as a graph:

```text
Person → works_at → Company
Company → acquired → Product
Product → uses → Technology
```

Best for:

- Multi-hop questions
- Entity-relationship reasoning
- Enterprise knowledge
- Investigations
- Research synthesis

It is usually excessive for ordinary document QA and more expensive to construct and maintain.

#### 6. Corrective RAG/CRAG

After retrieval, evaluate whether the retrieved context is sufficient and relevant:

```text
Retrieve → grade evidence
       → if sufficient: answer
       → if weak: rewrite query/search again
       → if still weak: use web/tool or say unknown
```

This is one of the most useful additions because it prevents the model from confidently answering from bad retrieval.

#### 7. Self-RAG/reflection

The model decides:

- Whether retrieval is needed.
- Whether retrieved evidence is relevant.
- Whether its answer is supported.
- Whether another retrieval round is required.

This reduces hallucinations but adds model calls and latency.

***

## C. Agentic RAG

Agentic RAG treats retrieval as a tool controlled by an agent rather than a fixed pipeline.

```text
User query
→ planner/reasoner
→ choose retrieval strategy
→ search
→ inspect results
→ rewrite/decompose query
→ call another tool
→ validate evidence
→ answer
```

NVIDIA and Redis both describe the key shift as moving from “retrieve once, generate once” to iterative retrieval, query refinement, tool use, memory, and multi-step reasoning ([NVIDIA](https://developer.nvidia.com/blog/traditional-rag-vs-agentic-rag-why-ai-agents-need-dynamic-knowledge-to-get-smarter/), [Redis](https://redis.io/blog/agentic-rag-how-enterprises-are-surmounting-the-limits-of-traditional-rag/)).

### Strengths

- Best for ambiguous and multi-step queries.
- Can select vector search, SQL, web search, code execution, graph traversal, or document navigation.
- Can detect missing evidence and search again.
- Works naturally with long-term and short-term memory.
- Better for research, coding agents, data analysis, and workflow automation.


### Weaknesses

- Higher latency.
- Higher token/API cost.
- Harder debugging and observability.
- More possible failure loops.
- Requires strict limits on iterations and tool calls.
- Not always worth it for simple questions.

Recent experimental comparisons also show that agentic RAG is not automatically superior for every workload; the added complexity should match query complexity ([arXiv comparison](https://arxiv.org/html/2601.07711v1)).

### Best for

- Research agents
- Coding agents
- Enterprise assistants
- Multi-document reasoning
- Queries requiring tools, SQL, APIs, or repeated retrieval
- Personalized assistants with memory

***

## D. Vectorless RAG

Vectorless RAG avoids approximate embedding search as the primary retrieval mechanism.

A prominent example is [PageIndex](https://github.com/VectifyAI/PageIndex), which:

1. Converts a document into a hierarchical tree, similar to an intelligent table of contents.
2. Uses an LLM to reason through the tree and navigate to relevant sections.

Its creators report 98.7% FinanceBench accuracy for a PageIndex-powered financial system, although that is a specific system/benchmark rather than proof that vectorless retrieval is universally superior.

### Strengths

- Preserves document hierarchy.
- No arbitrary chunk boundaries.
- More explainable retrieval path.
- Strong for contracts, annual reports, legal texts, textbooks, standards, and manuals.
- Avoids some embedding/index infrastructure.
- Better when section structure is meaningful.


### Weaknesses

- Multiple LLM navigation calls can be slow and expensive.
- Requires good document parsing and hierarchy extraction.
- Less suitable for millions of short, unstructured records.
- Not ideal for broad semantic discovery across a large heterogeneous corpus.
- Tree quality determines retrieval quality.


### Best for

- Long structured PDFs
- Financial filings
- Legal and compliance documents
- Academic textbooks
- Technical manuals
- Traceable professional-document QA

For your use case, vectorless retrieval is best treated as a **specialized retrieval tool**, not necessarily as a total replacement for a vector database.

***

## E. RLM — Recursive Language Model

The [Recursive Language Models paper](https://arxiv.org/html/2512.24601v1) proposes a fundamentally different long-context strategy.

Instead of feeding a giant prompt directly into the model:

1. The full prompt is stored as a variable in a persistent Python REPL.
2. The model writes code to inspect, search, filter, and decompose it.
3. It recursively calls sub-models over selected snippets.
4. Intermediate results are stored in variables.
5. The root model combines and verifies the result.

Conceptually:

```python
context = load_10_million_token_corpus()

# The root model decides what to inspect
matches = search(context, keywords)

# It can recursively ask submodels to process sections
results = [sub_lm.analyze(section) for section in selected_sections]

final_answer = root_lm.synthesize(results)
```


### Why it is important

Ordinary long-context models suffer from:

- Attention dilution
- Lost-in-the-middle behavior
- Context rot
- Higher cost as prompt length grows
- Physical context-window limits

RLM avoids loading the entire corpus into the neural context window at once. It treats the corpus as an external environment and brings only selected pieces into model context.

The RLM paper reports:

- Operation at **10M+ token scale**.
- GPT-5 RLM scoring **91.33%** on BrowseComp+ with 1K documents, versus 70.47% for a summary agent and 51% for CodeAct+BM25.
- OOLONG performance improving from 44% for base GPT-5 to 56.5% with RLM.
- OOLONG-Pairs improving from near zero for the base model to 58% with RLM.
- Comparable or lower average cost on several tasks, although runtime/cost variance is high.


### Strengths

- Best current approach for extremely long static inputs.
- Strong for global aggregation and dense reasoning.
- Every section can receive focused processing.
- Can scale beyond the physical context window.
- Can produce outputs larger than the normal model output window by assembling results programmatically.
- More consistent than simply increasing the context window.


### Weaknesses

- Complex infrastructure.
- Requires a model with strong coding/tool-use ability.
- Sequential recursive calls can be slow.
- Cost and latency can vary substantially.
- Requires sandboxing and safeguards.
- Poor decomposition can still produce bad results.
- Current models are not explicitly trained as RLMs, so behavior can be inefficient or brittle.
- It does not inherently solve freshness, permissions, persistent user memory, or source governance.


### Best for

- A 10M-token static corpus.
- Whole-codebase analysis.
- “Read every row and aggregate” tasks.
- Legal discovery over a fixed corpus.
- Deep research requiring global coverage.
- Tasks where missing one detail invalidates the answer.

***

# 2. Which gives the most consistent and accurate result?

## For massive context: RLM

If the task is:

- “Read this entire 8-million-token corpus”
- “Compare every section”
- “Aggregate every record”
- “Find all relationships across the whole repository”
- “Do not miss anything”

then **RLM should generally be more consistent than traditional RAG, agentic RAG, vectorless RAG, or a giant context window**.

Why:

- Traditional RAG only retrieves a subset.
- Reranking cannot recover evidence that first-stage retrieval missed.
- Long-context models degrade even inside their advertised token limits.
- Summarization loses details.
- RLM can systematically inspect the environment and recursively process dense sections.

So, purely for **global coverage over a fixed huge prompt/corpus**, RLM is currently the strongest paradigm.

## For production assistants: adaptive agentic RAG

If the task involves:

- Frequently changing data
- User-specific memory
- Permissions
- Fresh documents
- Low-latency interaction
- Source citations
- Millions of independently updated records
- Tools and APIs

then RLM alone is not the best architecture. An **adaptive agentic RAG system** is better.

The practical conclusion:

> **RLM is better for deep global reasoning over a static corpus. Agentic hybrid RAG is better as a general production memory/knowledge architecture.**

They are not strict competitors. The strongest system can use RLM as one of its tools.

***

# 3. Best chunking methods

There is no single best chunking method for every corpus. Redis’s 2026 chunking guide reaches the same conclusion: query type, document structure, and corpus shape should determine the method ([Redis chunking guide](https://redis.io/blog/chunking-strategy-rag-pipelines/)).

## Comparison

| Chunking method | How it works | Strengths | Weaknesses | Best use |
| :-- | :-- | :-- | :-- | :-- |
| Fixed-size | Split every N tokens with overlap | Simple, predictable | Breaks ideas and sections | Baseline/prototype |
| Recursive | Try paragraph, sentence, line, then token boundaries | Good default, coherent | Still can break structure | General text/docs |
| Semantic | Split where embedding similarity drops | Topic-aware | More ingestion cost, inconsistent gains | Essays, narratives |
| Structure-aware | Split using headings, sections, tables, AST | Preserves logical structure | Needs good parsing | Docs, legal, code, manuals |
| Parent-child | Small chunks for search, larger parents for generation | Precision + context | More storage/complexity | Best general production pattern |
| Late chunking | Embed full document first, split embeddings later | Retains global context | Requires long-context embedding model | Long documents |
| Contextual chunking | Add document/section context to each chunk | Better isolated retrieval | LLM cost during ingestion | High-value corpora |
| Agentic/LLM chunking | LLM creates atomic facts/propositions | Very precise | Slow and expensive | Legal, research, small high-value datasets |
| Vectorless/tree | Use natural hierarchy instead of chunks | Explainable, no artificial boundaries | Depends on document structure | Long structured documents |

## Recommended default

For a general RAG system:

> **Structure-aware recursive chunking + parent-child retrieval**

Example:

```text
Parent: 1,000–2,500 tokens
Child: 300–600 tokens
Overlap: 10–15%
Metadata: document, section, page, URL, timestamp, permissions
```

For fact-heavy retrieval:

```text
Child: 200–400 tokens
Parent: 800–1,500 tokens
```

For narrative or explanatory text:

```text
Child: 500–800 tokens
Parent: 1,500–3,000 tokens
```

For code:

- Use AST/function/class/module-aware chunking.
- Do not split functions randomly.
- Include imports, dependencies, file path, class name, and function signature as metadata.
- For repository-wide questions, use agentic retrieval or RLM-style analysis.

For PDFs/legal/financial reports:

- Prefer structure-aware or vectorless hierarchical indexing.
- Never cut a table away from its headers/caption.
- Keep section hierarchy and page numbers.


### My ranking for most workloads

1. **Parent-child hierarchical chunking**
2. **Structure-aware recursive chunking**
3. **Contextual retrieval**
4. **Late chunking**
5. **Semantic chunking**
6. **Fixed-size chunking**

But benchmark on your own data. Semantic chunking sounds better, yet its gains are not consistently large enough to justify its cost in every corpus.

***

# 4. Best retrieval pipeline

A strong 2026 retrieval pipeline looks like this:

```text
1. Classify query
2. Rewrite/decompose query if needed
3. Apply user/tenant/date/document filters
4. BM25/full-text retrieval
5. Dense-vector retrieval
6. Optional graph/tree/SQL retrieval
7. Merge using RRF
8. Deduplicate
9. Rerank top 50–100 candidates
10. Send top 5–10 high-quality sections to LLM
11. Check evidence sufficiency
12. Search again if needed
```

Do not retrieve 50 chunks and stuff all of them into the prompt. More context does not automatically mean better answers. It can increase cost, attention dilution, contradictions, and context rot.

A good default is:

```text
First-stage recall: top 50–100
After reranking: top 5–10
Maximum generation context: only what is necessary
```

For difficult multi-hop questions, send several smaller evidence groups through separate reasoning calls and then synthesize.

***

# 5. Reranking

Reranking is often the highest-leverage improvement after basic retrieval is working.

## Why it matters

The initial retriever is optimized for fast, broad recall. A reranker inspects the query and candidate together and improves precision before generation.

A 2026 AIMultiple benchmark reported one reranker raising Hit@1 from 62.67% to 83%, although results depend heavily on dataset and domain ([AIMultiple](https://aimultiple.com/rerankers)).

## Reranker types

### 1. Cross-encoder rerankers

Best general-purpose option.

Examples include:

- Qwen3-Reranker family
- BGE rerankers
- Jina rerankers
- Cohere rerank models
- Voyage rerank models
- NVIDIA rerank QA models

They read the query and document jointly and produce a relevance score.

**Recommended when:** accuracy matters and you are reranking roughly 20–100 candidates.

### 2. LLM rerankers

Ask an LLM to order or score documents.

**Advantages:**

- Can handle nuanced instructions.
- Useful for complex relevance definitions.

**Disadvantages:**

- Expensive
- Slow
- Can be less consistent than a trained cross-encoder
- Often unnecessary for standard RAG

Use LLM reranking only for difficult cases or a final evidence-validation step.

### 3. Late-interaction/multi-vector rerankers

Models such as ColBERT store token-level vectors and compute fine-grained similarity.

**Advantages:**

- Good quality
- More scalable than scoring every pair with a large cross-encoder

**Disadvantages:**

- Larger storage footprint
- More infrastructure complexity


## Practical recommendation

Use:

```text
Hybrid retrieval → cross-encoder reranker → top 5–8 sections
```

Possible 2026 choices:

- **Open/self-hosted:** Qwen3-Reranker-4B, BGE-reranker-v2-m3, Jina reranker for longer context
- **Managed:** Cohere/Voyage/ZeroEntropy-style hosted rerankers
- **High-volume fine-grained retrieval:** ColBERT/late interaction

The exact leaderboard changes frequently, so select based on:

- Your domain
- Language requirements
- Chunk length
- Latency budget
- Self-hosting requirements
- License
- nDCG/Hit@k on your own evaluation set

***

# 6. Long-term and short-term memory

Plain RAG is retrieval, not memory. A memory system requires a **write path**, retention rules, provenance, scoping, and prompt reassembly. Oracle’s 2026 architecture guide gives a useful typed-memory model ([Oracle memory-system guide](https://blogs.oracle.com/developers/from-rag-to-memory-systems-building-stateful-ai-architecture)).

## A. Short-term/working memory

Used for:

- Current conversation
- Current task state
- Scratchpad
- Intermediate tool results
- Temporary retrieved evidence
- Current plan

Storage:

- In-process state
- Redis/session cache
- Agent checkpointer

Lifecycle:

- Dies after the session/task or expires quickly.

Do not put every scratchpad thought into durable memory.

***

## B. Long-term memory

Should be typed instead of storing everything in one vector index.

### 1. Policy memory

Rules and constraints.

Examples:

- “Never reveal API keys.”
- “Use INR for pricing.”
- “Refunds above ₹5,000 require approval.”

Retrieval:

```text
Exact lookup by tenant/user/policy key
```

Do not retrieve policies through vector similarity.

### 2. Preference memory

Stable user preferences.

Examples:

- Prefers concise answers
- Uses Java/Spring Boot
- Wants code in TypeScript
- Date format DD/MM/YYYY

Retrieval:

```text
Exact user-scoped key lookup
```


### 3. Semantic/fact memory

Durable facts learned over time.

Examples:

- User is building an AI study agent.
- User prefers VS Code and IntelliJ.
- Project uses PostgreSQL and Redis.

Retrieval:

```text
Hybrid lexical + vector search
```

Each fact should include:

- Source
- Confidence
- Creation time
- Expiration
- User/tenant scope
- Superseded/revoked status


### 4. Episodic memory

Summaries of completed tasks or sessions.

Example:

```text
Task: Debugged Spring Security JWT filter
Outcome: Fixed by moving authentication filter before authorization filter
Important steps: ...
```

Useful when a future task resembles a previous task.

### 5. Trace memory

Raw append-only logs:

- User messages
- Tool calls
- Tool outputs
- Model decisions
- Retrieved evidence
- Token cost
- Latency

Trace memory is for replay, debugging, audit, and extraction—not direct semantic stuffing into every prompt.

***

## C. Memory promotion gate

Do not automatically save everything the user or model says.

Use this flow:

```text
Observation
→ classify memory type
→ decide whether durable
→ check user/tenant scope
→ extract atomic fact
→ compare with existing memory
→ resolve contradiction
→ assign confidence/provenance/TTL
→ write or reject
```

Example:

```text
Temporary statement:
"I might use MongoDB for this prototype."

Should not become:
"User uses MongoDB."

Better:
"User is evaluating MongoDB for project X; confidence 0.55; expires in 30 days."
```


***

## D. Prompt assembly

Rebuild the prompt every turn rather than continuously appending everything.

Recommended order:

```text
1. System/developer instructions
2. Active policies
3. User preferences
4. Short conversation summary
5. Current task state
6. Retrieved durable facts
7. Retrieved documents/evidence
8. Current user query
```

This keeps the context bounded and reduces context rot.

***

# 7. RAG memory vs RLM

| Dimension | Hybrid/Agentic RAG | Typed memory system | Vectorless RAG | RLM |
| :-- | :-- | :-- | :-- | :-- |
| Main purpose | Retrieve relevant knowledge | Maintain continuity | Navigate document structure | Process huge inputs recursively |
| Static documents | Excellent | Not primary purpose | Excellent for structured docs | Excellent |
| Frequently changing corpus | Excellent | Good if memory is updated | Moderate | Weak/less natural |
| User personalization | Limited without memory | Excellent | Limited | Limited |
| Multi-hop reasoning | Good with agentic layer | Depends on memory structure | Good within one document | Excellent |
| Global aggregation | Moderate | Poor/moderate | Moderate | Excellent |
| 10M-token input | Difficult | Not intended alone | Possible but expensive | Designed for it |
| Latency | Low/medium | Medium | Medium/high | High/variable |
| Cost predictability | Good | Good | Medium | Variable |
| Freshness | Excellent | Good | Good | Depends on reload |
| Permissions/scoping | Strong | Strong | Possible | Must be built separately |
| Citations | Strong | Strong | Strong | Possible but must be engineered |
| Consistency on huge context | Can degrade | Not applicable | Better than naive chunks | Best |
| Infrastructure complexity | Medium | Medium/high | Medium | High |

## Direct answer

- **Most accurate for global analysis over huge static context:** RLM
- **Most consistent long-context behavior:** RLM
- **Best all-around production system:** adaptive agentic hybrid RAG with typed memory
- **Best for long structured professional documents:** vectorless/hierarchical RAG
- **Best for low-latency retrieval at scale:** hybrid vector RAG
- **Best for personalization:** typed memory system, not RAG alone

***

# 8. Recommended architecture for your projects

For an AI study agent, autonomous workforce system, or web-search agent, I would build this:

```text
                         User query
                             │
                    Query/task classifier
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   Simple QA          Complex task        Huge/global analysis
        │                    │                    │
 Hybrid retrieval       Agent planner            RLM engine
 BM25 + vector          tool selection           REPL corpus
 metadata filter        multi-step search        recursive calls
        │                    │                    │
        └──────────────► Rerank/evidence check ◄──┘
                             │
                   Sufficient evidence?
                       │           │
                      Yes          No
                       │           │
                 Generate answer   Rewrite/query again
                 with citations    call web/tree/graph/SQL
                       │
                Memory extraction
                       │
               Promotion/dedup gate
                       │
        ┌──────────────┼───────────────┐
        │              │               │
  Short-term     Long-term facts   Episodic summary
  session state  + preferences     + trace
```


## Retrieval stack

Use:

- PostgreSQL/pgvector, Qdrant, Weaviate, Redis, or Milvus for vectors
- BM25/full-text search
- RRF fusion
- Parent-child chunking
- Cross-encoder reranking
- Metadata and permission filtering
- CRAG-style evidence check
- GraphRAG only where entity relationships justify it
- PageIndex/vectorless retrieval for long structured documents
- RLM for huge global-analysis jobs


## Memory stack

Use:

- Redis/session state for working memory
- PostgreSQL for typed durable memory
- Vector index only for facts/episodes that need semantic recall
- Append-only event store for traces
- Summarizer/extractor for episodic memory
- Promotion gate for memory writes
- TTL/confidence/provenance on inferred memories

***

# 9. Concrete “best defaults”

If you want to build immediately, start with:

### Ingestion

```text
Parser:
- structure-aware parser
- preserve headings, page numbers, tables, code symbols

Chunking:
- parent: 1,000–2,500 tokens
- child: 300–600 tokens
- overlap: 10–15%
- metadata on every chunk
```


### Retrieval

```text
BM25 top 50
+
vector top 50
+
metadata filters
→ RRF
→ deduplicate
→ rerank top 50
→ select top 5–8
```


### Agent behavior

```text
if evidence_score >= threshold:
    answer
else:
    rewrite query
    retrieve again
    optionally use web/tree/graph/SQL
    if still insufficient:
        state uncertainty
```


### Memory

```text
Every turn:
- exact-load policies/preferences
- retrieve relevant facts/episodes
- use short conversation summary
- answer
- append trace
- run memory extraction in background
- save only approved, deduplicated, scoped memories
```


### RLM trigger

Use RLM when:

```text
estimated corpus > model effective context
OR query requires global aggregation
OR every document/section matters
OR normal retrieval repeatedly misses evidence
```

Examples:

- Analyze an entire codebase.
- Read a full 1,000-page legal bundle.
- Aggregate every row in a huge dataset.
- Compare all chapters in a book.
- Audit every policy in a large corpus.

***

# 10. Final recommendation

If your priority is **maximum consistency over extremely long context**, choose:

> **RLM**

If your priority is **the best real-world AI assistant**, choose:

> **Adaptive agentic hybrid RAG with typed memory**

The most future-proof architecture is not RAG *or* RLM. It is:

> **Typed memory for continuity + hybrid retrieval for normal knowledge + agentic routing for complex tasks + vectorless tree retrieval for structured documents + RLM for very large global reasoning.**

That combination will outperform any one technique used alone.
<span style="display:none">[^1_1][^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_2][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_3][^1_30][^1_4][^1_5][^1_6][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: https://redis.io/blog/chunking-strategy-rag-pipelines/

[^1_2]: https://developer.nvidia.com/blog/traditional-rag-vs-agentic-rag-why-ai-agents-need-dynamic-knowledge-to-get-smarter/

[^1_3]: https://www.pingcap.com/article/agentic-rag-vs-traditional-rag-key-differences-benefits/

[^1_4]: https://arxiv.org/html/2606.00881v1

[^1_5]: https://arxiv.org/html/2601.07711v1

[^1_6]: https://arxiv.org/html/2512.24601v1

[^1_7]: https://redis.io/blog/agentic-rag-how-enterprises-are-surmounting-the-limits-of-traditional-rag/

[^1_8]: https://www.mindstudio.ai/blog/agentic-rag-vs-standard-rag-multi-layer-retrieval

[^1_9]: https://www.linkedin.com/posts/aryan-gupta-3a7036138_ai-generativeai-rag-activity-7482001744210604032-3l-M

[^1_10]: https://www.youtube.com/watch?v=qznFV59f3Uk

[^1_11]: https://towardsdatascience.com/going-beyond-the-context-window-recursive-language-models-in-action/

[^1_12]: https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089

[^1_13]: https://www.linkedin.com/posts/avi-chawla_5-chunking-strategies-for-rag-explained-activity-7351215096498487298-ag1v

[^1_14]: https://www.linkedin.com/pulse/recursive-language-models-killrag-devashish-mamgain-5wx4c

[^1_15]: https://www.reddit.com/r/Rag/comments/1ls6e3r/whats_the_best_rag_tech_stack_these_days_from/

[^1_16]: https://blogs.oracle.com/developers/from-rag-to-memory-systems-building-stateful-ai-architecture

[^1_17]: https://redis.io/blog/top-reranking-models-rag-accuracy/

[^1_18]: https://github.com/VectifyAI/PageIndex

[^1_19]: https://zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025/

[^1_20]: https://techcommunity.microsoft.com/blog/azuredevcommunityblog/vectorless-reasoning-based-rag-a-new-approach-to-retrieval-augmented-generation/4502238

[^1_21]: https://openrouter.ai/collections/rerank-models

[^1_22]: https://atlan.com/know/ai-memory-system-vs-rag/

[^1_23]: https://machinelearningmastery.com/top-5-reranking-models-to-improve-rag-results/

[^1_24]: https://aimultiple.com/rerankers

[^1_25]: https://www.linkedin.com/posts/yadavanshul36_ai-llm-rag-activity-7301703142889467904-ybwb

[^1_26]: https://supermemory.ai/blog/context-memory-guide-ai-systems/

[^1_27]: https://www.geeksforgeeks.org/artificial-intelligence/vectorless-rag-pageindex/

[^1_28]: https://fluid.ai/blogs/ai-memory-systems-explained

[^1_29]: https://www.linkedin.com/posts/dchanda_pageindex-activity-7433957139443875840-tCPc

[^1_30]: https://www.reddit.com/r/Rag/comments/1qhvzy7/compiled_a_list_of_%F0%9D%90%9A%F0%9D%90%B0%F0%9D%90%9E%F0%9D%90%AC%F0%9D%90%A8%F0%9D%90%A6%F0%9D%90%9E_%F0%9D%90%AB%F0%9D%90%9E%F0%9D%90%AB%F0%9D%90%9A%F0%9D%90%A7%F0%9D%90%A4%F0%9D%90%9E%F0%9D%90%AB%F0%9D%90%AC/


---

# can you create a deep architecture structure and PRD for all this to achieve  maximum consistency and which framework is best to use for this to implement, check the previous iterations of RLM on internet and github @GitHub

this is one implementation : [https://github.com/PrimeIntellect-ai/prime-agent](https://github.com/PrimeIntellect-ai/prime-agent)

and create skills for agents to use it (SKILL.md)  to make it from scratch

You can base this on a **prime-agent–style RLM harness plus agentic RAG and typed memory**, but build it from scratch with a clearer separation of concerns and evaluation.[^2_1][^2_2]

***

## System goals and core idea

You want an architecture that:

- Uses an **RLM engine** (prompt-as-variable + recursive subagents in a REPL) for very long, dense tasks.[^2_3][^2_2]
- Wraps this in an **agentic RAG + typed memory** stack for day-to-day work and personalization.
- Exposes **skills** as executable modules (SKILL.md + code) that agents can load, call, refine, and persist, similar to Prime Agent’s “skills as Python packages” concept.[^2_4][^2_1]
- Maximizes **consistency** by:
    - Keeping context in code/state, not huge prompts.
    - Using evaluation/verifiers, corrective retrieval, and self-refinement.
    - Making all harness changes small, reversible, and evidence-backed (continual harness).[^2_5][^2_4]

Below is a deep architecture, a PRD outline, framework recommendations, and a SKILL.md spec you can use directly.

***

## High-level architecture

### Layered view

```text
User / API Clients
│
├─ Presentation Layer
│   ├─ CLI / TUI (Textual / Rich)
│   ├─ HTTP API (FastAPI)
│   └─ JSON/RPC mode for integrations
│
├─ Agent Orchestration Layer
│   ├─ Agent Manager & Daemon
│   ├─ RLM Orchestrator (root + subagents)
│   ├─ Session & Goal Manager
│   └─ Schedules / Heartbeats / Autonomous mode
│
├─ Execution Layer
│   ├─ Persistent REPL (IPython kernel or similar)
│   ├─ Skill Runtime (Python packages)
│   ├─ Tooling Adapters (shell, git, HTTP, SQL, RAG)
│   └─ Verifier & Evaluation Engine
│
├─ Knowledge & Memory Layer
│   ├─ RAG Ingestion & Indexing
│   ├─ Retrieval (BM25 + vector + rerank)
│   ├─ Vectorless / tree index for structured docs
│   ├─ Typed Memory Stores (policy, prefs, facts, episodes)
│   └─ Trace Log & Metrics
│
└─ Infrastructure Layer
    ├─ Postgres (+ pgvector) / Qdrant
    ├─ Redis (sessions, queues)
    ├─ Object store (S3, MinIO)
    ├─ Message bus (NATS / Redis streams)
    └─ Orchestrator (systemd, Docker, K8s)
```

The key design choice: **RLM treats all context as variables in the REPL and recursive subagents as function calls**, while RAG and memory are accessed as tools from inside that environment.[^2_6][^2_3][^2_1]

***

## Core components (deep structure)

### 1. Agent Manager \& Daemon

Responsibilities:

- Maintain all live agent sessions over a local RPC/socket, like Prime Agent’s daemon.[^2_1][^2_5]
- Support:
    - `start_session(project_root, profile)`
    - `attach_session(session_id)`
    - `detach_session(session_id)`
    - `shutdown_session(session_id)`
- Track:
    - Agent state (idle, running, blocked, error)
    - Schedules and heartbeats (for recurring work)
    - Autonomous budgets (max turns, tokens, wall-clock time)

Internal structure:

- **Daemon process**
    - Owns:
        - Session registry
        - Background workers
        - IPython kernels
- **Worker processes**
    - Run individual agents (root and subagents).
- **Kernel processes**
    - Run the persistent REPL (IPython / Jupyter kernel) with isolated imports and variables per agent.[^2_4][^2_5]


### 2. RLM Orchestrator

Responsibilities:

- Implement the RLM loop:
    - Model → propose Python code → execute in REPL → observe results → repeat.[^2_7][^2_3]
- Manage recursive subagents:
    - `rlm_spawn(task_spec) → subagent_id`
    - `rlm_query(subagent_id, query)`
    - `rlm_collect(subagent_id) → result`
- Expose core variables/functions in REPL:

```python
context  # huge corpus or session state
goals  # current goals, milestones
tools  # registry of callable skills/tools
memories  # typed memory interface
retrieve()  # high-level RAG wrapper
spawn_agent()  # create subagents
log_event()  # append to trace
```

- Enforce recursion limits:
    - Max depth
    - Max parallel subagents
    - Per-agent budgets
- Provide **orchestration policies**:
    - When to use RLM for global analysis.
    - When to use simpler RAG queries.
    - When to stop recursion and return.

Use the MIT RLM reference implementation as inspiration for the core REPL + recursion semantics.[^2_3][^2_6][^2_7]

### 3. Persistent REPL / Kernel

Features:

- IPython kernel per agent, as in Prime Agent.[^2_5][^2_4][^2_1]
- Long-lived variables and imports:
    - Dataframes, partial results, code snippets, index handles.
- Tools exposed as Python functions:
    - `fs.*`, `shell.*`, `rag.*`, `memory.*`, `verify.*`, etc.
- Safeguards:
    - Sandboxed filesystem for project directory.
    - Command allowlist/denylist.
    - Resource limits (CPU, memory).
    - Timeouts on execution.

Implementation:

- Use `ipykernel` or `jupyter_client` to embed the kernel.
- Use an adapter layer to:
    - Execute code strings.
    - Capture stdout/stderr/results.
    - Marshal exceptions back to the agent loop.


### 4. RAG \& Retrieval Subsystem

Pipelines:

1. **Ingestion**
    - Structure-aware parsing (headings, sections, tables, code).[^2_8]
    - Parent-child chunking:
        - Parent: 1,000–2,500 tokens
        - Child: 300–600 tokens
    - Metadata: document id, section path, page, repository, timestamp, permissions.
    - Indexing:
        - Vector index (pgvector/Qdrant/Milvus).
        - BM25/full-text index (Postgres/Meilisearch/Elasticsearch).
        - Optional tree index for highly structured PDFs (vectorless RAG).[^2_8]
2. **Retrieval**
    - Hybrid:
        - BM25 top-N + dense top-N fused via RRF.[^2_8]
    - Reranking:
        - Cross-encoder reranker (e.g., Qwen/BGE) to select top 5–10 chunks.[^2_2]
    - Corrective step:
        - Grade evidence sufficiency.
        - If weak: re-query, widen filters, or call RLM for deeper search.
    - Expose to REPL via:

```python
results = rag.retrieve(query="...", k=10, mode="hybrid", filters={...}, rerank=True)
```

3. **Vectorless/tree navigation**
    - For long structured docs, expose:

```python
section = tree.navigate(doc_id, strategy="rlm", question="...")
```

    - Internally uses a document tree (PageIndex-style) and LLM navigation.[^2_8]

### 5. Typed Memory System

Stores:

- **Policy memory** (exact rules, per tenant/user).
- **Preferences** (user-level, stable).
- **Semantic facts** (vector or hybrid searchable).
- **Episodic summaries** (per task/session).
- **Trace log** (append-only events for audit).

Interfaces (exposed to REPL and agents):

```python
memory.get_policies(user_id, tenant_id)
memory.get_preferences(user_id)
memory.query_facts(query, k=20)
memory.save_fact(fact, scope, ttl, confidence)
memory.save_episode(summary, scope)
memory.trace.append(event)
```

Promotion pipeline:

1. Extract candidate memory from agent trajectory.
2. Classify type (policy, preference, fact, episode).
3. Validate:
    - Source (tool output vs speculation).
    - Confidence.
    - Conflicts with existing memory.
4. Write with TTL/provenance and versioning.

This mirrors Prime Intellect’s “continual harness” idea where prompts, memories, skills, and subagents are CRUD-managed and refined / rolled back.[^2_9][^2_4][^2_5]

### 6. Skills Runtime \& Registry

Skill representation:

- Each skill is:
    - A Python package/module.
    - A declarative manifest (SKILL.md).
- Registry handles:
    - Install/enable/disable skills.
    - Loading and exposing functions into REPL.
    - Versioning and rollback.

Core skills:

- `retrieval` (RAG tools).
- `memory` (typed memory access).
- `code_exec` (safe execution tasks).
- `verification` (unit tests, static analysis, invariants).
- `refinement` (self-improvement / harness edits).
- `planning` (planner/agentic decomposition).
- `web_search` (external search).
- `data_tools` (pandas, SQL, etc.).

***

## Framework recommendations

### Language \& runtime

- **Python** for RLM, REPL, skills, and infra:
    - Official RLM codebases and examples (alexzhang13/rlm, recursive-llm, Daytona guide) are Python-based and integrate easily with IPython.[^2_10][^2_3][^2_8]
    - Prime Agent uses a persistent IPython kernel and Python-based harness.[^2_4][^2_1][^2_5]


### Key frameworks

- **FastAPI** (or equivalent) for HTTP API and RPC.
- **ipykernel / Jupyter** for the embedded REPL.
- **Textual or Rich** for the CLI/TUI.
- **pgvector + Postgres** *or* **Qdrant/Milvus** for vector storage.
- **Meilisearch / Elasticsearch** for BM25/full-text.
- **Redis** for:
    - Session cache.
    - Message bus / lightweight queues.
- **litellm or similar** as an LLM abstraction layer (used in Daytona’s RLM agents).[^2_8]
- **Temporal.io or built-in scheduler** for scheduling, retries, heartbeats.
- **Prometheus + Grafana** for metrics/observability.


### RLM-specific libraries

- For inspiration or optional use:
    - `alexzhang13/rlm`: general plug-and-play inference library for RLMs with sandbox support.[^2_6][^2_3]
    - `recursive-llm` by ysz: Python RLM implementation.[^2_10]
    - Prime Agent repo itself: reference for daemon/agent/kernel layering.[^2_1][^2_5]

Given your “from scratch” requirement, you can:

- Use **alexzhang13/rlm** ideas for REPL and recursion semantics but implement your own orchestrator and harness.[^2_3][^2_6]
- Use a minimal RLM (rlm-minimal) as a starting point for experimental sandboxing.[^2_6]

***

## PRD outline (Product Requirements Document)

You can use this PRD skeleton and fill in project-specific details.

### 1. Product summary

**Product name:** Deep Context RLM Agent Platform
**Summary:** An agent platform combining Recursive Language Models, agentic RAG, and typed memory to deliver highly consistent long-running workflows over arbitrarily large knowledge bases.

### 2. Goals and success metrics

**Goals:**

- Maximize answer consistency and correctness for long-context, multi-step tasks.
- Support long-running agents with durable state, memory, and self-improvement.
- Make skills and subagents easy to author, reuse, and refine.
- Provide strong observability and auditability.

**Success metrics:**

- ≥ X% improvement in task success rate vs baseline agentic RAG on benchmark tasks.
- ≥ Y% reduction in hallucination/unsupported claims (measured by verifiers).
- Ability to process ≥ 1M-token corpora with stable quality using RLM.[^2_2][^2_7]
- Mean time to recovery (MTR) from agent errors below Z minutes.
- Skill reuse rate (skills used across ≥ N projects).


### 3. Target users and personas

- **Developer persona:** Uses agents to modify codebases, write tests, migrate frameworks.
- **Researcher persona:** Uses agents for literature review, data analysis, and multi-document synthesis.
- **Ops persona:** Uses agents for playbook execution, logs analysis, and incident retrospectives.


### 4. Use cases \& user stories

Examples:

- “As a developer, I want an agent that can understand my entire repo, run tests, and propose refactors without losing track of previous decisions.”
- “As a researcher, I want to ingest hundreds of papers and have the agent systematically compare methods and results using RLM recursion.”
- “As an ops engineer, I want long-running agents that monitor logs, refine their own alerting rules, and summarize incidents.”


### 5. Functional requirements

#### RLM Engine

- FR1: Provide a persistent REPL environment per agent session.
- FR2: Allow the model to propose and execute Python code, observe results, and iterate.
- FR3: Support recursive subagents with controlled depth and budgets.
- FR4: Expose core variables (context, goals, tools, memories) as Python objects.
- FR5: Support switching between RLM and simpler RAG workflows based on task classification.


#### Agent Management

- FR6: Daemon process to manage sessions and kernels.
- FR7: Attach/detach agents without stopping their work.
- FR8: Heartbeats and schedules for periodic re-entry.
- FR9: Persistent goals (/goal-like) across turns until completion.[^2_5][^2_4]
- FR10: Autonomous mode with bounded tokens/turns and quality gates.[^2_11][^2_4]


#### Skills

- FR11: Skills defined via SKILL.md and Python package interface.
- FR12: Skills loadable/enabled per project or user.
- FR13: Skill invocation available from REPL and agent-level planning.
- FR14: Skill refinement via `/refine` and self-improvement pipeline (edit harness state, not base prompt).[^2_9][^2_4][^2_5]


#### RAG \& Memory

- FR15: Hybrid BM25 + vector retrieval with cross-encoder reranking.[^2_2][^2_8]
- FR16: Vectorless/tree navigation for long structured documents where needed.[^2_8]
- FR17: Typed memory APIs for policies, preferences, facts, episodes.
- FR18: Memory promotion gate to prevent spam and contradictions.
- FR19: Per-tenant/user scoping and permission checks.


#### Verifiers \& Evaluation

- FR20: Configurable verifiers (tests, linters, evidence checks) callable as skills.
- FR21: Evaluation harness to compare RLM vs non-RLM approaches on test suites.[^2_7][^2_2]


#### Interfaces

- FR22: CLI/TUI for interactive work.
- FR23: HTTP/JSON RPC for programmatic integration.
- FR24: Logging and metrics endpoints.


### 6. Non-functional requirements

- NFR1: Latency budgets per operation type (interactive vs batch).
- NFR2: Resource isolation between agents and kernels.
- NFR3: Security: least-privilege file and shell access; sandboxing for untrusted projects.
- NFR4: Observability: structured logs, metrics, spans.
- NFR5: Reliability: crash recovery, kernel restart, resumption from last checkpoint.


### 7. Data model overview

Define core tables:

- `agents`: id, owner, status, budgets, created_at, updated_at.
- `sessions`: id, agent_id, project_root, kernel_ref, state_snapshot.
- `skills`: id, name, version, manifest, enabled_for.
- `memory_policy`, `memory_preference`, `memory_fact`, `memory_episode`.
- `rag_documents`, `rag_chunks`, `indexes`.
- `events_trace`: timestamp, agent_id, type, payload.


### 8. Release plan

Phases:

1. **MVP**
    - Single-agent RLM loop.
    - Basic REPL.
    - Simple RAG integration.
    - SKILL.md + a few core skills.
2. **Multi-agent / recursion**
    - Subagent support.
    - Typed memory.
    - Hybrid retrieval.
3. **Self-improvement**
    - `/refine` harness edits.
    - Skill refinement.
    - Verifier integration.
4. **Production hardening**
    - Daemon, schedules, goals.
    - Metrics, logging.
    - CI, evaluation harness.

***

## SKILL.md specification (from scratch)

Use this as the template for `SKILL.md` in each skill package and as a central skills index for the platform.

### SKILL.md top-level structure

```markdown
# Skills Overview

This document defines the skills available to agents in the Deep Context RLM Platform.
Each skill is an executable capability, implemented as a Python package, loadable in the REPL.

## Conventions

- **Name**: Unique skill identifier (snake_case).
- **Version**: Semantic version, e.g. 0.1.0.
- **Domain**: Category (retrieval, memory, planning, code_exec, verification, refinement, infra).
- **Entry point**: Python module and functions exposed.
- **Inputs/Outputs**: JSON-serializable types.
- **Triggers**: When agents should consider using this skill.
- **Safety**: Constraints, permissions, failure modes.
- **Examples**: Typical usage patterns.

---
```


### Example skill definitions

#### 1. Retrieval skill

```markdown
## Skill: retrieval

- **Name**: retrieval
- **Version**: 0.1.0
- **Domain**: retrieval
- **Entry point**: `skills.retrieval` (Python module)

### Description

Provides hybrid RAG retrieval (BM25 + vector + rerank) and optional tree-based navigation for long structured documents.

### API

```python
retrieve(
  query: str,
  k: int = 10,
  mode: Literal["hybrid", "bm25", "vector", "tree"] = "hybrid",
  filters: dict | None = None,
  rerank: bool = True
) -> list[RetrievedChunk]

navigate_tree(
  doc_id: str,
  question: str,
  strategy: Literal["rlm", "simple"] = "rlm"
) -> TreeSection
```


### Inputs

- `query`: User or agent query string.
- `k`: Number of chunks or sections to return.
- `filters`: Metadata filters (tenant, project, time range, permissions).


### Outputs

- List of `RetrievedChunk` objects containing:
    - `content`
    - `metadata`
    - `score`
    - `source_id`


### Triggers

- Agent needs external knowledge beyond current REPL variables.
- Corrective retrieval after weak evidence.
- RLM root planner wants to seed subagents with relevant context.


### Safety

- Enforces tenant/user permissions on documents.
- Limits `k` and payload size per call.
- Logs queries and sources to trace.


### Examples

- Answering “Explain how auth works in this repo.”
- Locating all usages of an API across codebase.

```

#### 2. Memory skill

```markdown
## Skill: memory

- **Name**: memory
- **Version**: 0.1.0
- **Domain**: memory
- **Entry point**: `skills.memory`

### Description

Reads and writes typed long-term memory (policies, preferences, facts, episodes).

### API

```python
get_policies(user_id: str, tenant_id: str) -> list[Policy]
get_preferences(user_id: str) -> Preferences
query_facts(query: str, k: int = 20) -> list[Fact]
save_fact(fact: Fact, scope: Scope, ttl: int | None, confidence: float) -> None
save_episode(summary: EpisodeSummary, scope: Scope) -> None
```


### Triggers

- Assembling prompt/harness context at turn start.
- Persisting a lesson from a completed task.
- Fetching user preferences before planning.


### Safety

- Write path goes through promotion gate.
- No direct speculative writes.
- All facts include source and confidence.


### Examples

- Storing that a project uses PostgreSQL + Redis.
- Retrieving policy on secrets handling before running shell commands.

```

#### 3. RLM orchestrator skill

```markdown
## Skill: rlm_orchestrator

- **Name**: rlm_orchestrator
- **Version**: 0.1.0
- **Domain**: orchestration
- **Entry point**: `skills.rlm_orchestrator`

### Description

Provides helpers to spawn subagents, manage recursion, and aggregate results.

### API

```python
spawn_agent(task_spec: TaskSpec) -> AgentId
query_agent(agent_id: AgentId, query: str) -> AgentResponse
collect_agent(agent_id: AgentId) -> AgentResult
set_recursion_limits(depth: int, max_agents: int) -> None
```


### Triggers

- Root agent decomposes a task into subproblems.
- Need parallel analysis of multiple repositories or documents.


### Safety

- Enforces global recursion and resource limits.
- Prevents infinite loops by tracking budgets.


### Examples

- Spawning separate agents for frontend, backend, and infra in a monorepo.

```

#### 4. Code execution skill

```markdown
## Skill: code_exec

- **Name**: code_exec
- **Version**: 0.1.0
- **Domain**: code_exec
- **Entry point**: `skills.code_exec`

### Description

Executes code and commands via the REPL in a controlled manner.

### API

```python
run_python(code: str) -> ExecutionResult
run_shell(command: str, cwd: str | None = None) -> ShellResult
```


### Safety

- Enforces allowlist on shell commands.
- Limits execution time and resource usage.
- Logs all invocations and outputs.


### Examples

- Running unit tests.
- Applying code transformations with `sed`/`awk` or custom scripts.

```

#### 5. Verification skill

```markdown
## Skill: verification

- **Name**: verification
- **Version**: 0.1.0
- **Domain**: verification
- **Entry point**: `skills.verification`

### Description

Provides verifiers for code, documents, and answers (tests, static checks, evidence scoring).

### API

```python
run_tests(test_command: str) -> TestReport
lint_code(paths: list[str]) -> LintReport
check_answer_support(answer: str, evidence: list[RetrievedChunk]) -> SupportScore
```


### Triggers

- Before proposing patches or refactors.
- Before finalizing long-form answers.
- Within autonomous mode quality gates.


### Examples

- Verifying that generated code compiles and passes tests.
- Checking that an answer is supported by retrieved evidence.

```

#### 6. Refinement skill

```markdown
## Skill: refinement

- **Name**: refinement
- **Version**: 0.1.0
- **Domain**: refinement
- **Entry point**: `skills.refinement`

### Description

Implements the self-improvement pipeline (/refine): reads trajectory and applies small, reversible changes to harness state (prompts, memory, skills, subagent configs).[web:4][web:14]

### API

```python
review_trajectory(agent_id: AgentId) -> list[ImprovementCandidate]
apply_improvement(candidate: ImprovementCandidate) -> None
rollback_improvement(improvement_id: str) -> None
```


### Safety

- Never edits the immutable base system prompt.
- Requires explicit evidence links for each change.
- Maintains a full refinement history for audit.


### Examples

- Updating a project-specific coding style note after repeated corrections.
- Adding a new skill description after a recurring workflow is discovered.

```

You can add more skills (planning, web_search, data_tools, etc.) following this pattern.

***

## How RLM iterations inform this design

- The RLM paper and repos show that treating long prompts as **external environment variables** in a REPL and recursively calling submodels yields better long-context performance and scalability than large static prompts.[^2_7][^2_3][^2_2]
- Prime Agent demonstrates a production harness where:
  - The **only tool** is a persistent IPython kernel.
  - Subagents are real child agents (`rlm(...)`).
  - The harness can self-improve via `/refine` over prompts, memory, skills, and subagents.[^2_4][^2_1][^2_5]
- Daytona’s RLM guide shows a multi-agent recursive architecture using sandboxed repos and litellm, confirming that **Python + REPL + recursive subagents + typed tools** is a practical blueprint.[^2_8]

This architecture and PRD pull those ideas together and extend them with a strong RAG/memory layer.

***

If you’d like, I can next:

- Turn this into concrete FastAPI/REPL skeleton code.
- Expand SKILL.md into a full file for your repo with all core skills pre-defined.
<span style="display:none">[^2_12][^2_13][^2_14][^2_15]</span>

<div align="center">⁂</div>

[^2_1]: https://github.com/PrimeIntellect-ai/prime-agent
[^2_2]: https://huggingface.co/papers/2512.24601
[^2_3]: https://github.com/alexzhang13/rlm
[^2_4]: https://www.primeintellect.ai/blog/prime-agent
[^2_5]: https://kingy.ai/blog/prime-agent-review-self-improving-rlm-harness/
[^2_6]: https://alexzhang13.github.io/blog/2025/rlm/
[^2_7]: https://arxiv.org/abs/2512.24601
[^2_8]: https://www.daytona.io/docs/en/guides/rlm/recursive-language-models/
[^2_9]: https://www.primeintellect.ai/blog/rlm
[^2_10]: https://github.com/ysz/recursive-llm
[^2_11]: https://discuss.google.dev/t/recursive-language-models-in-adk/323523
[^2_12]: https://www.youtube.com/watch?v=djEDs-c7eVM
[^2_13]: https://www.linkedin.com/posts/muhammad-naeem57_ai-artificialintelligence-llm-activity-7491485627528585216-61L4
[^2_14]: https://news.ycombinator.com/item?id=45596059
[^2_15]: https://navendu.me/posts/recursive-language-models/```

