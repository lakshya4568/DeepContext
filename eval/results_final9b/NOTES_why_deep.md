# Why did deep ingestion not help? Post-hoc analysis (run after the results were known)

Sections 1 to 5 were made with `eval/scripts_final9b/why_deep.py` (`ranks`, `summaries`, `vectors`, `title`, `chunks`). They read saved rows, the database and the embedding cache;
no GPU and no Gemini call was used. They are observational: they show what goes together with the loss, not a controlled test.
Sections 6 to 9 are the controlled tests and the remedies (`eval/scripts_final9b/fix_tests.py` and the `deepkw` mode of the adapter). Section 6 made new Gemini embeddings (see its cost line).
All of it is post-hoc: the questions are the benchmark's own, with one summary model (Qwen3.5-9B) and one prompt, so read it as an explanation of this run and not as a new benchmark.

## 1. Is there a bug? No bug was found

Checked on 600 random chunks per set (FiQA subset, SciFact, MultiHop-RAG, CovidQA):

- every stored summary belongs to its own chunk: the summary shares 43% to 71% of its words with its own chunk and 2% to 4% with a random other chunk;
- every stored vector, fast and deep, equals the cached vector of exactly the text it should be (`title: ... | text: <summary>

<chunk>` for deep, `title: ... | text: <chunk>` for fast): lowest cosine 1.00000, none missing from the cache;
- the questions are embedded the same way in both arms;
- no chunk of the deep indexes lacks a summary (`status`: 0 missing); on docs, thesis and novel the result reproduces the earlier 9B round (the same single clear gain, keyword-only search on the novel, +0.117).

deep schema fiqa_sub_7081bb213a_maindeep9b: sampled 600 child chunks

1. summary vs chunk word overlap: own chunk 43%, a random other chunk 2%
2. stored deep vector against the cached vector of (summary + chunk): 600 compared, 0 not in the cache, lowest cosine 1.00000
   stored fast vector against the cached vector of the chunk: 600 compared, lowest cosine 1.00000

3. cosine between the deep and the fast vector of the SAME chunk: mean 0.895 (1.0 would mean the summary changed nothing)
   mean cosine between two DIFFERENT random chunks: fast 0.577, deep 0.616

4. 250 questions: the best chunk of a relevant document against 3,000 random chunks
   cosine to the relevant document minus the average cosine to random chunks: fast 0.253, deep 0.224
   share of the random chunks that score higher than the relevant document: fast 0.01%, deep 0.03%

scifact: deep schema scifact_ff1b0ff0ff_maindeep9b: sampled 600 child chunks
1. summary vs chunk word overlap: own chunk 71%, a random other chunk 3%
2. stored deep vector against the cached vector of (summary + chunk): 600 compared, 0 not in the cache, lowest cosine 1.00000
   stored fast vector against the cached vector of the chunk: 600 compared, lowest cosine 1.00000
3. cosine between the deep and the fast vector of the SAME chunk: mean 0.966 (1.0 would mean the summary changed nothing)
   mean cosine between two DIFFERENT random chunks: fast 0.646, deep 0.642
4. 250 questions: the best chunk of a relevant document against 3,000 random chunks
   cosine to the relevant document minus the average cosine to random chunks: fast 0.245, deep 0.237
   share of the random chunks that score higher than the relevant document: fast 0.05%, deep 0.04%

multihoprag: deep schema multihoprag_03296ea13e_maindeep9b: sampled 600 child chunks
1. summary vs chunk word overlap: own chunk 53%, a random other chunk 2%
2. stored deep vector against the cached vector of (summary + chunk): 600 compared, 0 not in the cache, lowest cosine 1.00000
   stored fast vector against the cached vector of the chunk: 600 compared, lowest cosine 1.00000
3. cosine between the deep and the fast vector of the SAME chunk: mean 0.955 (1.0 would mean the summary changed nothing)
   mean cosine between two DIFFERENT random chunks: fast 0.549, deep 0.540
4. 250 questions: the best chunk of a relevant document against 3,000 random chunks
   cosine to the relevant document minus the average cosine to random chunks: fast 0.230, deep 0.224
   share of the random chunks that score higher than the relevant document: fast 0.01%, deep 0.01%

ragbench-covidqa: deep schema ragbench_covidqa_4f4740b162_maindeep9b: sampled 600 child chunks
1. summary vs chunk word overlap: own chunk 70%, a random other chunk 4%
2. stored deep vector against the cached vector of (summary + chunk): 600 compared, 0 not in the cache, lowest cosine 1.00000
   stored fast vector against the cached vector of the chunk: 600 compared, lowest cosine 1.00000
3. cosine between the deep and the fast vector of the SAME chunk: mean 0.924 (1.0 would mean the summary changed nothing)
   mean cosine between two DIFFERENT random chunks: fast 0.675, deep 0.661
4. 242 questions: the best chunk of a relevant document against 3,000 random chunks
   cosine to the relevant document minus the average cosine to random chunks: fast 0.176, deep 0.174
   share of the random chunks that score higher than the relevant document: fast 0.08%, deep 0.11%

## 2. Where the ranking changes

Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
## How the best relevant document moved (dense only and mixed 0.02; fast to deep)

| set | setting | questions | deep ranks the relevant document higher | same place | lower | median move of the changed (places) | in the top 10 in fast, not in deep | in the top 10 in deep, not in fast |
|---|---|---|---|---|---|---|---|---|
| ragbench-covidqa | dense only | 242 | 28 (12%) | 183 (76%) | 31 (13%) | +1 | 3 | 0 |
| ragbench-covidqa | mixed 0.02 | 242 | 28 (12%) | 184 (76%) | 30 (12%) | +1 | 5 | 2 |
| scifact | dense only | 300 | 17 (6%) | 261 (87%) | 22 (7%) | +1 | 4 | 4 |
| scifact | mixed 0.02 | 300 | 20 (7%) | 258 (86%) | 22 (7%) | +1 | 4 | 3 |
| multihoprag | dense only | 265 | 15 (6%) | 195 (74%) | 55 (21%) | +1 | 1 | 0 |
| multihoprag | mixed 0.02 | 265 | 15 (6%) | 197 (74%) | 53 (20%) | +1 | 1 | 0 |
| fiqa-sub | dense only | 648 | 86 (13%) | 382 (59%) | 180 (28%) | +2 | 39 | 12 |
| fiqa-sub | mixed 0.02 | 648 | 89 (14%) | 382 (59%) | 177 (27%) | +2 | 37 | 12 |

## Hub documents: who fills the top 10 of the dense ranking (all questions together)

| set | different documents in the top 10s, fast | deep | most frequent document appears in this many top 10s, fast | deep | share of all top-10 places held by the busiest 1% of documents, fast | deep | Gini of appearances, fast | deep |
|---|---|---|---|---|---|---|---|---|
| ragbench-covidqa | 859 | 840 | 10 | 13 | 3.2% | 3.3% | 0.32 | 0.34 |
| scifact | 1969 | 1897 | 8 | 10 | 4.0% | 4.1% | 0.25 | 0.27 |
| multihoprag | 452 | 438 | 70 | 79 | 9.3% | 9.8% | 0.61 | 0.61 |
| fiqa-sub | 5064 | 4936 | 8 | 9 | 3.6% | 3.6% | 0.18 | 0.20 |

## Length of the documents that deep brings into the top 10 and the ones it drops (dense only, characters of the document text)

| set | all documents (median) | brought in by deep (median) | dropped by deep (median) |
|---|---|---|---|
| ragbench-covidqa | 668 | 658 | 694 |
| scifact | 1330 | 1465 | 1218 |
| multihoprag | 7861 | 8282 | 7715 |
| fiqa-sub | 527 | 686 | 635 |

Reading: deep does not create hub documents (the busiest 1% of documents hold the same share of top-10 places, the Gini barely moves), and the documents it
brings in or drops are not clearly shorter or longer. What changes is a small, one-sided shift: on the FiQA subset the best relevant document moves down in 28% of the
questions and up in 13% (median 2 places); on MultiHop-RAG down in 21% and up in 6%; on SciFact and CovidQA it moves both ways equally.

## 3. What the summaries contain

## What the summaries contain (random sample of up to 3,000 per set)

| set | mean words in the summary | of which the opening anchor ('In <title> (<section>),') | mean words in the passage | share of the summary's content words found in the passage | found only in the parent context | found nowhere in the prompt | summaries that start with a bare number as title |
|---|---|---|---|---|---|---|---|
| ragbench-covidqa | 43.6 | 4.0 | 87 | 71% | 0% | 29% | 0% |
| scifact | 57.6 | 16.2 | 137 | 72% | 6% | 22% | 0% |
| multihoprag | 52.5 | 15.5 | 190 | 55% | 11% | 34% | 0% |
| fiqa-sub | 44.9 | 4.0 | 117 | 44% | 3% | 53% | 100% |

Reading: on FiQA more than half of the summary's content words occur nowhere in the passage or its context, and every summary opens with a bare document number
(FiQA posts have no title, so the opening words are "In 63 (Text Document),"). On SciFact and CovidQA about 71% of the words are copied from the passage.

## 4. Does the repeated title push documents up? (MultiHop-RAG and SciFact)

multihoprag: 2357 (question, document) pairs from 200 questions
   change in the question-to-document cosine (deep minus fast): relevant documents -0.0132, other candidates -0.0105
   correlation of that change with the share of the document's title words that the question uses: +0.06
   no title word in the question: 1156 pairs, mean change -0.0118
   up to a third of the title words: 1032 pairs, mean change -0.0108
   more than a third of the title words: 169 pairs, mean change -0.0086

scifact: 2430 (question, document) pairs from 200 questions
   change in the question-to-document cosine (deep minus fast): relevant documents -0.0244, other candidates -0.0237
   correlation of that change with the share of the document's title words that the question uses: +0.01
   no title word in the question: 1197 pairs, mean change -0.0248
   up to a third of the title words: 1137 pairs, mean change -0.0225
   more than a third of the title words: 96 pairs, mean change -0.0253

Reading: the title is already in the embedded text of the fast arm (`title: <headline> | text: ...`) and the summary repeats it in its first words. That repetition is
not what hurts: the change in similarity hardly depends on how much of the title the question uses (correlation +0.06 and +0.01).

## 5. Which property of the summary goes with the fall? (best chunk of each relevant document)

fiqa-sub: 665 (question, best chunk of a relevant document) pairs; mean change of the cosine -0.0418
   share of the summary's words that are not in the chunk: correlation with the change -0.06; lowest / middle / highest third: -0.0404 / -0.0414 / -0.0437
   share taken from the parent's opening instead of the chunk: correlation with the change +0.10; lowest / middle / highest third: -0.0428 / +nan / -0.0314
   summary length next to chunk length: correlation with the change -0.11; lowest / middle / highest third: -0.0396 / -0.0427 / -0.0432

multihoprag: 67 (question, best chunk of a relevant document) pairs; mean change of the cosine -0.0196
   share of the summary's words that are not in the chunk: correlation with the change +0.09; lowest / middle / highest third: -0.0250 / -0.0165 / -0.0170
   share taken from the parent's opening instead of the chunk: correlation with the change +0.21; lowest / middle / highest third: -0.0217 / +nan / -0.0146
   summary length next to chunk length: correlation with the change -0.07; lowest / middle / highest third: -0.0176 / -0.0198 / -0.0214

scifact: 277 (question, best chunk of a relevant document) pairs; mean change of the cosine -0.0278
   share of the summary's words that are not in the chunk: correlation with the change +0.05; lowest / middle / highest third: -0.0273 / -0.0305 / -0.0257
   share taken from the parent's opening instead of the chunk: correlation with the change +0.29; lowest / middle / highest third: -0.0299 / +nan / -0.0216
   summary length next to chunk length: correlation with the change -0.09; lowest / middle / highest third: -0.0266 / -0.0254 / -0.0313

ragbench-covidqa: 599 (question, best chunk of a relevant document) pairs; mean change of the cosine -0.0280
   share of the summary's words that are not in the chunk: correlation with the change -0.16; lowest / middle / highest third: -0.0237 / -0.0286 / -0.0318
   summary length next to chunk length: correlation with the change +0.05; lowest / middle / highest third: -0.0303 / -0.0262 / -0.0275

Reading: the question-to-relevant-chunk similarity falls on every set (-0.02 to -0.04). No single property of the summary (new words, words from the parent's opening,
length next to the chunk) explains it (all correlations are weak, at most 0.3).

## 6. Controlled test: which part of the summary does the damage?

`fix_tests.py ablation`. For each question the candidates are its relevant documents plus the top 10 documents of the fast and of the deep dense ranking (FiQA subset: 200 questions, 2,603 documents, 3,940 chunks).
Every chunk is embedded again with different text, the candidates are ranked by cosine and scored with nDCG@10. This re-ranks the candidate pool, it is not a search over the whole corpus, so the
levels differ a little from RETRIEVAL.md; read the differences between the columns. All five texts are compared with the same questions.

- **deep**: summary + chunk (the real deep text). **opening only**: the summary's first words ("In <title> (<section>),") + chunk. **body only**: the rest of the summary (the added paraphrase) + chunk.
- **mean**: the average of the fast cosine and the deep cosine. **larger**: the larger of the two cosines. These two need no new text; they use both stored vectors.

nDCG@10 of the fast text, and the difference to it with the 95% interval (paired bootstrap over the questions, 10,000 resamples):

| set (questions) | fast | deep | opening only | body only | mean of fast and deep | larger of fast and deep |
|---|---|---|---|---|---|---|
| fiqa-sub (200) | 0.672 | -0.072 [-0.099, -0.046] | +0.002 [-0.012, +0.016] | -0.051 [-0.074, -0.029] | -0.008 [-0.023, +0.007] | +0.001 [-0.001, +0.004] |
| scifact (200) | 0.891 | -0.008 [-0.025, +0.010] | -0.007 [-0.017, +0.002] | -0.003 [-0.013, +0.007] | +0.000 [-0.011, +0.012] | +0.002 [+0.000, +0.005] |
| ragbench-covidqa (200) | 0.845 | -0.012 [-0.034, +0.010] | +0.008 [-0.007, +0.024] | -0.014 [-0.034, +0.006] | +0.009 [-0.004, +0.023] | +0.007 [-0.001, +0.018] |
| docsqa (35) | 0.831 | -0.011 [-0.054, +0.032] | +0.001 [-0.030, +0.032] | -0.015 [-0.047, +0.017] | +0.010 [-0.002, +0.033] | -0.001 [-0.004, +0.000] |
| pdf-got, the novel (31) | 0.817 | -0.007 [-0.059, +0.040] | +0.022 [-0.017, +0.070] | -0.000 [-0.049, +0.045] | +0.015 [-0.023, +0.058] | -0.014 [-0.035, +0.005] |

Reading:

- **On FiQA the damage is in the body of the summary**, the paraphrase added after the opening: body only -0.051 [-0.074, -0.029]. The opening words alone cost nothing (+0.002). Both together are worse than either alone (-0.072);
  the extra -0.02 is inside the noise (the intervals overlap). This corrects the earlier guess (section 3) that the bare-number opening might be the problem: it is not.
- On the other four sets no text is clearly different from fast (every interval includes 0, apart from "larger" on SciFact, +0.002, tiny). The docs (35 questions) and the novel (31) have wide intervals (about plus or minus 0.05), so they can neither confirm nor deny.
- Combining the two cosines: "larger" never loses more than 0.014 and usually 0.001 or less, with no clear gain; "mean" is -0.008 on FiQA and neutral elsewhere.

Cost of this test: FiQA 7,878 new vectors (1.87 M tokens, $0.37 at the list price of $0.2 per million tokens, the part that was approved). It was also run, by mistake, on SciFact, CovidQA, docs and the novel
(the `rrf` option was not wired in the script and fell through to this test): another 10,220 new vectors, 3.04 M tokens, $0.61 at the list price. The results are valid and kept. In all about 18,100 new vectors,
about 4.9 M tokens, about $0.98 at the list price. The real bill comes from the Google Cloud account and was not looked at.

## 7. Full-corpus search with a combined dense score (no new embedding)

`fix_tests.py ensemble`: exact dense search over the whole corpus (pgvector scan, no approximate index), using the vectors already stored in both indexes. nDCG@10 of the fast vector, and the difference to it:

| set (questions) | fast vector | deep vector | mean of the two cosines | larger of the two cosines |
|---|---|---|---|---|
| fiqa-sub (648) | 0.651 | -0.067 [-0.082, -0.052] | -0.012 [-0.021, -0.002] | -0.001 [-0.003, +0.000] |
| multihoprag (273) | 0.804 | -0.045 [-0.060, -0.031] | -0.013 [-0.023, -0.004] | -0.002 [-0.006, +0.003] |
| scifact (300) | 0.890 | -0.009 [-0.025, +0.006] | +0.004 [-0.007, +0.015] | +0.001 [+0.000, +0.003] |
| ragbench-covidqa (242) | 0.850 | -0.019 [-0.040, +0.000] | +0.006 [-0.007, +0.019] | +0.007 [-0.001, +0.017] |

Reading: taking the larger of the two cosines never loses more than 0.002 on a whole corpus, and it gains nothing either (SciFact +0.001, CovidQA +0.007, both inside the noise). The mean still loses on FiQA and MultiHop-RAG (-0.012, -0.013).
Keeping both vectors doubles the storage and the embedding cost for no measured gain, so this is not a remedy; it only shows the loss is not hidden in the stored vectors.

## 8. Fusing the saved rankings across the two arms (document level)

`fix_tests.py rrf`: the dense list of one arm fused with the keyword list of the same or the other arm (document-level reciprocal rank fusion, k = 60). Difference of nDCG@10 to "fast and fast", with the 95% interval:

| set (questions) | keyword vote | deep dense + deep keyword | fast dense + deep keyword | deep dense + fast keyword |
|---|---|---|---|---|
| fiqa-sub (648) | 1.0 | -0.024 [-0.038, -0.009] | -0.001 [-0.014, +0.012] | -0.020 [-0.030, -0.011] |
| fiqa-sub (648) | 0.02 | -0.068 [-0.084, -0.052] | -0.001 [-0.003, +0.002] | -0.067 [-0.083, -0.051] |
| multihoprag (265) | 1.0 | -0.030 [-0.045, -0.015] | -0.011 [-0.024, +0.001] | -0.017 [-0.027, -0.006] |
| multihoprag (265) | 0.02 | -0.044 [-0.060, -0.030] | +0.001 [-0.000, +0.003] | -0.044 [-0.060, -0.030] |
| scifact (300) | 1.0 | +0.005 [-0.011, +0.021] | +0.005 [-0.009, +0.018] | -0.001 [-0.011, +0.010] |
| scifact (300) | 0.02 | -0.013 [-0.030, +0.003] | -0.004 [-0.010, +0.001] | -0.009 [-0.025, +0.007] |
| ragbench-covidqa (242) | 1.0 | -0.005 [-0.024, +0.013] | -0.003 [-0.017, +0.012] | -0.009 [-0.021, +0.004] |
| ragbench-covidqa (242) | 0.02 | -0.018 [-0.038, +0.002] | -0.000 [-0.001, +0.001] | -0.018 [-0.038, +0.002] |
| docsqa (35) | 1.0 | +0.010 [-0.046, +0.063] | -0.003 [-0.045, +0.039] | -0.010 [-0.053, +0.034] |
| docsqa (35) | 0.02 | -0.011 [-0.053, +0.031] | +0.000 [+0.000, +0.000] | -0.011 [-0.053, +0.031] |
| pdf-got, the novel (31) | 1.0 | +0.067 [-0.011, +0.150] | +0.071 [+0.005, +0.145] | -0.024 [-0.066, +0.011] |
| pdf-got, the novel (31) | 0.02 | -0.005 [-0.062, +0.048] | -0.004 [-0.033, +0.017] | +0.002 [-0.052, +0.052] |

Reading:

- **The dense list is what loses.** Wherever the dense list comes from deep, FiQA and MultiHop-RAG fall (-0.067 and -0.044 with a small keyword vote), whichever arm gives the keyword list.
- **The keyword list from deep does no harm** when the dense list stays fast: -0.001 on FiQA, +0.001 on MultiHop-RAG, -0.004 on SciFact, -0.000 on CovidQA at vote 0.02. At equal votes the largest loss is -0.011 on MultiHop-RAG (interval [-0.024, +0.001]),
  and the novel keeps its gain (+0.071 [+0.005, +0.145]).
- These are fusions of saved document rankings, not the application's own chunk-level fusion. They sit close to the application's saved numbers on FiQA, SciFact and docs, and differ on CovidQA (0.806 against 0.733 at equal votes)
  because the application's near-duplicate filter acts on mixed search (`NOTES_prefix_twins_fast.md`). Both arms are compared by the same method, so the differences are the useful part. Section 9 repeats the idea with the application itself.

## Conclusion

1. **No bug.** The pipeline stored the right summaries and the right vectors.
2. **The summary adds nothing a question can use on these sets.** The documents of SciFact, CovidQA, FiQA, HotpotQA and docs are short (an abstract, a forum post, a paragraph), so a chunk is most or all of
   its document and there is nothing to "situate"; the summary is a paraphrase of the same text. Embedding a longer text that is partly paraphrase moves every vector (cosine to the fast vector of the same chunk 0.90 to 0.97) and lowers
   every question-to-document similarity (-0.02 to -0.04).
3. **That is harmless when competing documents fall by the same amount** (SciFact, CovidQA: the gap between the relevant document and random chunks shrinks by 0.002 to 0.008), **and harmful when the
   summary drifts far from the text** (FiQA: 53% of the summary's content words occur nowhere in the text it was shown, all openings are a bare number, random chunks become more alike, 0.577 to 0.616, and the gap shrinks by 11%, 0.253 to 0.224).
4. **MultiHop-RAG is not fully explained.** The gap to random chunks hardly changes (0.230 to 0.224), but relevant articles lose slightly more similarity than competing articles
   (-0.0132 against -0.0105), which is enough to push them down a few places among near-tied candidates. The title repetition and hub documents were tested and are not the cause.
5. **The one clear gain** (keyword-only search on the novel) is most likely the 9B putting names into the text from its memory of the book (70% of its novel summaries name something it was not shown). That is not a general gain.
6. **Not proven, and how to prove it:** embed (a) the chunk with only the summary's opening words and (b) the summary body without its opening words, on a sample of candidate chunks (about 3,000 embeddings, roughly $0.3 to $0.5).
   That would separate "the opening words" from "the paraphrase". Not run.
