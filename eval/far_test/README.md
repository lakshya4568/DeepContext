# Far-reference test (small): fast ingestion against deep ingestion with a local 0.6B model

Written 2026-11-01. Results: [RESULTS.md](RESULTS.md) (generated). Nothing here is committed.

## The question

A name is stated on one page; far later, a page about the same entity never spells the name ("he", "the old man",
"this scheme"). A question that names the entity should find **both** pages. Does "deep ingestion" help, when the
model that writes the extra text is small and sees one chunk at a time?

## What was built

| step | file | what it does |
|---|---|---|
| window | `window.py` | pages indexed: novel pages 6-125, thesis pages 21-97 (small, so a 0.6B model can ingest them) |
| questions | `questions.py` | script picks the later page (name missing between two pages that have it, enough pronouns); **gemini-3.7-flash** writes one item per task from 3-4 pages; plain code checks quotes, names and gaps; **gemini-3.8-flash** reviews each item on the whole window and lists every page that states each fact |
| scoring | `metric.py`, `evaluate.py` | a question is answered when each part of its evidence has an acceptable page in the top K pages (any page that states the fact counts); wins/losses against `fast` and an exact sign test |
| tagging | `tagging.py`, `run_tagger.py` | the **local Qwen3-0.6B** (the only Qwen on this Mac; the application's summariser model) names the main person or topic of every chunk, choosing from candidate names taken from the document text |
| arms | `run_eval.py` | `fast`, `notes` (the application's own summariser), `views`, `fast+views`, `fast+names`, `fast+views+names` |

The questions are written and checked by Gemini, which can hold a whole window of pages; the ingestion is done by the
small local model, which cannot. Gemini never writes any index-time text.

## Run it

```bash
export EVAL_STORE=stock PYTHONPATH=.:src HF_HUB_OFFLINE=1
uv run --no-sync python -m eval.far_test.questions build --pdf pdf-got            # Gemini: questions
uv run --no-sync python -m eval.far_test.run_tagger --pdf pdf-got --device mps    # local Qwen: names
uv run --no-sync python -m eval.far_test.run_eval --no-notes                      # fast vs views vs names
# the application's summariser (slow) and the arm that uses it:
uv run --no-sync python -m eval.far_test.window export --out eval/cache/far_test/summary_jobs.jsonl
uv run --no-sync python eval/vm/summarize_prompts.py eval/cache/far_test/summary_jobs.jsonl eval/cache/far_test/summary_results.jsonl --batch-size 4 --device mps
uv run --no-sync python -m eval.far_test.run_eval
```

On an 8 GB Mac run **one** heavy job at a time (two Qwen runs, or batches of long prompts, filled the swap and froze the
machine twice). The tagger runs at about 1.3 chunks per second on the Apple GPU with batch 4.

## What happened (2026-11-01)

* **Local Qwen3-0.6B tagging works on the Mac**: 299 novel chunks and 262 thesis chunks, about 1 chunk per second on the Apple GPU
  with batch 4 (batch 2 for the thesis). It tags almost every novel chunk (298 of 299) and 195 of 262 thesis chunks.
* **The application's own note-writer did not run.** With batches of 4 of its long prompts it filled the swap of this 8 GB Mac and
  froze it three times (the cause was not isolated; long prompts times a 152k-token vocabulary is the likely one). It needs the GPU
  machine. No number exists for the `notes` arm.
* **Few valid questions.** The writer (Gemini) proposed 28 novel and 23 thesis items in the final build; 3 and 1 survived the checks
  (stricter versions of the rule kept 0 to 2). Main reasons: novel, the passage that states fact B names the entity (12) and 10 writer
  calls returned nothing; thesis, the two pages were closer than 10 pages (19 of 23).
* **Pilot retrieval (3 + 1 questions).** Fast ingestion already finds both pages for all of them; views change nothing; the name route
  lost one novel question (a part-B page was pushed out of the top 10). With n = 4 this shows nothing more than "no room to improve".
* **Tagging accuracy against the chapter headings (238 novel passages, no model needed for the truth).** The chapter's character is
  missing from only 20 of them. The local model is right on 52% overall (simple rules: 25% and 34%), but on the 20 passages that
  do not spell the name it is right on 1 (5%), while "most frequent recent name" is right on 5 (25%). A 0.6B model does not
  resolve "he"; it copies names that are in the passage. See [RESULTS.md](RESULTS.md).

## Limits (read before quoting a number)

* Small: one window per PDF, a few dozen questions at most; only large differences can show.
* The questions are written and reviewed by models, not by a person.
* Dense search only; no reranker, no keyword channel, no answer generation.
* The novel was the held-out set of the earlier benchmark; here it is a practice document. A final claim needs new documents.
* The tagger picks one person or topic per chunk; a chunk about two people is tagged with one.
