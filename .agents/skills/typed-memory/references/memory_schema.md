# Typed Memory — Schema Reference

Column-level contract for the four memory tables plus the trace log. Full DDL: `docs/DATA_MODEL.sql`. This file
adds the write-discipline notes that don't fit in SQL comments.

## `memory_policy`

| Column | Notes |
|---|---|
| `tenant_id` | Required. |
| `user_id` | `NULL` = tenant-wide policy. |
| `policy_key` | e.g. `"refund_approval_threshold"`. Unique per `(tenant_id, user_id, policy_key)`. |
| `policy_value` | JSONB — structured, not freeform text. |

**Write discipline:** never written by the promotion gate or by inference. Written only by an explicit
operator/admin path (a config UI, a migration script, an admin API call). If you find yourself tempted to have
the agent "infer a policy" from a conversation, that observation belongs in `memory_fact` with low confidence
and a note, not in `memory_policy`.

## `memory_preference`

| Column | Notes |
|---|---|
| `user_id` | Required, always user-scoped (no tenant-wide preferences). |
| `preference_key` | e.g. `"response_length"`, `"preferred_language"`. Unique per `(user_id, preference_key)`. |
| `confidence` | `0.0`–`1.0`. |
| `source` | `'explicit'` (user stated it directly) or `'inferred'`. |

**Write discipline:** `source = 'explicit'` can be written on a single observation (the user said it outright).
`source = 'inferred'` requires ≥2 corroborating observations before promotion, per the promotion gate. A single
inferred signal goes into a staging area (or a low-confidence `memory_fact` row) until corroborated, not
directly into `memory_preference`.

## `memory_fact`

| Column | Notes |
|---|---|
| `tenant_id` / `user_id` | `user_id = NULL` means tenant-scoped fact. |
| `content` | The **atomic claim**, not the raw observation sentence. |
| `source` | `'user_stated'` \| `'tool_output'` \| `'inferred'`. |
| `confidence` | `0.0`–`1.0`, required (`CHECK` constraint in the DDL). |
| `superseded_by` | Self-referencing FK. Set, never deleted, when a newer fact replaces this one. |
| `expires_at` | `NULL` = no TTL. See SKILL.md's TTL guidance — default to a real TTL for anything hedged/speculative. |

**Write discipline:** the table the promotion gate writes to most often. Every row must have defensible
`source`/`confidence`/provenance — this is the table the "memory promotion precision" metric
(`docs/PRD.md` §4) is measured against, via `WHERE source = 'inferred'` spot-checks.

## `memory_episode`

| Column | Notes |
|---|---|
| `user_id` | Required. |
| `session_id` | FK to `sessions`. |
| `task_type` | Freeform label used for similarity matching at read time (e.g. `"debugging"`, `"literature_review"`). |
| `summary` | One per completed task/session — not a raw transcript dump. |
| `outcome` | `'success'` \| `'partial'` \| `'failed'`. |

**Write discipline:** written automatically at session/task completion, from `events_trace`, not from raw
conversation history. Keep summaries short and specific enough to be useful for "has this happened before?"
style queries (see the debugging use case in `docs/PRD.md` §6).

## `events_trace`

Append-only. `event_type` values include `'retrieval'`, `'memory_write'`, `'rlm_spawn'`, `'tool_call'`,
`'router_decision'`. This is **not** a memory type — it's the raw material `memory_episode` rows get extracted
from, and the audit trail every promotion-gate decision (write or reject) should log to. Never inject rows from
this table directly into a prompt; summarize first.

## Read-time query shape (for prompt assembly step 6/7)

```sql
-- Policies / preferences: exact lookup, cheap, always first
SELECT policy_value FROM memory_policy
WHERE tenant_id = %s AND (user_id = %s OR user_id IS NULL) AND policy_key = %s;

SELECT preference_value FROM memory_preference
WHERE user_id = %s AND preference_key = %s;

-- Facts: hybrid search, same RRF pattern as skills/rag-retrieval/, scoped and TTL-filtered
SELECT content, confidence, source FROM memory_fact
WHERE (tenant_id, user_id) IN (...)
  AND superseded_by IS NULL
  AND (expires_at IS NULL OR expires_at > now())
  -- ... fused BM25 + vector recall against `content`/`embedding`, same as chunk retrieval
ORDER BY fused_score DESC LIMIT %s;
```
