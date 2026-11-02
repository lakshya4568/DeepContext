# Google Cloud setup: handoff notes

Written 2026-10-31. Carries the Google Cloud / MCP / ADC setup into a session that works on the
Deep Context RAG project. Contains no secrets. The API key string is deliberately not recorded.

## Accounts and projects

| Item | Value |
|---|---|
| gcloud active account | `lakshya4568@gmail.com` |
| gcloud default project | `gen-lang-client-0308909719` (Developer Hub) |
| Other signed-in account | `lakshya.sharma18.fea@gmail.com` (org account, owns `agentic-core`) |
| RAG / Vertex project | `project-d0052f4d-775b-4ef2-a56` (`agentic-core`), location `us-central1` |

## Credential layers (ADC is separate from the gcloud CLI login)

| Layer | Identity | Quota project |
|---|---|---|
| gcloud CLI + `gcloud` MCP | lakshya4568@gmail.com | Developer Hub |
| Global ADC `~/.config/gcloud/application_default_credentials.json` | lakshya4568@gmail.com | Developer Hub |
| RAG ADC `~/.config/gcloud-rag/application_default_credentials.json` | lakshya.sharma18.fea@gmail.com | agentic-core |

- The RAG project's `.env` sets `GOOGLE_APPLICATION_CREDENTIALS` to the RAG ADC file (line 24).
  `eval/strict_clients.py` calls `load_dotenv` at import, so the Vertex client uses that file.
- Why two ADC files: `agentic-core` has the org policy `constraints/iam.allowedPolicyMemberDomains`,
  so `lakshya4568@gmail.com` cannot be granted IAM there. Do not try to bypass the policy.
- Verified: `VertexRuntime().client()` called Gemini 2.5 Flash on agentic-core and returned "ok".
- Gemini API-key usage (`GEMINI_API_KEY` in `src/deep_context/core/llm_client.py`) is unaffected by ADC.

## MCP servers (user scope, in `~/.claude.json`)

| Server | Type | Status |
|---|---|---|
| `google-developer-knowledge` | HTTP `https://developerknowledge.googleapis.com/mcp`, header `X-Goog-Api-Key` | Working (returned real docs) |
| `gcloud` | stdio `npx -y @google-cloud/gcloud-mcp` (tool: `run_gcloud_command`) | Working, runs as the active gcloud account |

- API key: display name `claude-code-developer-knowledge`, restricted to the Developer Knowledge API,
  resource `projects/588075108531/locations/global/keys/3302ec2f-d5de-4f7d-b1f4-eb05a44bcc6a`.
  Retrieve the string only if needed, never print or commit it:
  `gcloud services api-keys get-key-string <resource> --project=gen-lang-client-0308909719`
- The `google-cloud-developer` plugin ships its own `developer-knowledge` server. It is a separate
  connection. In Claude Desktop it shows "not connected yet" with a Connect button, and in Claude Code
  it returned "API key not valid". Redundant with `google-developer-knowledge`, so ignore or disable it.
- A new MCP server's tools only load after Claude Code restarts.

## Skills

Plugin skills from `google-cloud-developer`: `gcloud`, `google-cloud-recipe-auth`,
`google-cloud-recipe-onboarding`, `retrieving-developer-knowledge`, `finding-google-skills`.
Gemini skills: `gemini-api-dev`, `gemini-live-api-dev`, `gemini-omni-flash-api`.

- Skill check done in the prior session: invoking `google-cloud-developer:google-cloud-recipe-auth`
  loaded and was fully readable.
- The auth skill requires reading its `references/attribution.md` before running gcloud, REST or MCP
  commands, and prefixing terminal `gcloud` calls with
  `CLOUDSDK_METRICS_REQUEST_ATTRIBUTION="agent-skills/1.0 (skill:google-cloud-recipe-auth)"`.

## Guardrails

- Never print, log or commit credential files or the API key. `.env` is gitignored.
- Use explicit `--account` and `--project` flags instead of changing global gcloud config.
- Ask before IAM changes, API enablement, key creation or anything that edits cloud resources.
- Start `gcloud` MCP use with read-only commands.
- RAG project rules: use `uv`, keep diffs minimal, no commits or pushes unless asked.

## Open items

- Confirm a full eval step runs end to end with the RAG ADC file (only a single Vertex call was tested).
- Decide whether to disable the plugin's own developer-knowledge server.

---

## Prompt to paste into the new session

```
Read GCLOUD_HANDOFF.md in this repo first. It describes my Google Cloud setup (accounts, the two ADC
files, and the google-developer-knowledge and gcloud MCP servers). Treat it as ground truth and don't
redo that setup.

Then verify, and report each as pass/fail in a short table:
1. Skills: invoke google-cloud-developer:google-cloud-recipe-auth and tell me whether you could load
   and read it.
2. Developer Knowledge MCP: call search_documents with "list Cloud Storage buckets Python" and
   summarize one result.
3. gcloud MCP: run `gcloud config list` through run_gcloud_command (read-only) and confirm the active
   account is lakshya4568@gmail.com and the project is gen-lang-client-0308909719.
4. RAG Vertex path: run a tiny script that imports eval.strict_clients.VertexRuntime, calls
   client().models.generate_content(model="gemini-2.5-flash", ...) and confirms it works under
   lakshya.sharma18.fea@gmail.com on project-d0052f4d-775b-4ef2-a56. Put the script outside the repo.

Rules: never print secrets or key strings, no IAM or resource changes without asking me, use uv for
Python, no commits. If a tool or skill is missing from this session, say so rather than working around
it. After the checks, ask what I want to build in the Deep Context RAG project.
```
