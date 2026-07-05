# HOS Agent — GCP

A multi-agent system for analysing Medicare Health Outcomes Survey (HOS) data, built with Google ADK, Gemini 2.5, FastAPI, and ChromaDB. Deployed on GCP Cloud Run with a full CI/CD pipeline.

---

## What it does

The system answers natural-language questions about HOS Public Use Files (PUFs) across three cohorts (c25a, c26b, c27b). It handles:

- **Definitions and methodology** — looks up what variables mean, their coded value labels, and survey design from the HOS PDF data dictionaries (RAG over ChromaDB)
- **Dataset and schema queries** — lists available datasets, columns, and schema entries
- **Statistical analysis** — Pearson correlation, Random Forest feature importance, logistic regression, chi-square/cross-tabulation, Mann-Whitney U / Kruskal-Wallis group comparisons
- **Multi-step analytical reports** — comprehensive health profile reports chaining all relevant analyses for a target variable, with per-step validation gates

---

## Architecture

```
User / API client
       │
       ▼
   orchestrator  (gemini-2.5-pro)
   ┌─────────────────────────────────────┐
   │  sub_agents:                        │
   │    pdf_agent  ── search_pdf_guidance│  ←── ChromaDB (all-MiniLM-L6-v2)
   │               └─ get_column_info   │  ←── schema_memory.json
   │    csv_agent  ── list_datasets     │
   │               └─ get_column_info   │
   │  tools:                            │
   │    analysis_agent (AgentTool)       │
   │      ├─ generate_health_report     │  ←── workflow engine (report_tools.py)
   │      ├─ run_correlation_analysis   │
   │      ├─ run_feature_importance     │
   │      ├─ run_logistic_regression    │  ←── HOS PUF CSVs (pandas + scipy/sklearn)
   │      ├─ run_categorical_analysis   │
   │      └─ run_group_comparison       │
   └─────────────────────────────────────┘
```

**Routing rules (orchestrator instruction):**
- PDF definitions / methodology questions → `pdf_agent`
- Dataset structure / column listings → `csv_agent`
- Statistical questions → `analysis_agent` directly (no csv_agent pre-validation)
- Multi-part analysis requests → `analysis_agent` using `generate_health_report`

**Column name resolution** (`_find_column`): three-tier fallback — exact match → description substring → embedding cosine similarity (threshold 0.35) — so the agent can pass plain-English phrasing like `"general health status"` and the tool resolves it to the actual column code (`B25VRGENHTH`).

---

## Tech stack

| Layer | Choice |
|---|---|
| Agent framework | Google ADK 2.1.0 |
| LLM — orchestrator | Gemini 2.5 Pro |
| LLM — specialists | Gemini 2.5 Flash |
| API | FastAPI + uvicorn |
| RAG index | ChromaDB (persistent, synced to GCS) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Analysis | pandas, scipy, scikit-learn |
| Schema cache | `schema_memory.json` (PDF-parsed + Gemini fallback) |
| Runtime | GCP Cloud Run (2 vCPU, 2 GiB, min-instances 0) |
| CI/CD | GitHub Actions → Artifact Registry → Cloud Run |

---

## Local setup

**Prerequisites:** Python 3.10+, HOS PDF data dictionaries in `data/pdfs/`, HOS PUF CSVs in `data/csvs/`.

```bash
# 1. Create and activate virtual environment
py -3.10 -m venv .venv
.\.venv\Scripts\activate          # Windows
source .venv/bin/activate         # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — set GOOGLE_API_KEY at minimum

# 4. Start the server
.\.venv\Scripts\uvicorn.exe main:app --port 8080
```

> **Do not use `--reload`**. The rotating log file handler writes `outputs/logs/app.log` on every line, which file-watchers detect as a change and trigger an infinite reload loop.

On first start the schema builder parses the PDFs (Pass 1) and runs a Gemini fallback for any columns not found in the data dictionary (Pass 2). Subsequent starts load from the `schema_memory.json` cache.

**Local cloud-mode** (test against the real GCS bucket):
```powershell
gcloud auth application-default login
$env:ENVIRONMENT="cloud"
$env:GCS_BUCKET="hos-agent-files-hos-agent"
$env:GOOGLE_CLOUD_PROJECT="hos-agent"
.\.venv\Scripts\uvicorn.exe main:app --port 8080
```

---

## API endpoints

### `GET /health`
Returns server status, active models, and readiness of the vector store and schema cache.

### `POST /query`
Single-turn question answering through the full agent pipeline.

```json
{ "question": "What does PCS mean?", "session_id": null }
```

Returns `{ "answer": "...", "session_id": "..." }`. Pass the same `session_id` on follow-up turns to maintain conversation context.

### `POST /clear`
Deletes all active sessions from the in-memory session store.

### `POST /reload`
Re-syncs input data from GCS (cloud mode) and forces a full re-index of the vector store and schema cache. Use after uploading new PDFs or CSVs.

### `GET /usage`
Returns cumulative Gemini token usage and cost breakdown by model and agent, read from `outputs/token_usage.jsonl`.

### `POST /evaluate`
Runs a question through the agent then evaluates the answer across five components.

```json
{ "question": "What predicts general health status in c25a?", "deep": false }
```

`deep` accepts `false` (embedding only), `"auto"` (embedding + selective Gemini escalation), or `true` (Gemini judge on all three scored dimensions). Returns the answer, session ID, and a full evaluation report — see [Evaluation](#evaluation) below.

### `POST /evaluate/suite`
Runs the full 13-case ground-truth test suite and returns per-case scores plus an aggregated instrumentation summary.

```
POST /evaluate/suite?deep=auto
```

Cases flagged `requires_prior_turn` (multi-turn follow-ups that need session history) are reported separately under `context_dependent_results` and excluded from the main pass count.

### `POST /report`
Directly executes a multi-step analytical workflow without an LLM hop. Returns both the markdown report and a structured execution trace — useful for programmatic access and testing.

```json
{
  "dataset": "c25a",
  "target_column": "general health status",
  "workflow": "health_profile",
  "max_missing_pct": 0.5,
  "max_unique_for_classification": 20
}
```

Returns `WorkflowResult.to_dict()` — see [Analytical workflows](#analytical-workflows) below.

---

## Evaluation

The evaluator scores each agent response across five components:

| Component | Method | Notes |
|---|---|---|
| `routing` | Rule-based keyword match | Did the orchestrator route to the right specialist? |
| `tool_selection` | Rule-based keyword match | Did the analysis agent pick the right test? |
| `retrieval_relevance` | Embedding sim (question ↔ context) | How relevant is the retrieved context to the question? |
| `answer_faithfulness` | Embedding sim (context ↔ answer) | Are all claims in the answer grounded in retrieved context? |
| `answer_quality` | Embedding sim (question ↔ answer) | Is the answer correct, complete, and plain-English? |

`retrieval_relevance` and `answer_faithfulness` are floored at 0.5 when no retrieval tool was called (e.g. for pure analysis-agent questions — there is no grounding text to evaluate against, by design).

**`deep` modes for the three similarity-scored dimensions:**

- `deep=false` — embedding similarity only. Fast, deterministic, quota-free. Default.
- `deep="auto"` — embedding first; escalates to one Gemini judge call per dimension when `rr < 0.5` and `aq > 0.6`, or `af < 0.6` and `aq > 0.6`, or `aq < 0.5` unconditionally. The `aq > 0.6` gate avoids spending Gemini quota confirming that a genuinely bad answer is bad — it targets cases where embeddings and observed answer quality disagree. Estimated ~15% of full-deep call volume.
- `deep=true` — Gemini-as-judge on all three dimensions for every case.

**Known limitations:**
- Short acronym questions (PCS, MCS) produce low `retrieval_relevance` via embedding even after acronym expansion improves actual retrieval — a length-mismatch artefact of cosine similarity between a 5-word query and a long technical passage. `deep="auto"` rescues these cases.
- Non-prose answers (raw column-code dumps) score poorly across all embedding dimensions for the same surface-form reason. Documented as a known blind spot, not further optimised.

**Current suite result: 12/13 pass** on scored cases (1 context-dependent case reported separately).

---

## Analytical workflows

`generate_health_report` (available as both an ADK tool on `analysis_agent` and directly via `POST /report`) runs a configurable sequence of analyses for a single target variable and returns a structured markdown report.

### `health_profile` workflow

Steps run in order, each guarded by a validation gate:

| Step | Gate condition (skip if…) |
|---|---|
| Frequency Distribution | target has < 2 unique values |
| Top Predictors (Random Forest) | target has > 20 unique values (likely continuous) or < 2 classes |
| Group Comparison by Sex | SEX column absent or single-valued |
| Group Comparison by Age | AGE column absent or single-valued |
| Top Correlates (Pearson) | target column is not numeric |

A global missing-data gate aborts the entire report before any step if the target column is > 50% null.

All gate thresholds are configurable via `GateConfig` (Python) or the `POST /report` request body. The report includes an Execution Summary table listing each step's status (completed / skipped / failed), timing in ms, and gate reason for any skipped steps.

The full execution trace (status, reason, output length, duration per step plus a summary dict) is written to `outputs/report_trace.json` and returned by `POST /report`.

### Adding a new workflow

Define a new list of `WorkflowStep` objects in `WORKFLOWS` in `tools/report_tools.py` — no other code changes required. The execution engine (`_run_workflow`) is workflow-agnostic.

---

## Project structure

```
agents/
  orchestrator.py        — root agent; routing rules
  pdf_agent.py           — HOS PDF specialist (RAG + schema lookup)
  csv_agent.py           — dataset and schema listing
  analysis_agent.py      — statistical analysis specialist

tools/
  pdf_tools.py           — search_pdf_guidance (ChromaDB), get_column_info, acronym expansion
  csv_tools.py           — list_datasets, get_dataset, column info delegation
  analysis_tools.py      — 5 statistical tools + 3-tier _find_column resolver
  report_tools.py        — workflow engine: GateConfig, WORKFLOWS, _run_workflow, generate_health_report

rag/
  embedder.py            — all-MiniLM-L6-v2 wrapper (embed + chunk_text)
  vector_store.py        — ChromaDB with manifest-based cache invalidation + GCS sync
  schema_builder.py      — PDF parser (Pass 1) + Gemini fallback (Pass 2) + GCS persistence

evaluation/
  evaluator.py           — ComponentEvaluator: 5-dimension scoring, 3 deep modes, instrumentation
  test_suite.py          — 14 ground-truth test cases (13 scored + 1 context-dependent)

config/
  settings.py            — pydantic-settings: local vs cloud switch, all file paths

tests/
  test_tools.py          — unit tests for analysis tools and report workflow engine
  test_analysis_agent.py
  test_csv_agent.py
  test_pdf_agent.py
  test_orchestrator.py

main.py                  — FastAPI app: all endpoints, _run_query with context capture
token_tracker.py         — per-event Gemini token and cost tracking → outputs/token_usage.jsonl
gcs_data.py              — GCS data sync helpers (download/upload directory)
Dockerfile               — python:3.11-slim image for Cloud Run
.github/workflows/deploy.yml — CI/CD: pytest → Docker build → Artifact Registry → Cloud Run
```

---

## GCP infrastructure

| Resource | Name |
|---|---|
| Project | `hos-agent` |
| Cloud Run service | `hos-agent` (us-east1) |
| Artifact Registry | `us-east1-docker.pkg.dev/hos-agent/hos-agent/app` |
| GCS bucket | `gs://hos-agent-files-hos-agent` |
| Secret Manager | `GOOGLE_API_KEY` |
| Runtime service account | `481249180021-compute@developer.gserviceaccount.com` |
| Deploy service account | `hos-agent-deployer@hos-agent.iam.gserviceaccount.com` |

**GCS bucket layout:**
```
gs://hos-agent-files-hos-agent/
  data/pdfs/          ← HOS PDF data dictionaries (downloaded at startup)
  data/csvs/          ← HOS PUF CSV files (downloaded at startup)
  chroma_db/          ← ChromaDB index (persisted across cold starts)
  memory/             ← schema_memory.json (persisted across cold starts)
```

**CI/CD:** every push to `main` runs `pytest tests/` first; the Docker build and Cloud Run deploy only proceed if all tests pass.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini API key (required) |
| `GOOGLE_GENAI_USE_VERTEXAI` | `FALSE` | Set `TRUE` to use Vertex AI instead of AI Studio |
| `ORCHESTRATOR_MODEL` | `gemini-2.5-pro` | Model for the root orchestrator agent |
| `SPECIALIST_MODEL` | `gemini-2.5-flash` | Model for pdf_agent, csv_agent, analysis_agent |
| `EVALUATOR_MODEL` | `gemini-2.5-flash` | Model used by Gemini-as-judge in deep/auto eval mode |
| `ENVIRONMENT` | `local` | `local` reads from disk; `cloud` reads from GCS |
| `GCS_BUCKET` | — | GCS bucket name (no `gs://` prefix); required in cloud mode |
| `GOOGLE_CLOUD_PROJECT` | — | GCP project ID; required when using user ADC (not a service account key) |
| `PDF_DIR` | `./data/pdfs` | Local PDF directory (local mode) |
| `CSV_DIR` | `./data/csvs` | Local CSV directory (local mode) |
| `CHROMA_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `SCHEMA_CACHE_PATH` | `./memory/schema_memory.json` | Schema cache file |
| `EVAL_RESULTS_PATH` | `./outputs/eval_results.json` | Evaluation suite results |
| `REPORT_TRACE_PATH` | `./outputs/report_trace.json` | Workflow execution trace |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `PORT` | `8080` | Server port |
