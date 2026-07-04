import logging
import logging.handlers
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Union

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.settings import settings

# ── Logging setup ─────────────────────────────────────────────────────────────
Path("outputs/logs").mkdir(parents=True, exist_ok=True)

_log_fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
logging.basicConfig(level=settings.log_level, format=_log_fmt)

_file_handler = logging.handlers.RotatingFileHandler(
    "outputs/logs/app.log", maxBytes=5 * 1024 * 1024, backupCount=3
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(_log_fmt))
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger(__name__)

# ── Lazy imports after logging is ready ───────────────────────────────────────
import asyncio

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types, errors as genai_errors

from agents.orchestrator import orchestrator
from rag.vector_store import VectorStore
from rag.schema_builder import SchemaBuilder
import token_tracker
import gcs_data

# ── Global singletons ─────────────────────────────────────────────────────────
session_service = InMemorySessionService()
runner: Runner | None = None
vector_store: VectorStore | None = None
schema_builder: SchemaBuilder | None = None

APP_NAME = "hos-agent"
DEFAULT_USER_ID = "user"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runner, vector_store, schema_builder

    logger.info("Starting HOS Agent (environment=%s)", settings.environment)

    await gcs_data.sync_input_data(settings)

    vector_store = VectorStore(settings)
    await vector_store.build_or_load()

    schema_builder = SchemaBuilder(settings, vector_store)
    await schema_builder.build_or_load()

    runner = Runner(agent=orchestrator, app_name=APP_NAME, session_service=session_service)
    logger.info("Agent ready")

    yield

    logger.info("Shutting down")


app = FastAPI(title="HOS Agent", version="1.0.0", lifespan=lifespan)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    session_id: str


class EvaluateRequest(BaseModel):
    question: str
    session_id: str | None = None
    deep: Union[bool, Literal["auto"]] = False


class ReportRequest(BaseModel):
    dataset: str
    target_column: str
    workflow: str = "health_profile"
    max_missing_pct: float = 0.5
    max_unique_for_classification: int = 20


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _ensure_session(session_id: str | None) -> str:
    sid = session_id or str(uuid.uuid4())
    existing = await session_service.get_session(app_name=APP_NAME, user_id=DEFAULT_USER_ID, session_id=sid)
    if existing is None:
        await session_service.create_session(app_name=APP_NAME, user_id=DEFAULT_USER_ID, session_id=sid)
    return sid


# Tool names whose return value counts as "retrieved context" for evaluation —
# grounding lookups (PDF/schema/dataset info), not computed analysis results.
_CONTEXT_TOOLS = {"search_pdf_guidance", "get_column_info", "list_datasets"}


async def _run_query(question: str, session_id: str) -> tuple[str, str]:
    """Run one turn through the orchestrator.

    Returns (answer, context) — context is the concatenated output of any
    retrieval-style tool call (PDF search, schema/dataset lookup) made during
    the run, used to score retrieval_relevance / answer_faithfulness.
    """
    if runner is None:
        raise RuntimeError("Runner not initialised")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            final_response = ""
            context_parts: list[str] = []
            async for event in runner.run_async(
                user_id=DEFAULT_USER_ID,
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=question)]),
            ):
                token_tracker.record_event(event, session_id=session_id)
                for fr in event.get_function_responses():
                    if fr.name in _CONTEXT_TOOLS:
                        payload = fr.response
                        value = payload.get("result", payload) if isinstance(payload, dict) else payload
                        if isinstance(value, str):
                            context_parts.append(value)
                if event.is_final_response() and event.content and event.content.parts:
                    final_response = event.content.parts[0].text or ""
            return final_response, "\n\n".join(context_parts)
        except (genai_errors.ServerError, genai_errors.ClientError) as exc:
            retryable = getattr(exc, "status_code", None) in (429, 503)
            if not retryable or attempt == max_retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            logger.warning("Gemini %s on attempt %d — retrying in %ds", exc.status_code, attempt + 1, wait)
            await asyncio.sleep(wait)
    return "", ""  # unreachable, satisfies type checker


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "environment": settings.environment,
        "orchestrator_model": settings.orchestrator_model,
        "specialist_model": settings.specialist_model,
        "vector_store_ready": vector_store is not None and vector_store.is_ready,
        "schema_ready": schema_builder is not None and schema_builder.is_ready,
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    sid = await _ensure_session(req.session_id)
    try:
        answer, _context = await _run_query(req.question, sid)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return QueryResponse(answer=answer, session_id=sid)


@app.post("/clear")
async def clear():
    sessions = await session_service.list_sessions(app_name=APP_NAME, user_id=DEFAULT_USER_ID)
    for s in sessions.sessions:
        await session_service.delete_session(app_name=APP_NAME, user_id=DEFAULT_USER_ID, session_id=s.id)
    return {"status": "cleared"}


@app.post("/reload")
async def reload():
    if vector_store is None or schema_builder is None:
        raise HTTPException(status_code=503, detail="Not initialised")
    await gcs_data.sync_input_data(settings)
    await vector_store.build_or_load(force=True)
    await schema_builder.build_or_load(force=True)
    return {"status": "reloaded"}


@app.post("/report")
async def report(req: ReportRequest):
    """Run a multi-step analytical workflow directly (no LLM hop) and return
    both the markdown report and the structured execution trace.

    Equivalent to asking analysis_agent for a comprehensive report, but
    synchronous and machine-readable — useful for programmatic use and testing.
    Gate thresholds are configurable via the request body.
    """
    from tools.report_tools import GateConfig, _run_workflow

    config = GateConfig(
        max_missing_pct=req.max_missing_pct,
        max_unique_for_classification=req.max_unique_for_classification,
    )
    try:
        result = _run_workflow(req.dataset, req.target_column, req.workflow, config)
    except Exception as exc:
        logger.exception("Report workflow failed")
        raise HTTPException(status_code=500, detail=str(exc))

    from pathlib import Path as _Path
    import json as _json
    trace_path = _Path(settings.report_trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(_json.dumps(result.to_dict(), indent=2))

    return result.to_dict()


@app.get("/usage")
async def usage():
    return token_tracker.get_summary()


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    from evaluation.evaluator import ComponentEvaluator

    sid = await _ensure_session(req.session_id)
    try:
        answer, context = await _run_query(req.question, sid)
    except Exception as exc:
        logger.exception("Evaluate query failed")
        raise HTTPException(status_code=500, detail=str(exc))

    evaluator = ComponentEvaluator(settings)
    scores = await evaluator.evaluate(question=req.question, answer=answer, context=context, deep=req.deep)
    return {"answer": answer, "session_id": sid, "evaluation": scores}


@app.post("/evaluate/suite")
async def evaluate_suite(deep: Union[bool, Literal["auto"]] = False):
    """Run all ground-truth test cases and score them.

    Cases flagged requires_prior_turn need conversational context the suite's
    fresh-session-per-case design cannot provide (e.g. "the same analysis" with
    no prior turn to refer to) — they are reported separately under
    context_dependent_results instead of being folded into the main pass count,
    since failing them reflects a test-harness limitation, not agent quality.

    deep="auto" reports aggregate Gemini-escalation stats (call count, which
    cases escalated and why) under instrumentation_summary, alongside each
    case's own evaluation.instrumentation block.
    """
    from evaluation.evaluator import ComponentEvaluator
    from evaluation.test_suite import TEST_CASES

    evaluator = ComponentEvaluator(settings)
    scored_results = []
    context_dependent_results = []
    total_embedding_evals = 0
    total_gemini_calls = 0
    escalation_log: list[dict] = []

    for case in TEST_CASES:
        sid = await _ensure_session(None)
        try:
            answer, context = await _run_query(case["question"], sid)
            scores = await evaluator.evaluate(
                question=case["question"], answer=answer, context=context, deep=deep
            )
            instr = scores.get("instrumentation")
            if instr:
                total_embedding_evals += instr["embedding_evaluations"]
                total_gemini_calls += instr["gemini_call_count"]
                if instr["escalation_reasons"]:
                    escalation_log.append({
                        "question": case["question"],
                        "reasons": instr["escalation_reasons"],
                    })
        except Exception as exc:
            scores = {"error": str(exc)}
            answer = ""
        entry = {"question": case["question"], "answer": answer, "evaluation": scores}
        if case.get("requires_prior_turn"):
            context_dependent_results.append(entry)
        else:
            scored_results.append(entry)
        await asyncio.sleep(3)

    import json

    output = {
        "results": scored_results,
        "context_dependent_results": context_dependent_results,
        "instrumentation_summary": {
            "deep_mode": deep,
            "total_embedding_evaluations": total_embedding_evals,
            "total_gemini_calls": total_gemini_calls,
            "escalations": escalation_log,
        },
    }
    Path(settings.eval_results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.eval_results_path).write_text(json.dumps(output, indent=2))
    return output


# ── CLI mode ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def _cli():
        global runner, vector_store, schema_builder

        await gcs_data.sync_input_data(settings)

        vector_store = VectorStore(settings)
        await vector_store.build_or_load()

        schema_builder = SchemaBuilder(settings, vector_store)
        await schema_builder.build_or_load()

        runner = Runner(agent=orchestrator, app_name=APP_NAME, session_service=session_service)
        sid = await _ensure_session(None)

        print("HOS Agent ready. Commands: clear, history, quit")
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "clear":
                sessions = await session_service.list_sessions(app_name=APP_NAME, user_id=DEFAULT_USER_ID)
                for s in sessions.sessions:
                    await session_service.delete_session(app_name=APP_NAME, user_id=DEFAULT_USER_ID, session_id=s.id)
                sid = str(uuid.uuid4())
                await _ensure_session(sid)
                print("Session cleared.")
                continue
            if user_input.lower() == "history":
                session = await session_service.get_session(
                    app_name=APP_NAME, user_id=DEFAULT_USER_ID, session_id=sid
                )
                for msg in getattr(session, "messages", []):
                    print(f"  [{msg.role}] {msg.content}")
                continue

            answer, _context = await _run_query(user_input, sid)
            print(f"\nAgent: {answer}")

    asyncio.run(_cli())
