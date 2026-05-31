import logging
import logging.handlers
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

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
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

from agents.orchestrator import orchestrator
from rag.vector_store import VectorStore
from rag.schema_builder import SchemaBuilder

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_session(session_id: str | None) -> str:
    sid = session_id or str(uuid.uuid4())
    try:
        session_service.get_session(app_name=APP_NAME, user_id=DEFAULT_USER_ID, session_id=sid)
    except Exception:
        session_service.create_session(app_name=APP_NAME, user_id=DEFAULT_USER_ID, session_id=sid)
    return sid


async def _run_query(question: str, session_id: str) -> str:
    if runner is None:
        raise RuntimeError("Runner not initialised")
    response = await runner.run_async(
        user_id=DEFAULT_USER_ID,
        session_id=session_id,
        new_message=question,
    )
    return str(response)


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
    sid = _ensure_session(req.session_id)
    try:
        answer = await _run_query(req.question, sid)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return QueryResponse(answer=answer, session_id=sid)


@app.post("/clear")
async def clear():
    session_service.clear_all_sessions()
    return {"status": "cleared"}


@app.post("/reload")
async def reload():
    if vector_store is None or schema_builder is None:
        raise HTTPException(status_code=503, detail="Not initialised")
    await vector_store.build_or_load(force=True)
    await schema_builder.build_or_load(force=True)
    return {"status": "reloaded"}


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    from evaluation.evaluator import ComponentEvaluator

    sid = _ensure_session(req.session_id)
    try:
        answer = await _run_query(req.question, sid)
    except Exception as exc:
        logger.exception("Evaluate query failed")
        raise HTTPException(status_code=500, detail=str(exc))

    evaluator = ComponentEvaluator(settings)
    scores = await evaluator.evaluate(question=req.question, answer=answer)
    return {"answer": answer, "session_id": sid, "evaluation": scores}


@app.post("/evaluate/suite")
async def evaluate_suite():
    from evaluation.evaluator import ComponentEvaluator
    from evaluation.test_suite import TEST_CASES

    evaluator = ComponentEvaluator(settings)
    results = []
    for case in TEST_CASES:
        sid = _ensure_session(None)
        try:
            answer = await _run_query(case["question"], sid)
            scores = await evaluator.evaluate(question=case["question"], answer=answer)
        except Exception as exc:
            scores = {"error": str(exc)}
            answer = ""
        results.append({"question": case["question"], "answer": answer, "evaluation": scores})

    import json

    Path(settings.eval_results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.eval_results_path).write_text(json.dumps(results, indent=2))
    return {"results": results}


# ── CLI mode ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def _cli():
        global runner, vector_store, schema_builder

        vector_store = VectorStore(settings)
        await vector_store.build_or_load()

        schema_builder = SchemaBuilder(settings, vector_store)
        await schema_builder.build_or_load()

        runner = Runner(agent=orchestrator, app_name=APP_NAME, session_service=session_service)
        sid = str(uuid.uuid4())

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
                session_service.clear_all_sessions()
                sid = str(uuid.uuid4())
                print("Session cleared.")
                continue
            if user_input.lower() == "history":
                session = session_service.get_session(
                    app_name=APP_NAME, user_id=DEFAULT_USER_ID, session_id=sid
                )
                for msg in getattr(session, "messages", []):
                    print(f"  [{msg.role}] {msg.content}")
                continue

            answer = await _run_query(user_input, sid)
            print(f"\nAgent: {answer}")

    asyncio.run(_cli())
