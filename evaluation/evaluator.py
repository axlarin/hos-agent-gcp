from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Literal, Union

from rag.embedder import embed

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 0.7

DeepMode = Union[bool, Literal["auto"]]

# deep="auto" escalation thresholds — calibrated from suite results where embedding
# scoring clearly undervalued a correct answer (PCS/MCS acronym definitions scoring
# retrieval_relevance/answer_faithfulness well below their peers; a valid
# logistic-regression clarifying question scoring answer_quality well below its
# peers). rr/af only escalate when answer_quality's embedding score is ALSO
# reasonably high (> _AUTO_AQ_GATE) — embeddings and observed answer quality
# disagreeing is the actual signal worth spending a Gemini call on; a low rr/af
# alongside a low aq usually just means the answer really is bad, which embedding
# scoring already gets right. aq escalates unconditionally below its own
# threshold since there's no second signal to gate it against.
_AUTO_RR_THRESHOLD = 0.5
_AUTO_AF_THRESHOLD = 0.6
_AUTO_AQ_THRESHOLD = 0.5
_AUTO_AQ_GATE = 0.55

# Gemini-judge prompts truncate context to this many characters. Context can be
# the concatenation of multiple retrieval-tool outputs (e.g. search_pdf_guidance
# returns up to 5 chunks of ~512 chars each — already ~2700 chars before any
# get_column_info text is appended). A too-small window silently drops whichever
# tool's output landed later in the concatenation, causing Gemini to judge
# "unfaithful" for claims it was simply never shown. 8000 chars (~2000 tokens)
# comfortably covers several retrieval calls and is cheap for gemini-2.5-flash.
_GEMINI_CONTEXT_CHARS = 8000

_ROUTING_KEYWORDS = {
    "pdf_agent": ["mean", "definition", "methodology", "survey", "designed", "pcs", "mcs", "what does"],
    "csv_agent": ["dataset", "columns", "available", "list", "what variables"],
    "analysis_agent": [
        "correlat", "regress", "predict", "compare", "associat", "chi-square", "mann-whitney",
        "frequency",
    ],
}

_TOOL_KEYWORDS = {
    "run_correlation_analysis": ["correlat"],
    "run_feature_importance": ["predict", "feature", "importance"],
    "run_logistic_regression": ["logistic", "regress"],
    "run_categorical_analysis": ["categorical", "chi-square", "frequency", "crosstab", "associat"],
    "run_group_comparison": ["group", "compare", "mann-whitney", "kruskal"],
}


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _embed_sim(text_a: str, text_b: str) -> float:
    vecs = embed([text_a[:1000], text_b[:1000]])
    return round(_cosine_sim(vecs[0], vecs[1]), 3)


class ComponentEvaluator:
    """Evaluates each agent component independently across 5 dimensions.

    Dimensions:
        routing              — rule-based: did orchestrator route correctly?
        retrieval_relevance  — embedding / Gemini / auto-escalated, see deep modes below
        tool_selection       — rule-based: did analysis_agent pick the right test?
        answer_faithfulness  — embedding / Gemini / auto-escalated, see deep modes below
        answer_quality       — embedding / Gemini / auto-escalated, see deep modes below

    deep modes for the 3 scored dimensions:
        False  — embedding similarity only (fast, deterministic, quota-free). Default.
        "auto" — embedding similarity first; escalate to a single Gemini-judge call
                 only when the embedding score is low AND disagrees with answer_quality
                 (see module-level _AUTO_* thresholds). Cheap, catches embedding's
                 known blind spots (acronym/short-query retrieval, terse-but-correct
                 clarifying answers).
        True   — Gemini-as-judge on all 3 dimensions, every case. Slowest, uses the
                 most quota.
    """

    def __init__(self, settings) -> None:
        self._settings = settings

    async def evaluate(
        self, question: str, answer: str, context: str = "", deep: DeepMode = False
    ) -> Dict[str, Any]:
        """Run all 5 evaluation dimensions and return a structured report.

        Args:
            question: The original user question.
            answer: The agent's final answer.
            context: Retrieved context (if available).
            deep: False (embedding only), "auto" (embedding + selective Gemini
                  escalation), or True (Gemini-as-judge on every case).

        Returns:
            Dict with overall_score, passed, per-component scores, eval_mode,
            and an instrumentation block reporting embedding vs Gemini usage.
        """
        components = []
        components.append(self._eval_routing(question))
        components.append(self._eval_tool_selection(question))

        # answer_quality's embedding score is computed first because auto-mode
        # rr/af escalation gates on it — see _AUTO_AQ_GATE rationale above.
        aq_embed = _embed_sim(question, answer) if answer else 0.0

        rr_comp = await self._eval_retrieval_relevance(question, context, deep=deep, aq_embed=aq_embed)
        af_comp = await self._eval_answer_faithfulness(answer, context, deep=deep, aq_embed=aq_embed)
        aq_comp = await self._eval_answer_quality(question, answer, deep=deep, embed_score=aq_embed)
        components.extend([rr_comp, af_comp, aq_comp])

        overall = round(sum(c["score"] for c in components) / len(components), 3)
        escalated = [c for c in (rr_comp, af_comp, aq_comp) if c.get("escalated")]

        return {
            "overall_score": overall,
            "passed": overall >= PASS_THRESHOLD,
            "eval_mode": deep if isinstance(deep, str) else ("deep" if deep else "embedding"),
            "components": components,
            "instrumentation": {
                "embedding_evaluations": 3 - len(escalated),
                "gemini_escalations": len(escalated),
                "escalation_reasons": [f"{c['component']}: {c['reason']}" for c in escalated],
                "gemini_call_count": len(escalated),
            },
        }

    # ── Rule-based dimensions ─────────────────────────────────────────────────

    def _eval_routing(self, question: str) -> Dict[str, Any]:
        q = question.lower()
        for agent, keywords in _ROUTING_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                return {"component": "routing", "score": 1.0, "passed": True,
                        "reason": f"Question matches {agent} keywords"}
        return {"component": "routing", "score": 0.5, "passed": False,
                "reason": "Routing pattern not matched — verify orchestrator description"}

    def _eval_tool_selection(self, question: str) -> Dict[str, Any]:
        q = question.lower()
        for tool, keywords in _TOOL_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                return {"component": "tool_selection", "score": 1.0, "passed": True,
                        "reason": f"Question matches {tool}"}
        # Only penalise when the question is clearly an analysis query that should
        # have matched a tool. Non-analysis questions (PDF lookups, dataset listing)
        # don't need an analysis tool, so scoring them 0.5 is misleading.
        if any(kw in q for kw in _ROUTING_KEYWORDS["analysis_agent"]):
            return {"component": "tool_selection", "score": 0.0, "passed": False,
                    "reason": "Analysis question but no tool keyword matched"}
        return {"component": "tool_selection", "score": 1.0, "passed": True,
                "reason": "Non-analysis query — tool selection not required"}

    # ── Similarity-scored dimensions — embedding / Gemini / auto-escalated ────

    async def _eval_retrieval_relevance(
        self, question: str, context: str, deep: DeepMode = False, aq_embed: float = 0.0
    ) -> Dict[str, Any]:
        if not context:
            return {"component": "retrieval_relevance", "score": 0.5, "passed": False,
                    "escalated": False, "reason": "No retrieval context provided for evaluation"}

        embed_score = _embed_sim(question, context)
        prompt = (
            f"Rate 0.0–1.0: How relevant is this context to the question?\n\n"
            f"Question: {question}\n\nContext: {context[:_GEMINI_CONTEXT_CHARS]}"
        )

        if deep is True:
            score = await self._gemini_judge(prompt)
            return {"component": "retrieval_relevance", "score": score,
                    "passed": score >= PASS_THRESHOLD, "escalated": True,
                    "reason": f"Gemini-as-judge score: {score}"}

        if deep == "auto" and embed_score < _AUTO_RR_THRESHOLD and aq_embed > _AUTO_AQ_GATE:
            score = await self._gemini_judge(prompt)
            return {"component": "retrieval_relevance", "score": score,
                    "passed": score >= PASS_THRESHOLD, "escalated": True,
                    "reason": (f"Auto-escalated (embedding={embed_score} < {_AUTO_RR_THRESHOLD} "
                               f"and answer_quality={aq_embed} > {_AUTO_AQ_GATE}): "
                               f"Gemini score {score}")}

        return {"component": "retrieval_relevance", "score": embed_score,
                "passed": embed_score >= PASS_THRESHOLD, "escalated": False,
                "reason": f"Embedding similarity: {embed_score}"}

    async def _eval_answer_faithfulness(
        self, answer: str, context: str, deep: DeepMode = False, aq_embed: float = 0.0
    ) -> Dict[str, Any]:
        if not context:
            return {"component": "answer_faithfulness", "score": 0.5, "passed": False,
                    "escalated": False, "reason": "No context provided to check faithfulness against"}

        embed_score = _embed_sim(context, answer)
        prompt = (
            f"Rate 0.0–1.0: Is every claim in the answer supported by the context?\n\n"
            f"Context: {context[:_GEMINI_CONTEXT_CHARS]}\n\nAnswer: {answer[:1000]}"
        )

        if deep is True:
            score = await self._gemini_judge(prompt)
            return {"component": "answer_faithfulness", "score": score,
                    "passed": score >= PASS_THRESHOLD, "escalated": True,
                    "reason": f"Gemini-as-judge score: {score}"}

        if deep == "auto" and embed_score < _AUTO_AF_THRESHOLD and aq_embed > _AUTO_AQ_GATE:
            score = await self._gemini_judge(prompt)
            return {"component": "answer_faithfulness", "score": score,
                    "passed": score >= PASS_THRESHOLD, "escalated": True,
                    "reason": (f"Auto-escalated (embedding={embed_score} < {_AUTO_AF_THRESHOLD} "
                               f"and answer_quality={aq_embed} > {_AUTO_AQ_GATE}): "
                               f"Gemini score {score}")}

        return {"component": "answer_faithfulness", "score": embed_score,
                "passed": embed_score >= PASS_THRESHOLD, "escalated": False,
                "reason": f"Embedding similarity: {embed_score}"}

    async def _eval_answer_quality(
        self, question: str, answer: str, deep: DeepMode = False, embed_score: float | None = None
    ) -> Dict[str, Any]:
        if embed_score is None:
            embed_score = _embed_sim(question, answer)
        prompt = (
            f"Rate 0.0–1.0: How correct, complete, and plain-English is this answer?\n\n"
            f"Question: {question}\n\nAnswer: {answer[:1000]}"
        )

        if deep is True:
            score = await self._gemini_judge(prompt)
            return {"component": "answer_quality", "score": score,
                    "passed": score >= PASS_THRESHOLD, "escalated": True,
                    "reason": f"Gemini-as-judge score: {score}"}

        if deep == "auto" and embed_score < _AUTO_AQ_THRESHOLD:
            score = await self._gemini_judge(prompt)
            return {"component": "answer_quality", "score": score,
                    "passed": score >= PASS_THRESHOLD, "escalated": True,
                    "reason": f"Auto-escalated (embedding={embed_score} < {_AUTO_AQ_THRESHOLD}): "
                              f"Gemini score {score}"}

        return {"component": "answer_quality", "score": embed_score,
                "passed": embed_score >= PASS_THRESHOLD, "escalated": False,
                "reason": f"Embedding similarity: {embed_score}"}

    async def _gemini_judge(self, prompt: str) -> float:
        """Call Gemini to score a prompt. Returns 0.0–1.0."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._settings.google_api_key)
            model = genai.GenerativeModel(self._settings.evaluator_model)
            response = model.generate_content(
                f"{prompt}\n\nRespond with only a number between 0.0 and 1.0."
            )
            text = response.text.strip()
            score = float(text)
            return max(0.0, min(1.0, score))
        except Exception as exc:
            logger.warning("Gemini judge failed: %s", exc)
            return 0.5
