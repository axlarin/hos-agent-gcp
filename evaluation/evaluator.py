from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

from rag.embedder import embed

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 0.7

_ROUTING_KEYWORDS = {
    "pdf_agent": ["mean", "definition", "methodology", "survey", "designed", "pcs", "mcs", "what does"],
    "csv_agent": ["dataset", "columns", "available", "list", "what variables"],
    "analysis_agent": ["correlat", "regress", "predict", "compare", "associat", "chi-square", "mann-whitney"],
}

_TOOL_KEYWORDS = {
    "run_correlation_analysis": ["correlat"],
    "run_feature_importance": ["predict", "feature", "importance"],
    "run_logistic_regression": ["logistic", "regress"],
    "run_categorical_analysis": ["categorical", "chi-square", "frequency", "crosstab"],
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
        retrieval_relevance  — embedding similarity by default; Gemini-as-judge if deep=True
        tool_selection       — rule-based: did analysis_agent pick the right test?
        answer_faithfulness  — embedding similarity by default; Gemini-as-judge if deep=True
        answer_quality       — embedding similarity by default; Gemini-as-judge if deep=True
    """

    def __init__(self, settings) -> None:
        self._settings = settings

    async def evaluate(
        self, question: str, answer: str, context: str = "", deep: bool = False
    ) -> Dict[str, Any]:
        """Run all 5 evaluation dimensions and return a structured report.

        Args:
            question: The original user question.
            answer: The agent's final answer.
            context: Retrieved context (if available).
            deep: If True, use Gemini-as-judge for the 3 scored dimensions instead of
                  embedding similarity. Slower and uses API quota.

        Returns:
            Dict with overall_score, passed, per-component scores, and eval_mode.
        """
        components = []
        components.append(self._eval_routing(question))
        components.append(self._eval_tool_selection(question))
        components.append(await self._eval_retrieval_relevance(question, context, deep=deep))
        components.append(await self._eval_answer_faithfulness(answer, context, deep=deep))
        components.append(await self._eval_answer_quality(question, answer, deep=deep))

        overall = round(sum(c["score"] for c in components) / len(components), 3)
        return {
            "overall_score": overall,
            "passed": overall >= PASS_THRESHOLD,
            "eval_mode": "deep" if deep else "embedding",
            "components": components,
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
        return {"component": "tool_selection", "score": 0.5, "passed": False,
                "reason": "No analysis tool keyword matched — may be a non-analysis query"}

    # ── Similarity-scored dimensions (default) / Gemini-as-judge (deep=True) ──

    async def _eval_retrieval_relevance(
        self, question: str, context: str, deep: bool = False
    ) -> Dict[str, Any]:
        if not context:
            return {"component": "retrieval_relevance", "score": 0.5, "passed": False,
                    "reason": "No retrieval context provided for evaluation"}
        if deep:
            score = await self._gemini_judge(
                f"Rate 0.0–1.0: How relevant is this context to the question?\n\n"
                f"Question: {question}\n\nContext: {context[:2000]}"
            )
            reason = f"Gemini-as-judge score: {score}"
        else:
            score = _embed_sim(question, context)
            reason = f"Embedding similarity: {score}"
        return {"component": "retrieval_relevance", "score": score,
                "passed": score >= PASS_THRESHOLD, "reason": reason}

    async def _eval_answer_faithfulness(
        self, answer: str, context: str, deep: bool = False
    ) -> Dict[str, Any]:
        if not context:
            return {"component": "answer_faithfulness", "score": 0.5, "passed": False,
                    "reason": "No context provided to check faithfulness against"}
        if deep:
            score = await self._gemini_judge(
                f"Rate 0.0–1.0: Is every claim in the answer supported by the context?\n\n"
                f"Context: {context[:2000]}\n\nAnswer: {answer[:1000]}"
            )
            reason = f"Gemini-as-judge score: {score}"
        else:
            score = _embed_sim(context, answer)
            reason = f"Embedding similarity: {score}"
        return {"component": "answer_faithfulness", "score": score,
                "passed": score >= PASS_THRESHOLD, "reason": reason}

    async def _eval_answer_quality(
        self, question: str, answer: str, deep: bool = False
    ) -> Dict[str, Any]:
        if deep:
            score = await self._gemini_judge(
                f"Rate 0.0–1.0: How correct, complete, and plain-English is this answer?\n\n"
                f"Question: {question}\n\nAnswer: {answer[:1000]}"
            )
            reason = f"Gemini-as-judge score: {score}"
        else:
            score = _embed_sim(question, answer)
            reason = f"Embedding similarity: {score}"
        return {"component": "answer_quality", "score": score,
                "passed": score >= PASS_THRESHOLD, "reason": reason}

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
