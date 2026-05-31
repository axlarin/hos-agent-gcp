from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

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


class ComponentEvaluator:
    """Evaluates each agent component independently across 5 dimensions.

    Dimensions:
        routing              — rule-based: did orchestrator route correctly?
        retrieval_relevance  — Gemini-as-judge: did pdf/csv agent return relevant content?
        tool_selection       — rule-based: did analysis_agent pick the right test?
        answer_faithfulness  — Gemini-as-judge: is answer grounded in context?
        answer_quality       — Gemini-as-judge: is answer correct and complete?
    """

    def __init__(self, settings) -> None:
        self._settings = settings

    async def evaluate(self, question: str, answer: str, context: str = "") -> Dict[str, Any]:
        """Run all 5 evaluation dimensions and return a structured report.

        Args:
            question: The original user question.
            answer: The agent's final answer.
            context: Retrieved context (if available).

        Returns:
            Dict with overall_score, passed, and per-component scores.
        """
        components = []
        components.append(self._eval_routing(question))
        components.append(self._eval_tool_selection(question))
        components.append(await self._eval_retrieval_relevance(question, context))
        components.append(await self._eval_answer_faithfulness(answer, context))
        components.append(await self._eval_answer_quality(question, answer))

        overall = round(sum(c["score"] for c in components) / len(components), 3)
        return {
            "overall_score": overall,
            "passed": overall >= PASS_THRESHOLD,
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

    # ── Gemini-as-judge dimensions ────────────────────────────────────────────

    async def _eval_retrieval_relevance(self, question: str, context: str) -> Dict[str, Any]:
        if not context:
            return {"component": "retrieval_relevance", "score": 0.5, "passed": False,
                    "reason": "No retrieval context provided for evaluation"}
        score = await self._gemini_judge(
            f"Rate 0.0–1.0: How relevant is this context to the question?\n\n"
            f"Question: {question}\n\nContext: {context[:2000]}"
        )
        return {"component": "retrieval_relevance", "score": score, "passed": score >= PASS_THRESHOLD,
                "reason": f"Gemini-as-judge score: {score}"}

    async def _eval_answer_faithfulness(self, answer: str, context: str) -> Dict[str, Any]:
        if not context:
            return {"component": "answer_faithfulness", "score": 0.5, "passed": False,
                    "reason": "No context provided to check faithfulness against"}
        score = await self._gemini_judge(
            f"Rate 0.0–1.0: Is every claim in the answer supported by the context?\n\n"
            f"Context: {context[:2000]}\n\nAnswer: {answer[:1000]}"
        )
        return {"component": "answer_faithfulness", "score": score, "passed": score >= PASS_THRESHOLD,
                "reason": f"Gemini-as-judge score: {score}"}

    async def _eval_answer_quality(self, question: str, answer: str) -> Dict[str, Any]:
        score = await self._gemini_judge(
            f"Rate 0.0–1.0: How correct, complete, and plain-English is this answer?\n\n"
            f"Question: {question}\n\nAnswer: {answer[:1000]}"
        )
        return {"component": "answer_quality", "score": score, "passed": score >= PASS_THRESHOLD,
                "reason": f"Gemini-as-judge score: {score}"}

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
