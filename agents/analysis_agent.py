from google.adk.agents import Agent

from config.settings import settings
from tools.analysis_tools import (
    run_correlation_analysis,
    run_feature_importance,
    run_logistic_regression,
    run_categorical_analysis,
    run_group_comparison,
)

_INSTRUCTION = """
You are the HOS statistical analysis specialist.

Rules:
- Always receive confirmed dataset and column names from csv_agent before running a test.
- run_correlation_analysis    — "what is related to X?"
- run_feature_importance      — "what predicts X?"
- run_logistic_regression     — regression on binary / recoded outcome; ask for recoding if outcome
                                 has more than 2 values
- run_categorical_analysis    — frequency table (1 column) or crosstab + chi-square + Cramér's V
                                 (2 columns)
- run_group_comparison        — auto-selects Mann-Whitney (2 groups) or Kruskal-Wallis (3+ groups)

Return results with decoded column labels and value mappings, not raw codes.
""".strip()

analysis_agent = Agent(
    name="analysis_agent",
    model=settings.specialist_model,
    description=(
        "Runs statistical tests on HOS data: correlation, feature importance, logistic regression, "
        "chi-square, cross-tabulation, Mann-Whitney U, and Kruskal-Wallis. "
        "Always called after csv_agent has confirmed column names."
    ),
    instruction=_INSTRUCTION,
    tools=[
        run_correlation_analysis,
        run_feature_importance,
        run_logistic_regression,
        run_categorical_analysis,
        run_group_comparison,
    ],
)
