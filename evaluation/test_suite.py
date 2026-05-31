"""Ground-truth test cases covering all agent types and all tool types."""

TEST_CASES = [
    # ── PDF agent — definitions ───────────────────────────────────────────────
    {
        "question": "What does PCS mean?",
        "expected_agent": "pdf_agent",
        "expected_tool": "search_pdf_guidance",
        "ground_truth_keywords": ["physical component summary", "sf-36"],
    },
    {
        "question": "What does MCS stand for?",
        "expected_agent": "pdf_agent",
        "expected_tool": "search_pdf_guidance",
        "ground_truth_keywords": ["mental component summary"],
    },
    {
        "question": "What is the HOS survey methodology?",
        "expected_agent": "pdf_agent",
        "expected_tool": "search_pdf_guidance",
        "ground_truth_keywords": ["medicare", "managed care", "cohort"],
    },
    {
        "question": "What does B25FRMPREV mean and what values does it have?",
        "expected_agent": "pdf_agent",
        "expected_tool": "get_column_info",
        "ground_truth_keywords": ["prevent falls", "yes", "no"],
    },
    # ── CSV agent — schema + listing ─────────────────────────────────────────
    {
        "question": "What datasets are available?",
        "expected_agent": "csv_agent",
        "expected_tool": "list_datasets",
        "ground_truth_keywords": ["c25a", "c25b"],
    },
    {
        "question": "What columns does c25a have?",
        "expected_agent": "csv_agent",
        "expected_tool": "get_column_info",
        "ground_truth_keywords": ["age", "sex"],
    },
    # ── Analysis agent — correlation ─────────────────────────────────────────
    {
        "question": "What variables are most correlated with AGE in c25a?",
        "expected_agent": "analysis_agent",
        "expected_tool": "run_correlation_analysis",
        "ground_truth_keywords": ["pearson", "correlation"],
    },
    # ── Analysis agent — feature importance ──────────────────────────────────
    {
        "question": "What predicts general health status in c25a?",
        "expected_agent": "analysis_agent",
        "expected_tool": "run_feature_importance",
        "ground_truth_keywords": ["importance", "random forest"],
    },
    # ── Analysis agent — categorical ─────────────────────────────────────────
    {
        "question": "Show the frequency distribution of health status in c25a",
        "expected_agent": "analysis_agent",
        "expected_tool": "run_categorical_analysis",
        "ground_truth_keywords": ["frequency"],
    },
    {
        "question": "Is there a significant association between AGE and health status in c25a?",
        "expected_agent": "analysis_agent",
        "expected_tool": "run_categorical_analysis",
        "ground_truth_keywords": ["chi-square", "cramér"],
    },
    # ── Analysis agent — group comparison ────────────────────────────────────
    {
        "question": "Compare health scores between males and females in c25a",
        "expected_agent": "analysis_agent",
        "expected_tool": "run_group_comparison",
        "ground_truth_keywords": ["mann-whitney"],
    },
    {
        "question": "Compare health scores across all age groups in c25a",
        "expected_agent": "analysis_agent",
        "expected_tool": "run_group_comparison",
        "ground_truth_keywords": ["kruskal-wallis"],
    },
    # ── Analysis agent — regression ──────────────────────────────────────────
    {
        "question": "Run a logistic regression predicting general health from age and sex in c25a",
        "expected_agent": "analysis_agent",
        "expected_tool": "run_logistic_regression",
        "ground_truth_keywords": ["logistic", "coefficient", "odds ratio"],
    },
    # ── Multi-turn / follow-up ────────────────────────────────────────────────
    {
        "question": "Now do the same analysis for the follow-up cohort",
        "expected_agent": "analysis_agent",
        "expected_tool": "run_correlation_analysis",
        "ground_truth_keywords": ["c25b"],
        "note": "Requires conversation context from a prior turn — run after a c25a query",
    },
]
