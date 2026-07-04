from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Gemini API — from AI Studio
    google_api_key: str = Field("", alias="GOOGLE_API_KEY")
    google_genai_use_vertexai: bool = Field(False, alias="GOOGLE_GENAI_USE_VERTEXAI")

    # Models
    orchestrator_model: str = Field("gemini-2.5-pro", alias="ORCHESTRATOR_MODEL")
    specialist_model: str = Field("gemini-2.5-flash", alias="SPECIALIST_MODEL")
    evaluator_model: str = Field("gemini-2.5-flash", alias="EVALUATOR_MODEL")

    # Environment switch — "local" reads from disk, "cloud" reads from GCS
    environment: str = Field("local", alias="ENVIRONMENT")

    # Local file paths
    pdf_dir: str = Field("./data/pdfs", alias="PDF_DIR")
    csv_dir: str = Field("./data/csvs", alias="CSV_DIR")
    chroma_dir: str = Field("./chroma_db", alias="CHROMA_DIR")
    schema_cache_path: str = Field("./memory/schema_memory.json", alias="SCHEMA_CACHE_PATH")
    eval_results_path: str = Field("./outputs/eval_results.json", alias="EVAL_RESULTS_PATH")
    report_trace_path: str = Field("./outputs/report_trace.json", alias="REPORT_TRACE_PATH")

    # Cloud (GCS) settings — only used when environment=cloud
    gcs_bucket: str = Field("", alias="GCS_BUCKET")

    # Logging
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Server
    port: int = Field(8080, alias="PORT")

    @property
    def is_cloud(self) -> bool:
        return self.environment == "cloud"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "populate_by_name": True}


settings = Settings()
