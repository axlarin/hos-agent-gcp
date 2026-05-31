FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for sentence-transformers and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (data/ and .env are gitignored — not copied)
COPY agents/ agents/
COPY tools/ tools/
COPY rag/ rag/
COPY evaluation/ evaluation/
COPY config/ config/
COPY main.py .

# Create runtime directories that are populated at startup
RUN mkdir -p outputs/logs memory chroma_db

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
