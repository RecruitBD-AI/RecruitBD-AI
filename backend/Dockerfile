# ── Stage 1: install dependencies ────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency manifests first (layer-cache friendly)
COPY pyproject.toml ./

# Install CPU-only torch + all project deps
RUN uv sync --extra cpu --no-dev

# ── Stage 2: pre-bake the SBERT model ────────────────────────────────────────
FROM builder AS model-cache

# Download the model into the image at build time so cold starts are instant
RUN uv run --extra cpu python -c "from sentence_transformers import SentenceTransformer;SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# ── Stage 3: final runtime image ──────────────────────────────────────────────
FROM model-cache AS runtime

# Copy application source
COPY . .

# Generate the job index (this will scrape fresh jobs if no CSV is found)
RUN uv run --extra cpu -m services.matcher.build_index

EXPOSE 8000

CMD ["sh", "-c", "uv run --extra cpu uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
