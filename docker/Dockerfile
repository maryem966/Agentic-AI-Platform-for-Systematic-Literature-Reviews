# ============================================================
# SLR-Nexus — Dockerfile
# Image de l'application Python
# ============================================================

FROM python:3.11-slim AS base

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# ----------------------------------------------------------
# Stage : dependencies
# ----------------------------------------------------------
FROM base AS deps

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------
# Stage : development
# ----------------------------------------------------------
FROM deps AS development

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ----------------------------------------------------------
# Stage : production
# ----------------------------------------------------------
FROM deps AS production

COPY src/ ./src/

# Utilisateur non-root
RUN addgroup --system slr && adduser --system --group slr
USER slr

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
