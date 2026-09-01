# Backend image. Build from the repo root:
#   docker build -t risk-manager-api .
#
# models/ is NOT baked into the image — the trained artifacts
# (risk_model.joblib etc.) are mounted as a volume (see
# docker-compose.yml) or generated inside the running container with
# `python src/train_model.py` once data/ is populated. Same story for
# data/ itself: it's ~680MB and not something that belongs in an image.
#
# 3.13, not 3.11: requirements.txt's pinned numpy==2.5.1 ships no cp311
# wheel at all (3.12+ only) — verified against PyPI, not a guess. 3.13 is
# also the exact version every pin in this file was resolved/tested
# against locally, so it's the safer choice over the 3.12 floor.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
