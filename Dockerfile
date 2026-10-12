# Small CPU-only image that serves the fine-tuned DistilBERT model with FastAPI.
#   docker build -t intentbench-api .
#   docker run -p 8000:8000 intentbench-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# CPU-only PyTorch: the GPU build adds several GB the API never uses.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install transformers==5.19.0 fastapi==0.143.0 "uvicorn==0.54.0" pydantic==2.13.5

RUN useradd --create-home app

# Only what serving needs: the serve package, the model and the validation-chosen threshold.
# Owned by `app`, so the server can read them without running as root.
COPY --chown=app:app src/intentbench/__init__.py src/intentbench/
COPY --chown=app:app src/intentbench/serve src/intentbench/serve
COPY --chown=app:app reports/threshold.json reports/
COPY --chown=app:app models/distilbert models/distilbert

USER app

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "intentbench.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
