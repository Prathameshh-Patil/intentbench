"""HTTP API: POST /classify, GET /health, GET /model-info.

Run locally:  uvicorn intentbench.serve.app:app --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from intentbench.serve.model import Predictor
from intentbench.serve.schemas import (
    MAX_TEXT_CHARS,
    ClassifyRequest,
    ClassifyResponse,
    ModelInfo,
)


def create_app(predictor_factory=Predictor) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor_factory()  # loaded once, before the first request
        yield

    app = FastAPI(
        title="IntentBench",
        description="Classify a short message into one of 150 intents, or out_of_scope.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/model-info", response_model=ModelInfo)
    def model_info(request: Request) -> ModelInfo:
        p = request.app.state.predictor
        return ModelInfo(
            model=p.source,
            base_model=p.base_model,
            num_labels=len(p.labels),
            oos_threshold=p.threshold,
            max_text_chars=MAX_TEXT_CHARS,
        )

    # A plain `def` endpoint runs in a worker thread, so the model call doesn't block the server.
    @app.post("/classify", response_model=ClassifyResponse)
    def classify(body: ClassifyRequest, request: Request) -> dict:
        return request.app.state.predictor.classify(body.text)

    return app


app = create_app()
