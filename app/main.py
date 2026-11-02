"""FastAPI service that serves the two-stage intent classifier."""
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model import TwoStageClassifier

MODEL_DIR = os.getenv("MODEL_DIR", "models")
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["clf"] = TwoStageClassifier(MODEL_DIR)
    yield
    state.clear()


app = FastAPI(title="Intent Classifier API", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, examples=["what's my checking account balance"])


class PredictResponse(BaseModel):
    domain: str
    intent: str
    confidence: float
    domain_confidence: float
    latency_ms: float


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


@app.get("/health")
def health():
    clf: TwoStageClassifier = state["clf"]
    return {"status": "ok", "domains": len(clf.domains), "loaded_domains": clf.loaded_domains}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is empty")
    start = time.perf_counter()
    p = state["clf"].predict(text)
    return PredictResponse(domain=p.domain, intent=p.intent, confidence=round(p.confidence, 4),
                           domain_confidence=round(p.domain_confidence, 4),
                           latency_ms=round((time.perf_counter() - start) * 1000, 2))


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(req: BatchRequest):
    return [predict(PredictRequest(text=t)) for t in req.texts]
