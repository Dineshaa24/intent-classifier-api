"""Two-stage intent classifier: first the domain, then the intent inside it.

One flat 150-way model keeps confusing intents that look alike across domains
("check balance" for a bank vs. a gift card). Splitting the problem lets each
domain model learn a small, clean label set, and only the domain models that
are actually needed get loaded into memory.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from app.preprocess import clean

OUT_OF_SCOPE = "out_of_scope"


def build_pipeline() -> Pipeline:
    # Word n-grams catch phrases, char n-grams catch typos and word variants.
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=clean, analyzer="char_wb", ngram_range=(2, 5),
                                 sublinear_tf=True, min_df=2)),
    ])
    return Pipeline([("features", features), ("clf", LogisticRegression(C=10, max_iter=2000))])


@dataclass
class Prediction:
    domain: str
    intent: str
    confidence: float
    domain_confidence: float


class TwoStageClassifier:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        meta = json.loads((self.model_dir / "meta.json").read_text())
        self.threshold: float = meta["threshold"]
        self.domains: list[str] = meta["domains"]
        self.domain_model: Pipeline = joblib.load(self.model_dir / "domain.joblib")
        self._intent_models: dict[str, Pipeline] = {}
        self._lock = threading.Lock()

    @property
    def loaded_domains(self) -> list[str]:
        return sorted(self._intent_models)

    def _intent_model(self, domain: str) -> Pipeline:
        # Load each domain's model on first use, so memory grows only with traffic.
        model = self._intent_models.get(domain)
        if model is None:
            with self._lock:
                model = self._intent_models.get(domain)
                if model is None:
                    model = joblib.load(self.model_dir / "intents" / f"{domain}.joblib")
                    self._intent_models[domain] = model
        return model

    def predict(self, text: str) -> Prediction:
        domain_probs = self.domain_model.predict_proba([text])[0]
        best = int(domain_probs.argmax())
        domain, domain_conf = str(self.domain_model.classes_[best]), float(domain_probs[best])

        intent_model = self._intent_model(domain)
        intent_probs = intent_model.predict_proba([text])[0]
        i = int(intent_probs.argmax())
        intent, intent_conf = str(intent_model.classes_[i]), float(intent_probs[i])

        # Overall confidence is the chance both stages are right.
        confidence = domain_conf * intent_conf
        if confidence < self.threshold:
            return Prediction(OUT_OF_SCOPE, OUT_OF_SCOPE, confidence, domain_conf)
        return Prediction(domain, intent, confidence, domain_conf)


def train(rows: list[tuple[str, str]], intent_to_domain: dict[str, str], model_dir: str | Path,
          threshold: float) -> None:
    """Fit the domain model and one intent model per domain, then save them."""
    model_dir = Path(model_dir)
    (model_dir / "intents").mkdir(parents=True, exist_ok=True)

    texts = [t for t, _ in rows]
    domain_model = build_pipeline().fit(texts, [intent_to_domain[i] for _, i in rows])
    joblib.dump(domain_model, model_dir / "domain.joblib")

    domains = sorted(set(intent_to_domain.values()))
    for domain in domains:
        subset = [(t, i) for t, i in rows if intent_to_domain[i] == domain]
        model = build_pipeline().fit([t for t, _ in subset], [i for _, i in subset])
        joblib.dump(model, model_dir / "intents" / f"{domain}.joblib")

    (model_dir / "meta.json").write_text(json.dumps({"threshold": threshold, "domains": domains}, indent=2))
