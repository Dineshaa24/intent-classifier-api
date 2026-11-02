"""Train the two-stage model on CLINC150, tune the out-of-scope threshold, and report test scores.

Usage: python -m scripts.train [--data data] [--out models]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from app.model import OUT_OF_SCOPE, TwoStageClassifier, build_pipeline, train


def load(data_dir: Path):
    data = json.loads((data_dir / "data_full.json").read_text())
    domains = json.loads((data_dir / "domains.json").read_text())
    intent_to_domain = {intent: domain for domain, intents in domains.items() for intent in intents}
    split = lambda name: [(t, i) for t, i in data[name]]
    oos = lambda name: [(t, OUT_OF_SCOPE) for t, _ in data[name]]
    return split("train"), split("val") + oos("oos_val"), split("test") + oos("oos_test"), intent_to_domain


def evaluate(clf: TwoStageClassifier, rows):
    preds = [clf.predict(t).intent for t, _ in rows]
    gold = [i for _, i in rows]
    in_scope = [k for k, g in enumerate(gold) if g != OUT_OF_SCOPE]
    oos = [k for k, g in enumerate(gold) if g == OUT_OF_SCOPE]
    return {
        "in_scope_accuracy": accuracy_score([gold[k] for k in in_scope], [preds[k] for k in in_scope]),
        "in_scope_macro_f1": f1_score([gold[k] for k in in_scope], [preds[k] for k in in_scope], average="macro"),
        "oos_recall": float(np.mean([preds[k] == OUT_OF_SCOPE for k in oos])) if oos else None,
        "overall_accuracy": accuracy_score(gold, preds),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    train_rows, val_rows, test_rows, intent_to_domain = load(Path(args.data))
    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} intents={len(intent_to_domain)}")

    start = time.time()
    train(train_rows, intent_to_domain, args.out, threshold=0.0)
    print(f"trained in {time.time() - start:.1f}s")

    # Pick the out-of-scope threshold that gives the best overall accuracy on validation.
    clf = TwoStageClassifier(args.out)
    raw = [clf.predict(t) for t, _ in val_rows]  # threshold is 0 here, so these are raw predictions
    gold = [i for _, i in val_rows]
    best = max(np.arange(0.0, 0.61, 0.02), key=lambda th: _accuracy_at(raw, gold, th))
    clf.threshold = float(best)
    meta_path = Path(args.out) / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["threshold"] = round(float(best), 2)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"threshold={best:.2f}")

    # Flat single-model baseline for comparison.
    flat = build_pipeline().fit([t for t, _ in train_rows], [i for _, i in train_rows])
    gold_in = [(t, i) for t, i in test_rows if i != OUT_OF_SCOPE]
    flat_acc = accuracy_score([i for _, i in gold_in], flat.predict([t for t, _ in gold_in]))

    results = evaluate(clf, test_rows)
    results["flat_baseline_in_scope_accuracy"] = flat_acc
    results = {k: round(v, 4) if isinstance(v, float) else v for k, v in results.items()}
    (Path(args.out) / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def _accuracy_at(preds, gold, threshold: float) -> float:
    labels = [p.intent if p.confidence >= threshold else OUT_OF_SCOPE for p in preds]
    return float(np.mean([p == g for p, g in zip(labels, gold)]))


if __name__ == "__main__":
    main()
