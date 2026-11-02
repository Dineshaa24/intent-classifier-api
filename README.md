# intent-classifier-api

A two-stage intent classifier served with FastAPI. It first decides which **domain** a sentence belongs to (banking, travel, utility, ...) and then picks the **intent** inside that domain. Anything it isn't confident about is returned as `out_of_scope` instead of a wrong guess.

I built this as a clean, public version of the approach I use for voice bot intent detection at work. It is trained on the public [CLINC150](https://github.com/clinc/oos-eval) dataset, so none of the data or code comes from any employer.

## Why two stages

With 150 intents in one model, intents that sound alike in different domains get mixed up, and every small change means retraining everything. Splitting it up gives:

- a small domain model (10 classes) that runs on every request
- one small intent model per domain, **loaded only when that domain is first needed**, so memory grows with real traffic instead of with the label count
- the option to retrain or replace one domain without touching the others

Out-of-scope detection uses the combined confidence (`domain_confidence × intent_confidence`) with a threshold tuned on the validation set.

## Results

CLINC150 test set (4,500 in-scope + 1,000 out-of-scope sentences), TF-IDF word + character n-grams with logistic regression, CPU only:

| Metric | Score |
|---|---|
| In-scope accuracy | 92.6% |
| In-scope macro F1 | 92.7% |
| Out-of-scope recall | 49.2% |
| Overall accuracy (incl. out-of-scope) | 84.8% |
| Flat 150-class baseline, in-scope accuracy | 92.5% |
| Prediction latency (median / p95, warm) | 6.1 ms / 8.8 ms |

Honest reading of this: on CLINC150 the two-stage setup matches the flat model's accuracy, so the gain here is in serving (lazy loading, independent retraining), not in raw accuracy. Out-of-scope recall is the weak spot. A threshold on TF-IDF probabilities only catches about half of the unrelated queries, which is the next thing to improve (see below).

## Run it

```bash
pip install -r requirements.txt
python -m scripts.download_data      # fetches CLINC150 into data/
python -m scripts.train              # trains, tunes the threshold, prints test scores
uvicorn app.main:app --reload
```

Or with Docker (the image downloads the data and trains during the build):

```bash
docker build -t intent-classifier-api .
docker run -p 8000:8000 intent-classifier-api
```

Interactive docs are at `http://localhost:8000/docs`.

## API

`POST /predict`

```json
{"text": "i lost my credit card"}
```

```json
{
  "domain": "credit_cards",
  "intent": "report_lost_card",
  "confidence": 0.9839,
  "domain_confidence": 1.0,
  "latency_ms": 6.4
}
```

A sentence outside every domain, such as "who won the cricket match yesterday", comes back as `out_of_scope` with low confidence.

`POST /predict/batch` takes `{"texts": [...]}` (up to 100) and returns a list of the same objects.

`GET /health` shows how many domains exist and which domain models are currently loaded.

## Project layout

```
app/
  preprocess.py   text cleaning that keeps negations ("not", "no") and numbers
  model.py        two-stage classifier, lazy loading, training
  main.py         FastAPI app
scripts/
  download_data.py
  train.py        training, threshold tuning, evaluation, flat baseline
tests/            pytest suite on a small built-in dataset (no download needed)
```

## Tests

```bash
pytest
```

## What I'd do next

- Add a Sentence-BERT embedding model as a second-opinion scorer for out-of-scope detection, since embeddings separate unrelated text far better than TF-IDF probabilities.
- Calibrate probabilities per domain instead of one global threshold.
- Export the models to ONNX for faster cold starts.

## License

MIT
