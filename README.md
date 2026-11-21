# Support Ticket Router (SWE015 Homework, Use Case 2)

Routes a customer support message to the right team using a four-model ladder, compared on the same
held-out test set, with a quantized Transformer inside a Gradio app.

| # | Model | Area |
|---|---|---|
| 1 | TF-IDF + logistic regression | B (classical NLP) |
| 2 | word2vec average + feedforward net (SGD, dropout) | A + B |
| 3 | BERT-base fine-tuned | C |
| 4 | DistilBERT distilled from #3, then dynamic INT8 quantization (**used by the app**) | C + D |

Preprocessing (Area B): regex normalization, model-code extraction, edit-distance product spell fix, ELIZA small-talk rules.
Data: CLINC150 "plus" (English, 150 intents + out-of-scope), loaded automatically from the Hugging Face Hub.

## Quick start (run the app from this repo)
The trained weights are hosted on the Hugging Face Hub (https://huggingface.co/Absa1eem/ticket-router-weights) and are
downloaded automatically the first time you start the app. Python 3.10+ is required.

```bash
git clone https://github.com/Absa1eem/swe015-ticket-router.git && cd swe015-ticket-router
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py          # first start downloads the weights (~1 GB), then open http://127.0.0.1:7860
```

## Retrain from scratch (optional, needs a GPU; Google Colab T4 works, see `colab.ipynb`)
```bash
python run_all.py      # trains all models, evaluates on the test set, writes artifacts/results.json + results.md
python upload_weights.py <user>/<repo>   # optional: publish your own weights
```

## Files
- `common.py` data, preprocessing, metrics, size/latency helpers
- `train_baselines.py` models 1 and 2; `train_transformers.py` models 3 and 4 (distillation + quantization)
- `inference.py` loads the four models (used by both `evaluate.py` and `app.py`)
- `evaluate.py` results table (accuracy, out-of-scope recall, size, latency); `app.py` Gradio app
- `make_diagram.py`, `make_slides.py` architecture diagram and slide deck generated from `results.json`
- `SMOKE=1 python run_all.py` runs the whole pipeline offline on tiny synthetic data (tests the code only, numbers meaningless)

## Method notes
- Out-of-scope: a message is sent to the human agent when max probability < tau, or the model predicts `oos`. tau is tuned per model on the validation split only.
- Latency: mean of 200 single-message CPU calls (tokenization + forward). Size: saved weights file(s).
- Quantization: `torch.quantization.quantize_dynamic` on all `nn.Linear` layers, INT8.

Data: Larson et al., 2019, "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction" (CLINC150).
