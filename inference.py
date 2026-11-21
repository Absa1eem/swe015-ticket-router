"""Single source of truth for loading the four models and predicting. Used by evaluate.py AND app.py,
so the architecture diagram matches the code. Each predictor maps a list of normalized texts -> probs (n, C)."""
import pickle

import numpy as np
import torch

from common import ART, MAX_LEN, SMOKE, load_json
from train_baselines import FFN, embed
from transformers_util import get_tokenizer, new_model

MODEL_KEYS = ["tfidf_lr", "w2v_ffn", "bert_fp32", "distil_int8"]  # the four ladder models shown in the app
EVAL_KEYS = MODEL_KEYS[:3] + ["distil_fp32", "distil_int8"]  # distil_fp32 = quantization "before"
MODEL_LABELS = {
    "tfidf_lr": "TF-IDF + Logistic Regression (FP32)",
    "w2v_ffn": "word2vec avg + FFN (FP32)",
    "bert_fp32": "BERT-base fine-tuned (FP32)",
    "distil_fp32": "DistilBERT distilled (FP32, before quantization)",
    "distil_int8": "DistilBERT distilled + dynamic INT8 (used by the app)",
}


def _hf_predict(model, tok, texts):
    enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt", return_token_type_ids=False)
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, -1).numpy()


def load_models():
    """Returns (classes, {key: predict_fn(list[str]) -> np.ndarray})."""
    with open(ART / "tfidf_lr.pkl", "rb") as f:
        d = pickle.load(f)
    classes = d["classes"]
    tfidf, lr = d["tfidf"], d["lr"]

    from gensim.models import Word2Vec

    w2v = Word2Vec.load(str(ART / "w2v.model"))
    ck = torch.load(ART / "ffn.pt")
    ffn = FFN(ck["dim"], len(classes))
    ffn.load_state_dict(ck["state"])
    ffn.eval()

    def p_tfidf(texts):
        return lr.predict_proba(tfidf.transform(texts))

    def p_ffn(texts):
        x = torch.tensor(np.stack([embed(t, w2v) for t in texts]))
        with torch.no_grad():
            return torch.softmax(ffn(x), -1).numpy()

    tok = get_tokenizer(str(ART / "teacher") if not SMOKE else None)
    from transformers import AutoModelForSequenceClassification

    bert = AutoModelForSequenceClassification.from_pretrained(ART / "teacher").eval()

    # Quantized student: rebuild the architecture, apply dynamic quantization, load the saved INT8 weights.
    student_fp = AutoModelForSequenceClassification.from_pretrained(ART / "student_fp32").eval()
    q = torch.quantization.quantize_dynamic(student_fp, {torch.nn.Linear}, dtype=torch.qint8)
    q.load_state_dict(torch.load(ART / "student_int8.pt"))
    q.eval()

    preds = {
        "tfidf_lr": p_tfidf,
        "w2v_ffn": p_ffn,
        "bert_fp32": lambda t: _hf_predict(bert, tok, t),
        "distil_fp32": lambda t: _hf_predict(student_fp, tok, t),
        "distil_int8": lambda t: _hf_predict(q, tok, t),
    }
    return classes, preds


def load_taus():
    p = ART / "taus.json"
    return load_json(p) if p.exists() else {k: 0.5 for k in EVAL_KEYS}
