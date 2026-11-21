"""Shared utilities: config, data loading, preprocessing (Area B), metrics, size and latency helpers."""
import json
import os
import platform
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
SMOKE = os.environ.get("SMOKE") == "1"
ART = ROOT / ("artifacts_smoke" if SMOKE else "artifacts")
ART.mkdir(exist_ok=True)

TEACHER_NAME = "bert-base-uncased"
STUDENT_NAME = "distilbert-base-uncased"
MAX_LEN = 64
SEED = 42
OOS_NAME = "oos"
HUMAN = "Human agent"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data():
    """Return dict(train, validation, test) of (texts, label_names) plus the label list.

    Real run: CLINC150 'plus' config (150 intents + out-of-scope), English.
    Smoke run (SMOKE=1): tiny synthetic data, only used to test that the code runs.
    """
    if SMOKE:
        return _smoke_data()
    from datasets import load_dataset

    ds = load_dataset("clinc/clinc_oos", "plus")
    names = ds["train"].features["intent"].names
    out = {}
    for split in ["train", "validation", "test"]:
        texts = list(ds[split]["text"])
        labels = [names[i] for i in ds[split]["intent"]]
        out[split] = (texts, labels)
    return out, names


def _smoke_data():
    rng = np.random.RandomState(0)
    templates = {
        "order_status": ["where is my order", "track my package", "order status please", "has my order shipped"],
        "pay_bill": ["pay my bill", "i want to pay the bill", "settle my invoice", "make a payment"],
        "damaged_card": ["my card is damaged", "card is broken", "my card got damaged"],
        "weather": ["what is the weather", "will it rain today", "weather forecast"],
        OOS_NAME: ["who won the world cup", "tell me about dinosaurs", "how tall is everest"],
    }
    names = list(templates)

    def make(n):
        texts, labels = [], []
        for _ in range(n):
            lab = names[rng.randint(len(names))]
            t = templates[lab][rng.randint(len(templates[lab]))]
            if rng.rand() < 0.3:
                t += " please"
            texts.append(t)
            labels.append(lab)
        return texts, labels

    return {"train": make(300), "validation": make(60), "test": make(100)}, names


# ----------------------------------------------------------------------------
# Area B: regex normalization, model-code extraction, edit distance, ELIZA
# ----------------------------------------------------------------------------
MODEL_CODE_RE = re.compile(r"\b([A-Za-z]{1,4}-?\d{3,5}[A-Za-z]?)\b")

PRODUCTS = [
    "dishwasher", "refrigerator", "fridge", "freezer", "washing machine", "washer", "dryer", "oven",
    "microwave", "vacuum", "kettle", "toaster", "blender", "stove", "cooktop", "air conditioner",
    "heater", "coffee machine", "dehumidifier", "airfryer",
]
PRODUCT_TOKENS = sorted({w for p in PRODUCTS for w in p.split()})


def normalize(text: str):
    """Return (normalized_text, model_code or None).

    Steps: extract model code, lowercase, strip punctuation noise, collapse repeated letters
    (sooooo -> soo) and whitespace.
    """
    m = MODEL_CODE_RE.search(text)
    code = m.group(1).upper() if m else None
    t = MODEL_CODE_RE.sub(" ", text) if m else text
    t = t.lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t, code


def edit_distance(a: str, b: str) -> int:
    """Minimum edit distance (Levenshtein: insert, delete, substitute all cost 1)."""
    n, m = len(a), len(b)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[m]


def correct_products(text: str, known_vocab: set):
    """Fix misspelled product names with minimum edit distance (<= 2).

    Only tokens that are NOT in the known training vocabulary are candidates, so valid words
    are never rewritten. Returns (text, list of (wrong, right)).
    """
    fixes, out = [], []
    for tok in text.split():
        if tok not in known_vocab and tok not in PRODUCT_TOKENS and len(tok) >= 5:
            best = min(PRODUCT_TOKENS, key=lambda p: edit_distance(tok, p))
            if edit_distance(tok, best) <= 2:
                fixes.append((tok, best))
                tok = best
        out.append(tok)
    return " ".join(out), fixes


ELIZA_RULES = [
    (r"^(hi|hello|hey|good (morning|afternoon|evening))\b", "Hello! How can I help you with your appliance today?"),
    (r"\bhow are you\b", "I'm doing well, thank you for asking. What can I help you with?"),
    (r"\b(thanks|thank you|thx)\b", "You're welcome! Is there anything else I can help with?"),
    (r"^(bye|goodbye|see you)\b", "Goodbye! Have a great day."),
    (r"\bwho are you\b|\bare you a (bot|robot|human)\b", "I'm an automated assistant that routes your message to the right support team."),
    (r"\bi (?:feel|am) (angry|upset|frustrated|annoyed|sad|worried)\b", "I'm sorry to hear that you are {0}. Can you tell me more about the problem with your product?"),
]


def eliza_reply(norm_text: str):
    """ELIZA-style small-talk rules. Only fires for short messages so real requests are never hijacked."""
    if len(norm_text.split()) > 6:
        return None
    for pat, resp in ELIZA_RULES:
        m = re.search(pat, norm_text)
        if m:
            groups = m.groups()
            try:
                return resp.format(*groups)
            except (IndexError, KeyError):
                return resp
    return None


def tokenize_simple(text: str):
    return text.split()


# ----------------------------------------------------------------------------
# Intent -> team routing
# ----------------------------------------------------------------------------
TEAM_KEYWORDS = [
    ("Billing & Payments", ["bill", "pay", "payment", "credit", "card", "interest", "apr", "transfer", "balance", "transactions",
                            "spending", "income", "taxes", "w2", "direct_deposit", "rewards", "fees", "limit", "routing", "checks", "insurance"]),
    ("Account & Security", ["account", "pin", "fraud", "lost", "blocked", "user_name", "freeze", "report", "damaged", "new_card", "replacement"]),
    ("Orders & Delivery", ["order", "shipping", "flight", "luggage", "package", "reservation", "book", "rental", "carry_on", "travel", "visa"]),
    ("Technical Support", ["sync", "reset", "settings", "device", "maintenance", "tire", "oil", "jump", "gas", "mpg", "plug", "smart_home",
                           "speed", "volume", "timer", "alarm", "cook", "recipe", "ingredient", "calories", "nutrition", "food"]),
]
DEFAULT_TEAM = "General Support"


def intent_to_team(intent: str) -> str:
    if intent == OOS_NAME:
        return HUMAN
    for team, kws in TEAM_KEYWORDS:
        if any(k in intent for k in kws):
            return team
    return DEFAULT_TEAM


# ----------------------------------------------------------------------------
# Out-of-scope decision, metrics
# ----------------------------------------------------------------------------
def decide(probs: np.ndarray, classes, tau: float):
    """probs: (n, C). Predict label; if max prob < tau, predict OOS."""
    idx = probs.argmax(1)
    conf = probs.max(1)
    preds = [classes[i] for i in idx]
    preds = [OOS_NAME if c < tau else p for p, c in zip(preds, conf)]
    return preds, conf


def tune_tau(probs, classes, y_true):
    """Pick the confidence threshold with best validation accuracy."""
    best, best_acc = 0.0, -1
    for tau in np.linspace(0.0, 0.95, 20):
        p, _ = decide(probs, classes, tau)
        acc = float(np.mean(np.array(p) == np.array(y_true)))
        if acc > best_acc:
            best, best_acc = float(tau), acc
    return best


def metrics(preds, y_true):
    preds, y = np.array(preds), np.array(y_true)
    acc = float((preds == y).mean())
    oos = y == OOS_NAME
    oos_recall = float((preds[oos] == OOS_NAME).mean()) if oos.any() else float("nan")
    return {"accuracy": acc, "oos_recall": oos_recall}


# ----------------------------------------------------------------------------
# Size / latency helpers
# ----------------------------------------------------------------------------
def file_mb(path) -> float:
    return os.path.getsize(path) / 1e6


def hardware_string() -> str:
    import torch

    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{cpu}, {os.cpu_count()} logical cores, torch {torch.__version__} threads={torch.get_num_threads()}, CPU inference"


def time_fn(fn, texts, runs=200, warmup=10):
    """Mean latency in ms of fn(text) for single-message inference, averaged over `runs` calls."""
    for i in range(warmup):
        fn(texts[i % len(texts)])
    t0 = time.perf_counter()
    for i in range(runs):
        fn(texts[i % len(texts)])
    return (time.perf_counter() - t0) / runs * 1000.0


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
