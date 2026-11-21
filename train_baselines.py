"""Model 1: tf-idf + logistic regression (Area B).  Model 2: word2vec-average + feedforward net (Area A)."""
import pickle

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import ART, SEED, load_data, normalize, save_json, tokenize_simple

torch.manual_seed(SEED)
np.random.seed(SEED)


class FFN(nn.Module):
    def __init__(self, dim, n_classes, hidden=256, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def embed(text, w2v):
    toks = [t for t in tokenize_simple(text) if t in w2v.wv]
    if not toks:
        return np.zeros(w2v.vector_size, dtype=np.float32)
    return np.mean([w2v.wv[t] for t in toks], axis=0).astype(np.float32)


def main():
    data, names = load_data()
    classes = sorted(set(names))
    cidx = {c: i for i, c in enumerate(classes)}
    Xtr = [normalize(t)[0] for t in data["train"][0]]
    ytr = np.array([cidx[l] for l in data["train"][1]])
    Xva = [normalize(t)[0] for t in data["validation"][0]]
    yva = np.array([cidx[l] for l in data["validation"][1]])

    # ---- Model 1: tf-idf + logistic regression ----
    tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    A = tfidf.fit_transform(Xtr)
    lr = LogisticRegression(max_iter=2000, C=20.0)
    lr.fit(A, ytr)
    print("tfidf+LR val acc (argmax):", (lr.predict(tfidf.transform(Xva)) == yva).mean())
    with open(ART / "tfidf_lr.pkl", "wb") as f:
        pickle.dump({"tfidf": tfidf, "lr": lr, "classes": classes}, f)

    # ---- Model 2: word2vec (trained on the training messages) + FFN trained with SGD ----
    w2v = Word2Vec([t.split() for t in Xtr], vector_size=100, window=5, min_count=1, sg=1, epochs=30, seed=SEED, workers=1)
    w2v.save(str(ART / "w2v.model"))
    Etr = torch.tensor(np.stack([embed(t, w2v) for t in Xtr]))
    Eva = torch.tensor(np.stack([embed(t, w2v) for t in Xva]))
    ytr_t, yva_t = torch.tensor(ytr), torch.tensor(yva)

    model = FFN(100, len(classes))
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  # SGD + backpropagation
    lossf = nn.CrossEntropyLoss()
    best, best_state = 0, None
    for epoch in range(60):
        model.train()
        perm = torch.randperm(len(Etr))
        for i in range(0, len(perm), 64):
            b = perm[i : i + 64]
            opt.zero_grad()
            loss = lossf(model(Etr[b]), ytr_t[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Eva).argmax(1) == yva_t).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"ffn epoch {epoch} loss {loss.item():.3f} val acc {acc:.3f}")
    torch.save({"state": best_state, "classes": classes, "dim": 100}, ART / "ffn.pt")
    print("FFN best val acc (argmax):", best)


if __name__ == "__main__":
    main()
