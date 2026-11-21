"""Model 3: BERT fine-tuned for intent classification (Area C, teacher).
Model 4: DistilBERT student trained by knowledge distillation from the teacher, then dynamic INT8
quantization (Area D). The quantized student is the model used by the application."""
import numpy as np
import torch
import torch.nn.functional as F

from common import ART, MAX_LEN, SEED, SMOKE, load_data, normalize
from transformers_util import get_tokenizer, new_model

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS_TEACHER = 1 if SMOKE else 4
EPOCHS_STUDENT = 1 if SMOKE else 6
BS = 32


def encode(tok, texts):
    return tok(texts, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt", return_token_type_ids=False)


def batches(enc, y, bs, shuffle):
    n = enc["input_ids"].shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        b = idx[i : i + bs]
        yield {k: v[b].to(DEVICE) for k, v in enc.items()}, y[b].to(DEVICE), b


@torch.no_grad()
def logits_of(model, enc, bs=128):
    model.eval()
    out = []
    for xb, _, _ in batches(enc, torch.zeros(enc["input_ids"].shape[0], dtype=torch.long), bs, False):
        out.append(model(**xb).logits.float().cpu())
    return torch.cat(out)


def fit(model, enc, y, epochs, lr, teacher_logits=None, T=2.0, alpha=0.5, val=None):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(y) + BS - 1) // BS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    for ep in range(epochs):
        model.train()
        tot = 0
        for xb, yb, b in batches(enc, y, BS, True):
            logits = model(**xb).logits
            loss = F.cross_entropy(logits, yb)
            if teacher_logits is not None:  # knowledge distillation: soft targets from the teacher
                tl = teacher_logits[b].to(DEVICE)
                kd = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(tl / T, -1), reduction="batchmean") * T * T
                loss = alpha * loss + (1 - alpha) * kd
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        msg = f"epoch {ep + 1}/{epochs} loss {tot:.2f}"
        if val is not None:
            pred = logits_of(model, val[0]).argmax(1)
            msg += f" val acc {(pred == val[1]).float().mean().item():.4f}"
        print(msg, flush=True)
    return model.cpu()


def main():
    data, names = load_data()
    classes = sorted(set(names))
    cidx = {c: i for i, c in enumerate(classes)}
    tok = get_tokenizer()
    enc, ys = {}, {}
    for s in ["train", "validation"]:
        enc[s] = encode(tok, [normalize(t)[0] for t in data[s][0]])
        ys[s] = torch.tensor([cidx[l] for l in data[s][1]])
    vocab = len(tok)

    # ---- Model 3: teacher ----
    teacher = new_model("teacher", len(classes), vocab)
    teacher = fit(teacher, enc["train"], ys["train"], EPOCHS_TEACHER, 5e-5, val=(enc["validation"], ys["validation"]))
    teacher.save_pretrained(ART / "teacher")
    tok.save_pretrained(ART / "teacher")
    t_logits = logits_of(teacher.to(DEVICE), enc["train"])
    teacher.cpu()

    # ---- Model 4a: distilled student ----
    student = new_model("student", len(classes), vocab)
    student = fit(student, enc["train"], ys["train"], EPOCHS_STUDENT, 8e-5, teacher_logits=t_logits,
                  val=(enc["validation"], ys["validation"]))
    student.save_pretrained(ART / "student_fp32")

    # ---- Model 4b: dynamic INT8 quantization of all nn.Linear layers (CPU) ----
    student.eval()
    q = torch.quantization.quantize_dynamic(student, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(q.state_dict(), ART / "student_int8.pt")
    print("saved", ART / "student_int8.pt")


if __name__ == "__main__":
    main()
