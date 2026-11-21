"""Tokenizer/model factory. In SMOKE mode builds tiny random models (offline) to test the code path only."""
from common import ART, MAX_LEN, SMOKE, STUDENT_NAME, TEACHER_NAME, load_data, normalize


def _smoke_tokenizer():
    from transformers import BertTokenizerFast

    d = ART / "smoke_tok"
    d.mkdir(exist_ok=True)
    data, _ = load_data()
    words = sorted({w for t in data["train"][0] + data["test"][0] for w in normalize(t)[0].split()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    (d / "vocab.txt").write_text("\n".join(vocab))
    return BertTokenizerFast(str(d / "vocab.txt"), do_lower_case=True)


def get_tokenizer(name=None):
    if SMOKE:
        return _smoke_tokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name or TEACHER_NAME)


def new_model(kind, n_labels, vocab_size=None):
    """kind: 'teacher' or 'student'."""
    from transformers import AutoModelForSequenceClassification, BertConfig, BertForSequenceClassification

    if SMOKE:
        layers = 2 if kind == "teacher" else 1
        cfg = BertConfig(vocab_size=vocab_size, hidden_size=32, num_hidden_layers=layers, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=MAX_LEN + 2, num_labels=n_labels)
        return BertForSequenceClassification(cfg)
    name = TEACHER_NAME if kind == "teacher" else STUDENT_NAME
    return AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_labels)
