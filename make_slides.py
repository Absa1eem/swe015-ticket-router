"""Builds the 12-slide presentation from artifacts/results.json (every number comes from evaluate.py).
Usage: python make_slides.py --id 2200XXXX --repo https://github.com/... --video https://youtu.be/...
Output: submission/presentation.pptx (and .pdf if LibreOffice is installed)."""
import argparse
import subprocess
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

from common import ART, ROOT, load_json
from inference import EVAL_KEYS

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="Abdullah Tahir Saleem")
ap.add_argument("--id", default="<STUDENT ID>")
ap.add_argument("--repo", default="<REPO LINK>")
ap.add_argument("--video", default="<VIDEO LINK>")
args = ap.parse_args()

R = load_json(ART / "results.json")
M = R["models"]
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)
BLANK = prs.slide_layouts[6]
NAVY = RGBColor(0x1F, 0x2A, 0x44)


def slide(title, bullets=None, size=20):
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12), Inches(0.9))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size, p.font.bold, p.font.color.rgb = Pt(32), True, NAVY
    if bullets:
        tb = s.shapes.add_textbox(Inches(0.7), Inches(1.4), Inches(12), Inches(5.6))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, b in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = "• " + b
            p.font.size = Pt(size)
            p.space_after = Pt(10)
    return s


def table(s, rows, left, top, width, height, fs=14):
    shape = s.shapes.add_table(len(rows), len(rows[0]), Inches(left), Inches(top), Inches(width), Inches(height))
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = shape.table.cell(i, j)
            c.text = str(v)
            for p in c.text_frame.paragraphs:
                p.font.size = Pt(fs)
                p.font.bold = i == 0


def pct(x):
    return f"{x * 100:.1f}%"


# 1 Title
s = prs.slides.add_slide(BLANK)
tb = s.shapes.add_textbox(Inches(0.8), Inches(1.0), Inches(11.8), Inches(5.5))
tf = tb.text_frame
tf.word_wrap = True
lines = [("SWE015 Homework: Building an Intelligent Application", 34, True),
         ("Use Case 2: Customer Support Ticket Router", 26, True), ("", 12, False),
         (f"{args.name}   |   Student ID: {args.id}", 20, False),
         (f"Repository: {args.repo}", 18, False), (f"Demo video: {args.video}", 18, False)]
for i, (t, sz, b) in enumerate(lines):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text, p.font.size, p.font.bold = t, Pt(sz), b

# 2 Introduction
slide("Introduction: the problem", [
    "A home appliance brand receives thousands of support messages a day; each must reach the right team instantly.",
    "Who it is for: support teams who need fast routing and fewer misrouted tickets.",
    "Task: free-text message -> normalize -> intent + confidence -> team; low-confidence or out-of-scope -> human agent; small talk -> rule-based reply.",
    "Success = high intent accuracy, high out-of-scope recall, and a small, fast model that runs on a CPU.",
    "Approach: a four-model ladder (classical, neural net, Transformer, quantized Transformer) compared on one test set."])

# 3 Data
slide("Development: data", [
    "Dataset: CLINC150 ('plus' configuration, English): 150 in-scope intents + an out-of-scope (oos) class (Larson et al., 2019).",
    f"Splits (official): train {R['n_train']}, validation {R['n_val']}, test {R['n_test']} (of which {R['n_test_oos']} out-of-scope).",
    "Preprocessing: model-code extraction (regex), lowercasing, URL and punctuation removal, repeated-letter collapsing, edit-distance product spell fixing.",
    "Intents are mapped to support teams by a keyword rule table (common.py).",
    "Decision rule: predict oos when max probability < tau; tau tuned per model on validation only."])

# 4 Architecture
s = slide("Development: architecture diagram")
s.shapes.add_picture(str(ART / "architecture.png"), Inches(0.4), Inches(1.3), width=Inches(12.5))

# 5 Technique map
s = slide("Development: technique map")
table(s, [["Area", "Technique", "Where it is used"],
          ["A  Neural networks", "Feedforward net, SGD + backprop, dropout", "Model 2: averaged word2vec -> FFN"],
          ["B  Classical NLP", "Regex, edit distance, ELIZA rules, word2vec, tf-idf", "Normalization, product spell-fix, small talk, Model 1, Model 2 inputs"],
          ["C  Transformers", "BERT fine-tuning with a classification head; WordPiece", "Model 3: BERT-base; Model 4 student: DistilBERT"],
          ["D  Efficient models", "Knowledge distillation, dynamic INT8 quantization, benchmarking", "Model 4: distilled student, quantized; size/latency table"]],
         0.6, 1.5, 12.1, 4.0, fs=15)

# 6 Models
slide("Development: the four models and how each was trained", [
    "Model 1 baseline: TF-IDF (1-2 grams) + logistic regression (scikit-learn).",
    "Model 2 neural net: word2vec (skip-gram, trained on the training messages) averaged -> 256-unit FFN, dropout 0.3, SGD with momentum.",
    "Model 3 Transformer: bert-base-uncased fine-tuned for intent classification (AdamW, 4 epochs).",
    "Model 4 quantized Transformer: DistilBERT student trained by distillation from Model 3 (soft targets, T=2, alpha=0.5), then dynamic INT8 quantization.",
    "All models use the same normalized text, the same test set and the same oos rule (per-model tau)."], size=19)

# 7 Quantization
slide("Development: quantization method", [
    "Post-training dynamic quantization (torch.quantization.quantize_dynamic): all nn.Linear weights stored as INT8, activations quantized on the fly.",
    "Runs on CPU, no calibration data needed; embeddings and layer norms stay FP32.",
    "The application loads the INT8 weights (student_int8.pt) and runs them for every request.",
    f"Quantization effect measured on the test set: DistilBERT FP32 {pct(M['distil_fp32']['accuracy'])} -> INT8 {pct(M['distil_int8']['accuracy'])} accuracy; "
    f"{M['distil_fp32']['size_mb']:.1f} MB -> {M['distil_int8']['size_mb']:.1f} MB; {M['distil_fp32']['latency_ms']:.1f} ms -> {M['distil_int8']['latency_ms']:.1f} ms."], size=19)

# 8 Results table
s = slide("Development: results table (same held-out test set)")
rows = [["Model", "Accuracy", "OOS recall", "Size (MB)", "Latency (ms)"]]
for k in EVAL_KEYS:
    r = M[k]
    rows.append([r["label"], pct(r["accuracy"]), pct(r["oos_recall"]), f"{r['size_mb']:.1f}", f"{r['latency_ms']:.1f}"])
table(s, rows, 0.5, 1.5, 12.3, 3.6, fs=14)
tb = s.shapes.add_textbox(Inches(0.5), Inches(5.6), Inches(12.3), Inches(1.4))
tb.text_frame.word_wrap = True
tb.text_frame.text = f"Hardware: {R['hardware']}. Latency = mean of {R['latency_runs']} single-message calls (tokenization + forward pass). Size = saved weights file(s)."
tb.text_frame.paragraphs[0].font.size = Pt(14)

# 9 Conclusion: comparison
best_acc = max(EVAL_KEYS, key=lambda k: M[k]["accuracy"])
fastest = min(EVAL_KEYS, key=lambda k: M[k]["latency_ms"])
smallest = min(EVAL_KEYS, key=lambda k: M[k]["size_mb"])
best_oos = max(EVAL_KEYS, key=lambda k: M[k]["oos_recall"])
q, t = M["distil_int8"], M["bert_fp32"]
slide("Conclusion: what the comparison showed", [
    f"Highest accuracy: {M[best_acc]['label']} ({pct(M[best_acc]['accuracy'])}). Best out-of-scope recall: {M[best_oos]['label']} ({pct(M[best_oos]['oos_recall'])}).",
    f"Fastest: {M[fastest]['label']} ({M[fastest]['latency_ms']:.1f} ms). Smallest: {M[smallest]['label']} ({M[smallest]['size_mb']:.2f} MB).",
    f"Quantized DistilBERT vs BERT-base: accuracy {pct(q['accuracy'])} vs {pct(t['accuracy'])}, size {q['size_mb']:.0f} MB vs {t['size_mb']:.0f} MB, latency {q['latency_ms']:.1f} ms vs {t['latency_ms']:.1f} ms.",
    "<EDIT: 1-2 sentences of your own interpretation: where each model wins and loses, and why>"], size=19)

# 10 Failure example + limitations
f = R["failures"][0] if R["failures"] else None
bul = [f"Quantized model errors on the test set: {R['n_failures']} of {R['n_test']}."]
if f:
    bul.append(f"Failure example: \"{f['text']}\" -> predicted '{f['pred']}' (confidence {f['confidence']:.2f}), true label '{f['true']}'.")
bul += ["Limitations: CLINC150 is general-domain, not appliance-specific; English only; the intent-to-team table is hand-written.",
        "Word2vec is trained on ~15k short messages only, which limits Model 2.",
        "Latency was measured on one CPU configuration; other hardware will differ."]
slide("Conclusion: limitations and a failure example", bul, size=19)

# 11 Next steps
slide("Conclusion: what I would do next", [
    "Add Turkish (MASSIVE) with a multilingual encoder such as XLM-R, as in the original scenario.",
    "Collect real appliance tickets and fine-tune on them; add the 50-100 domain examples the brief suggests.",
    "Try static INT8 quantization or ONNX Runtime and compare with dynamic quantization.",
    "Calibrate the confidence scores (temperature scaling) to improve out-of-scope detection."])

# 12 AI usage
slide("AI usage statement", [
    "Tool used: Claude (Anthropic).",
    "Used for: writing and debugging the code (preprocessing, training scripts, evaluation, Gradio app), the architecture diagram script, and a first draft of these slides.",
    "My own work: running all training and evaluation, measuring every number shown here, testing the app, recording the demo, and reviewing every file.",
    "<EDIT: adjust this to be exactly true for you>",
    "Data: CLINC150 (Larson et al., 2019, EMNLP). Models: bert-base-uncased, distilbert-base-uncased (Hugging Face)."], size=19)

out = ROOT / "submission"
out.mkdir(exist_ok=True)
prs.save(out / "presentation.pptx")
try:
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(out), str(out / "presentation.pptx")], check=True, capture_output=True)
    print("wrote", out / "presentation.pdf")
except Exception as e:
    print("pptx written; convert to PDF manually (File > Export as PDF):", e)
