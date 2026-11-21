"""Architecture diagram (R6). Mirrors app.py / inference.py. Output: artifacts/architecture.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from common import ART


def box(ax, x, y, w, h, text, fc="#e8f0fe", ec="#1a56db", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))


def main():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.text(6.5, 5.95, "Support Ticket Router: architecture (everything runs on CPU, in one Python process: Gradio app)",
            ha="center", fontsize=11, fontweight="bold")
    box(ax, 0.2, 2.5, 1.5, 1.2, "User message\n(Gradio UI,\ntext box)", "#fff4d6", "#b7791f", bold=True)
    box(ax, 2.1, 3.3, 2.2, 1.0, "Regex normalization\n+ model-code extraction\n(Area B)", "#e6f6ea", "#2f855a")
    box(ax, 2.1, 1.9, 2.2, 1.0, "Edit-distance product\nspell correction (Area B)", "#e6f6ea", "#2f855a")
    box(ax, 2.1, 0.4, 2.2, 1.0, "ELIZA small-talk rules\n(Area B) -> rule reply", "#e6f6ea", "#2f855a")
    ys = [4.5, 3.4, 2.3, 1.2]
    labels = ["TF-IDF + LogReg\nscikit-learn, FP32 (Area B)", "word2vec avg + FFN\nPyTorch, FP32, SGD (Area A)",
              "BERT-base fine-tuned\nPyTorch, FP32 (Area C)", "DistilBERT distilled\ndynamic INT8 (Areas C+D)\nUSED BY THE APP"]
    for y, t in zip(ys, labels):
        last = "INT8" in t
        box(ax, 5.2, y - 0.4, 2.6, 0.95 if not last else 1.05, t, "#fde8e8" if last else "#e8f0fe", "#c53030" if last else "#1a56db", fs=8.5, bold=last)
        arrow(ax, 4.4, 2.4, 5.2, y + 0.1)
    arrow(ax, 1.7, 3.1, 2.1, 3.8); arrow(ax, 3.2, 3.3, 3.2, 2.9); arrow(ax, 3.2, 1.9, 3.2, 1.4)
    box(ax, 8.4, 2.7, 1.9, 1.3, "Confidence\nthreshold tau\n(tuned on val set)\nlow conf -> OOS", "#f3e8ff", "#6b46c1")
    for y in ys:
        arrow(ax, 7.8, y + 0.1, 8.4, 3.35)
    box(ax, 10.7, 2.7, 2.1, 1.3, "Intent -> team map\nor \"Human agent\"\n(post-processing)", "#f3e8ff", "#6b46c1")
    arrow(ax, 10.3, 3.35, 10.7, 3.35)
    box(ax, 10.7, 0.9, 2.1, 1.3, "UI output:\nrouted team, confidence,\nside-by-side models,\nlatency", "#fff4d6", "#b7791f")
    arrow(ax, 11.75, 2.7, 11.75, 2.2)
    ax.text(0.2, 0.15, "Models are loaded from artifacts/ (weights hosted outside the repo). Small talk bypasses the models.", fontsize=8, style="italic")
    fig.savefig(ART / "architecture.png", dpi=170, bbox_inches="tight")


if __name__ == "__main__":
    main()
