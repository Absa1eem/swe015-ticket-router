"""Gradio app: Customer Support Ticket Router.
Flow: message -> regex normalize + model-code extraction -> edit-distance product correction ->
ELIZA small-talk rule (reply, no model) -> four models in parallel -> confidence threshold ->
intent -> team (or Human agent). The headline result comes from the quantized DistilBERT (INT8).
Run:  python app.py   (needs artifacts/ from run_all.py, or HF_REPO env var to download them)"""
import os
import time

import gradio as gr

from common import ART, HUMAN, OOS_NAME, correct_products, decide, eliza_reply, intent_to_team, load_data, normalize
from inference import MODEL_KEYS, MODEL_LABELS, load_models, load_taus


DEFAULT_HF_REPO = "Absa1eem/ticket-router-weights"  # trained weights are hosted on the Hugging Face Hub


def ensure_artifacts():
    if (ART / "student_int8.pt").exists():
        return
    repo = os.environ.get("HF_REPO", DEFAULT_HF_REPO)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, local_dir=str(ART))


ensure_artifacts()
CLASSES, PREDICT = load_models()
TAUS = load_taus()
_data, _ = load_data()
VOCAB = {w for t in _data["train"][0] for w in normalize(t)[0].split()}  # training vocabulary for spell-correction


def route(message: str):
    message = (message or "").strip()
    if not message:
        return "Type a message first.", "", []
    norm, code = normalize(message)
    norm, fixes = correct_products(norm, VOCAB)
    info = f"**Normalized:** `{norm}`"
    if code:
        info += f"  \n**Model code detected:** `{code}`"
    if fixes:
        info += "  \n**Spelling fixes:** " + ", ".join(f"{a} → {b}" for a, b in fixes)

    reply = eliza_reply(norm)
    if reply:
        return f"💬 **Small talk (rule-based):** {reply}", info, []

    rows, headline = [], ""
    for k in MODEL_KEYS:
        t0 = time.perf_counter()
        probs = PREDICT[k]([norm])
        ms = (time.perf_counter() - t0) * 1000
        pred, conf = decide(probs, CLASSES, TAUS[k])
        team = HUMAN if pred[0] == OOS_NAME else intent_to_team(pred[0])
        rows.append([MODEL_LABELS[k], pred[0], f"{conf[0]:.2f}", team, f"{ms:.1f}"])
        if k == "distil_int8":
            headline = f"### ➜ Routed to: **{team}**  \nIntent: `{pred[0]}` · confidence {conf[0]:.2f} (quantized DistilBERT, INT8)"
            if team == HUMAN:
                headline += "  \n_Low confidence or out of scope, sent to a human agent._"
    return headline, info, rows


with gr.Blocks(title="Support Ticket Router") as demo:
    gr.Markdown("# Support Ticket Router\nType a customer message. The quantized DistilBERT decides the route; all four models are shown side by side.")
    box = gr.Textbox(label="Customer message", placeholder="e.g. my dishwaser DW-6534 is not draining water", lines=2)
    btn = gr.Button("Route", variant="primary")
    out = gr.Markdown()
    info = gr.Markdown()
    table = gr.Dataframe(headers=["Model", "Intent", "Confidence", "Team", "Latency (ms)"], interactive=False)
    gr.Examples(["hello", "where is my order", "I want to pay my bill", "my card is damaged", "what is the capital of mongolia",
                 "pay my bil please", "thanks bye"], inputs=box)
    btn.click(route, box, [out, info, table])
    box.submit(route, box, [out, info, table])

if __name__ == "__main__":
    demo.launch(share=os.environ.get("SHARE") == "1")
