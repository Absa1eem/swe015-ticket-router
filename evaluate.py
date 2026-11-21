"""Evaluate all models on the SAME held-out CLINC150 test set. Writes artifacts/results.json and results.md.
Latency = mean over 200 single-message calls (normalize excluded, tokenization + forward included), CPU."""
import os

import numpy as np
import torch

from common import ART, OOS_NAME, decide, file_mb, hardware_string, load_data, metrics, normalize, save_json, time_fn, tune_tau
from inference import EVAL_KEYS, MODEL_LABELS, load_models

RUNS = 20 if os.environ.get("SMOKE") == "1" else 200


def dir_mb(path):
    return sum(os.path.getsize(p) for p in path.glob("**/*") if p.is_file()) / 1e6


def sizes():
    teacher = max((ART / "teacher").glob("*.safetensors"), key=os.path.getsize) if list((ART / "teacher").glob("*.safetensors")) else next((ART / "teacher").glob("pytorch_model.bin"))
    stu = max((ART / "student_fp32").glob("*.safetensors"), key=os.path.getsize) if list((ART / "student_fp32").glob("*.safetensors")) else next((ART / "student_fp32").glob("pytorch_model.bin"))
    w2v_files = sum(os.path.getsize(p) for p in ART.glob("w2v.model*")) / 1e6
    return {
        "tfidf_lr": file_mb(ART / "tfidf_lr.pkl"),
        "w2v_ffn": w2v_files + file_mb(ART / "ffn.pt"),
        "bert_fp32": file_mb(teacher),
        "distil_fp32": file_mb(stu),
        "distil_int8": file_mb(ART / "student_int8.pt"),
    }


def main():
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    data, _ = load_data()
    classes, preds = load_models()
    norm = {s: [normalize(t)[0] for t in data[s][0]] for s in ["validation", "test"]}
    y_val, y_test = data["validation"][1], data["test"][1]
    sz = sizes()
    results, taus, test_preds = {}, {}, {}
    for k in EVAL_KEYS:
        pv = preds[k](norm["validation"])
        tau = tune_tau(pv, classes, y_val)  # threshold tuned on VALIDATION only
        pt = preds[k](norm["test"])
        p, conf = decide(pt, classes, tau)
        m = metrics(p, y_test)
        lat = time_fn(lambda t, k=k: preds[k]([t]), norm["test"], runs=RUNS)
        taus[k] = tau
        test_preds[k] = (p, conf)
        results[k] = {"label": MODEL_LABELS[k], **m, "size_mb": sz[k], "latency_ms": lat, "tau": tau}
        print(k, results[k], flush=True)
    save_json(taus, ART / "taus.json")

    # failure examples of the model used in the application
    p, conf = test_preds["distil_int8"]
    fails = [{"text": data["test"][0][i], "true": y_test[i], "pred": p[i], "confidence": float(conf[i])}
             for i in range(len(y_test)) if p[i] != y_test[i]]
    out = {"hardware": hardware_string(), "n_test": len(y_test), "n_test_oos": int(sum(y == OOS_NAME for y in y_test)),
           "n_train": len(data["train"][0]), "n_val": len(data["validation"][0]), "n_intents": len(classes) - 1,
           "latency_runs": RUNS, "models": results, "failures": fails[:25], "n_failures": len(fails)}
    save_json(out, ART / "results.json")

    lines = ["| Model | Accuracy | OOS recall | Size (MB) | Latency (ms) |", "|---|---|---|---|---|"]
    for k in EVAL_KEYS:
        r = results[k]
        lines.append(f"| {r['label']} | {r['accuracy']*100:.1f}% | {r['oos_recall']*100:.1f}% | {r['size_mb']:.1f} | {r['latency_ms']:.1f} |")
    lines.append(f"\nHardware: {out['hardware']}. Latency: mean of {RUNS} single-message calls.")
    (ART / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
