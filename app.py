"""Research demo. Not a diagnostic tool."""
import os
from pathlib import Path
import gradio as gr
from inference import MAX_BYTES, REFER_BAND, load_gate, load_model, predict_bytes

HERE = Path(__file__).resolve().parent
MODEL, GATE = load_model(), load_gate()

BANNER = """
## Malaria cell classifier — research demo
**Not a diagnostic tool.** Trained on one public dataset of single RBC crops.
Uploads: PNG/JPEG, max 5 MB. Nothing is stored.
"""

def classify(upload):
    path = upload if isinstance(upload, str) else getattr(upload, "name", None)
    if not path:
        return None, "Upload a PNG or JPEG of a single cell."
    try:
        if os.path.getsize(path) > MAX_BYTES:
            return None, "**Rejected:** file too large (5 MB limit)."
        raw = open(path, "rb").read(MAX_BYTES + 1)
        res = predict_bytes(raw, MODEL, GATE)
    except Exception:
        return None, "**Error:** could not analyse this file."
    if res["status"] == "rejected":
        return None, f"**Rejected:** {res['reason']}."
    if res["status"] == "withheld":
        return None, ("**No prediction.** Image does not look like a training cell crop. "
                      + res["reason"])
    p = res["p_parasitized"]
    msg = [f"**{res['label']}** (P = {p:.3f})"]
    if res["refer_to_human"]:
        msg.append(f"Score in the uncertain band {REFER_BAND}; refer to a human.")
    msg.append("A confident score can still be wrong. Research model only.")
    return {"parasitized": p, "uninfected": 1 - p}, "\n\n".join(msg)

examples = sorted(str(p) for p in (HERE / "examples").glob("*.png"))
with gr.Blocks(title="Malaria cell classifier", delete_cache=(600, 600)) as demo:
    gr.Markdown(BANNER)
    with gr.Row():
        with gr.Column():
            f = gr.File(label="Cell image", file_types=[".png", ".jpg", ".jpeg"], type="filepath")
            btn = gr.Button("Analyse")
        with gr.Column():
            lab = gr.Label(num_top_classes=2)
            txt = gr.Markdown()
    if examples:
        gr.Examples(examples=[[p] for p in examples], inputs=f)
    btn.click(classify, f, [lab, txt])

if __name__ == "__main__":
    demo.launch()
