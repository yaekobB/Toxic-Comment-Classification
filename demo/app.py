# app.py — Classify + Explain (Captum IG) — polished UX

# (Optional) silence common warnings on Windows/HF
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gradio as gr
from transformers import AutoModel, AutoTokenizer, AutoConfig
from safetensors.torch import load_file
from captum.attr import LayerIntegratedGradients  # explainability

# ----------------------------
# Paths / labels / config
# ----------------------------
ARTI_DIR   = "artifacts"
BEST_DIR   = os.path.join(ARTI_DIR, "best")
THRESH_FP  = os.path.join(ARTI_DIR, "thresholds.json")

LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
NUM_LABELS = len(LABELS)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 256
BASE_MODEL = "distilbert-base-uncased"  # same backbone as training

# ----------------------------
# Model definition (same logic)
# ----------------------------
class ToxicMultiLabel(nn.Module):
    """
    DistilBERT backbone + single linear head -> multi-label logits.
    (We apply sigmoid at inference to get probabilities.)
    """
    def __init__(self, base_model_name: str, num_labels: int, head_dropout: float = 0.30):
        super().__init__()
        cfg = AutoConfig.from_pretrained(base_model_name)
        self.backbone = AutoModel.from_pretrained(base_model_name, config=cfg)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids=None, attention_mask=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]             # [CLS]-like token
        logits = self.classifier(self.dropout(cls))   # (B, L)
        return logits

# ----------------------------
# Load artifacts (tokenizer, model, thresholds)
# ----------------------------
def load_artifacts():
    # tokenizer (prefer the saved one if present)
    tok_src = BEST_DIR if os.path.isfile(os.path.join(BEST_DIR, "tokenizer.json")) else BASE_MODEL
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)

    # model weights
    model = ToxicMultiLabel(BASE_MODEL, NUM_LABELS)
    safep = os.path.join(BEST_DIR, "model.safetensors")
    binp  = os.path.join(BEST_DIR, "pytorch_model.bin")

    if os.path.isfile(safep):
        state = load_file(safep)
    elif os.path.isfile(binp):
        state = torch.load(binp, map_location="cpu")
    else:
        raise FileNotFoundError("No weights found (model.safetensors / pytorch_model.bin) in artifacts/best/")

    # strip training-only keys if any slipped in
    for k in list(state.keys()):
        if k.startswith("pos_weight") or k.startswith("loss_fn"):
            state.pop(k, None)

    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()

    # thresholds
    if os.path.isfile(THRESH_FP):
        with open(THRESH_FP) as f:
            thresholds = json.load(f)
    else:
        thresholds = {lab: 0.5 for lab in LABELS}
        os.makedirs(ARTI_DIR, exist_ok=True)
        with open(THRESH_FP, "w") as f:
            json.dump(thresholds, f, indent=2)

    return model, tok, thresholds

MODEL, TOK, THRESH = load_artifacts()

# =========================
# Inference (Classify tab)
# =========================
@torch.no_grad()
def classify_comment(text: str):
    """
    Returns: (DataFrame of per-label predictions, comma-separated positives)
    """
    text = (text or "").strip()
    if not text:
        return pd.DataFrame(columns=["label","probability","threshold","margin","decision"]), "(none)"

    enc = TOK(text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    logits = MODEL(**enc).squeeze(0).detach().cpu().numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    rows = []
    for i, lab in enumerate(LABELS):
        p = float(probs[i])
        t = float(THRESH.get(lab, 0.5))
        rows.append({
            "label": lab,
            "probability": round(p, 4),
            "threshold": round(t, 4),
            "margin": round(p - t, 4),
            "decision": "POS" if p >= t else "NEG",
        })

    df = pd.DataFrame(rows).sort_values(
        ["decision", "margin", "probability"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    positives = [r["label"] for r in rows if r["probability"] >= r["threshold"]]
    return df, ", ".join(positives) if positives else "(none)"

# =========================
# Explainability (IG tab)
# =========================
# Layer IG on embedding layer
EMB_LAYER = MODEL.backbone.embeddings.word_embeddings

# Captum forward: single logit for chosen label
def _forward_for_label(input_ids, attention_mask, class_index: int):
    logits = MODEL(input_ids=input_ids, attention_mask=attention_mask)  # (B, L)
    return logits[:, class_index]

LIG = LayerIntegratedGradients(_forward_for_label, EMB_LAYER)

def _tokenize_with_offsets(text: str):
    return TOK(text, truncation=True, padding=True, max_length=MAX_LEN,
               return_tensors="pt", return_offsets_mapping=True)

def _merge_wordpieces(tokens, offsets, scores):
    """Merge WordPiece tokens (##subwords) into words; sum scores."""
    words = []
    for tok_piece, (start, end), sc in zip(tokens, offsets, scores):
        # skip special tokens with (0,0) offsets
        if (start, end) == (0, 0) and tok_piece.startswith("[") and tok_piece.endswith("]"):
            continue
        if tok_piece.startswith("##") and words:
            words[-1]["text"] += tok_piece[2:]
            words[-1]["end"]   = end
            words[-1]["score"] += float(sc)
        else:
            words.append({"text": tok_piece, "start": start, "end": end, "score": float(sc)})
    return words

@torch.no_grad()
def _predict_probs(text: str):
    enc = TOK(text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    logits = MODEL(**enc).squeeze(0).detach().cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))  # (L,)

def explain_comment(text: str, target_label: str, steps: int = 30):
    """
    Returns (HTML with colored spans, selected label prob as string).
    Red = supports the label; Blue = opposes the label.
    """
    import html as ihtml

    text = (text or "").strip()
    if not text:
        return "<i>Provide a comment to explain.</i>", "0.000"

    idx = LABELS.index(target_label)
    enc = _tokenize_with_offsets(text)
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    offsets        = enc["offset_mapping"][0].tolist()
    tokens         = TOK.convert_ids_to_tokens(enc["input_ids"][0])

    # PAD baseline
    ref_ids = torch.full_like(input_ids, TOK.pad_token_id)

    # Be robust to Captum return signature
    res = LIG.attribute(
        inputs=input_ids,
        baselines=ref_ids,
        additional_forward_args=(attention_mask, idx),
        n_steps=int(max(4, steps)),
        return_convergence_delta=True,
    )
    attributions = res[0] if isinstance(res, tuple) else res
    token_attr = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    pieces = _merge_wordpieces(tokens, offsets, token_attr)
    arr = np.array([p["score"] for p in pieces], dtype=np.float32)
    denom = float(np.max(np.abs(arr))) if np.max(np.abs(arr)) > 1e-8 else 1.0
    for p in pieces:
        p["score_norm"] = p["score"] / denom

    def _color_for(s: float) -> str:
        alpha = min(1.0, max(0.06, abs(s)))
        return f"rgba(255,0,0,{alpha:.25f})" if s >= 0 else f"rgba(0,0,255,{alpha:.25f})"

    out, last = "", 0
    for p in pieces:
        out += ihtml.escape(text[last:p["start"]])
        out += (
            f'<span title="score={p["score_norm"]:+.3f}" '
            f'style="background:{_color_for(p["score_norm"])}; padding:1px 2px; border-radius:3px;">'
            f'{ihtml.escape(text[p["start"]:p["end"]])}</span>'
        )
        last = p["end"]
    out += ihtml.escape(text[last:])

    probs = _predict_probs(text)
    prob = float(probs[idx])
    header = (
        f"<h4 style='margin:6px 0;'>Label: <code>{target_label}</code> "
        f"| Prob: {prob:.3f}</h4>"
        "<div style='margin:4px 0 8px 0;'>Legend: "
        "<span style='background:rgba(255,0,0,.25);padding:0 6px;'>supports</span> &nbsp; "
        "<span style='background:rgba(0,0,255,.25);padding:0 6px;'>opposes</span></div>"
    )
    html_block = header + f"<div style='font-family:ui-sans-serif,system-ui;line-height:1.7;font-size:15px;'>{out}</div>"
    return html_block, f"{prob:.3f}"

# =========================
# Gradio UI (shared textbox)
# =========================
EXAMPLES = [
    "You are a complete idiot. Get banned already.",
    "I will kill you tomorrow. Watch your back.",
    "Thanks for your help—really appreciate your time!",
    "Shut up, this is the dumbest edit ever.",
    "Go away, you people don't belong here.",
]

with gr.Blocks(
    title="🧠 Toxic Comment Classifier & Explainer",
    theme=gr.themes.Soft(primary_hue="blue")
) as demo:
    gr.Markdown(
        f"""
# 🧠 Toxic Comment Classifier & Explainer
A DistilBERT-based **multi-label** model for detecting toxicity in online comments  
with **Integrated Gradients** explanations (Captum).

**Device:** `{DEVICE}` &nbsp;&nbsp;•&nbsp;&nbsp; **Max length:** {MAX_LEN}
"""
    )

    # Shared textbox (one input for both tabs)
    txt = gr.Textbox(
        label="Enter a comment",
        lines=4,
        value=EXAMPLES[1],
        placeholder="Type or paste a comment here…"
    )

    with gr.Tab("🔍 Classify"):
        btn = gr.Button("Classify", variant="primary")
        out_tbl = gr.Dataframe(
            headers=["label","probability","threshold","margin","decision"],
            label="Per-label predictions",
            interactive=False, wrap=True
        )
        out_pos = gr.Textbox(label="Predicted positive labels", interactive=False)
        btn.click(classify_comment, inputs=txt, outputs=[out_tbl, out_pos])
        gr.Examples(EXAMPLES, inputs=txt, label="Examples")

    with gr.Tab("🧩 Explain"):
        lab_dd = gr.Dropdown(choices=LABELS, value="toxic", label="Target label")
        steps_slider = gr.Slider(6, 80, value=30, step=2,
                                 label="IG steps (higher = smoother, slower)")
        explain_btn = gr.Button("Generate explanation", variant="primary")
        prob_box = gr.Textbox(label="Selected label probability", interactive=False)
        html_vis = gr.HTML(label="Attribution heatmap")
        explain_btn.click(
            fn=explain_comment,
            inputs=[txt, lab_dd, steps_slider],   # shared text
            outputs=[html_vis, prob_box]
        )
        gr.Examples(EXAMPLES, inputs=txt, label="Examples for Explain")

    with gr.Accordion("ℹ️ About & Responsible Use", open=False):
        gr.Markdown(
            """
**Labels:** `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`  
This demo is for **research/education**. Do not use as-is for moderation without
human oversight, bias assessment, and policy alignment. Explanations
(IG attributions) are **heuristics**, not proof of model causality.
"""
        )

if __name__ == "__main__":
    # For HF Spaces, you can use: demo.launch(share=False)
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)
