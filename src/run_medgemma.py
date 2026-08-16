from __future__ import annotations
import argparse
import os
from typing import Optional


def predict_mock(image_path: str) -> Optional[int]:
    return hash(image_path) % 5


HAS_TRANSFORMERS = False
try:
    import torch
    from transformers import AutoProcessor
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False


def predict_medgemma_real(image_path: str, model, processor) -> Optional[int]:
    try:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        prompt = "Classify DR severity 0-4. Respond with ONLY the digit."
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=10)
        decoded = processor.decode(output[0], skip_special_tokens=True).strip()
        digits = [c for c in decoded if c.isdigit() and c in '01234']
        return int(digits[-1]) if digits else None
    except Exception:
        return None


def run(input_csv: str, output_csv: str, mock: bool):
    import pandas as pd
    df = pd.read_csv(input_csv)
    rows = []

    model = None
    processor = None
    if not mock and HAS_TRANSFORMERS:
        from transformers import AutoProcessor
        MODEL_ID = os.environ.get("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")
        HF_TOKEN = os.environ.get("HF_TOKEN")
        try:
            processor = AutoProcessor.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
            # Lazy import of the model class to avoid heavy imports when mocking
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN, device_map="auto")
            model.eval()
        except Exception:
            model = None
            processor = None

    for r in df.itertuples(index=False):
        image_path = getattr(r, "image_path", None)
        if not image_path or not os.path.exists(image_path):
            pred = None
        elif mock:
            pred = predict_mock(image_path)
        else:
            if model is None or processor is None:
                pred = None
            else:
                pred = predict_medgemma_real(image_path, model, processor)
        rows.append({"id_code": getattr(r, "id_code"), "medgemma_pred": pred})

    outdf = pd.DataFrame(rows)
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        outdf = existing.merge(outdf, on="id_code", how="outer")
    outdf.to_csv(output_csv, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()
    run(args.input, args.output, args.mock)
