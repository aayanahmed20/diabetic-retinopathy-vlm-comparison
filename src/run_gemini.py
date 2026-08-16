"""
Minimal model-run adapters for the Diabetic Retinopathy repo.
Each script supports a --mock flag so CI can run without keys/weights.
Outputs a CSV with columns: id_code, gemini_pred/medgemma_pred/retizero_pred
"""

# src/run_gemini.py
from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import Optional


def predict_mock(image_path: str) -> Optional[int]:
    # deterministic mock: hash-based small pseudo-random grade 0-4
    return abs(hash(str(image_path))) % 5


try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False


def predict_gemini_real(client, image_path: str) -> Optional[int]:
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/png"), "Classify DR severity 0-4. Return ONLY the digit."]
        )
        text = getattr(response, "text", "")
        digits = ''.join(filter(str.isdigit, text))
        return int(digits[0]) if digits else None
    except Exception:
        return None


def run(input_csv: str, output_csv: str, mock: bool):
    import pandas as pd
    df = pd.read_csv(input_csv)
    rows = []

    # Lazy client creation only when needed and not mocking
    client = None
    if not mock and HAS_GENAI:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    for r in df.itertuples(index=False):
        image_path = getattr(r, "image_path", None)
        if mock:
            pred = predict_mock(image_path)
        else:
            if not image_path or not os.path.exists(image_path):
                pred = None
            elif client is None:
                pred = None
            else:
                pred = predict_gemini_real(client, image_path)
        rows.append({"id_code": getattr(r, "id_code"), "gemini_pred": pred})

    outdf = pd.DataFrame(rows)
    # merge with existing if present
    if os.path.exists(output_csv):
        try:
            existing = pd.read_csv(output_csv)
            outdf = existing.merge(outdf, on="id_code", how="outer")
        except Exception:
            pass
    outdf.to_csv(output_csv, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()
    run(args.input, args.output, args.mock)
