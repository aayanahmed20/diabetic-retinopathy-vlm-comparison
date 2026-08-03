"""
Builds notebooks/DR_VLM_Comparison.ipynb
Run once locally to (re)generate the notebook file.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ---------------------------------------------------------------------------
md("""\
# Diabetic Retinopathy Grading: Gemini vs. MedGemma vs. RetiZero

**Comparative evaluation of three vision-language models on ICDR-graded fundus images (APTOS 2019)**

This notebook:
1. Downloads the APTOS 2019 Blindness Detection dataset from Kaggle
2. Draws a stratified random sample of fundus images across ICDR grades 0–4
3. Runs inference with three models:
   - **Gemini** (Google's hosted multimodal API — `google-genai` SDK)
   - **MedGemma** (Google's open-weight medical VLM — `google/medgemma-4b-it` via Hugging Face)
   - **RetiZero** (open retinal foundation model, zero-shot fundus disease recognition)
4. Scores each model against ground-truth ICDR grades: accuracy, per-class accuracy,
   quadratic-weighted Cohen's kappa (the standard metric for ordinal DR grading), and
   mean absolute error
5. Writes results to `results/predictions.csv` and `results/metrics_summary.json`

**Before you run this:**
- Runtime → Change runtime type → **GPU** (T4 or better) — required for MedGemma and RetiZero
- You will need, as **Colab secrets** (key icon in the left sidebar), NOT hardcoded values:
  - `GEMINI_API_KEY` — from https://aistudio.google.com/apikey
  - `HF_TOKEN` — from https://huggingface.co/settings/tokens, and you must accept
    MedGemma's license at https://huggingface.co/google/medgemma-4b-it first
  - `KAGGLE_USERNAME` and `KAGGLE_KEY` — from your Kaggle account → Settings → API → Create New Token
- RetiZero's pretrained weights are not on pip/HF — they're a manual Google Drive download
  (linked in the RetiZero setup cell below). This notebook downloads them with `gdown`.

**⚠️ Sample size note:** `N_PER_GRADE = 25` below (125 images total) is the low end of what's
defensible for a paper — the original "10 images" ask (`N_PER_GRADE = 2`) is a pipeline
smoke-test only: one wrong prediction swings overall accuracy by 10 points, and 2 images/class
can't support a stable confusion matrix or per-class precision/recall. Raise further (30-40/class)
if your Gemini API quota and Colab session length allow it.
""")

# ---------------------------------------------------------------------------
md("## 0. Configuration")
code("""\
N_PER_GRADE = 25         # images sampled per ICDR grade (0-4). 25 -> 125 total images.
RANDOM_SEED = 42          # change for a different random draw; keep fixed for reproducibility
GEMINI_MODEL = "gemini-2.5-flash"          # any current Gemini vision-capable model
MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
RETIZERO_REPO = "https://github.com/LooKing9218/RetiZero.git"
RETIZERO_WEIGHTS_GDRIVE_ID = "14bMmnefO73_NL1Xc4x0A5qFNbuI7GqKM"  # from RetiZero README

GRADE_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}
""")

# ---------------------------------------------------------------------------
md("""\
## 1. Install dependencies
""")
code("""\
!pip install -q kaggle google-genai transformers accelerate pillow pandas scikit-learn matplotlib seaborn gdown
""")

# ---------------------------------------------------------------------------
md("""\
## 2. Download the APTOS 2019 dataset from Kaggle

Uses your Kaggle API credentials, read from Colab secrets — never hardcode `KAGGLE_KEY` in a cell.
""")
code("""\
import os
from google.colab import userdata

os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")

!kaggle competitions download -c aptos2019-blindness-detection -p /content/aptos2019 --force
!unzip -q -o /content/aptos2019/aptos2019-blindness-detection.zip -d /content/aptos2019
""")

code("""\
import pandas as pd

train_csv = pd.read_csv("/content/aptos2019/train.csv")  # columns: id_code, diagnosis
train_csv["image_path"] = train_csv["id_code"].apply(
    lambda x: f"/content/aptos2019/train_images/{x}.png"
)
train_csv["grade_name"] = train_csv["diagnosis"].map(GRADE_NAMES)
print(train_csv["diagnosis"].value_counts().sort_index())
train_csv.head()
""")

# ---------------------------------------------------------------------------
md("""\
## 3. Stratified random sample across ICDR grades 0–4
""")
code("""\
sample_df = (
    train_csv
    .groupby("diagnosis", group_keys=False)
    .apply(lambda g: g.sample(n=min(N_PER_GRADE, len(g)), random_state=RANDOM_SEED))
    .reset_index(drop=True)
)
sample_df = sample_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle
print(f"Sampled {len(sample_df)} images:")
sample_df[["id_code", "diagnosis", "grade_name"]]
""")

code("""\
import matplotlib.pyplot as plt
from PIL import Image

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
for ax, (_, row) in zip(axes.flat, sample_df.iterrows()):
    img = Image.open(row["image_path"])
    ax.imshow(img)
    ax.set_title(f'{row[\"id_code\"]}\\nGrade {row[\"diagnosis\"]}: {row[\"grade_name\"]}', fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("/content/sample_grid.png", dpi=150)
plt.show()
""")

# ---------------------------------------------------------------------------
md("""\
## 4. Model 1 — Gemini (hosted API)

The grading prompt below encodes the actual ICDR rubric rather than asking Gemini to
free-associate a number, which noticeably improves grading consistency.
""")
code("""\
from google import genai
from google.genai import types

gemini_client = genai.Client(api_key=userdata.get("GEMINI_API_KEY"))

ICDR_PROMPT = '''You are assisting with diabetic retinopathy severity grading on a color fundus
photograph, using the International Clinical Diabetic Retinopathy (ICDR) severity scale.

Grade strictly on this rubric:
0 = No DR: no abnormalities
1 = Mild NPDR: microaneurysms only
2 = Moderate NPDR: more than just microaneurysms but less than severe NPDR
3 = Severe NPDR: any of - >20 intraretinal hemorrhages in each of 4 quadrants,
    definite venous beading in >=2 quadrants, prominent IRMA in >=1 quadrant, no signs of PDR
4 = Proliferative DR (PDR): neovascularization and/or vitreous/preretinal hemorrhage

Respond with ONLY a single digit 0-4. No words, no punctuation, no explanation.'''

def predict_gemini(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ICDR_PROMPT,
        ],
    )
    text = response.text.strip()
    digits = [c for c in text if c.isdigit()]
    return int(digits[0]) if digits else None
""")

# ---------------------------------------------------------------------------
md("""\
## 5. Model 2 — MedGemma (open weights, local GPU inference)

Requires accepting the license on the [MedGemma model page](https://huggingface.co/google/medgemma-4b-it)
with the same account as your `HF_TOKEN`.
""")
code("""\
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

medgemma_processor = AutoProcessor.from_pretrained(MEDGEMMA_MODEL_ID, token=os.environ["HF_TOKEN"])
medgemma_model = AutoModelForImageTextToText.from_pretrained(
    MEDGEMMA_MODEL_ID,
    token=os.environ["HF_TOKEN"],
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
""")

code("""\
def predict_medgemma(image_path):
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": ICDR_PROMPT},
            ],
        }
    ]
    inputs = medgemma_processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(medgemma_model.device, dtype=torch.bfloat16)

    with torch.inference_mode():
        output = medgemma_model.generate(**inputs, max_new_tokens=8, do_sample=False)
    decoded = medgemma_processor.decode(
        output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()
    digits = [c for c in decoded if c.isdigit()]
    return int(digits[0]) if digits else None
""")

# ---------------------------------------------------------------------------
md("""\
## 6. Model 3 — RetiZero (zero-shot retinal foundation model)

RetiZero's weights are not hosted on pip/HF — they're a manual Google Drive file linked in
the repo's README. `gdown` fetches it by file ID. RetiZero exposes a `CLIPRModel` class
(`zeroshot/modeling/model.py`) called directly as `model(image, text_list)`, returning
`(probability, logits)` over `text_list` — confirmed against the repo's own `Zeroshot.py`
reference script.

**⚠️ Run this section in its own Colab runtime, separate from Section 5 (MedGemma).**
RetiZero's `requirements.txt` pins `torch~=1.13.1` / `transformers~=4.27.4`; MedGemma needs a
much newer `transformers` for its architecture. Installing both in one environment will likely
break one of them. Recommended flow: `Runtime → Disconnect and delete runtime` after Section 5,
run Section 6 fresh, export `retizero_predictions.csv`, then re-merge in Section 7/8.

**⚠️ Granularity caveat — disclose this in the report.** RetiZero's own demo evaluates 14
disease *types* (Normal, Retinal Vein Occlusion, glaucoma, AMD, etc.), where DR appears as just
two of those types — "Non-proliferative Diabetic Retinopathy" and "Proliferative Diabetic
Retinopathy" — not a mild/moderate/severe 5-way split. Below we pass the 5 ICDR grade names as
custom candidate text anyway (the model accepts arbitrary text, so it's mechanically possible),
but this is an untested extrapolation beyond what RetiZero's paper validated for this specific
task. Report both: (a) this 5-way ICDR attempt, and (b) a secondary, more faithful check using
RetiZero's native categories collapsed to "no DR" vs "any DR" as a binary sanity check.
""")
code("""\
!git clone -q {RETIZERO_REPO} /content/RetiZero
%cd /content/RetiZero
!pip install -q -r requirements.txt
!mkdir -p checkpoints
!gdown {RETIZERO_WEIGHTS_GDRIVE_ID} -O checkpoints/retizero_weights.pth
""")

code("""\
import sys
sys.path.insert(0, "/content/RetiZero")
from zeroshot import CLIPRModel

RETIZERO_WEIGHTS = "checkpoints/retizero_weights.pth"

retizero_model = CLIPRModel(
    vision_type="lora",
    from_checkpoint=False,
    weights_path=RETIZERO_WEIGHTS,
    R=8,
)
_state_dict = torch.load(RETIZERO_WEIGHTS, map_location="cuda")
retizero_model.load_state_dict(_state_dict, strict=True)
retizero_model = retizero_model.eval()

# 5-way ICDR attempt (off-label extrapolation -- see caveat above)
CANDIDATE_LABELS_ICDR = [GRADE_NAMES[i] for i in range(5)]

# RetiZero's native 14-way categories, collapsed to a binary DR-present sanity check
CANDIDATE_LABELS_NATIVE = [
    "Normal", "Retinal Vein Occlusion", "Central Serous Chorioretinopathy",
    "Non-proliferative Diabetic Retinopathy", "Proliferative Diabetic Retinopathy",
    "Epiretinal Membrane", "Glaucoma", "Macular Hole", "Pathologic Maculopathy",
    "Retinal Artery Occulusion", "Retinal Detachment", "Retinitis Pigmentosa",
    "Vogt-Koyanagi-Harada (VKH) disease", "Age-related Macular Degeneration",
]
_DR_NATIVE_INDICES = {3, 4}  # indices of the two DR-related native labels above

def predict_retizero(image_path, candidate_labels=CANDIDATE_LABELS_ICDR):
    '''Returns the index into candidate_labels with highest image-text similarity.'''
    image = Image.open(image_path).convert("RGB")
    probability, _logits = retizero_model(image, candidate_labels)
    return int(probability.argmax(-1))

def predict_retizero_dr_present(image_path):
    '''Binary sanity check using RetiZero's native 14-way categories: True if top match is DR-related.'''
    pred_idx = predict_retizero(image_path, candidate_labels=CANDIDATE_LABELS_NATIVE)
    return pred_idx in _DR_NATIVE_INDICES
""")

# ---------------------------------------------------------------------------
md("""\
## 7. Run all three models over the sample
""")
code("""\
import time

results = []
for _, row in sample_df.iterrows():
    record = {
        "id_code": row["id_code"],
        "image_path": row["image_path"],
        "ground_truth": int(row["diagnosis"]),
    }
    try:
        record["gemini_pred"] = predict_gemini(row["image_path"])
    except Exception as e:
        record["gemini_pred"] = None
        record["gemini_error"] = str(e)

    try:
        record["medgemma_pred"] = predict_medgemma(row["image_path"])
    except Exception as e:
        record["medgemma_pred"] = None
        record["medgemma_error"] = str(e)

    try:
        record["retizero_pred"] = predict_retizero(row["image_path"])
    except Exception as e:
        record["retizero_pred"] = None
        record["retizero_error"] = str(e)

    results.append(record)
    time.sleep(1)  # be polite to the Gemini API rate limit

results_df = pd.DataFrame(results)
os.makedirs("/content/results", exist_ok=True)
results_df.to_csv("/content/results/predictions.csv", index=False)
results_df
""")

# ---------------------------------------------------------------------------
md("""\
## 8. Score against ground truth

Diabetic retinopathy grades are **ordinal** (grade 3 is "closer to" grade 4 than to grade 0),
so alongside plain accuracy we report:
- **Quadratic-weighted Cohen's kappa** — the standard metric in the DR-grading literature
  (this is literally the Kaggle competition's own scoring metric); penalizes distant
  misclassifications more than adjacent ones, and corrects for chance agreement
- **Mean absolute error (MAE)** in grade steps
- **Per-class accuracy** and a **confusion matrix** per model
""")
code("""\
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    mean_absolute_error, classification_report
)
import json

MODELS = ["gemini_pred", "medgemma_pred", "retizero_pred"]
metrics_summary = {}

for model_col in MODELS:
    model_name = model_col.replace("_pred", "")
    valid = results_df.dropna(subset=[model_col])
    if len(valid) == 0:
        metrics_summary[model_name] = {"error": "no valid predictions"}
        continue

    y_true = valid["ground_truth"].astype(int)
    y_pred = valid[model_col].astype(int)

    metrics_summary[model_name] = {
        "n_scored": int(len(valid)),
        "n_total": int(len(results_df)),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "quadratic_weighted_kappa": round(cohen_kappa_score(y_true, y_pred, weights="quadratic"), 4),
        "mean_absolute_error_grades": round(mean_absolute_error(y_true, y_pred), 4),
    }
    print(f"=== {model_name} ===")
    print(classification_report(y_true, y_pred, zero_division=0))
    print()

with open("/content/results/metrics_summary.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

pd.DataFrame(metrics_summary).T
""")

code("""\
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, model_col in zip(axes, MODELS):
    model_name = model_col.replace("_pred", "")
    valid = results_df.dropna(subset=[model_col])
    if len(valid) == 0:
        ax.set_title(f"{model_name}: no predictions")
        ax.axis("off")
        continue
    cm = confusion_matrix(valid["ground_truth"].astype(int), valid[model_col].astype(int), labels=[0,1,2,3,4])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=range(5), yticklabels=range(5))
    ax.set_title(model_name)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
plt.tight_layout()
plt.savefig("/content/results/confusion_matrices.png", dpi=150)
plt.show()
""")

# ---------------------------------------------------------------------------
md("""\
## 9. Export a results table for the report
""")
code("""\
summary_table = pd.DataFrame(metrics_summary).T
summary_table.to_csv("/content/results/summary_table.csv")
print(summary_table.to_markdown())
""")

md("""\
## 10. Notes and limitations (read before writing up results)

- **Sample size.** `N_PER_GRADE = 25` (125 images total) is the low end of defensible for a
  paper. Raise it further if quota/session-length allow, and report a confidence interval on
  accuracy/kappa rather than a bare point estimate.
- **RetiZero is zero-shot embedding similarity, not generative digit-output**, unlike Gemini and
  MedGemma — it's being scored on a genuinely different task formulation. Worth stating
  explicitly rather than presenting the three numbers as directly equivalent.
- **RetiZero's 5-way ICDR grading here is an off-label extrapolation.** Its own published demo
  task is 14-way disease-type classification, where DR appears only as two coarse categories
  (NPDR / PDR), not a mild/moderate/severe split. Report the native binary DR-present check
  (`predict_retizero_dr_present`) alongside the 5-way attempt, and don't present the 5-way
  number as validating RetiZero's severity-grading ability specifically.
- **RetiZero and MedGemma have conflicting dependency pins** (old vs. new `torch`/`transformers`)
  — run them in separate Colab runtimes, not the same `pip install` environment.
- **Ground truth itself is noisy.** Published inter-grader agreement on ICDR grading across
  ophthalmologists sits at kappa 0.40-0.65 in the literature — so your "ground truth" labels
  are themselves one grader's opinion, not an infallible reference. Frame conclusions accordingly.
- **Gemini is a general-purpose model being asked to do a specialist task it wasn't trained
  for** (unlike MedGemma/RetiZero, which are medically pretrained) — that's a legitimate and
  interesting comparison, but the write-up should say so rather than implying a fair fight
  between equivalent systems.
- **This notebook was authored without running it end-to-end** (no Kaggle/Colab/GPU access in
  the environment that generated or updated it — RetiZero's API was confirmed by reading its
  source directly, not by executing it). Read through each cell before trusting it.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
    "accelerator": "GPU",
}

with open("notebooks/DR_VLM_Comparison.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/DR_VLM_Comparison.ipynb")
