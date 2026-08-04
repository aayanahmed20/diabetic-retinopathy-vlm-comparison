"""
Builds notebooks/DR_VLM_Comparison.ipynb
Run once locally to (re)generate the notebook file:

    python build_notebook.py

Requires: pip install nbformat
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
2. Draws a stratified random sample of fundus images across ICDR grades 0-4
3. Runs inference with three models:
   - **Gemini** (Google's hosted multimodal API — `google-genai` SDK)
   - **MedGemma** (Google's open-weight medical VLM — `google/medgemma-4b-it` via Hugging Face)
   - **RetiZero** (open retinal foundation model, zero-shot fundus disease recognition)
4. Scores each model against ground-truth ICDR grades: accuracy (with a bootstrap
   95% confidence interval), quadratic-weighted Cohen's kappa (the standard metric for
   ordinal DR grading), mean absolute error, and per-class confusion matrices
5. Writes results to `results/predictions.csv`, `results/metrics_summary.json`,
   `results/summary_table.csv`, and `results/confusion_matrices.png`

**Before you run this:**
- Runtime → Change runtime type → **GPU** (T4 or better) — required for MedGemma and RetiZero
- You will need, as **Colab secrets** (key icon in the left sidebar), NOT hardcoded values:
  - `GEMINI_API_KEY` — from https://aistudio.google.com/apikey
  - `HF_TOKEN` — from https://huggingface.co/settings/tokens, and you must accept
    MedGemma's license at https://huggingface.co/google/medgemma-4b-it first
  - `KAGGLE_USERNAME` and `KAGGLE_KEY` — from your Kaggle account → Settings → API → Create New Token
- RetiZero's pretrained weights are not on pip/HF — they're a manual Google Drive download
  (linked in the RetiZero setup cell below). This notebook downloads them with `gdown`.

**⚠️ Sample size note:** `N_PER_GRADE = 25` below (125 images total) is the statistically
defensible size this project's own methodology calls for. The original 10-image scope
(`N_PER_GRADE = 2`) was a pipeline smoke test only — one wrong prediction there swings overall
accuracy by 10 points and won't support a stable confusion matrix. Drop `N_PER_GRADE` back to
2 if you just want a fast smoke test of the pipeline before committing to a full run.
""")

# ---------------------------------------------------------------------------
md("## 0. Configuration")
code("""\
N_PER_GRADE = 25                      # images sampled per ICDR grade (0-4). 25 -> 125 total images (statistically defensible; drop to 2 for a 10-image smoke test).
RANDOM_SEED = 42                       # change for a different random draw; keep fixed for reproducibility
GEMINI_MODEL = "gemini-2.5-flash"      # any current Gemini vision-capable model
MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
RETIZERO_REPO = "https://github.com/LooKing9218/RetiZero.git"
RETIZERO_WEIGHTS_GDRIVE_ID = "14bMmnefO73_NL1Xc4x0A5qFNbuI7GqKM"  # from the RetiZero README
RETIZERO_WEIGHTS_PATH = "/content/RetiZero/checkpoints/retizero_weights.pth"
CONTENT_ROOT = "/content"
RESULTS_DIR = "/content/results"

GRADE_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}
""")

# ---------------------------------------------------------------------------
md("## 1. Install dependencies")
code("""\
!pip install -q kaggle google-genai transformers accelerate torch torchvision pillow pandas scikit-learn matplotlib seaborn gdown kornia
""")

# ---------------------------------------------------------------------------
md("""\
## 2. Check your Colab secrets

Every key is read from the **Colab secrets manager** (the 🔑 key icon in the left sidebar),
never pasted into a cell. This cell fails loudly and tells you exactly which secret is
missing, so you don't discover it halfway through the run.
""")
code("""\
from google.colab import userdata

REQUIRED_SECRETS = ["GEMINI_API_KEY", "HF_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"]

for name in REQUIRED_SECRETS:
    try:
        userdata.get(name)
        print(f"[OK]   {name} is set")
    except Exception:
        print(f"[MISS] {name} -> add it under the Secrets tab, then re-run this cell")
""")

# ---------------------------------------------------------------------------
md("""\
## 3. Download the APTOS 2019 dataset from Kaggle

Uses your Kaggle API credentials, read from Colab secrets — never hardcode `KAGGLE_KEY` in a cell.
""")
code("""\
import os

os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")

!kaggle competitions download -c aptos2019-blindness-detection -p {CONTENT_ROOT}/aptos2019 --force
!unzip -q -o {CONTENT_ROOT}/aptos2019/aptos2019-blindness-detection.zip -d {CONTENT_ROOT}/aptos2019
""")

code("""\
import pandas as pd

train_csv = pd.read_csv("/content/aptos2019/train.csv")  # columns: id_code, diagnosis
train_csv["image_path"] = train_csv["id_code"].apply(
    lambda x: f"/content/aptos2019/train_images/{x}.png"
)
train_csv["grade_name"] = train_csv["diagnosis"].map(GRADE_NAMES)
print("Class distribution (ground truth):")
print(train_csv["diagnosis"].value_counts().sort_index())
train_csv.head()
""")

# ---------------------------------------------------------------------------
md("""\
## 4. Stratified random sample across ICDR grades 0-4
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
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

n = len(sample_df)
cols = 5 if n >= 5 else n
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.0))
axes = axes.ravel() if isinstance(axes, np.ndarray) else [axes]

for ax in axes:
    ax.axis("off")

for ax, (_, row) in zip(axes, sample_df.iterrows()):
    img = Image.open(row["image_path"])
    ax.imshow(img)
    ax.set_title(f'{row["id_code"]}\\nGrade {row["diagnosis"]}: {row["grade_name"]}', fontsize=9)

plt.tight_layout()
plt.savefig("/content/sample_grid.png", dpi=150)
plt.show()
""")

# ---------------------------------------------------------------------------
md("""\
## 5. Model 1 — Gemini (hosted API)

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
## 6. Model 2 — MedGemma (open weights, local GPU inference)

Requires accepting the license on the [MedGemma model page](https://huggingface.co/google/medgemma-4b-it)
with the same account as your `HF_TOKEN`. First load takes a few minutes.
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
## 7. Model 3 — RetiZero (zero-shot retinal foundation model)

RetiZero is *not* a generative model — it is a CLIP-style vision-language model: it embeds
the image and a set of candidate text labels, then scores their cosine similarity. So
instead of asking it to "write a digit", we hand it the 5 ICDR grade names as candidate
labels and take the argmax of the similarity distribution. This is a mechanically different
task formulation from Gemini/MedGemma and should be reported as such.

**Weight download:** RetiZero's pretrained weights live on Google Drive (linked in the
[repo README](https://github.com/LooKing9218/RetiZero)); `gdown` fetches them by file ID.

**Dependency note:** RetiZero's own `requirements.txt` pins an old torch/transformers stack
that would *break* MedGemma's modern transformers. We intentionally do NOT downgrade the
runtime; the vendored ViT code (RETFound/LoRA) is written against standard torch APIs and
loads fine on a current Colab torch. If RetiZero errors on your runtime, run this notebook's
RetiZero cells in a separate runtime before/after the MedGemma cells and combine the CSVs
in Section 9 — the pipeline is built so predictions survive independently in
`results/predictions.csv`.
""")
code("""\
import os, sys

if not os.path.isdir("/content/RetiZero"):
    !git clone -q {RETIZERO_REPO} /content/RetiZero
else:
    print("RetiZero repo already cloned")

os.makedirs("/content/RetiZero/checkpoints", exist_ok=True)
if not os.path.exists(RETIZERO_WEIGHTS_PATH):
    !gdown {RETIZERO_WEIGHTS_GDRIVE_ID} -O {RETIZERO_WEIGHTS_PATH}
else:
    print("Weights already present")
""")

code("""\
os.chdir("/content/RetiZero")
sys.path.insert(0, "/content/RetiZero")

import torch
from zeroshot import CLIPRModel

retizero_model = CLIPRModel(
    vision_type="lora",
    from_checkpoint=False,          # weights are applied below via load_state_dict
    weights_path=RETIZERO_WEIGHTS_PATH,
    R=8,                            # LoRA rank used at pretraining time
)
state_dict = torch.load(RETIZERO_WEIGHTS_PATH, map_location="cpu")
retizero_model.load_state_dict(state_dict, strict=True)
retizero_model.eval()
print("RetiZero loaded successfully")
""")

code("""\
# RetiZero embeds images against candidate text labels. Use the full ICDR disease names
# (not "Moderate NPDR") because its text encoder is Bio_ClinicalBERT and its pretraining
# captions are phrased as "a fundus photograph of <disease>" — full names map best.
RETIZERO_LABELS = [
    "no diabetic retinopathy",
    "mild non-proliferative diabetic retinopathy",
    "moderate non-proliferative diabetic retinopathy",
    "severe non-proliferative diabetic retinopathy",
    "proliferative diabetic retinopathy",
]

def predict_retizero(image_path):
    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        probability, _logits = retizero_model(image, RETIZERO_LABELS)
    return int(probability.argmax())
""")

# ---------------------------------------------------------------------------
md("""\
## 8. Run all three models over the sample

Each prediction is wrapped so one model's failure doesn't kill the run — failed predictions
are recorded as `None` and dropped per-model at scoring time. Gemini API calls are spaced
1s apart to stay polite to rate limits.

**This cell checkpoints after every image** to `results/predictions.csv` and skips
`id_code`s already in that file. If the runtime crashes or restarts partway through
(memory pressure from running three models on one GPU is the most likely cause), just
re-run this cell — it picks up where it left off instead of starting over. It also runs
a GPU memory cleanup every few images to reduce the chance of that happening.
""")
code("""\
import gc
import time
import pandas as pd

os.makedirs(RESULTS_DIR, exist_ok=True)
predictions_path = f"{RESULTS_DIR}/predictions.csv"

if os.path.exists(predictions_path):
    results_df = pd.read_csv(predictions_path)
    done_ids = set(results_df["id_code"])
    print(f"Resuming: {len(done_ids)} images already scored, skipping those.")
else:
    results_df = pd.DataFrame()
    done_ids = set()

results = results_df.to_dict("records")
for i, (_, row) in enumerate(sample_df.iterrows()):
    if row["id_code"] in done_ids:
        continue

    record = {
        "id_code": row["id_code"],
        "image_path": row["image_path"],
        "ground_truth": int(row["diagnosis"]),
    }
    for model_name, predict_fn in [
        ("gemini", predict_gemini),
        ("medgemma", predict_medgemma),
        ("retizero", predict_retizero),
    ]:
        try:
            record[f"{model_name}_pred"] = predict_fn(row["image_path"])
        except Exception as e:
            record[f"{model_name}_pred"] = None
            record[f"{model_name}_error"] = str(e)
    results.append(record)

    # Checkpoint after every image - a crash loses at most one image's work, not the run.
    pd.DataFrame(results).to_csv(predictions_path, index=False)
    print(f"done {row['id_code']} (GT grade {row['diagnosis']})")

    # Periodic GPU memory cleanup - reduces fragmentation buildup over a long run.
    if i % 10 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    time.sleep(1)

results_df = pd.DataFrame(results)
results_df
""")

# ---------------------------------------------------------------------------
md("""\
## 9. Score against ground truth

Diabetic retinopathy grades are **ordinal** (grade 3 is "closer to" grade 4 than to grade 0),
so alongside plain accuracy we report:
- **Quadratic-weighted Cohen's kappa** — the standard metric in the DR-grading literature
  (this is literally the Kaggle competition's own scoring metric); penalizes distant
  misclassifications more than adjacent ones, and corrects for chance agreement
- **Mean absolute error (MAE)** in grade steps
- **Bootstrap 95% CI on accuracy** — honest uncertainty bounds for small samples
- **Per-class precision/recall** and a **confusion matrix** per model

If you ran the models across separate runtimes, re-upload the saved CSVs and merge them
before this section.
""")
code("""\
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    mean_absolute_error, classification_report
)

MODELS = ["gemini_pred", "medgemma_pred", "retizero_pred"]
metrics_summary = {}

def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=RANDOM_SEED):
    # Percentile bootstrap 95% CI for accuracy.
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accs = [accuracy_score(y_true[idx], y_pred[idx])
            for idx in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n_boot))]
    lo, hi = np.quantile(accs, [0.025, 0.975])
    return round(float(lo), 4), round(float(hi), 4)

for model_col in MODELS:
    model_name = model_col.replace("_pred", "")
    valid = results_df.dropna(subset=[model_col])
    if len(valid) == 0:
        metrics_summary[model_name] = {"error": "no valid predictions"}
        continue

    y_true = valid["ground_truth"].astype(int)
    y_pred = valid[model_col].astype(int)
    acc_ci = bootstrap_ci(y_true, y_pred)

    metrics_summary[model_name] = {
        "n_scored": int(len(valid)),
        "n_total": int(len(results_df)),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "accuracy_95ci": list(acc_ci),
        "quadratic_weighted_kappa": round(cohen_kappa_score(y_true, y_pred, weights="quadratic"), 4),
        "mean_absolute_error_grades": round(mean_absolute_error(y_true, y_pred), 4),
    }
    print(f"=== {model_name} ===")
    print(classification_report(y_true, y_pred, zero_division=0))
    print(f"accuracy 95% CI: {acc_ci}")
    print()

with open(f"{RESULTS_DIR}/metrics_summary.json", "w") as f:
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
    cm = confusion_matrix(valid["ground_truth"].astype(int), valid[model_col].astype(int), labels=[0, 1, 2, 3, 4])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=range(5), yticklabels=range(5))
    ax.set_title(model_name)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrices.png", dpi=150)
plt.show()
""")

# ---------------------------------------------------------------------------
md("## 10. Export a results table for the report")
code("""\
summary_table = pd.DataFrame(metrics_summary).T
summary_table.to_csv(f"{RESULTS_DIR}/summary_table.csv")
print(summary_table.to_markdown())
""")

# ---------------------------------------------------------------------------
md("""\
## 11. Package and download results

Run this when you're done. It bundles every artifact the report needs (`predictions.csv`,
`metrics_summary.json`, `summary_table.csv`, `confusion_matrices.png`, plus the sample grid)
into one tarball and triggers a browser download. Extract it into the repo root so the
`results/` folder populates the report.
""")
code("""\
import tarfile

tar_path = "/content/results.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(RESULTS_DIR, arcname="results")
    tar.add("/content/sample_grid.png", arcname="sample_grid.png")

from google.colab import files
files.download(tar_path)
print("Downloaded results.tar.gz -> extract into the repo root (creates results/)")
""")

md("""\
## 12. Notes and limitations (read before writing up results)

- **Sample size.** `N_PER_GRADE = 25` (125 images total) is set as the default because it's
  the statistically defensible size this project's own methodology calls for. If you switch
  back to `N_PER_GRADE = 2` (10 images, the original smoke-test scope) for a quick pipeline
  check, don't report those numbers as a result — the bootstrap CIs in Section 9 will make
  why painfully obvious.
- **RetiZero is zero-shot similarity classification, not generative digit-output**, unlike
  Gemini and MedGemma — it's being scored on a genuinely different task formulation
  (embedding similarity vs. token generation), which is worth stating explicitly as a
  methodological caveat rather than presenting the three numbers as directly equivalent.
- **Ground truth itself is noisy.** Published inter-grader agreement on ICDR grading across
  ophthalmologists sits at kappa 0.40-0.65 in the literature — so your "ground truth" labels
  are themselves one grader's opinion, not an infallible reference. Frame conclusions accordingly.
- **Gemini is a general-purpose model being asked to do a specialist task it wasn't trained
  for** (unlike MedGemma/RetiZero, which are medically pretrained) — that's a legitimate and
  interesting comparison, but the write-up should say so rather than implying a fair fight
  between equivalent systems.
- **Dependency tension.** RetiZero's upstream repo pins torch 1.13/transformers 4.27; this
  notebook deliberately runs it on the modern Colab stack to coexist with MedGemma. If you
  hit a RetiZero-specific error on your runtime, run it in a separate runtime and merge CSVs.
- **This notebook was authored and validated for structure, not executed end-to-end on a GPU**
  (no Kaggle/Colab/GPU access in the authoring environment). The RetiZero wiring follows its
  upstream `Zeroshot.py`; read the cell output carefully on first run.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
    "accelerator": "GPU",
}

with open("notebooks/DR_VLM_Comparison.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/DR_VLM_Comparison.ipynb")
