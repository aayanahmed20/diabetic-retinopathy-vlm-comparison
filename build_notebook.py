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

**Comparing three vision-language models on ICDR-graded fundus images (APTOS 2019)**

This notebook downloads the dataset, samples images across ICDR grades 0-4, runs all
three models, and scores each one (accuracy + 95% CI, quadratic-weighted kappa, MAE,
confusion matrices) — writing everything to `results/`.

**Sample size:** `N_PER_GRADE = 25` below (125 images total) is the statistically
defensible default. Drop it to `2` only for a quick smoke test of the pipeline — don't
report those numbers as a result.
""")

# ---------------------------------------------------------------------------
md("## 0. Configuration")
code("""\
N_PER_GRADE = 25                      # images sampled per ICDR grade (0-4). 25 -> 125 total images (statistically defensible; drop to 2 for a 10-image smoke test).
RANDOM_SEED = 42                       # change for a different random draw; keep fixed for reproducibility
GEMINI_MODEL = "gemini-3.5-flash"      # current stable Gemini vision model (gemini-2.5-flash is being retired Oct 2026 -- already seeing early "model not found" errors in the wild, so don't use it)
MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
RETIZERO_REPO = "https://github.com/LooKing9218/RetiZero.git"
RETIZERO_WEIGHTS_GDRIVE_ID = "14bMmnefO73_NL1Xc4x0A5qFNbuI7GqKM"  # from the RetiZero README
RETIZERO_WEIGHTS_PATH = "/content/RetiZero/checkpoints/retizero_weights.pth"
CONTENT_ROOT = "/content"
RESULTS_DIR = "/content/results"

# Maps the integer ICDR grade (as stored in the dataset's "diagnosis" column) to its
# clinical name, used for plot titles and the printed class distribution below.
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
# kaggle: dataset download | google-genai: Gemini API | transformers/accelerate/torch/torchvision: MedGemma + RetiZero
# pillow/pandas/scikit-learn/matplotlib/seaborn: image I/O, data handling, metrics, plots | gdown: RetiZero weight download
# tabulate: required by pandas.DataFrame.to_markdown() in Section 10, not pulled in by anything else here
!pip install -q kaggle google-genai transformers accelerate torch torchvision pillow pandas scikit-learn matplotlib seaborn gdown tabulate
""")

# ---------------------------------------------------------------------------
md("""\
## 2. Check your Colab secrets

This cell fails loudly and names any missing secret, rather than failing midway through the run.
""")
code("""\
from google.colab import userdata

# Kaggle accepts EITHER the classic KAGGLE_USERNAME+KAGGLE_KEY pair OR the newer
# single KAGGLE_API_TOKEN (KGAT_... format) -- only one of the two is required, not
# both, so it's checked separately below instead of being lumped into REQUIRED_SECRETS
# (which would incorrectly demand all four Kaggle-related secrets at once).
REQUIRED_SECRETS = ["GEMINI_API_KEY", "HF_TOKEN"]

# userdata.get() raises if a secret isn't set — catch that per-name so one missing
# secret is reported clearly instead of stopping at the first one.
for name in REQUIRED_SECRETS:
    try:
        userdata.get(name)
        print(f"[OK]   {name} is set")
    except Exception:
        print(f"[MISS] {name} -> add it under the Secrets tab, then re-run this cell")

def _secret_set(name):
    try:
        return bool(userdata.get(name))
    except Exception:
        return False

_has_kaggle_token = _secret_set("KAGGLE_API_TOKEN")
_has_kaggle_classic = _secret_set("KAGGLE_USERNAME") and _secret_set("KAGGLE_KEY")
if _has_kaggle_token:
    print("[OK]   KAGGLE_API_TOKEN is set")
elif _has_kaggle_classic:
    print("[OK]   KAGGLE_USERNAME + KAGGLE_KEY are set")
else:
    print(
        "[MISS] Kaggle credentials -> add EITHER KAGGLE_API_TOKEN (new KGAT_... token) "
        "OR both KAGGLE_USERNAME and KAGGLE_KEY (classic) under the Secrets tab. "
        "Not required if you're using Option B (manual zip upload) in Section 3."
    )
""")

# ---------------------------------------------------------------------------
md("""\
## 3. Get the APTOS 2019 dataset

Two ways to get the data in — use whichever actually works for you. The cell below tries
both automatically, in this order:

**Option A: Kaggle API** (works if `KAGGLE_USERNAME`/`KAGGLE_KEY` are set as Colab secrets
*and* you've accepted the competition's rules at
[kaggle.com/competitions/aptos2019-blindness-detection/rules](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules)
— the API returns 401 Unauthorized until you do this once, in a browser, even with valid keys).

**Option B: manual upload** (use this if the API keeps failing). In a browser, on the
[competition's Data tab](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data),
click **Download All** to get `aptos2019-blindness-detection.zip` (a few GB — this is the
full image set, not just the CSVs). Then get that zip file into Colab one of two ways:

- **Google Drive (recommended)** — upload the zip to your Drive, then run:
  ```python
  from google.colab import drive
  drive.mount("/content/drive")
  ```
  in a new cell above this one. Once mounted, the cell below will find the zip
  automatically at `/content/drive/MyDrive/aptos2019-blindness-detection.zip`. This
  survives a runtime restart, so you only upload once even across multiple sessions.
- **Direct upload** — use the Files panel (folder icon, left sidebar) to upload the zip
  straight into `/content/`. Faster to set up, but Colab deletes it when the runtime
  disconnects, so you'd need to re-upload after any restart.
""")
code("""\
import os

CANDIDATE_ZIPS = [
    "/content/drive/MyDrive/aptos2019-blindness-detection.zip",  # Option B, Drive-mounted
    "/content/aptos2019-blindness-detection.zip",                # Option B, direct upload
    f"{CONTENT_ROOT}/aptos2019/aptos2019-blindness-detection.zip",  # Option A, API download
]

os.makedirs(f"{CONTENT_ROOT}/aptos2019", exist_ok=True)
_zip_path = next((p for p in CANDIDATE_ZIPS if os.path.exists(p)), None)

if _zip_path is None:
    # No manually-provided zip found -- try the Kaggle API. Reads credentials from Colab
    # secrets rather than writing a kaggle.json file to disk. Kaggle accepts either the
    # new single-token format (KAGGLE_API_TOKEN, looks like "KGAT_...") or the classic
    # KAGGLE_USERNAME + KAGGLE_KEY pair -- try the new token first since that's what the
    # current Kaggle "Create New Token" flow issues by default.
    def _get_secret_or_none(name):
        try:
            val = userdata.get(name)
            return val if val else None
        except Exception:
            return None

    _kaggle_token = _get_secret_or_none("KAGGLE_API_TOKEN")
    _kaggle_user = _get_secret_or_none("KAGGLE_USERNAME")
    _kaggle_key = _get_secret_or_none("KAGGLE_KEY")

    if _kaggle_token:
        os.environ["KAGGLE_API_TOKEN"] = _kaggle_token
    elif _kaggle_user and _kaggle_key:
        os.environ["KAGGLE_USERNAME"] = _kaggle_user
        os.environ["KAGGLE_KEY"] = _kaggle_key
    else:
        raise RuntimeError(
            "No Kaggle credentials found and no manual zip either. Add KAGGLE_API_TOKEN "
            "(or KAGGLE_USERNAME + KAGGLE_KEY) as a Colab secret, or use Option B "
            "(manual upload) described above."
        )
    !kaggle competitions download -c aptos2019-blindness-detection -p {CONTENT_ROOT}/aptos2019 --force
    _zip_path = f"{CONTENT_ROOT}/aptos2019/aptos2019-blindness-detection.zip"

if os.path.exists(_zip_path):
    print(f"Using zip: {_zip_path}")
    !unzip -q -o {_zip_path} -d {CONTENT_ROOT}/aptos2019

# Verify this actually worked instead of assuming it did: a failed Kaggle API call (401,
# stale key) doesn't raise a Python exception, it just prints an error and lets the
# notebook keep going -- so the very next cell would otherwise fail with a confusing
# "FileNotFoundError: train.csv" instead of pointing at the actual cause.
_train_csv = f"{CONTENT_ROOT}/aptos2019/train.csv"
_train_images_dir = f"{CONTENT_ROOT}/aptos2019/train_images"
_n_images = len(os.listdir(_train_images_dir)) if os.path.isdir(_train_images_dir) else 0
if not os.path.exists(_train_csv) or _n_images < 1000:  # real dataset has 3,662 training images
    raise RuntimeError(
        "Still don't have the real dataset. If you saw a 401 error above from the Kaggle "
        "API: visit https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules "
        "and click 'I Understand and Accept', then re-run this cell -- or skip the API "
        "entirely and use Option B (manual upload) described above."
    )
print(f"Data OK: train.csv present, {_n_images} training images found.")
""")

code("""\
import pandas as pd

train_csv = pd.read_csv("/content/aptos2019/train.csv")  # columns: id_code, diagnosis
# Build the full path to each image file and attach a human-readable grade name,
# so downstream cells don't need to reconstruct either.
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
# Sample N_PER_GRADE rows from each grade independently and concatenate — this is what
# makes the sample "stratified": without it, a random draw from the raw dataset would be
# dominated by grade 0 (the most common class) and barely include grade 3/4.
sample_df = pd.concat([
    train_csv[train_csv["diagnosis"] == grade].sample(
        n=min(N_PER_GRADE, (train_csv["diagnosis"] == grade).sum()), random_state=RANDOM_SEED
    )
    for grade in sorted(train_csv["diagnosis"].unique())
]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle after stratifying
print(f"Sampled {len(sample_df)} images:")
sample_df[["id_code", "diagnosis", "grade_name"]]
""")

code("""\
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Lay the sample out in a grid, 5 images per row, as many rows as needed.
n = len(sample_df)
cols = 5 if n >= 5 else n
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.0))
axes = axes.ravel() if isinstance(axes, np.ndarray) else [axes]  # flatten to 1D regardless of grid shape

for ax in axes:
    ax.axis("off")  # hide axes for every slot, including any unused trailing ones

# Sanity check: a real fundus photo has real pixel variation. A blank, solid-color, or
# placeholder image (e.g. from a broken download step) has a pixel standard deviation
# near zero. This catches that immediately, with the exact filename, instead of letting
# it flow silently through every model and only showing up later as a suspicious result
# (e.g. every model predicting the same class regardless of ground truth).
flat_images = []
for _, row in sample_df.iterrows():
    img = Image.open(row["image_path"]).convert("RGB")
    flat_images.append((row["id_code"], np.asarray(img)))
low_variance = [(id_code, arr.std()) for id_code, arr in flat_images if arr.std() < 5.0]
if low_variance:
    raise RuntimeError(
        f"{len(low_variance)}/{len(flat_images)} sampled images look blank or nearly "
        f"solid-color (pixel std < 5.0), which real fundus photos never are: "
        f"{low_variance[:5]}\\n"
        "This means the images aren't real -- check that the Kaggle download in Section 3 "
        "actually completed (re-run it and check its output for errors) before continuing."
    )

# Each image's title shows its ground-truth grade, so this grid doubles as a visual
# ground-truth reference — compare it against model predictions later in the notebook.
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
import time
from google import genai
from google.genai import types

gemini_client = genai.Client(api_key=userdata.get("GEMINI_API_KEY"))

# Free-tier Gemini is capped at 5 requests/minute. Calling it in a plain serial loop
# with a flat time.sleep(1) between calls doesn't respect that limit either -- it just
# guarantees a 429 every 5th image, each one eating a 20-80s backoff (see the retry loop
# below). A sliding-window limiter lets Section 8 fire several requests concurrently
# (via a thread pool) while still never exceeding 5 calls in any 60s window, which is
# what actually gets close to the free tier's real throughput instead of serializing
# everything through avoidable rate-limit waits.
import threading

class RateLimiter:
    def __init__(self, max_calls, window_s):
        self.max_calls = max_calls
        self.window_s = window_s
        self.calls = []
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                self.calls = [t for t in self.calls if now - t < self.window_s]
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return
                sleep_for = self.window_s - (now - self.calls[0]) + 0.05
            time.sleep(sleep_for)

# max_calls=5 matches the free tier. If you're on a paid tier with a higher quota,
# raise this to match and Section 8's thread pool will speed up proportionally.
gemini_rate_limiter = RateLimiter(max_calls=5, window_s=60)

# The rubric text below is the actual ICDR grading criteria, not a paraphrase — giving
# Gemini the same definitions a human grader uses is what makes its answer comparable
# to the dataset's ground truth (and to MedGemma/RetiZero, which grade against the same scale).
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

# Gemini 3.x models "think" before answering by default (thinking_level defaults to
# "medium"), and thinking tokens count against the same output-token budget as the
# visible answer -- for a one-digit answer that's pure overhead, and it's also what
# was silently starving every prediction of real output (finish_reason=MAX_TOKENS,
# response.text raising instead of returning "0".."4"). thinking_level="low" keeps a
# fast sanity check without burning the whole budget on deliberation.
GEMINI_CONFIG = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_level="low"),
    max_output_tokens=512,
    # Fundus photographs depict a disease and are full of dark red/vascular detail --
    # Gemini's default safety thresholds routinely misfire on medical imagery like this
    # and silently return zero candidates instead of an answer. Grading known clinical
    # photographs from a public research dataset is exactly the case these thresholds
    # are too aggressive for, so they're relaxed here rather than left at the default.
    safety_settings=[
        types.SafetySetting(category=cat, threshold="BLOCK_ONLY_HIGH")
        for cat in [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        ]
    ],
)

def predict_gemini(image_path, max_retries=5):
    # Send the raw image bytes + prompt in one call; Gemini's vision models accept
    # image parts directly, no separate encoding step needed.
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Retrying with backoff is still here as a safety net for bursts that slip past the
    # limiter (clock skew, another process sharing the same API key) -- but with the
    # limiter in place this should rarely trigger.
    for attempt in range(max_retries):
        gemini_rate_limiter.acquire()
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    ICDR_PROMPT,
                ],
                config=GEMINI_CONFIG,
            )
            break
        except Exception as e:
            is_rate_limit = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            wait_s = 20 * (attempt + 1)  # 20s, 40s, 60s, 80s -- free-tier resets per minute
            print(f"  Gemini rate limit hit, waiting {wait_s}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_s)

    # response.text raises ValueError (not just returns empty) whenever a candidate has
    # no text part -- safety block, empty MAX_TOKENS cutoff, etc. Checking finish_reason
    # first turns that opaque crash into a specific, debuggable message instead of
    # letting the generic try/except in Section 8 swallow it as an unlabeled None.
    candidates = response.candidates or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates (prompt likely blocked)")
    finish_reason = str(candidates[0].finish_reason)
    if "SAFETY" in finish_reason or "PROHIBITED" in finish_reason:
        raise RuntimeError(f"Gemini blocked this image (finish_reason={finish_reason})")
    if not candidates[0].content or not candidates[0].content.parts:
        raise RuntimeError(f"Gemini returned no text (finish_reason={finish_reason})")

    # The prompt asks for a bare digit, but models occasionally add stray text anyway —
    # pulling the first digit out of the response is more robust than assuming int(text) works.
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

# AutoProcessor handles both image preprocessing and text tokenization for MedGemma.
# bfloat16 + device_map="auto" loads the 4B model in half precision, spread across
# whatever GPU memory Colab gives you — full fp32 wouldn't fit on a T4.
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
    # MedGemma expects a chat-style message list, same format as text-only chat models,
    # with the image and prompt as separate content items in one user turn.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": ICDR_PROMPT},
            ],
        }
    ]
    # apply_chat_template turns that message list into model-ready input tensors in one
    # call (formatting the prompt, tokenizing, and preparing the image together).
    inputs = medgemma_processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(medgemma_model.device, dtype=torch.bfloat16)

    # inference_mode disables gradient tracking (we're not training); max_new_tokens=8 is
    # plenty for a single-digit answer, and do_sample=False makes output deterministic.
    with torch.inference_mode():
        output = medgemma_model.generate(**inputs, max_new_tokens=8, do_sample=False)
    # Slice off the input tokens so only the newly generated text is decoded.
    decoded = medgemma_processor.decode(
        output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()
    digits = [c for c in decoded if c.isdigit()]
    return int(digits[0]) if digits else None
""")

# ---------------------------------------------------------------------------
md("""\
## 7. Model 3 — RetiZero (zero-shot retinal foundation model)

RetiZero isn't generative — it's CLIP-style: it embeds the image and 5 candidate ICDR
label strings, then takes the argmax of their cosine similarity. That's a mechanically
different task than "write a digit" and should be reported as such, not as a fair
three-way tie.

**Dependency note:** RetiZero's own `requirements.txt` pins an old torch/transformers
stack that would break MedGemma's modern one — we deliberately run it on the current
Colab stack instead (its ViT code uses standard torch APIs and loads fine).

**GPU memory note:** MedGemma alone uses several GB on a T4, and RetiZero needs its own
chunk too — loading both onto the GPU at the same time risks a CUDA out-of-memory error.
This cell only downloads RetiZero's weights; it doesn't load the model onto the GPU yet.
Section 8 runs Gemini and MedGemma first, frees MedGemma's GPU memory, *then* loads
RetiZero — so the two large models never compete for memory at the same time. (An earlier
version of this notebook tried moving RetiZero to the CPU instead to solve this, but that
introduced a device-mismatch bug — CPU model, GPU-loaded image tensors. Sequencing them
instead of splitting devices avoids that class of bug entirely.)
""")
code("""\
import os, sys

# Clone the upstream RetiZero repo (only once — reuse it if this cell is re-run).
if not os.path.isdir("/content/RetiZero"):
    !git clone -q {RETIZERO_REPO} /content/RetiZero
else:
    print("RetiZero repo already cloned")

# The pretrained weights aren't in the repo (too large for git) or on pip/HF — they're
# a Google Drive file, downloaded here by ID with gdown.
os.makedirs("/content/RetiZero/checkpoints", exist_ok=True)
if not os.path.exists(RETIZERO_WEIGHTS_PATH):
    !gdown {RETIZERO_WEIGHTS_GDRIVE_ID} -O {RETIZERO_WEIGHTS_PATH}
else:
    print("Weights already present")

# Sanity check: a large Google Drive file sometimes serves an HTML "can't scan this
# file for viruses" warning page instead of the actual file. gdown usually handles
# this automatically, but if it doesn't, torch.load() a few cells from now fails with
# a confusing pickle error -- this catches it immediately with a clear message instead.
_size_mb = os.path.getsize(RETIZERO_WEIGHTS_PATH) / 1e6 if os.path.exists(RETIZERO_WEIGHTS_PATH) else 0
if _size_mb < 10:
    with open(RETIZERO_WEIGHTS_PATH, "rb") as f:
        _head = f.read(200)
    raise RuntimeError(
        f"Downloaded weights file is only {_size_mb:.2f} MB -- this is almost certainly "
        f"not the real checkpoint (expect ~100s of MB). First bytes: {_head[:200]!r}\\n"
        "This usually means Google Drive served an HTML warning page instead of the file. "
        "Delete the file and retry, or download it manually from the RetiZero README's "
        "link and upload it to the path in RETIZERO_WEIGHTS_PATH above."
    )
print(f"Weights file OK: {_size_mb:.0f} MB")
""")

code("""\
os.chdir("/content/RetiZero")       # RetiZero's own code does relative imports, so run from its root
sys.path.insert(0, "/content/RetiZero")

import torch
from zeroshot import CLIPRModel     # RetiZero's own package (zeroshot/__init__.py re-exports this)

# Defined but not called yet -- Section 8 calls this only after MedGemma's GPU memory has
# been freed, so RetiZero always loads onto the GPU cleanly with no memory conflict and no
# CPU/GPU device split to introduce tensor-mismatch bugs.
def load_retizero():
    # from_checkpoint=False here because we load the weights ourselves right below via
    # load_state_dict, rather than having the constructor fetch them.
    model = CLIPRModel(
        vision_type="lora",
        from_checkpoint=False,          # weights are applied below via load_state_dict
        weights_path=RETIZERO_WEIGHTS_PATH,
        R=8,                            # LoRA rank used at pretraining time
    )
    state_dict = torch.load(RETIZERO_WEIGHTS_PATH, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(state_dict, strict=True)
    if torch.cuda.is_available():
        model.cuda()                    # keep model and image tensors on the same device throughout
    model.eval()                        # disables dropout/batchnorm training behavior for inference
    return model
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
    # forward() embeds the image and the 5 label strings, then returns their similarity
    # as (probability, logits) numpy arrays — argmax over probability gives the predicted grade.
    # retizero_model is set by Section 8, right after loading it post-MedGemma-cleanup.
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

**Two phases, not one loop.** Phase 1 runs Gemini and MedGemma over every image. Then
MedGemma's GPU memory is explicitly freed, and *only then* is RetiZero loaded and run in
Phase 2 — so MedGemma and RetiZero are never both resident on the GPU at once, which is
what a single combined loop risked (a CUDA out-of-memory error, since MedGemma alone uses
several GB on a T4). This also means RetiZero stays on the GPU the whole time, the same
device as its input images, avoiding the device-mismatch bugs that come from splitting a
model across CPU and GPU.

**Both phases checkpoint continuously** to `results/predictions.csv` and skip `id_code`s
already scored by every model in that phase. If the runtime crashes or restarts partway
through, just re-run from wherever it stopped — nothing already-scored is lost or redone.

**Phase 1 itself is two passes, for a reason.** Gemini calls are network-bound (waiting
on a response, not on your GPU) but MedGemma calls are GPU-bound — running them in the
same serial loop means every image pays for both delays back to back. Splitting them
lets Gemini's calls run several at once (bounded by the rate limiter defined in Section
5) while MedGemma runs after, at whatever speed the GPU allows, without also waiting on
network round-trips it doesn't need.
""")
code("""\
import gc
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Pre-flight check: confirm everything Phase 1 needs is actually loaded before starting.
# Without this, a model that failed to load a few cells back (or was skipped by running
# cells out of order) doesn't fail loudly here -- it fails silently on every single image
# inside the try/except below, and you only find out at the very end when that model's
# column is entirely empty.
_missing = [
    name for name, obj in [
        ("gemini_client", globals().get("gemini_client")),
        ("medgemma_model", globals().get("medgemma_model")),
        ("medgemma_processor", globals().get("medgemma_processor")),
    ] if obj is None
]
if _missing:
    raise RuntimeError(
        f"These models aren't loaded yet: {_missing}. Scroll up and re-run their "
        "setup cells (Sections 5-6) before running this cell -- otherwise every "
        "prediction from the missing model(s) will fail."
    )

os.makedirs(RESULTS_DIR, exist_ok=True)
predictions_path = f"{RESULTS_DIR}/predictions.csv"

# Every row keyed by id_code, not a plain list -- this is what makes "resume" mean
# "fill in only what's missing" instead of "append a second row for anything not
# already 100% done", which would silently duplicate partially-scored images.
if os.path.exists(predictions_path):
    _prior = pd.read_csv(predictions_path)
    results_by_id = {r["id_code"]: r for r in _prior.to_dict("records")}
    print(f"Resuming: {len(results_by_id)} images have at least a partial record.")
else:
    results_by_id = {}

def _get_or_init(id_code, image_path, diagnosis):
    if id_code not in results_by_id:
        results_by_id[id_code] = {
            "id_code": id_code,
            "image_path": image_path,
            "ground_truth": int(diagnosis),
        }
    return results_by_id[id_code]

def _checkpoint():
    pd.DataFrame(list(results_by_id.values())).to_csv(predictions_path, index=False)

# --- Phase 1a: Gemini, concurrently (network-bound -- a thread pool waits on many
# in-flight HTTP requests at once instead of one at a time; gemini_rate_limiter from
# Section 5 keeps the actual request rate under the API's per-minute cap regardless of
# how many workers are running). ---
todo = [
    row for _, row in sample_df.iterrows()
    if pd.isna(results_by_id.get(row["id_code"], {}).get("gemini_pred"))
]
print(f"=== Phase 1a: Gemini over {len(todo)} images ({len(sample_df) - len(todo)} already scored) ===")

def _score_gemini(row):
    try:
        return row["id_code"], predict_gemini(row["image_path"]), None
    except Exception as e:
        return row["id_code"], None, str(e)

with ThreadPoolExecutor(max_workers=5) as pool:
    futures = [pool.submit(_score_gemini, row) for _, row in pd.DataFrame(todo).iterrows()] if todo else []
    for n, fut in enumerate(as_completed(futures), start=1):
        id_code, pred, err = fut.result()
        row = sample_df.loc[sample_df["id_code"] == id_code].iloc[0]
        record = _get_or_init(id_code, row["image_path"], row["diagnosis"])
        record["gemini_pred"] = pred
        if err:
            record["gemini_error"] = err
        if n % 5 == 0 or n == len(futures):
            _checkpoint()
        print(f"gemini done {id_code} ({n}/{len(futures)})" + (f" -- {err}" if err else ""))
_checkpoint()

# --- Phase 1b: MedGemma, sequentially (GPU-bound -- concurrency wouldn't help here
# since a single GPU runs one forward pass at a time anyway; it would just add
# scheduling overhead). ---
todo = [
    row for _, row in sample_df.iterrows()
    if pd.isna(results_by_id.get(row["id_code"], {}).get("medgemma_pred"))
]
print(f"=== Phase 1b: MedGemma over {len(todo)} images ({len(sample_df) - len(todo)} already scored) ===")
for i, row in enumerate(todo):
    record = _get_or_init(row["id_code"], row["image_path"], row["diagnosis"])
    try:
        record["medgemma_pred"] = predict_medgemma(row["image_path"])
    except Exception as e:
        record["medgemma_pred"] = None
        record["medgemma_error"] = str(e)

    _checkpoint()
    print(f"medgemma done {row['id_code']} ({i + 1}/{len(todo)})")

    if i % 10 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# --- Free MedGemma's GPU memory before loading RetiZero ---
print("\\n=== Freeing MedGemma from GPU memory before loading RetiZero ===")
del medgemma_model, medgemma_processor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU memory now free: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

# --- Phase 2: load and run RetiZero now that the GPU has room for it ---
print("\\n=== Phase 2: loading RetiZero ===")
# This used to be an unguarded call: any failure here (a shape mismatch between the
# downloaded checkpoint and the model, a CUDA OOM, a partial/corrupt weights file)
# raised a raw traceback and skipped Phase 2 entirely -- with predictions.csv ending up
# with a gemini_pred/medgemma_pred filled in but retizero_pred entirely missing, and no
# indication of why. Catching it here gives an actionable message and still lets you
# keep the Gemini/MedGemma results already saved to disk instead of losing the whole run.
try:
    retizero_model = load_retizero()
    print("RetiZero loaded successfully")
except Exception as e:
    retizero_model = None
    print(
        f"RetiZero failed to load: {e}\\n"
        "Gemini + MedGemma predictions above are already saved to "
        f"{predictions_path} -- fix the error above (common causes: corrupt/partial "
        "download in Section 7, GPU out of memory, or a checkpoint whose keys don't "
        "match CLIPRModel) and re-run this cell; Phase 1 will skip already-scored images."
    )

if retizero_model is None:
    print("Skipping Phase 2 -- RetiZero didn't load (see error above). Re-run this cell after fixing it.")
    results_df = pd.DataFrame(results)
else:
    print(f"=== Phase 2: RetiZero over {len(sample_df)} images ===")
    results_df = pd.DataFrame(results)  # refresh with Phase 1's results before appending retizero_pred
    retizero_done_ids = set(results_df.loc[results_df["retizero_pred"].notna(), "id_code"]) if "retizero_pred" in results_df.columns else set()
    for i, (_, row) in enumerate(sample_df.iterrows()):
        if row["id_code"] in retizero_done_ids:
            continue
        idx = results_df.index[results_df["id_code"] == row["id_code"]][0]
        try:
            results_df.loc[idx, "retizero_pred"] = predict_retizero(row["image_path"])
        except Exception as e:
            results_df.loc[idx, "retizero_pred"] = None
            results_df.loc[idx, "retizero_error"] = str(e)

        results_df.to_csv(predictions_path, index=False)
        print(f"done {row['id_code']} (GT grade {row['diagnosis']})")

        if i % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
    # Percentile bootstrap 95% CI for accuracy: resample the predictions (with
    # replacement) n_boot times, compute accuracy each time, and take the 2.5th/97.5th
    # percentiles of that distribution as the interval. This is what makes the CI
    # honest about small-sample uncertainty rather than just reporting a point estimate.
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accs = [accuracy_score(y_true[idx], y_pred[idx])
            for idx in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n_boot))]
    lo, hi = np.quantile(accs, [0.025, 0.975])
    return round(float(lo), 4), round(float(hi), 4)

for model_col in MODELS:
    model_name = model_col.replace("_pred", "")
    # Drop rows where this model's prediction is missing (failed calls from the run
    # loop above) — each model is scored only on the images it actually predicted.
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
        # weights="quadratic" is what makes this ordinal-aware: a true grade 4 predicted
        # as grade 0 costs more than a true grade 4 predicted as grade 3.
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

# One confusion matrix per model, side by side, so grading errors (e.g. always confusing
# grade 2 and 3) are visible at a glance rather than buried in the summary metrics.
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
# Same numbers as metrics_summary.json, reshaped into a table (one row per model) and
# printed as Markdown so it can be pasted straight into reports/report_template.md.
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

# Bundle everything the write-up needs into one file so there's a single download instead
# of pulling each artifact out of the Colab file browser individually.
tar_path = "/content/results.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(RESULTS_DIR, arcname="results")
    tar.add("/content/sample_grid.png", arcname="sample_grid.png")

from google.colab import files
files.download(tar_path)  # triggers a browser download in Colab
print("Downloaded results.tar.gz -> extract into the repo root (creates results/)")
""")

md("""\
## 12. Notes and limitations (read before writing up results)

Quick recap for the write-up — see the README for the full version:

- **Ground truth is one grader's opinion, not an infallible reference** — published
  inter-ophthalmologist ICDR agreement sits at kappa 0.40-0.65. Frame conclusions accordingly.
- **Gemini is a generalist doing a specialist's task**; MedGemma and RetiZero were both
  medically pretrained. Say so rather than presenting three equivalent peers.
- **This notebook is structurally complete but not yet run end-to-end on a GPU** (no
  GPU/Kaggle access in the authoring environment) — the RetiZero wiring was checked
  line-by-line against its upstream source, but budget time for first-run surprises.
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
