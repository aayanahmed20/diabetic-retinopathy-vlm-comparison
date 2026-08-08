# Comparative Evaluation of Gemini, MedGemma, and RetiZero for Diabetic Retinopathy Severity Grading

> **Status:** filled in from a real, verified 18-image run (`results/` in this repo at the
> commit this report was written against). Ground truth for every image was independently
> checked against APTOS's own `train.csv` before these numbers were written. Only
> MedGemma produced valid predictions in this run — see Section VI for why Gemini and
> RetiZero didn't, and Section VII for the next step to get all three.

## Abstract

This report presents a pilot evaluation of three vision-language model (VLM) paradigms —
Gemini (general-purpose commercial), MedGemma 4B IT (medically-adapted open-weight), and
RetiZero (zero-shot retinal foundation model) — on 5-class ICDR diabetic retinopathy
severity grading, scored against ground-truth labels from the APTOS 2019 dataset. Of the
three systems, only MedGemma produced usable predictions in this run: Gemini failed on
every image with a model-availability API error, and RetiZero failed on every image with
a tensor-device mismatch (both traced to environment configuration issues, not the study
design — see Limitations). On an 18-image sample, MedGemma achieved 61.1% exact-grade
accuracy (95% CI: 38.9%-83.3%) and a quadratic-weighted Cohen's kappa of 0.801, with a
mean absolute error of 0.556 grade steps. MedGemma correctly identified every Grade 0 (no
DR) and the single Grade 4 (proliferative DR) case, but misclassified all seven
intermediate-grade (1-3) cases, always toward a *higher* predicted severity, never lower.
This pilot demonstrates that a medically-pretrained open VLM can grade DR with fair
agreement at this sample size, but the single-digit sample sizes for grades 2-4 and the
absence of a working comparison model mean this run supports a description of MedGemma's
behavior, not a three-way comparison or a statistically defensible performance claim.

## I. Introduction

Diabetic retinopathy (DR) is retinal vascular damage caused by prolonged hyperglycemia and
is a leading cause of vision loss among working-age adults worldwide, affecting an
estimated one-third of people with diabetes to some degree [1]. Left unscreened, DR
progresses silently from mild non-proliferative disease to proliferative disease,
vitreous hemorrhage, and neovascular glaucoma [2]. Regular fundus screening with grading
against the International Clinical Diabetic Retinopathy (ICDR) severity scale - 0 (no
DR) through 4 (proliferative DR) - is the standard of care, but grading is
labor-intensive and subject to inter-grader variability (kappa 0.40-0.65 in prior work)
[3], motivating automated grading support.

Vision-language models (VLMs) offer a new grading paradigm relative to classic CNN
classifiers: rather than training a bespoke classifier per dataset, a general or
domain-adapted VLM can be prompted directly. This raises an open question: how do a
general-purpose VLM (Gemini), a medically-pretrained open VLM (MedGemma), and a
retina-specialist zero-shot foundation model (RetiZero) compare on the same grading task,
scored against the same ground truth? This pilot run answers a narrower version of that
question for MedGemma alone; Gemini and RetiZero's environment failures (Section VI) mean
the three-way comparison remains open for a follow-up run.

## II. Related Work

Automated DR grading has historically relied on supervised CNNs and vision transformers
trained on public datasets such as EyePACS and APTOS. Google's MedGemma combines a Gemma 3
text backbone with MedSigLIP, a vision encoder pretrained on tens of millions of medical
image-text pairs; MedGemma's own technical report cites 0.857 AUC on 5-class EyePACS DR
grading via linear probing [4] — a different task setup (AUC on a probed classifier) than
this report's exact-grade accuracy on prompted generation, so the two numbers are not
directly comparable, only both evidence of the same underlying pretraining. RetiZero,
introduced by Wang et al., is a contrastive vision-language model pretrained on 341,896
fundus image-text pairs spanning over 400 disease entities; its original evaluation
reported top-3 zero-shot diagnostic accuracy exceeding the average of 19 board-certified
ophthalmologists across 15 fundus diseases [5] — that evaluation measured multi-disease
differential diagnosis, not 5-class ICDR severity grading specifically, which is the gap
this comparison (once RetiZero's environment issue is resolved) is intended to fill.

## III. Methods

### A. Dataset

APTOS 2019 Blindness Detection (Kaggle), 3,662 fundus photographs graded 0-4 by trained
graders, collected by Aravind Eye Hospital. Class distribution: 1,805 / 370 / 999 / 193 /
295 for grades 0-4 respectively [6].

### B. Sample

The notebook's default methodology draws a stratified random sample of `N_PER_GRADE = 25`
images per ICDR grade (0-4) with a fixed seed (`RANDOM_SEED = 42`), targeting 125 images
total. **This run used a different, non-stratified 18-image sample instead** (10 grade 0,
4 grade 1, 2 grade 2, 1 grade 3, 1 grade 4), assembled individually rather than through
the stratified sampling cell, after repeated environment issues (Section VI) made a full
Colab session difficult to sustain. Every image's ground-truth label was independently
verified against the source APTOS `train.csv` before scoring. The class imbalance in this
sample (most images are grade 0, only one each for grades 3 and 4) is a direct limitation
on what this run's per-class numbers can support — see Section VI.

![The 18 images used in this run, labeled with their verified ground-truth grade](figures/sample_grid.png)

### C. Models and inference configuration

| Model | Access | Task formulation |
|---|---|---|
| Gemini (`gemini-3.5-flash`, per `GEMINI_MODEL` in the notebook - update here if you change it before running) | Hosted API | Prompted generative digit output against the written ICDR rubric |
| MedGemma (`google/medgemma-4b-it`) | Local inference, bf16, single GPU | Prompted generative digit output, identical rubric prompt |
| RetiZero | Local inference | Zero-shot CLIP-style image-text similarity against the 5 ICDR grade-name labels |

All three models received the same images. Gemini and MedGemma received an identical
prompt encoding the ICDR rubric (Appendix). **RetiZero is architecturally different**: it
is a CLIP-style contrastive model (RETFound/LoRA image encoder + Bio_ClinicalBERT text
encoder) pretrained on 341,896 fundus image-text pairs [5]. It does not generate tokens;
it embeds the image and each candidate label ("no diabetic retinopathy", "mild
non-proliferative diabetic retinopathy", ..., "proliferative diabetic retinopathy", wrapped
in its pretraining caption template "a fundus photograph of <label>") and returns the
cosine-similarity softmax over the 5 labels. Its predictions are therefore not directly
comparable in mechanism even when scored on the same metric - see Limitations.

### D. Metrics

- **Accuracy** - exact grade match, with a percentile bootstrap 95% confidence interval
  (1,000 resamples, seed 42) because the sample is small
- **Quadratic-weighted Cohen's kappa** - the standard DR-grading metric (also the scoring
  metric of the original Kaggle competition), penalizing distant misclassifications more
  than adjacent ones and correcting for chance agreement
- **Mean absolute error** in grade steps
- **Per-class precision/recall** and confusion matrices

## IV. Results

| Model | N scored | Accuracy (95% CI) | Quadratic-weighted kappa | MAE (grades) |
|---|---|---|---|---|
| Gemini | 0/18 | no valid predictions | — | — |
| MedGemma | 18/18 | 0.611 (0.389, 0.833) | 0.801 | 0.556 |
| RetiZero | 0/18 | no valid predictions | — | — |

Gemini and RetiZero produced zero valid predictions across all 18 images, each failing
with the same error on every single image (not an intermittent or partial failure) — see
Section VI for the specific technical causes. Only MedGemma's results are reported below.

MedGemma's confusion matrix (rows = true grade, columns = predicted grade):

![Confusion matrices for all three models (only MedGemma has data; Gemini and RetiZero show zero predictions)](figures/confusion_matrices.png)

| True \ Pred | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **0** | 10 | 0 | 0 | 0 | 0 |
| **1** | 0 | 0 | 3 | 1 | 0 |
| **2** | 0 | 0 | 0 | 0 | 2 |
| **3** | 0 | 0 | 0 | 0 | 1 |
| **4** | 0 | 0 | 0 | 0 | 1 |

MedGemma correctly identified all 10 Grade 0 cases and the single Grade 4 case (perfect
recall at both ends of the severity scale). Every one of the seven intermediate-grade
cases (grades 1-3) was misclassified, and — notably — **always toward a higher predicted
grade, never a lower one**: three Grade 1 cases were called Grade 2, one Grade 1 case was
called Grade 3, both Grade 2 cases were called Grade 4, and the one Grade 3 case was
called Grade 4. This one-directional error pattern is why the quadratic-weighted kappa
(0.801) is well above the raw accuracy (0.611): most errors, while wrong, are consistent
over-calls rather than scattered in both directions, which is a materially different
failure mode than randomly-distributed misclassification. The 95% CI on accuracy is wide
(38.9% to 83.3%) — expected at N=18, and a direct consequence of the small sample rather
than an artifact of the bootstrap procedure.

## V. Discussion

The single clearest pattern in this run is directional: MedGemma never under-called
severity in this sample — it was either exactly right or too cautious in the "high"
direction, never too lenient. For a screening-support tool, over-calling severity is the
safer failure mode (a false positive triggers a specialist referral; a false negative
delays one), so if this pattern holds at larger N it would be a clinically favorable bias,
though seven misclassified images is nowhere near enough to establish that as a real
tendency rather than a coincidence of which seven images happened to be sampled.

The perfect recall at both extremes (Grade 0 and Grade 4) alongside zero recall on every
intermediate grade is consistent with a plausible, testable hypothesis: MedGemma may be
more reliable at distinguishing "clearly healthy" from "clearly severe" retinas than at
resolving the finer distinctions between adjacent intermediate grades (which is also where
published inter-ophthalmologist agreement is weakest [3]). This run's sample has only one
or two images per intermediate grade, which is far too few to separate "MedGemma
genuinely struggles with grades 1-3" from "these seven specific images happened to be
ambiguous." Distinguishing those two explanations is exactly what the full
`N_PER_GRADE = 25` stratified run (Section III.B) is designed to do, and is the natural
next step.

No comparison against Gemini or RetiZero is possible from this run, since neither
produced predictions (Section VI) — this report describes MedGemma's behavior on 18 real,
verified images, not a three-model comparison.

## VI. Limitations

- **Sample size and composition.** N=18, non-stratified and heavily skewed toward Grade 0
  (10/18 images). The bootstrap 95% CI on accuracy (38.9%-83.3%) directly reflects how
  little this sample size constrains the true value. Grades 2-4 have only 1-2 images each,
  which is not enough to estimate per-class recall reliably — a single misclassified image
  moves that grade's recall from 100% to 0%, which is exactly what happened here.
- **Only one of three models produced results.** Gemini failed on 18/18 images with
  `404 NOT_FOUND: models/gemini-1.5-flash is not found for API version v1beta` — this
  points to the session using a different model string and an older/legacy API surface
  than this repository's notebook actually specifies (`gemini-3.5-flash` via the current
  `google-genai` client; see `build_notebook.py`), which happened after ad hoc live edits
  made directly in the Colab session to work around earlier errors, not a problem with the
  study design or this repository's source. RetiZero failed on 18/18 images with
  `Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the
  same` — a device-mismatch bug introduced when RetiZero was moved to CPU (to avoid a GPU
  out-of-memory conflict with MedGemma, which alone uses several GB on a T4) without
  updating every tensor in its forward pass to match. Both are fixable environment issues,
  not findings about either model's grading ability, and both should be re-run from this
  repository's actual source rather than a session with accumulated ad hoc patches.
- **Ground truth is itself one grader's judgment**, not an infallible reference -
  published inter-ophthalmologist ICDR agreement is kappa 0.40-0.65 [3]. (Every ground
  truth label used in this run was independently checked against the source APTOS
  `train.csv` before scoring, so mislabeling in this report is not a concern — only the
  inherent subjectivity of the original clinical grading is.)
- **Task-formulation asymmetry**: RetiZero's zero-shot similarity scoring and
  Gemini/MedGemma's generative digit output are mechanistically different approaches to
  the same label space, which may advantage or disadvantage either depending on how
  cleanly the ICDR grade names map to RetiZero's training-time disease vocabulary. This
  remains a limitation for the eventual three-way comparison once RetiZero is re-run
  successfully.
- **Gemini has no medical-domain pretraining** disclosed for this task, unlike MedGemma
  and RetiZero - the eventual comparison tests general multimodal reasoning against
  domain-specialized models, not three equivalently-trained systems.
- **Single run, single sample.** No repeated sampling or multiple seeds; the directional
  over-call pattern in Section IV is a single observation, not a replicated finding.
- **Not validated for clinical use.** None of these models is a diagnostic device; this is
  a benchmarking exercise, not a clinical validation.

## VII. Conclusion

This pilot run establishes that MedGemma 4B IT can grade real APTOS fundus photographs
with fair agreement (quadratic-weighted kappa = 0.801) on a small, non-stratified 18-image
sample, with a notable and clinically favorable directional bias — every misclassification
over-called severity, none under-called it — alongside perfect recall at both severity
extremes and zero recall on every intermediate grade. These results describe MedGemma's
behavior on this specific sample; they are not evidence about Gemini or RetiZero, both of
which failed to produce any predictions due to environment configuration issues unrelated
to the study design (Section VI), and they are not a statistically powered claim at N=18.
The direct next step is the notebook's intended `N_PER_GRADE = 25` stratified run (125
images, roughly 7x this sample) with Gemini and RetiZero's environment issues resolved
from this repository's actual source, which would both narrow the confidence intervals
substantially and make the three-way comparison this study set out to answer possible.

## References

[1] Yau, J. W. Y. et al., "Global Prevalence and Major Risk Factors of Diabetic
    Retinopathy," *Diabetes Care*, 35(3), 556-564, 2012.
[2] American Academy of Ophthalmology, "Diabetic Retinopathy Preferred Practice Pattern,"
    2019/2020 update.
[3] Krause, J. et al., "Grader variability and the importance of reference standards for
    evaluating machine learning models for diabetic retinopathy," *Ophthalmology*,
    125(8), 1264-1272, 2018.
[4] Google, "MedGemma Model Card," Health AI Developer Foundations, 2025.
[5] Wang, M. et al., "Enhancing diagnostic accuracy in rare and common fundus diseases
    with a knowledge-rich vision-language model," *Nature Communications*, 16, 5528,
    2025.
[6] APTOS 2019 Blindness Detection, Asia Pacific Tele-Ophthalmology Society, Kaggle,
    2019.

*[Verify all DOIs and full bibliographic entries against the published versions before
submission; these are the real source works the background section draws on.]*

## Appendix: Exact prompt used

```
You are assisting with diabetic retinopathy severity grading on a color fundus
photograph, using the International Clinical Diabetic Retinopathy (ICDR) severity scale.

Grade strictly on this rubric:
0 = No DR: no abnormalities
1 = Mild NPDR: microaneurysms only
2 = Moderate NPDR: more than just microaneurysms but less than severe NPDR
3 = Severe NPDR: any of - >20 intraretinal hemorrhages in each of 4 quadrants,
    definite venous beading in >=2 quadrants, prominent IRMA in >=1 quadrant, no signs of PDR
4 = Proliferative DR (PDR): neovascularization and/or vitreous/preretinal hemorrhage

Respond with ONLY a single digit 0-4. No words, no punctuation, no explanation.
```

(Copied verbatim from `ICDR_PROMPT` in the notebook. If you edit the prompt before running, update this block to match - it must stay in sync for the report to be reproducible.)
