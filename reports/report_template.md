# Comparative Evaluation of Gemini, MedGemma, and RetiZero for Diabetic Retinopathy Severity Grading

> **Before submitting anywhere:** every `[FILL IN]` below must come from an actual run of
> `notebooks/DR_VLM_Comparison.ipynb` against `results/metrics_summary.json`,
> `results/summary_table.csv`, and `results/predictions.csv`. Do not estimate or
> approximate these numbers - an IEEE reviewer (or anyone re-running your code) will notice
> if reported numbers don't match the artifact.

## Abstract

*[FILL IN - 150-250 words. State the task (5-class ICDR severity grading on fundus
photographs), the three systems compared, the dataset, headline accuracy/kappa numbers,
and the one-sentence takeaway.]*

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
scored against the same ground truth?

## II. Related Work

*[FILL IN - briefly cite prior DR-grading CNN/ViT literature and the original MedGemma /
RetiZero papers. MedGemma reports 0.857 AUC on 5-class EyePACS DR grading [4]; RetiZero
reports top-3 zero-shot performance exceeding the average of 19 ophthalmologists across
15 fundus diseases in its original evaluation, though not specifically on DR grading [5] -
note that distinction rather than conflating the two claims.]*

## III. Methods

### A. Dataset

APTOS 2019 Blindness Detection (Kaggle), 3,662 fundus photographs graded 0-4 by trained
graders, collected by Aravind Eye Hospital. Class distribution: 1,805 / 370 / 999 / 193 /
295 for grades 0-4 respectively [6].

### B. Sample

*[FILL IN - actual N and per-grade counts used, from `sample_df` in the notebook. State
the random seed for reproducibility. Stratified sampling across grades 0-4, one image per
draw seeded by `RANDOM_SEED = 42`.]*

### C. Models and inference configuration

| Model | Access | Task formulation |
|---|---|---|
| Gemini (`[FILL IN exact model string, e.g. gemini-2.5-flash]`) | Hosted API | Prompted generative digit output against the written ICDR rubric |
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

*[FILL IN table from `results/summary_table.csv`]*

| Model | N scored | Accuracy (95% CI) | Quadratic-weighted kappa | MAE (grades) |
|---|---|---|---|---|
| Gemini | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| MedGemma | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| RetiZero | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |

*[FILL IN - insert `results/confusion_matrices.png`. Describe per-class error patterns:
e.g. is confusion concentrated between adjacent grades (1<->2), or does a model
systematically miss grade 4 / over-call grade 0? How wide are the bootstrap CIs - do any
two models' intervals overlap (i.e. no significant difference)?]*

## V. Discussion

*[FILL IN once real numbers exist. Suggested angles:]*
- Does medical pretraining (MedGemma, RetiZero) actually outperform general Gemini on
  this task, or is the gap smaller than expected?
- Where does each model fail - adjacent-grade confusion (clinically minor) vs.
  wildly-off predictions (clinically serious, e.g. calling grade 4 as grade 0)?
- Quadratic-weighted kappa vs. raw accuracy: do they tell the same story, or does one
  model look better on accuracy but worse on kappa (indicating its errors are more severe
  when it is wrong)?
- Do the bootstrap CIs overlap between models? At the current sample size, "model A beats
  model B" is usually not a defensible claim.

## VI. Limitations

- **Sample size.** *[FILL IN actual N]* images is [adequate/inadequate - state which,
  with justification] for statistically robust per-class estimates; the bootstrap CIs in
  Table IV quantify how wide the uncertainty is at this N.
- **Ground truth is itself one grader's judgment**, not an infallible reference -
  published inter-ophthalmologist ICDR agreement is kappa 0.40-0.65 [3].
- **Task-formulation asymmetry**: RetiZero's zero-shot similarity scoring and
  Gemini/MedGemma's generative digit output are mechanistically different approaches to
  the same label space, which may advantage or disadvantage either depending on how
  cleanly the ICDR grade names map to RetiZero's training-time disease vocabulary.
- **Gemini has no medical-domain pretraining** disclosed for this task, unlike MedGemma
  and RetiZero - the comparison tests general multimodal reasoning against
  domain-specialized models, not three equivalently-trained systems.
- **Single random seed.** *[FILL IN - did you re-run with multiple seeds/samples? If not,
  say so as a limitation.]*
- **Not validated for clinical use.** None of these models is a diagnostic device; this is
  a benchmarking exercise, not a clinical validation.

## VII. Conclusion

*[FILL IN]*

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
[paste ICDR_PROMPT from the notebook here, verbatim, for reproducibility]
```
