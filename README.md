# Diabetic Retinopathy Grading: Gemini vs. MedGemma vs. RetiZero

A mini research project comparing three vision-language models on diabetic retinopathy (DR) severity grading using the APTOS 2019 dataset. This repository contains the notebook and reproducible pipeline used for evaluation, plus scripts to run the models in Colab.

## Quickstart

1. Clone the repo:

   git clone https://github.com/aayanahmed20/diabetic-retinopathy-vlm-comparison.git
   cd diabetic-retinopathy-vlm-comparison

2. Install dependencies (for Colab you can run the cells that install packages):

   pip install -r requirements.txt

3. Provide secrets (Colab):
   - GEMINI_API_KEY (Gemini hosted API)
   - HF_TOKEN (Hugging Face token for gated MedGemma weights)
   - (Optional) KAGGLE_API_TOKEN for automated APTOS downloads

4. Run the notebook in Colab (recommended) or modify the notebook for local runs.

## Models status

- Gemini: requires `GEMINI_API_KEY` and the hosted API; inference is implemented in the notebook but may return no valid digit if response parsing fails. See `notebooks/DR_VLM_Comparison.ipynb` for details.
- MedGemma: supported via Hugging Face `google/medgemma-4b-it`; requires HF_TOKEN. The notebook includes environment patches to make it run in Colab/GPU. Expect heavy memory usage.
- RetiZero: a CLIP-style model that requires cloning its repo and downloading weights (see notebook). Its interface differs (embedding similarity) and notebook includes patches; ensure weights are present.

## Notes

- I removed the contributor username `aayan` from README and CONTRIBUTORS files as requested. If you intended collaborator removal from repository settings as well, confirm and I will proceed per repo.

## License

MIT
