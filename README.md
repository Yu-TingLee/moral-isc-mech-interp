# Explaining Intrinsic Moral Self-Correction with Mechanistic Interpretability

Official implementation for the paper: **Explaining Intrinsic Moral Self-Correction with Mechanistic Interpretability**.

## Abstract
**TL;DR:** We show that intrinsic moral self-correction functions by steering hidden representations along interpretable
latent directions.

Intrinsic moral self-correction refers to the phenomenon where a language model refines its ethical judgments or aligns its outputs purely through prompting. While effective across diverse tasks, its mechanism remains unclear. We hypothesize intrinsic moral self-correction functions by steering hidden representations along interpretable latent directions. Evaluating six LLMs across four morality-related tasks, we demonstrate that the representation shifts induced by self-correction prompts align with contrastive steering vectors. This alignment transfers even when the steering vectors are constructed from a disjoint corpus. Notably, when applied via activation addition, these prompt-induced shifts can alter model behavior more effectively than the self-correction prompts and the steering vectors. Our findings suggest representation steering is the mechanistic driver of intrinsic moral self-correction.

## Usage
**Warning:** Some data, prompts, and model outputs may contain toxic or offensive language.

To run the experiments, this repository uses the following datasets:

| Dataset | Source |
|---|---|
| RealToxicityPrompts | Included in the repository |
| Moral Stories, ETHICS | Fetched automatically from Hugging Face |
| Jigsaw 2018 | Manual: Download the [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) data and put `train.csv` in `data/jigsaw/`. |

Then, run the experiments with:
```sh
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN="<your token>"     # required for gated models
bash run.sh
```