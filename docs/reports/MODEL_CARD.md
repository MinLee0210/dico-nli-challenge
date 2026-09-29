---
license: mit
language:
  - en
  - es
  - eu
tags:
  - natural-language-inference
  - text-classification
  - cross-encoder
  - semeval-2027
  - dico-nli
  - multilingual
datasets:
  - ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI
pipeline_tag: text-classification
---

# DiCo-NLI cross-encoders (SemEval-2027 Task 2, development phase)

Checkpoints from our system for
[SemEval-2027 Task 2: Directional-Consistent Fine-Grained NLI (DiCo-NLI)](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI).
Given an ordered phrase pair `(text1, text2)`, the models predict
`EQUIVALENCE`, `FORWARD_ENTAILMENT` (text1 ⊨ text2), `BACKWARD_ENTAILMENT`,
or `NEGATIVE_OTHER`, in English, Spanish, Basque and mixed-language pairs.

These are **research checkpoints from the development phase**, not a
finished product. Scores below are on the official dev set, which was also
used for model selection, so they are optimistic.

## Model details

- **Architecture:** Hugging Face cross-encoder (`AutoModelForSequenceClassification`)
  with a 4-way head, wrapped in `PairClassifier` from the project code.
  `symmetric` checkpoints score both orders and average
  `f(a,b)` with the label-reversed `f(b,a)`, so they are exactly
  reversal-equivariant.
- **Backbones:** `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`,
  `joeddav/xlm-roberta-large-xnli`, `jhu-clsp/mmBERT-base`,
  `microsoft/deberta-v3-base`,
  `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (English only).
- **Decoding:** predictions should be decoded with the project's
  consistency-aware decoder (`src/decoding.py`), which pools the
  log-probabilities of a pair and its reversed twin (and other language views)
  in one canonical frame. This guarantees reversal consistency for every pair
  predicted as reversible.
- **Developed by:** Le Duc Minh. **License:** MIT for our fine-tuned weights;
  each backbone keeps its own license.

## Files

`checkpoints/<run>/best.pt` holds `{"model": state_dict, "step", "extra": {"cfg": ...}}`
(weights only). The saved `cfg` rebuilds the model without the training YAML.
`results/dev/logprobs/*.npz` holds dev log-probabilities per run, so ensembles
and decoding settings can be re-scored without a GPU. `configs/` holds every run
config.

| Run | Backbone | Training variant | Tracks |
|---|---|---|---|
| `e8_mdeberta_fine`, `e12_mdeberta_s{1,2,3}` | mDeBERTa-v3-base NLI | fine-grained validation, label smoothing 0.05 | all |
| `e7_xlmr_large`, `e12_xlmr_s{1,2}` | XLM-R-large XNLI | reversal-KL 0.5 | all |
| `e11_*_prior`, `e14_*_prior_*`, `e15_mdeberta_sym_prior`, `e17_xlmr_sym_prior` | mDeBERTa / XLM-R | **prior-aware**: reversible labels only (needs the structural prior at inference) | all |
| `e16_debl_t1`, `e16_debl_t1_prior` | DeBERTa-v3-large NLI | T1 only (clean / prior-aware) | T1 |
| `e3_*`, `e9_*`, `e13_*`, `e18_*`, `e19_*` | various | ablations (mmBERT, symmetric architecture, reverse-negative augmentation, NEG-weighted/focal loss, iSTS answers-students data) | all |
| `e1_deberta_base_t1` | DeBERTa-v3-base | baseline | T1 |

## How to use

```python
import torch
from huggingface_hub import hf_hub_download
from src.pipelines.predict import load_model   # from the project repository

path = hf_hub_download("LakoreAI/dico-nli-checkpoints",
                       "checkpoints/e8_mdeberta_fine/best.pt")
model, cfg, tokenizer = load_model(path, torch.device("cpu"))
enc = tokenizer(["two dogs play in the snow"], ["dogs play"], return_tensors="pt")
logits, _ = model(**enc)
print(logits.softmax(-1))  # EQ, FE, BE, NEG
```

For submissions, use the project CLI, which ensembles checkpoints and applies
consistency-aware decoding:

```bash
uv run python -m src.pipelines.predict --ckpt <best.pt> [<best.pt> ...] \
    --inputs <track files> --out_dir <dir> --decoding source
```

## Training data

- Official DiCo-NLI train split (all four tracks, 27,378 rows; the T1-only
  models use 3,042 English rows).
- `e19_*` runs also use 1,010 pairs converted from the iSTS 2016
  **answers-students** chunk alignments (EQUI→EQ, SPE1→FE, SPE2→BE,
  SIMI/REL/OPPO→NEG). The iSTS headlines and images subsets were **not** used,
  because they are the PhrasIS source of DiCo-NLI and contain its test pairs.
  None of the converted pairs occur in DiCo train/dev.

## Evaluation (dev set)

Official metrics: Weighted F1 / SoftCons / HardCons (%), averaged over the four
tracks (ALL-4), per-track source decoding.

| System | Prior | T1 EN | T2 ES | T3 EU | T4 Mixed | ALL-4 |
|---|:-:|---|---|---|---|---|
| mDeBERTa, fine-grained (`e8`) | no | 81.8 / 96.0 / 88.4 | 78.2 / 94.9 / 86.3 | 78.8 / 95.3 / 86.3 | 78.8 / 97.5 / 87.0 | 79.4 / 95.9 / 87.0 |
| XLM-R-large XNLI (`e7`) | no | 83.0 / 95.3 / 88.1 | 79.2 / 96.4 / 87.4 | 75.1 / 96.8 / 82.7 | 77.9 / 96.8 / 87.0 | 78.8 / 96.3 / 86.3 |
| **Clean-7 ensemble** | no | 83.7 / 95.3 / 89.9 | 80.6 / 96.0 / 88.1 | 79.4 / 95.7 / 87.0 | 79.4 / 95.7 / 87.0 | **80.8 / 95.7 / 88.0** |
| **Prior-14 ensemble** | yes | 94.2 / 100 / 93.1 | 92.4 / 100 / 91.0 | 91.3 / 100 / 89.5 | 92.5 / 100 / 91.0 | **92.6 / 100 / 91.2** |
| CodaBench dev #1 (29 Sep 2026) | – | 92.4 / 100 / 91.0 | 92.8 / 100 / 91.3 | 90.3 / 100 / 88.4 | 90.2 / 100 / 88.2 | 91.4 / 100 / 89.7 |

Cells are Weighted F1 / SoftCons / HardCons. *Clean-7* = `e8_mdeberta_fine`,
`e12_mdeberta_s{1,2,3}`, `e7_xlmr_large`, `e12_xlmr_s{1,2}`. *Prior-14* = clean-7
+ `e11_{mdeberta,xlmr}_prior`, `e14_mdeberta_prior_s{1,2}`, `e14_xlmr_prior_s1`,
`e15_mdeberta_sym_prior`, `e17_xlmr_sym_prior`. The two ensemble rows were
confirmed with the official scorer. Full per-model tables are in the technical
report (`report/dico_nli_report.pdf`).

The **structural prior** decodes a pair as reversible when its reversed twin is
present in the input file and as `NEGATIVE_OTHER` otherwise. It is exact on
train and dev, but it relies on how the dataset files were built, and its use
in official submissions has not been confirmed with the organizers. Results
without it are the model-only numbers.

## Limitations and risks

- Model and ensemble selection used the dev set (277 reversible pairs per
  track; HardCons standard error ≈2.6 points), so dev scores are optimistic.
- Without the structural prior, telling `NEGATIVE_OTHER` apart from the
  reversible labels is the main error source (Weighted F1 ≈80).
- The data are short phrases from image captions and news headlines; behaviour
  on longer or out-of-domain text is untested.
- Basque performance relies on multilingual transfer; a Basque-only encoder
  (JaunBERT) did worse in our tests.

## Citation

If you use these checkpoints or the code, please cite this project:

```bibtex
@software{le2026diconli,
  author  = {Le, Duc Minh},
  title   = {Consistency by Construction: Cross-Encoders with Reversal-Aware
             Decoding for {SemEval-2027} Task 2 ({DiCo-NLI})},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/MinLee0210/dico-nli-challenge},
  note    = {Technical report, development phase}
}
```

Please also cite the PhrasIS benchmark that DiCo-NLI is built on:

```bibtex
@article{lopezgazpio2024phrasis,
  title   = {PhrasIS: Phrase Inference and Similarity benchmark},
  author  = {Lopez-Gazpio, I. and others},
  journal = {Logic Journal of the IGPL},
  volume  = {32}, number = {6}, pages = {1088--1101}, year = {2024}
}
```
