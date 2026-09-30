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
- **Backbones** (all ≤ ~600M parameters):
  `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`,
  `joeddav/xlm-roberta-large-xnli`, `jhu-clsp/mmBERT-base`,
  `BAAI/bge-reranker-v2-m3`, `Qwen/Qwen3-Reranker-0.6B` (read through a short
  prompt), and for English only `microsoft/deberta-v3-base` and
  `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.
- **Head variants:** `symmetric` checkpoints are exactly reversal-equivariant
  (see above). `entail2` checkpoints compose the four labels from the NLI
  entailment probability in each direction. **Prior-aware** checkpoints are
  trained on the three reversible labels only.
- **Decoding:** predictions should be decoded with the project's
  consistency-aware decoder (`src/decoding.py`), which pools the
  log-probabilities of a pair and its reversed twin (and other language views)
  in one canonical frame. This guarantees reversal consistency for every pair
  predicted as reversible.
- **Hierarchical composition:** the best system without the structural prior
  takes p(NEG) from ordinary 4-way models (the *gate*) and the choice among
  EQ / FE / BE from prior-aware models:
  `p(NEG) = g`, `p(c) = (1 − g) · q(c)`. See
  `scripts/training/hierarchical_eval.py` in the code repository.
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
| `e28_mdeberta_sym_prior_wordnet`, `e29/e30_mdeberta_sym_prior_s{1,2}`, `e31_xlmr_prior_wordnet` | mDeBERTa / XLM-R | prior-aware (symmetric, WordNet pairs, extra seeds) | all |
| `e25_mdeberta_wordnet`, `e32_mdeberta_wordnet_s1`, `e26_xlmr_wordnet`, `e27_mdeberta_prior_wordnet` | mDeBERTa / XLM-R | + WordNet lexical-relation pairs | all |
| `e24_bgerr`, `e33_bgerr_s1`, `e24_bgerr_prior` | bge-reranker-v2-m3 | clean / prior-aware | all |
| `e23_qwen3rr`, `e23_qwen3rr_prior` | Qwen3-Reranker-0.6B | clean / prior-aware (weak; kept for completeness) | all |
| `e20_mdeberta_entail2`, `e22_debl_t1_entail2` | mDeBERTa / DeBERTa-v3-large | entail2 head | all / T1 |
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
- WordNet runs (`e25`–`e28`, `e31`, `e32`) add 9,000 rows of lexical-relation
  pairs in English, Spanish and Basque, built from the Open Multilingual
  Wordnet 1.4 through the `wn` library: hyponym → hypernym = FE, synonyms = EQ,
  co-hyponyms = NEG, 30% cross-lingual (`scripts/data/wordnet_pairs.py`).

## Evaluation (dev set)

Official metrics: Weighted F1 / SoftCons / HardCons (%), averaged over the four
tracks (ALL-4), per-track source decoding.

| System | Prior | T1 EN | T2 ES | T3 EU | T4 Mixed | ALL-4 |
|---|:-:|---|---|---|---|---|
| mDeBERTa, fine-grained (`e8`) | no | 81.8 / 96.0 / 88.4 | 78.2 / 94.9 / 86.3 | 78.8 / 95.3 / 86.3 | 78.8 / 97.5 / 87.0 | 79.4 / 95.9 / 87.0 |
| XLM-R-large XNLI (`e7`) | no | 83.0 / 95.3 / 88.1 | 79.2 / 96.4 / 87.4 | 75.1 / 96.8 / 82.7 | 77.9 / 96.8 / 87.0 | 78.8 / 96.3 / 86.3 |
| Clean-7 ensemble (`clean7`) | no | 83.7 / 95.3 / 89.9 | 80.6 / 96.0 / 88.1 | 79.4 / 95.7 / 87.0 | 79.4 / 95.7 / 87.0 | 80.8 / 95.7 / 88.0 |
| **Hierarchical, fixed** (`hier_fixed`) | no | 83.6 / 96.0 / 90.3 | 81.4 / 97.1 / 89.9 | 79.8 / 96.0 / 88.1 | 81.0 / 98.2 / 89.9 | **81.4 / 96.8 / 89.5** |
| Hierarchical, selected per track (`sel_hier`) † | no | 85.6 / 98.6 / 92.8 | 83.5 / 98.9 / 92.4 | 80.4 / 98.6 / 89.2 | 85.3 / 98.6 / 92.4 | 83.7 / 98.6 / 91.7 |
| Prior-14 ensemble (`prior14`) | yes | 94.2 / 100 / 93.1 | 92.4 / 100 / 91.0 | 91.3 / 100 / 89.5 | 92.5 / 100 / 91.0 | 92.6 / 100 / 91.2 |
| Selected per track (`sel_prior_v2`) † | yes | 96.1 / 100 / 95.3 | 93.6 / 100 / 92.4 | 92.8 / 100 / 91.3 | 94.6 / 100 / 93.5 | **94.3 / 100 / 93.1** |
| CodaBench dev #1 (29 Sep 2026) | – | 92.4 / 100 / 91.0 | 92.8 / 100 / 91.3 | 90.3 / 100 / 88.4 | 90.2 / 100 / 88.2 | 91.4 / 100 / 89.7 |

Cells are Weighted F1 / SoftCons / HardCons; every ensemble row was confirmed
with the official scorer.
- *Clean-7* = `e8_mdeberta_fine`, `e12_mdeberta_s{1,2,3}`, `e7_xlmr_large`,
  `e12_xlmr_s{1,2}`.
- *Hierarchical, fixed* = gate clean-7, direction ensemble of 11 prior-aware
  models (`e11_*_prior`, `e14_*`, `e15`, `e17`, `e28`–`e31`).
- *Prior-14* = clean-7 + `e11_{mdeberta,xlmr}_prior`,
  `e14_mdeberta_prior_s{1,2}`, `e14_xlmr_prior_s1`, `e15_mdeberta_sym_prior`,
  `e17_xlmr_sym_prior`.
- † Members chosen by greedy selection on the dev set, one ensemble per track.
  The member lists are in `results/final/<run>/selection.json`
  (`sel_prior_v2` is stored as `results/final/v2_prior/`). These scores
  are optimistic; the fixed ensembles are the better guide to unseen data.

Submission ZIPs for each run id are in
`results/submissions/<run_id>/submission.zip`. Full per-model tables are in
the technical report (`report/dico_nli_report.pdf`).

The **structural prior** decodes a pair as reversible when its reversed twin is
present in the input file and as `NEGATIVE_OTHER` otherwise. It is exact on
train and dev, but it relies on how the dataset files were built, and its use
in official submissions has not been confirmed with the organizers. Results
without it are the model-only numbers.

## Limitations and risks

- Model and ensemble selection used the dev set (277 reversible pairs per
  track; HardCons standard error ≈2.6 points), so dev scores are optimistic.
  This applies most to the per-track selected systems (†), whose member lists
  change when candidates are added.
- Without the structural prior, telling `NEGATIVE_OTHER` apart from the
  reversible labels is the main error source (Weighted F1 ≈80).
- The data are short phrases from image captions and news headlines; behaviour
  on longer or out-of-domain text is untested.
- Basque performance relies on multilingual transfer; a Basque-only encoder
  (JaunBERT) did worse in our tests.
- The rerankers (bge-reranker-v2-m3, Qwen3-Reranker-0.6B) and the WordNet data
  did not beat the NLI checkpoints on their own; they are published as
  ablations.

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
