# DiCo-NLI Challenge

Our system for [SemEval-2027 Task 2: Directional-Consistent Fine-Grained NLI
(DiCo-NLI)](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI)
([CodaBench](https://www.codabench.org/competitions/18038/)).

Given an ordered phrase pair `(text1, text2)`, predict `EQUIVALENCE`,
`FORWARD_ENTAILMENT`, `BACKWARD_ENTAILMENT`, or `NEGATIVE_OTHER`. Every
reversible pair also appears reversed, and systems are scored on
**Weighted F1**, **SoftCons** (the two directions agree under
`Rev: FE↔BE, EQ↔EQ`), and **HardCons** (both directions correct). Four tracks:
English, Spanish, Basque, and mixed-language.

The repo is a config-driven fine-tuning pipeline for Hugging Face
cross-encoders, with **consistency-aware training** (twin-aware batches plus a
reversal-KL loss) and **consistency-aware decoding** (pooling the log-probs of
twins, or of every language view of a source pair). Research notes are in
[`docs/RESEARCH.md`](docs/RESEARCH.md), and next steps are in
[`docs/TODO.md`](docs/TODO.md).

## Contents

- [Quickstart](#quickstart)
- [Data](#data)
- [Training](#training)
- [Decoding modes](#decoding-modes)
- [Evaluation, prediction, submission](#evaluation-prediction-submission)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Results, report, model card](#results-report-model-card)
- [Citation](#citation)
- [License](#license)

## Quickstart

Requires Python ≥ 3.12 and [`uv`](https://docs.astral.sh/uv/). Fine-tuning is
meant for a GPU training machine. Tests and the smoke run are CPU-only and
work offline.

```bash
uv sync                # core: torch, transformers>=5, sentencepiece
uv sync --extra rich   # optional: pretty training summary
uv sync --extra wandb  # optional: Weights & Biases logging
uv run pytest          # test suite (tiny random BERT, no downloads)

# end-to-end on synthetic data with a tiny offline backbone
uv run python scripts/training/smoke_test.py

# fetch the official data + scorer into data/raw/dico (gitignored)
uv run python scripts/data/fetch_data.py
```

Platform-aware torch builds resolve from `pyproject.toml`: Linux + NVIDIA uses
the CUDA wheels, and macOS resolves CPU/MPS wheels.

## Data

`scripts/data/fetch_data.py` clones the task repo into `data/raw/dico`. The
official CSVs are used as-is:

```
data/raw/dico/final_data/{train,dev}/
    dico_nli_<split>_track<N>_participant_labeled.csv   # instance_id,pair_id,text1_lang,text2_lang,text1,text2,label
    dico_nli_<split>_track<N>_submission_template.csv   # instance_id,label
    dico_nli_<split>_track<N>_reference.csv             # + reverse_pair_id (official scorer)
```

| Split | Track 1 EN | Track 2 ES | Track 3 EU | Track 4 mixed |
|---|--:|--:|--:|--:|
| Train | 3042 | 3042 | 3042 | 18252 |
| Dev | 660 | 660 | 660 | 3960 |

Structural facts the code relies on. You can check them with
`scripts/data/inspect_data.py`:

- An instance id is `<pair_id>__<l1>-<l2>__<original|flipped>`. The reversed
  twin swaps the languages and the direction tag. This matches
  `reverse_pair_id` on every train/dev reference file.
- Every reversible item has its twin in the same file. `NEGATIVE_OTHER` items
  are singletons, with no twin.
- A `pair_id` spans all tracks. It has 2 rows in each monolingual track and 12
  in Track 4 (6 language combinations × 2 directions), all in one canonical
  orientation.

## Training

```bash
# Track 1 baseline (DeBERTa-v3-base)
uv run python scripts/training/train.py --config configs/train.yaml

# one multilingual model for all tracks (source-pair decoding)
uv run python scripts/training/train.py --config configs/multilingual_mmbert.yaml

# overrides win over the YAML
uv run python scripts/training/train.py --config configs/train.yaml \
    --consistency_weight 1.0 --lr 3e-5 --decoding source
```

| Config | Backbone | Tracks |
|---|---|---|
| `train.yaml` | DeBERTa-v3-base (pilot reference) | T1 |
| `track1_deberta_v3_large_nli.yaml` | DeBERTa-v3-large, MNLI/FEVER/ANLI/Ling/WANLI | T1 |
| `xlmr_large_xnli_multilingual.yaml` | XLM-R-large XNLI | all |
| `multilingual_mmbert.yaml` | mmBERT-base (2025, 1,833 langs) | all |
| `multilingual_mrbert.yaml` | MrBERT (BSC, Dec 2025) | all |
| `multilingual_mdeberta.yaml` | mDeBERTa-v3-base multilingual NLI | all |
| `track3_jaunbert.yaml` | JaunBERT (HiTZ, Jul 2026, Basque) | T3 |
| `runs/e20_mdeberta_entail2.yaml` | mDeBERTa NLI, `entail2` head | all |
| `runs/e21_qwen35_9b_qlora.yaml` | Qwen3.5-9B, 4-bit QLoRA (`uv sync --extra llm`) | all |

**Data augmentation** (train only; `augment:` in the YAML or `--augment`):
`reverse_negatives` adds the reversed copy of every NEG pair, and `transitive`
adds entailments implied by chaining gold labels over shared phrases
(a ⊨ b, b ⊨ c ⇒ a ⊨ c). On Track 1 this grows train from 3,042 to 4,198 rows.
Training on all tracks is itself a 9-view augmentation of every source pair.
See `docs/RESEARCH.md` §3b for the wider plan.

A run writes `checkpoints/<run_name>/best.pt` (selected on `dico_mean`, the
mean of the three official scores) and a `train_log.json`. Checkpoints store
the `ModelConfig` (backbone id, max length), so the model can be rebuilt
without the YAML. Loading one re-downloads, or reads from the HF cache, the
backbone named there.

## Decoding modes

`src/decoding.py` maps each instance into its source pair's canonical frame
(`original`; `flipped` rows get Rev applied), averages log-probs within a
group, picks one label, and maps it back:

| `decoding` | group | effect |
|---|---|---|
| `independent` | each instance | plain argmax (baseline) |
| `twin` | instance + reversed twin | SoftCons = 1 on every pair not decoded as NEG |
| `source` | all rows sharing `pair_id` in the given files | also pools language views (Track 4, or Tracks 1–4 together) |

`neg_bias` shifts the `NEGATIVE_OTHER` decision and should be tuned on dev.
`structural_prior` decodes rows that have twins among the reversible labels
and singletons as NEG. It is exact on train/dev, but it exploits dataset
construction. Read the risk notes in `docs/RESEARCH.md` before enabling it
for a submission.

## Evaluation, prediction, submission

```bash
# metrics under all decoding modes
uv run python scripts/training/evaluate.py --ckpt checkpoints/<run>/best.pt \
    --data data/raw/dico/final_data/dev/dico_nli_dev_track1_participant_labeled.csv

# predictions (ensemble: pass several --ckpt; reuse saved --logprobs *.npz)
uv run python -m src.pipelines.predict --ckpt checkpoints/<run>/best.pt \
    --inputs data/raw/dico/final_data/dev/dico_nli_dev_track{1,2,3,4}_participant_labeled.csv \
    --out_dir results/predictions/dev --decoding source

# confirm with the organizers' scorer
uv run python scripts/submission/official_score.py \
    --gold data/raw/dico/final_data/dev/dico_nli_dev_track1_reference.csv \
    --predictions results/predictions/dev/track1_predictions.csv \
    --output_dir results/predictions/dev/track1_official

# validate against templates and zip for CodaBench (files at ZIP root)
uv run python scripts/submission/make_submission.py --pred_dir results/predictions/dev \
    --templates data/raw/dico/final_data/dev/dico_nli_dev_track{1,2,3,4}_submission_template.csv \
    --out results/submissions/dev_submission.zip
```

`src.pipelines.eval.dico_metrics` re-implements the official metrics. It
matches the official scorer exactly on all dev tracks and decoding modes.

## Repository layout

```
src/
├── labels.py            # label set, Rev permutation, instance-id / twin rules
├── config.py            # ModelConfig — backbone, max_length, head dropout
├── data.py              # CSV I/O, DicoDataset, PairCollator, TwinBatchSampler, synthetic data
├── decoding.py          # independent / twin / source consistency decoding
├── augment.py           # train-only augmentation: reverse_negatives, transitive
├── modules/
│   ├── model.py         # PairClassifier (HF cross-encoder -> logits, feature)
│   └── loss.py          # DicoLoss = CE + reversal-consistency KL
├── pipelines/
│   ├── config.py        # TrainingConfig — loop, loss, decoding, callbacks
│   ├── train.py         # training loop + callback wiring
│   ├── eval.py          # Weighted F1 / SoftCons / HardCons + diagnostics
│   └── predict.py       # checkpoints / log-probs -> track<N>_predictions.csv
├── callbacks/           # checkpoint, early_stopping, lr_scheduler, wandb
└── utils/               # io, model (incl. tiny offline backbone), device helpers
configs/                 # training YAML configs
scripts/
├── data/                # fetch_data, inspect_data, make_folds, ists_to_dico
├── training/            # train, evaluate, smoke_test
└── submission/          # official_score, make_submission
tests/                   # pytest suite (offline)
docs/                    # RESEARCH, TODO, NOTES, EXPERIMENTS, research/ survey
notebooks/               # exploratory notebooks
```

## Configuration

Configuration is split in two:

- **`src/config.py` → `ModelConfig`** holds the *architecture*: backbone id,
  tokenizer, max length, head dropout, and `fix_pair_template`. It is saved into every checkpoint.
- **`src/pipelines/config.py` → `TrainingConfig`** holds the *training loop*:
  data files, optimizer, loss weights, decoding, and callbacks. It is loaded
  from `configs/*.yaml`, overridable from the CLI, and applies ModelConfig
  overrides from its `arch:` block.

```yaml
consistency_weight: 0.5          # symmetric KL between p(x) and Rev p(x_rev)
decoding: twin                   # independent | twin | source
lr_scheduler: {type: warmup_linear, t_max: auto, warmup_ratio: 0.1}
early_stopping: {enabled: true, monitor: dico_mean, mode: max, patience: 3}
best_metric: dico_mean           # or weighted_f1 / soft_cons / hard_cons / val_loss
augment: [reverse_negatives]     # train-only augmentation (src/augment.py)
arch: {backbone: microsoft/deberta-v3-base, max_length: 128}
# arch: {backbone: HiTZ/JaunBERT, fix_pair_template: true}  # repair a broken pair template
# arch: {head: entail2}   # compose labels from p(a|=b), p(b|=a) of the NLI head
# arch: {backbone: Qwen/Qwen3.5-9B, lora: {r: 16}, load_in_4bit: true,
#        torch_dtype: bfloat16, pair_template: "A: {a}\nB: {b}"}  # + amp_dtype: bfloat16
```

LoRA checkpoints store only the adapter and head; the base model is
re-downloaded from the Hub when loading.

Grouped CV: `scripts/data/make_folds.py --k 5` writes `data/folds/f<k>/` in the
official layout (a `pair_id` never straddles folds, same folds on every
track), so a run only needs `--data_root data/folds/f<k>`.

## Results, report, model card

- Technical report: [`docs/reports/dico_nli_report.pdf`](docs/reports/dico_nli_report.pdf)
  (source `.tex` alongside; tables from `scripts/training/report_tables.py`).
- Model card: [`docs/reports/MODEL_CARD.md`](docs/reports/MODEL_CARD.md). Checkpoints
  are in the private Hugging Face repo `LakoreAI/dico-nli-checkpoints`.
- Run log: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

Dev-set ALL-4 Weighted F1 / SoftCons / HardCons (official scorer):
hierarchical fixed ensemble 81.4 / 96.8 / 89.5, selected 83.7 / 98.6 / 91.7;
with the structural prior (pending organizer approval), selected
94.3 / 100 / 93.1. Submissions: `results/submissions/<run_id>/submission.zip`
(gitignored; also on the HF repo).

## Citation

If you use this code, models or results, please cite (GitHub's "Cite this
repository" button reads [`CITATION.cff`](CITATION.cff)):

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

## License

This repository is released under the [LICENSE](LICENSE) (MIT). The task data,
scorer, and starter kit belong to the organizers (GPL-3.0). They are fetched
at runtime into `data/raw/dico` and not redistributed here.
