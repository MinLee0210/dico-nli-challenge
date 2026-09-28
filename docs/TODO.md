# TODO

Next steps, ordered by priority. Rationale is in [`RESEARCH.md`](RESEARCH.md);
log every run in [`EXPERIMENTS.md`](EXPERIMENTS.md). Fine-tuning happens on the
remote GPU machine. Key dates: **evaluation 10–31 Jan 2027**, papers due Feb 2027.

## Done (2026-09-28)

- [x] Survey the task, data, scorer, and 20 candidate approaches (`docs/research/`)
- [x] Adapt the template: HF cross-encoder, official CSV I/O, twin-aware
      batches, reversal-KL loss, `independent`/`twin`/`source` decoding,
      official-equivalent metrics, predict/ensemble CLI, CodaBench packager
- [x] Verify the twin-id rule on all reference files, and that `dico_metrics`
      equals the official scorer (diff 0.0, 4 tracks × 3 modes)
- [x] Survey latest models on the HF Hub (mmBERT, JaunBERT, MrBERT, Qwen3.5,
      Latxa-Qwen3.5, Gemma 4); move to `transformers>=5`; add
      `fix_pair_template` for broken upstream pair templates (JaunBERT, MrBERT)
- [x] mmBERT: config + real forward/backward check on T3 data
- [x] Train-only augmentation: `reverse_negatives`, `transitive` (`augment:`)

## P0 — now (October 2026)

- [ ] **Email the organizers** (inigo.lopez@ehu.eus) or open a GitHub issue:
  - [ ] Is using twin presence / singleton structure allowed?
  - [ ] May one track's predictions use other tracks' test inputs (same `pair_id`)?
  - [ ] Will test `instance_id`s keep `<pair_id>__l1-l2__original|flipped`?
  - [ ] PhrasIS / iSTS data-use policy
- [ ] Register on CodaBench and submit a trivial dev ZIP to test the format
      (`make_submission.py`)
- [ ] Set up the training machine: `uv sync --extra wandb`, `fetch_data.py`,
      `pytest`, `smoke_test.py`
- [ ] **E1** T1 baseline `configs/train.yaml` (DeBERTa-v3-base); report all 3
      decoding modes via `evaluate.py`; confirm with `official_score.py`
- [ ] **E1b** T1 `configs/track1_deberta_v3_large_nli.yaml`, 3 seeds
- [ ] **E3** `configs/multilingual_mmbert.yaml`,
      `configs/xlmr_large_xnli_multilingual.yaml`, and
      `configs/multilingual_mdeberta.yaml`: per-track dev scores with `twin` vs
      `source` decoding. Also compare mmBERT-small against mmBERT-base, and try
      lr 3e-5 / 5e-5 / 8e-5 for mmBERT
- [ ] **E5 augmentation** on the best E1/E3 setup: `augment: []` vs
      `[reverse_negatives]` vs `[reverse_negatives, transitive]`, and
      single-track vs all-track training. Judge on grouped CV
- [ ] **E4** `neg_bias` sweep on saved log-probs (`predict.py --logprobs`);
      add a small sweep script that writes a WF1/Soft/Hard vs bias table
- [ ] Grouped K-fold CV: add `scripts/training/make_folds.py` (split train+dev
      by `pair_id`, identical folds across all tracks) and a `fold` option in
      the configs; report OOF mean ± std
- [ ] Seed/fold ensembling via averaged log-probs (already supported by
      `predict.py --ckpt a b c`)

## P1 — November 2026

- [ ] **E2** `consistency_weight` ∈ {0.25, 0.5, 1.0}; add a cross-view KL term
      (all views of a `pair_id` in canonical frame) and a sampler that groups
      source pairs
- [ ] Map the NLI head onto the 4-way head at init (entailment → FE row, etc.)
      instead of a random re-init; compare on CV
- [ ] Hierarchical head: related-vs-NEG gate + direction head, symmetrized by
      scoring both orders (`p(b,a) = Rev p(a,b)` by construction)
- [ ] Basque: `configs/track3_jaunbert.yaml` (JaunBERT, Jul 2026; try the
      `periodistikoa` news revision too, since PhrasIS comes partly from
      headlines) and `configs/multilingual_mrbert.yaml`; intermediate
      fine-tuning on XNLIeu (`HiTZ/xnli-eu`) before DiCo
- [ ] LoRA sequence-classification path for decoder LLMs (add `peft`; a
      `lora` block in ModelConfig), latest first: **Qwen3.5-9B**,
      **Latxa-Qwen3.5-4B** (Basque), Qwen3.5-27B if budget allows. These are
      VLM checkpoints; load the text-only `Qwen3_5ForSequenceClassification`
- [ ] More augmentation (see RESEARCH §3b): iSTS **answers-students**
      (SPE1→FE, SPE2→BE, EQUI→EQ, else NEG); WordNet / MCR (EU, ES) hypernym
      pairs and PPDB 2.0 relations; rule-based minimal-pair edits (drop or add
      a modifier ⇒ FE/BE; swap a number or direction word ⇒ NEG). Never use
      PhrasIS or iSTS headlines/images
- [ ] Down-weight augmented rows (a per-example weight in `DicoExample` and
      the loss) if `transitive` helps but is noisy
- [ ] API-LLM teacher on dev (both directions in one prompt, canonical
      question); distill only if it beats the student
- [ ] Temperature scaling per model before averaging log-probs
- [ ] Text-matching fallback for twin detection if test ids change

## P2 — December 2026

- [ ] Translate-test extra views for ES/EU (NLLB-200 / HiTZ mt-hitz) with a
      label-preservation filter
- [ ] LLM-generated pairs and back-translated paraphrases with a
      relation-preservation filter (teacher: Latxa-Llama-3.1-70B-Instruct-v2
      or an API model)
- [ ] Diversity members: BART-large (T1), mT0-XL, ModernBERT-large NLI;
      Gemma 4 only with a custom head (no seq-cls class in transformers 5.17);
      older Basque encoders (RoBERTa-euscrawl-large, BERnaT) if JaunBERT
      underperforms
- [ ] Cross-track pooling, **only if the organizers allow it**
- [ ] Error analysis notebook: confusion by label and language, FE/BE
      specificity errors, NEG false positives on reversible pairs

## Evaluation phase (10–31 January 2027)

- [ ] Freeze the systems per track by 2027-01-09. Pin model revisions and
      record disclosure metadata (access regime, architecture family, external data)
- [ ] Run `predict.py` on the test files, validate with `make_submission.py`,
      and upload. Keep the log-probs for the paper's ablations
- [ ] Select the official submission bundle on CodaBench

## Paper (February 2027)

- [ ] System description: decoding ablation, consistency loss, view pooling,
      ensembles, per-track three-score profiles, and error analysis
- [ ] Document every external model and dataset, prompt, and access regime
