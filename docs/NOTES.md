# Notes

Decisions log. Append newest at the top. Record *why*, not just *what*, so the
reasoning survives after the code changes.

## 2026-09-28 (later)

- **Latest models first.** At the user's request, a Hub check (2026-09-28)
  found releases newer than the deep survey: mmBERT (still the newest massively
  multilingual encoder), JaunBERT (HiTZ, Jul 2026, Basque), MrBERT (BSC, Dec
  2025), Qwen3.5 (Feb 2026), Latxa-Qwen3.5 (Jun 2026), Gemma 4 (Mar 2026).
  mmBERT is promoted to P0. JaunBERT, Qwen3.5, and Latxa-Qwen3.5 are the first
  picks in their families.
- **`transformers>=5`** (was `>=4.48,<5`). 4.57 has no seq-cls class for
  `qwen3_5`; 5.17 does. The test suite and smoke run pass unchanged on 5.17.
  Gemma 4 still has no seq-cls class.
- **`fix_pair_template`.** JaunBERT's upstream tokenizer encodes a pair as
  `<unk> A<s><unk> B<s>`, and MrBERT drops the trailing separator. The option
  rebuilds `[CLS] A [SEP] B [SEP]` from the model config's ids. It is off by
  default, because it would *break* mmBERT (whose config sets `cls_token_id`
  to `<eos>`; its own template `<bos> A<eos> B<eos>` is fine).
- **Augmentation is train-only and opt-in** (`augment: []` in every config),
  so baselines stay clean and each augmentation is an ablation.
  `reverse_negatives` is label-safe (NEG relations are symmetric).
  `transitive` is logically sound, but it propagates PhrasIS's
  context-dependent labels (e.g. `'6 Killed' ⇒ 'Four dead'`), so it must win
  on CV.

## 2026-09-28

- **Template → DiCo-NLI.** Replaced the reference MLP-over-`.npz` pipeline with
  a Hugging Face cross-encoder over the official CSVs. Kept the template's
  architecture/training config split, callbacks, `forward -> (logits, feature)`
  contract, scripts layout, and offline test suite.
- **Consistency is decoded, not just learned.** Every reversible item's twin is
  in the same file, so `src/decoding.py` pools log-probs in the canonical
  (`original`) frame and decodes one label per group. `twin` mode makes every
  non-NEG pair SoftCons-consistent at zero training cost. The alternative
  (hoping a reversal-KL loss makes independent argmax consistent) stays
  available as `consistency_weight`, but it gives no guarantee.
- **Twin id rule, not language-set grouping.** The reverse of
  `en-es__original` is `es-en__flipped`, not `en-es__flipped`. Grouping by
  language set would merge two different twin pairs. Verified on all 8
  train/dev reference files (0 mismatches).
- **Own metric implementation + external official scorer.** The task repo is
  GPL-3.0 and this repo is MIT, so the scorer is fetched into `data/raw/dico`
  and called as a subprocess, not vendored. `dico_metrics` matches it exactly
  (max diff 0.0 over 4 tracks × 3 decoding modes on random predictions).
- **Model selection on `dico_mean`** (mean of Weighted F1, SoftCons, HardCons).
  The task reports a three-score profile with no single main metric.
- **`structural_prior` defaults to off.** "No twin ⇒ NEGATIVE_OTHER" is exact
  on train/dev, but it exploits dataset construction, may be read as
  "deceptive" under the rules, and breaks if the test layout differs. Ask the
  organizers first.
- **Do not train on raw PhrasIS / iSTS headlines+images.** Train+dev already
  hold ~1,559 of PhrasIS's 1,946 reversible source pairs. The rest are very
  likely the test pairs, and their labels are public.
- **Fine-tuning runs on a separate GPU machine.** The local box is not a
  constraint on model size.
- Dropped `scikit-learn` (metrics are self-implemented) and added
  `transformers`, `sentencepiece`, `protobuf`.
