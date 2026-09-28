# Experiments

Runbook for gated experiments. Each experiment records: hypothesis, config,
command, result, and decision. Report the three official scores on dev
(Weighted F1 / SoftCons / HardCons) and confirm them with
`scripts/submission/official_score.py` before quoting. HardCons on the 277 dev
pairs per track has a standard error of about ±2.6 points, so treat smaller
differences as noise unless they are confirmed by grouped CV or several seeds.

## Reference points

| System | Track | WF1 | SoftCons | HardCons | Source |
|---|---|--:|--:|--:|---|
| DeBERTa-v3-base (pilot; PhrasIS 6-label Positives-A, "unseen" coherence — not the 4-label DiCo set) | EN | 0.75 | 0.84 | 0.79 | LREC 2026 pilot |
| Leaderboard #1 (dev, 2026-09-23) | EN | 0.924 | 1.000 | 0.910 | CodaBench |
| Leaderboard #1 (dev, 2026-09-23) | EU | 0.903 | 1.000 | 0.884 | CodaBench |
| Random log-probs + `twin` decoding | EN | 0.268 | 0.733 | 0.274 | local sanity check |

## Planned

### E1 — Track 1 baseline, decoding ablation
- **Hypothesis:** `twin` decoding raises SoftCons to ~1.0 on non-NEG pairs and
  HardCons by several points, with WF1 unchanged or slightly up.
- **Config:** `configs/train.yaml`
- **Command:**

  ```bash
  uv run python scripts/training/train.py --config configs/train.yaml
  uv run python scripts/training/evaluate.py --ckpt checkpoints/<run>/best.pt \
      --data data/raw/dico/final_data/dev/dico_nli_dev_track1_participant_labeled.csv
  ```

- **Result:** _pending_
- **Decision:** _pending_

### E2 — Reversal-consistency loss
- **Hypothesis:** `consistency_weight` ∈ {0.5, 1.0} improves `independent`
  SoftCons and adds a little HardCons on top of `twin` decoding.
- **Config:** `configs/train.yaml` with `--consistency_weight 0.5|1.0`
- **Result:** _pending_

### E3 — One multilingual model, source-pair pooling
- **Hypothesis:** a multilingual encoder trained on Tracks 1–4 with
  `decoding: source` beats per-track models on ES/EU and on Track 4. mmBERT
  (newest) is competitive with XLM-R-large (best published Basque NLI).
- **Configs:** `configs/multilingual_mmbert.yaml`,
  `configs/xlmr_large_xnli_multilingual.yaml`,
  `configs/multilingual_mrbert.yaml`, `configs/multilingual_mdeberta.yaml`
- **Result:** _pending_

### E5 — Data augmentation
- **Hypothesis:** `reverse_negatives` improves NEG recall and HardCons at no
  cost. `transitive` adds FE/BE signal but may add noise.
- **Config:** best E1/E3 config with `--augment`, `--augment reverse_negatives`,
  and `--augment reverse_negatives transitive`
- **Result:** _pending_

### E4 — NEG threshold (`neg_bias`) sweep on saved log-probs
- **Hypothesis:** tuning `neg_bias` on dev trades WF1 against HardCons; pick the
  `dico_mean` maximum.
- **Command:** `src.pipelines.predict --logprobs ... --neg_bias <b>` (no re-inference)
- **Result:** _pending_

## Template

### <YYYY-MM-DD> — <short name>

- **Hypothesis:**
- **Config:** `configs/<file>.yaml`
- **Command:**

  ```bash
  uv run python scripts/training/train.py --config configs/<file>.yaml
  ```

- **Result:** _WF1 / SoftCons / HardCons, plus where the log lives._
- **Decision:** _adopt / discard / follow up with ..._
