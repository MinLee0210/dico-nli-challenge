# Research

**Question.** Which combination of backbone, training objective, and
consistency-aware decoding maximizes all three official DiCo-NLI scores
(Weighted F1, SoftCons, HardCons) on each track (EN, ES, EU, mixed)?

**Why it matters.** DiCo-NLI asks whether an NLI system's decision on
`(A, B)` is compatible with its decision on `(B, A)`. A system can be accurate
per item yet inconsistent under reversal, or consistent yet wrong. The official
result is a three-score profile per track with no single main metric, so
closing the accuracy/consistency gap is the whole game.

> Survey method: `/research` → `/research-deep`, 20 items × 25 fields, with
> outline, fields, and per-item JSON in [`research/`](research/). This page is
> the synthesis. Anything marked *(est.)* is extrapolated from related work,
> not measured on DiCo. As of 2026-09-28.

## 1. Task in one screen

| | |
|---|---|
| Input | ordered phrase pair `(text1, text2)`, ~6 words per pair in total, from PhrasIS (iSTS image-caption/headline chunks) |
| Labels | `EQUIVALENCE`, `FORWARD_ENTAILMENT` (text1 ⊨ text2), `BACKWARD_ENTAILMENT`, `NEGATIVE_OTHER` (any other PhrasIS relation) |
| Rev | EQ↔EQ, FE↔BE. NEG has no reverse. |
| Tracks | T1 EN, T2 ES, T3 EU, T4 mixed (EN/ES/EU cross-lingual combinations). Each track is ranked separately. |
| Metrics | **Weighted F1** (support-weighted over 4 labels), **SoftCons** (pred(x) is reversible and pred(x_rev) = Rev(pred(x)), gold ignored), **HardCons** (both directions correct). Consistency is computed only over gold-reversible pairs. |
| Dates | dev phase open. **Evaluation 10–31 Jan 2027.** Papers due Feb 2027. |
| Rules | External data and models (LLMs, APIs) are allowed but must be documented. Disclosure is mandatory: access regime, architecture family, external data. Hidden labels must not be used. "Deceptive" submissions can be withheld. |

**Scorer subtlety (from `evaluation_functions/metrics.py`).** On a
gold-reversible pair, predicting `NEGATIVE_OTHER` for *either* direction
scores SoftCons = 0 and HardCons = 0 for that pair. NEG/NEG is *not*
soft-consistent. Wrong NEG decisions on reversible pairs are therefore the
costliest error there is.

**Reference points.**

| System | Setup | WF1 | Soft | Hard |
|---|---|--:|--:|--:|
| DeBERTa-v3-base, LREC 2026 pilot | EN, PhrasIS **6-label** Positives-A, "unseen" coherence | 0.75 | 0.84 | 0.79 |
| DeBERTa-v3-large, pilot | same | 0.76 | 0.81 | 0.77 |
| BART-large, pilot | same | 0.68 | 0.80 | 0.72 |
| CodaBench dev #1 (2026-09-23) | T1 / T3 / T4 | 0.924 / 0.903 / 0.902 | 1.0 | 0.910 / 0.884 / 0.882 |

The pilot numbers are not on the 4-label DiCo set, so they are only loosely
comparable. The leaderboard leader's SoftCons of exactly 1.0 on every track
shows that they already use consistency-enforcing decoding.

## 2. Data facts the approach is built on

Verified locally on all train/dev files (`scripts/data/inspect_data.py`):

1. **Twins are always present.** Instance ids are
   `<pair_id>__<l1>-<l2>__<original|flipped>`. The reversed twin swaps the
   languages and the direction tag. This rule matches `reverse_pair_id` on 100%
   of the 8 reference files. Note that the twin of `en-es__original` is
   `es-en__flipped`, so twins cannot be grouped by language set.
2. **NEG items are singletons.** Every `NEGATIVE_OTHER` row is a lone
   `original` row, and every reversible row has its twin. "No twin ⇒ NEG" has
   precision and recall of 1.0 on train and dev. `pair_id` numbers also
   separate them: 1–1946 are reversible, 1947+ are NEG.
3. **A source pair spans every track.** The same `pair_id` has 2 rows in each
   of T1–T3 and 12 rows in T4 (6 language combinations × 2 directions), which
   makes 18 views of each reversible source pair. All `original` rows share one
   orientation: decoding gold one-hot scores in `source` mode reproduces gold
   exactly.
4. **Size.** Per monolingual track, train has 1,282 reversible pairs + 478 NEG
   (3,042 rows) and dev has 277 + 106 (660 rows). Labels are roughly balanced
   (BE 29 / EQ 26 / FE 29 / NEG 16 %). Splits are grouped by source pair
   (0 overlap between train and dev).
5. **Dev noise.** HardCons over 277 dev pairs has a standard error of about
   ±2.6 points. 5-fold grouped CV over train+dev brings it to about ±1.1.
6. **Contamination.** PhrasIS has 1,946 reversible English source pairs, and
   train+dev already contain 1,559. The remaining ~387 are almost certainly
   the test pairs, and their gold labels are public in PhrasIS and in the
   iSTS 2015/16 headlines/images alignment files. **Never train on raw PhrasIS
   or iSTS headlines/images.** iSTS *answers-students* is not part of PhrasIS
   and is clean.

## 3. Key findings

1. **Consistency is a decoding problem first.** Pool the log-probs of `x` and
   Rev(`x_rev`) and decode one label per twin group. Every pair not decoded as
   NEG becomes soft-consistent, HardCons becomes pair accuracy, and pooling two
   views usually adds accuracy. This needs no training and works with any model
   (ConCoRD, Li et al. 2019). Implemented as `decoding: twin`.
2. **The NEG decision drives the remaining gap.** With twin decoding, SoftCons
   loss comes only from NEG decisions on reversible pairs. Tune a NEG
   threshold (`neg_bias`) on dev or grouped CV to trade WF1 against HardCons.
   A hierarchical head (related-vs-NEG gate plus direction head) makes this
   explicit.
3. **Pool across language views.** In T4, the 12 views of a source pair share
   one canonical label. Pooling them (`decoding: source`) is safe within a
   track *(est. +2–5 WF1/HardCons on T4)*. Pooling *across tracks* (using EN
   views to decide EU) could add +3–8 on ES/EU *(est.)*, but it uses other
   tracks' test inputs, and the organizers describe the monolingual tracks as
   independent. **Ask before submitting such a run.**
4. **Start from NLI-fine-tuned checkpoints.** The pilot fine-tuned only plain
   hub checkpoints. NLI intermediate training encodes entailment direction
   (PhrasIS FORW = premise entails hypothesis, the standard NLI direction).
   *Est. +2–5 WF1*, lower seed variance.
5. **Backbone ranking by track.**
   - **T1:** DeBERTa-v3 (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`).
     In the pilot, large was not better than base on coherence and was
     seed-unstable, so run several seeds. ModernBERT trailed DeBERTa in the
     pilot (0.72/0.76/0.72 large).
   - **T3/T4:** XLM-R-large (`joeddav/xlm-roberta-large-xnli`, which includes
     cross-language XNLI pairs, matching T4). It has the best published Basque
     NLI results (XNLIeu 81.1 zero-shot / 83.8 translate-train, vs mDeBERTa
     79.0 / 81.4). No multilingual NLI checkpoint saw Basque NLI data, so
     add XNLIeu (`HiTZ/xnli-eu`) as an intermediate step.
   - **T2:** multilingual models beat Spanish-only ones on XNLI-es (XLM-R-large
     84.7, mDeBERTa 83.3, RoBERTa-large-BNE 82.6).
   - **Fast all-track baseline:** `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.
6. **LLMs are ensemble members, not replacements.** LoRA-tuned 4–12B decoders
   (Qwen3.5-9B, Latxa-Qwen3.5-4B for Basque; see §3a) are plausible +2–6 WF1
   on T2–T4 *(est.)* and diverse ensemble members. Basque NLI by prompting
   alone has been weak (50–67% vs 76–84% fine-tuned). API LLMs are best used
   as a teacher or pseudo-labeler with a reverse-consistency filter, and only
   if they beat the fine-tuned student on dev.
7. **Grouped-CV fold ensembles are the reliable gain.** They give *est.* +1–3 WF1
   and +2–4 HardCons, and make every other decision measurable at about ±1
   point instead of ±2.6.
8. **The data is small; augment it.** See §3b. Two label-safe augmentations
   derived from the training set itself are implemented, and they grow Track 1
   train from 3,042 to 4,198 rows.

## 3a. Latest relevant models (checked on the HF Hub, 2026-09-28)

The deep survey covered models up to 2025. A follow-up Hub check found newer
releases that should be tried **first** within each family. The pipeline now
targets `transformers>=5` (5.17 installed), which provides
`ModernBertForSequenceClassification` and `Qwen3_5ForSequenceClassification`.

| Model | Released | Type | Tracks | Why | Status in repo |
|---|---|---|---|---|---|
| **mmBERT-base / small** (`jhu-clsp/mmBERT-*`) | Sep 2025 (ICML 2026) | ModernBERT encoder, 1,833 langs incl. EU | all | newest massively multilingual encoder; beats XLM-R-base on XNLI (77.7 avg); fast | `configs/multilingual_mmbert.yaml`. Pair encoding `<bos> A<eos> B<eos>`. Forward and backward pass verified on T3 data. |
| **JaunBERT** (`HiTZ/JaunBERT`) | Jul 2026 | ModernBERT-style Basque encoder (308M), from MrBERT on latxa-corpus-v2 | T3 (EU views of T4) | newest Basque encoder from the Basque NLP group | `configs/track3_jaunbert.yaml`. The upstream tokenizer pair template is broken (`<unk> A<s><unk> B<s>`); fixed with `fix_pair_template: true`. |
| **MrBERT** (`BSC-LT/MrBERT`) | Dec 2025 | ModernBERT multilingual (308M, 6.1T tokens, EN/ES/EU + 30 more) | all | Iberian-strong new multilingual encoder, 8k context | `configs/multilingual_mrbert.yaml`, with `fix_pair_template: true` (upstream template drops the second separator). |
| **Qwen3.5** (`Qwen/Qwen3.5-{2B,4B,9B,27B}`) | Feb 2026 | decoder (VLM) | all | supersedes Qwen3; seq-cls class exists in transformers 5 | needs the LoRA path (TODO) |
| **Latxa-Qwen3.5-2B/4B** (`HiTZ/Latxa-Qwen3.5-*`) | Jun 2026 | Qwen3.5 adapted to EU/GL/CA | T3, T4 | newest Basque-adapted LLM; small enough for full LoRA sweeps | needs the LoRA path (TODO) |
| Latxa-Llama-3.1-70B-Instruct-v2 | Jun 2026 | decoder | T3, T4 | strongest Basque LLM; use as teacher/pseudo-labeler | API/vLLM-style teacher (TODO) |
| Gemma 4 (`google/gemma-4-*`) | Mar 2026 | multimodal decoder | all | newest Gemma | no seq-cls class in transformers 5.17; would need a custom head (P2) |

Priority change: **mmBERT is promoted to P0** as the modern multilingual
baseline, trained alongside XLM-R-large, which is still the published Basque
NLI leader. **JaunBERT replaces BERnaT/RoBERTa-euscrawl** as the first Basque
encoder to try. **Qwen3.5 / Latxa-Qwen3.5 replace Qwen3 / Gemma 3 / Latxa-3.1-8B**
as the first LoRA candidates. None of these has DiCo, XNLI-eu, or PhrasIS
numbers yet, so each must earn its place in grouped CV.

## 3b. Data augmentation

Train holds 1,282 reversible source pairs + 478 NEG per track (plus the same
pairs in two more languages and six mixed combinations). Ranked by safety:

| Augmentation | Label source | Yield | Status |
|---|---|---|---|
| **Train on all tracks** (EN/ES/EU views of the same source pair) | gold, human-verified translations | ×9 rows per source pair | supported (`train_files` lists all tracks) |
| **`reverse_negatives`**: add (B, A) for each NEG (A, B) | NEG relations (similar / related / opposite / unaligned) are symmetric | +478 per track | implemented (`augment:`). Also trains a symmetric NEG gate and puts NEG pairs under the consistency loss. |
| **`transitive`**: a ⊨ b, b ⊨ c ⇒ a ⊨ c over phrases shared between train pairs (EQ counts as both directions), per language | logical closure of gold labels | +339 pairs (678 rows) on T1; +6.2k FE/BE rows on all tracks | implemented. **Noisy:** PhrasIS labels are context-dependent (e.g. `'6 Killed' ⇒ 'Four dead'`), so gate on CV and consider down-weighting. |
| NLI intermediate training (MNLI/XNLI/XNLIeu) | external gold | 400k+ | via NLI checkpoints; XNLIeu step TODO |
| iSTS **answers-students** chunk alignments (SPE1→FE, SPE2→BE, EQUI→EQ, else NEG) | external gold, disjoint from PhrasIS | ~several thousand chunk pairs *(est.)* | TODO. Never use iSTS headlines/images (test leakage). |
| Lexical entailment: WordNet / MCR (EU, ES) hypernyms, PPDB 2.0 (Equivalence / ForwardEntailment / ReverseEntailment / OtherRelated) | curated resources | large | TODO. Targets FE/BE specificity errors, which dominate the pilot's error analysis. |
| Synthetic minimal-pair edits: add/remove a modifier ("black dog" ⊨ "dog"), quantifier swaps, number changes → NEG | rule-based | unbounded | TODO. Mirrors PhrasIS phenomena; verify with an LLM or a human on a sample. |
| Back-translation / paraphrase of one side | MT/LLM, label-preserving only for EQ-safe edits | ×2–3 | P2. Must filter for relation preservation: MT drops the modifiers that decide FE vs BE. |
| LLM-generated pairs (teacher labels, reverse-consistency filter) | LLM | unbounded | P1. Keep only if the teacher beats the student on dev. |

Every augmentation is train-only and must be judged on grouped CV (±1 pt
resolution). Report which ones were used in the system paper.

## 4. Approach landscape

Priority reflects expected gain per unit of effort for this task. Per-item
detail is in `research/results/<item>.json`.

| Item | Tracks | Priority | Expected effect *(est.)* |
|---|---|:--:|---|
| Joint pair decoding (+ NEG gate / `neg_bias`) | all | **P0** | Soft → ~1.0, Hard up several pts, WF1 flat/up |
| Within-track view pooling (`source`) | T4 (all) | **P0** | +2–5 WF1/Hard on T4 |
| NLI-initialized checkpoints (+ XNLIeu for EU) | all | **P0** | +2–5 WF1, lower variance |
| DeBERTa-v3-large NLI | T1 | **P0** | primary T1 backbone |
| XLM-R-large XNLI | T2–T4 | **P0** | primary multilingual backbone |
| **mmBERT-base** (latest multilingual encoder) | all | **P0** | modern multilingual baseline and ensemble member |
| Grouped CV + fold/seed ensembles + calibration | all | **P0** | +1–3 WF1, +2–4 Hard |
| **Augmentation: all-track training + `reverse_negatives`** | all | **P0** | more data at no label risk (§3b) |
| Ask organizers: structural cue, cross-track pooling, test id format | all | **P0** (action) | decides legal gains of several points |
| **JaunBERT** (Jul 2026, Basque) | T3 | P1 (first Basque encoder) | +0.5–2 WF1 in ensemble |
| **Qwen3.5-9B / Latxa-Qwen3.5-4B** LoRA | T2–T4 | P1 (first LLMs) | +2–6 WF1, diversity |
| **MrBERT** (Dec 2025, Iberian-strong) | all | P1 | diverse multilingual member |
| Augmentation: `transitive`, iSTS answers-students, WordNet/MCR/PPDB, minimal-pair edits | all | P1 | +0.5–2 WF1 on FE/BE specificity errors |
| mDeBERTa-v3-base 2mil7 | all | P1 | fast baseline and ensemble member |
| Reversal / cross-view consistency loss (`consistency_weight`) | all | P1 | +0.5–1.5 WF1/Hard on top of decoding |
| Antisymmetric / hierarchical two-head model | all | P1 | cleaner NEG gate, reversal-equivariant training |
| Older LLMs (Qwen3-8B, Gemma-3-12B, Latxa-3.1-8B, Salamandra-7B) | T2–T4 | P2 | superseded by the Qwen3.5 family |
| Older Basque encoders (RoBERTa-euscrawl-large, BERnaT-large) | T3 | P2 | only if JaunBERT underperforms |
| LLM teacher distillation (consistency-filtered) | all | P1 | +1–3 WF1 only if teacher > student on dev |
| API LLMs zero/few-shot | all | P1 | teacher / EU ensemble member |
| Translate-test as extra view; filtered back-translation | T2, T3 | P2 | +0.5–2 WF1 |
| Encoder-decoder (BART-large, mT0/mT5-XL, Aya-101) | varies | P2 | diversity only |
| ModernBERT / Ettin, EuroBERT, Spanish-only encoders | T1/T2 | P2 | ensemble diversity |
| bge-reranker-v2-m3 | all | P2 | NEG-gate feature only (no direction signal) |
| Structural cue in a ranked submission | all | P2 | +3–8 WF1 *if allowed*; disqualification and format-change risk |

## 5. Recommended system

1. **Baselines, one per track family.** DeBERTa-v3-large NLI (T1),
   XLM-R-large XNLI and mmBERT-base, both trained on all tracks (T2–T4), plus
   mDeBERTa as the fast reference. `twin`/`source` decoding. Run each with and
   without `augment: [reverse_negatives]`. Configs:
   `configs/track1_deberta_v3_large_nli.yaml`,
   `configs/xlmr_large_xnli_multilingual.yaml`,
   `configs/multilingual_mmbert.yaml`,
   `configs/multilingual_mdeberta.yaml`.
2. **Evaluation discipline.** 5-fold CV grouped by `pair_id` across all tracks
   (a source pair never straddles folds), multiple seeds, and official-scorer
   confirmation.
3. **Training improvements, gated by CV.** Consistency loss, XNLIeu
   intermediate step for Basque, NLI-head → 4-way head initialization,
   lexical/iSTS answers-students augmentation.
4. **Diversity, latest models first.** JaunBERT for T3, MrBERT, and a LoRA
   LLM (Qwen3.5-9B; Latxa-Qwen3.5-4B for Basque).
5. **Final.** Average log-probs across fold models and backbones, tune
   `neg_bias` on OOF predictions, decode with `source` (within track), and
   validate and package with `scripts/submission/make_submission.py`.

## 6. Risks and open questions for the organizers

| Question | Why it matters | Default until answered |
|---|---|---|
| May systems use the file structure (twin presence ⇒ reversible)? | Exact NEG oracle on train/dev, worth several WF1 points. It could be judged "deceptive". | Off (`structural_prior: false`). Use it only as a dev diagnostic. |
| May a track's predictions use other tracks' test inputs for the same `pair_id`? | Cross-track pooling could lift ES/EU by several points. | Pool only within a track. |
| Will test ids keep `<pair_id>__l1-l2__original/flipped`? | Twin/source decoding depends on it. Twins could otherwise be matched by swapped texts. | Assume yes. Add a text-matching fallback (see TODO). |
| PhrasIS data-use policy? | Test pairs are probably in public PhrasIS/iSTS. | Never train on raw PhrasIS or iSTS headlines/images. |

Also: phrase-level MT can drop the quantifiers and modifiers that decide
FE vs BE, so filter any translated or back-translated examples for label
preservation.

**Disclosure metadata to record per run** (mandatory in the system paper):
access regime (open-weight / API / hybrid), architecture family
(encoder / decoder / enc-dec / ensemble), and external data (none / public /
private). Keep it in each `docs/EXPERIMENTS.md` entry.

## 7. Scope

- **In scope:** the four tracks; HF cross-encoders; LoRA decoders; consistency
  decoding, loss, and architecture; public auxiliary NLI data (MNLI, XNLI,
  XNLIeu, iSTS answers-students, PPDB/WordNet/MCR); API LLMs as teachers or
  ensemble members.
- **Out of scope:** training on PhrasIS or iSTS headlines/images; using hidden
  test labels; the structural cue in ranked submissions unless the organizers
  explicitly allow it.

## 8. Sources

- Task repo, rules, data, scorer: <https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI>,
  <https://inigolopezgazpio.net/SemEval-2027-Task-2-DiCo-NLI/>
- CodaBench: <https://www.codabench.org/competitions/18038/>
- Pilot: Apaolaza et al., *Assessing Logical Coherence of LLMs via Fine-Grained
  NLI*, LREC 2026, <https://lrec.elra.info/lrec2026-main-423>
- PhrasIS: Lopez-Gazpio et al., Logic Journal of the IGPL 32(6), 2024,
  doi:10.1093/jigpal/jzae037
- XNLIeu: Heredia et al., NAACL 2024 (arXiv:2404.06996)
- Consistency: Li et al. 2019 (arXiv:1909.00126), ConCoRD (arXiv:2211.11875),
  xTune (arXiv:2106.08226), Asai & Hajishirzi 2020 (arXiv:2004.10157),
  Berglund et al. 2024, *The Reversal Curse* (arXiv:2309.12288)
- Models: DeBERTa-v3 / MoritzLaurer NLI checkpoints; XLM-R (joeddav XNLI);
  mmBERT (arXiv:2509.06888); EuroBERT (arXiv:2503.05500); BERnaT
  (arXiv:2512.03903); Latxa 3.1 (arXiv:2506.07597); Qwen3 (arXiv:2505.09388);
  Gemma 3 (arXiv:2503.19786); Salamandra (arXiv:2502.08489)
- Latest models (HF Hub cards, checked 2026-09-28): `jhu-clsp/mmBERT-base`
  (ICML 2026 poster), `HiTZ/JaunBERT`, `BSC-LT/MrBERT`, `Qwen/Qwen3.5-*`,
  `HiTZ/Latxa-Qwen3.5-*`, `HiTZ/Latxa-Llama-3.1-70B-Instruct-v2`,
  `google/gemma-4-*`
- Augmentation: PPDB 2.0 (Pavlick et al. 2015), iSTS answers-students
  (SemEval-2016 Task 2), logic-guided augmentation (Asai & Hajishirzi 2020)
- Full per-item source lists: `docs/research/results/*.json` (`links` field)

## Results

Record runs in [`EXPERIMENTS.md`](EXPERIMENTS.md), with raw outputs in
`docs/reports/<date>/`. Link them here rather than copying tables by hand.
