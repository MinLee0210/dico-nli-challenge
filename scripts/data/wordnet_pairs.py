"""Synthetic lexical-relation pairs from WordNet (EN / ES / EU) in DiCo CSV format.

Targets the main dev error: related-but-not-entailing pairs (co-hyponyms such
as creek / river) read as entailment. Uses Open Multilingual Wordnet 1.4 via
the `wn` package; English, Spanish and Basque synsets share ILI ids, so the
English hypernym graph labels all three languages.

    hyponym  -> hypernym   FORWARD_ENTAILMENT (reversed twin: BACKWARD)
    synonyms (one synset)  EQUIVALENCE        (reversed twin: EQUIVALENCE)
    co-hyponyms (siblings) NEGATIVE_OTHER     (single row, like DiCo NEG)

A share of pairs is cross-lingual (text1 and text2 in different languages),
matching Track 4. Pair ids are prefixed `wn` so they never collide with DiCo.

    uv run --with wn python scripts/data/wordnet_pairs.py --per_type 600 \
        --out data/raw/dico/final_data/train/wordnet_pairs.csv
"""

import argparse
import csv
import random
from pathlib import Path

LEXICONS = {"en": "omw-en:1.4", "es": "omw-es:1.4", "eu": "omw-eu:1.4"}
REVERSE = {
    "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT",
    "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT",
    "EQUIVALENCE": "EQUIVALENCE",
}


def ili_id(ss):
    """The synset's ILI id (a str in newer `wn`, an object with .id in older)."""
    ili = ss.ili
    return getattr(ili, "id", ili) or None


def lemmas_by_ili(wordnets):
    """ili -> {lang: [lemma, ...]} for noun and verb synsets with an ILI id."""
    table = {}
    for lang, w in wordnets.items():
        for ss in w.synsets():
            if ss.pos not in ("n", "v") or not ili_id(ss):
                continue
            forms = [f.replace("_", " ") for f in ss.lemmas()]
            forms = [f for f in forms if 1 <= len(f.split()) <= 3 and not f.isupper()]
            if forms:
                table.setdefault(ili_id(ss), {})[lang] = forms
    return table


def main() -> None:
    import wn

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per_type", type=int, default=600, help="pairs per label per language")
    ap.add_argument("--cross_lingual", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    for spec in LEXICONS.values():
        try:
            wn.Wordnet(spec)
        except Exception:
            wn.download(spec)
    wordnets = {lang: wn.Wordnet(spec) for lang, spec in LEXICONS.items()}
    ili = lemmas_by_ili(wordnets)
    en = wordnets["en"]
    print({lang: sum(lang in v for v in ili.values()) for lang in LEXICONS}, "synsets with lemmas")

    # English graph: hyponym -> direct hypernyms, and siblings.
    hyper, children = [], {}
    for ss in en.synsets():
        if ss.pos not in ("n", "v") or ili_id(ss) not in ili:
            continue
        for h in ss.hypernyms():
            if ili_id(h) in ili:
                hyper.append((ili_id(ss), ili_id(h)))
                children.setdefault(ili_id(h), []).append(ili_id(ss))
    siblings = [(a, b) for kids in children.values() if len(kids) > 1
                for a, b in [tuple(rng.sample(kids, 2))]]
    rng.shuffle(hyper)
    rng.shuffle(siblings)

    def pick(sid, lang):
        forms = ili.get(sid, {}).get(lang)
        return rng.choice(forms) if forms else None

    def langs():
        l1 = rng.choice(list(LEXICONS))
        l2 = rng.choice(list(LEXICONS)) if rng.random() < args.cross_lingual else l1
        return l1, l2

    rows, seen, n = [], set(), 0

    def add(a, b, l1, l2, label):
        nonlocal n
        if not a or not b or a.lower() == b.lower() or (a.lower(), b.lower()) in seen:
            return False
        seen.add((a.lower(), b.lower()))
        n += 1
        pid = f"wn{n}"
        rows.append(dict(instance_id=f"{pid}__{l1}-{l2}__original", pair_id=pid,
                         text1_lang=l1, text2_lang=l2, text1=a, text2=b, label=label))
        if label in REVERSE:
            rows.append(dict(instance_id=f"{pid}__{l2}-{l1}__flipped", pair_id=pid,
                             text1_lang=l2, text2_lang=l1, text1=b, text2=a,
                             label=REVERSE[label]))
        return True

    target = args.per_type * len(LEXICONS)
    counts = {"FORWARD_ENTAILMENT": 0, "EQUIVALENCE": 0, "NEGATIVE_OTHER": 0}
    for hypo, hyp in hyper:
        if counts["FORWARD_ENTAILMENT"] >= target:
            break
        l1, l2 = langs()
        counts["FORWARD_ENTAILMENT"] += add(pick(hypo, l1), pick(hyp, l2), l1, l2, "FORWARD_ENTAILMENT")
    for sid in rng.sample(list(ili), len(ili)):
        if counts["EQUIVALENCE"] >= target:
            break
        l1, l2 = langs()
        a, b = pick(sid, l1), pick(sid, l2)
        if l1 == l2 and len(ili[sid].get(l1, [])) < 2:
            continue
        counts["EQUIVALENCE"] += add(a, b, l1, l2, "EQUIVALENCE")
    for a_id, b_id in siblings:
        if counts["NEGATIVE_OTHER"] >= target:
            break
        l1, l2 = langs()
        counts["NEGATIVE_OTHER"] += add(pick(a_id, l1), pick(b_id, l2), l1, l2, "NEGATIVE_OTHER")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["instance_id", "pair_id", "text1_lang", "text2_lang", "text1", "text2", "label"]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} rows, pairs per label {counts}")


if __name__ == "__main__":
    main()
