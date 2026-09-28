import pytest
import torch

from src.data import (
    DicoDataset,
    PairCollator,
    TwinBatchSampler,
    make_synthetic_dico,
    read_dico_csv,
    read_dico_files,
    twin_indices,
    write_dico_csv,
    write_predictions,
)
from src.labels import NEGATIVE_OTHER, reverse_label


def test_synthetic_structure_matches_official():
    examples = make_synthetic_dico(
        n_pairs=50, langs=(("en", "es"), ("es", "en")), seed=0
    )
    twins = twin_indices(examples)
    for ex, t in zip(examples, twins):
        if ex.label == NEGATIVE_OTHER:
            assert t is None
        else:
            assert t is not None
            assert examples[t].label == reverse_label(ex.label)


def test_csv_roundtrip(tmp_path):
    examples = make_synthetic_dico(n_pairs=10, seed=1)
    path = tmp_path / "train.csv"
    write_dico_csv(path, examples)
    assert read_dico_csv(path) == examples


def test_read_unlabeled_file(tmp_path):
    path = tmp_path / "test.csv"
    path.write_text(
        "instance_id,pair_id,text1_lang,text2_lang,text1,text2\n"
        "p__en-en__original,p,en,en,  a   dog ,dog\n"
    )
    [ex] = read_dico_csv(path)
    assert ex.label is None
    assert ex.text1 == "a dog"
    assert not DicoDataset([ex]).has_labels


def test_read_rejects_bad_label_and_duplicates(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "instance_id,pair_id,text1,text2,label\np__en-en__original,p,a,b,CONTRADICTION\n"
    )
    with pytest.raises(ValueError):
        read_dico_csv(path)
    dup = tmp_path / "dup.csv"
    write_dico_csv(dup, make_synthetic_dico(n_pairs=3, seed=0))
    with pytest.raises(ValueError):
        read_dico_files([dup, dup])


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_dico_csv(tmp_path / "nope.csv")


def test_write_predictions_format(tmp_path):
    path = tmp_path / "track1_predictions.csv"
    write_predictions(path, ["a__en-en__original"], ["EQUIVALENCE"])
    assert path.read_text() == "instance_id,label\na__en-en__original,EQUIVALENCE\n"
    with pytest.raises(ValueError):
        write_predictions(path, ["a"], ["bogus"])


def test_twin_batch_sampler_keeps_twins_together():
    dataset = DicoDataset(make_synthetic_dico(n_pairs=40, seed=2))
    sampler = TwinBatchSampler(dataset, batch_size=5, shuffle=True, seed=0)
    batches = list(sampler)
    seen = [i for b in batches for i in b]
    assert sorted(seen) == list(range(len(dataset)))
    for batch in batches:
        assert len(batch) <= 5
        members = set(batch)
        for i in batch:
            twin = dataset.twins[i]
            assert twin is None or twin in members


def test_collator_builds_twin_positions(tiny_tokenizer):
    dataset = DicoDataset(make_synthetic_dico(n_pairs=10, seed=3))
    collator = PairCollator(dataset, tiny_tokenizer, max_length=16)
    batch = collator(list(range(len(dataset))))
    assert batch["input_ids"].shape[0] == len(dataset)
    assert batch["labels"].tolist() == dataset.labels.tolist()
    for pos, twin in enumerate(batch["twin"].tolist()):
        expected = dataset.twins[pos]
        assert twin == (-1 if expected is None else expected)
    # twin outside the batch -> -1
    first_with_twin = next(i for i, t in enumerate(dataset.twins) if t is not None)
    lone = collator([first_with_twin])
    assert lone["twin"].tolist() == [-1]
    assert lone["input_ids"].dtype == torch.long
