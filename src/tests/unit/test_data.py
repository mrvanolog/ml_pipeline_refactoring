import pandas as pd
import pytest

from utils.data import build_labels, build_vocab, reader


@pytest.mark.parametrize('partition', ['train', 'dev', 'test'])
def test_reader(partition, data_dir):
    """Check that reader utility function works and returns output of the desired type.
    """
    data, targets = reader(partition, data_dir)

    assert isinstance(data, pd.Series)
    assert isinstance(targets, pd.Series)


@pytest.mark.parametrize('partition, no_labels', [
    ('train', 17930),
    ('dev', 13072),
    ('test', 13072),
])
def test_build_labels(partition, no_labels, data_dir):
    """Check that build_labels utility function works, returns output of the desired type,
    and the number of output labels is correct.
    """
    _, targets = reader(partition, data_dir)
    fam2label = build_labels(targets)

    assert isinstance(fam2label, dict)
    assert len(fam2label) == no_labels


@pytest.mark.parametrize('partition, no_amino_acids', [
    ('train', 22),
    ('dev', 22),
    ('test', 22),
])
def test_build_vocab(partition, no_amino_acids, data_dir):
    """Check that build_vocab utility function works, returns output of the desired type,
    and the number of output labels is correct.
    """
    data, _ = reader(partition, data_dir)
    word2id = build_vocab(data)

    assert isinstance(word2id, dict)
    assert len(word2id) == no_amino_acids
