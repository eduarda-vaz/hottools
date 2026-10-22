import numpy as np
import pytest
from conftest import AMBIGUOUS_POS, REF_LENGTH

from hottools.reference import (
    ReferenceReader,
    one_hot_reference,
    read_reference_interval,
)


def test_ambiguous_reference_is_zero():
    onehot, idx = one_hot_reference("ACGTN")
    np.testing.assert_array_equal(
        onehot,
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
    )
    assert idx[-1] == 255


def test_soft_masked_bases_are_encoded():
    onehot, idx = one_hot_reference("acgt")
    np.testing.assert_array_equal(onehot, np.eye(4, dtype=np.float32))
    np.testing.assert_array_equal(idx, [0, 1, 2, 3])


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_one_hot_respects_dtype(dtype):
    onehot, _ = one_hot_reference("ACGT", dtype=dtype)
    assert onehot.dtype == np.dtype(dtype)


def test_one_hot_row_sums(reference_sequence, reference_onehot):
    onehot, _ = one_hot_reference(reference_sequence)
    np.testing.assert_array_equal(onehot, reference_onehot)
    # Exactly one hot bit per position, except at the ambiguous base.
    sums = onehot.sum(axis=1)
    assert sums[AMBIGUOUS_POS - 1] == 0
    assert np.count_nonzero(sums == 1) == REF_LENGTH - 1


def test_interval_is_one_based_inclusive(fasta_path, reference_sequence):
    assert read_reference_interval(fasta_path, "chr1", 1, 1) == reference_sequence[0]
    assert read_reference_interval(fasta_path, "chr1", 10, 12) == reference_sequence[9:12]
    whole = read_reference_interval(fasta_path, "chr1", 1, REF_LENGTH)
    assert whole == reference_sequence


def test_reader_reuses_one_handle(fasta_path, reference_sequence):
    with ReferenceReader(fasta_path) as reader:
        assert reader.contigs == ["chr1"]
        first = reader.interval("chr1", 1, 50)
        second = reader.interval("chr1", 51, 100)
    assert first + second == reference_sequence[:100]


def test_missing_contig_names_alternatives(fasta_path):
    with pytest.raises(ValueError, match="not found in"):
        read_reference_interval(fasta_path, "chrZ", 1, 10)


def test_naming_convention_hint(fasta_path):
    with pytest.raises(ValueError, match="Did you mean 'chr1'"):
        read_reference_interval(fasta_path, "1", 1, 10)


def test_interval_past_contig_end_is_rejected(fasta_path):
    with pytest.raises(ValueError, match="expected 5000 bp"):
        read_reference_interval(fasta_path, "chr1", 1, 5000)


@pytest.mark.parametrize("start,end", [(0, 10), (10, 5)])
def test_invalid_interval_bounds(fasta_path, start, end):
    with pytest.raises(ValueError):
        read_reference_interval(fasta_path, "chr1", start, end)
