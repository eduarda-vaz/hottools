"""Signature grouping and fixed-length structural transformations."""

from __future__ import annotations

import numpy as np
import pytest

from hottools.constants import N_ALLELE_SLOTS
from hottools.edits import Edit, build_structural_edits
from hottools.transforms import (
    FROM_NOWHERE,
    apply_transform,
    compile_all,
    compile_transform,
    haplotype_signatures,
)

BASES = "ACGT"


def pack(a1: int, a2: int) -> int:
    """One packed genotype byte from an allele pair."""
    return a1 * N_ALLELE_SLOTS + a2


def onehot(sequence: str, dtype=np.float32) -> np.ndarray:
    out = np.zeros((len(sequence), 4), dtype=dtype)
    for i, base in enumerate(sequence):
        out[i, BASES.index(base)] = 1
    return out


def decode(block: np.ndarray) -> str:
    return "".join(
        BASES[int(row.argmax())] if row.sum() > 0 else "-" for row in block
    )


def make_edits(specs, genotypes):
    """``specs`` is a list of ``(record, Edit)``; positions follow it."""
    positions = np.array(
        sorted({edit.start - 1 for _, edit in specs}), dtype=np.int32
    )
    if positions.size != genotypes.shape[0]:
        positions = np.arange(genotypes.shape[0], dtype=np.int32)
    return build_structural_edits(specs, positions, genotypes)


class TestHaplotypeSignatures:
    def test_haplotypes_with_the_same_edits_share_a_group(self):
        # Three records; S1 carries all on haplotype 1, S2 carries all on
        # both, S3 carries none.
        # Rows are records, columns are samples: S1 is 1|0, S2 is 1|1
        # and S3 is 0|0 at every one of the three records.
        genotypes = np.array(
            [[pack(1, 0), pack(1, 1), pack(0, 0)]] * 3, dtype=np.uint8
        )
        edits = make_edits(
            [
                (0, Edit(11, 14, "", "del", 1)),
                (1, Edit(21, 21, "GG", "ins", 1)),
                (2, Edit(31, 34, "", "del", 1)),
            ],
            genotypes,
        )
        sig = haplotype_signatures(edits)
        assert sig.n_haplotypes == 6
        # S1H1 = all, S1H2 = none, S2H1 = all, S2H2 = all, S3 = none.
        assert sig.n_groups == 2
        assert sig.group[0] == sig.group[2] == sig.group[3]
        assert sig.group[1] == sig.group[4] == sig.group[5]

    def test_a_missing_allele_carries_nothing(self):
        genotypes = np.array([[pack(4, 1)]], dtype=np.uint8)
        edits = make_edits([(0, Edit(11, 14, "", "del", 1))], genotypes)
        sig = haplotype_signatures(edits)
        assert not sig.carriers[0, 0]      # the uncalled slot
        assert sig.carriers[1, 0]          # the called slot

    def test_reference_group_carries_no_edits(self):
        genotypes = np.array([[pack(0, 0), pack(1, 1)]], dtype=np.uint8)
        edits = make_edits([(0, Edit(11, 14, "", "del", 1))], genotypes)
        sig = haplotype_signatures(edits)
        reference = sig.reference_group
        assert reference is not None
        assert not sig.unique[reference].any()


class TestCompileTransform:
    def _compile(self, specs, carriers, work, out_length, anchor_ref,
                 anchor_offset=0):
        n = max(record for record, _ in specs) + 1
        genotypes = np.full((n, 1), pack(1, 1), dtype=np.uint8)
        edits = make_edits(specs, genotypes)
        active = np.flatnonzero(np.asarray(carriers, dtype=bool))
        return compile_transform(
            active, edits,
            work_start=work[0], work_end=work[1],
            out_length=out_length,
            anchor_ref=anchor_ref, anchor_offset=anchor_offset,
        ), edits

    def test_no_active_edits_is_the_identity(self):
        transform, _ = self._compile(
            [(0, Edit(11, 14, "", "del", 1))], [False],
            work=(1, 20), out_length=20, anchor_ref=1,
        )
        assert transform.is_identity
        np.testing.assert_array_equal(transform.source, np.arange(20))

    def test_a_deletion_pulls_downstream_bases_in(self):
        work = "ACGTACGTACGTACGTACGT"          # 20 bases at 1..20
        transform, edits = self._compile(
            [(0, Edit(5, 9, "", "del", 1))], [True],
            work=(1, 20), out_length=16, anchor_ref=1,
        )
        out = np.zeros((1, 16, 4), dtype=np.float32)
        apply_transform(
            onehot(work)[None, :, :], np.array([0]), transform,
            edits.insert_onehot(np.dtype(np.float32)), out,
        )
        # Positions 5..8 are gone; the rest closes up.
        assert decode(out[0]) == (work[:4] + work[8:])[:16]

    def test_an_insertion_pushes_bases_out(self):
        work = "ACGTACGTACGTACGTACGT"
        transform, edits = self._compile(
            [(0, Edit(5, 5, "TTT", "ins", 1))], [True],
            work=(1, 20), out_length=20, anchor_ref=1,
        )
        out = np.zeros((1, 20, 4), dtype=np.float32)
        apply_transform(
            onehot(work)[None, :, :], np.array([0]), transform,
            edits.insert_onehot(np.dtype(np.float32)), out,
        )
        assert decode(out[0]) == (work[:4] + "TTT" + work[4:])[:20]

    def test_the_anchor_lands_at_its_requested_offset(self):
        work = "ACGTACGTACGTACGTACGT"
        # Delete 3..6, anchor on position 11 at output offset 5.
        transform, edits = self._compile(
            [(0, Edit(3, 7, "", "del", 1))], [True],
            work=(1, 20), out_length=12, anchor_ref=11, anchor_offset=5,
        )
        out = np.zeros((1, 12, 4), dtype=np.float32)
        apply_transform(
            onehot(work)[None, :, :], np.array([0]), transform,
            edits.insert_onehot(np.dtype(np.float32)), out,
        )
        assert decode(out[0])[5] == work[10]

    def test_running_out_of_reference_is_counted_not_hidden(self):
        work = "ACGTACGTAC"                      # 10 bases
        transform, edits = self._compile(
            [(0, Edit(2, 9, "", "del", 1))], [True],
            work=(1, 10), out_length=10, anchor_ref=1,
        )
        assert transform.n_unfilled == 7
        assert (transform.source == FROM_NOWHERE).sum() == 7
        out = np.zeros((1, 10, 4), dtype=np.float32)
        apply_transform(
            onehot(work)[None, :, :], np.array([0]), transform,
            edits.insert_onehot(np.dtype(np.float32)), out,
        )
        # Unfilled positions are all-zero rows, not invented bases.
        # One surviving base before the deletion, then positions 9-10.
        assert decode(out[0]) == work[0] + work[8:] + "-" * 7

    def test_an_edit_overlapping_an_earlier_one_is_skipped(self):
        transform, _ = self._compile(
            [(0, Edit(5, 11, "", "del", 1)), (1, Edit(8, 15, "", "del", 1))],
            [True, True], work=(1, 30), out_length=20, anchor_ref=1,
        )
        assert transform.n_skipped == 1

    def test_edits_sharing_only_an_anchor_are_both_applied(self):
        # Trimmed intervals [5,9) and [9,13) touch but do not overlap.
        transform, _ = self._compile(
            [(0, Edit(5, 9, "", "del", 1)), (1, Edit(9, 13, "", "del", 1))],
            [True, True], work=(1, 30), out_length=20, anchor_ref=1,
        )
        assert transform.n_skipped == 0

    def test_an_edit_clipped_by_the_window_still_applies(self):
        # The deletion starts upstream of the window but removes bases
        # inside it.
        work = "ACGTACGTACGTACGTACGT"             # positions 11..30
        transform, edits = self._compile(
            [(0, Edit(8, 15, "", "del", 1))], [True],
            work=(11, 30), out_length=14, anchor_ref=11,
        )
        assert transform.n_skipped == 0
        out = np.zeros((1, 14, 4), dtype=np.float32)
        apply_transform(
            onehot(work)[None, :, :], np.array([0]), transform,
            edits.insert_onehot(np.dtype(np.float32)), out,
        )
        # Positions 11..14 are removed, so the output starts at 15.
        assert decode(out[0]) == work[4:18]


class TestCompileAll:
    def test_one_transform_per_signature_group(self):
        genotypes = np.array(
            [[pack(1, 0), pack(1, 1), pack(0, 0), pack(0, 1)]], dtype=np.uint8
        )
        edits = make_edits([(0, Edit(11, 15, "", "del", 1))], genotypes)
        sig = haplotype_signatures(edits)
        transforms = compile_all(
            sig, edits, work_start=1, work_end=40, out_length=30,
            anchor_ref=1, anchor_offset=0,
        )
        assert len(transforms) == sig.n_groups
        assert any(t.is_identity for t in transforms)
        assert any(not t.is_identity for t in transforms)

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_the_insert_pool_follows_the_output_dtype(self, dtype):
        genotypes = np.array([[pack(1, 1)]], dtype=np.uint8)
        edits = make_edits([(0, Edit(11, 11, "ACGT", "ins", 1))], genotypes)
        pool = edits.insert_onehot(np.dtype(dtype))
        assert pool.dtype == dtype
        np.testing.assert_array_equal(pool.argmax(axis=1), [0, 1, 2, 3])
