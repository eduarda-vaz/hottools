"""End-to-end behaviour of the opt-in length-changing variant classes."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import (
    DEL_POS,
    INS_POS,
    MNV_POS,
    SYMDEL_END,
    SYMDEL_POS,
)

from hottools.pipeline import RunConfig, run_hottools
from hottools.regions import Region
from hottools.variants import StructuralEditWarning

BASES = "ACGT"

#: The region used throughout. It stops short of the contig end so a
#: deletion has real reference left to pull downstream bases from.
LENGTH = 180


def decode(block: np.ndarray) -> str:
    return "".join(
        BASES[int(row.argmax())] if row.sum() > 0 else "-" for row in block
    )


def as_decoded(sequence: str) -> str:
    """Render a reference string the way the tensor decodes it.

    An ambiguous base has no channel to set, so it comes back as an
    all-zero row rather than a letter.
    """
    return "".join(b if b in BASES else "-" for b in sequence)


def run(tmp_path, fasta_path, vcf_path, **kwargs):
    """Run the pipeline over chr1:1-200 and return (tensor, metadata)."""
    output = tmp_path / "out.npy"
    config = RunConfig(
        fasta_path=str(fasta_path),
        vcf_path=str(vcf_path),
        region=kwargs.pop("region", Region("chr1", 1, LENGTH)),
        output_path=str(output),
        mode=kwargs.pop("mode", "haplotypes"),
        missing_policy="reference",
        overwrite="replace",
        **kwargs,
    )
    metadata = run_hottools(config)
    return np.load(output), metadata


class TestDefaultIsSnvOnly:
    def test_indels_are_absent_unless_requested(
        self, tmp_path, fasta_path, indel_vcf_path, reference_sequence
    ):
        tensor, meta = run(tmp_path, fasta_path, indel_vcf_path)
        assert meta["variant_classes"] == ["snv"]
        assert meta["structural_edits"] == 0
        assert meta["working_length"] == LENGTH
        assert meta["working_pad_right"] == 0
        # Nothing in the file is a plain SNV, so every haplotype is the
        # reference and the output is what it has always been.
        for sample in range(tensor.shape[0]):
            for hap in range(2):
                assert decode(tensor[sample, hap]) == as_decoded(reference_sequence)[:LENGTH]

    def test_the_transform_is_skipped_entirely_without_edits(
        self, tmp_path, fasta_path, clean_vcf_path
    ):
        _, meta = run(
            tmp_path, fasta_path, clean_vcf_path, variant_classes="all"
        )
        assert meta["structural_edits"] == 0
        assert meta["structural_signatures"] == 0
        assert meta["working_pad_right"] == 0


class TestDeletion:
    def test_a_deletion_closes_the_gap_and_pulls_bases_in(
        self, tmp_path, fasta_path, indel_vcf_path, reference_sequence
    ):
        tensor, meta = run(
            tmp_path, fasta_path, indel_vcf_path, variant_classes="del"
        )
        assert meta["variant_deletion_edits"] == 1
        # 3 bases removed, so the working window reads 3 bases further.
        assert meta["working_pad_right"] == 3
        assert meta["working_length"] == LENGTH + 3

        carrier = decode(tensor[1, 0])          # S2 is 1|1
        expected = (
            reference_sequence[:DEL_POS]
            + reference_sequence[DEL_POS + 3:]
        )
        assert carrier == as_decoded(expected)[:LENGTH]
        # S3 is homozygous reference and is untouched.
        assert decode(tensor[2, 0]) == as_decoded(reference_sequence)[:LENGTH]

    def test_only_the_carrier_haplotype_changes(
        self, tmp_path, fasta_path, indel_vcf_path, reference_sequence
    ):
        tensor, _ = run(
            tmp_path, fasta_path, indel_vcf_path, variant_classes="del"
        )
        # S1 is 1|0: haplotype 1 carries the deletion, haplotype 2 does
        # not, and both tracks stay the same length.
        assert decode(tensor[0, 0]) != as_decoded(reference_sequence)[:LENGTH]
        assert decode(tensor[0, 1]) == as_decoded(reference_sequence)[:LENGTH]


class TestInsertion:
    def test_an_insertion_pushes_the_tail_out_of_the_window(
        self, tmp_path, fasta_path, indel_vcf_path, reference_sequence
    ):
        tensor, meta = run(
            tmp_path, fasta_path, indel_vcf_path, variant_classes="ins"
        )
        assert meta["variant_insertion_edits"] == 1
        # An insertion needs no extra reference: the window already has
        # more bases than the output can hold.
        assert meta["working_pad_right"] == 0

        expected = (
            reference_sequence[:INS_POS]
            + "GGG"
            + reference_sequence[INS_POS:]
        )
        assert decode(tensor[1, 0]) == as_decoded(expected)[:LENGTH]


class TestSymbolicDeletion:
    def test_end_defines_the_deleted_interval(
        self, tmp_path, fasta_path, indel_vcf_path, reference_sequence
    ):
        tensor, meta = run(
            tmp_path, fasta_path, indel_vcf_path,
            variant_classes="symbolic-del",
        )
        assert meta["variant_symbolic_deletion_edits"] == 1
        removed = SYMDEL_END - SYMDEL_POS
        assert meta["working_pad_right"] == removed

        # POS is the base before the event, so it survives.
        expected = (
            reference_sequence[:SYMDEL_POS]
            + reference_sequence[SYMDEL_POS + removed:]
        )
        assert decode(tensor[1, 0]) == as_decoded(expected)[:LENGTH]

    def test_a_symbolic_deletion_without_end_is_refused_loudly(
        self, tmp_path, fasta_path, symbolic_no_end_vcf_path
    ):
        with pytest.warns(Warning):
            _, meta = run(
                tmp_path, fasta_path, symbolic_no_end_vcf_path,
                variant_classes="all",
            )
        assert meta["variant_unsupported_alt_alleles"] == 1
        assert meta["variant_kept"] == 0


class TestMnv:
    def test_every_base_of_the_block_is_substituted(
        self, tmp_path, fasta_path, indel_vcf_path, reference_sequence
    ):
        tensor, meta = run(
            tmp_path, fasta_path, indel_vcf_path, variant_classes="mnv"
        )
        assert meta["variant_mnv_records"] == 1
        # An MNV changes no lengths, so it needs no transform at all.
        assert meta["structural_edits"] == 0
        assert meta["working_pad_right"] == 0

        got = decode(tensor[1, 0])
        reference = as_decoded(reference_sequence)
        block = got[MNV_POS - 1:MNV_POS + 2]
        reference_block = reference[MNV_POS - 1:MNV_POS + 2]
        assert block != reference_block
        assert all(a != b for a, b in zip(block, reference_block, strict=True))
        # Nothing outside the block moved.
        assert got[:MNV_POS - 1] == reference[:MNV_POS - 1]
        assert got[MNV_POS + 2:] == reference[MNV_POS + 2:LENGTH]


class TestOverlappingRecords:
    def test_the_first_record_in_pos_order_wins(
        self, tmp_path, fasta_path, overlapping_del_vcf_path,
        reference_sequence,
    ):
        with pytest.warns(StructuralEditWarning):
            tensor, meta = run(
                tmp_path, fasta_path, overlapping_del_vcf_path,
                variant_classes="del",
            )
        assert meta["variant_overlapping_records_suppressed"] > 0
        # Only the deletion at 100 is realised: 6 bases from 101.
        expected = reference_sequence[:100] + reference_sequence[106:]
        assert decode(tensor[0, 0]) == as_decoded(expected)[:LENGTH]

    def test_records_sharing_only_an_anchor_both_apply(
        self, tmp_path, fasta_path, adjacent_del_vcf_path, reference_sequence
    ):
        tensor, meta = run(
            tmp_path, fasta_path, adjacent_del_vcf_path,
            variant_classes="del",
        )
        # Trimmed intervals [101,104) and [105,109) do not overlap, so
        # neither record suppresses the other.
        assert meta["variant_overlapping_records_suppressed"] == 0
        # The first record removes 101-104, the second 105-109.
        expected = reference_sequence[:100] + reference_sequence[109:]
        assert decode(tensor[0, 0]) == as_decoded(expected)[:LENGTH]


class TestAnchor:
    def test_start_is_the_default_and_pins_the_first_base(
        self, tmp_path, fasta_path, indel_vcf_path, reference_sequence
    ):
        _, meta = run(
            tmp_path, fasta_path, indel_vcf_path, variant_classes="all"
        )
        assert meta["anchor_position"] == 1
        assert meta["anchor_output_offset"] == 0
        assert meta["working_pad_left"] == 0

    def test_center_pins_the_middle_position(
        self, tmp_path, fasta_path, indel_vcf_path
    ):
        _, meta = run(
            tmp_path, fasta_path, indel_vcf_path,
            variant_classes="all", anchor="center",
        )
        assert meta["anchor_output_offset"] == (LENGTH - 1) // 2
        assert meta["anchor_position"] == 1 + (LENGTH - 1) // 2
        # A pinned interior anchor needs reference on both sides.
        assert meta["working_pad_left"] > 0

    def test_an_explicit_position_is_honoured(
        self, tmp_path, fasta_path, indel_vcf_path, reference_sequence
    ):
        tensor, meta = run(
            tmp_path, fasta_path, indel_vcf_path,
            variant_classes="mnv", anchor="150",
        )
        assert meta["anchor_position"] == 150
        assert meta["anchor_output_offset"] == 149
        # No length changes here, so the anchor lands on its own base.
        assert decode(tensor[2, 0])[149] == as_decoded(reference_sequence)[149]

    def test_an_anchor_outside_the_region_is_an_error(
        self, tmp_path, fasta_path, indel_vcf_path
    ):
        with pytest.raises(ValueError, match="outside the region"):
            run(
                tmp_path, fasta_path, indel_vcf_path,
                variant_classes="all", anchor="900",
            )

    def test_a_nonsense_anchor_is_an_error(
        self, tmp_path, fasta_path, indel_vcf_path
    ):
        with pytest.raises(ValueError, match="1-based position"):
            run(
                tmp_path, fasta_path, indel_vcf_path,
                variant_classes="all", anchor="middle-ish",
            )


class TestModeGuard:
    @pytest.mark.parametrize("mode", ["dosage", "consensus"])
    @pytest.mark.parametrize("classes", ["del", "ins", "symbolic-del"])
    def test_length_changing_classes_need_haplotype_mode(
        self, tmp_path, fasta_path, indel_vcf_path, mode, classes
    ):
        with pytest.raises(ValueError, match="only meaningful in"):
            run(
                tmp_path, fasta_path, indel_vcf_path,
                variant_classes=classes, mode=mode,
            )

    @pytest.mark.parametrize("mode", ["dosage", "consensus"])
    def test_equal_length_classes_are_fine_in_every_mode(
        self, tmp_path, fasta_path, indel_vcf_path, mode
    ):
        _, meta = run(
            tmp_path, fasta_path, indel_vcf_path,
            variant_classes="mnv", mode=mode,
        )
        assert meta["variant_mnv_records"] == 1


class TestStrandAndMetadata:
    def test_minus_strand_is_the_reverse_complement_of_plus(
        self, tmp_path, fasta_path, indel_vcf_path
    ):
        complement = {"A": "T", "C": "G", "G": "C", "T": "A", "-": "-"}
        plus, _ = run(
            tmp_path, fasta_path, indel_vcf_path, variant_classes="all"
        )
        minus, _ = run(
            tmp_path, fasta_path, indel_vcf_path, variant_classes="all",
            region=Region("chr1", 1, LENGTH, strand="-"),
        )
        for sample in range(plus.shape[0]):
            for hap in range(2):
                expected = "".join(
                    complement[b] for b in reversed(decode(plus[sample, hap]))
                )
                assert decode(minus[sample, hap]) == expected

    def test_metadata_records_the_working_window(
        self, tmp_path, fasta_path, indel_vcf_path
    ):
        output = tmp_path / "meta.npy"
        config = RunConfig(
            fasta_path=str(fasta_path),
            vcf_path=str(indel_vcf_path),
            region=Region("chr1", 1, LENGTH),
            output_path=str(output),
            mode="haplotypes",
            missing_policy="reference",
            variant_classes="all",
            overwrite="replace",
        )
        run_hottools(config)
        meta = json.loads(
            (tmp_path / "meta.npy.metadata.json").read_text("utf-8")
        )
        for key in (
            "variant_classes", "anchor_position", "anchor_output_offset",
            "working_start", "working_end", "working_length",
            "working_pad_left", "working_pad_right", "structural_edits",
            "structural_signatures", "unfilled_positions",
            "overlapping_edits_skipped", "overlap_chain_lookback",
        ):
            assert key in meta, key
        assert meta["working_length"] == (
            meta["working_end"] - meta["working_start"] + 1
        )
        assert meta["structural_signatures"] >= 1
