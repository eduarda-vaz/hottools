"""Normalisation and classification of reference-space edits."""

from __future__ import annotations

import numpy as np
import pytest

from hottools.edits import (
    EDIT_DELETION,
    EDIT_INSERTION,
    EDIT_REPLACEMENT,
    LENGTH_CHANGING,
    VARIANT_CLASSES,
    Edit,
    UnsupportedVariant,
    build_structural_edits,
    classify_alt,
    parse_variant_classes,
)


class TestParseVariantClasses:
    def test_default_is_snv_only(self):
        assert parse_variant_classes(None) == frozenset({"snv"})
        assert parse_variant_classes("") == frozenset({"snv"})

    def test_snv_is_always_included(self):
        assert "snv" in parse_variant_classes("del")
        assert "snv" in parse_variant_classes("symbolic-del")

    def test_indel_expands_to_both_directions(self):
        # A group name must not be shadowed by a single-class alias:
        # 'indel' means deletions *and* insertions.
        assert parse_variant_classes("indel") == frozenset(
            {"snv", "del", "ins"}
        )
        assert parse_variant_classes("indels") == parse_variant_classes(
            "indel"
        )

    def test_all_covers_every_class(self):
        assert parse_variant_classes("all") == frozenset(VARIANT_CLASSES)

    def test_aliases_and_case(self):
        assert parse_variant_classes("SNP, Deletion") == frozenset(
            {"snv", "del"}
        )
        assert parse_variant_classes("mnp") == frozenset({"snv", "mnv"})

    def test_unknown_class_names_the_choices(self):
        with pytest.raises(ValueError, match="symbolic-del"):
            parse_variant_classes("inversion")

    def test_length_changing_set_is_the_opt_in_group(self):
        assert LENGTH_CHANGING == {"del", "ins", "symbolic-del"}
        assert "snv" not in LENGTH_CHANGING
        assert "mnv" not in LENGTH_CHANGING


class TestClassifyAlt:
    def test_single_base_substitution(self):
        edit = classify_alt(10, "A", "G", 1)
        assert (edit.variant_class, edit.start, edit.end) == ("snv", 10, 11)
        assert edit.alt == "G" and not edit.is_length_changing
        assert edit.kind == EDIT_REPLACEMENT

    def test_multi_base_substitution_is_an_mnv(self):
        edit = classify_alt(10, "ACG", "TTT", 1)
        assert edit.variant_class == "mnv"
        assert (edit.start, edit.end, edit.alt) == (10, 13, "TTT")
        assert edit.ref_length == 3 and edit.length_change == 0

    def test_shared_flanks_are_trimmed_off(self):
        # Only the middle base changes, so this normalises to an SNV -
        # and, critically, to the *middle* coordinate.
        edit = classify_alt(10, "GTA", "GCA", 1)
        assert (edit.variant_class, edit.start, edit.end) == ("snv", 11, 12)
        assert edit.alt == "C"

    def test_deletion_drops_the_anchor_base(self):
        edit = classify_alt(10, "ATTT", "A", 1)
        assert (edit.variant_class, edit.start, edit.end) == ("del", 11, 14)
        assert edit.alt == "" and edit.length_change == -3
        assert edit.kind == EDIT_DELETION

    def test_insertion_is_a_zero_length_interval(self):
        edit = classify_alt(10, "A", "AGGG", 1)
        assert (edit.variant_class, edit.start, edit.end) == ("ins", 11, 11)
        assert edit.alt == "GGG" and edit.length_change == 3
        assert edit.kind == EDIT_INSERTION

    def test_replacement_is_routed_by_which_way_it_changes_length(self):
        assert classify_alt(10, "ACGT", "TT", 1).variant_class == "del"
        assert classify_alt(10, "AC", "TTTT", 1).variant_class == "ins"

    def test_symbolic_deletion_uses_end_and_skips_the_anchor(self):
        # POS is the base *before* the event, so the deleted interval
        # runs from POS+1 through END inclusive.
        edit = classify_alt(100, "G", "<DEL>", 1, symbolic_end=105)
        assert (edit.variant_class, edit.start, edit.end) == (
            "symbolic-del", 101, 106,
        )
        assert edit.length_change == -5

    def test_symbolic_deletion_without_end_is_refused(self):
        with pytest.raises(UnsupportedVariant, match="END"):
            classify_alt(100, "G", "<DEL>", 1)

    @pytest.mark.parametrize(
        "alt", ["<INS>", "<DUP>", "<INV>", "<NON_REF>", "<CNV>"]
    )
    def test_other_symbolic_alleles_are_refused(self, alt):
        with pytest.raises(UnsupportedVariant):
            classify_alt(100, "G", alt, 1, symbolic_end=120)

    def test_spanning_deletion_star_is_refused(self):
        with pytest.raises(UnsupportedVariant):
            classify_alt(100, "G", "*", 1)

    @pytest.mark.parametrize("alt", ["G]chr2:321]", "]chr2:321]G", "G."])
    def test_breakends_are_refused(self, alt):
        with pytest.raises(UnsupportedVariant):
            classify_alt(100, "G", alt, 1)

    def test_non_acgt_bases_are_refused(self):
        with pytest.raises(UnsupportedVariant):
            classify_alt(100, "G", "N", 1)
        with pytest.raises(UnsupportedVariant):
            classify_alt(100, "R", "G", 1)

    def test_identical_ref_and_alt_is_refused(self):
        with pytest.raises(UnsupportedVariant):
            classify_alt(100, "ACG", "ACG", 1)


class TestStructuralEdits:
    def _edits(self):
        return [
            (0, Edit(11, 14, "", "del", 1)),
            (1, Edit(21, 21, "GGG", "ins", 1)),
            (1, Edit(31, 36, "", "symbolic-del", 2)),
        ]

    def test_arrays_line_up_with_the_edits(self):
        pairs = self._edits()
        positions = np.array([10, 20], dtype=np.int32)
        genotypes = np.zeros((2, 4), dtype=np.uint8)
        built = build_structural_edits(pairs, positions, genotypes)

        assert built.n_edits == 3 and built.n_records == 2
        np.testing.assert_array_equal(built.start, [11, 21, 31])
        np.testing.assert_array_equal(built.end, [14, 21, 36])
        np.testing.assert_array_equal(built.record, [0, 1, 1])
        np.testing.assert_array_equal(built.allele, [1, 1, 2])
        np.testing.assert_array_equal(built.insert_length, [0, 3, 0])

    def test_deleted_bases_counts_only_what_is_removed(self):
        built = build_structural_edits(
            self._edits(),
            np.array([10, 20], dtype=np.int32),
            np.zeros((2, 4), dtype=np.uint8),
        )
        # 3 from the deletion, 5 from the symbolic deletion; the
        # insertion removes nothing.
        assert built.deleted_bases == 8

    def test_insert_pool_is_one_hot_and_offsets_point_into_it(self):
        built = build_structural_edits(
            self._edits(),
            np.array([10, 20], dtype=np.int32),
            np.zeros((2, 4), dtype=np.uint8),
        )
        pool = built.insert_onehot(np.dtype(np.float32))
        assert pool.shape == (3, 4) and pool.dtype == np.float32
        np.testing.assert_array_equal(pool.sum(axis=1), np.ones(3))
        offset = int(built.insert_offset[1])
        # 'GGG' - the G channel is index 2.
        np.testing.assert_array_equal(
            pool[offset:offset + 3].argmax(axis=1), [2, 2, 2]
        )

    def test_no_edits_is_representable(self):
        built = build_structural_edits(
            [], np.empty(0, dtype=np.int32), np.zeros((0, 4), dtype=np.uint8)
        )
        assert built.n_edits == 0 and built.deleted_bases == 0
        assert built.insert_onehot(np.dtype(np.float32)).shape == (0, 4)
