"""Regression cases for bcftools' per-haplotype, first-applicable-ALT rule."""
import numpy as np
import pytest
from sv_helpers import Rec, run_sv


@pytest.mark.parametrize("records,bases", [
    ([("C", "1|0"), ("G", "0|1")], "CG"),
    ([("C", "1|0"), ("G", "1|0")], "CA"),
    ([("G", "1|0"), ("C", "1|0")], "GA"),
    ([("C", "1|0"), ("C", "1|0")], "CA"),
    ([("C", "0|0"), ("G", "1|0")], "GA"),
    ([("C", "0|1"), ("G", "1|0"), ("T", "1|1")], "GC"),
    ([("C,G", "1|2"), ("T", "1|1")], "CG"),
    ([("C", "./."), ("G", "1|0")], "GA"),
])
@pytest.mark.parametrize("mode", ["haplotypes", "dosage"])
def test_duplicate_first_applied_alt(tmp_path, records, bases, mode):
    variants = [Rec(7, "A", alt, (gt,) * 4) for alt, gt in records]
    tensor, metadata = run_sv(tmp_path, "A" * 300, variants,
                              mode=mode, variant_classes="snv", missing_policy="reference")
    expected = np.eye(4)[["ACGT".index(base) for base in bases]]
    if mode == "dosage":
        np.testing.assert_array_equal(tensor[:, 6], np.broadcast_to(expected.mean(axis=0), (4, 4)))
    else:
        np.testing.assert_array_equal(tensor[:, :, 6], np.broadcast_to(expected, (4, 2, 4)))
    assert metadata["variant_duplicate_positions_removed"] == 0
    assert metadata["variant_duplicate_positions_resolved"] == 1
