"""SNV-only selection must reject whole mixed or non-single-base records."""
import numpy as np
import pytest
from sv_helpers import Rec, run_sv


@pytest.mark.parametrize("ref,alt", [
    ("A", "C,AT"), ("A", "AT,C"), ("AA", "CA,A"),
    ("A", "C,<DEL>"), ("A", "C,<INS>"), ("A", "C,*"),
    ("A", "C,A]chr1:90]"), ("AA", "CA"), ("AA", "CG"),
])
@pytest.mark.parametrize("mode", ["haplotypes", "dosage"])
def test_reject_entire_non_snv_record(tmp_path, ref, alt, mode):
    # Even homozygous SNV calls cannot rescue a mixed record.
    records = [Rec(7, ref, alt, ("1|1", "0|1", "1|0", "0|0"))]
    tensor, _ = run_sv(tmp_path, "A" * 300, records,
                       variant_classes="snv", mode=mode)
    expected = np.zeros_like(tensor)
    expected[..., 0] = 1
    np.testing.assert_array_equal(tensor, expected)


@pytest.mark.parametrize("mode", ["haplotypes", "dosage"])
def test_rejected_mixed_record_does_not_block_valid_duplicate(tmp_path, mode):
    records = [Rec(7, "A", "C,AT", ("1|1",) * 4),
               Rec(7, "A", "G", ("1|1",) * 4)]
    tensor, meta = run_sv(tmp_path, "A" * 300, records,
                          variant_classes="snv", mode=mode)
    assert np.all(tensor[..., 6, 2] == 1)
    assert np.all(tensor[..., 6, [0, 1, 3]] == 0)
    assert meta["variant_non_snv_records"] == 1


def test_broader_mode_retains_mixed_allele_support(tmp_path):
    records = [Rec(7, "A", "C,AT", ("1|0",) * 4)]
    tensor, _ = run_sv(tmp_path, "A" * 300, records,
                       variant_classes="all", mode="haplotypes")
    assert np.all(tensor[:, 0, 6, 1] == 1)
    assert np.all(tensor[:, 1, 6, 0] == 1)
