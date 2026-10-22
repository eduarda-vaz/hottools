import numpy as np
import pytest
from sv_helpers import Rec, run_sv


@pytest.mark.parametrize(
    "alt,expected",
    [
        ("C", "C"),
        ("C,G", "C"),
        ("C,G,T", "C"),
        ("C,AT", "A"),
        ("C,<DEL>", "A"),
        ("C,*", "A"),
    ],
)
@pytest.mark.parametrize("biallelic", [False, True])
def test_whole_record_selection(tmp_path, alt, expected, biallelic):
    if biallelic and "," in alt:
        expected = "A"
    tensor, meta = run_sv(
        tmp_path,
        "A" * 300,
        [Rec(7, "A", alt, ("1|1",) * 4, "END=12" if "<DEL>" in alt else ".")],
        variant_classes="snv",
        biallelic_only=biallelic,
    )
    np.testing.assert_array_equal(
        tensor[:, :, 6], np.broadcast_to(np.eye(4)["ACGT".index(expected)], (4, 2, 4))
    )
    assert meta["biallelic_only"] == biallelic
    assert meta["mixed_record_policy"] == "exclude"
    if biallelic and "," in alt:
        assert meta["variant_filtered_multiallelic"] == 1


def test_explicit_partial_selection(tmp_path):
    tensor, meta = run_sv(
        tmp_path,
        "A" * 300,
        [Rec(7, "A", "C,AT", ("1|2",) * 4)],
        variant_classes="snv",
        mixed_record_policy="apply-selected",
    )
    np.testing.assert_array_equal(tensor[0, :, 6], np.eye(4)[[1, 0]])
    assert meta["mixed_record_policy"] == "apply-selected"


def test_invalid_record_policy(tmp_path):
    with pytest.raises(ValueError, match="mixed_record_policy"):
        run_sv(
            tmp_path,
            "A" * 300,
            [Rec(7, "A", "C", ("1|1",) * 4)],
            mixed_record_policy="invalid",
        )


@pytest.mark.parametrize("other", ["AT", "<DEL>", "<INS>", "*", "A]chr1:90]"])
@pytest.mark.parametrize("mode", ["dosage", "haplotypes"])
def test_explicit_snv_projection_ignores_other_alt(tmp_path, other, mode):
    tensor, _ = run_sv(
        tmp_path,
        "A" * 300,
        [
            Rec(
                7,
                "A",
                "C," + other,
                ("1|2",) * 4,
                "END=12" if other == "<DEL>" else ".",
            )
        ],
        variant_classes="snv",
        mixed_record_policy="apply-selected",
        mode=mode,
    )
    expected = np.eye(4)[[1, 0]]
    if mode == "dosage":
        np.testing.assert_array_equal(tensor[0, 6], expected.mean(axis=0))
    else:
        np.testing.assert_array_equal(tensor[0, :, 6], expected)
