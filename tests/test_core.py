import pytest
import shutil
import numpy as np

from hottools.core import (
    _gt_str_to_dosage,
    _gt_str_to_haps,
    estimate_onehot_bytes,
    plan_batch_size_from_budget,
    load_reference_window,
    save_onehot,
    normalize_chrom_to_vcf
)


@pytest.mark.parametrize(
    "gt, expected",
    [
        ("0|0", 0.0),
        ("0/0", 0.0),
        ("0|1", 0.5),
        ("1|0", 0.5),
        ("0/1", 0.5),
        ("1/1", 1.0),
        ("1|1", 1.0),
        ("./.", 0.0),
        (".|.", 0.0),
        (".", 0.0),
    ],
)
def test_gt_str_to_dosage(gt, expected):
    assert _gt_str_to_dosage(gt) == expected


@pytest.mark.parametrize(
    "gt, expected",
    [
        ("0|0", (0.0, 0.0)),
        ("0|1", (0.0, 1.0)),
        ("1|0", (1.0, 0.0)),
        ("1|1", (1.0, 1.0)),
        ("./.", (0.0, 0.0)),
        (".|.", (0.0, 0.0)),
        # Intended behavior in Hottools: unphased genotypes cannot be split into haplotypes
        ("0/1", (0.0, 0.0)),
        ("1/0", (0.0, 0.0)),
    ],
)
def test_gt_str_to_haps(gt, expected):
    assert _gt_str_to_haps(gt) == expected


def test_load_reference_window(tmp_path):
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\nACGTACGTACGTACGT\n")

    seq = load_reference_window(str(fa), chrom="chr1", start_1based=5, end_1based=8)
    # positions 5..8 in ACGTACGT... => ACGT
    assert "".join(seq.tolist()) == "ACGT"
    assert seq.shape == (4,)


def test_estimate_onehot_bytes_float32_avg():
    # 1 sample, length 10, 4 channels, float32(4 bytes): 1 * 10 * 4 * 4 = 160
    assert estimate_onehot_bytes(1, 10, "float32", "average") == 1 * 1 * 10 * 4 * 4


def test_estimate_onehot_bytes_float16_separate():
    # Separate haplotypes -> doubles "sample dimension"
    # 3 samples * 2 haps * length 10 * 4 channels * float16(2 bytes)
    assert estimate_onehot_bytes(3, 10, "float16", "separate") == 3 * 2 * 10 * 4 * 2


def test_plan_batch_size_with_tiny_budget_still_valid():
    bs = plan_batch_size_from_budget(
        num_samples=100,
        seq_len=1000,
        dtype="float32",
        haplotype_mode="average",
        max_memory_gb=0.0005,
        overhead_factor=2.0,
    )
    assert 1 <= bs <= 100


def test_plan_batch_size_decreases_with_overhead():
    bs1 = plan_batch_size_from_budget(
        100, 1000, "float32", "average", max_memory_gb=0.5, overhead_factor=1.0
    )
    bs2 = plan_batch_size_from_budget(
        100, 1000, "float32", "average", max_memory_gb=0.5, overhead_factor=3.0
    )
    assert bs2 <= bs1


def test_save_onehot_combined_npy(tmp_path):
    x = np.zeros((2, 8, 4), dtype=np.float32)
    x[0, 0, 0] = 1.0

    save_onehot(
        seq=x,
        sample_ids=["S1", "S2"],
        out_dir=str(tmp_path),
        prefix="toy",
        fmt="npy",
        per_sample=False,
    )

    out = tmp_path / "toy.onehot.npy"
    assert out.exists()

    y = np.load(out)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert np.allclose(y, x)


def test_save_onehot_per_sample_npy(tmp_path):
    x = np.zeros((2, 8, 4), dtype=np.float32)
    save_onehot(x, ["S1", "S2"], str(tmp_path), "toy", fmt="npy", per_sample=True)

    assert (tmp_path / "toy.S1.onehot.npy").exists()
    assert (tmp_path / "toy.S2.onehot.npy").exists()


def test_normalize_chrom_to_vcf_accepts_chr_and_nochr(chr_prefix_vcf, no_chr_prefix_vcf):
    if shutil.which("bcftools") is None:
        pytest.skip("bcftools not installed")

    from hottools.core import normalize_chrom_to_vcf

    # VCF uses chr-prefix: both inputs should normalize to chr12
    assert normalize_chrom_to_vcf("12", str(chr_prefix_vcf)) == "chr12"
    assert normalize_chrom_to_vcf("chr12", str(chr_prefix_vcf)) == "chr12"

    # VCF does not use chr-prefix: both inputs should normalize to 12
    assert normalize_chrom_to_vcf("12", str(no_chr_prefix_vcf)) == "12"
    assert normalize_chrom_to_vcf("chr12", str(no_chr_prefix_vcf)) == "12"