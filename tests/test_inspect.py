import shutil
import subprocess
from collections import Counter
from pathlib import Path
import pytest

from hottools.inspect import (
    ensure_biallelic_snps,
    assert_phased_genotypes
)

# Helper functions 

def _require_bcftools():
    if shutil.which("bcftools") is None:
        pytest.skip("bcftools not installed / not in PATH")


def _bcftools_records(vcf_path: str, region: str | None = None):
    _require_bcftools()

    cmd = ["bcftools", "view", "-H"]
    if region:
        cmd += ["-r", region]
    cmd.append(vcf_path)

    p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True)

    recs = []
    for line in p.stdout.splitlines():
        f = line.split("\t")
        recs.append((f[0], int(f[1]), f[3], f[4]))
    return recs


def _assert_all_biallelic_snvs(recs):
    assert len(recs) > 0
    for chrom, pos, ref, alt in recs:
        assert "," not in alt
        assert len(ref) == 1
        assert len(alt) == 1


def _assert_no_duplicate_positions(recs):
    counts = Counter((chrom, pos) for chrom, pos, _, _ in recs)
    dups = [(k, c) for k, c in counts.items() if c > 1]
    assert not dups, f"Found duplicate positions: {dups}"


# Test phased genotypes
def test_assert_phased_genotypes_passes(phased_vcf):
    _require_bcftools()

    assert_phased_genotypes(
        vcf_path=str(phased_vcf),
        region="chr1:1-1000000",
        samples_file=None,
        n_check_variants=2000,
    )

def test_assert_phased_genotypes_raises(unphased_vcf):
    _require_bcftools()

    with pytest.raises(ValueError):
        assert_phased_genotypes(
            vcf_path=str(unphased_vcf),
            region="chr1:1-1000000",
            samples_file=None,
            n_check_variants=2000,
        )

# Ensure clean phased VCF file stays valid
def test_ensure_biallelic_snps_keeps_phased_clean_vcf(tmp_path, phased_vcf):
    _require_bcftools()

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    out_vcf = ensure_biallelic_snps(
        path=str(phased_vcf),
        out_dir=str(out_dir),
        region="chr1:1-1000000",
    )

    assert Path(out_vcf).exists()

    recs = _bcftools_records(out_vcf, region="chr1:1-1000000")
    _assert_all_biallelic_snvs(recs)
    _assert_no_duplicate_positions(recs)


# Ensure multiallelic filtered
def test_ensure_biallelic_snps_filters_multiallelic(tmp_path, multiallelic_vcf):
    _require_bcftools()

    in_recs = _bcftools_records(str(multiallelic_vcf), region="chr1:1-1000000")
    assert any("," in alt for _, _, _, alt in in_recs)

    out_vcf = ensure_biallelic_snps(
        path=str(multiallelic_vcf),
        out_dir=str(tmp_path),
        region="chr1:1-1000000",
    )

    recs = _bcftools_records(out_vcf, region="chr1:1-1000000")

    _assert_all_biallelic_snvs(recs)
    assert all("," not in alt for _, _, _, alt in recs)


# Ensure indels filtered
def test_ensure_biallelic_snps_filters_indels(tmp_path, indel_vcf):
    _require_bcftools()

    in_recs = _bcftools_records(str(indel_vcf), region="chr1:1-1000000")
    assert any(
        len(ref) != 1 or len(alt) != 1 or "," in alt
        for _, _, ref, alt in in_recs
    )

    out_vcf = ensure_biallelic_snps(
        path=str(indel_vcf),
        out_dir=str(tmp_path),
        region="chr1:1-1000000",
    )

    recs = _bcftools_records(out_vcf, region="chr1:1-1000000")
    _assert_all_biallelic_snvs(recs)

# Ensure duplicate positions are removed
def test_ensure_biallelic_snps_drops_all_duplicate_positions(tmp_path, duplicates_vcf):
    _require_bcftools()

    # Read input records and identify duplicated positions
    in_recs = _bcftools_records(str(duplicates_vcf), region="chr1:1-1000000")
    in_counts = Counter((chrom, pos) for chrom, pos, _, _ in in_recs)
    dup_positions = {k for k, c in in_counts.items() if c > 1}

    assert dup_positions, "duplicates_vcf fixture does not contain any duplicated positions to test"

    # Run filtering
    out_vcf = ensure_biallelic_snps(
        path=str(duplicates_vcf),
        out_dir=str(tmp_path),
        region="chr1:1-1000000",
    )
    assert Path(out_vcf).exists()

    # Read output records
    out_recs = _bcftools_records(out_vcf, region="chr1:1-1000000")
    _assert_all_biallelic_snvs(out_recs)
    _assert_no_duplicate_positions(out_recs)

    out_positions = {(chrom, pos) for chrom, pos, _, _ in out_recs}

    # Key assertion: duplicated positions are fully removed (0 occurrences in output)
    still_present = dup_positions & out_positions
    assert not still_present, f"Duplicated positions were not fully removed: {sorted(still_present)[:10]}"

# Ensure unsorted input raises error
def test_ensure_biallelic_snps_raises_on_unsorted_input(tmp_path, unsorted_vcf):
    _require_bcftools()

    with pytest.raises(ValueError, match=r"not coordinate-sorted|coordinate-sorted|bcftools sort"):
        ensure_biallelic_snps(
            path=str(unsorted_vcf),
            out_dir=str(tmp_path),
            region="chr1:1-1000000",
        )
