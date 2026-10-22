"""
End-to-end coverage for the site filters, mask, multiallelic support,
consensus mode and haploid padding.
"""
import json

import numpy as np
import pytest
from conftest import BASES, REF_LENGTH

from hottools.pipeline import RunConfig, run_hottools
from hottools.regions import Region, read_mask
from hottools.variants import HaploidWarning, extract_variants

REGION = Region("chr1", 1, REF_LENGTH)


def run(tmp_path, vcf, fasta, name="out.npy", **overrides):
    kwargs = dict(
        vcf_path=str(vcf), fasta_path=str(fasta), region=REGION,
        output_path=str(tmp_path / name), batch_size="none",
    )
    kwargs.update(overrides)
    return run_hottools(RunConfig(**kwargs))


def decode(track):
    """One-hot L x 4 -> sequence string; all-zero -> 'N'."""
    sums = track.sum(axis=1)
    out = np.full(track.shape[0], "?", dtype="<U1")
    hot = np.isclose(sums, 1.0)
    out[hot] = np.asarray(list(BASES))[track[hot].argmax(axis=1)]
    out[np.isclose(sums, 0.0)] = "N"
    return "".join(out.tolist())


# ------------------------------------------------------------------
# site filters: FILTER, QUAL, AF
# ------------------------------------------------------------------

def test_no_filters_keeps_every_record(tmp_path, filtered_vcf_path, fasta_path):
    meta = run(tmp_path, filtered_vcf_path, fasta_path, dry_run=True)
    assert meta["variants_kept"] == 3


def test_pass_only_drops_the_failing_record(
    tmp_path, filtered_vcf_path, fasta_path
):
    meta = run(
        tmp_path, filtered_vcf_path, fasta_path, dry_run=True, pass_only=True
    )
    assert meta["variants_kept"] == 2
    assert meta["variant_filtered_not_pass"] == 1
    # the LowQual record at position 20 is the one dropped
    assert 20 not in _kept_positions(filtered_vcf_path, fasta_path, pass_only=True)


def _kept_positions(vcf, fasta, **kwargs):
    from hottools.reference import read_reference_interval

    seq = read_reference_interval(fasta, "chr1", 1, REF_LENGTH)
    data = extract_variants(
        vcf_path=vcf, chrom="chr1", start=1, end=REF_LENGTH,
        reference_seq=seq, mode="dosage", **kwargs,
    )
    return data.positions.tolist()


def test_min_qual(tmp_path, filtered_vcf_path, fasta_path):
    meta = run(
        tmp_path, filtered_vcf_path, fasta_path, dry_run=True, min_qual=40
    )
    assert meta["variants_kept"] == 2
    assert meta["variant_filtered_low_qual"] == 1


def test_min_qual_is_inclusive(tmp_path, filtered_vcf_path, fasta_path):
    meta = run(
        tmp_path, filtered_vcf_path, fasta_path, dry_run=True, min_qual=50
    )
    assert meta["variants_kept"] == 2   # QUAL 50 and 99 survive


def test_min_af_uses_info_field(tmp_path, filtered_vcf_path, fasta_path):
    meta = run(
        tmp_path, filtered_vcf_path, fasta_path, dry_run=True, min_af=0.1
    )
    assert meta["variants_kept"] == 2          # AF 0.40 and 0.90
    assert meta["variant_filtered_allele_frequency"] == 1


def test_max_af_uses_info_field(tmp_path, filtered_vcf_path, fasta_path):
    meta = run(
        tmp_path, filtered_vcf_path, fasta_path, dry_run=True, max_af=0.5
    )
    assert meta["variants_kept"] == 2          # AF 0.40 and 0.02


def test_af_band(tmp_path, filtered_vcf_path, fasta_path):
    meta = run(
        tmp_path, filtered_vcf_path, fasta_path, dry_run=True,
        min_af=0.1, max_af=0.5,
    )
    assert meta["variants_kept"] == 1


def test_af_from_samples_ignores_the_info_field(
    tmp_path, filtered_vcf_path, fasta_path
):
    """INFO/AF says 0.02 at position 20, but the samples say 0.125."""
    meta = run(
        tmp_path, filtered_vcf_path, fasta_path, dry_run=True,
        min_af=0.1, af_source="samples",
    )
    assert meta["variants_kept"] == 3


def test_af_source_info_errors_without_the_field(
    tmp_path, clean_vcf_path, fasta_path
):
    with pytest.raises(ValueError, match="no INFO/AF field"):
        run(
            tmp_path, clean_vcf_path, fasta_path, dry_run=True,
            min_af=0.1, af_source="info",
        )


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_af_bounds_are_validated(tmp_path, filtered_vcf_path, fasta_path, bad):
    with pytest.raises(ValueError, match="between 0 and 1"):
        run(tmp_path, filtered_vcf_path, fasta_path, dry_run=True, min_af=bad)


def test_filters_compose(tmp_path, filtered_vcf_path, fasta_path):
    meta = run(
        tmp_path, filtered_vcf_path, fasta_path, dry_run=True,
        pass_only=True, min_qual=60, min_af=0.5,
    )
    assert meta["variants_kept"] == 1          # only the QUAL 99 / AF 0.90 one


# ------------------------------------------------------------------
# mask file
# ------------------------------------------------------------------

def test_read_mask_bed_is_zero_based(tmp_path):
    bed = tmp_path / "m.bed"
    bed.write_text("chr1\t0\t10\nchr1\t20\t25\n")
    mask = read_mask(bed, Region("chr1", 1, 30))
    assert mask[:10].all() and not mask[10:20].any()
    assert mask[20:25].all() and not mask[25:].any()
    assert mask.sum() == 15


def test_read_mask_ignores_other_contigs_and_clips(tmp_path):
    bed = tmp_path / "m.bed"
    bed.write_text("chr2\t0\t10\nchr1\t5\t1000\n")
    mask = read_mask(bed, Region("chr1", 1, 20))
    assert mask.sum() == 15          # 1-based 6..20
    assert not mask[:5].any()


def test_mask_zeroes_reference_and_drops_variants(
    tmp_path, clean_vcf_path, fasta_path, variant_spec
):
    pos = variant_spec["clean_phased"]["pos"]
    bed = tmp_path / "m.bed"
    bed.write_text(f"chr1\t{pos - 3}\t{pos + 2}\n")      # 0-based half-open
    meta = run(
        tmp_path, clean_vcf_path, fasta_path, mask_path=str(bed)
    )
    assert meta["masked_bases"] == 5
    assert meta["variant_masked_out"] == 1
    assert meta["variants_kept"] == 1

    out = np.load(tmp_path / "out.npy")
    masked = slice(pos - 3, pos + 2)
    assert np.all(out[:, masked, :] == 0)
    assert out[:, : pos - 3, :].sum() > 0


def test_mask_applies_to_every_sample(tmp_path, clean_vcf_path, fasta_path):
    bed = tmp_path / "m.bed"
    bed.write_text("chr1\t0\t20\n")
    run(tmp_path, clean_vcf_path, fasta_path, mask_path=str(bed))
    out = np.load(tmp_path / "out.npy")
    assert np.all(out[:, :20, :] == 0)


# ------------------------------------------------------------------
# multiallelic sites
# ------------------------------------------------------------------

def test_multiallelic_record_is_kept(
    tmp_path, triallelic_vcf_path, fasta_path, variant_spec
):
    meta = run(tmp_path, triallelic_vcf_path, fasta_path, dry_run=True)
    assert meta["variants_kept"] == 2
    assert meta["multiallelic_records"] == 1
    assert meta["variant_non_snv_records"] == 0


def test_multiallelic_haplotypes_end_to_end(
    tmp_path, triallelic_vcf_path, fasta_path, variant_spec, reference_sequence
):
    run(tmp_path, triallelic_vcf_path, fasta_path, mode="haplotypes")
    out = np.load(tmp_path / "out.npy")
    pos = variant_spec["clean_phased"]["pos"]
    col = pos - 1
    ref = reference_sequence[col]
    alt1, alt2 = [b for b in BASES if b != ref][:2]

    # samples are 0|1, 0|2, 1|2, 2|2
    assert decode(out[0, 0])[col] == ref
    assert decode(out[0, 1])[col] == alt1
    assert decode(out[1, 1])[col] == alt2
    assert decode(out[2, 0])[col] == alt1
    assert decode(out[2, 1])[col] == alt2
    assert decode(out[3, 0])[col] == alt2


def test_multiallelic_dosage_splits_across_two_alts(
    tmp_path, triallelic_vcf_path, fasta_path, variant_spec, reference_sequence
):
    run(tmp_path, triallelic_vcf_path, fasta_path, mode="dosage")
    out = np.load(tmp_path / "out.npy")
    col = variant_spec["clean_phased"]["pos"] - 1
    ref = reference_sequence[col]
    alt1, alt2 = [b for b in BASES if b != ref][:2]
    vec = out[2, col]            # sample 1|2
    assert vec[BASES.index(alt1)] == 0.5
    assert vec[BASES.index(alt2)] == 0.5
    assert vec[BASES.index(ref)] == 0.0


def test_too_many_alt_alleles_is_reported(
    tmp_path, quadallelic_vcf_path, fasta_path
):
    with pytest.warns(RuntimeWarning, match="ALT alleles"):
        meta = run(tmp_path, quadallelic_vcf_path, fasta_path, dry_run=True)
    assert meta["variant_too_many_alt_alleles"] == 1


# ------------------------------------------------------------------
# consensus mode
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "het_policy,expected", [("reference", "ref"), ("alternate", "alt")]
)
def test_consensus_het_policies_end_to_end(
    tmp_path, clean_vcf_path, fasta_path, variant_spec, reference_sequence,
    het_policy, expected,
):
    meta = run(
        tmp_path, clean_vcf_path, fasta_path, mode="consensus",
        het_policy=het_policy, missing_policy="reference",
    )
    assert meta["het_policy"] == het_policy
    out = np.load(tmp_path / "out.npy")
    assert out.shape == (4, REF_LENGTH, 4)

    spec = variant_spec["clean_phased"]
    col = spec["pos"] - 1
    want = spec["ref"] if expected == "ref" else spec["alt"]
    # sample 2 is the heterozygote
    assert decode(out[1])[col] == want


def test_consensus_output_is_strictly_one_hot(
    tmp_path, clean_vcf_path, fasta_path
):
    run(
        tmp_path, clean_vcf_path, fasta_path, mode="consensus",
        missing_policy="reference",
    )
    out = np.load(tmp_path / "out.npy")
    # the single ambiguous reference base stays all-zero by design
    from conftest import AMBIGUOUS_POS

    keep = [i for i in range(REF_LENGTH) if i != AMBIGUOUS_POS - 1]
    np.testing.assert_allclose(out[:, keep, :].sum(axis=2), 1.0)
    assert set(np.unique(out).tolist()) <= {0.0, 1.0}


def test_consensus_major_uses_cohort_frequency(
    tmp_path, clean_vcf_path, fasta_path, variant_spec
):
    """Genotypes are 0|0, 0|1, 1|1, .|. so REF and ALT are tied at 2:2."""
    run(
        tmp_path, clean_vcf_path, fasta_path, mode="consensus",
        het_policy="major", missing_policy="reference",
    )
    out = np.load(tmp_path / "out.npy")
    spec = variant_spec["clean_phased"]
    col = spec["pos"] - 1
    # a tie resolves to the lower allele index, which is REF
    assert decode(out[1])[col] == spec["ref"]


def test_consensus_shape_matches_dosage(tmp_path, clean_vcf_path, fasta_path):
    a = run(tmp_path, clean_vcf_path, fasta_path, mode="dosage", name="a.npy")
    b = run(
        tmp_path, clean_vcf_path, fasta_path, mode="consensus", name="b.npy"
    )
    assert a["output_shape"] == b["output_shape"]
    assert b["het_policy"] == "alternate"
    assert a["het_policy"] is None


# ------------------------------------------------------------------
# haploid support
# ------------------------------------------------------------------

def test_haploid_duplicate_is_the_default(
    tmp_path, haploid_vcf_path, fasta_path, variant_spec
):
    with pytest.warns(HaploidWarning, match="duplicated onto both"):
        meta = run(tmp_path, haploid_vcf_path, fasta_path, mode="haplotypes")
    out = np.load(tmp_path / "out.npy")
    col = variant_spec["clean_phased"]["pos"] - 1
    np.testing.assert_array_equal(out[0, 0, col], out[0, 1, col])
    assert meta["haploid_policy"] == "duplicate"
    assert meta["variant_haploid_genotypes"] == 2
    assert meta["sample_ploidy"] == [1, 2, 2, 2]


def test_haploid_blank_leaves_second_track_empty(
    tmp_path, haploid_vcf_path, fasta_path, variant_spec
):
    with pytest.warns(HaploidWarning, match="left blank"):
        run(
            tmp_path, haploid_vcf_path, fasta_path, mode="haplotypes",
            haploid_policy="blank",
        )
    out = np.load(tmp_path / "out.npy")
    col = variant_spec["clean_phased"]["pos"] - 1
    assert out[0, 0, col].sum() == 1.0
    assert out[0, 1, col].sum() == 0.0


def test_haploid_tensor_stays_rectangular(
    tmp_path, haploid_vcf_path, fasta_path
):
    """Both policies keep two tracks, so nothing downstream changes."""
    for policy in ("duplicate", "blank"):
        with pytest.warns(HaploidWarning):
            meta = run(
                tmp_path, haploid_vcf_path, fasta_path, mode="haplotypes",
                haploid_policy=policy, name=f"{policy}.npy",
            )
        assert meta["output_shape"] == [4, 2, REF_LENGTH, 4]


def test_haploid_message_reaches_the_sidecar(
    tmp_path, haploid_vcf_path, fasta_path
):
    with pytest.warns(HaploidWarning):
        run(tmp_path, haploid_vcf_path, fasta_path, mode="haplotypes")
    sidecar = json.loads((tmp_path / "out.npy.metadata.json").read_text())
    assert sidecar["variant_haploid_records"] == 2
    assert sidecar["sample_ploidy"] == [1, 2, 2, 2]
    assert sidecar["haploid_policy"] == "duplicate"


def test_haploid_dosage_duplicate_is_full_weight(
    tmp_path, haploid_vcf_path, fasta_path, variant_spec, reference_sequence
):
    with pytest.warns(HaploidWarning):
        run(tmp_path, haploid_vcf_path, fasta_path, mode="dosage")
    out = np.load(tmp_path / "out.npy")
    col = variant_spec["clean_phased"]["pos"] - 1
    # S1 is hemizygous REF here: full weight on the reference base
    assert out[0, col, BASES.index(reference_sequence[col])] == 1.0
