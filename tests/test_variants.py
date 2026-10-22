import numpy as np
import pytest
from conftest import REF_LENGTH, SAMPLES

from hottools.constants import (
    ALLELE_ABSENT,
    ALLELE_MISSING,
    pack_genotype,
)
from hottools.variants import (
    EmptyRegionWarning,
    HaploidWarning,
    MultiallelicWarning,
    extract_variants,
    resolve_sample_selection,
)

REF, ALT, MISS, ABS = 0, 1, int(ALLELE_MISSING), int(ALLELE_ABSENT)


def gt(a1, a2):
    """Packed genotype code, for readable expectations."""
    return int(pack_genotype(a1, a2))


def extract(vcf_path, reference_sequence, **overrides):
    kwargs = dict(
        vcf_path=vcf_path,
        chrom="chr1",
        start=1,
        end=REF_LENGTH,
        reference_seq=reference_sequence,
        mode="dosage",
    )
    kwargs.update(overrides)
    return extract_variants(**kwargs)


# --------------------------------------------------------------------
# sample selection
# --------------------------------------------------------------------

def test_resolve_sample_selection_passthrough():
    names, order = resolve_sample_selection(["a", "b"], None)
    assert names == ["a", "b"] and order is None


def test_resolve_sample_selection_reorders():
    names, order = resolve_sample_selection(["a", "b", "c"], ["c", "a"])
    assert names == ["c", "a"]
    assert order == [2, 0]


def test_resolve_sample_selection_rejects_missing():
    with pytest.raises(ValueError, match="not present in the VCF"):
        resolve_sample_selection(["a", "b"], ["a", "zz"])


def test_resolve_sample_selection_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate sample"):
        resolve_sample_selection(["a", "b"], ["a", "a"])


def test_absent_sample_is_an_error(clean_vcf_path, reference_sequence):
    with pytest.raises(ValueError, match="not present in the VCF"):
        extract(clean_vcf_path, reference_sequence, samples=["S1", "NOPE"])


def test_requested_sample_order_is_honoured(
    clean_vcf_path, reference_sequence, variant_spec
):
    """Row order must follow the request, not the VCF header order."""
    full = extract(clean_vcf_path, reference_sequence)
    subset = extract(
        clean_vcf_path, reference_sequence, samples=["S3", "S1"]
    )
    assert subset.samples == ["S3", "S1"]

    pos = variant_spec["clean_phased"]["pos"]
    row = int(np.flatnonzero(full.positions == pos)[0])
    expected = [
        full.genotypes[row, SAMPLES.index("S3")],
        full.genotypes[row, SAMPLES.index("S1")],
    ]
    np.testing.assert_array_equal(subset.genotypes[row], expected)


def test_sample_order_honoured_in_haplotype_mode(
    clean_vcf_path, reference_sequence
):
    full = extract(clean_vcf_path, reference_sequence, mode="haplotypes")
    subset = extract(
        clean_vcf_path, reference_sequence, mode="haplotypes",
        samples=["S4", "S2"],
    )
    assert subset.samples == ["S4", "S2"]
    np.testing.assert_array_equal(subset.genotypes[:, 0], full.genotypes[:, 3])
    np.testing.assert_array_equal(subset.genotypes[:, 1], full.genotypes[:, 1])


# --------------------------------------------------------------------
# filtering
# --------------------------------------------------------------------

def test_full_filter_accounting(vcf_path, reference_sequence, variant_spec):
    data = extract(
        vcf_path,
        reference_sequence,
        ambiguous_ref_policy="skip",
        ref_mismatch_policy="skip",
    )
    stats = data.stats.as_dict()
    assert stats["records_seen"] == 8
    assert stats["non_snv_records"] == 1     # the insertion at pos 30
    assert stats["ambiguous_fasta"] == 1       # FASTA 'N' at pos 50
    assert stats["ref_mismatch"] == 1          # pos 60
    assert stats["duplicate_records"] == 1     # second record at pos 40
    assert stats["duplicate_positions_removed"] == 0
    assert stats["duplicate_positions_resolved"] == 1
    assert stats["outside_region"] == 0
    assert stats["haploid_records"] == 0
    assert stats["multiallelic_sites_partial"] == 0
    assert stats["records_genotyped"] == 5
    assert stats["kept"] == 4
    np.testing.assert_array_equal(
        data.positions,
        [variant_spec["clean_phased"]["pos"],
         variant_spec["unphased_het"]["pos"],
         variant_spec["duplicate"]["pos"],
         variant_spec["clean_phased_2"]["pos"]],
    )
    assert data.samples == SAMPLES
    assert data.genotypes.shape == (4, 4)
    assert data.genotypes.dtype == np.uint8
    assert data.allele_bases.shape == (4, 4)


def test_duplicate_position_keeps_first_applicable_alt_per_haplotype(vcf_path, reference_sequence):
    data = extract(
        vcf_path, reference_sequence,
        ambiguous_ref_policy="skip", ref_mismatch_policy="skip",
    )
    row = data.positions.tolist().index(40)
    np.testing.assert_array_equal(data.genotypes[row], [gt(ALT, ALT)] * 4)


def test_ambiguous_reference_errors_by_default(vcf_path, reference_sequence):
    with pytest.raises(ValueError, match="Ambiguous FASTA base in chr1:50"):
        extract(vcf_path, reference_sequence, ref_mismatch_policy="skip")


def test_ref_mismatch_errors_by_default(vcf_path, reference_sequence):
    with pytest.raises(ValueError, match="REF mismatch at chr1:60"):
        extract(vcf_path, reference_sequence, ambiguous_ref_policy="skip")


@pytest.mark.parametrize("policy", ["skip", "warn"])
def test_record_policies_keep_going(vcf_path, reference_sequence, policy):
    with pytest.warns() if policy == "warn" else _nullcontext():
        data = extract(
            vcf_path, reference_sequence,
            ambiguous_ref_policy=policy, ref_mismatch_policy=policy,
        )
    assert data.stats.kept == 4


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_invalid_policy_rejected_up_front(vcf_path, reference_sequence):
    with pytest.raises(ValueError, match="missing_policy must be one of"):
        extract(vcf_path, reference_sequence, missing_policy="nonsense")


def test_lowercase_vcf_alleles_are_accepted(
    lowercase_vcf_path, reference_sequence, variant_spec
):
    """bcftools applies lower-case REF/ALT; hottools must not drop them."""
    data = extract(lowercase_vcf_path, reference_sequence)
    assert data.stats.lowercase_alleles == 1
    assert data.stats.non_snv_records == 0
    assert data.stats.kept == 2
    assert data.positions.tolist() == [
        variant_spec["clean_phased"]["pos"],
        variant_spec["clean_phased_2"]["pos"],
    ]
    row = int(np.flatnonzero(
        data.positions == variant_spec["clean_phased"]["pos"]
    )[0])
    np.testing.assert_array_equal(
        data.genotypes[row],
        [gt(REF, REF), gt(REF, ALT), gt(ALT, ALT), gt(MISS, MISS)],
    )


def test_window_subset_excludes_outside_variants(
    vcf_path, reference_sequence, variant_spec
):
    sub = reference_sequence[14:34]  # 1-based 15..34
    data = extract_variants(
        vcf_path=vcf_path, chrom="chr1", start=15, end=34,
        reference_seq=sub, mode="dosage",
    )
    assert data.positions.tolist() == [variant_spec["unphased_het"]["pos"]]


def test_reference_length_mismatch_is_rejected(vcf_path, reference_sequence):
    with pytest.raises(ValueError, match="reference_seq length"):
        extract_variants(
            vcf_path=vcf_path, chrom="chr1", start=1, end=50,
            reference_seq=reference_sequence, mode="dosage",
        )


# --------------------------------------------------------------------
# genotype decoding
# --------------------------------------------------------------------

def test_dosage_genotype_codes(clean_vcf_path, reference_sequence, variant_spec):
    data = extract(clean_vcf_path, reference_sequence)
    row = int(np.flatnonzero(
        data.positions == variant_spec["clean_phased"]["pos"]
    )[0])
    np.testing.assert_array_equal(
        data.genotypes[row],
        [gt(REF, REF), gt(REF, ALT), gt(ALT, ALT), gt(MISS, MISS)],
    )
    assert data.stats.missing_genotypes == 1


def test_missing_policy_reference_rewrites_to_hom_ref(
    clean_vcf_path, reference_sequence, variant_spec
):
    data = extract(
        clean_vcf_path, reference_sequence, missing_policy="reference"
    )
    row = int(np.flatnonzero(
        data.positions == variant_spec["clean_phased"]["pos"]
    )[0])
    np.testing.assert_array_equal(
        data.genotypes[row],
        [gt(REF, REF), gt(REF, ALT), gt(ALT, ALT), gt(MISS, MISS)],
    )


def test_missing_policy_error(clean_vcf_path, reference_sequence):
    with pytest.raises(ValueError, match="missing allele"):
        extract(clean_vcf_path, reference_sequence, missing_policy="error")


def test_half_call_keeps_the_called_allele(
    halfcall_vcf_path, reference_sequence, variant_spec
):
    """'.|1' must not collapse to UNKNOWN and lose its ALT allele."""
    data = extract(halfcall_vcf_path, reference_sequence)
    row = int(np.flatnonzero(
        data.positions == variant_spec["clean_phased"]["pos"]
    )[0])
    np.testing.assert_array_equal(
        data.genotypes[row],
        [gt(MISS, ALT), gt(ALT, MISS), gt(MISS, REF), gt(MISS, MISS)],
    )
    assert data.stats.half_calls == 3
    assert data.stats.missing_genotypes == 1


def test_half_call_under_reference_policy(
    halfcall_vcf_path, reference_sequence, variant_spec
):
    data = extract(
        halfcall_vcf_path, reference_sequence, missing_policy="reference"
    )
    row = int(np.flatnonzero(
        data.positions == variant_spec["clean_phased"]["pos"]
    )[0])
    # The stored codes keep the truth; missing_policy is applied by the
    # encoder's lookup table, not by rewriting the genotypes.
    np.testing.assert_array_equal(
        data.genotypes[row],
        [gt(MISS, ALT), gt(ALT, MISS), gt(MISS, REF), gt(MISS, MISS)],
    )


def test_half_call_counters_agree_across_modes(
    halfcall_vcf_path, reference_sequence
):
    dosage = extract(halfcall_vcf_path, reference_sequence)
    haps = extract(halfcall_vcf_path, reference_sequence, mode="haplotypes")
    assert dosage.stats.half_calls == haps.stats.half_calls == 3
    assert dosage.stats.missing_genotypes == haps.stats.missing_genotypes == 1


def test_half_call_errors_under_missing_policy_error(
    halfcall_vcf_path, reference_sequence
):
    with pytest.raises(ValueError, match="missing allele"):
        extract(
            halfcall_vcf_path, reference_sequence, missing_policy="error"
        )


def test_haplotype_alleles_and_phase(
    clean_vcf_path, reference_sequence, variant_spec
):
    data = extract(clean_vcf_path, reference_sequence, mode="haplotypes")
    row = int(np.flatnonzero(
        data.positions == variant_spec["clean_phased"]["pos"]
    )[0])
    np.testing.assert_array_equal(
        data.genotypes[row],
        [gt(REF, REF), gt(REF, ALT), gt(ALT, ALT), gt(MISS, MISS)],
    )
    assert data.payload is data.genotypes


def test_unphased_het_errors_in_haplotype_mode(vcf_path, reference_sequence):
    with pytest.raises(ValueError, match="unphased heterozygous"):
        extract(
            vcf_path, reference_sequence, mode="haplotypes",
            ambiguous_ref_policy="skip", ref_mismatch_policy="skip",
        )


def test_unphased_policy_missing(vcf_path, reference_sequence, variant_spec):
    data = extract(
        vcf_path, reference_sequence, mode="haplotypes",
        unphased_policy="missing",
        ambiguous_ref_policy="skip", ref_mismatch_policy="skip",
    )
    row = int(np.flatnonzero(
        data.positions == variant_spec["unphased_het"]["pos"]
    )[0])
    # S2 is 0/1 unphased -> masked; S1/S3 are homozygous and survive.
    assert data.genotypes[row, 1] == gt(MISS, MISS)
    assert data.genotypes[row, 0] == gt(REF, REF)
    assert data.stats.unphased_heterozygotes == 1


def test_unphased_policy_missing_with_reference_fill(
    vcf_path, reference_sequence, variant_spec
):
    data = extract(
        vcf_path, reference_sequence, mode="haplotypes",
        unphased_policy="missing", missing_policy="reference",
        ambiguous_ref_policy="skip", ref_mismatch_policy="skip",
    )
    row = int(np.flatnonzero(
        data.positions == variant_spec["unphased_het"]["pos"]
    )[0])
    assert data.genotypes[row, 1] == gt(MISS, MISS)


# --------------------------------------------------------------------
# ploidy, multiallelic sites, empty regions
# --------------------------------------------------------------------

def test_haploid_call_is_kept_and_announced(
    haploid_vcf_path, reference_sequence, variant_spec
):
    """A hemizygous call must survive, and the user must be told."""
    with pytest.warns(HaploidWarning, match="haploid genotype"):
        data = extract(haploid_vcf_path, reference_sequence, mode="haplotypes")
    row = int(np.flatnonzero(
        data.positions == variant_spec["clean_phased"]["pos"]
    )[0])
    assert data.genotypes[row, 0] == gt(REF, ABS)
    assert data.stats.haploid_genotypes == 2
    assert data.stats.haploid_records == 2
    assert data.sample_ploidy.tolist() == [1, 2, 2, 2]


def test_haploid_is_distinct_from_a_half_call(
    haploid_vcf_path, halfcall_vcf_path, reference_sequence, variant_spec
):
    """The design trap: hemizygous must not look like a half-call."""
    pos = variant_spec["clean_phased"]["pos"]
    with pytest.warns(HaploidWarning):
        hap = extract(haploid_vcf_path, reference_sequence)
    half = extract(halfcall_vcf_path, reference_sequence)
    r1 = int(np.flatnonzero(hap.positions == pos)[0])
    r2 = int(np.flatnonzero(half.positions == pos)[0])
    assert hap.genotypes[r1, 0] == gt(REF, ABS)
    assert half.genotypes[r2, 0] == gt(MISS, ALT)
    assert hap.genotypes[r1, 0] != half.genotypes[r2, 0]
    assert hap.stats.haploid_genotypes == 2 and hap.stats.half_calls == 0
    assert half.stats.half_calls == 3 and half.stats.haploid_genotypes == 0


def test_haploid_policy_error(haploid_vcf_path, reference_sequence):
    with pytest.raises(ValueError, match="haploid genotype"):
        extract(
            haploid_vcf_path, reference_sequence, mode="haplotypes",
            haploid_policy="error",
        )


def test_haploid_policy_skip(haploid_vcf_path, reference_sequence):
    data = extract(
        haploid_vcf_path, reference_sequence, mode="haplotypes",
        haploid_policy="skip",
    )
    assert data.stats.kept == 0


def test_polyploid_record_is_rejected(triploid_vcf_path, reference_sequence):
    with pytest.raises(ValueError, match="ploidy > 2"):
        extract(triploid_vcf_path, reference_sequence, mode="haplotypes")


def test_polyploid_policy_skip(triploid_vcf_path, reference_sequence):
    with pytest.warns(RuntimeWarning, match="ploidy > 2"):
        data = extract(
            triploid_vcf_path, reference_sequence, mode="haplotypes",
            ploidy_policy="skip",
        )
    assert data.stats.polyploid_records == 1
    assert data.stats.kept == 1


def test_split_multiallelic_site_is_reported(
    multiallelic_vcf_path, reference_sequence
):
    with pytest.warns(MultiallelicWarning, match="chr1:80"):
        data = extract(multiallelic_vcf_path, reference_sequence)
    assert data.stats.multiallelic_sites_partial == 1
    assert 80 in data.positions.tolist()


def test_split_multiallelic_site_errors_under_strict(
    multiallelic_vcf_path, reference_sequence
):
    with pytest.raises(ValueError, match="split multiallelic"):
        extract(multiallelic_vcf_path, reference_sequence, strict=True)


def test_empty_region_warns(clean_vcf_path, reference_sequence):
    sub = reference_sequence[149:200]
    with pytest.warns(EmptyRegionWarning, match="No usable variants"):
        data = extract_variants(
            vcf_path=clean_vcf_path, chrom="chr1", start=150, end=200,
            reference_seq=sub, mode="dosage",
        )
    assert data.stats.kept == 0


def test_empty_region_errors_under_strict(clean_vcf_path, reference_sequence):
    sub = reference_sequence[149:200]
    with pytest.raises(ValueError, match="No usable variants"):
        extract_variants(
            vcf_path=clean_vcf_path, chrom="chr1", start=150, end=200,
            reference_seq=sub, mode="dosage", strict=True,
        )


def test_contig_naming_mismatch_is_caught(nochr_vcf_path, reference_sequence):
    with pytest.raises(ValueError, match="Did you mean '1'"):
        extract(nochr_vcf_path, reference_sequence)
