"""
Regression tests for the defects found in review of 0.4.0.

Each test names the behaviour that was wrong, so a reappearance is
recognisable rather than just red.
"""
import gzip

import numpy as np
import pytest
from conftest import REF_LENGTH

from hottools.pipeline import RunConfig, run_hottools
from hottools.regions import Region, read_mask, read_region_file
from hottools.variants import HaploidWarning, resolve_sample_selection
from hottools.writers import (
    SampleNameCollisionWarning,
    check_destination,
    safe_sample_names,
)

REGION = Region("chr1", 1, REF_LENGTH)
COMP = [3, 2, 1, 0]


def run(tmp_path, vcf, fasta, name="out.npy", **overrides):
    kwargs = dict(
        vcf_path=str(vcf), fasta_path=str(fasta), region=REGION,
        output_path=str(tmp_path / name), batch_size="none",
        overwrite="replace",
    )
    kwargs.update(overrides)
    return run_hottools(RunConfig(**kwargs))


# ------------------------------------------------------------------
# 1. --af-source auto must use one source per record, not both
# ------------------------------------------------------------------

def test_af_auto_does_not_stack_both_filters(
    tmp_path, filtered_vcf_path, fasta_path
):
    """
    Position 10 carries INFO/AF=0.40 but a sample-derived frequency of
    0.50. With --max-af 0.45 the INFO value clears it and the
    sample-derived value would not, so stacking both filters drops a
    record that 'auto' should keep. 'auto' must agree with 'info' for
    every record that has an INFO/AF field.
    """
    auto = run(
        tmp_path, filtered_vcf_path, fasta_path, dry_run=True,
        max_af=0.45, af_source="auto",
    )
    info = run(
        tmp_path, filtered_vcf_path, fasta_path, dry_run=True,
        max_af=0.45, af_source="info",
    )
    samples = run(
        tmp_path, filtered_vcf_path, fasta_path, dry_run=True,
        max_af=0.45, af_source="samples",
    )
    assert auto["variants_kept"] == info["variants_kept"] == 2
    # The sample-derived frequencies genuinely differ, which is what
    # makes this test able to fail.
    assert samples["variants_kept"] == 1


def test_af_auto_falls_back_per_record(
    tmp_path, mixed_af_vcf_path, fasta_path
):
    """
    One record carries INFO/AF=0.90, the other has none and a
    sample-derived frequency of 0.125. At a 0.5 threshold 'auto' keeps
    the first on its INFO value and drops the second on its samples.
    """
    meta = run(
        tmp_path, mixed_af_vcf_path, fasta_path, dry_run=True,
        min_af=0.5, af_source="auto",
    )
    assert meta["variants_kept"] == 1
    assert meta["variant_filtered_allele_frequency"] == 1


def test_af_samples_ignores_info_entirely(
    tmp_path, mixed_af_vcf_path, fasta_path
):
    """Sample frequencies are 0.375 and 0.125; both fail at 0.5."""
    meta = run(
        tmp_path, mixed_af_vcf_path, fasta_path, dry_run=True,
        min_af=0.5, af_source="samples",
    )
    assert meta["variants_kept"] == 0


def test_af_info_requires_the_field(tmp_path, mixed_af_vcf_path, fasta_path):
    with pytest.raises(ValueError, match="no INFO/AF field"):
        run(
            tmp_path, mixed_af_vcf_path, fasta_path, dry_run=True,
            min_af=0.5, af_source="info",
        )


# ------------------------------------------------------------------
# 2. negative-strand consensus mode
# ------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["samples_first", "channels_first"])
def test_consensus_reverse_strand(tmp_path, clean_vcf_path, fasta_path, layout):
    """Consensus is single-track; it must not take the haplotype path."""
    run(
        tmp_path, clean_vcf_path, fasta_path, mode="consensus",
        output_layout=layout, missing_policy="reference", name="fwd.npy",
    )
    meta = run(
        tmp_path, clean_vcf_path, fasta_path, mode="consensus",
        output_layout=layout, missing_policy="reference",
        region=Region("chr1", 1, REF_LENGTH, "-"), name="rev.npy",
    )
    assert meta["reverse_complement"] is True
    fwd = np.load(tmp_path / "fwd.npy")
    rev = np.load(tmp_path / "rev.npy")
    if layout == "samples_first":
        np.testing.assert_array_equal(rev, fwd[:, ::-1, :][:, :, COMP])
    else:
        np.testing.assert_array_equal(rev, fwd[:, COMP, :][:, :, ::-1])


def test_consensus_reverse_strand_is_an_involution(
    tmp_path, clean_vcf_path, fasta_path
):
    from hottools.pipeline import _apply_reverse_complement

    batch = np.random.default_rng(0).random((3, 11, 4)).astype(np.float32)
    once = _apply_reverse_complement(
        batch, mode="consensus", layout="samples_first"
    )
    twice = _apply_reverse_complement(
        once, mode="consensus", layout="samples_first"
    )
    np.testing.assert_array_equal(twice, batch)


def test_reverse_complement_rejects_unknown_mode():
    from hottools.pipeline import _apply_reverse_complement

    with pytest.raises(ValueError, match="mode must be one of"):
        _apply_reverse_complement(
            np.zeros((1, 2, 4), dtype=np.float32),
            mode="nonsense", layout="samples_first",
        )


# ------------------------------------------------------------------
# 3. compressed region and mask files
# ------------------------------------------------------------------

def test_gzipped_bed_region_file(tmp_path):
    path = tmp_path / "r.bed.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("chr1\t0\t100\tname\t0\t-\n")
    assert read_region_file(path) == [Region("chr1", 1, 100, "-")]


def test_bgz_suffix_is_read_as_gzip(tmp_path):
    path = tmp_path / "r.bed.bgz"
    with gzip.open(path, "wt") as handle:
        handle.write("chr1\t4\t15\n")
    assert read_region_file(path) == [Region("chr1", 5, 15, "+")]


def test_gzipped_mask_file(tmp_path):
    path = tmp_path / "m.bed.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("chr1\t0\t10\n")
    assert read_mask(path, Region("chr1", 1, 30)).sum() == 10


def test_gzipped_mask_through_the_pipeline(
    tmp_path, clean_vcf_path, fasta_path
):
    path = tmp_path / "m.bed.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("chr1\t0\t25\n")
    meta = run(tmp_path, clean_vcf_path, fasta_path, mask_path=str(path))
    assert meta["masked_bases"] == 25
    assert np.all(np.load(tmp_path / "out.npy")[:, :25, :] == 0)


# ------------------------------------------------------------------
# 4. sample selection must not be quadratic
# ------------------------------------------------------------------

def test_duplicate_detection_scales():
    """
    The duplicate check used list.count inside a comprehension, which is
    O(S^2); at cohort scale that alone takes minutes. This runs in well
    under a second with a counter.
    """
    names = [f"s{i}" for i in range(60_000)]
    resolved, order = resolve_sample_selection(names, names[::-1])
    assert resolved == names[::-1]
    assert order[0] == 59_999


def test_duplicate_detection_still_correct():
    with pytest.raises(ValueError, match="Duplicate sample"):
        resolve_sample_selection(["a", "b", "c"], ["a", "b", "a", "c", "b"])


# ------------------------------------------------------------------
# 6. genotype memory is reported honestly
# ------------------------------------------------------------------

def test_genotype_peak_and_table_bytes_are_reported(
    tmp_path, clean_vcf_path, fasta_path
):
    meta = run(tmp_path, clean_vcf_path, fasta_path)
    assert meta["genotype_array_bytes"] == 2 * 4        # 2 variants, 4 samples
    # An estimate, and labelled as one: rows and matrix coexist (2x the
    # array) plus a measured per-variant object overhead.
    assert "genotype_extraction_peak_bytes" not in meta
    assert meta["estimated_genotype_extraction_peak_bytes"] == (
        2 * meta["genotype_array_bytes"] + 512 * meta["variants_kept"]
    )
    assert meta["variant_table_bytes"] == 2 * 36 * 4 * 4
    assert meta["memory_budget_bytes"] is None or meta["memory_budget_bytes"] >= 0


def test_dry_run_reports_table_cost_it_did_not_pay(
    tmp_path, clean_vcf_path, fasta_path
):
    wet = run(tmp_path, clean_vcf_path, fasta_path, name="wet.npy")
    dry = run(tmp_path, clean_vcf_path, fasta_path, dry_run=True)
    assert dry["variant_table_bytes"] == wet["variant_table_bytes"]


def test_haplotype_mode_builds_two_tables(tmp_path, clean_vcf_path, fasta_path):
    single = run(tmp_path, clean_vcf_path, fasta_path, mode="dosage")
    double = run(
        tmp_path, clean_vcf_path, fasta_path, mode="haplotypes",
        name="hap.npy",
    )
    assert double["variant_table_bytes"] == 2 * single["variant_table_bytes"]


# ------------------------------------------------------------------
# 7. per-sample filename collisions and overwrite policy
# ------------------------------------------------------------------

def test_sanitised_names_are_disambiguated():
    with pytest.warns(SampleNameCollisionWarning, match="numeric suffix"):
        names = safe_sample_names(["S/2", "S:2", "S 2", "ok"])
    assert names == ["S_2", "S_2-2", "S_2-3", "ok"]
    assert len(set(names)) == len(names)


def test_distinct_names_are_untouched():
    assert safe_sample_names(["A", "B.1", "C-2"]) == ["A", "B.1", "C-2"]


def test_empty_name_falls_back():
    assert safe_sample_names([""]) == ["sample"]
    assert safe_sample_names(["///"]) == ["_"]


def test_per_sample_collision_writes_distinct_files(
    tmp_path, collision_vcf_path, fasta_path
):
    """Two IDs that sanitise alike must not overwrite each other."""
    out = tmp_path / "per_sample"
    with pytest.warns(SampleNameCollisionWarning):
        meta = run(
            tmp_path, collision_vcf_path, fasta_path,
            per_sample=True, output_path=str(out),
        )
    files = sorted(p.name for p in out.glob("*.npy"))
    assert files == ["S_2-2.npy", "S_2.npy", "other.npy"]
    assert meta["n_output_files"] == 3
    first = np.load(out / "S_2.npy")
    second = np.load(out / "S_2-2.npy")
    assert not np.array_equal(first, second)


def test_overwrite_error_refuses_an_existing_output(
    tmp_path, clean_vcf_path, fasta_path
):
    run(tmp_path, clean_vcf_path, fasta_path)
    with pytest.raises(FileExistsError, match="--overwrite replace"):
        run(tmp_path, clean_vcf_path, fasta_path, overwrite="error")


def test_overwrite_replace_allows_it(tmp_path, clean_vcf_path, fasta_path):
    run(tmp_path, clean_vcf_path, fasta_path)
    output = tmp_path / "out.npy"
    before = np.load(output)
    old_bytes = output.read_bytes()
    metadata = run(tmp_path, clean_vcf_path, fasta_path,
                   overwrite="replace", samples=["S2"], mode="haplotypes")
    after = np.load(output)
    assert after.shape == (1, 2, REF_LENGTH, 4)
    np.testing.assert_array_equal(after[0].mean(axis=0), before[1])
    assert output.read_bytes() != old_bytes
    assert metadata["sample_ids"] == ["S2"]
    import json
    sidecar = json.loads(output.with_suffix(output.suffix + ".metadata.json").read_text())
    assert sidecar["sample_ids"] == ["S2"]
    assert sidecar["mode"] == "haplotypes"


def test_overwrite_error_protects_per_sample_files(
    tmp_path, clean_vcf_path, fasta_path
):
    out = tmp_path / "ps"
    run(tmp_path, clean_vcf_path, fasta_path, per_sample=True,
        output_path=str(out))
    with pytest.raises(FileExistsError):
        run(tmp_path, clean_vcf_path, fasta_path, per_sample=True,
            output_path=str(out), overwrite="error")


def test_check_destination_validates_its_policy(tmp_path):
    with pytest.raises(ValueError, match="overwrite must be one of"):
        check_destination(tmp_path / "x", "maybe")


# ------------------------------------------------------------------
# second audit: counting, allele validation, redundant work
# ------------------------------------------------------------------

def test_hemizygous_nocall_counts_as_missing(
    tmp_path, hemizygous_nocall_vcf_path, fasta_path
):
    """
    A '.' on a haploid contig has no allele called at all. It was
    counted as neither missing nor half-called, because the old test
    required two existing slots.
    """
    with pytest.warns(HaploidWarning):
        meta = run(
            tmp_path, hemizygous_nocall_vcf_path, fasta_path,
            mode="haplotypes", dry_run=True,
        )
    assert meta["variant_haploid_genotypes"] == 4
    assert meta["variant_missing_genotypes"] == 2   # the two '.' samples
    assert meta["variant_half_calls"] == 0          # nothing is half-called


def test_hemizygous_nocall_encodes_as_empty(
    tmp_path, hemizygous_nocall_vcf_path, fasta_path, variant_spec
):
    with pytest.warns(HaploidWarning):
        run(
            tmp_path, hemizygous_nocall_vcf_path, fasta_path,
            mode="haplotypes", missing_policy="zero",
        )
    out = np.load(tmp_path / "out.npy")
    col = variant_spec["clean_phased"]["pos"] - 1
    assert out[0, 0, col].sum() == 0.0      # '.' -> nothing known
    assert out[1, 0, col].sum() == 1.0      # '0' -> REF


def test_half_call_still_counted_for_diploids(
    tmp_path, halfcall_vcf_path, fasta_path
):
    meta = run(tmp_path, halfcall_vcf_path, fasta_path, dry_run=True)
    assert meta["variant_half_calls"] == 3
    assert meta["variant_missing_genotypes"] == 1


def test_genotype_naming_an_undeclared_allele_is_rejected(
    tmp_path, bad_allele_index_vcf_path, fasta_path
):
    """
    Allele index 4 collides with the ALLELE_MISSING slot value, so it
    used to be read as 'uncalled' instead of rejected. Validation now
    happens against the record's own ALT count.
    """
    with pytest.raises(ValueError, match="allele index 4"):
        run(tmp_path, bad_allele_index_vcf_path, fasta_path, dry_run=True)


def test_counters_describe_records_that_were_kept(
    tmp_path, dropped_multiallelic_vcf_path, fasta_path
):
    """
    The multiallelic and lower-case counters were incremented before the
    FASTA checks, so a record discarded later still showed up in them.
    """
    meta = run(
        tmp_path, dropped_multiallelic_vcf_path, fasta_path, dry_run=True,
        ambiguous_ref_policy="skip",
    )
    assert meta["variant_ambiguous_fasta"] == 1   # the position-50 record
    assert meta["variants_kept"] == 1
    assert meta["multiallelic_records"] == 0      # it was not kept
    assert meta["variant_lowercase_alleles"] == 0


def test_allele_frequencies_are_computed_on_demand(
    clean_vcf_path, reference_sequence
):
    """
    Frequencies cost several passes over the V x S matrix and only the
    AF filter and het_policy='major' need them, so a plain run must not
    pay for them.
    """
    from hottools.reference import read_reference_interval  # noqa: F401
    from hottools.variants import extract_variants

    data = extract_variants(
        vcf_path=clean_vcf_path, chrom="chr1", start=1, end=REF_LENGTH,
        reference_seq=reference_sequence, mode="dosage",
    )
    assert data._allele_freq is None
    freq = data.allele_freq                      # computed here
    assert data._allele_freq is not None
    assert freq.shape == (2, 4)
    np.testing.assert_allclose(freq.sum(axis=1), 1.0)
    assert freq is data.allele_freq              # and cached


def test_af_filter_decides_per_record_without_a_frequency_table(
    clean_vcf_path, reference_sequence
):
    """
    Since 0.7.0 the sample AF filter decides each record as it is read,
    so it never builds the matrix-wide table; the table is computed only
    if something (het_policy='major') asks for it, and is still right.
    """
    from hottools.variants import _allele_frequencies, extract_variants

    data = extract_variants(
        vcf_path=clean_vcf_path, chrom="chr1", start=1, end=REF_LENGTH,
        reference_seq=reference_sequence, mode="dosage",
        min_af=0.01, af_source="samples",
    )
    assert data._allele_freq is None
    np.testing.assert_array_equal(
        data.allele_freq, _allele_frequencies(data.genotypes)
    )


def test_consensus_non_major_policies_need_no_frequencies():
    """Only 'major' is variant-dependent, so the others must not require
    an allele-frequency table at all."""
    from hottools.constants import NO_ALLELE
    from hottools.encoding import allele_slot_vectors, consensus_tables

    ab = np.array([[1, 3, NO_ALLELE, NO_ALLELE]], dtype=np.uint8)
    vec = allele_slot_vectors(ab, np.float32, "zero")
    for policy in ("reference", "alternate"):
        table = consensus_tables(vec, policy)
        assert table.shape == (1, 36, 4)
    with pytest.raises(ValueError, match="requires allele_freq"):
        consensus_tables(vec, "major")


def test_haplotype_tables_are_independent_arrays():
    """Dropping the redundant .copy() must not alias the two tracks."""
    from hottools.constants import NO_ALLELE
    from hottools.encoding import allele_slot_vectors, haplotype_track_tables

    ab = np.array([[1, 3, NO_ALLELE, NO_ALLELE]], dtype=np.uint8)
    vec = allele_slot_vectors(ab, np.float32, "zero")
    t1, t2 = haplotype_track_tables(vec, "duplicate")
    assert not np.shares_memory(t1, t2)
    assert not np.shares_memory(t1, vec)
    before = t1.copy()
    t2[:] = 0
    np.testing.assert_array_equal(t1, before)


def test_dosage_table_does_not_alias_the_slot_vectors():
    from hottools.constants import NO_ALLELE
    from hottools.encoding import allele_slot_vectors, dosage_tables

    ab = np.array([[1, 3, NO_ALLELE, NO_ALLELE]], dtype=np.uint8)
    vec = allele_slot_vectors(ab, np.float32, "reference")
    before = vec.copy()
    table = dosage_tables(vec)
    np.testing.assert_array_equal(vec, before)   # in-place math stayed local
    assert not np.shares_memory(table, vec)
