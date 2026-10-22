"""
Structural variants end to end: interactions, genotype edge cases,
run options, filters and boundaries.

Expected haplotypes are built by hand from string slices of the
reference wherever that is short enough to read; larger mixed scenarios
are checked against ``sv_helpers.oracle_window``, an independent text
model. Numbers in comments are 1-based reference positions.
"""

from __future__ import annotations

import contextlib
import dataclasses
import warnings

import numpy as np
import pytest
from sv_helpers import (
    SAMPLES,
    Rec,
    allele_of,
    decode,
    deletion,
    haps,
    insertion,
    mnv,
    oracle_window,
    other,
    random_sequence,
    revcomp,
    run_paths,
    run_sv,
    snv,
    symdel,
    write_case,
)

from hottools.edits import UnsupportedVariant, classify_alt, parse_variant_classes
from hottools.transforms import haplotype_signatures
from hottools.variants import StructuralEditWarning, extract_variants

SEQ = random_sequence(2026, 400)
L = 200
REF = SEQ[:L]


@contextlib.contextmanager
def quiet():
    """Silence warnings where the warning itself is not under test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def cut(s: str) -> str:
    return s[:L]


def snv_at(s: str, pos: int, k: int = 0) -> str:
    """``s`` (a reference copy) with an SNV at 1-based ``pos``."""
    return s[: pos - 1] + other(s[pos - 1], k) + s[pos:]


def expect(records, gts_row: int, hap: int, *, start=1, length=L,
           offset=0, unrequested=()):
    """Oracle expectation for sample ``gts_row``'s haplotype ``hap``."""
    alleles = []
    for rec in records:
        call = rec.gts[gts_row].replace(".", "0")
        a = allele_of(call if ("|" in call or "/" in call) else f"{call}|{call}",
                      hap)
        alleles.append(0 if (rec.pos, a) in unrequested else a)
    return oracle_window(SEQ, records, alleles, anchor=start + offset,
                         offset=offset, length=length)


def extract(fasta, vcf, samples=None, **kwargs):
    kwargs.setdefault("variant_classes", parse_variant_classes("all"))
    return extract_variants(
        vcf_path=vcf, chrom="chr1", start=1, end=len(SEQ),
        reference_seq=SEQ, samples=samples, mode="haplotypes", **kwargs,
    )


# ====================================================================
# 1. substitutions and structural edits on the same haplotype
# ====================================================================

def test_snv_inside_a_deletion_disappears_with_it(tmp_path):
    d = deletion(SEQ, 40, 5, ["1|1", "1|0", "0|0", "0|0"])     # drops 41-45
    s = snv(SEQ, 43, ["1|1", "0|1", "1|1", "0|0"])
    with pytest.warns(StructuralEditWarning):
        t, meta = run_sv(tmp_path, SEQ, [d, s])
    gone = cut(SEQ[:40] + SEQ[45:])
    with_snv = cut(snv_at(SEQ, 43))
    h = haps(t)
    assert h[0] == (gone, gone)          # 43 is deleted; nothing to change
    assert h[1] == (gone, with_snv)      # deletion and SNV on different haps
    assert h[2] == (with_snv, with_snv)
    assert h[3] == (REF, REF)
    # The SNV was set aside on S1's two haplotypes, and nowhere else.
    assert meta["variant_overlapping_records_suppressed"] == 2


def test_snv_downstream_of_a_deletion_shifts_left(tmp_path):
    d = deletion(SEQ, 40, 5, ["1|0", "0|0", "0|0", "0|0"])
    s = snv(SEQ, 60, ["1|1", "0|0", "0|0", "0|0"])
    t, _ = run_sv(tmp_path, SEQ, [d, s])
    h1, h2 = haps(t)[0]
    assert h1 == cut(SEQ[:40] + SEQ[45:59] + other(SEQ[59]) + SEQ[60:])
    assert h1[59 - 5] == other(SEQ[59])       # five places to the left
    assert h2 == cut(snv_at(SEQ, 60))


def test_snv_downstream_of_an_insertion_shifts_right(tmp_path):
    i = insertion(SEQ, 40, "GGG", ["1|0", "0|0", "0|0", "0|0"])
    s = snv(SEQ, 60, ["1|1", "0|0", "0|0", "0|0"])
    t, _ = run_sv(tmp_path, SEQ, [i, s])
    h1, h2 = haps(t)[0]
    assert h1 == cut(SEQ[:40] + "GGG" + SEQ[40:59] + other(SEQ[59]) + SEQ[60:])
    assert h1[59 + 3] == other(SEQ[59])       # three places to the right
    assert h2 == cut(snv_at(SEQ, 60))


def test_mnv_partly_inside_a_deletion(tmp_path):
    d = deletion(SEQ, 40, 5, ["1|1", "0|0", "1|0", "0|0"])     # 41-45
    m = mnv(SEQ, 44, 3, ["1|1", "1|1", "0|1", "0|0"])          # 44-46
    with pytest.warns(StructuralEditWarning):
        t, _ = run_sv(tmp_path, SEQ, [d, m])
    gone = cut(SEQ[:40] + SEQ[45:])
    block = "".join(other(b) for b in SEQ[43:46])
    with_mnv = cut(SEQ[:43] + block + SEQ[46:])
    h = haps(t)
    # The deletion has the lower POS, so it wins where both are carried.
    assert h[0] == (gone, gone)
    assert h[1] == (with_mnv, with_mnv)
    assert h[2] == (gone, with_mnv)


def test_mnv_fully_inside_a_deletion(tmp_path):
    d = deletion(SEQ, 40, 5, ["1|1", "0|0", "0|0", "0|0"])
    m = mnv(SEQ, 42, 2, ["1|1", "1|1", "0|0", "0|0"])          # 42-43
    with pytest.warns(StructuralEditWarning):
        t, _ = run_sv(tmp_path, SEQ, [d, m])
    h = haps(t)
    assert h[0] == (cut(SEQ[:40] + SEQ[45:]),) * 2
    block = "".join(other(b) for b in SEQ[41:43])
    assert h[1] == (cut(SEQ[:41] + block + SEQ[43:]),) * 2


def test_an_earlier_mnv_overlapping_a_deletion_wins(tmp_path):
    m = mnv(SEQ, 38, 4, ["1|1", "0|0", "0|0", "0|0"])          # 38-41
    d = deletion(SEQ, 40, 5, ["1|1", "1|1", "0|0", "0|0"])     # 41-45
    with pytest.warns(StructuralEditWarning):
        t, meta = run_sv(tmp_path, SEQ, [m, d])
    block = "".join(other(b) for b in SEQ[37:41])
    h = haps(t)
    assert h[0] == (cut(SEQ[:37] + block + SEQ[41:]),) * 2
    assert h[1] == (cut(SEQ[:40] + SEQ[45:]),) * 2
    assert meta["variant_overlapping_records_suppressed"] == 2


def test_snv_on_the_base_after_a_deletion_applies(tmp_path):
    d = deletion(SEQ, 40, 5, ["1|1"] * 4)                      # 41-45
    s = snv(SEQ, 46, ["1|1"] * 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", StructuralEditWarning)
        t, _ = run_sv(tmp_path, SEQ, [d, s])
    assert haps(t)[0][0] == cut(SEQ[:40] + other(SEQ[45]) + SEQ[46:])


def test_snv_on_the_base_after_an_insertion_point_applies(tmp_path):
    i = insertion(SEQ, 40, "GGG", ["1|1"] * 4)                 # before 41
    s = snv(SEQ, 41, ["1|1"] * 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", StructuralEditWarning)
        t, _ = run_sv(tmp_path, SEQ, [i, s])
    assert haps(t)[0][0] == cut(SEQ[:40] + "GGG" + other(SEQ[40]) + SEQ[41:])


def test_snv_on_the_base_before_a_deletion_anchor_applies(tmp_path):
    s = snv(SEQ, 39, ["1|1"] * 4)
    d = deletion(SEQ, 40, 5, ["1|1"] * 4)
    t, _ = run_sv(tmp_path, SEQ, [s, d])
    assert haps(t)[0][0] == cut(snv_at(SEQ, 39)[:40] + SEQ[45:])


def test_an_snv_sharing_a_deletions_pos_is_a_duplicate(tmp_path):
    """A called SNV and the following deletion can share an anchor."""
    s = snv(SEQ, 40, ["1|1"] * 4)
    d = deletion(SEQ, 40, 5, ["1|1"] * 4)
    with quiet():
        t, meta = run_sv(tmp_path, SEQ, [s, d])
    assert meta["variant_duplicate_records"] == 1
    assert meta["structural_edits"] == 1
    expected = cut(SEQ[:39] + s.alt + SEQ[45:])
    assert haps(t)[0] == (expected, expected)


def test_a_duplicated_mnv_preserves_first_applicable_block(tmp_path):
    """The first called MNV is preserved; the conflicting SNV is skipped."""
    m = mnv(SEQ, 40, 3, ["1|1"] * 4)
    s = snv(SEQ, 40, ["1|1"] * 4)
    with quiet():
        t, _ = run_sv(tmp_path, SEQ, [m, s])
    expected = cut(SEQ[:39] + m.alt + SEQ[42:])
    assert haps(t)[0] == (expected, expected)


# ====================================================================
# 2. several structural edits on one haplotype
# ====================================================================

def test_three_edits_on_one_haplotype_compose(tmp_path):
    d1 = deletion(SEQ, 40, 4, ["1|1", "1|0", "0|0", "0|0"])    # 41-44
    i = insertion(SEQ, 80, "TTAC", ["1|1", "0|1", "0|0", "0|0"])
    d2 = deletion(SEQ, 120, 6, ["1|1", "1|1", "0|0", "0|0"])   # 121-126
    t, meta = run_sv(tmp_path, SEQ, [d1, i, d2])
    h = haps(t)
    every = cut(SEQ[:40] + SEQ[44:80] + "TTAC" + SEQ[80:120] + SEQ[126:])
    assert h[0] == (every, every)
    assert h[1] == (
        cut(SEQ[:40] + SEQ[44:120] + SEQ[126:]),
        cut(SEQ[:80] + "TTAC" + SEQ[80:120] + SEQ[126:]),
    )
    assert h[2] == (REF, REF)
    # all three, d1+d2, ins+d2, and none
    assert meta["structural_signatures"] == 4


# ====================================================================
# 3, 8-12. run options give the same answer with structural edits
# ====================================================================

MIX_SAMPLES = ["A", "B", "C", "D", "E", "F"]
MIX = [
    snv(SEQ, 20, ["0|1", "1|1", "0|0", "1|0", "0|0", "1|1"]),
    deletion(SEQ, 40, 5, ["1|0", "0|1", "1|1", "0|0", "1|0", "0|0"]),
    insertion(SEQ, 70, "ACGTA", ["0|1", "1|0", "0|0", "1|1", "0|1", "0|0"]),
    mnv(SEQ, 100, 3, ["1|1", "0|0", "0|1", "1|0", "0|0", "1|0"]),
    symdel(SEQ, 130, 138, ["0|0", "1|1", "1|0", "0|1", "1|0", "0|1"]),
    snv(SEQ, 160, ["1|0", "0|1", "1|1", "0|0", "1|1", "0|0"], k=1),
]


@pytest.fixture
def mix(tmp_path):
    fasta, vcf = write_case(tmp_path, SEQ, MIX, MIX_SAMPLES, name="mix")
    reference, meta = run_paths(tmp_path, fasta, vcf, name="mix_ref",
                                batch_size=len(MIX_SAMPLES))
    return fasta, vcf, reference, meta


def test_mixed_fixture_matches_the_oracle(mix):
    _, _, reference, meta = mix
    assert meta["structural_signatures"] > 2
    for row, (h1, h2) in enumerate(haps(reference)):
        assert h1 == expect(MIX, row, 0), MIX_SAMPLES[row]
        assert h2 == expect(MIX, row, 1), MIX_SAMPLES[row]


@pytest.mark.parametrize("batch_size", [1, 2, 4, "auto"])
def test_structural_batching_invariance(tmp_path, mix, batch_size):
    fasta, vcf, reference, _ = mix
    t, _ = run_paths(tmp_path, fasta, vcf, name=f"b{batch_size}",
                        batch_size=batch_size)
    np.testing.assert_array_equal(t, reference)


@pytest.mark.parametrize("fmt", ["npy", "npz", "h5", "pt"])
@pytest.mark.parametrize("per_sample", [False, True])
@pytest.mark.parametrize("batch_size", [1, 6])
def test_structural_formats_and_per_sample(tmp_path, mix, fmt, per_sample,
                                           batch_size):
    if fmt == "pt":
        pytest.importorskip("torch")
    fasta, vcf, reference, _ = mix
    t, _ = run_paths(tmp_path, fasta, vcf, name="fmt", fmt=fmt,
                     per_sample=per_sample, batch_size=batch_size)
    np.testing.assert_array_equal(t, reference)


def test_structural_channels_first_is_a_transpose(tmp_path, mix):
    fasta, vcf, reference, _ = mix
    t, _ = run_paths(tmp_path, fasta, vcf, name="cf",
                     output_layout="channels_first", batch_size=2)
    np.testing.assert_array_equal(t, np.transpose(reference, (0, 1, 3, 2)))


@pytest.mark.parametrize("layout", ["samples_first", "channels_first"])
def test_structural_minus_strand_is_the_reverse_complement(tmp_path, mix,
                                                           layout):
    fasta, vcf, _, _ = mix
    plus, _ = run_paths(tmp_path, fasta, vcf, name="p", output_layout=layout)
    minus, _ = run_paths(tmp_path, fasta, vcf, name="m", output_layout=layout,
                         strand="-", batch_size=1)
    seq_axis, chan_axis = (2, 3) if layout == "samples_first" else (3, 2)
    np.testing.assert_array_equal(
        minus, np.flip(np.flip(plus, axis=seq_axis), axis=chan_axis)
    )
    if layout == "samples_first":
        for (p1, p2), (m1, m2) in zip(haps(plus), haps(minus), strict=True):
            assert (m1, m2) == (revcomp(p1), revcomp(p2))


def test_structural_float16_matches_float32(tmp_path, mix):
    fasta, vcf, reference, _ = mix
    t16, _ = run_paths(tmp_path, fasta, vcf, name="f16", dtype="float16",
                          batch_size=1)
    assert t16.dtype == np.float16
    np.testing.assert_array_equal(t16.astype(np.float32), reference)
    # The inserted ACGTA really is in the half-precision output.
    assert "ACGTA" in haps(t16)[0][1]


# ====================================================================
# 4. sample reordering and subsetting
# ====================================================================

def test_structural_sample_reordering(tmp_path):
    d = deletion(SEQ, 40, 5, ["0|0", "1|0", "0|0", "0|1"])
    i = insertion(SEQ, 80, "GG", ["1|1", "0|1", "0|0", "1|0"])
    fasta, vcf = write_case(tmp_path, SEQ, [d, i])
    full, _ = run_paths(tmp_path, fasta, vcf, name="full")
    sub, meta = run_paths(tmp_path, fasta, vcf, name="sub",
                          samples=["S4", "S2"])
    assert meta["sample_ids"] == ["S4", "S2"]
    np.testing.assert_array_equal(sub, full[[3, 1]])
    s4, s2 = haps(sub)
    # S4: insertion on haplotype 1, deletion on haplotype 2.
    assert s4 == (cut(SEQ[:80] + "GG" + SEQ[80:]), cut(SEQ[:40] + SEQ[45:]))
    assert s2 == (cut(SEQ[:40] + SEQ[45:]), cut(SEQ[:80] + "GG" + SEQ[80:]))

    # Both genotype matrices follow the request.
    every = extract(fasta, vcf)
    picked = extract(fasta, vcf, samples=["S4", "S2"])
    np.testing.assert_array_equal(picked.structural.genotypes,
                                  every.structural.genotypes[:, [3, 1]])


# ====================================================================
# 5-7. unphased, missing, half-called and haploid structural calls
# ====================================================================

def test_unphased_structural_heterozygote(tmp_path):
    d = deletion(SEQ, 40, 5, ["0/1", "1|0", "0|0", "0|0"])
    with pytest.raises(ValueError, match="nphased"):
        run_sv(tmp_path, SEQ, [d], name="refused")
    assert not (tmp_path / "refused.npy").exists()

    t, _ = run_sv(tmp_path, SEQ, [d], name="missing",
                     unphased_policy="missing")
    h = haps(t)
    assert h[0] == (REF, REF)            # no deletion invented on either
    assert h[1] == (cut(SEQ[:40] + SEQ[45:]), REF)
    fasta, vcf = write_case(tmp_path, SEQ, [d], name="missing")
    edits = extract(fasta, vcf, unphased_policy="missing").structural
    carriers = haplotype_signatures(edits).carriers
    assert not carriers[0].any() and not carriers[1].any()


MISSING_CALLS = [
    deletion(SEQ, 40, 5, ["./.", "0|.", "1|.", "1|1"]),
    insertion(SEQ, 80, "GG", ["1|.", "./.", ".|1", "0|0"]),
]


@pytest.mark.parametrize("missing_policy", ["zero", "reference"])
def test_a_missing_allele_never_carries_a_structural_edit(tmp_path,
                                                          missing_policy):
    t, _ = run_sv(tmp_path, SEQ, MISSING_CALLS, missing_policy=missing_policy)
    for row, (h1, h2) in enumerate(haps(t)):
        assert h1 == expect(MISSING_CALLS, row, 0), SAMPLES[row]
        assert h2 == expect(MISSING_CALLS, row, 1), SAMPLES[row]
    fasta, vcf = write_case(tmp_path, SEQ, MISSING_CALLS)
    carriers = haplotype_signatures(
        extract(fasta, vcf, missing_policy=missing_policy).structural
    ).carriers
    # edit 0 = deletion, edit 1 = insertion; rows are S1h1, S1h2, S2h1 ...
    assert carriers[:, 0].tolist() == [0, 0, 0, 0, 1, 0, 1, 1]
    assert carriers[:, 1].tolist() == [1, 0, 0, 0, 0, 1, 0, 0]


def test_missing_policy_error_refuses_missing_structural_calls(tmp_path):
    with pytest.raises(ValueError, match="missing allele"):
        run_sv(tmp_path, SEQ, MISSING_CALLS, missing_policy="error")
    assert not (tmp_path / "out.npy").exists()


HAPLOID = [
    deletion(SEQ, 40, 5, ["0", "1", "1|0", "0|0"]),
    snv(SEQ, 60, ["1", "1", "0|1", "0|0"]),
]


def test_haploid_structural_call_is_duplicated_like_its_snvs(tmp_path):
    t, _ = run_sv(tmp_path, SEQ, HAPLOID, haploid_policy="duplicate")
    h = haps(t)
    deleted = cut(SEQ[:40] + SEQ[45:59] + other(SEQ[59]) + SEQ[60:])
    assert h[0] == (cut(snv_at(SEQ, 60)),) * 2
    # Track 2 copies the single allele - the deletion as well as the SNV.
    assert h[1] == (deleted, deleted)
    fasta, vcf = write_case(tmp_path, SEQ, HAPLOID)
    edits = extract(fasta, vcf, haploid_policy="duplicate").structural
    carriers = haplotype_signatures(edits, haploid_policy="duplicate").carriers
    assert carriers[:, 0].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]


def test_haploid_blank_leaves_track_two_without_the_edit(tmp_path):
    t, _ = run_sv(tmp_path, SEQ, HAPLOID, haploid_policy="blank")
    h = haps(t)
    blank_snv = cut(SEQ[:59] + "-" + SEQ[60:])
    assert h[1] == (cut(SEQ[:40] + SEQ[45:59] + other(SEQ[59]) + SEQ[60:]),
                    blank_snv)
    assert h[0] == (cut(snv_at(SEQ, 60)), blank_snv)


# ====================================================================
# 13. allele-frequency filtering of structural-only records
# ====================================================================

#: INFO/AF and the sample frequency deliberately disagree, so the source
#: actually used is visible in the result.
AF_RECORDS = [
    dataclasses.replace(
        deletion(SEQ, 40, 5, ["0|1", "1|0", "0|0", "0|0"]), info="AF=0.9"
    ),                                                 # samples: 0.25
    dataclasses.replace(
        insertion(SEQ, 80, "GG", ["1|1", "1|1", "1|0", "0|0"]), info="AF=0.1"
    ),                                                 # samples: 0.625
]


def _carries(t, sample: int, hap: int, deletion: bool, insertion: bool):
    want = SEQ[:40] + SEQ[45:] if deletion else SEQ
    if insertion:
        cut_at = 80 - (5 if deletion else 0)
        want = want[:cut_at] + "GG" + want[cut_at:]
    return haps(t)[sample][hap] == cut(want)


@pytest.mark.parametrize("af_source, kwargs, keep_del, keep_ins", [
    ("samples", {"min_af": 0.5}, False, True),
    ("samples", {"max_af": 0.5}, True, False),
    ("info", {"min_af": 0.5}, True, False),
    ("info", {"max_af": 0.5}, False, True),
    ("auto", {"min_af": 0.5}, True, False),       # both records have INFO/AF
])
def test_af_filter_on_structural_only_records(tmp_path, af_source, kwargs,
                                              keep_del, keep_ins):
    t, meta = run_sv(tmp_path, SEQ, AF_RECORDS, af_source=af_source, **kwargs)
    # S1 is 0|1 for the deletion and 1|1 for the insertion.
    assert _carries(t, 0, 1, keep_del, keep_ins)
    assert _carries(t, 0, 0, False, keep_ins)
    # One record filtered, counted once as a record.
    assert meta["variant_filtered_allele_frequency"] == 1
    assert meta["structural_edits"] == 1


def test_af_auto_uses_samples_where_info_is_absent(tmp_path):
    records = [
        AF_RECORDS[0],                                         # INFO 0.9
        deletion(SEQ, 120, 4, ["0|1", "0|0", "0|0", "0|0"]),  # samples 0.125
    ]
    t, meta = run_sv(tmp_path, SEQ, records, af_source="auto", min_af=0.5)
    assert meta["variant_filtered_allele_frequency"] == 1
    assert haps(t)[0][1] == cut(SEQ[:40] + SEQ[45:])    # 120 deletion gone


def test_af_info_requires_info_af_on_structural_records(tmp_path):
    records = [deletion(SEQ, 120, 4, ["0|1", "0|0", "0|0", "0|0"])]
    with pytest.raises(ValueError, match="INFO/AF"):
        run_sv(tmp_path, SEQ, records, af_source="info", min_af=0.5)


def test_a_filtered_deletion_does_not_suppress_what_it_covered(tmp_path):
    d = deletion(SEQ, 40, 5, ["1|1", "0|0", "0|0", "0|0"])    # AF 0.25
    s = snv(SEQ, 43, ["1|1", "1|1", "0|0", "0|0"])            # AF 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("error", StructuralEditWarning)
        t, meta = run_sv(tmp_path, SEQ, [d, s], af_source="samples",
                         min_af=0.3)
    assert meta["variant_overlapping_records_suppressed"] == 0
    assert haps(t)[0] == (cut(snv_at(SEQ, 43)),) * 2


# ====================================================================
# 14-15. multiallelic structural records
# ====================================================================

def test_two_insertions_in_one_record(tmp_path):
    base = SEQ[79]
    rec = Rec(80, base, f"{base}T,{base}TTT", ("1|2", "2|1", "0|1", "2|2"))
    t, meta = run_sv(tmp_path, SEQ, [rec])
    short = cut(SEQ[:80] + "T" + SEQ[80:])
    long_ = cut(SEQ[:80] + "TTT" + SEQ[80:])
    h = haps(t)
    assert h[0] == (short, long_)
    assert h[1] == (long_, short)
    assert h[2] == (REF, short)
    assert h[3] == (long_, long_)
    assert meta["structural_edits"] == 2

    fasta, vcf = write_case(tmp_path, SEQ, [rec])
    edits = extract(fasta, vcf).structural
    assert edits.n_edits == 2 and edits.n_records == 1
    assert edits.record.tolist() == [0, 0]               # same VCF record
    assert int(edits.positions[0]) == 80
    groups = haplotype_signatures(edits).group
    assert groups[0] != groups[1]                        # S1h1 vs S1h2
    assert groups[0] == groups[3]                        # both 'T'


def test_a_deletion_and_a_substitution_in_one_record(tmp_path):
    ref = SEQ[79:81]                                     # bases 80-81
    rec = Rec(80, ref, f"{ref[0]},{other(ref[0])}{ref[1]}",
              ("1|2", "2|2", "0|1", "0|0"))
    t, meta = run_sv(tmp_path, SEQ, [rec])
    deleted = cut(SEQ[:80] + SEQ[81:])                   # drops base 81
    swapped = cut(snv_at(SEQ, 80))                       # 80 changes
    h = haps(t)
    assert h[0] == (deleted, swapped)
    assert h[1] == (swapped, swapped)
    assert h[2] == (REF, deleted)
    assert meta["variant_deletion_edits"] == 1


MIXED_CLASSES = Rec(
    100, SEQ[99:102],
    "".join(other(b) for b in SEQ[99:102]) + "," + SEQ[99:102] + "TT",
    ("1|2", "2|1", "1|1", "0|0"),
)


@pytest.mark.parametrize("classes", ["mnv", "ins"])
def test_an_unrequested_allele_reads_as_reference(tmp_path, classes):
    t, meta = run_sv(tmp_path, SEQ, [MIXED_CLASSES], variant_classes=classes,
                     mixed_record_policy="apply-selected")
    block = "".join(other(b) for b in SEQ[99:102])
    with_mnv = cut(SEQ[:99] + block + SEQ[102:])
    with_ins = cut(SEQ[:102] + "TT" + SEQ[102:])
    mnv_hap = with_mnv if classes == "mnv" else REF
    ins_hap = with_ins if classes == "ins" else REF
    h = haps(t)
    assert h[0] == (mnv_hap, ins_hap)
    assert h[1] == (ins_hap, mnv_hap)
    assert h[2] == (mnv_hap, mnv_hap)
    assert "-" not in "".join(h[0])        # reference, not an uncalled gap
    assert meta["variant_class_not_requested"] == 1


# ====================================================================
# 16-17. unsupported representations and symbolic coordinates
# ====================================================================

def test_unsupported_representations_are_refused_and_counted(tmp_path):
    def b(pos):
        return SEQ[pos - 1]

    records = [
        Rec(20, b(20), "<INS>", ("1|1",) * 4, "END=20"),
        Rec(30, b(30), "<INV>", ("1|1",) * 4, "END=40"),
        Rec(50, b(50), "<DUP>", ("1|1",) * 4, "END=60"),
        Rec(70, b(70), "*", ("1|1",) * 4),
        Rec(80, b(80), f"{b(80)}]chr2:100]", ("1|1",) * 4),
        Rec(90, b(90), f"]chr2:100]{b(90)}", ("1|1",) * 4),
        Rec(100, b(100), "<DEL>", ("1|1",) * 4),                # no END
        Rec(110, b(110), "<DEL>", ("1|1",) * 4, "END=105"),     # END < POS
        Rec(120, b(120), "<DEL>", ("1|1",) * 4, "END=120"),     # END == POS
        Rec(150, b(150), f"{other(b(150))},<INV>", ("1|2",) * 4, "END=155"),
        snv(SEQ, 140, ["1|1"] * 4),
    ]
    with quiet():
        t, meta = run_sv(tmp_path, SEQ, records)
    assert meta["variant_unsupported_alt_alleles"] == 10
    assert meta["variants_kept"] == 1
    assert meta["structural_edits"] == 0
    # Nothing was silently encoded as reference-with-a-twist: every
    # refused record leaves the reference, and the one valid SNV applies.
    assert haps(t)[0] == (cut(snv_at(SEQ, 140)),) * 2


@pytest.mark.parametrize("end, deleted", [
    (11, [11]),
    (12, [11, 12]),
    (14, [11, 12, 13, 14]),
])
def test_symbolic_deletion_removes_pos_plus_one_through_end(tmp_path, end,
                                                            deleted):
    edit = classify_alt(10, SEQ[9], "<DEL>", 1, symbolic_end=end)
    assert (edit.start, edit.end) == (11, end + 1)
    t, _ = run_sv(tmp_path, SEQ, [symdel(SEQ, 10, end, ["1|1"] * 4)])
    kept = "".join(SEQ[p - 1] for p in range(1, len(SEQ) + 1)
                   if p not in deleted)
    assert haps(t)[0][0] == cut(kept)


def test_symbolic_deletion_with_end_equal_to_pos_is_malformed():
    with pytest.raises(UnsupportedVariant, match="END"):
        classify_alt(10, SEQ[9], "<DEL>", 1, symbolic_end=10)
    with pytest.raises(UnsupportedVariant, match="END"):
        classify_alt(10, SEQ[9], "<DEL>", 1, symbolic_end=9)


# ====================================================================
# 18, 20. insertions at window boundaries and at the anchor
# ====================================================================

@pytest.mark.parametrize("pos, expected", [
    (50, SEQ[50:150]),                                  # before the start
    (51, (SEQ[50:51] + "GGGG" + SEQ[51:])[:100]),       # after the start
    (149, (SEQ[50:149] + "GGGG" + SEQ[149:])[:100]),    # before the end
    (150, SEQ[50:150]),                                 # after the end
    (151, SEQ[50:150]),                                 # beyond the end
])
def test_insertion_at_window_boundaries(tmp_path, pos, expected):
    t, _ = run_sv(tmp_path, SEQ, [insertion(SEQ, pos, "GGGG", ["1|1"] * 4)],
                  start=51, end=150)
    assert haps(t)[0][0] == expected


def test_insertion_just_before_the_anchor_lands_left_of_it(tmp_path):
    """The anchor base keeps its offset; bases inserted before it sit to
    its left. This is the permanent convention."""
    t, meta = run_sv(tmp_path, SEQ, [insertion(SEQ, 50, "TTT", ["1|1"] * 4)],
                     start=1, end=101, anchor="center")
    assert meta["anchor_position"] == 51 and meta["anchor_output_offset"] == 50
    h = haps(t)[0][0]
    assert h[50] == SEQ[50]
    assert h[47:50] == "TTT"
    assert h == SEQ[3:50] + "TTT" + SEQ[50:101]


def test_insertion_just_after_the_anchor_lands_right_of_it(tmp_path):
    t, _ = run_sv(tmp_path, SEQ, [insertion(SEQ, 51, "TTT", ["1|1"] * 4)],
                  start=1, end=101, anchor="center")
    h = haps(t)[0][0]
    assert h[50] == SEQ[50]
    assert h[51:54] == "TTT"


# ====================================================================
# 19. a deletion that swallows the anchor
# ====================================================================

ANCHOR_DEL = deletion(SEQ, 48, 6, ["1|1", "0|0", "0|0", "0|0"])  # 49-54


def test_deleted_anchor_moves_right_by_default(tmp_path):
    t, _ = run_sv(tmp_path, SEQ, [ANCHOR_DEL], start=1, end=101,
                     anchor="center")
    h = haps(t)[0][0]
    assert h[50] == SEQ[54]                     # base 55, first survivor
    assert h == expect([ANCHOR_DEL], 0, 0, length=101, offset=50)
    assert haps(t)[1][0][50] == SEQ[50]         # non-carrier unaffected


def test_deleted_anchor_can_be_an_error(tmp_path):
    with pytest.raises(ValueError, match="anchor"):
        run_sv(tmp_path, SEQ, [ANCHOR_DEL], start=1, end=101,
               anchor="center", anchor_deletion_policy="error")
    assert not (tmp_path / "out.npy").exists()


# ====================================================================
# 21-22. adjacent and overlapping edits
# ====================================================================

def test_adjacent_deletions_both_apply(tmp_path):
    first = deletion(SEQ, 9, 5, ["1|1"] * 4)        # 10-14
    second = deletion(SEQ, 14, 5, ["1|1"] * 4)      # 15-19
    with warnings.catch_warnings():
        warnings.simplefilter("error", StructuralEditWarning)
        t, meta = run_sv(tmp_path, SEQ, [first, second])
    assert haps(t)[0][0] == cut(SEQ[:9] + SEQ[19:])
    assert meta["variant_overlapping_records_suppressed"] == 0


def test_an_insertion_and_deletion_at_one_pos_are_duplicates(tmp_path):
    """In a VCF they can only share a POS, so the duplicate rule decides."""
    i = insertion(SEQ, 14, "GG", ["1|1"] * 4)
    d = deletion(SEQ, 14, 5, ["1|1"] * 4)
    with quiet():
        t, meta = run_sv(tmp_path, SEQ, [i, d])
    assert meta["variant_duplicate_records"] == 1
    expected = cut(SEQ[:14] + "GG" + SEQ[14:])
    assert haps(t)[0] == (expected, expected)


OVERLAPPING = [
    deletion(SEQ, 40, 8, ["1|1", "0|1", "0|0", "0|0"]),   # 41-48
    deletion(SEQ, 44, 8, ["1|1", "1|0", "0|0", "0|0"]),   # 45-52
]


def test_overlap_policy_first(tmp_path):
    with pytest.warns(StructuralEditWarning):
        t, meta = run_sv(tmp_path, SEQ, OVERLAPPING, overlap_policy="first")
    h = haps(t)
    assert h[0] == (cut(SEQ[:40] + SEQ[48:]),) * 2
    assert h[1] == (cut(SEQ[:44] + SEQ[52:]), cut(SEQ[:40] + SEQ[48:]))
    assert meta["variant_overlapping_records_suppressed"] == 2


def test_overlap_policy_error_fails_before_writing(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        run_sv(tmp_path, SEQ, OVERLAPPING, overlap_policy="error")
    assert not (tmp_path / "out.npy").exists()
    assert not any(p.name.endswith(".hottools-partial")
                   for p in tmp_path.iterdir())


# ====================================================================
# 26-27. edits large against the window, and the right contig edge
# ====================================================================

def test_deletion_longer_than_the_window(tmp_path):
    t, meta = run_sv(tmp_path, SEQ, [deletion(SEQ, 10, 80, ["1|1"] * 4)],
                     start=1, end=50)
    h = haps(t)[0][0]
    assert len(h) == 50 and h == (SEQ[:10] + SEQ[90:])[:50]
    assert meta["unfilled_positions"] == 0


def test_insertion_longer_than_the_window(tmp_path):
    long_ins = random_sequence(7, 120)
    t, meta = run_sv(tmp_path, SEQ, [insertion(SEQ, 10, long_ins, ["1|1"] * 4)],
                     start=1, end=50)
    assert haps(t)[0][0] == (SEQ[:10] + long_ins)[:50]
    assert meta["unfilled_positions"] == 0


def test_deletion_running_off_the_contig_end_is_zero_filled(tmp_path):
    t, meta = run_sv(tmp_path, SEQ, [deletion(SEQ, 360, 40, ["1|1"] * 4)],
                     start=351, end=400)
    h = haps(t)[0][0]
    # 361-400 are gone; nothing wraps round from the start of the window.
    assert h == SEQ[350:360] + "-" * 40
    assert meta["unfilled_positions"] == 40


def test_deletion_near_the_contig_end(tmp_path):
    t, meta = run_sv(tmp_path, SEQ, [deletion(SEQ, 380, 10, ["1|1"] * 4)],
                     start=351, end=400)
    assert haps(t)[0][0] == SEQ[350:380] + SEQ[390:400] + "-" * 10
    assert meta["unfilled_positions"] == 10


def test_insertion_near_the_contig_end(tmp_path):
    t, meta = run_sv(tmp_path, SEQ, [insertion(SEQ, 390, "AAAA", ["1|1"] * 4)],
                     start=351, end=400)
    assert haps(t)[0][0] == (SEQ[350:390] + "AAAA" + SEQ[390:400])[:50]
    assert meta["unfilled_positions"] == 0


def test_centre_anchor_without_enough_downstream_reference(tmp_path):
    d = deletion(SEQ, 360, 20, ["1|1"] * 4)
    t, meta = run_sv(tmp_path, SEQ, [d], start=301, end=400, anchor="center")
    h = haps(t)[0][0]
    assert h == expect([d], 0, 0, start=301, length=100, offset=49)
    assert h.endswith("-" * 20) and meta["unfilled_positions"] == 20


def test_centre_anchor_without_enough_upstream_reference(tmp_path):
    d = deletion(SEQ, 10, 15, ["1|1"] * 4)
    t, _ = run_sv(tmp_path, SEQ, [d], start=1, end=100, anchor="center")
    h = haps(t)[0][0]
    assert h == expect([d], 0, 0, start=1, length=100, offset=49)
    assert h.startswith("-" * 15)


# ====================================================================
# 28. masks that touch part of an event
# ====================================================================

def _mask(tmp_path, first, last):
    path = tmp_path / "mask.bed"
    path.write_text(f"chr1\t{first - 1}\t{last}\n")
    return str(path)


def test_mask_on_deleted_bases_drops_the_deletion(tmp_path):
    t, meta = run_sv(tmp_path, SEQ, [deletion(SEQ, 40, 5, ["1|1"] * 4)],
                     mask_path=_mask(tmp_path, 43, 43))
    assert meta["variant_masked_out"] == 1
    assert haps(t)[0][0] == cut(SEQ[:42] + "-" + SEQ[43:])


def test_mask_on_only_the_anchor_keeps_the_deletion(tmp_path):
    """The anchor base is quoted in REF but not changed by the deletion."""
    t, meta = run_sv(tmp_path, SEQ, [deletion(SEQ, 40, 5, ["1|1"] * 4)],
                     mask_path=_mask(tmp_path, 40, 40))
    assert meta["variant_masked_out"] == 0
    assert haps(t)[0][0] == cut(SEQ[:39] + "-" + SEQ[45:])


@pytest.mark.parametrize("masked, dropped", [
    (80, True),     # the base the insertion follows
    (81, True),     # the base it precedes
    (79, False),    # not adjacent to the insertion point
])
def test_mask_at_an_insertion_point(tmp_path, masked, dropped):
    t, meta = run_sv(tmp_path, SEQ, [insertion(SEQ, 80, "GG", ["1|1"] * 4)],
                     mask_path=_mask(tmp_path, masked, masked))
    assert meta["variant_masked_out"] == int(dropped)
    masked_ref = SEQ[: masked - 1] + "-" + SEQ[masked:]
    want = masked_ref if dropped else masked_ref[:80] + "GG" + masked_ref[80:]
    assert haps(t)[0][0] == cut(want)


# ====================================================================
# 29. REF is validated across its whole span
# ====================================================================

@pytest.mark.parametrize("record", [
    # deletion: first base right, second wrong
    Rec(40, SEQ[39] + other(SEQ[40]) + SEQ[41:44], SEQ[39], ("1|1",) * 4),
    # MNV: first two bases right, third wrong
    Rec(60, SEQ[59:61] + other(SEQ[61]),
        "".join(other(b) for b in SEQ[59:62]), ("1|1",) * 4),
])
def test_ref_mismatch_anywhere_in_the_span_is_caught(tmp_path, record):
    with pytest.raises(ValueError, match="REF mismatch"):
        run_sv(tmp_path, SEQ, [record], name="err")
    t, meta = run_sv(tmp_path, SEQ, [record], name="skip",
                     ref_mismatch_policy="skip")
    assert meta["variant_ref_mismatch"] == 1
    assert haps(t)[0] == (REF, REF)


def test_decode_helper_marks_zero_rows():
    assert decode(np.zeros((2, 4))) == "--"


# ====================================================================
# Working-window padding reaches a fixed point
# ====================================================================

def test_padding_follows_a_chain_of_deletions_pulled_into_the_window(tmp_path):
    # Each deletion's replacement bases carry the next deletion. Region
    # 101-140 with deletions at 120, 145 and 155: the first scan sees one
    # (pad 10), the second two (pad 20), and only a third sees the one at
    # 155, which needs bases out to 170. Two fixed scans left 5 unfilled.
    records = [deletion(SEQ, p, 10, ["1|1"] * 4) for p in (120, 145, 155)]
    t, meta = run_sv(tmp_path, SEQ, records, start=101, end=140)
    want = oracle_window(SEQ, records, [1, 1, 1], anchor=101, offset=0,
                         length=40)
    assert "-" not in want
    assert meta["unfilled_positions"] == 0
    assert meta["working_end"] >= 170
    assert haps(t)[0] == (want, want)
