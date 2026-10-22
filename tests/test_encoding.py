import numpy as np
import pytest

from hottools.constants import (
    ALLELE_ABSENT,
    ALLELE_MISSING,
    NO_ALLELE,
    pack_genotype,
)
from hottools.encoding import (
    _build_variant_tables,
    _encode_haplotype_tracks,
    _encode_single_track,
    encode_consensus_batch,
    encode_dosage_batch,
    encode_haplotypes,
)

REF, ALT, ALT2, MISS, ABS = 0, 1, 2, int(ALLELE_MISSING), int(ALLELE_ABSENT)

#: Reference window A C G; the variant sits at offset 1 (C).
POS = np.array([1], dtype=np.int32)

#: One variant, REF=C (base 1), ALT=T (base 3).
ALLELES = np.array([[1, 3, NO_ALLELE, NO_ALLELE]], dtype=np.uint8)

#: Same variant with a second ALT=A (base 0).
ALLELES_TRI = np.array([[1, 3, 0, NO_ALLELE]], dtype=np.uint8)


def gt(a1, a2):
    return int(pack_genotype(a1, a2))


def _ref():
    # A C G
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.float32,
    )


def _codes(*pairs):
    return np.array([[gt(a, b) for a, b in pairs]], dtype=np.uint8)


def _dosage(codes=None, **overrides):
    kwargs = dict(
        ref_onehot=_ref(),
        variant_pos0=POS,
        allele_bases=ALLELES,
        genotypes=_codes((REF, REF), (REF, ALT), (ALT, ALT), (MISS, MISS))
        if codes is None else codes,
        sample_start=0,
        sample_end=4,
    )
    kwargs.update(overrides)
    kwargs["sample_end"] = kwargs["genotypes"].shape[1]
    return encode_dosage_batch(**kwargs)


def _haps(codes=None, **overrides):
    kwargs = dict(
        ref_onehot=_ref(),
        variant_pos0=POS,
        allele_bases=ALLELES,
        genotypes=_codes((REF, REF), (REF, ALT), (ALT, REF), (MISS, MISS))
        if codes is None else codes,
        sample_start=0,
        sample_end=4,
    )
    kwargs.update(overrides)
    kwargs["sample_end"] = kwargs["genotypes"].shape[1]
    return encode_haplotypes(**kwargs)


def _consensus(codes=None, **overrides):
    kwargs = dict(
        ref_onehot=_ref(),
        variant_pos0=POS,
        allele_bases=ALLELES,
        genotypes=_codes((REF, REF), (REF, ALT), (ALT, ALT), (MISS, MISS))
        if codes is None else codes,
        allele_freq=np.array([[0.7, 0.3, 0.0, 0.0]], dtype=np.float32),
        sample_start=0,
        sample_end=4,
    )
    kwargs.update(overrides)
    kwargs["sample_end"] = kwargs["genotypes"].shape[1]
    return encode_consensus_batch(**kwargs)


# ------------------------------------------------------------------
# dosage
# ------------------------------------------------------------------

def test_dosage_variant_across_samples():
    out = _dosage(missing_policy="zero")
    np.testing.assert_array_equal(out[0, 1], [0, 1, 0, 0])
    np.testing.assert_array_equal(out[1, 1], [0, .5, 0, .5])
    np.testing.assert_array_equal(out[2, 1], [0, 0, 0, 1])
    np.testing.assert_array_equal(out[3, 1], [0, 0, 0, 0])


def test_dosage_missing_policy_reference():
    out = _dosage(missing_policy="reference")
    np.testing.assert_array_equal(out[3, 1], [0, 1, 0, 0])


def test_dosage_leaves_non_variant_positions_alone():
    out = _dosage()
    for sample in range(4):
        np.testing.assert_array_equal(out[sample, 0], [1, 0, 0, 0])
        np.testing.assert_array_equal(out[sample, 2], [0, 0, 1, 0])


def test_dosage_rows_sum_to_one_except_missing():
    out = _dosage()
    sums = out.sum(axis=2)
    np.testing.assert_allclose(sums[:3], 1.0)
    assert sums[3, 1] == 0.0


# ------------------------------------------------------------------
# haplotypes
# ------------------------------------------------------------------

def test_haplotype_variant_across_samples():
    out = _haps()
    np.testing.assert_array_equal(out[1, 0, 1], [0, 1, 0, 0])
    np.testing.assert_array_equal(out[1, 1, 1], [0, 0, 0, 1])
    np.testing.assert_array_equal(out[2, 0, 1], [0, 0, 0, 1])
    np.testing.assert_array_equal(out[2, 1, 1], [0, 1, 0, 0])


def test_haplotype_missing_is_zero_then_reference():
    np.testing.assert_array_equal(_haps()[3, 0, 1], [0, 0, 0, 0])
    np.testing.assert_array_equal(
        _haps(missing_policy="reference")[3, 0, 1], [0, 1, 0, 0]
    )


def test_haplotype_both_strands_share_the_reference_backbone():
    out = _haps()
    for sample in range(4):
        for hap in range(2):
            np.testing.assert_array_equal(out[sample, hap, 0], [1, 0, 0, 0])


# ------------------------------------------------------------------
# haploid padding
# ------------------------------------------------------------------

def test_haploid_duplicate_copies_the_allele_onto_both_tracks():
    codes = _codes((REF, ABS), (ALT, ABS))
    out = _haps(codes, haploid_policy="duplicate")
    np.testing.assert_array_equal(out[0, 0, 1], out[0, 1, 1])
    np.testing.assert_array_equal(out[0, 0, 1], [0, 1, 0, 0])
    np.testing.assert_array_equal(out[1, 0, 1], out[1, 1, 1])
    np.testing.assert_array_equal(out[1, 0, 1], [0, 0, 0, 1])


def test_haploid_blank_leaves_the_second_track_empty():
    codes = _codes((REF, ABS), (ALT, ABS))
    out = _haps(codes, haploid_policy="blank")
    np.testing.assert_array_equal(out[1, 0, 1], [0, 0, 0, 1])
    np.testing.assert_array_equal(out[1, 1, 1], [0, 0, 0, 0])


def test_dosage_ignores_the_haploid_track_policy():
    """
    --haploid-policy governs the second haplotype *track*. Dosage is
    ploidy-aware: a hemizygous ALT is full-weight ALT either way.
    """
    codes = _codes((ALT, ABS))
    np.testing.assert_array_equal(_dosage(codes)[0, 1], [0, 0, 0, 1])
    for policy in ("duplicate", "blank"):
        haps = _haps(codes, haploid_policy=policy)
        np.testing.assert_array_equal(haps[0, 0, 1], [0, 0, 0, 1])
    assert _haps(codes, haploid_policy="blank")[0, 1, 1].sum() == 0
    np.testing.assert_array_equal(
        _haps(codes, haploid_policy="duplicate")[0, 1, 1], [0, 0, 0, 1]
    )


def test_consensus_ignores_the_haploid_track_policy():
    codes = _codes((ALT, ABS))
    out = _consensus(codes)
    np.testing.assert_array_equal(out[0, 1], [0, 0, 0, 1])


def test_haploid_is_not_a_half_call():
    """A hemizygous ALT and a half-called ALT must encode differently."""
    hemi = _dosage(_codes((ALT, ABS)))
    half = _dosage(_codes((ALT, MISS)))
    assert not np.array_equal(hemi[0, 1], half[0, 1])
    np.testing.assert_array_equal(hemi[0, 1], [0, 0, 0, 1])
    np.testing.assert_array_equal(half[0, 1], [0, 0, 0, .5])


# ------------------------------------------------------------------
# multiallelic
# ------------------------------------------------------------------

def test_multiallelic_haplotypes():
    """REF=C, ALT1=T, ALT2=A; genotype 1|2 carries two different ALTs."""
    codes = _codes((REF, ALT2), (ALT, ALT2), (ALT2, ALT2))
    out = _haps(codes, allele_bases=ALLELES_TRI)
    np.testing.assert_array_equal(out[0, 0, 1], [0, 1, 0, 0])   # REF  C
    np.testing.assert_array_equal(out[0, 1, 1], [1, 0, 0, 0])   # ALT2 A
    np.testing.assert_array_equal(out[1, 0, 1], [0, 0, 0, 1])   # ALT1 T
    np.testing.assert_array_equal(out[1, 1, 1], [1, 0, 0, 0])   # ALT2 A
    np.testing.assert_array_equal(out[2, 0, 1], [1, 0, 0, 0])


def test_multiallelic_dosage_splits_between_two_alts():
    codes = _codes((ALT, ALT2))
    out = _dosage(codes, allele_bases=ALLELES_TRI)
    # half on T (ALT1) and half on A (ALT2); nothing on the reference C
    np.testing.assert_array_equal(out[0, 1], [.5, 0, 0, .5])


# ------------------------------------------------------------------
# consensus / het collapse
# ------------------------------------------------------------------

def test_consensus_het_policy_reference():
    out = _consensus(het_policy="reference")
    np.testing.assert_array_equal(out[1, 1], [0, 1, 0, 0])   # het -> REF C
    np.testing.assert_array_equal(out[2, 1], [0, 0, 0, 1])   # hom-alt -> T


def test_consensus_het_policy_alternate():
    out = _consensus(het_policy="alternate")
    np.testing.assert_array_equal(out[1, 1], [0, 0, 0, 1])   # het -> ALT T
    np.testing.assert_array_equal(out[0, 1], [0, 1, 0, 0])   # hom-ref -> C


def test_consensus_het_policy_major_follows_frequency():
    codes = _codes((REF, ALT))
    ref_major = _consensus(
        codes, het_policy="major",
        allele_freq=np.array([[0.8, 0.2, 0, 0]], dtype=np.float32),
    )
    alt_major = _consensus(
        codes, het_policy="major",
        allele_freq=np.array([[0.2, 0.8, 0, 0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(ref_major[0, 1], [0, 1, 0, 0])
    np.testing.assert_array_equal(alt_major[0, 1], [0, 0, 0, 1])


def test_consensus_is_one_hot_where_a_base_is_known():
    """
    Consensus carries a single allele, so it is one-hot wherever one is
    known. Uncalled genotypes under missing_policy='zero' are the
    all-zero vector, as are masked and ambiguous-reference positions.
    """
    called = _consensus(het_policy="alternate", missing_policy="reference")
    np.testing.assert_allclose(called.sum(axis=2), 1.0)
    assert set(np.unique(called).tolist()) <= {0.0, 1.0}

    uncalled = _consensus(het_policy="alternate", missing_policy="zero")
    assert uncalled[3, 1].sum() == 0.0        # ".|." stays empty
    assert set(np.unique(uncalled).tolist()) <= {0.0, 1.0}


def test_consensus_multiallelic_het_picks_lower_alt_index():
    codes = _codes((ALT, ALT2))
    out = _consensus(
        codes, allele_bases=ALLELES_TRI, het_policy="alternate",
        allele_freq=np.array([[0.2, 0.5, 0.3, 0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(out[0, 1], [0, 0, 0, 1])   # ALT1 = T


def test_consensus_rejects_unknown_policy():
    with pytest.raises(ValueError, match="het_policy"):
        _consensus(het_policy="whatever")


# ------------------------------------------------------------------
# invariants, dtypes, layouts
# ------------------------------------------------------------------

@pytest.mark.parametrize("missing_policy", ["zero", "reference"])
@pytest.mark.parametrize("haploid_policy", ["duplicate", "blank"])
def test_dosage_is_the_superposition_of_the_two_haplotypes(
    missing_policy, haploid_policy
):
    """
    The invariant that ties the modes together. For a diploid genotype
    -- homozygous, heterozygous, multiallelic, half-called or missing --
    dosage is exactly the mean of the two haplotype tracks. For a
    hemizygous genotype dosage is the single allele at full weight,
    which equals the mean only under the 'duplicate' track policy.
    """
    pairs = [
        (REF, REF), (REF, ALT), (ALT, REF), (ALT, ALT),
        (REF, ALT2), (ALT, ALT2), (ALT2, ALT2),
        (MISS, REF), (REF, MISS), (MISS, ALT), (ALT, MISS),
        (MISS, MISS), (REF, ABS), (ALT, ABS), (ALT2, ABS),
    ]
    codes = _codes(*pairs)
    common = dict(
        ref_onehot=_ref(), variant_pos0=POS, allele_bases=ALLELES_TRI,
        genotypes=codes, sample_start=0, sample_end=len(pairs),
        missing_policy=missing_policy,
    )
    dose = encode_dosage_batch(**common)
    haps = encode_haplotypes(haploid_policy=haploid_policy, **common)

    hemizygous = np.array([b == ABS for _, b in pairs])
    # For diploid genotypes dosage is exactly the mean of the two tracks.
    np.testing.assert_allclose(
        dose[~hemizygous], haps[~hemizygous].mean(axis=1), rtol=0, atol=1e-6,
    )
    # A hemizygous call has one allele, so dosage is that allele at full
    # weight whatever the second *track* was told to carry. With
    # 'duplicate' the mean happens to agree; with 'blank' it does not,
    # and dosage is the one that is right.
    np.testing.assert_allclose(
        dose[hemizygous], haps[hemizygous, 0], rtol=0, atol=1e-6,
    )
    if haploid_policy == "duplicate":
        np.testing.assert_allclose(
            dose, haps.mean(axis=1), rtol=0, atol=1e-6,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_dtype_is_inherited_from_reference(dtype):
    ref = _ref().astype(dtype)
    assert _dosage(ref_onehot=ref).dtype == dtype
    assert _haps(ref_onehot=ref).dtype == dtype
    assert _consensus(ref_onehot=ref).dtype == dtype
    assert _dosage(ref_onehot=ref)[1, 1, 1] == dtype(0.5)


def test_channels_first_layout_is_a_transpose():
    sf = _dosage(output_layout="samples_first")
    cf = _dosage(output_layout="channels_first")
    assert cf.shape == (4, 4, 3)
    np.testing.assert_array_equal(cf, np.transpose(sf, (0, 2, 1)))


def test_haplotype_channels_first_layout():
    sf = _haps(output_layout="samples_first")
    cf = _haps(output_layout="channels_first")
    assert cf.shape == (4, 2, 4, 3)
    np.testing.assert_array_equal(cf, np.transpose(sf, (0, 1, 3, 2)))


def test_consensus_channels_first_layout():
    sf = _consensus(output_layout="samples_first")
    cf = _consensus(output_layout="channels_first")
    assert cf.shape == (4, 4, 3)
    np.testing.assert_array_equal(cf, np.transpose(sf, (0, 2, 1)))


@pytest.mark.parametrize("fn", [_dosage, _haps, _consensus])
def test_invalid_layout_rejected(fn):
    with pytest.raises(ValueError, match="output_layout"):
        fn(output_layout="sideways")


def test_sample_slicing_matches_full_batch():
    codes = _codes((REF, REF), (REF, ALT), (ALT, ALT), (MISS, MISS))
    full = encode_dosage_batch(
        ref_onehot=_ref(), variant_pos0=POS, allele_bases=ALLELES,
        genotypes=codes, sample_start=0, sample_end=4,
    )
    tail = encode_dosage_batch(
        ref_onehot=_ref(), variant_pos0=POS, allele_bases=ALLELES,
        genotypes=codes, sample_start=2, sample_end=4,
    )
    np.testing.assert_array_equal(tail, full[2:4])


def test_no_variants_returns_broadcast_reference():
    out = encode_dosage_batch(
        ref_onehot=_ref(),
        variant_pos0=np.array([], dtype=np.int32),
        allele_bases=np.zeros((0, 4), dtype=np.uint8),
        genotypes=np.zeros((0, 4), dtype=np.uint8),
        sample_start=0,
        sample_end=4,
    )
    for sample in range(4):
        np.testing.assert_array_equal(out[sample], _ref())


def test_multiple_variants_are_independent():
    ref = np.eye(4, dtype=np.float32)        # A C G T
    alleles = np.array(
        [[0, 2, NO_ALLELE, NO_ALLELE],       # pos 0: A -> G
         [3, 1, NO_ALLELE, NO_ALLELE]],      # pos 3: T -> C
        dtype=np.uint8,
    )
    codes = np.array(
        [[gt(ALT, ALT), gt(REF, REF)],
         [gt(REF, REF), gt(ALT, ALT)]],
        dtype=np.uint8,
    )
    out = encode_dosage_batch(
        ref_onehot=ref,
        variant_pos0=np.array([0, 3], dtype=np.int32),
        allele_bases=alleles,
        genotypes=codes,
        sample_start=0,
        sample_end=2,
    )
    np.testing.assert_array_equal(out[0, 0], [0, 0, 1, 0])   # hom-alt G
    np.testing.assert_array_equal(out[0, 3], [0, 0, 0, 1])   # hom-ref T
    np.testing.assert_array_equal(out[1, 0], [1, 0, 0, 0])   # hom-ref A
    np.testing.assert_array_equal(out[1, 3], [0, 1, 0, 0])   # hom-alt C


def test_tables_are_reusable_across_batches():
    """
    Lookup tables depend only on the variants and policies, so the
    pipeline builds them once and every batch reads the same arrays.
    """
    codes = _codes((REF, REF), (REF, ALT), (ALT, ALT), (MISS, MISS))
    tables = _build_variant_tables(
        allele_bases=ALLELES, variant_pos0=POS, sequence_length=3, dtype=np.float32, mode="dosage"
    )
    assert tables.primary.shape == (1, 36, 4)
    # Positions are converted to a list once, at build time.
    assert tables.positions == [1]
    assert tables.nbytes == tables.primary.nbytes
    assert len(tables) == 1

    whole = _encode_single_track(
        ref_onehot=_ref(), genotypes=codes,
        tables=tables, sample_start=0, sample_end=4,
    )
    halves = [
        _encode_single_track(
            ref_onehot=_ref(), genotypes=codes,
            tables=tables, sample_start=a, sample_end=b,
        )
        for a, b in ((0, 2), (2, 4))
    ]
    np.testing.assert_array_equal(np.concatenate(halves), whole)


def test_haplotype_tables_rejected_for_single_track():
    tables = _build_variant_tables(
        allele_bases=ALLELES, variant_pos0=POS, sequence_length=3, dtype=np.float32, mode="haplotypes"
    )
    with pytest.raises(ValueError, match=r"dosage.*consensus"):
        _encode_single_track(
            ref_onehot=_ref(),
            genotypes=_codes((REF, ALT)), tables=tables,
            sample_start=0, sample_end=1,
        )


def test_single_track_tables_rejected_for_haplotypes():
    tables = _build_variant_tables(
        allele_bases=ALLELES, variant_pos0=POS, sequence_length=3, dtype=np.float32, mode="dosage"
    )
    with pytest.raises(ValueError, match="built for mode"):
        _encode_haplotype_tracks(
            ref_onehot=_ref(),
            genotypes=_codes((REF, ALT)), tables=tables,
            sample_start=0, sample_end=1,
        )


def test_invalid_haploid_track_policy():
    with pytest.raises(ValueError, match="haploid_policy"):
        _build_variant_tables(
            allele_bases=ALLELES, variant_pos0=POS, sequence_length=3, dtype=np.float32, mode="haplotypes",
            haploid_policy="sideways",
        )
