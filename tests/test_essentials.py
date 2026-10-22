"""
The cases every release must get right, end to end through the pipeline.

Each expected tensor here is written out from the documented semantics,
not recomputed by a second implementation, so a test failure points at a
behaviour change rather than at two pieces of code disagreeing.

Genotype patterns: 0|0, 0|1, 1|0, 1|1, unphased 0/1, ./., 0/., haploid 0,
haploid 1, multiallelic 1|2. Input problems: missing REF/ALT, an
ambiguous FASTA base, a REF mismatch. Run options: sample reordering,
consensus reference/alternate/major, both strands, float16 vs float32,
all four output formats, batch_size=1 vs all samples, per-sample vs
combined output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import (
    ALT_OF,
    AMBIGUOUS_POS,
    ESSENTIAL_GTS,
    ESSENTIAL_MULTI_POS,
    ESSENTIAL_SAMPLES,
    ESSENTIAL_SNV_POS,
    REF_LENGTH,
)

from hottools.pipeline import RunConfig, run_hottools
from hottools.regions import Region

BASES = "ACGT"
MODES = ("dosage", "haplotypes", "consensus")
FORMATS = ("npy", "npz", "h5", "pt")
N = len(ESSENTIAL_SAMPLES)
ROW = {name: i for i, name in enumerate(ESSENTIAL_SAMPLES)}
SNV = ESSENTIAL_SNV_POS - 1          # 0-based column
MULTI = ESSENTIAL_MULTI_POS - 1

ZERO = np.zeros(4)


def hot(base: str) -> np.ndarray:
    out = np.zeros(4)
    out[BASES.index(base)] = 1.0
    return out


def run(tmp_path: Path, vcf, *, name: str = "out", fmt: str = "npy",
        per_sample: bool = False, strand: str = "+", **kwargs):
    """Run over the whole 200 bp contig; return (tensor, metadata)."""
    out = tmp_path / (name if per_sample else f"{name}.{fmt}")
    config = RunConfig(
        vcf_path=str(vcf),
        fasta_path=kwargs.pop("fasta"),
        region=Region("chr1", 1, REF_LENGTH, strand=strand),
        output_path=str(out),
        output_format=fmt,
        per_sample=per_sample,
        **kwargs,
    )
    meta = run_hottools(config)
    return load_output(meta), meta


def _load_one(path: Path, fmt: str, per_sample: bool) -> np.ndarray:
    if fmt == "npy":
        return np.load(path)
    if fmt == "npz":
        return np.load(path)["sequences"]
    if fmt == "h5":
        import h5py
        with h5py.File(path) as f:
            return f["sequence" if per_sample else "sequences"][:]
    import torch
    return torch.load(path).numpy()


def load_output(meta: dict) -> np.ndarray:
    """Reassemble whatever a run wrote into one ``S x ...`` array."""
    files = [Path(p) for p in meta["output_files"]]
    arrays = [
        _load_one(p, meta["output_format"], meta["per_sample"])
        for p in files
    ]
    return np.stack(arrays) if meta["per_sample"] else np.concatenate(arrays)


@pytest.fixture
def snv_bases(reference_sequence):
    ref = reference_sequence[SNV]
    return ref, ALT_OF[ref]


@pytest.fixture
def multi_bases(reference_sequence):
    ref = reference_sequence[MULTI]
    others = [b for b in BASES if b != ref]
    return ref, others[0], others[1]


# --------------------------------------------------------------------
# genotype patterns
# --------------------------------------------------------------------

def expected_haplotypes(r: str, a: str, missing_policy: str) -> dict:
    """Per sample: (track 1, track 2) at the biallelic site."""
    miss = hot(r) if missing_policy == "reference" else ZERO
    return {
        "P00": (hot(r), hot(r)),
        "P01": (hot(r), hot(a)),
        "P10": (hot(a), hot(r)),
        "P11": (hot(a), hot(a)),
        "MISS": (miss, miss),
        "HALF": (hot(r), miss),
        # Haploid: the single allele is duplicated onto track 2.
        "HAP0": (hot(r), hot(r)),
        "HAP1": (hot(a), hot(a)),
        "MULTI": (hot(r), hot(r)),
    }


def expected_dosage(r: str, a: str, missing_policy: str) -> dict:
    """Mean over the alleles that exist; an uncalled allele still counts."""
    miss = hot(r) if missing_policy == "reference" else ZERO
    return {
        "P00": hot(r),
        "P01": 0.5 * hot(r) + 0.5 * hot(a),
        "P10": 0.5 * hot(r) + 0.5 * hot(a),
        "P11": hot(a),
        "MISS": miss,
        # Half-called diploid: the called REF keeps half weight under
        # 'zero' - unlike a hemizygous call, it is not full weight.
        "HALF": 0.5 * hot(r) + 0.5 * miss,
        # Hemizygous: the single allele is the whole dosage.
        "HAP0": hot(r),
        "HAP1": hot(a),
        "MULTI": hot(r),
    }


def expected_consensus(r: str, a: str, missing_policy: str,
                       het_policy: str) -> dict:
    miss = hot(r) if missing_policy == "reference" else ZERO
    het = hot(r) if het_policy == "reference" else hot(a)
    return {
        "P00": hot(r), "P01": het, "P10": het, "P11": hot(a),
        "MISS": miss,
        # One known allele: that allele, whatever the het policy.
        "HALF": hot(r),
        "HAP0": hot(r), "HAP1": hot(a), "MULTI": hot(r),
    }


@pytest.mark.parametrize("missing_policy", ["zero", "reference"])
def test_haplotype_tracks_for_every_pattern(
    tmp_path, fasta_path, essential_vcf_path, snv_bases, missing_policy
):
    t, meta = run(tmp_path, essential_vcf_path, fasta=fasta_path,
                  mode="haplotypes", missing_policy=missing_policy)
    assert t.shape == (N, 2, REF_LENGTH, 4)
    assert meta["sample_ids"] == ESSENTIAL_SAMPLES
    for sample, (h1, h2) in expected_haplotypes(
        *snv_bases, missing_policy
    ).items():
        np.testing.assert_array_equal(
            t[ROW[sample], 0, SNV], h1, err_msg=f"{sample} track 1"
        )
        np.testing.assert_array_equal(
            t[ROW[sample], 1, SNV], h2, err_msg=f"{sample} track 2"
        )


@pytest.mark.parametrize("missing_policy", ["zero", "reference"])
def test_dosage_for_every_pattern(
    tmp_path, fasta_path, essential_vcf_path, snv_bases, missing_policy
):
    t, _ = run(tmp_path, essential_vcf_path, fasta=fasta_path,
               mode="dosage", missing_policy=missing_policy)
    assert t.shape == (N, REF_LENGTH, 4)
    for sample, want in expected_dosage(*snv_bases, missing_policy).items():
        np.testing.assert_array_equal(t[ROW[sample], SNV], want,
                                      err_msg=sample)


@pytest.mark.parametrize("missing_policy", ["zero", "reference"])
@pytest.mark.parametrize("het_policy", ["reference", "alternate"])
def test_consensus_for_every_pattern(
    tmp_path, fasta_path, essential_vcf_path, snv_bases,
    missing_policy, het_policy,
):
    t, meta = run(tmp_path, essential_vcf_path, fasta=fasta_path,
                  mode="consensus", missing_policy=missing_policy,
                  het_policy=het_policy)
    assert meta["het_policy"] == het_policy
    for sample, want in expected_consensus(
        *snv_bases, missing_policy, het_policy
    ).items():
        np.testing.assert_array_equal(t[ROW[sample], SNV], want,
                                      err_msg=sample)


def test_haploid_blank_leaves_the_second_track_empty(
    tmp_path, fasta_path, essential_vcf_path, snv_bases
):
    _, a = snv_bases
    t, _ = run(tmp_path, essential_vcf_path, fasta=fasta_path,
               mode="haplotypes", haploid_policy="blank")
    np.testing.assert_array_equal(t[ROW["HAP1"], 0, SNV], hot(a))
    np.testing.assert_array_equal(t[ROW["HAP1"], 1, SNV], ZERO)
    # Diploid samples are unaffected by the haploid policy.
    np.testing.assert_array_equal(t[ROW["P01"], 1, SNV], hot(a))


def test_multiallelic_one_two_in_every_mode(
    tmp_path, fasta_path, essential_vcf_path, multi_bases
):
    r, a1, a2 = multi_bases
    row = ROW["MULTI"]
    haps, meta = run(tmp_path, essential_vcf_path, fasta=fasta_path,
                     name="h", mode="haplotypes")
    assert meta["variant_multiallelic_records"] == 1
    np.testing.assert_array_equal(haps[row, 0, MULTI], hot(a1))
    np.testing.assert_array_equal(haps[row, 1, MULTI], hot(a2))

    dose, _ = run(tmp_path, essential_vcf_path, fasta=fasta_path,
                  name="d", mode="dosage")
    np.testing.assert_array_equal(dose[row, MULTI],
                                  0.5 * hot(a1) + 0.5 * hot(a2))
    # Every other sample is REF there.
    np.testing.assert_array_equal(dose[ROW["P11"], MULTI], hot(r))

    for policy in ("reference", "alternate", "major"):
        cons, _ = run(tmp_path, essential_vcf_path, fasta=fasta_path,
                      name=f"c_{policy}", mode="consensus",
                      het_policy=policy)
        # Two different ALTs: no REF to choose, so the lower-indexed ALT
        # wins under every policy (and the frequency tie under 'major').
        np.testing.assert_array_equal(cons[row, MULTI], hot(a1),
                                      err_msg=policy)


def test_unphased_heterozygote(
    tmp_path, fasta_path, unphased_het_vcf_path, snv_bases
):
    r, a = snv_bases
    # Haplotype mode refuses to invent phase.
    with pytest.raises(ValueError, match="nphased"):
        run(tmp_path, unphased_het_vcf_path, fasta=fasta_path,
            name="refused", mode="haplotypes")
    assert not (tmp_path / "refused.npy").exists()

    # ... or treats the call as missing, when told to.
    haps, meta = run(tmp_path, unphased_het_vcf_path, fasta=fasta_path,
                     name="as_missing", mode="haplotypes",
                     unphased_policy="missing")
    assert meta["variant_unphased_heterozygotes"] == 1
    np.testing.assert_array_equal(haps[0, 0, SNV], ZERO)
    np.testing.assert_array_equal(haps[0, 1, SNV], ZERO)
    np.testing.assert_array_equal(haps[1, 1, SNV], hot(a))   # phased S2

    # Dosage and consensus need no phase.
    dose, _ = run(tmp_path, unphased_het_vcf_path, fasta=fasta_path,
                  name="dose", mode="dosage")
    np.testing.assert_array_equal(dose[0, SNV], 0.5 * hot(r) + 0.5 * hot(a))
    cons, _ = run(tmp_path, unphased_het_vcf_path, fasta=fasta_path,
                  name="cons", mode="consensus", het_policy="alternate")
    np.testing.assert_array_equal(cons[0, SNV], hot(a))


def test_consensus_major_follows_the_cohort(
    tmp_path, fasta_path, major_vcf_path, reference_sequence
):
    t, _ = run(tmp_path, major_vcf_path, fasta=fasta_path,
               mode="consensus", het_policy="major")
    # S1 is 0|1 at both sites. ALT is the majority at 20 (7 of 8
    # haplotypes) and the minority at 40 (1 of 8).
    np.testing.assert_array_equal(t[0, 19], hot(ALT_OF[reference_sequence[19]]))
    np.testing.assert_array_equal(t[0, 39], hot(reference_sequence[39]))


# --------------------------------------------------------------------
# input problems
# --------------------------------------------------------------------

def test_missing_ref_or_alt_is_dropped_and_counted(
    tmp_path, fasta_path, missing_ref_alt_vcf_path, reference_sequence
):
    t, meta = run(tmp_path, missing_ref_alt_vcf_path, fasta=fasta_path,
                  mode="haplotypes")
    # ALT '.', REF 'N' and REF '.' are refused; the SNV at 60 survives.
    assert meta["variants_kept"] == 1
    assert meta["variant_non_snv_records"] == 3
    for col in (19, 29, 39):
        np.testing.assert_array_equal(t[0, 0, col], hot(reference_sequence[col]))
    np.testing.assert_array_equal(
        t[0, 0, 59], hot(ALT_OF[reference_sequence[59]])
    )


def test_ambiguous_fasta_base(tmp_path, fasta_path, ambiguous_site_vcf_path):
    col = AMBIGUOUS_POS - 1
    with pytest.raises(ValueError, match="Ambiguous FASTA"):
        run(tmp_path, ambiguous_site_vcf_path, fasta=fasta_path,
            name="err", mode="haplotypes")
    assert not (tmp_path / "err.npy").exists()

    with pytest.warns(RuntimeWarning, match="Ambiguous FASTA"):
        run(tmp_path, ambiguous_site_vcf_path, fasta=fasta_path,
            name="warn", mode="haplotypes", ambiguous_ref_policy="warn")

    t, meta = run(tmp_path, ambiguous_site_vcf_path, fasta=fasta_path,
                  name="skip", mode="haplotypes", ambiguous_ref_policy="skip")
    assert meta["variants_kept"] == 1
    assert meta["variant_ambiguous_fasta"] == 1
    # An N has no channel to set: all-zero for every sample and track,
    # and the variant on it is not applied.
    np.testing.assert_array_equal(t[:, :, col], np.zeros((4, 2, 4)))
    assert meta["reference_ambiguous_bases"] == 1


def test_ref_mismatch(tmp_path, fasta_path, ref_mismatch_vcf_path,
                      reference_sequence):
    with pytest.raises(ValueError, match="REF mismatch"):
        run(tmp_path, ref_mismatch_vcf_path, fasta=fasta_path,
            name="err", mode="dosage")
    assert not (tmp_path / "err.npy").exists()

    with pytest.warns(RuntimeWarning, match="REF mismatch"):
        run(tmp_path, ref_mismatch_vcf_path, fasta=fasta_path,
            name="warn", mode="dosage", ref_mismatch_policy="warn")

    t, meta = run(tmp_path, ref_mismatch_vcf_path, fasta=fasta_path,
                  name="skip", mode="dosage", ref_mismatch_policy="skip")
    assert meta["variants_kept"] == 1
    assert meta["variant_ref_mismatch"] == 1
    # The mismatching record is dropped, so position 20 is the FASTA base.
    np.testing.assert_array_equal(t[0, 19], hot(reference_sequence[19]))
    np.testing.assert_array_equal(t[0, 29], hot(ALT_OF[reference_sequence[29]]))


# --------------------------------------------------------------------
# run options
# --------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_requested_sample_order_is_the_row_order(
    tmp_path, fasta_path, essential_vcf_path, mode
):
    full, _ = run(tmp_path, essential_vcf_path, fasta=fasta_path,
                  name="all", mode=mode)
    wanted = ["HAP1", "P00", "MULTI", "P01"]
    sub, meta = run(tmp_path, essential_vcf_path, fasta=fasta_path,
                    name="sub", mode=mode, samples=wanted)
    assert meta["sample_ids"] == wanted
    np.testing.assert_array_equal(sub, full[[ROW[s] for s in wanted]])


def _revcomp(t: np.ndarray, mode: str, layout: str) -> np.ndarray:
    """Reverse the sequence axis and complement A<->T, C<->G.

    With channels in A,C,G,T order the complement of channel i is 3-i,
    so complementing is a reversal of the channel axis.
    """
    haps = mode == "haplotypes"
    if layout == "samples_first":
        seq_axis, chan_axis = (2, 3) if haps else (1, 2)
    else:
        seq_axis, chan_axis = (3, 2) if haps else (2, 1)
    return np.flip(np.flip(t, axis=seq_axis), axis=chan_axis)


@pytest.mark.parametrize("layout", ["samples_first", "channels_first"])
@pytest.mark.parametrize("mode", MODES)
def test_minus_strand_is_the_reverse_complement_of_plus(
    tmp_path, fasta_path, essential_vcf_path, mode, layout
):
    plus, meta_p = run(tmp_path, essential_vcf_path, fasta=fasta_path,
                       name="plus", mode=mode, output_layout=layout)
    minus, meta_m = run(tmp_path, essential_vcf_path, fasta=fasta_path,
                        name="minus", mode=mode, output_layout=layout,
                        strand="-")
    assert meta_p["strand"] == "+" and not meta_p["reverse_complement"]
    assert meta_m["strand"] == "-" and meta_m["reverse_complement"]
    np.testing.assert_array_equal(minus, _revcomp(plus, mode, layout))
    assert not np.array_equal(minus, plus)


@pytest.mark.parametrize("mode", MODES)
def test_float16_matches_float32(tmp_path, fasta_path, essential_vcf_path,
                                 mode):
    t32, _ = run(tmp_path, essential_vcf_path, fasta=fasta_path,
                   name="f32", mode=mode, dtype="float32")
    t16, m16 = run(tmp_path, essential_vcf_path, fasta=fasta_path,
                   name="f16", mode=mode, dtype="float16")
    assert t32.dtype == np.float32 and t16.dtype == np.float16
    assert m16["dtype"] == "float16"
    # Every value the encoder produces (0, 0.5, 1) is exact in float16.
    np.testing.assert_array_equal(t16.astype(np.float32), t32)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fmt", FORMATS)
def test_every_format_batching_and_layout_give_the_same_tensor(
    tmp_path, fasta_path, essential_vcf_path, fmt, mode
):
    """npy/npz/h5/pt x batch_size 1 or all x per-sample or combined."""
    if fmt == "pt":
        pytest.importorskip("torch")
    reference, _ = run(tmp_path, essential_vcf_path, fasta=fasta_path,
                       name="ref", fmt="npy", mode=mode, batch_size=N)

    variants = {
        "combined, one batch": {"batch_size": N},
        "combined, batch_size=1": {"batch_size": 1},
        "per-sample, one batch": {"batch_size": N, "per_sample": True},
        "per-sample, batch_size=1": {"batch_size": 1, "per_sample": True},
    }
    for i, (label, extra) in enumerate(variants.items()):
        t, meta = run(tmp_path, essential_vcf_path, fasta=fasta_path,
                      name=f"v{i}", fmt=fmt, mode=mode, **extra)
        assert meta["output_format"] == fmt
        if extra.get("per_sample"):
            assert meta["n_output_files"] == N
        elif fmt in {"npz", "pt"} and extra["batch_size"] == 1:
            # Not appendable: split into one file per batch.
            assert meta["n_output_files"] == N
        else:
            assert meta["n_output_files"] == 1
        np.testing.assert_array_equal(t, reference, err_msg=label)


def test_essential_fixture_covers_every_listed_pattern():
    """Guard against the fixture drifting away from the checklist."""
    assert set(ESSENTIAL_GTS.values()) >= {
        "0|0", "0|1", "1|0", "1|1", "./.", "0/.", "0", "1",
    }
