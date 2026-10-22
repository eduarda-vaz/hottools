"""Small published fixtures with expectations independent of Hottools encoders."""

from pathlib import Path

import numpy as np
import pytest
from conftest import _bgzip_and_index
from sv_helpers import run_paths, write_fasta

from hottools.pipeline import RunConfig
from hottools.regions import Region

pytestmark = [
    pytest.mark.filterwarnings("ignore::hottools.variants.HaploidWarning"),
    pytest.mark.filterwarnings("ignore::hottools.variants.StructuralEditWarning"),
    pytest.mark.filterwarnings("ignore::hottools.variants.EmptyRegionWarning"),
    pytest.mark.filterwarnings("ignore::hottools.variants.MultiallelicWarning"),
]

DATA = Path(__file__).parent / "data" / "snv_edges"
BASES = "ACGT"
SAMPLES = ["S1", "S2", "S3", "HALF", "HAP", "MISSING", "REF", "HAP_MISSING"]
# Explicit expected allele bases, in ORIGINAL VCF allele-index order.
# None means reject the entire record. Excluded ALTs in opt-in mixed records
# map to REF; retaining ALT order prevents GT=2 from being mistaken for GT=1.
ALLELES = {
    "boundary_left": ["A", "C"],
    "two_alts": ["A", "C", "G"],
    "indel_first": ["A", "A", "C"],
    "symbolic_second": ["A", "G", "A"],
    "star_first": ["A", "A", "T"],
    "three_alts": ["A", "C", "G", "T"],
    "boundary_right": ["T", "A"],
}
MIXED = {"indel_first", "symbolic_second", "star_first"}
PROFILES = [
    dict(
        batch_size=None,
        threads=1,
        dtype="float32",
        strand="+",
        output_layout="samples_first",
        samples=None,
    ),
    dict(
        batch_size=1,
        threads=2,
        dtype="float16",
        strand="-",
        output_layout="channels_first",
        samples=["HAP", "S2", "HALF"],
    ),
    dict(
        batch_size=3,
        threads=1,
        dtype="float32",
        strand="-",
        output_layout="samples_first",
        samples=None,
    ),
    dict(
        batch_size="auto",
        threads=2,
        dtype="float16",
        strand="+",
        output_layout="channels_first",
        samples=None,
    ),
]
POLICIES = ["biallelic", "whole-snv", "selected-snv"]


@pytest.fixture
def edge_files(tmp_path):
    reference = "".join((DATA / "reference.fa").read_text().splitlines()[1:])
    fasta = write_fasta(tmp_path / "reference.fa", reference)
    phased = _bgzip_and_index(
        (DATA / "phased.vcf").read_text(), tmp_path / "phased.vcf.gz"
    )
    unphased = _bgzip_and_index(
        (DATA / "unphased.vcf").read_text(), tmp_path / "unphased.vcf.gz"
    )
    return fasta, phased, unphased


def options(policy):
    return dict(
        variant_classes="snv",
        biallelic_only=policy == "biallelic",
        mixed_record_policy="apply-selected" if policy == "selected-snv" else "exclude",
    )


def vector(base):
    return np.zeros(4) if base is None else np.eye(4)[BASES.index(base)]


def encode_gt(gt, bases, mode, missing, haploid, het="alternate"):
    """Simple scalar oracle: no Hottools packing, tables, or transforms."""
    indices = [None if a == "." else int(a) for a in gt.replace("/", "|").split("|")]
    # Unrequested ALT calls are REF genotypes, not merely REF-looking ALT slots.
    indices = [0 if a is not None and bases[a] == bases[0] else a for a in indices]
    values = [
        vector(
            bases[a] if a is not None else bases[0] if missing == "reference" else None
        )
        for a in indices
    ]
    if mode == "dosage":
        return np.mean(values, axis=0)
    if mode == "consensus":
        called = [a for a in indices if a is not None]
        if not called:
            return values[0]
        if het == "reference":
            chosen = min(called)
        else:
            chosen = min([a for a in called if a > 0] or called)
        return vector(bases[chosen])
    if len(values) == 1:
        values.append(values[0] if haploid == "duplicate" else vector(None))
    return np.stack(values)


def expected_phased(policy, mode, missing, haploid, samples=None, het="alternate"):
    selected = samples or SAMPLES
    reference = np.stack([vector(b) for b in "ACGT" * 16])
    shape = (
        (len(selected), 2, 64, 4) if mode == "haplotypes" else (len(selected), 64, 4)
    )
    expected = np.broadcast_to(reference, shape).copy()
    for line in (DATA / "phased.vcf").read_text().splitlines():
        if line.startswith("#"):
            continue
        _, pos, name, _, _, _, _, _, _, *gts = line.split("\t")
        if name not in ALLELES:  # Duplicate sites have explicit expectations below.
            continue
        bases = ALLELES[name]
        if policy == "biallelic" and len(bases) != 2:
            continue
        if policy != "selected-snv" and name in MIXED:
            continue
        for j, sample in enumerate(selected):
            value = encode_gt(
                gts[SAMPLES.index(sample)], bases, mode, missing, haploid, het
            )
            if mode == "haplotypes":
                expected[j, :, int(pos) - 1] = value
            else:
                expected[j, int(pos) - 1] = value
    # First applicable ALT on each haplotype: disjoint duplicate calls combine;
    # conflicting calls keep C. REF and missing calls do not claim a position.
    for j, sample in enumerate(selected):
        for pos, alt in [(25, "C"), (29, "C" if policy == "selected-snv" else "G")]:
            if sample == "REF":
                gt, bases = "0|0", ["A", alt]
            elif sample == "HAP_MISSING":
                gt, bases = ".", ["A", alt]
            elif sample == "HAP":
                gt, bases = "1", ["A", alt]
            elif pos == 25:
                gt, bases = "1|2", ["A", "C", "G"]
            else:
                gt, bases = "1|1", ["A", alt]
            value = encode_gt(gt, bases, mode, missing, haploid, het)
            if mode == "haplotypes":
                expected[j, :, pos - 1] = value
            else:
                expected[j, pos - 1] = value
    return expected


def orient(expected, mode, strand, layout):
    if strand == "-":
        expected = expected[..., ::-1, :][..., [3, 2, 1, 0]]
    return np.swapaxes(expected, -1, -2) if layout == "channels_first" else expected


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("mode", ["haplotypes", "dosage", "consensus"])
@pytest.mark.parametrize("missing", ["zero", "reference"])
@pytest.mark.parametrize("haploid", ["duplicate", "blank"])
@pytest.mark.parametrize(
    "profile", PROFILES, ids=["full", "subset-batch1", "batch3", "auto"]
)
def test_selection_configuration_matrix(
    tmp_path, edge_files, policy, mode, missing, haploid, profile
):
    fasta, vcf, _ = edge_files
    result, meta = run_paths(
        tmp_path,
        fasta,
        vcf,
        end=64,
        mode=mode,
        missing_policy=missing,
        haploid_policy=haploid,
        **options(policy),
        **profile,
    )
    expected = expected_phased(policy, mode, missing, haploid, profile["samples"])
    expected = orient(expected, mode, profile["strand"], profile["output_layout"])
    np.testing.assert_array_equal(result, expected.astype(profile["dtype"]))
    assert result.dtype == np.dtype(profile["dtype"])
    assert meta["samples"] == len(profile["samples"] or SAMPLES)
    assert meta["biallelic_only"] == (policy == "biallelic")
    assert meta["mixed_record_policy"] == options(policy)["mixed_record_policy"]


@pytest.mark.parametrize("mode", ["haplotypes", "dosage", "consensus"])
@pytest.mark.parametrize("missing", ["zero", "reference"])
@pytest.mark.parametrize("unphased", ["error", "missing"])
def test_unphased_policy_is_mode_specific(
    tmp_path, edge_files, mode, missing, unphased
):
    fasta, _, vcf = edge_files
    if mode == "haplotypes" and unphased == "error":
        with pytest.raises(ValueError, match="unphased"):
            run_paths(
                tmp_path,
                fasta,
                vcf,
                end=64,
                mode=mode,
                variant_classes="snv",
                missing_policy=missing,
                unphased_policy=unphased,
            )
        assert not (tmp_path / "out.npy").exists()
        return
    result, _ = run_paths(
        tmp_path,
        fasta,
        vcf,
        end=64,
        mode=mode,
        variant_classes="snv",
        missing_policy=missing,
        unphased_policy=unphased,
    )
    gts = ["0/1", "2/1", "1/1", "./1", "1", "./.", "0/0", "."]
    for j, gt in enumerate(gts):
        if mode == "haplotypes" and j in [0, 1]:
            gt = ".|."
        value = encode_gt(gt, ["A", "C", "G"], mode, missing, "duplicate")
        np.testing.assert_array_equal(
            result[j, :, 4] if mode == "haplotypes" else result[j, 4], value
        )


@pytest.mark.parametrize("mode", ["haplotypes", "dosage", "consensus"])
@pytest.mark.parametrize("policy", POLICIES)
def test_missing_error_does_not_publish_output(tmp_path, edge_files, mode, policy):
    fasta, vcf, _ = edge_files
    with pytest.raises(ValueError, match="missing allele"):
        run_paths(
            tmp_path,
            fasta,
            vcf,
            end=64,
            mode=mode,
            missing_policy="error",
            **options(policy),
        )
    assert not (tmp_path / "out.npy").exists()
    assert not list(tmp_path.glob("*.metadata.json"))


@pytest.mark.parametrize("fmt", ["npy", "npz", "h5", "pt"])
@pytest.mark.parametrize("per_sample", [False, True])
@pytest.mark.parametrize("mode", ["haplotypes", "dosage", "consensus"])
def test_edge_fixture_output_roundtrip(tmp_path, edge_files, fmt, per_sample, mode):
    if fmt in ["h5", "pt"]:
        pytest.importorskip("h5py" if fmt == "h5" else "torch")
    fasta, vcf, _ = edge_files
    result, _ = run_paths(
        tmp_path,
        fasta,
        vcf,
        end=64,
        mode=mode,
        fmt=fmt,
        per_sample=per_sample,
        missing_policy="reference",
        batch_size=3,
        **options("selected-snv"),
    )
    np.testing.assert_array_equal(
        result, expected_phased("selected-snv", mode, "reference", "duplicate")
    )


def test_record_policy_validates_before_file_io(tmp_path):
    config = RunConfig(
        vcf_path="missing.vcf.gz",
        fasta_path="missing.fa",
        region=Region("chr1", 1, 64),
        output_path=str(tmp_path / "out.npy"),
        mixed_record_policy="typo",
    )
    with pytest.raises(ValueError, match="mixed_record_policy"):
        config.validate()


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("het", ["reference", "alternate"])
def test_consensus_het_policy_with_original_alt_indices(
    tmp_path, edge_files, policy, het
):
    fasta, vcf, _ = edge_files
    result, _ = run_paths(
        tmp_path,
        fasta,
        vcf,
        end=64,
        mode="consensus",
        missing_policy="reference",
        het_policy=het,
        **options(policy),
    )
    np.testing.assert_array_equal(
        result, expected_phased(policy, "consensus", "reference", "duplicate", het=het)
    )


@pytest.mark.parametrize("batch_size", [None, 1, 3, "auto"])
@pytest.mark.parametrize(
    "samples,chosen",
    [
        (["MAJOR2", "MAJOR", "TARGET", "MINOR"], "G"),
        (["TARGET", "MINOR"], "C"),
    ],
)
def test_major_allele_depends_on_selected_cohort_not_batch(
    tmp_path, batch_size, samples, chosen
):
    from sv_helpers import Rec, write_case

    # All cohort: C=3, G=5. TARGET+MINOR only: C=3, G=1.
    names = ["MAJOR", "MAJOR2", "TARGET", "MINOR"]
    fasta, vcf = write_case(
        tmp_path,
        "A" * 64,
        [Rec(5, "A", "C,G", ("2|2", "2|2", "1|2", "1|1"))],
        samples=names,
    )
    result, _ = run_paths(
        tmp_path,
        fasta,
        vcf,
        end=64,
        mode="consensus",
        variant_classes="snv",
        het_policy="major",
        batch_size=batch_size,
        samples=samples,
    )
    np.testing.assert_array_equal(result[samples.index("TARGET"), 4], vector(chosen))


@pytest.mark.parametrize("mode", ["haplotypes", "dosage", "consensus"])
@pytest.mark.parametrize("haploid_policy", ["error", "skip"])
@pytest.mark.parametrize("biallelic_only", [False, True])
def test_record_filter_precedes_haploid_rejection(
    tmp_path, mode, haploid_policy, biallelic_only
):
    from sv_helpers import Rec, run_sv

    kwargs = dict(
        end=64,
        mode=mode,
        variant_classes="snv",
        haploid_policy=haploid_policy,
        biallelic_only=biallelic_only,
    )
    if haploid_policy == "error" and not biallelic_only:
        with pytest.raises(ValueError, match="haploid"):
            run_sv(
                tmp_path,
                "A" * 64,
                [Rec(5, "A", "C,G", ("2",))],
                samples=["H"],
                **kwargs,
            )
        assert not (tmp_path / "out.npy").exists()
    else:
        result, _ = run_sv(
            tmp_path, "A" * 64, [Rec(5, "A", "C,G", ("2",))], samples=["H"], **kwargs
        )
        np.testing.assert_array_equal(result[..., 0], 1)
        np.testing.assert_array_equal(result[..., 1:], 0)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("mode", ["haplotypes", "dosage", "consensus"])
def test_masked_duplicate_site_remains_zero(tmp_path, edge_files, policy, mode):
    fasta, vcf, _ = edge_files
    mask = tmp_path / "mask.bed"
    mask.write_text("chr1\t24\t25\nchr1\t28\t29\n")
    result, _ = run_paths(
        tmp_path,
        fasta,
        vcf,
        end=64,
        mode=mode,
        missing_policy="reference",
        mask_path=str(mask),
        **options(policy),
    )
    expected = expected_phased(policy, mode, "reference", "duplicate")
    expected[..., [24, 28], :] = 0
    np.testing.assert_array_equal(result, expected)


def test_extraction_record_policy_validates_before_vcf_open(monkeypatch):
    import hottools.variants as variants

    def forbidden_open(*args, **kwargs):
        pytest.fail("Invalid record policy reached VCF file I/O")

    monkeypatch.setattr(variants, "VCF", forbidden_open)
    with pytest.raises(ValueError, match="mixed_record_policy"):
        variants.extract_variants(
            vcf_path="missing.vcf.gz",
            chrom="chr1",
            start=1,
            end=64,
            reference_seq="ACGT" * 16,
            mixed_record_policy="typo",
        )


@pytest.mark.parametrize("policy", ["biallelic", "whole-snv"])
def test_strict_distinguishes_explicit_filter_from_partial_site(
    tmp_path, edge_files, policy
):
    fasta, vcf, _ = edge_files
    kwargs = dict(
        end=64,
        mode="haplotypes",
        missing_policy="reference",
        strict=True,
        **options(policy),
    )
    if policy == "whole-snv":
        # The SNV classifier discarded a mixed record at a retained position.
        with pytest.raises(ValueError, match="discarded record"):
            run_paths(tmp_path, fasta, vcf, **kwargs)
        assert not (tmp_path / "out.npy").exists()
    else:
        # Explicit biallelic filtering is deliberate and does not escalate.
        result, _ = run_paths(tmp_path, fasta, vcf, **kwargs)
        np.testing.assert_array_equal(
            result, expected_phased(policy, "haplotypes", "reference", "duplicate")
        )
