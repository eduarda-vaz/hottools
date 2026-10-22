"""Invalid options fail before I/O; unusual but meaningful combinations work."""

from dataclasses import fields

import numpy as np
import pytest
from conftest import _bgzip_and_index
from sv_helpers import Rec, run_paths, run_sv, write_fasta

from hottools.cli import main
from hottools.pipeline import RunConfig, run_hottools
from hottools.regions import Region
from hottools.variants import EmptyRegionWarning, extract_variants

# Every public configuration field is classified here; additions must be reviewed.
OPTION_GROUPS = {
    "inputs": {
        "vcf_path",
        "fasta_path",
        "region",
        "reference_reader",
        "contig_alias_policy",
        "samples",
        "mask_path",
    },
    "encoding": {
        "mode",
        "dtype",
        "output_layout",
        "missing_policy",
        "unphased_policy",
        "random_seed",
        "ref_mismatch_policy",
        "ambiguous_ref_policy",
        "haploid_policy",
        "ploidy_policy",
        "het_policy",
    },
    "selection": {
        "variant_classes",
        "biallelic_only",
        "mixed_record_policy",
        "pass_only",
        "min_qual",
        "min_af",
        "max_af",
        "af_source",
        "strict",
    },
    "structural": {"anchor", "anchor_deletion_policy", "overlap_policy"},
    "execution": {"batch_size", "memory_fraction", "threads", "dry_run"},
    "output": {
        "output_path",
        "output_format",
        "per_sample",
        "metadata_json",
        "overwrite",
        "h5_compression",
        "h5_chunk_samples",
        "h5_chunk_length",
    },
}

SITE_BAD = [
    ("threads", 0),
    ("threads", -1),
    ("threads", 1.5),
    ("threads", True),
    ("threads", "2"),
    ("min_qual", -1),
    ("min_qual", float("nan")),
    ("min_qual", float("inf")),
    ("min_qual", True),
    ("min_af", -0.1),
    ("min_af", 1.1),
    ("min_af", float("nan")),
    ("min_af", float("inf")),
    ("max_af", -0.1),
    ("max_af", 1.1),
    ("max_af", float("nan")),
    ("max_af", "0.5"),
    ("samples", "S1"),
    ("samples", []),
    ("samples", ["S1", "S1"]),
    ("samples", [""]),
    ("samples", [None]),
    ("samples", ["   "]),
]
OTHER_BAD = [
    ("contig_alias_policy", "guess"),
    ("contig_alias_policy", True),
    ("batch_size", 0),
    ("batch_size", -1),
    ("batch_size", True),
    ("batch_size", 1.5),
    ("batch_size", "2"),
    ("memory_fraction", 0),
    ("memory_fraction", -1),
    ("memory_fraction", 1.1),
    ("memory_fraction", float("nan")),
    ("memory_fraction", float("inf")),
    ("memory_fraction", True),
    ("h5_chunk_samples", 0),
    ("h5_chunk_samples", -1),
    ("h5_chunk_samples", 1.5),
    ("h5_chunk_length", 0),
    ("h5_chunk_length", True),
    ("h5_compression", "zip"),
    ("per_sample", "false"),
    ("pass_only", 1),
    ("biallelic_only", "false"),
    ("metadata_json", "false"),
    ("dry_run", 1),
    ("strict", "false"),
    ("anchor", "outside"),
    ("anchor", "chr1:65"),
    ("anchor", "0"),
    ("variant_classes", "bnd"),
    ("output_format", "fasta"),
    ("overlap_policy", "last"),
    ("anchor_deletion_policy", "skip"),
]


def config(tmp_path, **kwargs):
    return RunConfig(
        vcf_path="absent.vcf.gz",
        fasta_path="absent.fa",
        region=Region("chr1", 1, 64),
        output_path=str(tmp_path / "out.npy"),
        **kwargs,
    )


def test_every_config_field_has_a_coverage_category():
    recorded = [name for group in OPTION_GROUPS.values() for name in group]
    assert len(recorded) == len(set(recorded))
    assert set(recorded) == {field.name for field in fields(RunConfig)}


@pytest.mark.parametrize("name,value", SITE_BAD + OTHER_BAD)
def test_invalid_config_fails_before_reference_open(tmp_path, monkeypatch, name, value):
    import hottools.pipeline as pipeline

    def forbidden_open(*args, **kwargs):
        pytest.fail("Invalid configuration reached reference I/O")

    monkeypatch.setattr(pipeline, "ReferenceReader", forbidden_open)
    with pytest.raises(ValueError):
        run_hottools(config(tmp_path, **{name: value}))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name,value", SITE_BAD)
def test_invalid_extraction_options_fail_before_vcf_open(monkeypatch, name, value):
    import hottools.variants as variants

    def forbidden_open(*args, **kwargs):
        pytest.fail("Invalid extraction options reached VCF I/O")

    monkeypatch.setattr(variants, "VCF", forbidden_open)
    with pytest.raises(ValueError):
        extract_variants(
            vcf_path="absent.vcf.gz",
            chrom="chr1",
            start=1,
            end=64,
            reference_seq="A" * 64,
            **{name: value},
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(min_af=0.8, max_af=0.2),
        dict(mode="dosage", variant_classes="indel"),
        dict(mode="consensus", variant_classes="all"),
        dict(mode="haplotypes", het_policy="major"),
        dict(mode="dosage", het_policy="reference"),
        dict(output_format="npy", h5_compression="gzip"),
        dict(output_format="npz", h5_chunk_length=8),
        dict(output_format="pt", h5_chunk_samples=2),
    ],
)
def test_incompatible_configuration_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        config(tmp_path, **kwargs).validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(min_af=0, max_af=1, min_qual=0, threads=1),
        dict(min_af=0.5, max_af=0.5),
        dict(batch_size=None),
        dict(batch_size="auto"),
        dict(batch_size="none"),
        dict(batch_size=1),
        dict(memory_fraction=1),
        dict(biallelic_only=True, mixed_record_policy="apply-selected"),
        dict(mode="dosage", unphased_policy="missing", haploid_policy="blank"),
        dict(mode="consensus", unphased_policy="error", het_policy="major"),
        dict(
            output_format="h5",
            h5_compression="gzip",
            h5_chunk_samples=2,
            h5_chunk_length=8,
        ),
        dict(
            mode="haplotypes",
            variant_classes="all",
            anchor="chr1:32",
            anchor_deletion_policy="error",
        ),
        dict(dry_run=True, overwrite="replace", metadata_json=False),
    ],
)
def test_meaningful_combinations_still_validate(tmp_path, kwargs):
    assert config(tmp_path, **kwargs).validate() == np.dtype("float32")


@pytest.mark.parametrize(
    "flags,message",
    [
        (["--threads", "0"], "threads"),
        (["--min-af", ".8", "--max-af", ".2"], "min_af"),
        (["--min-qual", "nan"], "min_qual"),
        (["--memory-fraction", "nan"], "memory_fraction"),
        (["--h5-chunk-length", "-1"], "h5_chunk_length"),
        (["--h5-compression", "gzip"], "h5_"),
        (["--mode", "dosage", "--variant-classes", "all"], "haplotypes"),
        (["--mode", "haplotypes", "--het-policy", "major"], "consensus"),
        (["--samples", "S1,S1"], "Duplicate"),
    ],
)
def test_cli_invalid_options_checked_before_reference_open(
    tmp_path, monkeypatch, capsys, flags, message
):
    import hottools.cli as cli

    def forbidden_open(*args, **kwargs):
        pytest.fail("Invalid CLI configuration reached reference I/O")

    monkeypatch.setattr(cli, "ReferenceReader", forbidden_open)
    code = main(
        [
            "absent.vcf.gz",
            "--reference",
            "absent.fa",
            "--region",
            "chr1:1-64",
            "--output",
            str(tmp_path / "out.npy"),
            *flags,
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert message in captured.err
    assert "Traceback" not in captured.err
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["haplotypes", "dosage", "consensus"])
@pytest.mark.parametrize("missing", ["zero", "reference", "error"])
@pytest.mark.parametrize("unphased", ["error", "missing"])
def test_phase_and_missing_policy_interaction(tmp_path, mode, missing, unphased):
    kwargs = dict(
        end=64,
        mode=mode,
        variant_classes="snv",
        missing_policy=missing,
        unphased_policy=unphased,
    )
    if mode == "haplotypes" and (unphased == "error" or missing == "error"):
        with pytest.raises(ValueError, match="unphased"):
            run_sv(
                tmp_path,
                "A" * 64,
                [Rec(5, "A", "C", ("0/1",))],
                samples=["S1"],
                **kwargs,
            )
        assert not (tmp_path / "out.npy").exists()
    else:
        tensor, _ = run_sv(
            tmp_path, "A" * 64, [Rec(5, "A", "C", ("0/1",))], samples=["S1"], **kwargs
        )
        if mode == "haplotypes":
            expected = np.zeros((2, 4)) if missing == "zero" else np.eye(4)[[0, 0]]
            np.testing.assert_array_equal(tensor[0, :, 4], expected)
        elif mode == "dosage":
            np.testing.assert_array_equal(tensor[0, 4], [0.5, 0.5, 0, 0])
        else:
            np.testing.assert_array_equal(tensor[0, 4], [0, 1, 0, 0])


@pytest.mark.parametrize("mode", ["haplotypes", "dosage", "consensus"])
def test_biallelic_filter_takes_precedence_over_mixed_opt_in(tmp_path, mode):
    tensor, _ = run_sv(
        tmp_path,
        "A" * 64,
        [Rec(5, "A", "AT,C", ("2|1",))],
        samples=["S1"],
        end=64,
        mode=mode,
        variant_classes="snv",
        biallelic_only=True,
        mixed_record_policy="apply-selected",
    )
    np.testing.assert_array_equal(tensor[..., 0], 1)
    np.testing.assert_array_equal(tensor[..., 1:], 0)


@pytest.mark.parametrize("filter_kind", ["qual", "pass", "mixed", "biallelic"])
def test_excluded_record_cannot_trigger_genotype_error(tmp_path, filter_kind):
    fasta = write_fasta(tmp_path / "reference.fa", "A" * 64)
    header = (
        "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=64>\n"
        '##FILTER=<ID=q10,Description="Low quality">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
    )
    alt = (
        "C,AT"
        if filter_kind == "mixed"
        else "C,G"
        if filter_kind == "biallelic"
        else "C"
    )
    row = f"chr1\t5\t.\tA\t{alt}\t1\tq10\t.\tGT\t./1\n"
    vcf = _bgzip_and_index(header + row, tmp_path / "filtered.vcf.gz")
    kwargs = (
        dict(min_qual=2)
        if filter_kind == "qual"
        else dict(pass_only=True)
        if filter_kind == "pass"
        else dict(biallelic_only=True)
        if filter_kind == "biallelic"
        else {}
    )
    with pytest.warns(EmptyRegionWarning):
        tensor, _ = run_paths(
            tmp_path,
            fasta,
            vcf,
            end=64,
            variant_classes="snv",
            mode="haplotypes",
            missing_policy="error",
            **kwargs,
        )
    np.testing.assert_array_equal(tensor[..., 0], 1)
    np.testing.assert_array_equal(tensor[..., 1:], 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(start=1.5),
        dict(end=8.5),
        dict(start=True),
        dict(end=float("nan")),
        dict(chrom=1),
        dict(chrom=" "),
    ],
)
def test_region_rejects_noninteger_or_invalid_identity(kwargs):
    values = dict(chrom="chr1", start=1, end=64)
    values.update(kwargs)
    with pytest.raises(ValueError):
        Region(**values)


def small_input(tmp_path, *, info=".", qual="60", fmt="GT", call="1|1"):
    fasta = write_fasta(tmp_path / "reference.fa", "A" * 64)
    header = (
        "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=64>\n"
        '##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
    )
    row = f"chr1\t5\t.\tA\tC\t{qual}\tPASS\t{info}\t{fmt}\t{call}\n"
    return fasta, _bgzip_and_index(header + row, tmp_path / "input.vcf.gz")


@pytest.mark.parametrize("info", ["AF=nan", "AF=."])
def test_missing_info_af_falls_back_in_auto_mode(tmp_path, info):
    fasta, vcf = small_input(tmp_path, info=info, call="0|0")
    with pytest.warns(EmptyRegionWarning):
        tensor, _ = run_paths(
            tmp_path,
            fasta,
            vcf,
            end=64,
            mode="dosage",
            variant_classes="snv",
            min_af=0.5,
            af_source="auto",
        )
    np.testing.assert_array_equal(tensor[..., 0], 1)
    np.testing.assert_array_equal(tensor[..., 1:], 0)


@pytest.mark.parametrize("af", ["-0.1", "1.1", "inf"])
def test_invalid_info_af_is_an_input_error(tmp_path, af):
    fasta, vcf = small_input(tmp_path, info="AF=" + af)
    with pytest.raises(ValueError, match="INFO/AF"):
        run_paths(
            tmp_path,
            fasta,
            vcf,
            end=64,
            mode="dosage",
            variant_classes="snv",
            min_af=0.5,
        )
    assert not (tmp_path / "out.npy").exists()


def test_retained_record_requires_format_GT(tmp_path):
    fasta, vcf = small_input(tmp_path, fmt="DP", call="20")
    with pytest.raises(ValueError, match="FORMAT/GT"):
        run_paths(tmp_path, fasta, vcf, end=64, mode="dosage", variant_classes="snv")
    assert not (tmp_path / "out.npy").exists()


def test_nonfinite_QUAL_does_not_pass_quality_filter(tmp_path):
    fasta, vcf = small_input(tmp_path, qual="nan")
    with pytest.warns(EmptyRegionWarning):
        tensor, _ = run_paths(
            tmp_path,
            fasta,
            vcf,
            end=64,
            mode="dosage",
            variant_classes="snv",
            min_qual=10,
        )
    np.testing.assert_array_equal(tensor[..., 0], 1)
    np.testing.assert_array_equal(tensor[..., 1:], 0)


@pytest.mark.parametrize(
    "name,value",
    [
        ("region", None),
        ("region", "chr1:1-64"),
        ("region", {"start": 1}),
        ("vcf_path", None),
        ("fasta_path", ""),
        ("output_path", " "),
        ("mask_path", False),
    ],
)
def test_invalid_input_field_has_parameter_error(tmp_path, name, value):
    request = config(tmp_path)
    setattr(request, name, value)
    with pytest.raises(ValueError, match=name):
        request.validate()


@pytest.mark.parametrize("name", ["biallelic_only", "pass_only", "strict"])
def test_extraction_rejects_strings_used_as_boolean_flags(name):
    with pytest.raises(ValueError, match=name):
        extract_variants(
            vcf_path="absent.vcf.gz",
            chrom="chr1",
            start=1,
            end=64,
            reference_seq="A" * 64,
            **{name: "false"},
        )


@pytest.mark.parametrize("field", ["chunk_samples", "chunk_length"])
@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_h5_chunk_helper_rejects_invalid_overrides(field, bad):
    from hottools.writers import auto_h5_chunks

    with pytest.raises(ValueError, match=field):
        auto_h5_chunks((4, 64, 4), 4, sequence_axis=1, **{field: bad})


@pytest.mark.parametrize("compression", ["gzip", "lzf"])
@pytest.mark.parametrize("per_sample", [False, True])
@pytest.mark.parametrize("layout", ["samples_first", "channels_first"])
def test_valid_h5_configuration_reconstructs_expected_arrays(
    tmp_path, compression, per_sample, layout
):
    pytest.importorskip("h5py")
    tensor, _ = run_sv(
        tmp_path,
        "A" * 64,
        [Rec(5, "A", "C", ("0|1", "1|0"))],
        samples=["S1", "S2"],
        end=64,
        variant_classes="snv",
        fmt="h5",
        per_sample=per_sample,
        h5_compression=compression,
        h5_chunk_samples=999,
        h5_chunk_length=999,
        output_layout=layout,
        batch_size=1,
    )
    expected = np.broadcast_to(np.eye(4)[0], (2, 2, 64, 4)).copy()
    expected[0, 1, 4] = np.eye(4)[1]
    expected[1, 0, 4] = np.eye(4)[1]
    if layout == "channels_first":
        expected = expected.swapaxes(-1, -2)
    np.testing.assert_array_equal(tensor, expected)


@pytest.mark.parametrize("alt", ["<NON_REF>", "<DUP>", "<INV>", "N"])
@pytest.mark.parametrize("mixed_record_policy", ["exclude", "apply-selected"])
def test_nonreconstructible_alleles_in_mixed_snv_records(
    tmp_path, alt, mixed_record_policy
):
    tensor, _ = run_sv(
        tmp_path,
        "A" * 64,
        [Rec(5, "A", "C," + alt, ("1|2",))],
        samples=["S1"],
        end=64,
        variant_classes="snv",
        mixed_record_policy=mixed_record_policy,
    )
    expected = np.eye(4)[[0, 0] if mixed_record_policy == "exclude" else [1, 0]]
    np.testing.assert_array_equal(tensor[0, :, 4], expected)


@pytest.mark.parametrize("alt,gt", [("AT,C", "1|2"), ("C,AT", "1|2")])
@pytest.mark.parametrize(
    "het_policy,expected", [("reference", "A"), ("alternate", "C")]
)
def test_projected_consensus_choice_is_independent_of_alt_order(
    tmp_path, alt, gt, het_policy, expected
):
    tensor, _ = run_sv(
        tmp_path,
        "A" * 64,
        [Rec(5, "A", alt, (gt,))],
        samples=["S1"],
        end=64,
        mode="consensus",
        variant_classes="snv",
        mixed_record_policy="apply-selected",
        het_policy=het_policy,
    )
    np.testing.assert_array_equal(tensor[0, 4], np.eye(4)["ACGT".index(expected)])


def test_unphased_only_excluded_alts_become_phased_equivalent_reference(tmp_path):
    tensor, _ = run_sv(
        tmp_path,
        "A" * 64,
        [Rec(5, "A", "AT,<DEL>,C", ("1/2",), "END=8")],
        samples=["S1"],
        end=64,
        mode="haplotypes",
        variant_classes="snv",
        mixed_record_policy="apply-selected",
        unphased_policy="error",
    )
    np.testing.assert_array_equal(tensor[..., 0], 1)
    np.testing.assert_array_equal(tensor[..., 1:], 0)


def test_excluded_alt_does_not_claim_duplicate_position(tmp_path):
    tensor, _ = run_sv(
        tmp_path,
        "A" * 64,
        [Rec(5, "A", "AT,C", ("1|0",)), Rec(5, "A", "G", ("1|0",))],
        samples=["S1"],
        end=64,
        variant_classes="snv",
        mixed_record_policy="apply-selected",
    )
    np.testing.assert_array_equal(tensor[0, :, 4], np.eye(4)[[2, 0]])


@pytest.mark.parametrize("batch_size", [1, "none"])
def test_projected_ref_calls_count_together_for_major_consensus(tmp_path, batch_size):
    # After projection, TARGET=0|2 and the two donors are 0|0: REF=5, C=1.
    tensor, _ = run_sv(
        tmp_path,
        "A" * 64,
        [Rec(5, "A", "AT,C", ("1|2", "1|1", "0|0"))],
        samples=["TARGET", "EXCLUDED", "REF"],
        end=64,
        mode="consensus",
        variant_classes="snv",
        mixed_record_policy="apply-selected",
        het_policy="major",
        batch_size=batch_size,
    )
    np.testing.assert_array_equal(tensor[0, 4], [1, 0, 0, 0])


@pytest.mark.parametrize("alt", ["AT,C", "C,AT"])
@pytest.mark.parametrize(
    "het_policy,bcf_policy", [("reference", "R"), ("alternate", "A")]
)
def test_mixed_consensus_matches_native_projected_genotypes(
    tmp_path, alt, het_policy, bcf_policy, native_bcftools
):
    import gzip

    from sv_helpers import write_case
    from test_synthetic_cli_bcftools import project_literal_snvs

    bcftools = native_bcftools
    fasta, vcf = write_case(
        tmp_path, "A" * 64, [Rec(5, "A", alt, ("1|2",))], samples=["S1"]
    )
    with gzip.open(vcf, "rt") as handle:
        text = project_literal_snvs(handle.read(), "selected-snv")
    projected = _bgzip_and_index(text, tmp_path / "projected.vcf.gz")
    tensor, _ = run_paths(
        tmp_path,
        fasta,
        vcf,
        end=64,
        mode="consensus",
        variant_classes="snv",
        mixed_record_policy="apply-selected",
        het_policy=het_policy,
    )
    output = bcftools.consensus(
        "-f", str(fasta), "-s", "S1", "-H", bcf_policy, str(projected)
    )
    sequence = "".join(line for line in output.splitlines() if not line.startswith(">"))
    assert len(sequence) == 64
    expected = np.eye(4)[["ACGT".index(base) for base in sequence]]
    np.testing.assert_array_equal(tensor[0], expected)


@pytest.mark.parametrize("mode", ["haplotypes", "dosage", "consensus"])
def test_selected_snv_policy_still_requires_literal_single_base_REF(tmp_path, mode):
    with pytest.warns(EmptyRegionWarning):
        tensor, _ = run_sv(
            tmp_path,
            "A" * 64,
            [Rec(5, "AA", "CA", ("1|1",))],
            samples=["S1"],
            end=64,
            mode=mode,
            variant_classes="snv",
            mixed_record_policy="apply-selected",
        )
    np.testing.assert_array_equal(tensor[..., 0], 1)
    np.testing.assert_array_equal(tensor[..., 1:], 0)


def test_broader_classes_allow_normalized_substitution_representation(tmp_path):
    tensor, _ = run_sv(
        tmp_path,
        "A" * 64,
        [Rec(5, "AA", "CA", ("1|1",))],
        samples=["S1"],
        end=64,
        variant_classes="all",
        mixed_record_policy="apply-selected",
    )
    np.testing.assert_array_equal(tensor[0, :, 4], np.eye(4)[[1, 1]])
