"""Synthetic cross-file aliases: positive reconstructions and refusal paths."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pysam
import pysam.bcftools
import pytest
from conftest import _bgzip_and_index

from hottools import RunConfig, parse_region, run_hottools
from hottools.cli import main
from hottools.contigs import resolve_contig
from hottools.regions import read_mask

DATA = Path(__file__).parent / "data" / "contig_alias"


@pytest.fixture
def inputs(tmp_path):
    fasta = tmp_path / "ref.fa"
    fasta.write_text((DATA / "reference.fa").read_text())
    pysam.faidx(str(fasta))
    text = (DATA / "numeric.vcf").read_text()
    numeric = _bgzip_and_index(text, tmp_path / "numeric.vcf.gz")
    prefixed = _bgzip_and_index(
        text.replace("ID=1,", "ID=chr1,").replace("\n1\t", "\nchr1\t"),
        tmp_path / "prefixed.vcf.gz",
    )
    return fasta, numeric, prefixed


def config(inputs, tmp_path, name="out", **kwargs):
    fasta, numeric, _ = inputs
    return RunConfig(
        vcf_path=str(numeric), fasta_path=str(fasta),
        region=parse_region("chr1:1-32"), output_path=str(tmp_path / f"{name}.npy"),
        batch_size="none", **kwargs,
    )


@pytest.mark.parametrize("mode", ["dosage", "haplotypes", "consensus"])
@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("batch_size", [1, "none"])
def test_auto_equals_exact_and_records_mapping(inputs, tmp_path, mode, strand, batch_size):
    auto = config(inputs, tmp_path, "auto", mode=mode, contig_alias_policy="auto")
    auto.region = parse_region("chr1:1-32", strand)
    auto.batch_size = batch_size
    exact = config(inputs, tmp_path, "exact", mode=mode)
    exact.region = auto.region
    exact.batch_size = batch_size
    exact.vcf_path = str(inputs[2])
    metadata = run_hottools(auto)
    run_hottools(exact)
    np.testing.assert_array_equal(np.load(auto.output_path), np.load(exact.output_path))
    assert metadata["requested_chrom"] == "chr1"
    assert metadata["vcf_chrom"] == "1"
    assert metadata["fasta_chrom"] == "chr1"
    assert metadata["contig_alias_policy"] == "auto"
    if mode == "dosage" and strand == "+":
        np.testing.assert_array_equal(np.load(auto.output_path)[0, 4], [.5, 0, .5, 0])
    sidecar = json.loads(Path(auto.output_path + ".metadata.json").read_text())
    assert sidecar["vcf_chrom"] == "1"


@pytest.mark.parametrize("requested", ["1", "chr1"])
@pytest.mark.parametrize("anchor", ["center", "1:17", "chr1:17"])
def test_structural_aliases_masks_and_anchors(inputs, tmp_path, requested, anchor):
    mask = tmp_path / "mask.bed"
    mask.write_text("1\t4\t5\n")
    auto = config(inputs, tmp_path, "auto", mode="haplotypes",
                  variant_classes="all", contig_alias_policy="auto",
                  anchor=anchor, mask_path=str(mask))
    auto.region = parse_region(f"{requested}:1-32")
    exact = config(inputs, tmp_path, "exact", mode="haplotypes",
                   variant_classes="all", anchor="chr1:17")
    exact.vcf_path = str(inputs[2])
    if anchor == "center":
        exact.anchor = "center"
    exact_mask = tmp_path / "exact.bed"
    exact_mask.write_text("chr1\t4\t5\n")
    exact.mask_path = str(exact_mask)
    run_hottools(auto)
    run_hottools(exact)
    np.testing.assert_array_equal(np.load(auto.output_path), np.load(exact.output_path))


@pytest.mark.parametrize("vcf_prefixed", [False, True])
def test_resolves_either_fasta_direction(inputs, tmp_path, vcf_prefixed):
    fasta, numeric, prefixed = inputs
    fasta.write_text((DATA / "reference.fa").read_text().replace(">chr1", ">1"))
    Path(str(fasta) + ".fai").unlink()
    pysam.faidx(str(fasta))
    cfg = config(inputs, tmp_path, contig_alias_policy="auto")
    cfg.vcf_path = str(prefixed if vcf_prefixed else numeric)
    meta = run_hottools(cfg)
    assert meta["fasta_chrom"] == "1"
    np.testing.assert_array_equal(np.load(cfg.output_path)[0, 4], [.5, 0, .5, 0])


def test_exact_default_refuses_mismatch(inputs, tmp_path):
    cfg = config(inputs, tmp_path)
    with pytest.raises(ValueError, match="not present in the VCF"):
        run_hottools(cfg)
    assert not Path(cfg.output_path).exists()


def test_invalid_policy_refused_before_io(inputs, tmp_path):
    cfg = config(inputs, tmp_path, contig_alias_policy="guess")
    with pytest.raises(ValueError, match="contig_alias_policy"):
        cfg.validate()


@pytest.mark.parametrize("requested,names,expected", [
    ("1", ["1", "chr1"], "1"),
    ("chr1", ["1"], "1"),
    ("1", ["chr1"], "chr1"),
    ("MT", ["chrM"], "chrM"),
    ("chrX", ["X"], "X"),
    ("chrY", ["Y"], "Y"),
])
def test_supported_unique_aliases(requested, names, expected):
    assert resolve_contig(requested, names, source="test") == expected


@pytest.mark.parametrize("requested,names", [
    ("MT", ["M", "chrM"]),
    ("1", ["chr2"]),
    ("chr01", ["1"]),
    ("Chr1", ["1"]),
    ("chrUn_random", ["Un_random"]),
])
def test_ambiguous_or_nonstandard_aliases_refused(requested, names):
    with pytest.raises(ValueError):
        resolve_contig(requested, names, source="test")


def test_length_mismatch_refused_before_output(inputs, tmp_path):
    text = (DATA / "numeric.vcf").read_text().replace("length=64", "length=63")
    vcf = _bgzip_and_index(text, tmp_path / "bad.vcf.gz")
    cfg = config(inputs, tmp_path, contig_alias_policy="auto")
    cfg.vcf_path = str(vcf)
    with pytest.raises(ValueError, match="length mismatch"):
        run_hottools(cfg)
    assert not Path(cfg.output_path).exists()


def test_unknown_header_length_allowed(inputs, tmp_path):
    text = (DATA / "numeric.vcf").read_text().replace(",length=64", "")
    vcf = _bgzip_and_index(text, tmp_path / "unknown.vcf.gz")
    cfg = config(inputs, tmp_path, contig_alias_policy="auto")
    cfg.vcf_path = str(vcf)
    run_hottools(cfg)
    assert np.load(cfg.output_path).shape == (2, 32, 4)


def test_unrelated_anchor_still_refused(inputs, tmp_path):
    cfg = config(inputs, tmp_path, contig_alias_policy="auto", anchor="chr2:5")
    with pytest.raises(ValueError, match="does not match"):
        cfg.validate()


def test_ambiguous_mask_refused(tmp_path):
    mask = tmp_path / "mask.bed"
    mask.write_text("M\t1\t2\nchrM\t2\t3\n")
    with pytest.raises(ValueError, match="Ambiguous"):
        read_mask(mask, parse_region("MT:1-10"), contig_alias_policy="auto")


def test_mask_exact_precedence_and_other_contigs(tmp_path):
    mask = tmp_path / "mask.bed"
    mask.write_text("1\t1\t2\nchr1\t3\t4\n")
    result = read_mask(mask, parse_region("chr1:1-10"), contig_alias_policy="auto")
    assert np.flatnonzero(result).tolist() == [3]
    assert not read_mask(mask, parse_region("chr2:1-10"),
                         contig_alias_policy="auto").any()


def test_cli_region_file_aliases_and_dry_run(inputs, tmp_path, capsys):
    fasta, numeric, _ = inputs
    regions = tmp_path / "regions.bed"
    regions.write_text("1\t0\t32\n")
    base = [str(numeric), "--reference", str(fasta), "--region-file", str(regions),
            "--contig-alias-policy", "auto", "--batch-size", "none"]
    assert main([*base, "--output", str(tmp_path / "dry"), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["fasta_chrom"] == "chr1"
    assert not (tmp_path / "dry").exists()
    assert main([*base, "--output", str(tmp_path / "result")]) == 0
    assert np.load(tmp_path / "result/1_1_32.npy").shape == (2, 32, 4)


def test_indexed_bcf_aliases(inputs, tmp_path):
    with pysam.VariantFile(str(inputs[1])) as source:
        bcf = tmp_path / "input.bcf"
        with pysam.VariantFile(str(bcf), "wb", header=source.header) as target:
            for record in source:
                target.write(record)
    pysam.bcftools.index(str(bcf))
    cfg = config(inputs, tmp_path, contig_alias_policy="auto")
    cfg.vcf_path = str(bcf)
    run_hottools(cfg)
    np.testing.assert_array_equal(np.load(cfg.output_path)[1, 4], [0, 0, 1, 0])


def test_auto_preserves_ref_validation(inputs, tmp_path):
    text = (DATA / "numeric.vcf").read_text().replace("\t5\tsnv\tA\tG", "\t5\tsnv\tC\tG")
    bad = _bgzip_and_index(text, tmp_path / "wrong_ref.vcf.gz")
    cfg = config(inputs, tmp_path, contig_alias_policy="auto")
    cfg.vcf_path = str(bad)
    with pytest.raises(ValueError, match="REF mismatch"):
        run_hottools(cfg)
    assert not Path(cfg.output_path).exists()


def test_ambiguous_fasta_mapping_stops_pipeline(inputs, tmp_path):
    fasta = inputs[0]
    fasta.write_text(">M\n" + "A" * 64 + "\n>chrM\n" + "A" * 64 + "\n")
    Path(str(fasta) + ".fai").unlink()
    pysam.faidx(str(fasta))
    cfg = config(inputs, tmp_path, contig_alias_policy="auto")
    cfg.region = parse_region("MT:1-32")
    with pytest.raises(ValueError, match=r"Ambiguous.*FASTA"):
        run_hottools(cfg)
    assert not Path(cfg.output_path).exists()


def test_shared_reference_handle_and_mitochondrial_alias(tmp_path):
    from hottools import ReferenceReader

    fasta = tmp_path / "mito.fa"
    fasta.write_text(">chrM\n" + "A" * 64 + "\n")
    pysam.faidx(str(fasta))
    text = (DATA / "numeric.vcf").read_text().replace("ID=1,", "ID=MT,")
    text = text.replace("\n1\t", "\nMT\t").split("\nMT\t9")[0] + "\n"
    vcf = _bgzip_and_index(text, tmp_path / "mito.vcf.gz")
    with ReferenceReader(fasta) as reader:
        for i, name in enumerate(["M", "MT", "chrM"]):
            cfg = RunConfig(vcf_path=str(vcf), fasta_path=str(fasta),
                            region=parse_region(f"{name}:1-32"),
                            output_path=str(tmp_path / f"mito{i}.npy"),
                            batch_size="none", contig_alias_policy="auto",
                            reference_reader=reader)
            metadata = run_hottools(cfg)
            assert metadata["vcf_chrom"] == "MT"
            assert reader.interval("chrM", 1, 1) == "A"
            np.testing.assert_array_equal(np.load(cfg.output_path)[0, 4], [.5, 0, .5, 0])
