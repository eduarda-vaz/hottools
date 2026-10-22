"""Targeted execution and input-format gaps from the external suite review."""
import numpy as np
import pytest
from test_synthetic_configuration_matrix import edge_files as _edge_files
from test_synthetic_configuration_matrix import options

from hottools.pipeline import RunConfig, run_hottools
from hottools.regions import Region, parse_region
from hottools.variants import EmptyRegionWarning

edge_files = _edge_files


def execute(tmp_path, vcf, fasta, name, **kwargs):
    path = tmp_path / f"{name}.npy"
    config = RunConfig(vcf_path=str(vcf), fasta_path=str(fasta),
                       region=Region("chr1", 1, 200), output_path=str(path),
                       batch_size="none", **kwargs)
    metadata = run_hottools(config)
    return np.load(path), metadata


@pytest.mark.parametrize("threads", [4, 8])
@pytest.mark.parametrize("mode", ["dosage", "haplotypes", "consensus"])
def test_additional_thread_counts_preserve_output(
    tmp_path, clean_vcf_path, fasta_path, threads, mode
):
    expected, _ = execute(tmp_path, clean_vcf_path, fasta_path, "serial",
                          threads=1, mode=mode)
    actual, _ = execute(tmp_path, clean_vcf_path, fasta_path, "threaded",
                        threads=threads, mode=mode)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("policy", ["biallelic", "whole-snv", "selected-snv"])
@pytest.mark.parametrize("mode", ["dosage", "haplotypes", "consensus"])
def test_bcf_mixed_record_policies_match_vcf(tmp_path, edge_files, policy, mode):
    import pysam
    import pysam.bcftools

    fasta, vcf, _ = edge_files
    bcf = tmp_path / "mixed.bcf"
    with pysam.VariantFile(str(vcf)) as source:
        with pysam.VariantFile(str(bcf), "wb", header=source.header) as target:
            for record in source:
                target.write(record)
    pysam.bcftools.index("-f", str(bcf))
    outputs = []
    for index, input_path in enumerate((vcf, bcf)):
        output = tmp_path / f"mixed{index}.npy"
        run_hottools(RunConfig(
            vcf_path=str(input_path), fasta_path=str(fasta),
            region=Region("chr1", 1, 64, "-"), output_path=str(output),
            mode=mode, samples=["S2", "S1"], batch_size=1, threads=4,
            output_layout="channels_first", **options(policy)))
        outputs.append(np.load(output))
    np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize("mode", ["dosage", "haplotypes", "consensus"])
@pytest.mark.parametrize("layout", ["samples_first", "channels_first"])
def test_mask_with_per_sample_output(
    tmp_path, clean_vcf_path, fasta_path, mode, layout
):
    mask = tmp_path / "mask.bed"
    mask.write_text("chr1\t0\t3\nchr1\t9\t10\n")
    output = tmp_path / "samples"
    metadata = run_hottools(RunConfig(
        vcf_path=str(clean_vcf_path), fasta_path=str(fasta_path),
        region=Region("chr1", 1, 200), output_path=str(output),
        per_sample=True, samples=["S2", "S1"], mode=mode,
        output_layout=layout, batch_size=1, mask_path=str(mask)))
    combined, _ = execute(tmp_path, clean_vcf_path, fasta_path, "combined",
                          mode=mode, output_layout=layout,
                          samples=["S2", "S1"], mask_path=str(mask))
    assert metadata["masked_bases"] == 4
    for index, sample in enumerate(("S2", "S1")):
        track = np.load(output / f"{sample}.npy")
        np.testing.assert_array_equal(track, combined[index])
        positions = [0, 1, 2, 9]
        masked = track[..., positions] if layout == "channels_first" else track[..., positions, :]
        assert np.count_nonzero(masked) == 0


@pytest.mark.parametrize("het_policy", ["reference", "alternate", "major"])
@pytest.mark.parametrize("strand", ["+", "-"])
def test_empty_consensus_is_reference(
    tmp_path, clean_vcf_path, fasta_path, het_policy, strand
):
    import pysam
    output = tmp_path / "empty.npy"
    with pytest.warns(EmptyRegionWarning):
        metadata = run_hottools(RunConfig(
            vcf_path=str(clean_vcf_path), fasta_path=str(fasta_path),
            region=Region("chr1", 150, 200, strand), output_path=str(output),
            mode="consensus", het_policy=het_policy))
    with pysam.FastaFile(str(fasta_path)) as reference:
        sequence = reference.fetch("chr1", 149, 200)
    if strand == "-":
        sequence = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    expected = np.eye(4, dtype=np.float32)[["ACGT".index(base) for base in sequence]]
    actual = np.load(output)
    assert metadata["variants_kept"] == 0
    np.testing.assert_array_equal(actual, np.broadcast_to(expected, actual.shape))


def test_unsupported_ploidy_policy_is_rejected():
    with pytest.raises(ValueError, match="ploidy_policy"):
        RunConfig(vcf_path="missing.vcf", fasta_path="missing.fa",
                  region=Region("chr1", 1, 2), output_path="unused.npy",
                  ploidy_policy="warn").validate()


def test_single_region_cannot_span_contigs():
    with pytest.raises(ValueError):
        parse_region("chr1:1-chr2:10")


@pytest.mark.parametrize("strand", ["+", "-"])
def test_bcf_structural_edits_match_vcf(tmp_path, strand):
    import pysam
    import pysam.bcftools
    from sv_helpers import Rec, run_paths, write_case
    fasta, vcf = write_case(tmp_path, "A" * 200, [
        Rec(20, "AAA", "A", ("0|1",)),
        Rec(40, "A", "ATG", ("1|0",)),
        Rec(60, "AA", "CG", ("1|1",)),
        Rec(80, "A", "<DEL>", ("1|0",), "END=83"),
    ], samples=["S1"])
    bcf = tmp_path / "structural.bcf"
    with pysam.VariantFile(str(vcf)) as source:
        with pysam.VariantFile(str(bcf), "wb", header=source.header) as target:
            for record in source:
                target.write(record)
    pysam.bcftools.index("-f", str(bcf))
    expected, _ = run_paths(tmp_path, fasta, vcf, name="vcf", strand=strand)
    actual, _ = run_paths(tmp_path, fasta, bcf, name="bcf", strand=strand)
    np.testing.assert_array_equal(actual, expected)


def test_region_file_across_two_contigs(tmp_path, capsys):
    import pysam
    from conftest import _bgzip_and_index

    from hottools.cli import main
    fasta = tmp_path / "two.fa"
    fasta.write_text(">chr1\nAAAAAAAA\n>chr2\nCCCCCCCC\n")
    pysam.faidx(str(fasta))
    vcf = _bgzip_and_index(
        '##fileformat=VCFv4.2\n##contig=<ID=chr1,length=8>\n'
        '##contig=<ID=chr2,length=8>\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n'
        'chr1\t2\t.\tA\tT\t.\tPASS\t.\tGT\t1|1\n'
        'chr2\t3\t.\tC\tG\t.\tPASS\t.\tGT\t1|1\n',
        tmp_path / "two.vcf.gz")
    regions = tmp_path / "regions.txt"
    regions.write_text("chr1 1 8\nchr2 1 8\n")
    output = tmp_path / "regions"
    assert main([str(vcf), "--reference", str(fasta), "--region-file",
                 str(regions), "-o", str(output), "--mode", "consensus",
                 "--threads", "4"]) == 0
    for contig, sequence in (("chr1", "ATAAAAAA"), ("chr2", "CCGCCCCC")):
        expected = np.eye(4, dtype=np.float32)[["ACGT".index(b) for b in sequence]]
        np.testing.assert_array_equal(np.load(output / f"{contig}_1_8.npy")[0], expected)
