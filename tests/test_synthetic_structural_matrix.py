"""Readable structural VCF tested against the independent text-edit oracle."""

import numpy as np
import pytest
from conftest import _bgzip_and_index
from sv_helpers import (
    Rec,
    allele_of,
    haps,
    oracle_window,
    revcomp,
    run_paths,
    write_fasta,
)
from test_synthetic_configuration_matrix import DATA, vector

pytestmark = [
    pytest.mark.filterwarnings("ignore::hottools.variants.HaploidWarning"),
    pytest.mark.filterwarnings("ignore::hottools.variants.StructuralEditWarning"),
    pytest.mark.filterwarnings("ignore::hottools.variants.EmptyRegionWarning"),
]

CLASSES = {
    "snv": {"snv"},
    "snv,mnv": {"snv", "mnv"},
    "indel": {"snv", "ins", "del"},
    "all": {"snv", "ins", "del", "mnv", "symbolic-del"},
}
RECORD_CLASSES = {
    "insertion": ["ins"],
    "deletion": ["del"],
    "mnv": ["mnv"],
    "symbolic_deletion": ["symbolic-del"],
    "mixed_snv_insertion": ["snv", "ins"],
}


@pytest.fixture
def structural_files(tmp_path):
    reference = "".join((DATA / "reference.fa").read_text().splitlines()[1:])
    fasta = write_fasta(tmp_path / "reference.fa", reference)
    vcf = _bgzip_and_index(
        (DATA / "structural.vcf").read_text(), tmp_path / "structural.vcf.gz"
    )
    return fasta, vcf


def oracle_records(classes, policy):
    records, sample_alleles = [], [[] for _ in range(4)]
    for line in (DATA / "structural.vcf").read_text().splitlines():
        if line.startswith("#"):
            continue
        _, pos, name, ref, alt, _, _, info, _, *gts = line.split("\t")
        if name not in RECORD_CLASSES:  # Unsupported INS/BND cannot be reconstructed.
            continue
        kinds = RECORD_CLASSES[name]
        if policy == "biallelic" and len(kinds) != 1:
            continue
        if policy != "selected" and any(kind not in CLASSES[classes] for kind in kinds):
            continue
        if not any(kind in CLASSES[classes] for kind in kinds):
            continue
        records.append(Rec(int(pos), ref, alt, tuple(gts), info))
        for sample, gt in enumerate(gts):
            pair = [allele_of(gt, hap) for hap in [0, 1]]
            sample_alleles[sample].append(
                [a if a == 0 or kinds[a - 1] in CLASSES[classes] else 0 for a in pair]
            )
    return records, sample_alleles


@pytest.mark.parametrize("classes", CLASSES)
@pytest.mark.parametrize("policy", ["whole", "biallelic", "selected"])
@pytest.mark.parametrize("window", [(1, 64, "start"), (9, 56, "center")])
@pytest.mark.parametrize("strand,batch_size", [("+", None), ("-", 1)])
def test_structural_class_anchor_policy_matrix(
    tmp_path, structural_files, classes, policy, window, strand, batch_size
):
    fasta, vcf = structural_files
    start, end, anchor = window
    result, _ = run_paths(
        tmp_path,
        fasta,
        vcf,
        start=start,
        end=end,
        anchor=anchor,
        strand=strand,
        batch_size=batch_size,
        variant_classes=classes,
        biallelic_only=policy == "biallelic",
        mixed_record_policy="apply-selected" if policy == "selected" else "exclude",
    )
    offset = 0 if anchor == "start" else (end - start) // 2
    records, alleles = oracle_records(classes, policy)
    expected = []
    for sample in range(4):
        tracks = []
        for hap in [0, 1]:
            sequence = oracle_window(
                "ACGT" * 16,
                records,
                [pair[hap] for pair in alleles[sample]],
                anchor=start + offset,
                offset=offset,
                length=end - start + 1,
            )
            tracks.append(revcomp(sequence) if strand == "-" else sequence)
        expected.append(tuple(tracks))
    assert haps(result) == expected


@pytest.mark.parametrize("biallelic_only", [False, True])
def test_native_structural_consensus(tmp_path, structural_files, biallelic_only, native_bcftools):
    bcftools = native_bcftools
    fasta, vcf = structural_files
    lines = []
    for line in (DATA / "structural.vcf").read_text().splitlines():
        if line.startswith("#"):
            lines.append(line)
            continue
        fields = line.split("\t")
        if fields[2] not in RECORD_CLASSES or (biallelic_only and "," in fields[4]):
            continue
        lines.append(line)
    filtered = _bgzip_and_index("\n".join(lines) + "\n", tmp_path / "eligible.vcf.gz")
    result, _ = run_paths(
        tmp_path,
        fasta,
        vcf,
        end=64,
        variant_classes="all",
        biallelic_only=biallelic_only,
    )
    expected = []
    for sample in ["S1", "S2", "S3", "S4"]:
        tracks = []
        for hap in [1, 2]:
            output = bcftools.consensus(
                "-f", str(fasta), "-s", sample, "-H", str(hap), str(filtered)
            )
            sequence = "".join(
                line for line in output.splitlines() if not line.startswith(">")
            )
            # Fixed reference-start window: truncate/pad AFTER whole-contig reconstruction.
            sequence = sequence[:64].ljust(64, "-")
            tracks.append(
                np.stack([vector(None if base == "-" else base) for base in sequence])
            )
        expected.append(tracks)
    np.testing.assert_array_equal(result, np.asarray(expected))
