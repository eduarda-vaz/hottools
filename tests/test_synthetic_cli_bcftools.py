"""CLI wiring and native bcftools concordance on the same eligible SNVs."""

import numpy as np
import pytest
from conftest import _bgzip_and_index
from sv_helpers import run_paths
from test_synthetic_configuration_matrix import (
    DATA,
    SAMPLES,
    expected_phased,
    options,
    vector,
)
from test_synthetic_configuration_matrix import (
    edge_files as edge_files,
)

from hottools.cli import main

pytestmark = [
    pytest.mark.filterwarnings("ignore::hottools.variants.HaploidWarning"),
    pytest.mark.filterwarnings("ignore::hottools.variants.StructuralEditWarning"),
    pytest.mark.filterwarnings("ignore::hottools.variants.EmptyRegionWarning"),
    pytest.mark.filterwarnings("ignore::hottools.variants.MultiallelicWarning"),
]


@pytest.mark.parametrize("policy", ["biallelic", "whole-snv", "selected-snv"])
@pytest.mark.parametrize("mode", ["haplotypes", "dosage", "consensus"])
def test_selection_flags_cli_matches_oracle(tmp_path, edge_files, capsys, policy, mode):
    fasta, vcf, _ = edge_files
    out = tmp_path / "cli.npy"
    flags = ["--biallelic-only"] if policy == "biallelic" else []
    if policy == "selected-snv":
        flags += ["--mixed-record-policy", "apply-selected"]
    code = main(
        [
            str(vcf),
            "--reference",
            str(fasta),
            "--region",
            "chr1:1-64",
            "--output",
            str(out),
            "--mode",
            mode,
            "--variant-classes",
            "snv",
            "--missing-policy",
            "reference",
            "--batch-size",
            "3",
            *flags,
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    np.testing.assert_array_equal(
        np.load(out), expected_phased(policy, mode, "reference", "duplicate")
    )


def project_literal_snvs(text, policy):
    """Independent, simple VCF projection; never uses Hottools selection code.

    Non-SNV alleles map to REF under selected-snv. GT allele numbers are
    remapped to the retained ALT list without altering phase or missing calls.
    """
    lines = []
    for line in text.splitlines():
        if line.startswith("#"):
            lines.append(line)
            continue
        fields = line.split("\t")
        ref, alts = fields[3], fields[4].split(",")
        if len(ref) != 1 or ref not in "ACGT":
            continue
        keep = [i for i, alt in enumerate(alts, 1) if len(alt) == 1 and alt in "ACGT"]
        if not keep or (policy != "selected-snv" and len(keep) != len(alts)):
            continue
        if policy == "biallelic" and len(alts) != 1:
            continue
        mapping = {0: 0, **{old: new for new, old in enumerate(keep, 1)}}
        fields[4] = ",".join(alts[i - 1] for i in keep)
        fields[7] = "."  # END from a discarded symbolic ALT is irrelevant.
        for col in range(9, len(fields)):
            gt = fields[col]
            separator = "|" if "|" in gt else "/"
            fields[col] = separator.join(
                "." if a == "." else str(mapping.get(int(a), 0))
                for a in gt.split(separator)
            )
        lines.append("\t".join(fields))
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("policy", ["biallelic", "whole-snv", "selected-snv"])
@pytest.mark.parametrize("mode", ["haplotypes", "dosage"])
def test_native_bcftools_consensus_same_eligible_variants(
    tmp_path, edge_files, policy, mode, native_bcftools
):
    # pysam.bcftools calls compiled bcftools, not a Python reimplementation.
    bcftools = native_bcftools
    fasta, vcf, _ = edge_files
    projected = _bgzip_and_index(
        project_literal_snvs((DATA / "phased.vcf").read_text(), policy),
        tmp_path / "consensus.vcf.gz",
    )
    result, _ = run_paths(
        tmp_path,
        fasta,
        vcf,
        end=64,
        mode=mode,
        missing_policy="reference",
        batch_size=3,
        **options(policy),
    )
    tracks = []
    for sample in SAMPLES:
        haplotypes = []
        for hap in [1, 2]:
            # A haploid call has one biological allele; Hottools duplicate
            # exposes it on both tracks. Request that allele from bcftools.
            requested = 1 if sample in ["HAP", "HAP_MISSING"] else hap
            output = bcftools.consensus(
                "-f", str(fasta), "-s", sample, "-H", str(requested), str(projected)
            )
            sequence = "".join(
                line.strip() for line in output.splitlines() if not line.startswith(">")
            )
            assert len(sequence) == 64, (policy, sample, hap, sequence)
            haplotypes.append(np.stack([vector(base) for base in sequence]))
        tracks.append(haplotypes)
    expected = np.asarray(tracks)
    if mode == "dosage":
        expected = expected.mean(axis=1)
    np.testing.assert_array_equal(result, expected)
