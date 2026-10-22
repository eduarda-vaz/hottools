"""Window edges, overlap chains and representational limits."""

from __future__ import annotations

import re
import subprocess
import sys

import numpy as np
import pytest
from conftest import ALT_OF
from test_structural import LENGTH, as_decoded, decode, run

from hottools.edits import parse_variant_classes
from hottools.regions import Region
from hottools.variants import (
    StructuralEditWarning,
    find_chain_start,
    scan_structural_span,
)

#: A window that starts partway along the 200 bp test contig has to be
#: short enough to leave reference beyond it.
EDGE_LENGTH = 80


class TestPreScan:
    def test_snv_only_costs_nothing(self, indel_vcf_path):
        assert scan_structural_span(
            vcf_path=indel_vcf_path, chrom="chr1", start=1, end=200,
            variant_classes=frozenset({"snv"}),
        ) == (0, 0)
        assert find_chain_start(
            vcf_path=indel_vcf_path, chrom="chr1", start=100,
            max_lookback=500, variant_classes=frozenset({"snv"}),
        ) == (100, True)

    def test_it_reports_the_deletion_load_and_the_widest_claim(
        self, indel_vcf_path
    ):
        removed, widest = scan_structural_span(
            vcf_path=indel_vcf_path, chrom="chr1", start=1, end=200,
            variant_classes=parse_variant_classes("all"),
        )
        # Three removed by the deletion, five by the symbolic deletion.
        assert removed == 8
        # The symbolic deletion claims the widest interval.
        assert widest == 5

    def test_a_window_at_the_contig_start_needs_no_lookback(
        self, chain_vcf_path
    ):
        assert find_chain_start(
            vcf_path=chain_vcf_path, chrom="chr1", start=1,
            max_lookback=200, variant_classes=parse_variant_classes("del"),
        ) == (1, True)

    def test_the_chain_starts_at_a_position_no_record_straddles(
        self, chain_vcf_path
    ):
        start, clean = find_chain_start(
            vcf_path=chain_vcf_path, chrom="chr1", start=105,
            max_lookback=200, variant_classes=parse_variant_classes("del"),
        )
        # The record at 90 claims 91-104, so the chain has to be picked
        # up at 91 - where its own claim begins - and not later.
        assert (start, clean) == (91, True)


class TestWindowEdges:
    def test_a_record_reaching_into_the_window_is_applied(
        self, tmp_path, fasta_path, straddling_del_vcf_path,
        reference_sequence,
    ):
        tensor, meta = run(
            tmp_path, fasta_path, straddling_del_vcf_path,
            variant_classes="del", region=Region("chr1", 100, 100 + EDGE_LENGTH - 1),
        )
        assert meta["variant_clipped_at_edge"] == 1
        assert meta["structural_edits"] == 1
        # The record removes 96-103; inside the window that is 100-103.
        expected = reference_sequence[103:]
        assert decode(tensor[1, 0]) == as_decoded(expected)[:EDGE_LENGTH]
        # A non-carrier keeps the plain reference.
        assert decode(tensor[2, 0]) == as_decoded(
            reference_sequence[99:]
        )[:EDGE_LENGTH]

    def test_an_upstream_record_can_suppress_one_inside_the_window(
        self, tmp_path, fasta_path, chain_vcf_path, reference_sequence
    ):
        with pytest.warns(StructuralEditWarning):
            tensor, meta = run(
                tmp_path, fasta_path, chain_vcf_path,
                variant_classes="del",
                region=Region("chr1", 105, 105 + EDGE_LENGTH - 1),
            )
        # The record at 90 wins, so the one at 100 is set aside even
        # though it is the only one the window itself contains.
        assert meta["overlap_chain_lookback"] > 0
        assert meta["variant_overlapping_records_suppressed"] > 0
        # 91-104 is removed, which covers everything before the window,
        # so the output is simply the reference from 105 on.
        assert decode(tensor[0, 0]) == as_decoded(
            reference_sequence[104:]
        )[:EDGE_LENGTH]


class TestRepresentationalLimit:
    def test_a_second_record_claiming_the_same_position_is_dropped(
        self, tmp_path, fasta_path, colliding_mnv_vcf_path,
        reference_sequence,
    ):
        with pytest.warns(StructuralEditWarning, match="same reference"):
            tensor, meta = run(
                tmp_path, fasta_path, colliding_mnv_vcf_path,
                variant_classes="mnv",
            )
        assert meta["variant_overlapping_blocks_dropped"] == 0
        assert meta["variant_overlapping_records_suppressed"] > 0
        # Only the first block is encoded, at 100-102.
        got = decode(tensor[0, 0])
        reference = as_decoded(reference_sequence)[:LENGTH]
        assert got[99:102] == "".join(
            ALT_OF[b] for b in reference_sequence[99:102]
        )
        assert got[102:] == reference[102:]


class TestCli:
    def _run(self, args):
        return subprocess.run(
            [sys.executable, "-m", "hottools.cli", *args],
            capture_output=True, text=True,
        )

    @pytest.mark.parametrize("columns", [40, 60, 80, 100, 120])
    def test_the_flags_are_documented_in_help(self, monkeypatch, columns):
        monkeypatch.setenv("COLUMNS", str(columns))
        result = self._run(["--help"])
        assert result.returncode == 0
        assert "--variant-classes" in result.stdout
        assert "--anchor" in result.stdout
        # argparse may wrap a hyphenated class name between physical lines.
        # Keep checking the rendered help, without depending on terminal width.
        unwrapped = re.sub(r"-[ \t]*\n[ \t]*", "-", result.stdout)
        assert "symbolic-del" in unwrapped

    def test_classes_and_anchor_reach_the_run(
        self, tmp_path, fasta_path, indel_vcf_path
    ):
        output = tmp_path / "cli.npy"
        result = self._run([
            "--reference", str(fasta_path), str(indel_vcf_path),
            "--region", f"chr1:1-{LENGTH}", "--mode", "haplotypes",
            "--variant-classes", "indel", "--anchor", "center",
            "--missing-policy", "reference", "-o", str(output),
            "--overwrite", "replace",
        ])
        assert result.returncode == 0, result.stderr
        tensor = np.load(output)
        assert tensor.shape == (4, 2, LENGTH, 4)

    def test_an_unknown_class_is_rejected_before_any_work(
        self, tmp_path, fasta_path, indel_vcf_path
    ):
        result = self._run([
            "--reference", str(fasta_path), str(indel_vcf_path),
            "--region", f"chr1:1-{LENGTH}", "--variant-classes", "inversion",
            "-o", str(tmp_path / "x.npy"),
        ])
        assert result.returncode != 0
        assert "Unknown variant class" in result.stderr
        assert not (tmp_path / "x.npy").exists()

    def test_length_changing_classes_outside_haplotype_mode_fail(
        self, tmp_path, fasta_path, indel_vcf_path
    ):
        result = self._run([
            "--reference", str(fasta_path), str(indel_vcf_path),
            "--region", f"chr1:1-{LENGTH}", "--mode", "dosage",
            "--variant-classes", "del", "-o", str(tmp_path / "y.npy"),
        ])
        assert result.returncode != 0
        assert "only meaningful in --mode haplotypes" in result.stderr
        assert not (tmp_path / "y.npy").exists()
