import json

import numpy as np
import pytest
from conftest import REF_LENGTH

from hottools.cli import build_parser, main

COMP = [3, 2, 1, 0]


def run_cli(capsys, *argv):
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    payload = json.loads(captured.out) if captured.out.strip() else None
    return code, payload, captured.err


def base_args(vcf, fasta, out, *extra):
    return [str(vcf), "--reference", str(fasta), "-o", str(out), *extra]


# --------------------------------------------------------------------
# argument handling
# --------------------------------------------------------------------

def test_length_without_center_is_rejected(capsys, clean_vcf_path, fasta_path, tmp_path):
    code, _, err = run_cli(
        capsys,
        *base_args(clean_vcf_path, fasta_path, tmp_path / "o.npy",
                   "--region", "chr1:1-50", "--length", "999"),
    )
    assert code == 2
    assert "--length is only valid together with --center" in err


def test_center_without_length_is_rejected(capsys, clean_vcf_path, fasta_path, tmp_path):
    code, _, err = run_cli(
        capsys,
        *base_args(clean_vcf_path, fasta_path, tmp_path / "o.npy",
                   "--center", "chr1:100"),
    )
    assert code == 2
    assert "--center requires --length" in err


def test_center_and_length(capsys, clean_vcf_path, fasta_path, tmp_path):
    code, meta, _ = run_cli(
        capsys,
        *base_args(clean_vcf_path, fasta_path, tmp_path / "o.npy",
                   "--center", "chr1:100", "--length", "51", "--dry-run"),
    )
    assert code == 0
    assert (meta["start"], meta["end"], meta["length"]) == (75, 125, 51)


def test_samples_and_sample_file_are_exclusive(
    capsys, clean_vcf_path, fasta_path, tmp_path
):
    sample_file = tmp_path / "s.txt"
    sample_file.write_text("S1\n")
    code, _, err = run_cli(
        capsys,
        *base_args(clean_vcf_path, fasta_path, tmp_path / "o.npy",
                   "--region", "chr1:1-50", "--samples", "S1",
                   "--sample-file", sample_file),
    )
    assert code == 2 and "not both" in err


def test_region_is_required():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["v.vcf.gz", "--reference", "r.fa", "-o", "o.npy"])


# --------------------------------------------------------------------
# strand
# --------------------------------------------------------------------

def test_strand_flag_is_recorded_and_applied(
    capsys, clean_vcf_path, fasta_path, tmp_path
):
    run_cli(capsys, *base_args(clean_vcf_path, fasta_path, tmp_path / "fwd.npy",
                               "--region", f"chr1:1-{REF_LENGTH}"))
    code, meta, _ = run_cli(
        capsys,
        *base_args(clean_vcf_path, fasta_path, tmp_path / "rev.npy",
                   "--region", f"chr1:1-{REF_LENGTH}", "--strand", "-"),
    )
    assert code == 0
    assert meta["strand"] == "-" and meta["reverse_complement"] is True
    fwd = np.load(tmp_path / "fwd.npy")
    rev = np.load(tmp_path / "rev.npy")
    np.testing.assert_array_equal(rev, fwd[:, ::-1, :][:, :, COMP])


def test_region_file_strand_column_is_applied(
    capsys, clean_vcf_path, fasta_path, tmp_path
):
    """The bug this guards: the column was parsed, stored, then ignored."""
    regions = tmp_path / "regions.txt"
    regions.write_text(f"chr1 1 {REF_LENGTH} -\n")
    code, meta, _ = run_cli(
        capsys,
        *base_args(clean_vcf_path, fasta_path, tmp_path / "rf",
                   "--region-file", regions),
    )
    assert code == 0
    assert meta["strand"] == "-" and meta["reverse_complement"] is True

    run_cli(capsys, *base_args(clean_vcf_path, fasta_path, tmp_path / "fwd.npy",
                               "--region", f"chr1:1-{REF_LENGTH}"))
    fwd = np.load(tmp_path / "fwd.npy")
    rev = np.load(tmp_path / "rf" / f"chr1_1_{REF_LENGTH}_minus.npy")
    np.testing.assert_array_equal(rev, fwd[:, ::-1, :][:, :, COMP])


# --------------------------------------------------------------------
# region files
# --------------------------------------------------------------------

def test_single_region_file_still_writes_a_directory(
    capsys, clean_vcf_path, fasta_path, tmp_path
):
    """One-region files used to write a bare, suffix-less file."""
    regions = tmp_path / "regions.txt"
    regions.write_text("chr1 1 100\n")
    out = tmp_path / "rf"
    code, _, _ = run_cli(
        capsys,
        *base_args(clean_vcf_path, fasta_path, out, "--region-file", regions),
    )
    assert code == 0
    assert out.is_dir()
    assert (out / "chr1_1_100.npy").exists()


def test_multi_region_file(capsys, clean_vcf_path, fasta_path, tmp_path):
    regions = tmp_path / "regions.txt"
    regions.write_text("chr1 1 100\nchr1 101 200\n")
    out = tmp_path / "rf"
    code, payload, _ = run_cli(
        capsys,
        *base_args(clean_vcf_path, fasta_path, out, "--region-file", regions),
    )
    assert code == 0 and isinstance(payload, list) and len(payload) == 2
    assert {p.name for p in out.glob("*.npy")} == {
        "chr1_1_100.npy", "chr1_101_200.npy"
    }


def test_bed_region_file_is_zero_based(capsys, clean_vcf_path, fasta_path, tmp_path):
    bed = tmp_path / "regions.bed"
    bed.write_text("chr1\t0\t100\n")
    code, payload, _ = run_cli(
        capsys,
        *base_args(clean_vcf_path, fasta_path, tmp_path / "rf",
                   "--region-file", bed, "--dry-run"),
    )
    assert code == 0
    assert (payload["start"], payload["end"]) == (1, 100)


def test_region_file_format_override(capsys, clean_vcf_path, fasta_path, tmp_path):
    bed = tmp_path / "regions.bed"
    bed.write_text("chr1\t1\t100\n")
    code, payload, _ = run_cli(
        capsys,
        *base_args(clean_vcf_path, fasta_path, tmp_path / "rf",
                   "--region-file", bed, "--region-file-format", "1based",
                   "--dry-run"),
    )
    assert code == 0
    assert (payload["start"], payload["end"]) == (1, 100)


# --------------------------------------------------------------------
# error reporting
# --------------------------------------------------------------------

def test_missing_contig_is_a_clean_error(
    capsys, clean_vcf_path, fasta_path, tmp_path
):
    code, _, err = run_cli(
        capsys,
        *base_args(clean_vcf_path, fasta_path, tmp_path / "o.npy",
                   "--region", "chrZ:1-10"),
    )
    assert code == 2
    assert err.startswith("hottools: error:")
    assert "Traceback" not in err


def test_debug_flag_restores_the_traceback(
    capsys, clean_vcf_path, fasta_path, tmp_path
):
    with pytest.raises(ValueError):
        main([str(clean_vcf_path), "--reference", str(fasta_path),
              "-o", str(tmp_path / "o.npy"), "--region", "chrZ:1-10",
              "--debug"])


def test_unknown_sample_is_a_clean_error(
    capsys, clean_vcf_path, fasta_path, tmp_path
):
    code, _, err = run_cli(
        capsys,
        *base_args(clean_vcf_path, fasta_path, tmp_path / "o.npy",
                   "--region", "chr1:1-50", "--samples", "S1,GHOST"),
    )
    assert code == 2 and "GHOST" in err


def test_ambiguous_reference_is_a_clean_error(
    capsys, vcf_path, fasta_path, tmp_path
):
    code, _, err = run_cli(
        capsys,
        *base_args(vcf_path, fasta_path, tmp_path / "o.npy",
                   "--region", f"chr1:1-{REF_LENGTH}",
                   "--ref-mismatch-policy", "skip"),
    )
    assert code == 2 and "Ambiguous FASTA base" in err


def test_strict_turns_the_empty_region_into_an_error(
    capsys, clean_vcf_path, fasta_path, tmp_path
):
    code, _, err = run_cli(
        capsys,
        *base_args(clean_vcf_path, fasta_path, tmp_path / "o.npy",
                   "--region", "chr1:150-200", "--strict"),
    )
    assert code == 2 and "No usable variants" in err


# --------------------------------------------------------------------
# sample selection and output plumbing
# --------------------------------------------------------------------

def test_sample_file_order_is_preserved(
    capsys, clean_vcf_path, fasta_path, tmp_path
):
    sample_file = tmp_path / "s.txt"
    sample_file.write_text("# subset\nS4\nS2\n")
    code, meta, _ = run_cli(
        capsys,
        *base_args(clean_vcf_path, fasta_path, tmp_path / "o.npy",
                   "--region", f"chr1:1-{REF_LENGTH}",
                   "--sample-file", sample_file),
    )
    assert code == 0
    assert meta["sample_ids"] == ["S4", "S2"]
    assert np.load(tmp_path / "o.npy").shape[0] == 2


def test_dry_run_reports_without_writing(
    capsys, vcf_path, fasta_path, tmp_path
):
    code, meta, _ = run_cli(
        capsys,
        *base_args(vcf_path, fasta_path, tmp_path / "o.npy",
                   "--region", f"chr1:1-{REF_LENGTH}", "--dry-run",
                   "--ambiguous-ref-policy", "skip",
                   "--ref-mismatch-policy", "skip"),
    )
    assert code == 0
    assert meta["variant_records_seen"] == 8
    assert meta["variants_kept"] == 4
    assert not (tmp_path / "o.npy").exists()


def test_no_metadata_json(capsys, clean_vcf_path, fasta_path, tmp_path):
    run_cli(capsys, *base_args(clean_vcf_path, fasta_path, tmp_path / "o.npy",
                               "--region", "chr1:1-50", "--no-metadata-json"))
    assert not (tmp_path / "o.npy.metadata.json").exists()


def test_h5_chunk_overrides_reach_the_writer(
    capsys, clean_vcf_path, fasta_path, tmp_path
):
    pytest.importorskip("h5py")
    code, meta, _ = run_cli(
        capsys,
        *base_args(clean_vcf_path, fasta_path, tmp_path / "o.h5",
                   "--region", f"chr1:1-{REF_LENGTH}", "--format", "h5",
                   "--h5-chunk-samples", "1", "--h5-chunk-length", "50"),
    )
    assert code == 0
    assert meta["h5_chunks"] == [1, 50, 4]


# --------------------------------------------------------------------
# site filters, mask, consensus mode, haploid padding
# --------------------------------------------------------------------

def test_pass_only_flag(capsys, filtered_vcf_path, fasta_path, tmp_path):
    code, meta, _ = run_cli(
        capsys,
        *base_args(filtered_vcf_path, fasta_path, tmp_path / "o.npy",
                   "--region", f"chr1:1-{REF_LENGTH}", "--pass-only",
                   "--dry-run"),
    )
    assert code == 0
    assert meta["variants_kept"] == 2
    assert meta["variant_filtered_not_pass"] == 1


def test_qual_and_af_flags(capsys, filtered_vcf_path, fasta_path, tmp_path):
    code, meta, _ = run_cli(
        capsys,
        *base_args(filtered_vcf_path, fasta_path, tmp_path / "o.npy",
                   "--region", f"chr1:1-{REF_LENGTH}",
                   "--min-qual", "40", "--min-af", "0.5", "--dry-run"),
    )
    assert code == 0
    assert meta["variants_kept"] == 1


def test_af_source_flag(capsys, filtered_vcf_path, fasta_path, tmp_path):
    code, meta, _ = run_cli(
        capsys,
        *base_args(filtered_vcf_path, fasta_path, tmp_path / "o.npy",
                   "--region", f"chr1:1-{REF_LENGTH}", "--min-af", "0.1",
                   "--af-source", "samples", "--dry-run"),
    )
    assert code == 0 and meta["variants_kept"] == 3


def test_mask_flag(capsys, clean_vcf_path, fasta_path, tmp_path):
    bed = tmp_path / "m.bed"
    bed.write_text("chr1\t0\t30\n")
    code, meta, _ = run_cli(
        capsys,
        *base_args(clean_vcf_path, fasta_path, tmp_path / "o.npy",
                   "--region", f"chr1:1-{REF_LENGTH}", "--mask", str(bed)),
    )
    assert code == 0
    assert meta["masked_bases"] == 30
    assert np.all(np.load(tmp_path / "o.npy")[:, :30, :] == 0)


@pytest.mark.parametrize("policy", ["reference", "alternate", "major"])
def test_consensus_mode_flags(
    capsys, clean_vcf_path, fasta_path, tmp_path, policy
):
    code, meta, _ = run_cli(
        capsys,
        *base_args(clean_vcf_path, fasta_path, tmp_path / f"{policy}.npy",
                   "--region", f"chr1:1-{REF_LENGTH}", "--mode", "consensus",
                   "--het-policy", policy, "--missing-policy", "reference"),
    )
    assert code == 0
    assert meta["het_policy"] == policy
    out = np.load(tmp_path / f"{policy}.npy")
    assert out.shape == (4, REF_LENGTH, 4)
    assert set(np.unique(out).tolist()) <= {0.0, 1.0}


def test_multiallelic_through_the_cli(
    capsys, triallelic_vcf_path, fasta_path, tmp_path
):
    code, meta, _ = run_cli(
        capsys,
        *base_args(triallelic_vcf_path, fasta_path, tmp_path / "o.npy",
                   "--region", f"chr1:1-{REF_LENGTH}", "--mode", "haplotypes"),
    )
    assert code == 0
    assert meta["multiallelic_records"] == 1
    assert meta["variants_kept"] == 2


@pytest.mark.parametrize("policy", ["duplicate", "blank"])
def test_haploid_policy_flags(
    capsys, haploid_vcf_path, fasta_path, tmp_path, policy
):
    code, meta, _ = run_cli(
        capsys,
        *base_args(haploid_vcf_path, fasta_path, tmp_path / f"{policy}.npy",
                   "--region", f"chr1:1-{REF_LENGTH}", "--mode", "haplotypes",
                   "--haploid-policy", policy),
    )
    assert code == 0
    assert meta["haploid_policy"] == policy
    assert meta["sample_ploidy"] == [1, 2, 2, 2]
    out = np.load(tmp_path / f"{policy}.npy")
    assert out.shape == (4, 2, REF_LENGTH, 4)


def test_haploid_policy_error_flag(
    capsys, haploid_vcf_path, fasta_path, tmp_path
):
    code, _, err = run_cli(
        capsys,
        *base_args(haploid_vcf_path, fasta_path, tmp_path / "o.npy",
                   "--region", f"chr1:1-{REF_LENGTH}", "--mode", "haplotypes",
                   "--haploid-policy", "error"),
    )
    assert code == 2 and "haploid genotype" in err


def test_polyploid_is_a_clean_error(
    capsys, triploid_vcf_path, fasta_path, tmp_path
):
    code, _, err = run_cli(
        capsys,
        *base_args(triploid_vcf_path, fasta_path, tmp_path / "o.npy",
                   "--region", f"chr1:1-{REF_LENGTH}", "--mode", "haplotypes"),
    )
    assert code == 2 and "ploidy > 2" in err
