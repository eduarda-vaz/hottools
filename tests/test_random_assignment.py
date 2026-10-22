"""Opt-in pseudo-phasing preserves known phase and records reproducible choices."""
import json

import numpy as np
import pytest
from sv_helpers import Rec, decode, run_paths, write_case

from hottools.cli import main
from hottools.pipeline import RunConfig
from hottools.regions import Region
from hottools.variants import EmptyRegionWarning

SAMPLES = ["PHASED", "U1", "U2"]


@pytest.fixture
def mixed_phase_files(tmp_path):
    records = [
        Rec(5, "A", "C", ("0|1", "1/0", "0/1")),
        Rec(10, "A", "C,G", ("2|1", "1/2", "2/1")),
        Rec(15, "A", "C", ("1/1", "0/0", "1/1")),
    ]
    records.extend(Rec(pos, "A", "T", ("1|0", "0/1", "1/0"))
                   for pos in range(20, 61, 5))
    return write_case(tmp_path, "A" * 96, records, samples=SAMPLES)


def run_random(tmp_path, files, name="random", **kwargs):
    return run_paths(tmp_path, *files, end=96, name=name,
                     unphased_policy="random", random_seed=kwargs.pop("random_seed", 42),
                     variant_classes=kwargs.pop("variant_classes", "snv"), **kwargs)


def choices(metadata):
    return {(x["chrom"], x["pos"], x["sample"]): x["assigned_gt"]
            for x in metadata["random_assignments"]}


def test_random_preserves_phased_and_homozygous_calls_and_audits(
    tmp_path, mixed_phase_files
):
    tensor, metadata = run_random(tmp_path, mixed_phase_files, missing_policy="error")
    phased_tracks = [decode(track) for track in tensor[0]]
    assert (phased_tracks[0][4], phased_tracks[1][4]) == ("A", "C")
    assert (phased_tracks[0][9], phased_tracks[1][9]) == ("G", "C")
    for pos in range(20, 61, 5):
        assert (phased_tracks[0][pos-1], phased_tracks[1][pos-1]) == ("T", "A")
    assert np.all(tensor[0, :, 14, 1] == 1)
    assert np.all(tensor[1, :, 14, 0] == 1)
    assert np.all(tensor[2, :, 14, 1] == 1)
    assert metadata["random_seed"] == 42
    assert metadata["contains_pseudo_haplotypes"] is True
    assert metadata["random_assignment_algorithm"] == "blake2b-call-v1"
    assert metadata["variant_randomly_assigned_genotypes"] == 22
    assert len(metadata["random_assignments"]) == 22
    assert {x["sample"] for x in metadata["random_assignments"]} == {"U1", "U2"}
    for call in metadata["random_assignments"]:
        assert sorted(call["input_gt"].split("/")) == sorted(call["assigned_gt"].split("|"))
        sample_index = SAMPLES.index(call["sample"])
        allele_bases = [call["ref"], *call["alts"]]
        for hap, allele in enumerate(call["assigned_gt"].split("|")):
            expected_base = allele_bases[int(allele)]
            assert decode(tensor[sample_index, hap])[call["pos"]-1] == expected_base
    sidecar = tmp_path / "random.npy.metadata.json"
    assert json.loads(sidecar.read_text())["random_assignments"] == metadata["random_assignments"]


@pytest.mark.parametrize("batch_size,threads", [("none", 1), (1, 2), (2, 4)])
def test_random_reproducibility_and_dosage_preservation(
    tmp_path, mixed_phase_files, batch_size, threads
):
    expected, original_metadata = run_random(tmp_path, mixed_phase_files, name="original", batch_size="none")
    actual, metadata = run_random(tmp_path, mixed_phase_files, name="profile",
                                  batch_size=batch_size, threads=threads)
    np.testing.assert_array_equal(actual, expected)
    assert choices(metadata) == choices(original_metadata)
    dosage, _ = run_paths(tmp_path, *mixed_phase_files, end=96,
                          name="dosage", mode="dosage", variant_classes="snv")
    np.testing.assert_array_equal(actual.mean(axis=1), dosage)


def test_random_sample_order_subset_and_region_do_not_change_choices(tmp_path, mixed_phase_files):
    expected, full_metadata = run_random(tmp_path, mixed_phase_files, name="full")
    subset, metadata = run_random(tmp_path, mixed_phase_files, name="subset", samples=["U2", "PHASED"])
    np.testing.assert_array_equal(subset, expected[[2, 0]])
    assert choices(metadata) == {k: v for k, v in choices(full_metadata).items() if k[2] == "U2"}
    _, regional = run_random(tmp_path, mixed_phase_files, name="regional", start=20)
    assert choices(regional) == {k: v for k, v in choices(full_metadata).items() if k[1] >= 20}


def test_different_seeds_change_only_unphased_assignments(tmp_path, mixed_phase_files):
    first, first_metadata = run_random(tmp_path, mixed_phase_files, name="seed1", random_seed=1)
    second, second_metadata = run_random(tmp_path, mixed_phase_files, name="seed2", random_seed=2)
    assert choices(first_metadata) != choices(second_metadata)
    assert not np.array_equal(first[1:], second[1:])
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first.mean(axis=1), second.mean(axis=1))


@pytest.mark.parametrize("missing_policy", ["zero", "reference"])
def test_random_does_not_assign_missing_half_called_or_haploid_calls(tmp_path, missing_policy):
    files = write_case(tmp_path, "A" * 96, [Rec(5, "A", "C", ("0/.", "./.", "1"))], samples=SAMPLES)
    tensor, metadata = run_random(tmp_path, files, missing_policy=missing_policy)
    assert metadata["random_assignments"] == []
    assert metadata["contains_pseudo_haplotypes"] is False
    assert np.all(tensor[2, :, 4, 1] == 1)
    if missing_policy == "zero":
        assert tensor[0, 1, 4].sum() == 0
        assert tensor[1, :, 4].sum() == 0
    else:
        assert np.all(tensor[:2, :, 4, 0] == 1)


def test_random_default_remains_error(tmp_path, mixed_phase_files):
    with pytest.raises(ValueError, match="unphased heterozygous"):
        run_paths(tmp_path, *mixed_phase_files, end=96, variant_classes="snv")


@pytest.mark.parametrize("seed", [None, True, -1, 1.5, "42", 2**64])
def test_invalid_random_seeds_fail_before_input_io(tmp_path, seed):
    with pytest.raises(ValueError, match="random_seed"):
        RunConfig(vcf_path="absent.vcf", fasta_path="absent.fa", region=Region("chr1", 1, 2),
                  output_path=str(tmp_path / "unused.npy"), mode="haplotypes",
                  unphased_policy="random", random_seed=seed).validate()


@pytest.mark.parametrize("mode", ["dosage", "consensus"])
def test_random_is_haplotype_only(tmp_path, mode):
    with pytest.raises(ValueError, match="requires mode='haplotypes'"):
        RunConfig(vcf_path="absent.vcf", fasta_path="absent.fa", region=Region("chr1", 1, 2),
                  output_path=str(tmp_path / "unused.npy"), mode=mode,
                  unphased_policy="random", random_seed=42).validate()


def test_seed_without_random_policy_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="requires unphased_policy='random'"):
        RunConfig(vcf_path="absent.vcf", fasta_path="absent.fa", region=Region("chr1", 1, 2),
                  output_path=str(tmp_path / "unused.npy"), random_seed=42).validate()


def test_filtered_records_are_not_reported_as_assigned(tmp_path, mixed_phase_files):
    with pytest.warns(EmptyRegionWarning):
        _, metadata = run_random(tmp_path, mixed_phase_files, min_af=0.95, af_source="samples")
    assert metadata["random_assignments"] == []
    assert metadata["variant_randomly_assigned_genotypes"] == 0


def test_random_projection_precedes_assignment(tmp_path):
    files = write_case(tmp_path, "A" * 96, [Rec(5, "A", "C,AT", ("0/2", "1/2", "2/2"))], samples=SAMPLES)
    tensor, metadata = run_random(tmp_path, files, mixed_record_policy="apply-selected")
    assert np.all(tensor[[0, 2], :, 4, 0] == 1)
    assert len(metadata["random_assignments"]) == 1
    call = metadata["random_assignments"][0]
    assert call["input_gt"] == "1/2"
    assert call["projected_gt"] == "1/0"
    assert sorted(call["assigned_gt"].split("|")) == ["0", "1"]


def test_random_structural_genotype_preserves_alleles(tmp_path):
    files = write_case(tmp_path, "A" * 96, [Rec(10, "A", "ATG", ("0|1", "0/1", "1/0"))], samples=SAMPLES)
    tensor, metadata = run_random(tmp_path, files, variant_classes="all")
    assert decode(tensor[0, 0]) == "A" * 96
    assert decode(tensor[0, 1])[9:13] == "ATGA"
    for sample in (1, 2):
        assert sorted([decode(track)[9:13] for track in tensor[sample]]) == ["AAAA", "ATGA"]
    assert len(metadata["random_assignments"]) == 2


def test_random_cli_requires_seed_and_writes_audit(tmp_path, mixed_phase_files, capsys):
    fasta, vcf = mixed_phase_files
    arguments = [str(vcf), "--reference", str(fasta), "--region", "chr1:1-96",
                 "--mode", "haplotypes", "--unphased-policy", "random",
                 "-o", str(tmp_path / "cli.npy")]
    assert main(arguments) == 2
    assert "random_seed" in capsys.readouterr().err
    assert main([*arguments, "--random-seed", "42"]) == 0
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["random_seed"] == 42
    assert len(metadata["random_assignments"]) == 22


@pytest.mark.parametrize("seed", [0, 2**64 - 1])
def test_seed_boundary_values_are_reproducible(tmp_path, mixed_phase_files, seed):
    first, _ = run_random(tmp_path, mixed_phase_files, name="first", random_seed=seed)
    second, _ = run_random(tmp_path, mixed_phase_files, name="second", random_seed=seed)
    np.testing.assert_array_equal(first, second)


def test_unphased_allele_order_does_not_change_assignment(tmp_path):
    first = write_case(tmp_path, "A" * 96,
                       [Rec(5, "A", "C,G", ("1/2",))], samples=["S1"], name="first")
    second = write_case(tmp_path, "A" * 96,
                        [Rec(5, "A", "C,G", ("2/1",))], samples=["S1"], name="second")
    left, _ = run_random(tmp_path, first, name="left")
    right, _ = run_random(tmp_path, second, name="right")
    np.testing.assert_array_equal(left, right)


def test_reverse_complement_does_not_change_assignments(tmp_path, mixed_phase_files):
    forward, forward_metadata = run_random(tmp_path, mixed_phase_files, name="forward")
    reverse, reverse_metadata = run_random(tmp_path, mixed_phase_files, name="reverse", strand="-")
    assert choices(forward_metadata) == choices(reverse_metadata)
    np.testing.assert_array_equal(reverse, forward[..., ::-1, :][..., [3, 2, 1, 0]])


def test_random_bcf_matches_vcf(tmp_path, mixed_phase_files):
    import pysam
    import pysam.bcftools

    fasta, vcf = mixed_phase_files
    bcf = tmp_path / "input.bcf"
    with pysam.VariantFile(str(vcf)) as source:
        with pysam.VariantFile(str(bcf), "wb", header=source.header) as target:
            for record in source:
                target.write(record)
    pysam.bcftools.index("-f", str(bcf))
    expected, expected_metadata = run_random(tmp_path, mixed_phase_files, name="vcf")
    actual, metadata = run_random(tmp_path, (fasta, bcf), name="bcf", threads=4)
    np.testing.assert_array_equal(actual, expected)
    assert metadata["random_assignments"] == expected_metadata["random_assignments"]


def test_random_audited_assignments_match_native_consensus(tmp_path, mixed_phase_files, native_bcftools):
    import pysam

    fasta, vcf = mixed_phase_files
    tensor, metadata = run_random(tmp_path, mixed_phase_files)
    assignments = choices(metadata)
    phased = tmp_path / "assigned.vcf.gz"
    with pysam.VariantFile(str(vcf)) as source:
        with pysam.VariantFile(str(phased), "wz", header=source.header) as target:
            for record in source:
                for sample in SAMPLES:
                    key = (record.contig, record.pos, sample)
                    if key in assignments:
                        record.samples[sample]["GT"] = tuple(int(x) for x in assignments[key].split("|"))
                        record.samples[sample].phased = True
                target.write(record)
    pysam.tabix_index(str(phased), preset="vcf", force=True)
    for index, sample in enumerate(SAMPLES):
        for haplotype in (1, 2):
            result = native_bcftools.consensus("-f", str(fasta), "-s", sample,
                                               "-H", str(haplotype), str(phased))
            sequence = "".join(line.strip() for line in result.splitlines()
                               if not line.startswith(">"))
            assert decode(tensor[index, haplotype-1]) == sequence
