"""Regressions fixed since 0.7.0; failures on the current release are errors."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hottools.writers import make_writer


@pytest.fixture
def comparison():
    """Load optional checkout-only validation without collection-time effects."""
    validation = Path(__file__).resolve().parents[1] / "validation"
    if not all((validation / name).is_file() for name in
               ("compare_with_bcftools.py", "synthetic_cases.py")):
        pytest.skip("optional validation comparator is absent from this checkout")
    old_path = sys.path[:]
    names = ("review_comparator", "synthetic_cases")
    old_modules = {name: sys.modules.get(name) for name in names}
    try:
        sys.path.insert(0, str(validation))
        # Avoid accidentally importing a comparator helper from another checkout.
        sys.modules.pop("synthetic_cases", None)
        spec = importlib.util.spec_from_file_location(
            "review_comparator", validation / "compare_with_bcftools.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = old_path
        for name, previous in old_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_commit_failure_preserves_previous_run(tmp_path, monkeypatch):
    out = tmp_path / "samples"
    out.mkdir()
    for name in ("S1", "S2"):
        np.save(out / f"{name}.npy", np.zeros((2, 4), dtype=np.float32))
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    import hottools.writers as writers
    real_replace = writers.os.replace
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected commit failure")
        return real_replace(source, target)

    monkeypatch.setattr(writers.os, "replace", fail_second)
    with pytest.raises(OSError, match="injected"):
        with make_writer(output_path=out, fmt="npy", full_shape=(2, 2, 4),
                         dtype=np.float32, samples=["S1", "S2"],
                         per_sample=True, metadata={}, overwrite="replace") as writer:
            writer.write(np.ones((2, 2, 4), dtype=np.float32), 0, 2)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert not list(tmp_path.glob("*.hottools-partial"))


def prepare_comparison(comparison, monkeypatch, samples):
    case = SimpleNamespace(name="review", description="review", contig="chr1",
                           window=4, samples=samples, structural=True,
                           haploid_source=False, strand="+")
    mat = SimpleNamespace(case=case, fasta=Path("ref.fa"), consensus_input=lambda: Path("v.vcf.gz"))
    monkeypatch.setattr(comparison, "hottools_run", lambda *args: (0, {"variants_kept": 1, "working_pad_right": 0}, ""))
    data = np.zeros((len(samples), 2, 4, 4))
    data[..., 0] = 1
    monkeypatch.setattr(comparison.np, "load", lambda *args: data)
    return mat


def test_validation_rejects_shortened_consensus(tmp_path, monkeypatch, comparison):
    mat = prepare_comparison(comparison, monkeypatch, ["S1"])
    monkeypatch.setattr(comparison, "bcftools_haplotype", lambda *args: "AAA")
    assert comparison.compare_case(mat, tmp_path).status != "match"


def test_validation_rejects_partial_reference_tool_failure(tmp_path, monkeypatch, comparison):
    mat = prepare_comparison(comparison, monkeypatch, ["S1", "S2"])
    monkeypatch.setattr(comparison, "bcftools_haplotype", lambda fasta, vcf, region, sample, hap: "AAAA" if sample == "S1" else None)
    assert comparison.compare_case(mat, tmp_path).status != "match"


def test_validation_exit_fails_on_unexpected_edge_result(tmp_path, monkeypatch, comparison):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["compare_with_bcftools.py", "0"])
    monkeypatch.setattr(comparison, "build_concordance_cases", lambda: [])
    monkeypatch.setattr(comparison, "build_fuzz_cases", lambda n: [])
    monkeypatch.setattr(comparison, "build_structural_fuzz_cases", lambda n: [])
    monkeypatch.setattr(comparison, "build_edge_cases", lambda: [SimpleNamespace(materialise=lambda root: None)])
    result = comparison.EdgeResult("edge", "review", "ok", 2, "unexpected", 0, "", "unexpected-error", "")
    monkeypatch.setattr(comparison, "probe_edge", lambda *args: result)
    assert comparison.main() != 0


def test_structural_memory_plan_accounts_for_simultaneous_canvases(tmp_path, fasta_path, indel_vcf_path):
    from hottools.pipeline import RunConfig, run_hottools
    from hottools.regions import Region
    meta = run_hottools(RunConfig(
        vcf_path=str(indel_vcf_path), fasta_path=str(fasta_path),
        region=Region("chr1", 1, 100), output_path=str(tmp_path / "out.npy"),
        mode="haplotypes", variant_classes="all", batch_size=4,
    ))
    work_length = meta["working_length"]
    minimum_multiplier = 1 + work_length / meta["length"]
    assert meta["peak_multiplier"] >= minimum_multiplier


@pytest.mark.parametrize("mode", ["dosage", "haplotypes", "consensus"])
def test_indexed_bcf_matches_vcf_with_reordered_samples(tmp_path, fasta_path, clean_vcf_path, mode):
    import pysam
    import pysam.bcftools

    from hottools.pipeline import RunConfig, run_hottools
    from hottools.regions import Region
    bcf = tmp_path / "input.bcf"
    with pysam.VariantFile(str(clean_vcf_path)) as source:
        with pysam.VariantFile(str(bcf), "wb", header=source.header) as target:
            for record in source:
                target.write(record)
    pysam.bcftools.index("-f", str(bcf))
    arrays = []
    for index, source in enumerate([clean_vcf_path, bcf]):
        output = tmp_path / f"output{index}.npy"
        meta = run_hottools(RunConfig(
            vcf_path=str(source), fasta_path=str(fasta_path),
            region=Region("chr1", 1, 200), output_path=str(output),
            mode=mode, batch_size=1, samples=["S4", "S2"],
        ))
        assert meta["sample_ids"] == ["S4", "S2"]
        arrays.append(np.load(output))
    np.testing.assert_array_equal(*arrays)


def test_failed_dosage_run_does_not_load_stale_data(tmp_path, monkeypatch, comparison):
    mat = prepare_comparison(comparison, monkeypatch, ["S1"])
    mat.case.structural = False
    loaded = []

    def run(mat, output, mode, extra):
        if mode == "dosage":
            return 2, None, "dosage failed"
        return 0, {"variants_kept": 1}, ""

    monkeypatch.setattr(comparison, "hottools_run", run)
    monkeypatch.setattr(comparison.np, "load", lambda path: loaded.append(path))
    result = comparison.compare_case(mat, tmp_path)
    assert result.status == "hottools-error"
    assert "dosage failed" in result.note
    assert not loaded


def test_decoder_rejects_fractional_and_cancelling_channels(comparison):
    track = np.array([[0.5, 0.5, 0, 0], [-0.5, 0.5, 0, 0], [0, 0, 0, 0]])
    assert comparison.decode_onehot(track) == "??N"


def test_structural_peak_includes_gather_and_resident_arrays(
    tmp_path, fasta_path, indel_vcf_path, monkeypatch
):
    import hottools.pipeline as pipeline
    from hottools.regions import Region

    chosen = []
    choose = pipeline.choose_batch_size

    def capture(**kwargs):
        chosen.append(kwargs)
        return choose(**kwargs)

    monkeypatch.setattr(pipeline, "choose_batch_size", capture)
    metadata = pipeline.run_hottools(pipeline.RunConfig(
        vcf_path=str(indel_vcf_path), fasta_path=str(fasta_path),
        region=Region("chr1", 1, 100), output_path=str(tmp_path / "out.npy"),
        mode="haplotypes", variant_classes="all", batch_size=4,
    ))
    ratio = metadata["working_length"] / metadata["length"]
    assert metadata["peak_multiplier"] >= ratio + 2
    assert chosen[0]["extra_bytes"] > (
        metadata["genotype_array_bytes"] + metadata["variant_table_bytes"]
        + metadata["working_length"] * 4 * 4
    )
