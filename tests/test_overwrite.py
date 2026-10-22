"""
Output transactions: what --overwrite error and replace leave on disk.

A run writes into a staging directory and commits only after its last
batch, so a refused or failed run must leave the output location exactly
as it found it, and a replacing run must leave exactly one coherent set
of files.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import hottools.pipeline as pipeline
from hottools.pipeline import RunConfig, run_hottools
from hottools.regions import Region
from hottools.writers import (
    MANIFEST_NAME,
    SampleNameCollisionWarning,
    plan_outputs,
    previous_run_outputs,
    safe_sample_names,
)


def config(vcf, fasta, out, **kwargs) -> RunConfig:
    return RunConfig(
        vcf_path=str(vcf), fasta_path=str(fasta),
        region=Region("chr1", 1, 200), output_path=str(out),
        missing_policy="reference", ambiguous_ref_policy="skip",
        ref_mismatch_policy="skip", unphased_policy="missing",
        **kwargs,
    )


def snapshot(directory: Path) -> dict[str, bytes]:
    """Every file under ``directory``, by relative path, with contents."""
    return {
        str(p.relative_to(directory)): p.read_bytes()
        for p in sorted(directory.rglob("*")) if p.is_file()
    }


def no_staging_left(directory: Path) -> bool:
    return not any(
        p.name.endswith(".hottools-partial") for p in directory.rglob("*")
    )


# --------------------------------------------------------------------
# overwrite='error'
# --------------------------------------------------------------------

def test_error_refuses_a_conflicting_per_sample_run_before_writing(
    tmp_path, vcf_path, fasta_path
):
    out = tmp_path / "per_sample"
    out.mkdir()
    (out / "S3.npy").write_text("someone else's data")
    before = snapshot(tmp_path)

    with pytest.raises(FileExistsError, match="Nothing was written"):
        run_hottools(config(vcf_path, fasta_path, out, per_sample=True,
                            batch_size=1))

    # The bug this guards against: S1 and S2 were written before S3 was
    # found to exist, leaving a half-new, half-old directory.
    assert snapshot(tmp_path) == before
    assert no_staging_left(tmp_path)


def test_error_is_checked_before_any_vcf_work(
    tmp_path, vcf_path, fasta_path, monkeypatch
):
    out = tmp_path / "out.npy"
    run_hottools(config(vcf_path, fasta_path, out))

    def must_not_run(**_):
        raise AssertionError("extraction ran for a run that must fail")

    monkeypatch.setattr(pipeline, "extract_variants", must_not_run)
    with pytest.raises(FileExistsError):
        run_hottools(config(vcf_path, fasta_path, out))


def test_error_refuses_a_stale_sidecar_and_leaves_it_alone(
    tmp_path, vcf_path, fasta_path
):
    out = tmp_path / "out.npy"
    sidecar = tmp_path / "out.npy.metadata.json"
    sidecar.write_text(json.dumps(
        {"mode": "dosage", "output_files": ["out.npy"], "marker": 1}
    ))
    with pytest.raises(FileExistsError, match=r"metadata\.json"):
        run_hottools(config(vcf_path, fasta_path, out))
    assert json.loads(sidecar.read_text())["marker"] == 1
    assert not out.exists()


def test_error_refuses_split_files_from_an_earlier_run(
    tmp_path, vcf_path, fasta_path
):
    out = tmp_path / "out.npz"
    run_hottools(config(vcf_path, fasta_path, out, output_format="npz",
                        batch_size=2, metadata_json=False))
    # No sidecar was written, so the batch files themselves are the only
    # evidence of the earlier run - and they are enough.
    with pytest.raises(FileExistsError, match="batch"):
        run_hottools(config(vcf_path, fasta_path, out, output_format="npz"))


def test_error_dry_run_raises_what_the_real_run_would(
    tmp_path, vcf_path, fasta_path
):
    out = tmp_path / "out.npy"
    run_hottools(config(vcf_path, fasta_path, out))
    with pytest.raises(FileExistsError):
        run_hottools(config(vcf_path, fasta_path, out, dry_run=True))


def test_dry_run_reports_planned_files_without_claiming_them(
    tmp_path, vcf_path, fasta_path
):
    meta = run_hottools(config(vcf_path, fasta_path, tmp_path / "x",
                               per_sample=True, dry_run=True))
    assert "output_files" not in meta
    assert [Path(p).name for p in meta["planned_output_files"]] == [
        "S1.npy", "S2.npy", "S3.npy", "S4.npy",
    ]
    assert not (tmp_path / "x").exists()


# --------------------------------------------------------------------
# overwrite='replace'
# --------------------------------------------------------------------

def test_replace_removes_stale_per_sample_files_only(
    tmp_path, vcf_path, fasta_path
):
    out = tmp_path / "per_sample"
    run_hottools(config(vcf_path, fasta_path, out, per_sample=True))
    assert {p.name for p in out.iterdir()} >= {"S3.npy", "S4.npy"}
    # Files hottools did not write must survive a replace, even with a
    # data suffix.
    (out / "notes.txt").write_text("keep me")
    (out / "my_analysis.npy").write_text("keep me too")

    meta = run_hottools(config(vcf_path, fasta_path, out, per_sample=True,
                               samples=["S1", "S2"], overwrite="replace"))

    assert sorted(p.name for p in out.iterdir()) == sorted([
        MANIFEST_NAME, "S1.npy", "S2.npy", "hottools.metadata.json",
        "my_analysis.npy", "notes.txt",
    ])
    assert sorted(Path(p).name for p in meta["stale_outputs_removed"]) == [
        "S3.npy", "S4.npy",
    ]
    sidecar = json.loads((out / "hottools.metadata.json").read_text())
    assert sidecar["overwrite"] == "replace"
    assert sidecar["sample_ids"] == ["S1", "S2"]


def test_replace_cleans_up_without_a_sidecar(tmp_path, vcf_path, fasta_path):
    """The manifest, not the optional sidecar, is what makes cleanup safe."""
    out = tmp_path / "per_sample"
    run_hottools(config(vcf_path, fasta_path, out, per_sample=True,
                        metadata_json=False))
    assert (out / MANIFEST_NAME).exists()
    run_hottools(config(vcf_path, fasta_path, out, per_sample=True,
                        samples=["S4"], metadata_json=False,
                        overwrite="replace"))
    assert sorted(p.name for p in out.iterdir()) == [MANIFEST_NAME, "S4.npy"]


def test_replace_switching_from_split_to_single_file(
    tmp_path, vcf_path, fasta_path
):
    out = tmp_path / "out.npz"
    run_hottools(config(vcf_path, fasta_path, out, output_format="npz",
                        batch_size=2))
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "out.batch000000-000001.npz", "out.batch000002-000003.npz",
        "out.metadata.json",
    ]
    run_hottools(config(vcf_path, fasta_path, out, output_format="npz",
                        overwrite="replace"))
    # Before the fix, the old batch files and their sidecar survived next
    # to the new file: two sidecars describing two different runs.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "out.npz", "out.npz.metadata.json",
    ]


def test_replace_leaves_a_different_output_with_the_same_stem_alone(
    tmp_path, vcf_path, fasta_path
):
    run_hottools(config(vcf_path, fasta_path, tmp_path / "out.npz",
                        output_format="npz", batch_size=2))
    npz_family = snapshot(tmp_path)
    run_hottools(config(vcf_path, fasta_path, tmp_path / "out.npy",
                        overwrite="replace"))
    after = snapshot(tmp_path)
    for name, data in npz_family.items():
        assert after[name] == data, name


def test_metadata_records_the_overwrite_policy(tmp_path, vcf_path,
                                               fasta_path):
    out = tmp_path / "out.npy"
    meta = run_hottools(config(vcf_path, fasta_path, out))
    assert meta["overwrite"] == "error"
    assert meta["stale_outputs_removed"] == []
    sidecar = json.loads((tmp_path / "out.npy.metadata.json").read_text())
    assert sidecar["overwrite"] == "error"

    run_hottools(config(vcf_path, fasta_path, out, overwrite="replace"))
    sidecar = json.loads((tmp_path / "out.npy.metadata.json").read_text())
    assert sidecar["overwrite"] == "replace"


# --------------------------------------------------------------------
# failure part-way through
# --------------------------------------------------------------------

def fail_on_second_batch(monkeypatch) -> None:
    """Make the encoder raise on its second call, as a crash would."""
    original = pipeline._encode_single_track
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated failure in batch 2")
        return original(**kwargs)

    monkeypatch.setattr(pipeline, "_encode_single_track", flaky)


@pytest.mark.parametrize("fmt", ["npy", "npz", "h5", "pt"])
@pytest.mark.parametrize("per_sample", [False, True])
def test_a_failed_replace_leaves_the_earlier_run_intact(
    tmp_path, vcf_path, fasta_path, fmt, per_sample, monkeypatch
):
    if fmt == "pt":
        pytest.importorskip("torch")
    out = tmp_path / ("dir" if per_sample else f"out.{fmt}")
    kwargs = {"output_format": fmt, "per_sample": per_sample}
    run_hottools(config(vcf_path, fasta_path, out, **kwargs))
    before = snapshot(tmp_path)

    fail_on_second_batch(monkeypatch)
    with pytest.raises(RuntimeError, match="batch 2"):
        run_hottools(config(vcf_path, fasta_path, out, batch_size=1,
                            overwrite="replace", **kwargs))

    assert snapshot(tmp_path) == before
    assert no_staging_left(tmp_path)


def test_a_failed_first_run_creates_nothing(
    tmp_path, vcf_path, fasta_path, monkeypatch
):
    fail_on_second_batch(monkeypatch)
    out = tmp_path / "fresh" / "out.npy"
    with pytest.raises(RuntimeError):
        run_hottools(config(vcf_path, fasta_path, out, batch_size=1))
    assert not out.exists()
    assert not (out.parent / "out.npy.metadata.json").exists()
    assert no_staging_left(tmp_path)


# --------------------------------------------------------------------
# filenames
# --------------------------------------------------------------------

@pytest.mark.parametrize("samples", [
    ["S_2", "S/2", "S_2-2"],          # suffix used to collide with a real name
    ["a-2", "a", "a"],                # same, generated before the real one
    ["S1", "s1"],                     # same file on a case-insensitive disk
    ["x", "X", "x-2", "X-2", "x"],
    ["", "sample", "///", "_"],
])
def test_safe_names_are_unique_case_insensitively(samples):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SampleNameCollisionWarning)
        names = safe_sample_names(samples)
    assert len(names) == len(samples)
    assert len({n.casefold() for n in names}) == len(names), names


def test_a_name_that_needs_no_change_is_never_renamed():
    with pytest.warns(SampleNameCollisionWarning):
        names = safe_sample_names(["S/2", "S_2", "S_2-2"])
    # S_2 and S_2-2 are valid IDs as given and keep their names; it is
    # the sanitised S/2 that has to move, to the first free suffix.
    assert names == ["S_2-3", "S_2", "S_2-2"]


def test_unchanged_names_raise_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert safe_sample_names(["A", "B.1", "C-2"]) == ["A", "B.1", "C-2"]


@pytest.mark.parametrize("name, expected", [
    (".hidden", "_hidden"),
    ("..", "__"),
    ("CON", "_CON"),
    ("nul.txt", "_nul.txt"),
])
def test_names_that_misbehave_as_files_are_rewritten(name, expected):
    assert safe_sample_names([name]) == [expected]


def test_overlong_names_fit_with_room_for_a_suffix():
    long = "x" * 400
    with pytest.warns(SampleNameCollisionWarning):
        a, b = safe_sample_names([long, long])
    assert len(a) <= 200 and len(b) <= 200 and a != b


def test_case_collision_reaches_distinct_files(tmp_path, fasta_path,
                                               reference_sequence):
    """End to end: S1 and s1 must not overwrite each other."""
    from conftest import ALT_OF, _bgzip_and_index, _row, _vcf_text

    ref = reference_sequence[19]
    vcf = _bgzip_and_index(
        _vcf_text([_row(20, ref, ALT_OF[ref], ["1|1", "0|0"])],
                  samples=["S1", "s1"]),
        tmp_path / "case.vcf.gz",
    )
    out = tmp_path / "per_sample"
    with pytest.warns(SampleNameCollisionWarning):
        meta = run_hottools(config(vcf, fasta_path, out, per_sample=True,
                                   mode="dosage"))
    files = [Path(p) for p in meta["output_files"]]
    assert len({p.name.casefold() for p in files}) == 2
    first, second = (np.load(p) for p in files)
    assert not np.array_equal(first, second)


# --------------------------------------------------------------------
# planning helpers
# --------------------------------------------------------------------

def test_plan_lists_every_file_before_anything_exists(tmp_path):
    plan = plan_outputs(output_path=tmp_path / "out.pt", fmt="pt",
                        per_sample=False, samples=["a", "b", "c"],
                        batch_size=2)
    assert [p.name for p in plan.data_files] == [
        "out.batch000000-000001.pt", "out.batch000002-000002.pt",
    ]
    assert plan.metadata_path.name == "out.metadata.json"
    assert plan.previous == [] and plan.conflicts == []


def test_a_recorded_path_is_never_followed_outside_the_output(tmp_path):
    out = tmp_path / "dir"
    out.mkdir()
    outside = tmp_path / "precious.npy"
    outside.write_text("do not delete")
    (out / MANIFEST_NAME).write_text(json.dumps(
        {"files": [str(outside), "../precious.npy", "S1.npy"]}
    ))
    (out / "S1.npy").write_text("old")
    found = previous_run_outputs(out, "npy", per_sample=True)
    assert outside not in found
    assert sorted(p.name for p in found) == [MANIFEST_NAME, "S1.npy"]


def test_per_sample_output_onto_a_file_is_refused(tmp_path):
    target = tmp_path / "not_a_dir"
    target.write_text("x")
    with pytest.raises(NotADirectoryError):
        previous_run_outputs(target, "npy", per_sample=True)


# --------------------------------------------------------------------
# region files: one preflight for all regions
# --------------------------------------------------------------------

def _cli(*argv: str) -> int:
    from hottools.cli import main
    return main(list(argv))


def _region_file(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "regions.bed"
    path.write_text("".join(f"{ln}\n" for ln in lines))
    return path


def test_region_file_conflict_stops_before_any_region_is_written(
    tmp_path, vcf_path, fasta_path, capsys
):
    regions = _region_file(
        tmp_path, ["chr1\t0\t60", "chr1\t60\t120", "chr1\t120\t180"]
    )
    out = tmp_path / "out"
    inputs = ("--reference", str(fasta_path), str(vcf_path))
    policies = ("--missing-policy", "reference", "--ambiguous-ref-policy",
                "skip", "--ref-mismatch-policy", "skip",
                "--unphased-policy", "missing")
    # An earlier run of the *third* region only, at the name the region
    # file would give it.
    assert _cli(*inputs, "--region", "chr1:121-180",
                "-o", str(out / "chr1_121_180.npy"), *policies) == 0
    before = snapshot(tmp_path)

    assert _cli(*inputs, "--region-file", str(regions), "-o", str(out),
                *policies) == 2
    assert "No region was written" in capsys.readouterr().err
    assert snapshot(tmp_path) == before


def test_duplicate_regions_are_refused(tmp_path, vcf_path, fasta_path,
                                       capsys):
    regions = _region_file(tmp_path, ["chr1\t0\t60", "chr1\t0\t60"])
    code = _cli("--reference", str(fasta_path), str(vcf_path),
                "--region-file", str(regions), "-o", str(tmp_path / "out"))
    assert code == 2
    assert "would both write" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_a_refused_region_run_creates_no_directory(
    tmp_path, vcf_path, fasta_path
):
    regions = _region_file(tmp_path, ["chr1\t0\t60", "chr1\t0\t60"])
    _cli("--reference", str(fasta_path), str(vcf_path),
         "--region-file", str(regions), "-o", str(tmp_path / "never"))
    assert not (tmp_path / "never").exists()


@pytest.mark.parametrize("fmt", ["npy", "h5"])
def test_a_failure_while_opening_the_output_leaves_no_staging(
    tmp_path, monkeypatch, fmt
):
    """A constructor that raises never reaches ``__exit__``."""
    from hottools import writers

    def refuse(*_args, **_kwargs):
        raise OSError("No space left on device")

    if fmt == "npy":
        import numpy.lib.format
        monkeypatch.setattr(numpy.lib.format, "open_memmap", refuse)
    else:
        import h5py
        monkeypatch.setattr(h5py, "File", refuse)
    with pytest.raises(OSError, match="No space"):
        writers.make_writer(
            output_path=tmp_path / f"out.{fmt}", fmt=fmt,
            full_shape=(2, 10, 4), dtype=np.dtype("float32"),
            samples=["a", "b"], per_sample=False, metadata={},
        )
    assert list(tmp_path.iterdir()) == []
