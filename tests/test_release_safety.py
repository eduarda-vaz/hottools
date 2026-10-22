"""Release regressions fixed in 0.7.6; these must pass on current code."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sv_helpers import Rec, write_case

from hottools.memory import choose_batch_size
from hottools.pipeline import RunConfig, run_hottools
from hottools.regions import Region
from hottools.writers import make_writer


@pytest.mark.parametrize("af_source", ["samples", "auto"])
def test_filtered_upstream_mnv_does_not_change_regional_output(tmp_path, af_source):
    fasta, vcf = write_case(
        tmp_path,
        "A" * 100,
        [
            Rec(5, "AAA", "CCC", ("1|1", "1|1")),
            Rec(7, "AAAAA", "GGGGG", ("1|0", "0|0")),
        ],
        samples=["S1", "S2"],
    )

    def reconstruct(start, end):
        out = tmp_path / f"{start}_{end}.npy"
        run_hottools(
            RunConfig(
                vcf_path=str(vcf),
                fasta_path=str(fasta),
                region=Region("chr1", start, end),
                output_path=str(out),
                mode="haplotypes",
                variant_classes="mnv",
                max_af=0.75,
                af_source=af_source,
                batch_size="none",
            )
        )
        return np.load(out)

    whole = reconstruct(1, 20)
    regional = reconstruct(10, 20)
    np.testing.assert_array_equal(regional, whole[:, :, 9:])


def test_auto_batch_respects_smaller_slurm_allocation(monkeypatch):
    import psutil

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=512 * 1024**3)
    )
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "1024")
    plan = choose_batch_size(
        n_samples=20000, sequence_length=49152, mode="haplotypes", dtype="float32"
    )
    assert plan.estimated_peak_batch_bytes <= 1024**3


@pytest.mark.parametrize("fmt", ["npy", "npz", "h5", "pt"])
def test_overwrite_error_rejects_output_published_after_writer_creation(tmp_path, fmt):
    if fmt in {"h5", "pt"}:
        pytest.importorskip("h5py" if fmt == "h5" else "torch")
    path = tmp_path / f"out.{fmt}"
    writers = [
        make_writer(
            output_path=path,
            fmt=fmt,
            full_shape=(1, 1, 4),
            dtype=np.float32,
            samples=["S"],
            per_sample=False,
            metadata={},
            overwrite="error",
        )
        for _ in range(2)
    ]
    try:
        writers[0].write(np.array([[[1, 0, 0, 0]]], np.float32), 0, 1)
        writers[1].write(np.array([[[0, 1, 0, 0]]], np.float32), 0, 1)
        writers[0].commit()
        before = path.read_bytes()
        with pytest.raises(FileExistsError):
            writers[1].commit()
        assert path.read_bytes() == before
    finally:
        for writer in writers:
            writer.abort()


def test_anchor_contig_must_match_region():
    cfg = RunConfig(
        vcf_path="unused.vcf",
        fasta_path="unused.fa",
        region=Region("chr1", 100, 200),
        output_path="unused.npy",
        anchor="chr2:150",
    )
    with pytest.raises(ValueError):
        cfg.validate()


@pytest.mark.parametrize("anchor", ["chr1:150", "150", "1:150"])
def test_explicit_anchor_forms_preserve_contig_case(anchor):
    cfg = RunConfig(
        vcf_path="unused.vcf",
        fasta_path="unused.fa",
        region=Region("chr1", 100, 200),
        output_path="unused.npy",
        anchor=anchor,
    )
    if anchor == "1:150":
        with pytest.raises(ValueError, match="contig"):
            cfg.validate()
    else:
        assert cfg.resolve_anchor() == (150, 50)
    cfg.region = Region("CHR1", 100, 200)
    cfg.anchor = "CHR1:150"
    cfg.validate()
    assert cfg.resolve_anchor() == (150, 50)


def test_live_publication_lock_rejects_second_writer(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    import hottools.writers as writers_module

    path = tmp_path / "out.npy"
    pair = [
        make_writer(
            output_path=path,
            fmt="npy",
            full_shape=(1, 1, 4),
            dtype=np.float32,
            samples=["S"],
            per_sample=False,
            metadata={},
        )
        for _ in range(2)
    ]
    pair[0].write(np.array([[[1, 0, 0, 0]]], np.float32), 0, 1)
    pair[1].write(np.array([[[0, 1, 0, 0]]], np.float32), 0, 1)
    entered, release = Event(), Event()
    real_link = writers_module.os.link

    def hold(source, target, **kwargs):
        entered.set()
        assert release.wait(5)
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(writers_module.os, "link", hold)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(pair[0].commit)
            try:
                assert entered.wait(5)
                with pytest.raises(FileExistsError, match="publishing"):
                    pair[1].commit()
            finally:
                release.set()
            future.result(timeout=5)
        np.testing.assert_array_equal(np.load(path), [[[1, 0, 0, 0]]])
    finally:
        for writer in pair:
            writer.abort()
    assert not list(tmp_path.glob("*.publish-lock"))


def test_external_late_file_is_not_clobbered(tmp_path, monkeypatch):
    import hottools.writers as writers_module

    path = tmp_path / "out.npy"
    real_link = writers_module.os.link

    def race(source, target, **kwargs):
        if Path(target) == path:
            path.write_bytes(b"external writer")
        return real_link(source, target, **kwargs)

    monkeypatch.setattr(writers_module.os, "link", race)
    with pytest.raises(FileExistsError):
        with make_writer(
            output_path=path,
            fmt="npy",
            full_shape=(1, 1, 4),
            dtype=np.float32,
            samples=["S"],
            per_sample=False,
            metadata={},
        ) as writer:
            writer.write(np.ones((1, 1, 4), np.float32), 0, 1)
    assert path.read_bytes() == b"external writer"
    assert not list(tmp_path.glob("*.publish-lock"))
    assert not list(tmp_path.glob("*.hottools-partial"))


def test_replace_refreshes_stale_per_sample_files(tmp_path):
    out = tmp_path / "samples"
    a = make_writer(
        output_path=out,
        fmt="npy",
        full_shape=(2, 1, 4),
        dtype=np.float32,
        samples=["S1", "S2"],
        per_sample=True,
        metadata={},
        overwrite="replace",
    )
    b = make_writer(
        output_path=out,
        fmt="npy",
        full_shape=(1, 1, 4),
        dtype=np.float32,
        samples=["S1"],
        per_sample=True,
        metadata={},
        overwrite="replace",
    )
    try:
        a.write(np.ones((2, 1, 4), np.float32), 0, 2)
        b.write(np.zeros((1, 1, 4), np.float32), 0, 1)
        a.commit()
        removed = b.commit()
        assert out / "S2.npy" in removed
        assert not (out / "S2.npy").exists()
        assert not np.load(out / "S1.npy").any()
    finally:
        a.abort()
        b.abort()


def test_upstream_sample_af_matches_consensus_after_record_filtering(tmp_path):
    import pysam.bcftools
    from sv_helpers import decode

    records = [
        Rec(5, "AAA", "CCC", ("1|1", "1|1")),
        Rec(7, "AAAAA", "GGGGG", ("1|0", "0|0")),
    ]
    fasta, vcf = write_case(
        tmp_path, "A" * 100, records, samples=["S1", "S2"], name="original"
    )
    _, filtered = write_case(
        tmp_path, "A" * 100, records[1:], samples=["S1", "S2"], name="filtered"
    )
    out = tmp_path / "regional.npy"
    run_hottools(
        RunConfig(
            vcf_path=str(vcf),
            fasta_path=str(fasta),
            region=Region("chr1", 10, 20),
            output_path=str(out),
            mode="haplotypes",
            variant_classes="mnv",
            af_source="samples",
            max_af=0.75,
            batch_size="none",
        )
    )
    sequence = pysam.bcftools.consensus(
        "-f", str(fasta), "-s", "S1", "-H", "1", str(filtered)
    )
    expected = "".join(
        line for line in sequence.splitlines() if not line.startswith(">")
    )[9:20]
    assert expected == "GGAAAAAAAAA"
    assert decode(np.load(out)[0, 0]) == expected


def test_standalone_metadata_error_mode_is_exclusive(tmp_path, monkeypatch):
    import hottools.writers as writers_module

    path = tmp_path / "metadata.json"
    original = writers_module.check_destination

    def late_file(candidate, overwrite):
        checked = original(candidate, overwrite)
        checked.write_text("external metadata")
        return checked

    monkeypatch.setattr(writers_module, "check_destination", late_file)
    with pytest.raises(FileExistsError):
        writers_module.write_metadata_json(path, {"new": True})
    assert path.read_text() == "external metadata"


def test_replace_lock_stays_stable_when_output_was_symlink(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    import hottools.writers as writers_module

    original = tmp_path / "original.npy"
    np.save(original, np.zeros((1, 1, 4), np.float32))
    out = tmp_path / "out.npy"
    try:
        out.symlink_to(original)
    except OSError:
        pytest.skip("symlink creation unavailable")
    pair = [
        make_writer(
            output_path=out,
            fmt="npy",
            full_shape=(1, 1, 4),
            dtype=np.float32,
            samples=["S"],
            per_sample=False,
            metadata={},
            overwrite="replace",
        )
        for _ in range(2)
    ]
    pair[0].write(np.array([[[1, 0, 0, 0]]], np.float32), 0, 1)
    pair[1].write(np.array([[[0, 1, 0, 0]]], np.float32), 0, 1)
    replaced, release = Event(), Event()
    real_replace = writers_module.os.replace

    def hold(source, target):
        result = real_replace(source, target)
        if Path(target) == out:
            replaced.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(writers_module.os, "replace", hold)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(pair[0].commit)
            try:
                assert replaced.wait(5)
                assert not out.is_symlink()
                with pytest.raises(FileExistsError, match="publishing"):
                    pair[1].commit()
            finally:
                release.set()
            future.result(timeout=5)
        np.testing.assert_array_equal(np.load(out), [[[1, 0, 0, 0]]])
        assert not np.load(original).any()
    finally:
        for writer in pair:
            writer.abort()
