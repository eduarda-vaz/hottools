
import numpy as np
import pytest
from test_overwrite import config, no_staging_left, snapshot

import hottools.writers as writers
from hottools.pipeline import run_hottools
from hottools.writers import make_writer


@pytest.mark.parametrize("failure_at", range(1, 7))
def test_each_commit_rename_restores_complete_previous_run(
    tmp_path, vcf_path, fasta_path, monkeypatch, failure_at
):
    out = tmp_path / "samples"
    run_hottools(config(vcf_path, fasta_path, out, per_sample=True))
    before = snapshot(tmp_path)
    real_replace = writers.os.replace
    calls = 0

    def fail_once(source, target):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise OSError("commit fault")
        return real_replace(source, target)

    monkeypatch.setattr(writers.os, "replace", fail_once)
    # Two data files, manifest, sidecar, then retiring S3 and S4.
    with pytest.raises(OSError, match="commit fault"):
        run_hottools(config(vcf_path, fasta_path, out, per_sample=True,
                            samples=["S1", "S2"], overwrite="replace"))
    assert snapshot(tmp_path) == before
    assert no_staging_left(tmp_path)


@pytest.mark.parametrize("failure_at", range(1, 5))
def test_first_run_commit_failure_removes_new_files_and_directory(
    tmp_path, vcf_path, fasta_path, monkeypatch, failure_at
):
    out = tmp_path / "samples"
    real_link = writers.os.link
    calls = 0

    def fail_once(source, target):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise OSError("commit fault")
        return real_link(source, target)

    monkeypatch.setattr(writers.os, "link", fail_once)
    with pytest.raises(OSError, match="commit fault"):
        run_hottools(config(vcf_path, fasta_path, out, per_sample=True,
                            samples=["S1", "S2"]))
    assert not out.exists()
    assert no_staging_left(tmp_path)


def test_rollback_failure_preserves_original_backup(tmp_path, monkeypatch):
    out = tmp_path / "samples"
    out.mkdir()
    np.save(out / "S1.npy", np.zeros((2, 4), dtype=np.float32))
    np.save(out / "S2.npy", np.zeros((2, 4), dtype=np.float32))
    real_replace = writers.os.replace
    calls = 0

    def fail_commit_and_restore(source, target):
        nonlocal calls
        calls += 1
        if calls in (2, 3):
            raise OSError("persistent fault")
        return real_replace(source, target)

    monkeypatch.setattr(writers.os, "replace", fail_commit_and_restore)
    with pytest.raises(RuntimeError, match="recovery files preserved"):
        with make_writer(output_path=out, fmt="npy", full_shape=(2, 2, 4),
                         dtype=np.float32, samples=["S1", "S2"],
                         per_sample=True, metadata={}, overwrite="replace") as writer:
            writer.write(np.ones((2, 2, 4), dtype=np.float32), 0, 2)
    backups = list(tmp_path.glob("*.hottools-partial/.rollback/original-*"))
    assert backups
    np.testing.assert_array_equal(np.load(backups[0]), np.zeros((2, 4)))


def test_metadata_staging_failure_cleans_up(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("metadata fault")

    monkeypatch.setattr(writers, "write_metadata_json", fail)
    out = tmp_path / "out.npy"
    with pytest.raises(OSError, match="metadata fault"):
        with make_writer(output_path=out, fmt="npy", full_shape=(1, 2, 4),
                         dtype=np.float32, samples=["S1"], per_sample=False,
                         metadata={}, metadata_json=True) as writer:
            writer.write(np.ones((1, 2, 4), dtype=np.float32), 0, 1)
            writer.commit({"mode": "dosage"})
    assert not out.exists()
    assert no_staging_left(tmp_path)


def test_backup_copy_failure_leaves_existing_output_intact(
    tmp_path, vcf_path, fasta_path, monkeypatch
):
    out = tmp_path / "samples"
    run_hottools(config(vcf_path, fasta_path, out, per_sample=True))
    before = snapshot(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("backup fault")

    monkeypatch.setattr(writers.os, "link", fail)
    monkeypatch.setattr(writers.shutil, "copy2", fail)
    with pytest.raises(OSError, match="backup fault"):
        run_hottools(config(vcf_path, fasta_path, out, per_sample=True,
                            overwrite="replace"))
    assert snapshot(tmp_path) == before
    assert no_staging_left(tmp_path)
