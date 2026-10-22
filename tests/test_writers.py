import json

import numpy as np
import pytest

from hottools.writers import (
    DEFAULT_H5_CHUNK_BYTES,
    auto_h5_chunks,
    make_writer,
    metadata_path_for,
    write_metadata_json,
)

h5py = pytest.importorskip("h5py")

SAMPLES = ["S1", "S/2", "S3"]
METADATA = {"mode": "dosage", "chrom": "chr1", "samples": 3, "sample_ids": SAMPLES}


def _batches(full_shape, dtype=np.float32, batch=2):
    rng = np.random.default_rng(0)
    data = rng.random(full_shape).astype(dtype)
    bounds = [
        (s, min(s + batch, full_shape[0]))
        for s in range(0, full_shape[0], batch)
    ]
    return data, bounds


def _run_writer(tmp_path, fmt, full_shape, per_sample=False, batch=2, **kwargs):
    out = tmp_path / ("out_dir" if per_sample else f"out.{fmt}")
    data, bounds = _batches(full_shape, batch=batch)
    with make_writer(
        output_path=out, fmt=fmt, full_shape=full_shape,
        dtype=np.dtype("float32"), samples=SAMPLES[: full_shape[0]],
        per_sample=per_sample, metadata=METADATA, batch_size=batch,
        **kwargs,
    ) as writer:
        for start, end in bounds:
            writer.write(data[start:end], start, end)
        written = list(writer.written_paths)
    return out, data, written


# --------------------------------------------------------------------
# HDF5 chunk sizing
# --------------------------------------------------------------------

def test_auto_chunks_keeps_whole_samples_when_small():
    chunks = auto_h5_chunks((100, 1000, 4), 4, sequence_axis=1)
    assert chunks[1] == 1000
    assert chunks[0] > 1
    assert np.prod(chunks) * 4 <= DEFAULT_H5_CHUNK_BYTES


def test_auto_chunks_splits_the_sequence_axis_for_long_regions():
    """A 524 kb window must not produce a 512 MiB chunk."""
    full = (2504, 524288, 4)
    chunks = auto_h5_chunks(full, 4, sequence_axis=1)
    assert chunks[0] == 1
    assert chunks[1] < 524288
    assert np.prod(chunks) * 4 <= DEFAULT_H5_CHUNK_BYTES


def test_auto_chunks_haplotypes_and_channels_first():
    hap = auto_h5_chunks((2504, 2, 524288, 4), 4, sequence_axis=2)
    assert np.prod(hap) * 4 <= DEFAULT_H5_CHUNK_BYTES
    cf = auto_h5_chunks((2504, 4, 524288), 4, sequence_axis=2)
    assert np.prod(cf) * 4 <= DEFAULT_H5_CHUNK_BYTES


def test_auto_chunks_never_exceeds_the_dataset():
    chunks = auto_h5_chunks((2, 10, 4), 4, sequence_axis=1)
    assert chunks == (2, 10, 4)


def test_auto_chunks_honours_explicit_overrides():
    chunks = auto_h5_chunks(
        (100, 524288, 4), 4, sequence_axis=1,
        chunk_samples=8, chunk_length=1024,
    )
    assert chunks == (8, 1024, 4)


def test_auto_chunks_rejects_bad_axis():
    with pytest.raises(ValueError):
        auto_h5_chunks((10, 20, 4), 4, sequence_axis=0)


# --------------------------------------------------------------------
# writers
# --------------------------------------------------------------------

def test_npy_writer_roundtrip(tmp_path):
    out, data, written = _run_writer(tmp_path, "npy", (3, 10, 4))
    np.testing.assert_array_equal(np.load(out), data)
    assert written == [out]


def test_h5_writer_roundtrip_and_metadata(tmp_path):
    out, data, written = _run_writer(tmp_path, "h5", (3, 10, 4))
    with h5py.File(out) as f:
        np.testing.assert_array_equal(f["sequences"][:], data)
        assert [s.decode() if isinstance(s, bytes) else s
                for s in f["samples"][:]] == SAMPLES
        assert f.attrs["chrom"] == "chr1"
        assert f["sequences"].chunks is not None
    assert written == [out]


def test_h5_chunk_bytes_are_bounded_for_a_long_region(tmp_path):
    out = tmp_path / "long.h5"
    full = (2, 300_000, 4)
    with make_writer(
        output_path=out, fmt="h5", full_shape=full,
        dtype=np.dtype("float32"), samples=SAMPLES[:2], per_sample=False,
        metadata=METADATA, sequence_axis=1,
    ) as writer:
        chunks = writer.chunks
    assert np.prod(chunks) * 4 <= DEFAULT_H5_CHUNK_BYTES
    with h5py.File(out) as f:
        assert f["sequences"].chunks == chunks


def test_npz_single_batch_writes_requested_path(tmp_path):
    out, data, written = _run_writer(tmp_path, "npz", (2, 10, 4), batch=2)
    assert written == [out]
    np.testing.assert_array_equal(np.load(out)["sequences"], data)


def test_npz_multiple_batches_split_files(tmp_path):
    out = tmp_path / "out.npz"
    full = (3, 10, 4)
    data, bounds = _batches(full, batch=2)
    with make_writer(
        output_path=out, fmt="npz", full_shape=full,
        dtype=np.dtype("float32"), samples=SAMPLES, per_sample=False,
        metadata=METADATA, batch_size=2,
    ) as writer:
        for start, end in bounds:
            writer.write(data[start:end], start, end)
        written = list(writer.written_paths)
    assert [p.name for p in written] == [
        "out.batch000000-000001.npz", "out.batch000002-000002.npz"
    ]
    assert not out.exists()
    rejoined = np.concatenate([np.load(p)["sequences"] for p in written])
    np.testing.assert_array_equal(rejoined, data)


@pytest.mark.parametrize("fmt", ["npy", "h5", "npz"])
def test_per_sample_files_are_named_and_sanitised(tmp_path, fmt):
    _out, data, written = _run_writer(
        tmp_path / fmt, fmt, (3, 10, 4), per_sample=True
    )
    names = sorted(p.name for p in written)
    assert names == sorted(f"{n}.{fmt}" for n in ["S1", "S_2", "S3"])
    first = next(p for p in written if p.name.startswith("S1"))
    if fmt == "npy":
        loaded = np.load(first)
    elif fmt == "npz":
        loaded = np.load(first)["sequences"]
    else:
        with h5py.File(first) as f:
            loaded = f["sequence"][:]
    np.testing.assert_array_equal(loaded, data[0])


def test_unknown_format_rejected(tmp_path):
    with pytest.raises(ValueError, match="format must be one of"):
        make_writer(
            output_path=tmp_path / "x.foo", fmt="foo", full_shape=(1, 2, 4),
            dtype=np.dtype("float32"), samples=["S1"], per_sample=False,
            metadata={},
        )


# --------------------------------------------------------------------
# metadata sidecar
# --------------------------------------------------------------------

def test_metadata_path_variants(tmp_path):
    out = tmp_path / "x.npy"
    assert metadata_path_for(out, per_sample=False).name == "x.npy.metadata.json"
    assert metadata_path_for(out, per_sample=True).name == "hottools.metadata.json"
    batched = metadata_path_for(tmp_path / "x.npz", per_sample=False, batched=True)
    assert batched.name == "x.metadata.json"


def test_metadata_json_is_serialisable(tmp_path):
    path = write_metadata_json(
        tmp_path / "m.json",
        {"a": np.int64(3), "b": np.float32(1.5), "c": [tmp_path / "f.npy"], "d": "x"},
    )
    loaded = json.loads(path.read_text())
    assert loaded["a"] == 3
    assert loaded["b"] == pytest.approx(1.5)
    assert loaded["c"] == [str(tmp_path / "f.npy")]
