import json

import numpy as np
import pytest
from conftest import REF_LENGTH, SAMPLES

from hottools.pipeline import RunConfig, _apply_reverse_complement, run_hottools
from hottools.regions import Region

COMP = [3, 2, 1, 0]


def config_for(tmp_path, vcf, fasta, **overrides):
    kwargs = dict(
        vcf_path=str(vcf),
        fasta_path=str(fasta),
        region=Region("chr1", 1, REF_LENGTH),
        output_path=str(tmp_path / "out.npy"),
        batch_size="none",
    )
    kwargs.update(overrides)
    return RunConfig(**kwargs)


# --------------------------------------------------------------------
# strand: one source of truth
# --------------------------------------------------------------------

def test_reverse_complement_is_derived_from_the_region(tmp_path, clean_vcf_path, fasta_path):
    forward = config_for(tmp_path, clean_vcf_path, fasta_path)
    assert forward.reverse_complement is False and forward.strand == "+"
    reverse = config_for(
        tmp_path, clean_vcf_path, fasta_path,
        region=Region("chr1", 1, REF_LENGTH, "-"),
    )
    assert reverse.reverse_complement is True and reverse.strand == "-"


def test_metadata_strand_agrees_with_the_data(tmp_path, clean_vcf_path, fasta_path):
    fwd = run_hottools(config_for(tmp_path, clean_vcf_path, fasta_path))
    rev = run_hottools(config_for(
        tmp_path, clean_vcf_path, fasta_path,
        region=Region("chr1", 1, REF_LENGTH, "-"),
        output_path=str(tmp_path / "rev.npy"),
    ))
    assert (fwd["strand"], fwd["reverse_complement"]) == ("+", False)
    assert (rev["strand"], rev["reverse_complement"]) == ("-", True)

    forward = np.load(tmp_path / "out.npy")
    reverse = np.load(tmp_path / "rev.npy")
    np.testing.assert_array_equal(reverse, forward[:, ::-1, :][:, :, COMP])


def test_sidecar_strand_matches_metadata(tmp_path, clean_vcf_path, fasta_path):
    run_hottools(config_for(
        tmp_path, clean_vcf_path, fasta_path,
        region=Region("chr1", 1, REF_LENGTH, "-"),
    ))
    sidecar = json.loads((tmp_path / "out.npy.metadata.json").read_text())
    assert sidecar["strand"] == "-"
    assert sidecar["reverse_complement"] is True


@pytest.mark.parametrize("mode,layout", [
    ("dosage", "samples_first"), ("dosage", "channels_first"),
    ("haplotypes", "samples_first"), ("haplotypes", "channels_first"),
])
def test_reverse_complement_is_an_involution(mode, layout):
    rng = np.random.default_rng(0)
    shape = {
        ("dosage", "samples_first"): (3, 11, 4),
        ("dosage", "channels_first"): (3, 4, 11),
        ("haplotypes", "samples_first"): (3, 2, 11, 4),
        ("haplotypes", "channels_first"): (3, 2, 4, 11),
    }[(mode, layout)]
    batch = rng.random(shape).astype(np.float32)
    once = _apply_reverse_complement(batch, mode=mode, layout=layout)
    twice = _apply_reverse_complement(once, mode=mode, layout=layout)
    np.testing.assert_array_equal(twice, batch)
    assert not np.array_equal(once, batch)


def test_reverse_complement_matches_hand_computation():
    # A C G T  ->  reverse complement is  A C G T reversed and complemented
    ref = np.eye(4, dtype=np.float32)[None]  # 1 x 4 x 4 (one sample, ACGT)
    out = _apply_reverse_complement(ref, mode="dosage", layout="samples_first")
    np.testing.assert_array_equal(out[0], np.eye(4, dtype=np.float32))


# --------------------------------------------------------------------
# end-to-end content
# --------------------------------------------------------------------

def test_dosage_output_matches_reference_plus_variants(
    tmp_path, clean_vcf_path, fasta_path, reference_onehot, variant_spec
):
    meta = run_hottools(config_for(tmp_path, clean_vcf_path, fasta_path))
    out = np.load(tmp_path / "out.npy")
    assert out.shape == (4, REF_LENGTH, 4)
    assert meta["output_shape"] == [4, REF_LENGTH, 4]

    variant_pos0 = [variant_spec["clean_phased"]["pos"] - 1,
                    variant_spec["clean_phased_2"]["pos"] - 1]
    untouched = [i for i in range(REF_LENGTH) if i not in variant_pos0]
    for sample in range(4):
        np.testing.assert_array_equal(
            out[sample][untouched], reference_onehot[untouched]
        )

    bases = "ACGT"
    for key in ("clean_phased", "clean_phased_2"):
        spec = variant_spec[key]
        col = spec["pos"] - 1
        ref_i, alt_i = bases.index(spec["ref"]), bases.index(spec["alt"])
        np.testing.assert_array_equal(out[0, col], reference_onehot[col])
        assert out[1, col, ref_i] == 0.5 and out[1, col, alt_i] == 0.5
        assert out[2, col, alt_i] == 1.0


def test_haplotype_output_shape_and_phase(
    tmp_path, clean_vcf_path, fasta_path, variant_spec
):
    run_hottools(config_for(
        tmp_path, clean_vcf_path, fasta_path, mode="haplotypes"
    ))
    out = np.load(tmp_path / "out.npy")
    assert out.shape == (4, 2, REF_LENGTH, 4)
    spec = variant_spec["clean_phased"]
    col = spec["pos"] - 1
    bases = "ACGT"
    # S2 is 0|1: hap0 carries REF, hap1 carries ALT.
    assert out[1, 0, col, bases.index(spec["ref"])] == 1.0
    assert out[1, 1, col, bases.index(spec["alt"])] == 1.0


def test_sample_rows_follow_the_request(tmp_path, clean_vcf_path, fasta_path):
    run_hottools(config_for(tmp_path, clean_vcf_path, fasta_path))
    full = np.load(tmp_path / "out.npy")
    meta = run_hottools(config_for(
        tmp_path, clean_vcf_path, fasta_path,
        samples=["S4", "S1"], output_path=str(tmp_path / "sub.npy"),
    ))
    sub = np.load(tmp_path / "sub.npy")
    assert meta["sample_ids"] == ["S4", "S1"]
    np.testing.assert_array_equal(sub[0], full[SAMPLES.index("S4")])
    np.testing.assert_array_equal(sub[1], full[SAMPLES.index("S1")])


def test_batching_produces_the_same_tensor(tmp_path, clean_vcf_path, fasta_path):
    run_hottools(config_for(tmp_path, clean_vcf_path, fasta_path, batch_size="none"))
    whole = np.load(tmp_path / "out.npy")
    run_hottools(config_for(
        tmp_path, clean_vcf_path, fasta_path, batch_size=1,
        output_path=str(tmp_path / "batched.npy"),
    ))
    np.testing.assert_array_equal(np.load(tmp_path / "batched.npy"), whole)


@pytest.mark.parametrize("layout,expected", [
    ("samples_first", (4, REF_LENGTH, 4)),
    ("channels_first", (4, 4, REF_LENGTH)),
])
def test_output_layouts(tmp_path, clean_vcf_path, fasta_path, layout, expected):
    meta = run_hottools(config_for(
        tmp_path, clean_vcf_path, fasta_path, output_layout=layout
    ))
    assert np.load(tmp_path / "out.npy").shape == expected
    assert tuple(meta["output_shape"]) == expected


def test_float16_output(tmp_path, clean_vcf_path, fasta_path):
    run_hottools(config_for(tmp_path, clean_vcf_path, fasta_path, dtype="float16"))
    out = np.load(tmp_path / "out.npy")
    assert out.dtype == np.float16
    assert 0.5 in set(np.unique(out).tolist())


@pytest.mark.parametrize("fmt", ["npy", "npz", "h5"])
def test_formats_roundtrip(tmp_path, clean_vcf_path, fasta_path, fmt):
    pytest.importorskip("h5py") if fmt == "h5" else None
    out = tmp_path / f"out.{fmt}"
    meta = run_hottools(config_for(
        tmp_path, clean_vcf_path, fasta_path,
        output_format=fmt, output_path=str(out),
    ))
    assert meta["n_output_files"] == 1
    assert out.exists()


def test_per_sample_outputs(tmp_path, clean_vcf_path, fasta_path):
    out_dir = tmp_path / "persample"
    meta = run_hottools(config_for(
        tmp_path, clean_vcf_path, fasta_path,
        per_sample=True, output_path=str(out_dir),
    ))
    assert meta["n_output_files"] == 4
    assert sorted(p.name for p in out_dir.glob("*.npy")) == [
        "S1.npy", "S2.npy", "S3.npy", "S4.npy"
    ]
    assert (out_dir / "hottools.metadata.json").exists()
    assert np.load(out_dir / "S1.npy").shape == (REF_LENGTH, 4)


# --------------------------------------------------------------------
# metadata, memory plan, validation
# --------------------------------------------------------------------

def test_metadata_is_complete(tmp_path, clean_vcf_path, fasta_path):
    meta = run_hottools(config_for(tmp_path, clean_vcf_path, fasta_path))
    for key in (
        "mode", "chrom", "start", "end", "length", "strand",
        "reverse_complement", "samples", "sample_ids", "variants_kept",
        "reference_ambiguous_bases", "dtype", "output_layout",
        "output_shape", "batch_size", "n_batches", "output_files",
        "estimated_peak_batch_bytes", "peak_multiplier",
        "variant_records_seen", "variant_kept", "timings",
    ):
        assert key in meta, key
    assert meta["sample_ids"] == SAMPLES
    assert meta["reference_ambiguous_bases"] == 1  # the single 'N'


def test_peak_multiplier_reflects_reverse_complement(
    tmp_path, clean_vcf_path, fasta_path
):
    plain = run_hottools(config_for(
        tmp_path, clean_vcf_path, fasta_path, batch_size="auto", dry_run=True
    ))
    rc = run_hottools(config_for(
        tmp_path, clean_vcf_path, fasta_path, batch_size="auto", dry_run=True,
        region=Region("chr1", 1, REF_LENGTH, "-"),
    ))
    assert plain["peak_multiplier"] == 1.0
    assert rc["peak_multiplier"] == 2.0


def test_dry_run_writes_nothing(tmp_path, clean_vcf_path, fasta_path):
    meta = run_hottools(config_for(tmp_path, clean_vcf_path, fasta_path, dry_run=True))
    assert meta["variants_kept"] == 2
    assert not (tmp_path / "out.npy").exists()
    assert "output_files" not in meta


def test_empty_region_warns_through_the_pipeline(
    tmp_path, clean_vcf_path, fasta_path
):
    from hottools.variants import EmptyRegionWarning

    with pytest.warns(EmptyRegionWarning):
        run_hottools(config_for(
            tmp_path, clean_vcf_path, fasta_path,
            region=Region("chr1", 150, 200),
        ))
    out = np.load(tmp_path / "out.npy")
    # Every sample collapses to the reference: this is the failure mode
    # the warning exists to announce.
    assert np.all(out == out[0])


@pytest.mark.parametrize("field,value", [
    ("dtype", "float64"), ("output_layout", "sideways"),
    ("output_format", "parquet"), ("mode", "nonsense"),
])
def test_config_validation(tmp_path, clean_vcf_path, fasta_path, field, value):
    config = config_for(tmp_path, clean_vcf_path, fasta_path, **{field: value})
    with pytest.raises(ValueError):
        run_hottools(config)


def test_shared_reference_reader_is_reused(tmp_path, clean_vcf_path, fasta_path):
    from hottools.reference import ReferenceReader

    with ReferenceReader(fasta_path) as reader:
        for i in range(3):
            run_hottools(config_for(
                tmp_path, clean_vcf_path, fasta_path,
                reference_reader=reader,
                output_path=str(tmp_path / f"r{i}.npy"),
            ))
        # The handle must still be usable afterwards.
        assert len(reader.interval("chr1", 1, 10)) == 10
