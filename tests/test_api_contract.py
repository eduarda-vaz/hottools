"""
The public API's contract: one vocabulary, validated everywhere.

Policies and modes are defined once in ``hottools.constants``. Every
entry point - ``RunConfig``, the public encoders, the table builders -
must reject a value outside that vocabulary instead of quietly falling
back to some other behaviour.
"""

from __future__ import annotations

import inspect
import typing

import numpy as np
import pytest

import hottools
import hottools.constants as C
from hottools import encoding
from hottools.constants import NO_ALLELE, pack_genotype
from hottools.encoding import (
    encode_consensus_batch,
    encode_dosage_batch,
    encode_haplotype_batch,
    encode_haplotypes,
)
from hottools.pipeline import RunConfig, run_hottools
from hottools.regions import Region

ENCODERS = (encode_dosage_batch, encode_consensus_batch,
            encode_haplotype_batch)

REF = np.eye(4, dtype=np.float32)[[0, 1, 2]]       # A C G
POS = np.array([1])
ALLELES = np.array([[1, 3, NO_ALLELE, NO_ALLELE]], dtype=np.uint8)
CODES = np.array([[pack_genotype(0, 1), pack_genotype(1, 1)]],
                 dtype=np.uint8)


def call(fn, **overrides):
    kwargs = {
        "ref_onehot": REF, "variant_pos0": POS, "allele_bases": ALLELES,
        "genotypes": CODES,
    }
    kwargs.update(overrides)
    return fn(**kwargs)


# --------------------------------------------------------------------
# one vocabulary
# --------------------------------------------------------------------

@pytest.mark.parametrize("alias, values", [
    (C.Mode, C.MODES),
    (C.MissingPolicy, C.MISSING_POLICIES),
    (C.HetPolicy, C.HET_POLICIES),
    (C.HaploidPolicy, C.HAPLOID_POLICIES),
    (C.OverwritePolicy, C.OVERWRITE_POLICIES),
    (C.OutputLayout, C.LAYOUTS),
    (C.OutputFormat, C.FORMATS),
    (C.AfSource, C.AF_SOURCES),
    (C.OverlapPolicy, C.OVERLAP_POLICIES),
    (C.AnchorDeletionPolicy, C.ANCHOR_DELETION_POLICIES),
])
def test_each_type_alias_matches_its_runtime_vocabulary(alias, values):
    assert typing.get_args(alias) == values


def test_consensus_is_a_mode_wherever_mode_is_typed():
    """The pipeline once typed ``mode`` as dosage|haplotypes only."""
    from hottools import memory, pipeline, variants
    for module in (encoding, memory, pipeline, variants):
        assert "consensus" in typing.get_args(module.Mode), module.__name__
    hints = typing.get_type_hints(RunConfig)
    assert hints["mode"] is C.Mode
    assert hints["het_policy"] is C.HetPolicy
    assert hints["mixed_record_policy"] is C.MixedRecordPolicy
    assert hints["missing_policy"] is C.MissingPolicy
    assert hints["overwrite"] is C.OverwritePolicy


# --------------------------------------------------------------------
# RunConfig validates every choice before any file is opened
# --------------------------------------------------------------------

@pytest.mark.parametrize("field, bad", [
    ("mode", "Consensus"),
    ("missing_policy", "Reference"),
    ("missing_policy", "ref"),
    ("unphased_policy", "ignore"),
    ("mixed_record_policy", "ignore"),
    ("ref_mismatch_policy", "drop"),
    ("ambiguous_ref_policy", "N"),
    ("haploid_policy", "copy"),
    ("ploidy_policy", "warn"),
    ("het_policy", "alt"),
    ("af_source", "INFO"),
    ("overwrite", "force"),
    ("dtype", "float64"),
    ("output_layout", "sideways"),
])
def test_run_config_rejects_values_outside_the_vocabulary(
    tmp_path, vcf_path, fasta_path, field, bad
):
    config = RunConfig(
        vcf_path=str(vcf_path), fasta_path=str(fasta_path),
        region=Region("chr1", 1, 200), output_path=str(tmp_path / "x.npy"),
        **{field: bad},
    )
    with pytest.raises(ValueError, match=field):
        run_hottools(config)
    assert not any(tmp_path.iterdir())


# --------------------------------------------------------------------
# public encoders
# --------------------------------------------------------------------

def test_public_encoders_share_one_signature():
    common = {"ref_onehot", "variant_pos0", "allele_bases", "genotypes",
              "sample_start", "sample_end", "missing_policy",
              "output_layout"}
    for fn in ENCODERS:
        params = inspect.signature(fn).parameters
        assert common <= set(params), fn.__name__
        assert all(p.kind is p.KEYWORD_ONLY for p in params.values())
        assert params["sample_start"].default == 0
        assert params["sample_end"].default is None


def test_haplotype_encoder_name_matches_the_others():
    assert encode_haplotypes is encode_haplotype_batch


@pytest.mark.parametrize("het_policy", ["reference", "alternate"])
def test_consensus_needs_no_allele_freq_unless_major(het_policy):
    out = call(encode_consensus_batch, het_policy=het_policy)
    assert out.shape == (2, 3, 4)


def test_consensus_major_demands_allele_freq_of_the_right_shape():
    with pytest.raises(ValueError, match="requires allele_freq"):
        call(encode_consensus_batch, het_policy="major")
    with pytest.raises(ValueError, match="shape"):
        call(encode_consensus_batch, het_policy="major",
             allele_freq=np.zeros((2, 4)))
    freq = np.array([[0.25, 0.75, 0.0, 0.0]])
    out = call(encode_consensus_batch, het_policy="major", allele_freq=freq)
    np.testing.assert_array_equal(out[0, 1], [0, 0, 0, 1])   # ALT T


@pytest.mark.parametrize("fn", ENCODERS)
@pytest.mark.parametrize("bad", ["Reference", "ref", "bogus", None])
def test_encoders_validate_missing_policy(fn, bad):
    """'Reference', 'ref' and 'bogus' used to encode silently as 'zero'."""
    with pytest.raises(ValueError, match="missing_policy"):
        call(fn, missing_policy=bad)


@pytest.mark.parametrize("fn", ENCODERS)
def test_missing_policy_error_refuses_an_uncalled_allele(fn):
    codes = np.array([[pack_genotype(0, 1), pack_genotype(4, 4)]],
                     dtype=np.uint8)
    with pytest.raises(ValueError, match="uncalled"):
        call(fn, genotypes=codes, missing_policy="error")
    # The same call without the missing sample is fine.
    call(fn, genotypes=codes, missing_policy="error", sample_end=1)


@pytest.mark.parametrize("fn", ENCODERS)
@pytest.mark.parametrize("positions, message", [
    (np.array([-1]), r"\[0, 3\)"),         # used to write the last base
    (np.array([3]), r"\[0, 3\)"),
    (np.array([1.0]), "integers"),
    (np.array([[1]]), "one-dimensional"),
])
def test_encoders_validate_positions(fn, positions, message):
    with pytest.raises(ValueError, match=message):
        call(fn, variant_pos0=positions)


def test_duplicate_positions_are_refused():
    alleles = np.repeat(ALLELES, 2, axis=0)
    codes = np.repeat(CODES, 2, axis=0)
    with pytest.raises(ValueError, match="duplicate"):
        call(encode_dosage_batch, variant_pos0=np.array([1, 1]),
             allele_bases=alleles, genotypes=codes)


@pytest.mark.parametrize("overrides, message", [
    ({"genotypes": CODES.astype(np.int16)}, "uint8"),
    ({"genotypes": np.full((1, 2), 36, np.uint8)}, "codes >= 36"),
    ({"genotypes": CODES[0]}, r"shape \(V, S\)"),
    ({"allele_bases": ALLELES[:, :3]}, "allele_bases"),
    ({"allele_bases": np.array([[7, 3, NO_ALLELE, NO_ALLELE]], np.uint8)},
     "REF"),
    ({"allele_bases": np.array([[1, 9, NO_ALLELE, NO_ALLELE]], np.uint8)},
     "ALT"),
    ({"ref_onehot": REF.astype(np.int8)}, "floating"),
    ({"ref_onehot": REF[:, :3]}, r"\(L, 4\)"),
    ({"sample_start": 1, "sample_end": 1}, "sample range"),
    ({"sample_end": 3}, "sample range"),
    ({"output_layout": "sideways"}, "output_layout"),
])
def test_encoders_validate_their_inputs(overrides, message):
    for fn in ENCODERS:
        with pytest.raises(ValueError, match=message):
            call(fn, **overrides)


def test_sample_range_defaults_to_every_sample():
    whole = call(encode_haplotype_batch)
    part = call(encode_haplotype_batch, sample_start=1, sample_end=2)
    assert whole.shape == (2, 2, 3, 4)
    np.testing.assert_array_equal(part, whole[1:2])


# --------------------------------------------------------------------
# what is public and what is not
# --------------------------------------------------------------------

def test_table_machinery_is_internal():
    for name in ("VariantTables", "build_variant_tables",
                 "encode_single_track_batch"):
        assert not hasattr(hottools, name), name
        assert name not in hottools.__all__
        assert not hasattr(encoding, name), name
    assert set(encoding.__all__) >= {
        "encode_dosage_batch", "encode_consensus_batch",
        "encode_haplotype_batch",
    }
    assert all(not n.startswith("_") for n in hottools.__all__)


def test_tables_convert_positions_once():
    tables = encoding._build_variant_tables(
        allele_bases=ALLELES, variant_pos0=np.array([1], dtype=np.int32),
        sequence_length=3, dtype=np.float32, mode="dosage",
    )
    assert tables.positions == [1]
    assert all(type(p) is int for p in tables.positions)


@pytest.mark.parametrize('frequency', [np.nan, np.inf, -.1, 1.1, 'bad', 1j])
def test_major_consensus_rejects_invalid_frequency_values(frequency):
    freq = np.asarray([[frequency, 0, 0, 0]])
    with pytest.raises(ValueError, match='allele_freq'):
        call(encode_consensus_batch, het_policy='major', allele_freq=freq)
