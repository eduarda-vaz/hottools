"""
Turn compact genotypes into one-hot channel tensors.

Public API
----------
``encode_dosage_batch``, ``encode_consensus_batch`` and
``encode_haplotype_batch`` share one keyword signature and are
self-contained: each validates its inputs, builds the lookup tables it
needs and encodes one sample range. They suit one-off calls and
notebooks.

The pipeline instead builds the tables once per run with the internal
``_build_variant_tables`` and reuses them for every sample batch. That
object, and the ``_encode_*`` functions that consume it, are internal:
their layout (one ``V x 36 x 4`` table per track) is an implementation
detail free to change between releases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    ALLELE_ABSENT,
    ALLELE_MISSING,
    CODE_ALLELE1,
    CODE_ALLELE2,
    CODE_PLOIDY,
    HAPLOID_TRACK_POLICIES,
    HET_POLICIES,
    LAYOUTS,
    MISSING_POLICIES,
    MODES,
    N_ALLELE_SLOTS,
    N_ALLELES,
    N_GENOTYPE_CODES,
    NO_ALLELE,
    HaploidTrackPolicy,
    HetPolicy,
    MissingPolicy,
    Mode,
    OutputLayout,
    check_choice,
)

__all__ = [
    "allele_slot_vectors",
    "consensus_tables",
    "dosage_tables",
    "encode_consensus_batch",
    "encode_dosage_batch",
    "encode_haplotype_batch",
    "encode_haplotypes",
    "haplotype_track_tables",
]

_I1 = CODE_ALLELE1.astype(np.int16)
_I2 = CODE_ALLELE2.astype(np.int16)
_ABSENT1 = _I1 == int(ALLELE_ABSENT)
_ABSENT2 = _I2 == int(ALLELE_ABSENT)

#: Packed codes in which at least one existing allele slot is uncalled.
_CODES_WITH_MISSING = np.flatnonzero(
    (_I1 == int(ALLELE_MISSING)) | (_I2 == int(ALLELE_MISSING))
).astype(np.uint8)


def _float_dtype(dtype) -> np.dtype:
    """Refuse integer and other non-floating dtypes.

    Dosage is fractional, so an integer tensor would silently truncate a
    heterozygote's 0.5 to 0.
    """
    out = np.dtype(dtype)
    if not np.issubdtype(out, np.floating):
        raise ValueError(f"dtype must be a floating type; got {out}")
    return out


# ------------------------------------------------------------------
# per-variant lookup tables
# ------------------------------------------------------------------


def allele_slot_vectors(
    allele_bases: np.ndarray,
    dtype: np.dtype | str,
    missing_policy: str,
) -> np.ndarray:
    """
    ``V x N_ALLELE_SLOTS x 4`` channel vector for each allele slot.

    ``allele_bases`` is the per-variant allele table: entry *k* is the
    base index of allele *k*, or ``NO_ALLELE`` where the record has
    fewer alternates. The ``ALLELE_MISSING`` slot resolves per
    ``missing_policy``; the ``ALLELE_ABSENT`` slot stays zero, because
    a hemizygous sample contributes only one allele and the dosage
    divisor -- not a filler vector -- is what accounts for that.
    """
    check_choice("missing_policy", missing_policy, MISSING_POLICIES)
    dtype = _float_dtype(dtype)
    n_variants = allele_bases.shape[0]
    vec = np.zeros((n_variants, N_ALLELE_SLOTS, 4), dtype=dtype)
    rows = np.arange(n_variants)
    for k in range(N_ALLELES):
        bases = allele_bases[:, k]
        present = bases != NO_ALLELE
        if np.any(present):
            vec[rows[present], k, bases[present]] = 1
    if missing_policy == "reference":
        vec[:, ALLELE_MISSING, :] = vec[:, 0, :]
    # 'zero' leaves the missing slot all-zero. 'error' does too: under
    # that policy no missing call may reach the encoder, and the public
    # wrappers check for one before encoding.
    return vec


def haplotype_track_tables(
    vec: np.ndarray,
    haploid_policy: HaploidTrackPolicy = "duplicate",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Channel vectors for haplotype track 1 and track 2.

    ``haploid_policy`` governs the second *track* only: ``duplicate``
    copies a hemizygous sample's single allele onto it, ``blank`` leaves
    it empty. Neither choice changes the dosage or consensus encodings,
    which are ploidy-aware instead.
    """
    check_choice("haploid_policy", haploid_policy, HAPLOID_TRACK_POLICIES)

    # Fancy indexing already returns a fresh array, so no .copy() here.
    table1 = vec[:, _I1, :]
    table2 = vec[:, _I2, :]

    if haploid_policy == "duplicate":
        table2[:, _ABSENT2, :] = table1[:, _ABSENT2, :]
    # 'blank' leaves table2 at the zero vector already in vec[ABSENT].

    # A first slot can only be absent for a malformed record; read it as
    # missing rather than as silence.
    if np.any(_ABSENT1):
        table1[:, _ABSENT1, :] = vec[:, ALLELE_MISSING, :][:, None, :]
    return table1, table2


def dosage_tables(vec: np.ndarray) -> np.ndarray:
    """
    Ploidy-aware allele dosage.

    Each existing allele contributes an equal share, so the divisor is
    the number of allele slots that exist (``CODE_PLOIDY``), not always
    two. A hemizygous ALT call therefore encodes as full-weight ALT,
    while a half-called diploid genotype (one allele uncalled) keeps
    half weight under ``missing_policy='zero'`` -- the two situations
    are genuinely different and encode differently.
    """
    contrib = vec[:, _I1, :]
    contrib += vec[:, _I2, :]
    contrib /= CODE_PLOIDY.astype(vec.dtype).reshape(1, -1, 1)
    return contrib


def consensus_tables(
    vec: np.ndarray,
    het_policy: HetPolicy,
    allele_freq: np.ndarray | None = None,
) -> np.ndarray:
    """
    One channel vector per genotype code, carrying a single allele.

    Heterozygotes are collapsed:

    ``reference``   the REF allele (or the lower-indexed ALT when the
                    genotype carries two different ALTs)
    ``alternate``   the ALT allele (the lower-indexed one if both
                    alleles are ALT)
    ``major``       whichever allele is more frequent across the
                    selected samples, ties going to the lower index

    The result is one-hot wherever the chosen allele is a real base. It
    is the all-zero vector where the genotype is uncalled under
    ``missing_policy='zero'``; masked and ambiguous-reference positions
    are likewise zero, so the output is not one-hot everywhere.

    This encoding does not depend on the haploid track policy: a
    hemizygous call has exactly one allele, and that allele is used.
    """
    check_choice("het_policy", het_policy, HET_POLICIES)
    if het_policy == "major":
        if allele_freq is None:
            raise ValueError("het_policy='major' requires allele_freq")
        allele_freq = np.asarray(allele_freq)
        if allele_freq.shape != (vec.shape[0], N_ALLELES):
            raise ValueError(
                f"allele_freq must have shape ({vec.shape[0]}, {N_ALLELES}); "
                f"got {allele_freq.shape}"
            )

        if (
            not (
                np.issubdtype(allele_freq.dtype, np.floating)
                or np.issubdtype(allele_freq.dtype, np.integer)
            )
            or not np.all(np.isfinite(allele_freq))
            or np.any((allele_freq < 0) | (allele_freq > 1))
        ):
            raise ValueError(
                "allele_freq must contain finite frequencies between 0 and 1"
            )

    a1 = np.where(_ABSENT1, int(ALLELE_MISSING), _I1)
    a2 = np.where(_ABSENT2, int(ALLELE_MISSING), _I2)

    known1 = a1 < N_ALLELES
    known2 = a2 < N_ALLELES
    chosen = np.full(N_GENOTYPE_CODES, int(ALLELE_MISSING), dtype=np.intp)
    chosen = np.where(known1 & ~known2, a1, chosen)
    chosen = np.where(~known1 & known2, a2, chosen)

    both = known1 & known2
    chosen = np.where(both & (a1 == a2), a1, chosen)
    het = both & (a1 != a2)

    if het_policy == "reference":
        pick = np.where((a1 == 0) | (a2 == 0), 0, np.minimum(a1, a2))
    elif het_policy == "alternate":
        both_alt = (a1 != 0) & (a2 != 0)
        first_alt = np.where(a1 != 0, a1, a2)
        pick = np.where(both_alt, np.minimum(a1, a2), first_alt)
    else:
        safe1 = np.clip(a1, 0, N_ALLELES - 1)
        safe2 = np.clip(a2, 0, N_ALLELES - 1)
        f1 = allele_freq[:, safe1]  # V x codes
        f2 = allele_freq[:, safe2]
        lower = np.minimum(a1, a2)
        pick = np.where(f1 > f2, a1, np.where(f2 > f1, a2, lower))
        chosen_v = np.where(het[None, :], pick, chosen[None, :])
        rows = np.arange(vec.shape[0])[:, None]
        return vec[rows, chosen_v, :]

    # 'reference' and 'alternate' choose the same allele slot for every
    # variant, so one 1-D take along the slot axis does the whole job.
    chosen = np.where(het, pick, chosen)
    return vec[:, chosen, :]


@dataclass(slots=True)
class _VariantTables:
    """
    Everything per-run that the batch encoders need. Internal.

    Built once per run rather than once per sample batch: the tables
    depend only on the variants and the policies, never on which samples
    are being encoded. ``positions`` is the 0-based variant offsets as a
    plain list, converted once here so no batch pays for it again.
    Each table has shape ``V x N_GENOTYPE_CODES x 4``.
    """

    mode: Mode
    positions: list[int]
    length: int
    primary: np.ndarray  # dosage/consensus, or haplotype 1
    secondary: np.ndarray | None = None  # haplotype 2

    @property
    def nbytes(self) -> int:
        total = self.primary.nbytes
        if self.secondary is not None:
            total += self.secondary.nbytes
        return total

    def __len__(self) -> int:
        return self.primary.shape[0]


def _check_positions(variant_pos0: np.ndarray, length: int) -> np.ndarray:
    """Validate 0-based variant offsets against a sequence of ``length``.

    NumPy accepts negative indices, so an out-of-range offset would not
    fail: -1 would quietly write the last base. Duplicates would let one
    variant overwrite another. Both are refused here instead.
    """
    pos = np.asarray(variant_pos0)
    if pos.ndim != 1:
        raise ValueError("variant_pos0 must be one-dimensional")
    if pos.size == 0:
        return pos.astype(np.int64)
    if not np.issubdtype(pos.dtype, np.integer):
        raise ValueError(f"variant_pos0 must hold integers; got dtype {pos.dtype}")
    lo, hi = int(pos.min()), int(pos.max())
    if lo < 0 or hi >= length:
        raise ValueError(
            f"variant_pos0 must lie in [0, {length}); got range [{lo}, {hi}]"
        )
    if np.unique(pos).size != pos.size:
        raise ValueError("variant_pos0 contains duplicate positions")
    return pos


def _build_variant_tables(
    *,
    allele_bases: np.ndarray,
    variant_pos0: np.ndarray,
    sequence_length: int,
    dtype: np.dtype | str,
    mode: Mode = "dosage",
    missing_policy: MissingPolicy = "zero",
    haploid_policy: HaploidTrackPolicy = "duplicate",
    het_policy: HetPolicy = "alternate",
    allele_freq: np.ndarray | None = None,
) -> _VariantTables:
    """Build the lookup tables one run needs, once. Internal."""
    check_choice("mode", mode, MODES)
    pos = _check_positions(variant_pos0, sequence_length)
    if pos.shape[0] != allele_bases.shape[0]:
        raise ValueError(
            f"{pos.shape[0]} variant positions but "
            f"{allele_bases.shape[0]} allele-table rows"
        )
    positions = pos.tolist()
    vec = allele_slot_vectors(allele_bases, dtype, missing_policy)

    if mode == "haplotypes":
        table1, table2 = haplotype_track_tables(vec, haploid_policy)
        return _VariantTables("haplotypes", positions, sequence_length, table1, table2)
    if mode == "dosage":
        return _VariantTables("dosage", positions, sequence_length, dosage_tables(vec))
    return _VariantTables(
        "consensus",
        positions,
        sequence_length,
        consensus_tables(vec, het_policy, allele_freq),
    )


def _table_bytes(n_variants: int, mode: Mode, dtype: np.dtype) -> int:
    """What ``_build_variant_tables`` would allocate, without building."""
    n_tables = 2 if mode == "haplotypes" else 1
    return n_variants * N_GENOTYPE_CODES * 4 * np.dtype(dtype).itemsize * n_tables


# ------------------------------------------------------------------
# batch encoders (internal: they trust a validated _VariantTables)
# ------------------------------------------------------------------


def _apply_layout(
    seq: np.ndarray, output_layout: OutputLayout, haplotypes: bool
) -> np.ndarray:
    if output_layout == "samples_first":
        return seq
    if output_layout == "channels_first":
        return (
            np.transpose(seq, (0, 1, 3, 2))
            if haplotypes
            else np.transpose(seq, (0, 2, 1))
        )
    raise ValueError(
        f"output_layout must be one of: {', '.join(LAYOUTS)}; got {output_layout!r}"
    )


def _canvas(ref_onehot: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.broadcast_to(ref_onehot, shape).copy()


def _encode_single_track(
    *,
    ref_onehot: np.ndarray,
    genotypes: np.ndarray,
    tables: _VariantTables,
    sample_start: int,
    sample_end: int,
    output_layout: OutputLayout = "samples_first",
) -> np.ndarray:
    """One sample batch as a single ``B x L x 4`` track (dosage, consensus)."""
    if tables.mode not in ("dosage", "consensus"):
        raise ValueError(
            "tables must have been built for 'dosage' or 'consensus'; "
            f"got {tables.mode!r}"
        )
    gt = genotypes[:, sample_start:sample_end]
    seq = _canvas(ref_onehot, (sample_end - sample_start, ref_onehot.shape[0], 4))
    table = tables.primary
    for i, pos in enumerate(tables.positions):
        seq[:, pos, :] = table[i][gt[i]]
    return _apply_layout(seq, output_layout, haplotypes=False)


def _encode_haplotype_tracks(
    *,
    ref_onehot: np.ndarray,
    genotypes: np.ndarray,
    tables: _VariantTables,
    sample_start: int,
    sample_end: int,
    output_layout: OutputLayout = "samples_first",
) -> np.ndarray:
    """
    One sample batch as two ``B x 2 x L x 4`` haplotype tracks.

    Hemizygous samples still occupy two tracks so the tensor stays
    rectangular; which allele the second track carries was decided when
    the tables were built.
    """
    if tables.mode != "haplotypes":
        raise ValueError(
            f"tables were built for mode {tables.mode!r}, not 'haplotypes'"
        )
    gt = genotypes[:, sample_start:sample_end]
    seq = _canvas(ref_onehot, (sample_end - sample_start, 2, ref_onehot.shape[0], 4))
    table1, table2 = tables.primary, tables.secondary
    for i, pos in enumerate(tables.positions):
        codes = gt[i]
        seq[:, 0, pos, :] = table1[i][codes]
        seq[:, 1, pos, :] = table2[i][codes]
    return _apply_layout(seq, output_layout, haplotypes=True)


# ------------------------------------------------------------------
# public, self-contained encoders
# ------------------------------------------------------------------


def _validate_batch_inputs(
    *,
    ref_onehot: np.ndarray,
    variant_pos0: np.ndarray,
    allele_bases: np.ndarray,
    genotypes: np.ndarray,
    sample_start: int,
    sample_end: int | None,
    missing_policy: MissingPolicy,
    output_layout: OutputLayout,
) -> int:
    """Check everything a public encoder is given; return ``sample_end``."""
    check_choice("missing_policy", missing_policy, MISSING_POLICIES)
    check_choice("output_layout", output_layout, LAYOUTS)

    if ref_onehot.ndim != 2 or ref_onehot.shape[1] != 4:
        raise ValueError(f"ref_onehot must have shape (L, 4); got {ref_onehot.shape}")
    _float_dtype(ref_onehot.dtype)

    if genotypes.ndim != 2:
        raise ValueError(f"genotypes must have shape (V, S); got {genotypes.shape}")
    if genotypes.dtype != np.uint8:
        raise ValueError(
            f"genotypes must be packed uint8 codes; got dtype {genotypes.dtype}"
        )
    if genotypes.size and int(genotypes.max()) >= N_GENOTYPE_CODES:
        raise ValueError(
            f"genotypes holds codes >= {N_GENOTYPE_CODES}, which are not "
            "valid packed allele pairs"
        )

    if allele_bases.ndim != 2 or allele_bases.shape[1] != N_ALLELES:
        raise ValueError(
            f"allele_bases must have shape (V, {N_ALLELES}); got {allele_bases.shape}"
        )
    if allele_bases.shape[0] != genotypes.shape[0]:
        raise ValueError(
            f"{allele_bases.shape[0]} allele-table rows but "
            f"{genotypes.shape[0]} genotype rows"
        )
    if allele_bases.size:
        if int(allele_bases[:, 0].max()) >= 4:
            raise ValueError("allele_bases column 0 (REF) must be a base 0-3")
        alts = allele_bases[:, 1:]
        if np.any((alts >= 4) & (alts != NO_ALLELE)):
            raise ValueError("allele_bases ALT entries must be a base 0-3 or NO_ALLELE")

    n_samples = genotypes.shape[1]
    end = n_samples if sample_end is None else int(sample_end)
    if not 0 <= int(sample_start) < end <= n_samples:
        raise ValueError(
            f"sample range [{sample_start}, {end}) is not a non-empty range "
            f"within the {n_samples} samples"
        )

    if missing_policy == "error":
        window = genotypes[:, sample_start:end]
        if np.isin(window, _CODES_WITH_MISSING).any():
            raise ValueError(
                "missing_policy='error' but the genotypes contain an uncalled allele"
            )
    return end


def encode_dosage_batch(
    *,
    ref_onehot: np.ndarray,
    variant_pos0: np.ndarray,
    allele_bases: np.ndarray,
    genotypes: np.ndarray,
    sample_start: int = 0,
    sample_end: int | None = None,
    missing_policy: MissingPolicy = "zero",
    output_layout: OutputLayout = "samples_first",
) -> np.ndarray:
    """
    Allele dosage for samples ``[sample_start, sample_end)``: ``B x L x 4``.

    A heterozygote is 0.5/0.5 across its two bases; a hemizygous call is
    full weight.
    """
    end = _validate_batch_inputs(
        ref_onehot=ref_onehot,
        variant_pos0=variant_pos0,
        allele_bases=allele_bases,
        genotypes=genotypes,
        sample_start=sample_start,
        sample_end=sample_end,
        missing_policy=missing_policy,
        output_layout=output_layout,
    )
    tables = _build_variant_tables(
        allele_bases=allele_bases,
        variant_pos0=variant_pos0,
        sequence_length=ref_onehot.shape[0],
        dtype=ref_onehot.dtype,
        mode="dosage",
        missing_policy=missing_policy,
    )
    return _encode_single_track(
        ref_onehot=ref_onehot,
        genotypes=genotypes,
        tables=tables,
        sample_start=sample_start,
        sample_end=end,
        output_layout=output_layout,
    )


def encode_consensus_batch(
    *,
    ref_onehot: np.ndarray,
    variant_pos0: np.ndarray,
    allele_bases: np.ndarray,
    genotypes: np.ndarray,
    sample_start: int = 0,
    sample_end: int | None = None,
    het_policy: HetPolicy = "alternate",
    allele_freq: np.ndarray | None = None,
    missing_policy: MissingPolicy = "zero",
    output_layout: OutputLayout = "samples_first",
) -> np.ndarray:
    """
    One allele per sample and position: ``B x L x 4``.

    ``allele_freq`` (``V x 4``) is needed only for ``het_policy='major'``;
    ``reference`` and ``alternate`` ignore it.
    """
    end = _validate_batch_inputs(
        ref_onehot=ref_onehot,
        variant_pos0=variant_pos0,
        allele_bases=allele_bases,
        genotypes=genotypes,
        sample_start=sample_start,
        sample_end=sample_end,
        missing_policy=missing_policy,
        output_layout=output_layout,
    )
    tables = _build_variant_tables(
        allele_bases=allele_bases,
        variant_pos0=variant_pos0,
        sequence_length=ref_onehot.shape[0],
        dtype=ref_onehot.dtype,
        mode="consensus",
        missing_policy=missing_policy,
        het_policy=het_policy,
        allele_freq=allele_freq,
    )
    return _encode_single_track(
        ref_onehot=ref_onehot,
        genotypes=genotypes,
        tables=tables,
        sample_start=sample_start,
        sample_end=end,
        output_layout=output_layout,
    )


def encode_haplotype_batch(
    *,
    ref_onehot: np.ndarray,
    variant_pos0: np.ndarray,
    allele_bases: np.ndarray,
    genotypes: np.ndarray,
    sample_start: int = 0,
    sample_end: int | None = None,
    haploid_policy: HaploidTrackPolicy = "duplicate",
    missing_policy: MissingPolicy = "zero",
    output_layout: OutputLayout = "samples_first",
) -> np.ndarray:
    """
    Two haplotype tracks per sample: ``B x 2 x L x 4``.

    ``haploid_policy`` decides what the second track carries for a
    hemizygous sample: a copy of its allele (``duplicate``) or nothing
    (``blank``).
    """
    end = _validate_batch_inputs(
        ref_onehot=ref_onehot,
        variant_pos0=variant_pos0,
        allele_bases=allele_bases,
        genotypes=genotypes,
        sample_start=sample_start,
        sample_end=sample_end,
        missing_policy=missing_policy,
        output_layout=output_layout,
    )
    tables = _build_variant_tables(
        allele_bases=allele_bases,
        variant_pos0=variant_pos0,
        sequence_length=ref_onehot.shape[0],
        dtype=ref_onehot.dtype,
        mode="haplotypes",
        missing_policy=missing_policy,
        haploid_policy=haploid_policy,
    )
    return _encode_haplotype_tracks(
        ref_onehot=ref_onehot,
        genotypes=genotypes,
        tables=tables,
        sample_start=sample_start,
        sample_end=end,
        output_layout=output_layout,
    )


#: Earlier name for :func:`encode_haplotype_batch`, kept so existing
#: callers keep working. New code should use the ``_batch`` name, which
#: matches the dosage and consensus encoders.
encode_haplotypes = encode_haplotype_batch
