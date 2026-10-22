"""
Fixed-length structural transformations.

A length-changing edit breaks the one-to-one mapping between reference
and personalised coordinates, so it cannot be applied by substitution.
Reconstructing a sequence per haplotype would also give up the one
property that makes hottools fast, so instead:

1. Each haplotype gets a **signature** -- which length-changing edits it
   carries. Structural variants are far rarer than SNVs, so across a
   cohort only a handful of distinct signatures usually occur.
2. Each distinct signature is compiled **once** into a fixed-length
   index map saying where every output position comes from: a position
   in the SNV-personalised working sequence, or a base from the insert
   pool.
3. Every haplotype sharing a signature is transformed together with one
   advanced-indexing gather.

The map is compiled directly at the requested output length around an
anchor, rather than building a variable-length sequence and cropping it
afterwards, so a deletion upstream of the anchor does not slide the
anchor within the model input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    ALLELE_ABSENT,
    ALLELE_MISSING,
    ANCHOR_DELETION_POLICIES,
    CODE_ALLELE1,
    CODE_ALLELE2,
    HAPLOID_TRACK_POLICIES,
    check_choice,
)
from .edits import StructuralEdits

#: Marks an output position that comes from the insert pool.
FROM_INSERT = np.int32(-1)
#: Marks an output position with no source at all (ran off the contig).
FROM_NOWHERE = np.int32(-2)


@dataclass(slots=True)
class HaplotypeSignatures:
    """Which length-changing edits each haplotype carries."""

    #: (2S, E) bool -- row ``s * 2 + h`` is sample *s*, haplotype *h*.
    carriers: np.ndarray
    #: (G, E) bool, the distinct rows of ``carriers``.
    unique: np.ndarray
    #: (2S,) int, index into ``unique`` for each haplotype row.
    group: np.ndarray

    @property
    def n_groups(self) -> int:
        return int(self.unique.shape[0])

    @property
    def n_haplotypes(self) -> int:
        return int(self.carriers.shape[0])

    @property
    def reference_group(self) -> int | None:
        """The group carrying no edits at all, if present."""
        if self.unique.shape[1] == 0:
            return 0 if self.n_groups else None
        empty = np.flatnonzero(~self.unique.any(axis=1))
        return int(empty[0]) if empty.size else None


def haplotype_signatures(
    edits: StructuralEdits,
    haploid_policy: str = "duplicate",
) -> HaplotypeSignatures:
    """
    Group haplotypes by the set of length-changing edits they carry.

    A haplotype carries edit *e* when its allele at ``edits.record[e]``
    equals ``edits.allele[e]``. Missing alleles carry nothing: an
    uncalled genotype is not evidence of a structural variant.

    ``haploid_policy`` must match the one the substitution tables were
    built with. Under ``duplicate`` a hemizygous sample's second track
    copies its single allele, so it carries that allele's structural
    edits too; under ``blank`` the second track has no allele and carries
    none. Without this the sequence track and the structural track of
    the same haplotype would disagree.
    """
    check_choice("haploid_policy", haploid_policy, HAPLOID_TRACK_POLICIES)
    codes = edits.genotypes                      # R x S
    n_samples = codes.shape[1]
    n_edits = edits.n_edits

    allele1 = CODE_ALLELE1[codes]                # R x S
    allele2 = CODE_ALLELE2[codes]
    if haploid_policy == "duplicate":
        allele2 = np.where(allele2 == ALLELE_ABSENT, allele1, allele2)
    # (R, 2S): haplotype-major within each sample, matching the encoder's
    # (sample, haplotype) axis order when flattened.
    per_hap = np.empty((codes.shape[0], n_samples * 2), dtype=np.uint8)
    per_hap[:, 0::2] = allele1
    per_hap[:, 1::2] = allele2

    carriers = np.zeros((n_samples * 2, n_edits), dtype=bool)
    for e in range(n_edits):
        wanted = edits.allele[e]
        row = per_hap[edits.record[e]]
        carriers[:, e] = (row == wanted) & (row < ALLELE_MISSING)

    if n_edits == 0:
        unique = np.zeros((1, 0), dtype=bool)
        group = np.zeros(n_samples * 2, dtype=np.intp)
    else:
        packed = np.packbits(carriers, axis=1)
        _, first, inverse = np.unique(
            packed, axis=0, return_index=True, return_inverse=True
        )
        unique = carriers[first]
        group = np.asarray(inverse, dtype=np.intp).reshape(-1)
    return HaplotypeSignatures(carriers, unique, group)


@dataclass(slots=True)
class Transform:
    """A compiled fixed-length index map for one structural signature."""

    #: (L,) int32 -- offset into the working sequence, or FROM_INSERT /
    #: FROM_NOWHERE.
    source: np.ndarray
    #: (L,) int32 -- index into the insert pool where source is FROM_INSERT.
    insert: np.ndarray
    #: Number of output positions with no source (contig ran out).
    n_unfilled: int
    #: Edits dropped for overlapping an earlier edit on the same
    #: haplotype.
    n_skipped: int = 0

    @property
    def is_identity(self) -> bool:
        return bool(
            self.n_unfilled == 0
            and not np.any(self.source < 0)
            and np.array_equal(
                self.source, np.arange(self.source.shape[0], dtype=np.int32)
            )
        )


def _segments(
    active: np.ndarray,
    edits: StructuralEdits,
    work_start: int,
    work_end: int,
) -> list[tuple[int, int, int]]:
    """
    Walk the working interval under ``active`` edits.

    Returns ``(segments, n_skipped)``. Each segment is a
    ``(source_kind, a, b)`` triple in output order, where
    ``source_kind`` is 0 for a reference run ``[a, b)`` in working
    offsets and 1 for an insert-pool run ``[a, b)``. ``n_skipped``
    counts edits dropped for overlapping an earlier one.
    """
    # By start, and at an equal start the zero-length insertion first: it
    # sits before the base the other edit begins at, so it can never be
    # inside it. Without the tie-break the result depended on input order.
    starts = edits.start[active]
    lengths = edits.end[active] - starts
    chosen = active[np.lexsort((lengths, starts))]

    out: list[tuple[int, int, int]] = []
    skipped = 0
    cursor = work_start                      # 1-based reference position
    for e in chosen:
        start = int(edits.start[e])
        end = int(edits.end[e])
        # An edit is outside the window if it ends before it starts or
        # begins after it ends. A zero-length insertion exactly at either
        # edge is inside: it sits between the window's first base and the
        # one before it, or after its last base, just as it would anywhere
        # else.
        if start > work_end + 1 or end < work_start or (
            end == work_start and end > start
        ):
            continue
        if start < work_start:
            # Clipped by the window edge, not by another edit: an event
            # that begins upstream still removes the bases it covers
            # inside the window, so the visible part is applied.
            start = work_start
        if start < cursor:
            # Two records on one haplotype claiming the same reference
            # bases are contradictory input. Applying the leftover piece
            # of the second would synthesise a composite event neither
            # record describes, so the later record is skipped whole -
            # the same resolution bcftools consensus uses.
            skipped += 1
            continue
        if start > cursor:
            out.append((0, cursor - work_start, start - work_start))
        length = int(edits.insert_length[e])
        if length:
            offset = int(edits.insert_offset[e])
            out.append((1, offset, offset + length))
        cursor = max(cursor, end)
    if cursor <= work_end:
        out.append((0, cursor - work_start, work_end - work_start + 1))
    return out, skipped


def compile_transform(
    active: np.ndarray,
    edits: StructuralEdits,
    *,
    work_start: int,
    work_end: int,
    out_length: int,
    anchor_ref: int,
    anchor_offset: int,
    anchor_deletion_policy: str = "right",
) -> Transform:
    """
    Compile one signature into a fixed-length index map.

    ``work_start``/``work_end`` bound the working reference interval
    (1-based inclusive) that the SNV stage personalised.
    ``anchor_ref`` is the reference position pinned to output offset
    ``anchor_offset``.
    """
    segments, n_skipped = _segments(active, edits, work_start, work_end)

    pieces_src: list[np.ndarray] = []
    pieces_ins: list[np.ndarray] = []
    anchor_index: int | None = None
    produced = 0
    for kind, a, b in segments:
        length = b - a
        if length <= 0:
            continue
        if kind == 0:
            pieces_src.append(np.arange(a, b, dtype=np.int32))
            pieces_ins.append(np.full(length, FROM_INSERT, dtype=np.int32))
            ref_lo = work_start + a
            if anchor_index is None and ref_lo <= anchor_ref < work_start + b:
                anchor_index = produced + (anchor_ref - ref_lo)
        else:
            pieces_src.append(np.full(length, FROM_INSERT, dtype=np.int32))
            pieces_ins.append(np.arange(a, b, dtype=np.int32))
        produced += length

    if pieces_src:
        stream_src = np.concatenate(pieces_src)
        stream_ins = np.concatenate(pieces_ins)
    else:
        stream_src = np.empty(0, dtype=np.int32)
        stream_ins = np.empty(0, dtype=np.int32)

    if anchor_index is None:
        # The anchor base itself was deleted on this haplotype.
        if anchor_deletion_policy == "error":
            raise ValueError(
                f"The anchor base at position {anchor_ref} is deleted on "
                "at least one haplotype, so it has no place in the output. "
                "Use --anchor-deletion-policy right to pin the first base "
                "after the deletion instead."
            )
        # 'right': the first surviving base after it takes its place. If
        # nothing survives after it, the anchor sits just past the end of
        # the stream - never at its start, which would put the window
        # upstream of where it belongs.
        after = np.flatnonzero(stream_src >= (anchor_ref - work_start))
        anchor_index = int(after[0]) if after.size else stream_src.shape[0]

    lo = anchor_index - anchor_offset
    source = np.full(out_length, FROM_NOWHERE, dtype=np.int32)
    insert = np.full(out_length, FROM_INSERT, dtype=np.int32)

    src_lo = max(lo, 0)
    src_hi = min(lo + out_length, stream_src.shape[0])
    if src_hi > src_lo:
        dst_lo = src_lo - lo
        dst_hi = dst_lo + (src_hi - src_lo)
        source[dst_lo:dst_hi] = stream_src[src_lo:src_hi]
        insert[dst_lo:dst_hi] = stream_ins[src_lo:src_hi]

    n_unfilled = int(np.count_nonzero(source == FROM_NOWHERE))
    return Transform(source, insert, n_unfilled, n_skipped)


def apply_transform(
    work: np.ndarray,
    rows: np.ndarray,
    transform: Transform,
    pool: np.ndarray,
    out: np.ndarray,
) -> None:
    """
    Write the transformed haplotypes of one signature group into ``out``.

    ``work`` is ``(H, W, 4)`` and ``out`` is ``(H, L, 4)``, both flattened
    over (sample, haplotype). ``rows`` selects the haplotypes in this
    group; every one of them shares ``transform``.
    """
    source = transform.source
    from_ref = source >= 0
    if rows.size == 0:
        return

    rows_2d = rows[:, None]
    if np.all(from_ref):
        out[rows_2d, np.arange(source.shape[0])[None, :], :] = work[
            rows_2d, source[None, :], :
        ]
        return

    ref_positions = np.flatnonzero(from_ref)
    if ref_positions.size:
        out[rows_2d, ref_positions[None, :], :] = work[
            rows_2d, source[ref_positions][None, :], :
        ]

    ins_positions = np.flatnonzero(source == FROM_INSERT)
    if ins_positions.size:
        out[rows_2d, ins_positions[None, :], :] = pool[
            transform.insert[ins_positions]
        ][None, :, :]

    gap_positions = np.flatnonzero(source == FROM_NOWHERE)
    if gap_positions.size:
        out[rows_2d, gap_positions[None, :], :] = 0


def compile_all(
    signatures: HaplotypeSignatures,
    edits: StructuralEdits,
    *,
    work_start: int,
    work_end: int,
    out_length: int,
    anchor_ref: int,
    anchor_offset: int,
    anchor_deletion_policy: str = "right",
) -> list[Transform]:
    """Compile every distinct signature once."""
    check_choice(
        "anchor_deletion_policy", anchor_deletion_policy,
        ANCHOR_DELETION_POLICIES,
    )
    return [
        compile_transform(
            np.flatnonzero(signatures.unique[g]),
            edits,
            work_start=work_start,
            work_end=work_end,
            out_length=out_length,
            anchor_ref=anchor_ref,
            anchor_offset=anchor_offset,
            anchor_deletion_policy=anchor_deletion_policy,
        )
        for g in range(signatures.n_groups)
    ]
