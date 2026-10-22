"""
Normalised reference-space edits.

Every variant hottools can apply is expressed as "replace the half-open
reference interval ``[start, end)`` with ``alt``":

====================  ==================  =========================
variant               interval            alt
====================  ==================  =========================
SNV                   one base            one base
MNV                   *n* bases           *n* bases
deletion              *n* bases           empty
insertion             empty               *n* bases
replacement           *n* bases           *m* bases, ``m != n``
precise ``<DEL>``     POS+1 .. END        empty
====================  ==================  =========================

Edits whose interval and replacement have the same length do not move
any coordinate, so they stay on the vectorised substitution path in
:mod:`hottools.encoding`. The rest change the mapping between reference
and personalised coordinates and are handled by
:mod:`hottools.transforms`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .constants import BASES, VALID_BASES

#: Variant classes the caller can opt into. ``snv`` is always enabled.
VariantClass = Literal["snv", "mnv", "del", "ins", "symbolic-del"]
VARIANT_CLASSES = ("snv", "mnv", "del", "ins", "symbolic-del")
#: Classes that change sequence length, and so need the transform stage.
LENGTH_CHANGING = frozenset({"del", "ins", "symbolic-del"})

EDIT_DELETION = np.uint8(0)
EDIT_INSERTION = np.uint8(1)
EDIT_REPLACEMENT = np.uint8(2)

EDIT_KIND_NAMES = {
    int(EDIT_DELETION): "deletion",
    int(EDIT_INSERTION): "insertion",
    int(EDIT_REPLACEMENT): "replacement",
}


class UnsupportedVariant(ValueError):
    """The record cannot be expressed as an exact reference-space edit."""


def parse_variant_classes(spec: str | None) -> frozenset[str]:
    """
    Parse a ``--variant-classes`` specification.

    ``snv`` is always included; an empty or ``None`` spec means SNVs
    only, which is the default behaviour of the package.
    """
    if not spec:
        return frozenset({"snv"})
    requested = {part.strip().lower() for part in spec.split(",") if part.strip()}
    #: Groups expand to several classes; plain aliases rename one.
    groups = {
        "all": frozenset(VARIANT_CLASSES),
        "indel": frozenset({"del", "ins"}),
        "indels": frozenset({"del", "ins"}),
    }
    aliases = {
        "snvs": "snv", "snp": "snv", "snps": "snv",
        "mnp": "mnv", "mnps": "mnv", "mnvs": "mnv",
        "deletion": "del", "deletions": "del", "dels": "del",
        "insertion": "ins", "insertions": "ins",
        "symbolic_del": "symbolic-del", "symbolicdel": "symbolic-del",
        "symbolic-deletion": "symbolic-del",
    }
    resolved: set[str] = set()
    for item in requested:
        if item in groups:
            resolved.update(groups[item])
            continue
        name = aliases.get(item, item)
        if name not in VARIANT_CLASSES:
            raise ValueError(
                f"Unknown variant class {item!r}; choose from "
                f"{', '.join(VARIANT_CLASSES)}, indel, or all"
            )
        resolved.add(name)
    resolved.add("snv")
    return frozenset(resolved)


@dataclass(frozen=True, slots=True)
class Edit:
    """One normalised edit of one ALT allele."""

    start: int          # 1-based, first replaced reference base
    end: int            # 1-based exclusive; start == end means an insertion
    alt: str            # replacement bases, "" for a deletion
    variant_class: str  # snv | mnv | del | ins | symbolic-del
    allele: int         # which ALT allele of its record (1-based)

    @property
    def ref_length(self) -> int:
        return self.end - self.start

    @property
    def length_change(self) -> int:
        return len(self.alt) - self.ref_length

    @property
    def is_length_changing(self) -> bool:
        return self.length_change != 0

    @property
    def kind(self) -> np.uint8:
        if not self.alt:
            return EDIT_DELETION
        if self.ref_length == 0:
            return EDIT_INSERTION
        return EDIT_REPLACEMENT


def _trim(pos: int, ref: str, alt: str) -> tuple[int, int, str]:
    """
    Strip the bases an ALT shares with REF.

    Returns ``(start, end, alt_core)`` with ``start`` 1-based inclusive
    and ``end`` 1-based exclusive. ``REF=ATTT ALT=A`` at POS=10 becomes
    ``(11, 14, "")`` -- the anchor base stays put and only the three
    trailing bases are removed.
    """
    prefix = 0
    while (
        prefix < len(ref) and prefix < len(alt)
        and ref[prefix] == alt[prefix]
    ):
        prefix += 1
    suffix = 0
    while (
        suffix < len(ref) - prefix and suffix < len(alt) - prefix
        and ref[len(ref) - 1 - suffix] == alt[len(alt) - 1 - suffix]
    ):
        suffix += 1
    start = pos + prefix
    end = pos + len(ref) - suffix
    return start, end, alt[prefix : len(alt) - suffix]


def classify_alt(
    pos: int,
    ref: str,
    alt: str,
    allele: int,
    *,
    symbolic_end: int | None = None,
) -> Edit:
    """
    Normalise one ALT allele into an :class:`Edit`.

    Raises :class:`UnsupportedVariant` for anything that cannot be
    turned into an exact reference-space edit: symbolic alleles other
    than a precise ``<DEL>``, breakends, the ``*`` spanning-deletion
    placeholder, and non-ACGT sequence.
    """
    if not alt:
        raise UnsupportedVariant("empty ALT allele")

    if alt.startswith("<") and alt.endswith(">"):
        symbol = alt[1:-1].upper()
        if symbol != "DEL":
            raise UnsupportedVariant(
                f"symbolic ALT <{symbol}> cannot be expanded to sequence"
            )
        if symbolic_end is None:
            raise UnsupportedVariant(
                "<DEL> without an INFO/END coordinate is not precise"
            )
        if symbolic_end <= pos:
            # POS is the kept base before the event, so the deleted
            # interval is POS+1 .. END; END == POS would delete nothing.
            raise UnsupportedVariant(
                f"<DEL> END={symbolic_end} must be greater than POS={pos}: "
                "the deleted bases are POS+1 through END"
            )
        # POS is the base before the event for symbolic SV records.
        return Edit(pos + 1, symbolic_end + 1, "", "symbolic-del", allele)

    if alt == "*":
        raise UnsupportedVariant(
            "'*' marks a deletion called elsewhere, not an edit of its own"
        )
    if "[" in alt or "]" in alt or alt.startswith("."):
        raise UnsupportedVariant("breakend ALT is not a local edit")

    upper_ref, upper_alt = ref.upper(), alt.upper()
    if any(b not in VALID_BASES for b in upper_ref + upper_alt):
        raise UnsupportedVariant("REF/ALT contain non-ACGT characters")

    start, end, core = _trim(pos, upper_ref, upper_alt)
    if end == start and not core:
        # REF and ALT describe the same bases. A record that changes
        # nothing has no edit to apply, and silently accepting it would
        # hide a malformed file.
        raise UnsupportedVariant(
            f"{pos}: REF and ALT are identical ({ref}); the record "
            "describes no change."
        )
    ref_len = end - start
    if ref_len == len(core):
        variant_class = "snv" if ref_len == 1 else "mnv"
    elif not core:
        variant_class = "del"
    elif ref_len == 0:
        variant_class = "ins"
    else:
        # A replacement of unequal length is a deletion and an insertion
        # at one locus; it needs whichever of the two the caller enabled.
        variant_class = "del" if len(core) < ref_len else "ins"
    return Edit(start, end, core, variant_class, allele)


@dataclass(slots=True)
class StructuralEdits:
    """
    Length-changing edits for one region, in compact arrays.

    ``record`` indexes the rows of the structural genotype matrix and
    ``allele`` says which ALT allele of that record the edit belongs to,
    so a haplotype carries edit *e* exactly when its allele at
    ``record[e]`` equals ``allele[e]``.
    """

    start: np.ndarray            # E int32, 1-based inclusive
    end: np.ndarray              # E int32, 1-based exclusive
    insert_offset: np.ndarray    # E int32, offset into insert_bases
    insert_length: np.ndarray    # E int32
    kind: np.ndarray             # E uint8
    record: np.ndarray           # E int32
    allele: np.ndarray           # E uint8
    insert_bases: np.ndarray     # P uint8, base indices 0..3
    positions: np.ndarray        # R int32, 1-based position of each record
    genotypes: np.ndarray        # R x S uint8, packed allele pairs

    @property
    def n_edits(self) -> int:
        return int(self.start.shape[0])

    @property
    def n_records(self) -> int:
        return int(self.genotypes.shape[0])

    @property
    def deleted_bases(self) -> int:
        """Total reference bases any single haplotype could lose."""
        if self.n_edits == 0:
            return 0
        removed = (self.end - self.start) - self.insert_length
        return int(np.clip(removed, 0, None).sum())

    @property
    def nbytes(self) -> int:
        return sum(
            a.nbytes for a in (
                self.start, self.end, self.insert_offset, self.insert_length,
                self.kind, self.record, self.allele, self.insert_bases,
                self.positions, self.genotypes,
            )
        )

    def insert_onehot(self, dtype: np.dtype | str) -> np.ndarray:
        """``P x 4`` one-hot pool for every inserted base in the region."""
        pool = np.zeros((self.insert_bases.shape[0], 4), dtype=dtype)
        if self.insert_bases.size:
            pool[np.arange(self.insert_bases.shape[0]), self.insert_bases] = 1
        return pool


def build_structural_edits(
    edits: list[tuple[int, Edit]],
    positions: np.ndarray,
    genotypes: np.ndarray,
) -> StructuralEdits:
    """
    Pack ``(record_index, Edit)`` pairs into :class:`StructuralEdits`.

    ``positions`` and ``genotypes`` describe the records the edits came
    from, in the same row order as ``record_index``.
    """
    n = len(edits)
    start = np.empty(n, dtype=np.int32)
    end = np.empty(n, dtype=np.int32)
    insert_offset = np.zeros(n, dtype=np.int32)
    insert_length = np.zeros(n, dtype=np.int32)
    kind = np.empty(n, dtype=np.uint8)
    record = np.empty(n, dtype=np.int32)
    allele = np.empty(n, dtype=np.uint8)

    pool: list[int] = []
    for i, (record_index, edit) in enumerate(edits):
        start[i] = edit.start
        end[i] = edit.end
        kind[i] = edit.kind
        record[i] = record_index
        allele[i] = edit.allele
        insert_offset[i] = len(pool)
        insert_length[i] = len(edit.alt)
        pool.extend(BASES.index(b) for b in edit.alt)

    return StructuralEdits(
        start=start,
        end=end,
        insert_offset=insert_offset,
        insert_length=insert_length,
        kind=kind,
        record=record,
        allele=allele,
        insert_bases=np.asarray(pool, dtype=np.uint8),
        positions=np.asarray(positions, dtype=np.int32),
        genotypes=np.asarray(genotypes, dtype=np.uint8),
    )
