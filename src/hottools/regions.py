from __future__ import annotations

import gzip
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

from .config_validation import positive_integer
from .contigs import alias_key, resolve_contig, validate_alias_policy

VALID_STRANDS = frozenset({"+", "-"})

CoordinateSystem = Literal["auto", "1based", "bed"]

#: Region-file suffixes that imply 0-based half-open (BED) coordinates.
BED_SUFFIXES = (".bed", ".bed.gz", ".bed.bgz")

#: Suffixes opened through gzip. bgzip output is gzip-compatible, so the
#: same reader handles .bed.bgz.
GZIP_SUFFIXES = (".gz", ".bgz")


def _open_text(path: str | Path):
    """Open a region/mask file, transparently handling gzip and bgzip."""
    if str(path).lower().endswith(GZIP_SUFFIXES):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


@dataclass(frozen=True, slots=True)
class Region:
    """A 1-based, fully-closed genomic interval with an explicit strand."""

    chrom: str
    start: int
    end: int
    strand: str = "+"

    def __post_init__(self) -> None:
        if not isinstance(self.chrom, str) or not self.chrom.strip():
            raise ValueError("Region chrom must be a non-empty string")
        positive_integer("Region start", self.start)
        positive_integer("Region end", self.end)
        if self.start < 1:
            raise ValueError(f"Region start must be >= 1, got {self.start}")
        if self.end < self.start:
            raise ValueError(
                f"Region end must be >= start, got {self.chrom}:{self.start}-{self.end}"
            )
        if self.strand not in VALID_STRANDS:
            raise ValueError(f"Region strand must be '+' or '-', got {self.strand!r}")

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def label(self) -> str:
        return f"{self.chrom}:{self.start}-{self.end}"

    @property
    def reverse_complement(self) -> bool:
        """Single source of truth for output orientation."""
        return self.strand == "-"

    def with_strand(self, strand: str) -> Region:
        return replace(self, strand=strand)


def parse_region(text: str, strand: str = "+") -> Region:
    """Parse ``CHROM:START-END`` as a 1-based, fully-closed interval."""
    try:
        chrom, coords = text.rsplit(":", 1)
        start_s, end_s = coords.replace(",", "").split("-", 1)
        start = int(start_s)
        end = int(end_s)
    except Exception as exc:
        raise ValueError(f"Invalid region {text!r}; expected CHROM:START-END") from exc

    return Region(chrom, start, end, strand)


def region_from_center(
    center: str,
    length: int,
    strand: str = "+",
) -> Region:
    """Build a ``length`` bp window centred on ``CHROM:POS``."""
    if length < 1:
        raise ValueError("length must be >= 1")
    try:
        chrom, pos_s = center.rsplit(":", 1)
        center_pos = int(pos_s.replace(",", ""))
    except Exception as exc:
        raise ValueError(f"Invalid center {center!r}; expected CHROM:POS") from exc

    left = (length - 1) // 2
    start = center_pos - left
    end = start + length - 1
    if start < 1:
        raise ValueError(
            f"Centered interval extends before coordinate 1: {chrom}:{start}-{end}"
        )
    return Region(chrom, start, end, strand)


def resolve_coordinate_system(
    path: str | Path,
    coordinate_system: CoordinateSystem = "auto",
) -> str:
    """Resolve ``auto`` to ``bed`` or ``1based`` from the file suffix."""
    if coordinate_system not in {"auto", "1based", "bed"}:
        raise ValueError("coordinate_system must be 'auto', '1based' or 'bed'")
    if coordinate_system != "auto":
        return coordinate_system
    name = str(path).lower()
    return "bed" if name.endswith(BED_SUFFIXES) else "1based"


def read_mask(
    path: str | Path,
    region: Region,
    coordinate_system: CoordinateSystem = "auto",
    *,
    contig_alias_policy: str = "exact",
) -> np.ndarray:
    """
    Read a mask file into a boolean array over ``region``.

    ``True`` marks a position to blank out: its reference channels are
    zeroed and any variant there is dropped. The file uses the same
    layouts as :func:`read_region_file`, so a plain BED of unreliable
    intervals works directly.
    """
    import numpy as np

    validate_alias_policy(contig_alias_policy)
    mask = np.zeros(region.length, dtype=bool)
    intervals = read_region_file(path, coordinate_system=coordinate_system)
    chrom = region.chrom
    if contig_alias_policy == "auto":
        names = {interval.chrom for interval in intervals}
        if not any(alias_key(name) == alias_key(chrom) for name in names):
            return mask
        chrom = resolve_contig(chrom, names, source="mask")
    for interval in intervals:
        if interval.chrom != chrom:
            continue
        lo = max(interval.start, region.start) - region.start
        hi = min(interval.end, region.end) - region.start
        if hi >= lo:
            mask[lo : hi + 1] = True
    return mask


def read_region_file(
    path: str | Path,
    default_strand: str = "+",
    coordinate_system: CoordinateSystem = "auto",
) -> list[Region]:
    """
    Read regions from a text file.

    Two layouts are supported, selected by ``coordinate_system``:

    ``1based`` (default for any non-BED suffix)
        ``CHROM:START-END`` on its own, or whitespace-separated
        ``CHROM START END [STRAND]``. Coordinates are 1-based and
        fully closed, matching ``--region``.

    ``bed`` (default for ``.bed`` / ``.bed.gz`` / ``.bed.bgz``)
        Standard BED: ``CHROM START END [NAME] [SCORE] [STRAND]`` with
        0-based half-open coordinates, converted on read.

    ``auto`` picks ``bed`` for BED suffixes and ``1based`` otherwise.
    """
    if default_strand not in VALID_STRANDS:
        raise ValueError(f"default_strand must be '+' or '-', got {default_strand!r}")
    system = resolve_coordinate_system(path, coordinate_system)
    strand_column = 5 if system == "bed" else 3

    regions: list[Region] = []
    with _open_text(path) as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text or text.startswith(("#", "track", "browser")):
                continue
            fields = text.split()
            try:
                if len(fields) == 1:
                    if system == "bed":
                        raise ValueError("BED lines need at least 3 columns")
                    regions.append(parse_region(fields[0], default_strand))
                    continue
                if len(fields) < 3:
                    raise ValueError("expected CHROM START END, or CHROM:START-END")
                chrom, start_s, end_s = fields[:3]
                start, end = int(start_s), int(end_s)
                if system == "bed":
                    if end <= start:
                        raise ValueError(
                            "BED end must be greater than start (0-based half-open)"
                        )
                    start += 1  # half-open 0-based -> closed 1-based
                strand = (
                    fields[strand_column]
                    if len(fields) > strand_column
                    else default_strand
                )
                regions.append(Region(chrom, start, end, strand))
            except Exception as exc:
                raise ValueError(
                    f"Invalid region file line {line_no} ({system} coordinates): {text}"
                ) from exc
    if not regions:
        raise ValueError(f"No regions found in {path}")
    return regions
