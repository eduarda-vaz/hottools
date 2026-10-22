from __future__ import annotations

from pathlib import Path

import numpy as np
from pyfaidx import Fasta

from .constants import BASE_TO_INDEX


def _contig_hint(chrom: str, available: list[str], fasta_path) -> str:
    """Build an actionable message for a contig that is not in the FASTA."""
    shown = ", ".join(available[:8])
    if len(available) > 8:
        shown += f", ... ({len(available)} total)"
    hint = ""
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    if alt in available:
        hint = (
            f" Did you mean {alt!r}? The FASTA uses the "
            f"{'un' if chrom.startswith('chr') else ''}prefixed naming "
            "convention."
        )
    return (
        f"Contig {chrom!r} not found in {fasta_path}. "
        f"Available: {shown}.{hint}"
    )


class ReferenceReader:
    """
    Keep one ``pyfaidx.Fasta`` handle open across many interval reads.

    Opening a ``Fasta`` re-reads the ``.fai``; doing that once per region
    is wasteful for region files with many entries.
    """

    def __init__(self, fasta_path: str | Path) -> None:
        self.fasta_path = str(fasta_path)
        self._fasta = Fasta(
            self.fasta_path,
            as_raw=True,
            sequence_always_upper=True,
        )

    @property
    def contigs(self) -> list[str]:
        return list(self._fasta.keys())

    def length(self, chrom: str) -> int:
        """Length of ``chrom`` in the FASTA index."""
        try:
            return len(self._fasta[chrom])
        except KeyError as exc:
            raise ValueError(
                _contig_hint(chrom, self.contigs, self.fasta_path)
            ) from exc

    def interval(self, chrom: str, start: int, end: int) -> str:
        """Read a 1-based, fully-closed interval as an uppercase string."""
        if start < 1:
            raise ValueError("start must be >= 1")
        if end < start:
            raise ValueError("end must be >= start")

        try:
            record = self._fasta[chrom]
        except KeyError as exc:
            raise ValueError(
                _contig_hint(chrom, self.contigs, self.fasta_path)
            ) from exc

        seq = record[start - 1 : end]
        expected = end - start + 1
        if len(seq) != expected:
            raise ValueError(
                f"FASTA returned {len(seq)} bp for {chrom}:{start}-{end}; "
                f"expected {expected} bp. Contig {chrom!r} is "
                f"{len(record)} bp long."
            )
        return str(seq).upper()

    def close(self) -> None:
        self._fasta.close()

    def __enter__(self) -> ReferenceReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_reference_interval(
    fasta_path: str | Path,
    chrom: str,
    start: int,
    end: int,
) -> str:
    """Read a 1-based inclusive FASTA interval (opens and closes a handle)."""
    with ReferenceReader(fasta_path) as reader:
        return reader.interval(chrom, start, end)


def one_hot_reference(
    sequence: str,
    dtype: np.dtype | str = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode A/C/G/T as one-hot and ambiguous reference bases as zeros.

    Returns
    -------
    onehot
        L x 4
    base_indices
        L uint8 array. A/C/G/T -> 0/1/2/3, ambiguous -> 255.
    """
    try:
        seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    except UnicodeEncodeError as exc:
        raise ValueError("Reference sequence contains non-ASCII characters.") from exc

    base_indices = BASE_TO_INDEX[seq_bytes]
    onehot = np.zeros((len(sequence), 4), dtype=dtype)

    # Boolean row selection avoids materialising an L-sized index array.
    valid = base_indices < 4
    onehot[valid, base_indices[valid]] = 1
    return onehot, base_indices
