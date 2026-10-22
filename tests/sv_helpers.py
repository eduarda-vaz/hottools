"""
Shared helpers for the structural-variant tests.

Scenarios are built from small, readable record constructors over a
random reference with no ambiguous bases, run through the real pipeline,
and decoded back to strings (``-`` for an all-zero row).

``oracle_window`` is a deliberately slow, independent model of what one
haplotype should look like: it edits the whole contig as text - POS
order, normalised intervals, a record overlapping an earlier one on the
same haplotype skipped - and only then cuts the output window around the
anchor. It shares no code with hottools.
"""

from __future__ import annotations

import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from conftest import _bgzip_and_index

from hottools.pipeline import RunConfig, run_hottools
from hottools.regions import Region

BASES = "ACGT"
SAMPLES = ["S1", "S2", "S3", "S4"]
_COMPLEMENT = str.maketrans("ACGT-", "TGCA-")


def random_sequence(seed: int, length: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(BASES) for _ in range(length))


def other(base: str, k: int = 0) -> str:
    """The k-th base that differs from ``base``."""
    return [b for b in BASES if b != base.upper()][k]


def revcomp(s: str) -> str:
    return s.translate(_COMPLEMENT)[::-1]


# --------------------------------------------------------------------
# records
# --------------------------------------------------------------------

@dataclass(frozen=True)
class Rec:
    pos: int
    ref: str
    alt: str
    gts: tuple[str, ...]
    info: str = "."


def snv(seq: str, pos: int, gts, k: int = 0) -> Rec:
    return Rec(pos, seq[pos - 1], other(seq[pos - 1], k), tuple(gts))


def mnv(seq: str, pos: int, n: int, gts) -> Rec:
    ref = seq[pos - 1 : pos - 1 + n]
    return Rec(pos, ref, "".join(other(b) for b in ref), tuple(gts))


def deletion(seq: str, pos: int, n: int, gts) -> Rec:
    """Delete the ``n`` bases after ``pos`` (POS is the kept anchor)."""
    return Rec(pos, seq[pos - 1 : pos + n], seq[pos - 1], tuple(gts))


def insertion(seq: str, pos: int, bases: str, gts) -> Rec:
    """Insert ``bases`` after ``pos``."""
    return Rec(pos, seq[pos - 1], seq[pos - 1] + bases, tuple(gts))


def symdel(seq: str, pos: int, end: int, gts) -> Rec:
    """Precise ``<DEL>``: removes ``pos+1 .. end``."""
    return Rec(pos, seq[pos - 1], "<DEL>", tuple(gts), f"END={end}")


# --------------------------------------------------------------------
# files and runs
# --------------------------------------------------------------------

def write_fasta(path: Path, seq: str, contig: str = "chr1") -> Path:
    lines = [seq[i : i + 60] for i in range(0, len(seq), 60)]
    path.write_text(f">{contig}\n" + "\n".join(lines) + "\n", encoding="utf-8")
    try:
        import pysam
        pysam.faidx(str(path))
    except ImportError:
        assert shutil.which("samtools"), "pysam or samtools is required"
        subprocess.run(["samtools", "faidx", str(path)], check=True)
    return path


def write_vcf(path: Path, records, samples, contig: str, length: int) -> Path:
    header = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID={contig},length={length}>",
        '##INFO=<ID=END,Number=1,Type=Integer,Description="End position">',
        '##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(samples),
    ]
    rows = [
        f"{contig}\t{r.pos}\t.\t{r.ref}\t{r.alt}\t.\tPASS\t{r.info}\tGT\t"
        + "\t".join(r.gts)
        for r in sorted(records, key=lambda r: r.pos)
    ]
    return _bgzip_and_index("\n".join(header + rows) + "\n", path)


def write_case(tmp_path: Path, seq: str, records, samples=SAMPLES,
               name: str = "case") -> tuple[Path, Path]:
    """Write a FASTA and an indexed VCF; return their paths."""
    directory = tmp_path / f"{name}_inputs"
    directory.mkdir(parents=True, exist_ok=True)
    fasta = write_fasta(directory / "ref.fa", seq)
    vcf = write_vcf(directory / "var.vcf.gz", records, samples, "chr1",
                    len(seq))
    return fasta, vcf


def run_paths(tmp_path: Path, fasta: Path, vcf: Path, *, start: int = 1,
              end: int = 200, strand: str = "+", name: str = "out",
              fmt: str = "npy", per_sample: bool = False, **kwargs):
    """Run the pipeline; return (reassembled tensor, metadata)."""
    kwargs.setdefault("mode", "haplotypes")
    kwargs.setdefault("variant_classes", "all")
    out = tmp_path / (name if per_sample else f"{name}.{fmt}")
    config = RunConfig(
        vcf_path=str(vcf), fasta_path=str(fasta),
        region=Region("chr1", start, end, strand=strand),
        output_path=str(out), output_format=fmt, per_sample=per_sample,
        overwrite="replace", **kwargs,
    )
    meta = run_hottools(config)
    return load_output(meta), meta


def run_sv(tmp_path: Path, seq: str, records, samples=SAMPLES,
           name: str = "out", **kwargs):
    fasta, vcf = write_case(tmp_path, seq, records, samples, name=name)
    return run_paths(tmp_path, fasta, vcf, name=name, **kwargs)


def _load_one(path: Path, fmt: str, per_sample: bool) -> np.ndarray:
    if fmt == "npy":
        return np.load(path)
    if fmt == "npz":
        return np.load(path)["sequences"]
    if fmt == "h5":
        import h5py
        with h5py.File(path) as f:
            return f["sequence" if per_sample else "sequences"][:]
    import torch
    return torch.load(path).numpy()


def load_output(meta: dict) -> np.ndarray | None:
    if "output_files" not in meta:
        return None
    arrays = [
        _load_one(Path(p), meta["output_format"], meta["per_sample"])
        for p in meta["output_files"]
    ]
    return np.stack(arrays) if meta["per_sample"] else np.concatenate(arrays)


def decode(track: np.ndarray) -> str:
    """``L x 4`` -> string; an all-zero row is ``-``."""
    return "".join(
        BASES[int(row.argmax())] if row.sum() > 0 else "-" for row in track
    )


def haps(tensor: np.ndarray) -> list[tuple[str, str]]:
    """Decode an ``S x 2 x L x 4`` haplotype tensor."""
    return [(decode(s[0]), decode(s[1])) for s in tensor]


# --------------------------------------------------------------------
# the oracle
# --------------------------------------------------------------------

def normalised(rec: Rec, allele: int) -> tuple[int, int, str]:
    """Half-open reference interval an ALT replaces, and its new bases."""
    alt = rec.alt.split(",")[allele - 1]
    if alt == "<DEL>":
        end = int(rec.info.split("END=")[1].split(";")[0])
        return rec.pos + 1, end + 1, ""
    ref = rec.ref
    i = 0
    while i < len(ref) and i < len(alt) and ref[i] == alt[i]:
        i += 1
    j = 0
    while (j < len(ref) - i and j < len(alt) - i
           and ref[len(ref) - 1 - j] == alt[len(alt) - 1 - j]):
        j += 1
    return rec.pos + i, rec.pos + len(ref) - j, alt[i : len(alt) - j]


def edited_contig(seq: str, records, alleles) -> list[tuple[int | None, str]]:
    """
    The whole contig after one haplotype's edits, as (source, base) pairs.

    ``source`` is the 1-based reference position a base came from, or
    ``None`` for an inserted base. Records apply in POS order; one whose
    interval starts inside an earlier applied record's is skipped.
    """
    chosen = []
    last_end = -1
    for rec, allele in sorted(zip(records, alleles, strict=True),
                              key=lambda t: t[0].pos):
        if allele == 0:
            continue
        start, end, new = normalised(rec, allele)
        if start < last_end:
            continue
        chosen.append((start, end, new))
        last_end = end
    out: list[tuple[int | None, str]] = []
    cursor = 1
    for start, end, new in chosen:
        out.extend((p, seq[p - 1]) for p in range(cursor, start))
        if end - start == len(new):
            # A substitution changes bases, not coordinates: each new
            # base still stands at its reference position.
            out.extend((start + k, b) for k, b in enumerate(new))
        else:
            out.extend((None, b) for b in new)
        cursor = max(cursor, end)
    out.extend((p, seq[p - 1]) for p in range(cursor, len(seq) + 1))
    return out


def oracle_window(seq: str, records, alleles, *, anchor: int, offset: int,
                  length: int) -> str:
    """
    The output window: the first surviving base at or after ``anchor``
    lands at ``offset``; positions with no base are ``-``.
    """
    entries = edited_contig(seq, records, alleles)
    index = next(
        (i for i, (p, _) in enumerate(entries) if p is not None and p >= anchor),
        len(entries),
    )
    lo = index - offset
    return "".join(
        entries[i][1] if 0 <= i < len(entries) else "-"
        for i in range(lo, lo + length)
    )


def allele_of(gt: str, hap: int) -> int:
    """Allele index of haplotype ``hap`` (0 or 1) in a phased call."""
    return int(gt.replace("/", "|").split("|")[hap])
