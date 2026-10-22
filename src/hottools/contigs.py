"""Conservative, opt-in contig aliases; no coordinate or assembly conversion."""
from __future__ import annotations

from collections.abc import Iterable

from cyvcf2 import VCF

from .constants import CONTIG_ALIAS_POLICIES, check_choice


def alias_key(name: str) -> str:
    """Group standard numeric, X/Y and mitochondrial names, case-sensitively."""
    bare = name[3:] if name.startswith("chr") else name
    if bare in {"M", "MT"}:
        return "mitochondrial"
    if bare in {"X", "Y"} or (bare.isascii() and bare.isdecimal()
                               and str(int(bare)) == bare and int(bare) > 0):
        return "standard:" + bare
    return "literal:" + name


def resolve_contig(requested: str, available: Iterable[str], *, source: str) -> str:
    names = set(available)
    if requested in names:
        return requested
    candidates = sorted(
        name for name in names if alias_key(name) == alias_key(requested)
    )
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError(
            f"Ambiguous contig alias for {requested!r} in {source}: {candidates}. "
            "Use an exact name or rename contigs explicitly."
        )
    raise ValueError(
        f"No contig matching {requested!r} in {source} (including aliases)"
    )


def resolve_input_contigs(vcf_path, reader, requested: str) -> tuple[str, str]:
    """Resolve each file independently and reject known length disagreements."""
    fasta_chrom = resolve_contig(requested, reader.contigs, source="FASTA")
    handle = VCF(str(vcf_path), lazy=True)
    try:
        names = list(handle.seqnames)
        vcf_chrom = resolve_contig(requested, names, source="VCF/BCF")
        try:
            length = handle.seqlens[names.index(vcf_chrom)]
        except (AttributeError, ValueError, IndexError, TypeError):
            length = None
        fasta_length = reader.length(fasta_chrom)
        if length is not None and length > 0 and length != fasta_length:
            raise ValueError(
                f"Contig length mismatch: VCF {vcf_chrom!r} has {length} bases, "
                f"FASTA {fasta_chrom!r} has {fasta_length}. "
                "Aliases do not convert genome assemblies."
            )
    finally:
        handle.close()
    return vcf_chrom, fasta_chrom


def validate_alias_policy(value: str) -> None:
    check_choice("contig_alias_policy", value, CONTIG_ALIAS_POLICIES)
