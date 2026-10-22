from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from cyvcf2 import VCF

from .config_validation import (
    boolean,
    validate_random_assignment,
    validate_site_options,
)
from .constants import (
    AF_SOURCES,
    ALLELE_ABSENT,
    ALLELE_MISSING,
    BASES,
    CODE_ALLELE1,
    CODE_ALLELE2,
    DIPLOID_GENOTYPE_WIDTH,
    GENOTYPE_UNKNOWN,
    HAPLOID_GENOTYPE_WIDTH,
    HAPLOID_POLICIES,
    MAX_ALT_ALLELES,
    MISSING_POLICIES,
    MIXED_RECORD_POLICIES,
    MODES,
    N_ALLELE_SLOTS,
    N_ALLELES,
    NO_ALLELE,
    OVERLAP_POLICIES,
    PLOIDY_POLICIES,
    RECORD_POLICIES,
    UNPHASED_POLICIES,
    VALID_BASES,
    AfSource,
    HaploidPolicy,
    MissingPolicy,
    Mode,
    OverlapPolicy,
    PloidyPolicy,
    RecordPolicy,
    UnphasedPolicy,
    check_choice,
    pack_genotype,
)
from .edits import (
    LENGTH_CHANGING,
    Edit,
    StructuralEdits,
    UnsupportedVariant,
    build_structural_edits,
    classify_alt,
)

#: Re-exported from :mod:`hottools.constants`, the single definition.
#: ``duplicate``/``blank`` haploid policies decide what the *second
#: haplotype track* carries for a hemizygous sample; ``error``/``skip``
#: reject the record outright. Dosage and consensus are ploidy-aware and
#: unaffected by the first two.
Policy = RecordPolicy
AFSource = AfSource


class EmptyRegionWarning(RuntimeWarning):
    """No usable variants were found in the requested interval."""


class MultiallelicWarning(RuntimeWarning):
    """A retained SNV shares its position with a discarded record."""


class HaploidWarning(RuntimeWarning):
    """Haploid genotypes were encountered and padded to two tracks."""


class StructuralEditWarning(RuntimeWarning):
    """A length-changing edit could not be applied as written."""


@dataclass(slots=True)
class VariantStats:
    records_seen: int = 0
    filtered_not_pass: int = 0
    filtered_low_qual: int = 0
    filtered_allele_frequency: int = 0
    masked_out: int = 0
    non_snv_records: int = 0
    multiallelic_records: int = 0
    too_many_alt_alleles: int = 0
    filtered_multiallelic: int = 0
    lowercase_alleles: int = 0
    outside_region: int = 0
    ambiguous_fasta: int = 0
    ref_mismatch: int = 0
    haploid_records: int = 0
    haploid_genotypes: int = 0
    polyploid_records: int = 0
    duplicate_records: int = 0
    duplicate_positions_removed: int = 0
    duplicate_positions_resolved: int = 0
    clipped_at_edge: int = 0
    overlapping_records_suppressed: int = 0
    overlapping_blocks_dropped: int = 0
    multiallelic_sites_partial: int = 0
    mnv_records: int = 0
    deletion_edits: int = 0
    insertion_edits: int = 0
    symbolic_deletion_edits: int = 0
    structural_records: int = 0
    unsupported_alt_alleles: int = 0
    class_not_requested: int = 0
    records_genotyped: int = 0
    missing_genotypes: int = 0
    half_calls: int = 0
    unphased_heterozygotes: int = 0
    randomly_assigned_genotypes: int = 0
    kept: int = 0

    def as_dict(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(slots=True)
class VariantData:
    mode: Mode
    samples: list[str]
    positions: np.ndarray  # V, 1-based genomic positions
    allele_bases: np.ndarray  # V x 4 uint8, allele index -> base index
    n_alt: np.ndarray  # V uint8, number of ALT alleles
    genotypes: np.ndarray  # V x S uint8, packed allele pairs
    sample_ploidy: np.ndarray  # S uint8, largest ploidy seen (0 if absent)
    stats: VariantStats
    #: Length-changing edits, when any variant class that produces them
    #: was requested. ``None`` on the SNV-only default path.
    structural: StructuralEdits | None = None
    #: Backing store for :attr:`allele_freq`. Computing it costs a few
    #: passes over the V x S matrix and only the allele-frequency filter
    #: and ``het_policy='major'`` need it, so it is filled on demand.
    _allele_freq: np.ndarray | None = None
    random_assignments: list[dict] = field(default_factory=list)

    @property
    def allele_freq(self) -> np.ndarray:
        """V x 4 frequency of each allele among the called alleles."""
        if self._allele_freq is None:
            self._allele_freq = _allele_frequencies(self.genotypes)
        return self._allele_freq

    @property
    def ref_idx(self) -> np.ndarray:
        """Base index of the REF allele, for convenience."""
        return self.allele_bases[:, 0]

    @property
    def payload(self) -> np.ndarray:
        """The compact genotype array; one byte per sample per variant."""
        return self.genotypes


def _base_index(base: str) -> int:
    return BASES.index(base)


def _validate_policies(
    mode: str,
    missing_policy: str,
    unphased_policy: str,
    ref_mismatch_policy: str,
    ambiguous_ref_policy: str,
    haploid_policy: str,
    ploidy_policy: str,
    af_source: str,
) -> None:
    """Reject bad policy strings up front rather than mid-iteration."""
    for name, value, allowed in (
        ("mode", mode, MODES),
        ("missing_policy", missing_policy, MISSING_POLICIES),
        ("unphased_policy", unphased_policy, UNPHASED_POLICIES),
        ("ref_mismatch_policy", ref_mismatch_policy, RECORD_POLICIES),
        ("ambiguous_ref_policy", ambiguous_ref_policy, RECORD_POLICIES),
        ("haploid_policy", haploid_policy, HAPLOID_POLICIES),
        ("ploidy_policy", ploidy_policy, PLOIDY_POLICIES),
        ("af_source", af_source, AF_SOURCES),
    ):
        check_choice(name, value, allowed)


def resolve_sample_selection(
    vcf_samples: Sequence[str],
    requested: Sequence[str] | None,
) -> tuple[list[str], list[int] | None]:
    """
    Validate a sample request and return ``(names, column_order)``.

    cyvcf2 silently ignores sample IDs that are absent from the VCF and
    always returns samples in file order. Both are corrected here: absent
    IDs raise, and ``column_order`` permutes the genotype columns back
    into the order the caller asked for.
    """
    vcf_samples = list(vcf_samples)
    if requested is None:
        return vcf_samples, None

    requested = list(requested)
    if not requested:
        raise ValueError("Sample selection is empty.")

    duplicates = sorted(name for name, n in Counter(requested).items() if n > 1)
    if duplicates:
        raise ValueError(f"Duplicate sample ID(s) requested: {', '.join(duplicates)}")

    available = set(vcf_samples)
    missing = [s for s in requested if s not in available]
    if missing:
        shown = ", ".join(missing[:10])
        if len(missing) > 10:
            shown += f", ... ({len(missing)} total)"
        raise ValueError(
            f"Requested sample(s) not present in the VCF: {shown}. "
            f"The VCF contains {len(vcf_samples)} sample(s)."
        )

    index = {name: i for i, name in enumerate(vcf_samples)}
    return requested, [index[name] for name in requested]


def _check_region_contig(vcf: VCF, chrom: str) -> None:
    """Fail loudly on a contig the VCF index does not know about."""
    try:
        contigs = list(vcf.seqnames)
    except Exception:  # pragma: no cover - index without a contig list
        return
    if not contigs or chrom in contigs:
        return
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    hint = ""
    if alt in contigs:
        hint = (
            f" Did you mean {alt!r}? The VCF and the region use different "
            "chromosome-naming conventions."
        )
    shown = ", ".join(contigs[:8])
    if len(contigs) > 8:
        shown += f", ... ({len(contigs)} total)"
    raise ValueError(
        f"Contig {chrom!r} is not present in the VCF. Available: {shown}.{hint}"
    )


def _record_allele_frequency(var) -> float | None:
    """Largest ALT allele frequency from INFO/AF, if the record has one."""
    try:
        value = var.INFO.get("AF")
    except KeyError:
        return None
    if value is None:
        return None
    raw_values = value if isinstance(value, (tuple, list, np.ndarray)) else [value]
    values = []
    for entry in raw_values:
        if entry is None:
            continue
        frequency = float(entry)
        if np.isnan(frequency):
            # htslib can represent a missing float as NaN; treat it like '.'.
            continue
        if not np.isfinite(frequency) or not 0 <= frequency <= 1:
            raise ValueError(f"{var.CHROM}:{var.POS}: INFO/AF must be between 0 and 1")
        values.append(frequency)
    return max(values) if values else None


def _decode_genotypes(
    var,
    vcf_sample_names: Sequence[str],
    n_alt: int,
    haploid_policy: HaploidPolicy,
    ploidy_policy: PloidyPolicy,
    stats: VariantStats,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Decode one record into packed genotype codes.

    Returns ``(codes, ploidy, phased)`` or ``None`` when the record is
    skipped by a ploidy policy. ``codes`` is S uint8, ``ploidy`` is
    S uint8 and ``phased`` is S bool.
    """
    try:
        raw = var.genotypes
    except Exception as exc:
        # cyvcf2 raises a generic Exception for absent/malformed GT. Reading
        # FORMAT separately is unsafe with lazy records in some versions.
        raise ValueError(
            f"{var.CHROM}:{var.POS}: FORMAT/GT is missing or cannot be decoded"
        ) from exc
    n_samples = len(vcf_sample_names)

    try:
        grid = np.asarray(raw)
        uniform = grid.ndim == 2
    except ValueError:
        grid = None
        uniform = False

    if uniform and grid.shape[1] == DIPLOID_GENOTYPE_WIDTH:
        allele1 = grid[:, 0].astype(np.int16, copy=False)
        allele2 = grid[:, 1].astype(np.int16, copy=False)
        phased = grid[:, 2].astype(bool, copy=False)
        ploidy = np.full(n_samples, 2, dtype=np.uint8)
    elif uniform and grid.shape[1] == HAPLOID_GENOTYPE_WIDTH:
        allele1 = grid[:, 0].astype(np.int16, copy=False)
        allele2 = np.full(n_samples, -2, dtype=np.int16)  # -2 = absent
        phased = grid[:, 1].astype(bool, copy=False)
        ploidy = np.full(n_samples, 1, dtype=np.uint8)
    else:
        # Mixed or higher ploidy: fall back to a per-sample walk. This is
        # the slow path and only runs for records that need it.
        allele1 = np.empty(n_samples, dtype=np.int16)
        allele2 = np.empty(n_samples, dtype=np.int16)
        phased = np.zeros(n_samples, dtype=bool)
        ploidy = np.empty(n_samples, dtype=np.uint8)
        polyploid: list[str] = []
        for i, entry in enumerate(raw):
            alleles = list(entry[:-1])
            phased[i] = bool(entry[-1])
            ploidy[i] = len(alleles)
            if len(alleles) == 1:
                allele1[i], allele2[i] = alleles[0], -2
            elif len(alleles) == 2:
                allele1[i], allele2[i] = alleles[0], alleles[1]
            else:
                polyploid.append(
                    vcf_sample_names[i] if i < len(vcf_sample_names) else f"sample{i}"
                )
                allele1[i], allele2[i] = alleles[0], alleles[1]
        if polyploid:
            stats.polyploid_records += 1
            shown = ", ".join(polyploid[:5])
            if len(polyploid) > 5:
                shown += f", ... ({len(polyploid)} total)"
            msg = (
                f"{var.CHROM}:{var.POS}: genotype(s) with ploidy > 2 for "
                f"{shown}. hottools encodes at most two alleles per "
                "sample; use --ploidy-policy skip to drop such records."
            )
            if ploidy_policy == "error":
                raise ValueError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            return None

    n_haploid = int(np.count_nonzero(ploidy == 1))
    if n_haploid:
        stats.haploid_records += 1
        stats.haploid_genotypes += n_haploid
        if haploid_policy == "error":
            offenders = [
                vcf_sample_names[i]
                for i in np.flatnonzero(ploidy == 1)[:5]
                if i < len(vcf_sample_names)
            ]
            shown = ", ".join(offenders)
            if n_haploid > 5:
                shown += f", ... ({n_haploid} total)"
            raise ValueError(
                f"{var.CHROM}:{var.POS}: haploid genotype(s) for {shown}. "
                "Use --haploid-policy duplicate (default) or blank, or "
                "--mode dosage."
            )
        if haploid_policy == "skip":
            return None

    # A GT entry may only name an allele the record declares. This has
    # to be checked against n_alt rather than against the slot encoding:
    # allele index 4 would otherwise land on ALLELE_MISSING and be read
    # as "uncalled" instead of being rejected. Negative values are the
    # missing (-1) and absent (-2) sentinels and are never too large.
    too_large = int(np.max(allele1, initial=0)), int(np.max(allele2, initial=0))
    if max(too_large) > n_alt:
        raise ValueError(
            f"{var.CHROM}:{var.POS}: genotype names allele index "
            f"{max(too_large)} but the record declares only {n_alt} ALT "
            f"allele(s). hottools supports at most {MAX_ALT_ALLELES}."
        )

    slot1 = np.where(allele1 >= 0, allele1, int(ALLELE_MISSING)).astype(np.uint8)
    slot2 = np.where(allele2 >= 0, allele2, int(ALLELE_MISSING)).astype(np.uint8)
    slot2[allele2 == -2] = ALLELE_ABSENT

    codes = (slot1 * np.uint8(N_ALLELE_SLOTS) + slot2).astype(np.uint8)
    return codes, ploidy, phased


def _project_excluded_alleles(codes: np.ndarray, excluded: list[int]) -> np.ndarray:
    """Map unrequested ALT calls to REF before phase, frequency and conflicts."""
    if not excluded:
        return codes
    a1 = CODE_ALLELE1[codes].copy()
    a2 = CODE_ALLELE2[codes].copy()
    a1[np.isin(a1, excluded)] = 0
    a2[np.isin(a2, excluded)] = 0
    return pack_genotype(a1, a2)


def _apply_missing_and_phase(
    var,
    codes: np.ndarray,
    phased: np.ndarray,
    missing_policy: MissingPolicy,
    unphased_policy: UnphasedPolicy,
    stats: VariantStats,
    check_phase: bool = True,
    random_seed: int | None = None,
    sample_names: Sequence[str] = (),
    assignments: list[dict] | None = None,
    original_codes: np.ndarray | None = None,
) -> np.ndarray:
    """
    Count and resolve missing alleles and unphased heterozygotes.

    ``check_phase`` is False for the dosage and consensus modes, which
    are phase-agnostic: an unphased heterozygote is perfectly usable
    there, so it is neither counted nor rewritten.
    """
    a1 = CODE_ALLELE1[codes]
    a2 = CODE_ALLELE2[codes]

    known1 = a1 < N_ALLELES
    known2 = a2 < N_ALLELES
    diploid = a2 != ALLELE_ABSENT

    # A genotype is missing when none of the alleles it *has* were
    # called -- which includes a hemizygous '.' on chrY, where there is
    # only one allele to miss. It is a half-call when exactly one of two
    # existing alleles was called.
    n_called = known1.astype(np.int8) + known2.astype(np.int8)
    n_missing = int(np.count_nonzero(n_called == 0))
    n_half = int(np.count_nonzero((n_called == 1) & diploid))
    stats.missing_genotypes += n_missing
    stats.half_calls += n_half

    if (n_missing or n_half) and missing_policy == "error":
        raise ValueError(
            f"{var.CHROM}:{var.POS}: {n_missing + n_half} genotype(s) "
            "with a missing allele."
        )

    if not check_phase:
        return codes

    heterozygous = known1 & known2 & (a1 != a2)
    unphased = heterozygous & (~phased)
    n_unphased = int(unphased.sum())
    stats.unphased_heterozygotes += n_unphased

    if n_unphased:
        if unphased_policy == "error":
            raise ValueError(
                f"{var.CHROM}:{var.POS}: {n_unphased} unphased "
                "heterozygous genotype(s). Haplotype mode requires phase."
            )
        if unphased_policy == "random":
            codes = codes.copy()
            for index in np.flatnonzero(unphased):
                first, second = int(a1[index]), int(a2[index])
                low, high = sorted((first, second))
                sample = sample_names[index]
                # A stable per-call coin flip: independent of batches, threads,
                # sample order, other records and Python's salted hash().
                key = json.dumps([
                    int(random_seed), str(var.CHROM), int(var.POS),
                    var.REF.upper(), [alt.upper() for alt in var.ALT],
                    sample, low, high,
                ], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                digest = hashlib.blake2b(
                    key, digest_size=8, person=b"ht-phase-v1"
                ).digest()
                flip = digest[0] & 1
                pair = (high, low) if flip else (low, high)
                codes[index] = pack_genotype(*pair)
                original = (original_codes[index] if original_codes is not None
                            else pack_genotype(first, second))
                original_first = int(CODE_ALLELE1[original])
                original_second = int(CODE_ALLELE2[original])
                if assignments is not None:
                    assignments.append({
                        "chrom": str(var.CHROM), "pos": int(var.POS),
                        "ref": var.REF, "alts": list(var.ALT), "sample": sample,
                        "input_gt": f"{original_first}/{original_second}",
                        "projected_gt": f"{first}/{second}",
                        "assigned_gt": f"{pair[0]}|{pair[1]}",
                    })
            return codes
        if missing_policy == "error":
            raise ValueError(
                f"{var.CHROM}:{var.POS}: unphased heterozygotes cannot "
                "be represented with missing_policy='error'."
            )
        codes = codes.copy()
        codes[unphased] = GENOTYPE_UNKNOWN
    return codes


#: Copies of each allele index (0-3) carried by each packed code.
_CODE_ALLELE_COUNTS = np.stack(
    [
        (CODE_ALLELE1 == allele).astype(np.int64)
        + (CODE_ALLELE2 == allele).astype(np.int64)
        for allele in range(N_ALLELES)
    ],
    axis=1,
)


def _record_alt_frequency(codes: np.ndarray) -> float:
    """
    Highest ALT frequency among the called alleles of one record.

    Missing and absent allele slots are not counted, so a hemizygous
    call contributes one allele and an uncalled one contributes none.
    """
    counts = _CODE_ALLELE_COUNTS[codes].sum(axis=0)
    total = int(counts.sum())
    if total == 0:
        return 0.0
    return float(counts[1:].max()) / total


def _touches_mask(mask: np.ndarray, changed, offset: int) -> bool:
    """
    Whether a mask covers a reference base a record changes.

    ``changed`` holds 1-based half-open intervals; ``offset`` is the
    1-based coordinate of ``mask[0]``. An insertion changes no base, so
    it counts as masked when either base it is inserted between is
    masked. A deletion's VCF anchor base is quoted in REF but left
    unchanged, so masking only the anchor does not drop the deletion.
    """
    n = mask.shape[0]
    for begin, finish in changed:
        lo, hi = begin - offset, finish - offset
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        lo, hi = max(0, lo), min(n, hi)
        if hi > lo and mask[lo:hi].any():
            return True
    return False


def _allele_frequencies(codes: np.ndarray) -> np.ndarray:
    """Per-variant frequency of each allele among the called alleles."""
    n_variants = codes.shape[0]
    freq = np.zeros((n_variants, N_ALLELES), dtype=np.float32)
    if n_variants == 0:
        return freq
    a1 = CODE_ALLELE1[codes]
    a2 = CODE_ALLELE2[codes]
    for allele in range(N_ALLELES):
        freq[:, allele] = (a1 == allele).sum(axis=1) + (a2 == allele).sum(axis=1)
    total = freq.sum(axis=1, keepdims=True)
    np.divide(freq, total, out=freq, where=total > 0)
    return freq


def _classify_record(
    pos: int,
    ref: str,
    alts: list[str],
    var,
    variant_classes: frozenset[str],
    stats: VariantStats,
    mixed_record_policy: str = "exclude",
) -> (
    tuple[
        list[tuple[int, np.ndarray]],
        list[Edit],
        bool,
        int,
        int,
        list[tuple[int, int]],
        list[int],
    ]
    | None
):
    """
    Split one record's ALT alleles into fixed-length and structural work.

    Returns ``(allele_tables, structural_edits, has_substitution,
    claim_start, claim_end, changed, excluded_alleles)``, or ``None`` when the record
    contributes nothing. ``changed`` lists the normalised half-open
    interval of every usable allele - the reference bases it actually
    changes, which is what a mask is tested against.

    ``allele_tables`` is a list of ``(reference_position, table)``
    pairs: one ``N_ALLELES`` row of base indices per position the record
    substitutes. A plain SNV yields one pair; an MNV is a coordinated
    block of substitutions and yields one pair per base of the block, so
    the existing vectorized substitution stage applies the whole block
    with no new machinery. A length-changing allele is entered as the
    REF base, because the substitution stage must leave those positions
    alone and let the transform stage make the edit.

    Everything is expressed in *trimmed* coordinates - the interval each
    ALT actually changes, with shared flanking bases removed. That is
    the interval that decides overlap, and it is why two deletions
    sharing only their anchor base are not in conflict.
    """
    symbolic_end = None
    if any(a.startswith("<") for a in alts):
        try:
            value = var.INFO.get("END")
        except KeyError:
            value = None
        symbolic_end = None if value is None else int(value)

    if any(b not in VALID_BASES for b in ref):
        stats.non_snv_records += 1
        return None

    # SNV-only allele projection still requires a literal single-base REF;
    # normalizing a multi-base representation must not broaden this policy.
    if variant_classes == frozenset({"snv"}) and len(ref) != 1:
        stats.non_snv_records += 1
        return None

    classified: list[Edit] = []
    for index, alt in enumerate(alts, start=1):
        try:
            classified.append(
                classify_alt(pos, ref, alt, index, symbolic_end=symbolic_end)
            )
        except UnsupportedVariant:
            stats.unsupported_alt_alleles += 1
            if mixed_record_policy != "apply-selected":
                return None
            # Explicit allele projection: preserve its GT slot as REF,
            # while supported requested alleles can still be applied.
            classified.append(Edit(pos, pos, "", "unsupported", index))

    # Whole-record selection: every ALT must belong to a requested class.
    # SNV-only additionally requires literal single-base REF and ALT fields.
    excluded_class = any(
        edit.variant_class not in variant_classes for edit in classified
    )
    nonliteral_snv = variant_classes == frozenset({"snv"}) and (
        len(ref) != 1 or any(len(alt) != 1 for alt in alts)
    )
    if mixed_record_policy == "exclude" and (excluded_class or nonliteral_snv):
        stats.class_not_requested += sum(
            edit.variant_class not in variant_classes for edit in classified
        )
        stats.non_snv_records += 1
        return None

    usable = [edit for edit in classified if edit.variant_class in variant_classes]
    stats.class_not_requested += len(classified) - len(usable)
    if not usable:
        stats.non_snv_records += 1
        return None

    structural = [edit for edit in usable if edit.is_length_changing]
    substitutions = [edit for edit in usable if not edit.is_length_changing]
    if any(edit.ref_length > 1 for edit in substitutions):
        stats.mnv_records += 1

    claim_start = min(edit.start for edit in usable)
    claim_end = max(edit.end for edit in usable)

    tables: list[tuple[int, np.ndarray]] = []
    if substitutions:
        block_start = min(edit.start for edit in substitutions)
        block_end = max(edit.end for edit in substitutions)
        for position in range(block_start, block_end):
            table = np.full(N_ALLELES, NO_ALLELE, dtype=np.uint8)
            table[0] = _base_index(ref[position - pos])
            for edit in classified:
                # Any allele that does not change this position - not
                # requested, length-changing, or a substitution block
                # that does not reach here - carries the reference base.
                base = table[0]
                if edit in substitutions and edit.start <= position < edit.end:
                    base = _base_index(edit.alt[position - edit.start])
                table[edit.allele] = base
            tables.append((position, table))

    changed = [(edit.start, edit.end) for edit in usable]
    return (
        tables,
        structural,
        bool(substitutions),
        claim_start,
        claim_end,
        changed,
        [edit.allele for edit in classified if edit not in usable],
    )


def _suppress_overlapping_records(
    kept: list[tuple[int, int, int, np.ndarray]],
    stats: VariantStats,
) -> None:
    """
    Apply one record per haplotype where records claim the same bases.

    Two records whose REF spans overlap describe the same reference
    bases twice, and only one of them can be realised on a given
    haplotype. Walking the records in POS order and keeping the first
    one that reaches each haplotype is the resolution ``bcftools
    consensus`` uses ("overlaps with another variant, skipping"), and it
    is what a caller's own phasing implies: the earlier record defines
    the haplotype, the later one contradicts it.

    The suppression is written back into the packed genotype byte, so
    the substitution stage and the structural stage both follow it with
    no extra machinery: a haplotype whose slot now reads 0 carries the
    reference at that record and is not a carrier of its edits.

    ``kept`` is ``(pos, ref_span_start, ref_span_end, codes)`` per kept
    record, in POS order; ``codes`` is modified in place.
    """
    if not kept:
        return
    n_samples = kept[0][3].shape[0]
    # Reference coordinate past the last record applied to each
    # haplotype; -1 means nothing applied yet.
    last_end = np.full((n_samples, 2), -1, dtype=np.int64)
    last_insertion_pos = np.full((n_samples, 2), -1, dtype=np.int64)
    allele1 = CODE_ALLELE1.astype(np.int16)
    allele2 = CODE_ALLELE2.astype(np.int16)

    for rec_pos, rec_start, rec_end, codes in kept:
        slots = np.stack([allele1[codes], allele2[codes]], axis=1)  # S x 2, int16
        # A slot claims the record only when it names a real ALT.
        claims = (slots >= 1) & (slots <= MAX_ALT_ALLELES)
        if not claims.any():
            continue
        blocked = claims & ((rec_start < last_end) | (last_insertion_pos == rec_pos))
        if blocked.any():
            stats.overlapping_records_suppressed += int(blocked.sum())
            patched = np.where(blocked, 0, slots)
            codes[:] = (patched[:, 0] * N_ALLELE_SLOTS + patched[:, 1]).astype(
                codes.dtype
            )
        applied = claims & ~blocked
        np.maximum(last_end, np.where(applied, rec_end, -1), out=last_end)
        last_insertion_pos[applied] = rec_pos if rec_start == rec_end else -1


def _merge_substitution_rows(rows, missing_policy):
    """Combine already overlap-resolved records into one base table per locus.

    REF calls do not overwrite an earlier applied ALT. Actual per-haplotype
    conflicts were suppressed in original record order before this step.
    """
    if len(rows) == 1:
        return rows[0]
    ref = int(rows[0][0][0])
    bases = list(
        dict.fromkeys(
            int(base)
            for source, _, _ in rows
            for base in source
            if int(base) < 4 and int(base) != ref
        )
    )
    table = np.full(N_ALLELES, NO_ALLELE, dtype=np.uint8)
    table[: len(bases) + 1] = [ref, *bases]
    mapping = np.full(256, ALLELE_MISSING, dtype=np.uint8)
    mapping[table[table != NO_ALLELE]] = np.arange(len(bases) + 1, dtype=np.uint8)
    chosen = None
    for source_table, _, codes in rows:
        slots = np.stack([CODE_ALLELE1[codes], CODE_ALLELE2[codes]], axis=1)
        resolved = slots.copy()
        called = slots < N_ALLELES
        resolved[called] = mapping[source_table[slots[called]]]
        if chosen is None:
            chosen = resolved
            continue
        # Reference/missing-as-reference permits a later called ALT. Missing-as-
        # zero remains unknown, as an explicitly masked earlier call should.
        open_slot = (chosen == 0) | (
            (chosen == ALLELE_MISSING) & (missing_policy == "reference")
        )
        take = open_slot & (resolved > 0) & (resolved < N_ALLELES)
        chosen[take] = resolved[take]
    codes = (chosen[:, 0] * N_ALLELE_SLOTS + chosen[:, 1]).astype(np.uint8)
    return table, len(bases), codes


def scan_structural_span(
    *,
    vcf_path: str | Path,
    chrom: str,
    start: int,
    end: int,
    variant_classes: frozenset[str],
    threads: int | None = None,
) -> tuple[int, int]:
    """
    Deletion load and widest claimed interval in this reference range.

    Returns ``(deleted_bases, max_claim_span)``. The first sizes the
    working window, because a deletion pulls downstream bases into a
    fixed-length output and those bases have to be read first. The
    second sizes how far left the extractor must look to see a record
    that overlaps - and therefore suppresses - one inside the window.

    This is a deliberately cheap pass: it reads REF/ALT and INFO/END but
    never touches genotypes, so it costs a fraction of a full
    extraction. The pipeline uses it to size the working window before
    doing the real work, because a deletion pulls downstream bases into
    a fixed-length output and those bases have to be read first.
    """
    if variant_classes == frozenset({"snv"}):
        return 0, 0
    kwargs = {"gts012": False, "strict_gt": True, "lazy": True}
    if threads is not None:
        kwargs["threads"] = int(threads)
    vcf = VCF(str(vcf_path), **kwargs)
    removed = 0
    widest = 0
    try:
        for var in vcf(f"{chrom}:{start}-{end}"):
            ref = var.REF
            alts = list(var.ALT)
            if not alts or len(alts) > MAX_ALT_ALLELES:
                continue
            symbolic_end = None
            if any(a.startswith("<") for a in alts):
                try:
                    value = var.INFO.get("END")
                except KeyError:
                    value = None
                symbolic_end = None if value is None else int(value)
            worst = 0
            for index, alt in enumerate(alts, start=1):
                try:
                    edit = classify_alt(
                        int(var.POS),
                        ref,
                        alt,
                        index,
                        symbolic_end=symbolic_end,
                    )
                except UnsupportedVariant:
                    continue
                if edit.variant_class not in variant_classes:
                    continue
                worst = max(worst, -edit.length_change)
                widest = max(widest, edit.end - edit.start)
            removed += max(0, worst)
            widest = max(widest, len(ref))
    finally:
        vcf.close()
    return removed, widest


def find_chain_start(
    *,
    vcf_path: str | Path,
    chrom: str,
    start: int,
    max_lookback: int,
    variant_classes: frozenset[str],
    threads: int | None = None,
) -> tuple[int, bool]:
    """
    Find where the overlap chain feeding this window can safely begin.

    Which of two overlapping records gets applied depends on the one
    before it, so a record just upstream of the window can decide
    whether a record *inside* it is applied. Reading a fixed distance
    to the left does not settle the question: the chain has to be picked
    up at a reference position that no record's claimed interval
    straddles, because no record upstream of such a gap can reach past
    it.

    Returns ``(chain_start, is_clean)``. ``is_clean`` is False when no
    gap was found within ``max_lookback``, which means the resolution of
    overlapping records near the window start is not guaranteed to match
    what reading the whole file would give.
    """
    if variant_classes == frozenset({"snv"}) or max_lookback <= 0:
        return start, True
    low = max(1, start - max_lookback)
    if low == start:
        return start, True
    kwargs = {"gts012": False, "strict_gt": True, "lazy": True}
    if threads is not None:
        kwargs["threads"] = int(threads)
    vcf = VCF(str(vcf_path), **kwargs)
    spans: list[tuple[int, int]] = []
    try:
        for var in vcf(f"{chrom}:{low}-{start}"):
            ref = var.REF
            alts = list(var.ALT)
            if not alts or len(alts) > MAX_ALT_ALLELES:
                continue
            symbolic_end = None
            if any(a.startswith("<") for a in alts):
                try:
                    value = var.INFO.get("END")
                except KeyError:
                    value = None
                symbolic_end = None if value is None else int(value)
            for index, alt in enumerate(alts, start=1):
                try:
                    edit = classify_alt(
                        int(var.POS),
                        ref,
                        alt,
                        index,
                        symbolic_end=symbolic_end,
                    )
                except UnsupportedVariant:
                    continue
                if edit.variant_class in variant_classes:
                    spans.append((edit.start, edit.end))
    finally:
        vcf.close()
    if not spans:
        return start, True
    # Walk left from the window edge to the first position no claimed
    # interval straddles.
    crossings = sorted(spans, key=lambda t: t[0], reverse=True)
    position = start
    for claim_start, claim_end in crossings:
        if claim_start < position < claim_end:
            position = claim_start
    if position <= low:
        return low, False
    return position, True


def extract_variants(
    *,
    vcf_path: str | Path,
    chrom: str,
    start: int,
    end: int,
    reference_seq: str,
    mode: Mode = "dosage",
    samples: Sequence[str] | None = None,
    threads: int | None = None,
    missing_policy: MissingPolicy = "zero",
    unphased_policy: UnphasedPolicy = "error",
    random_seed: int | None = None,
    ref_mismatch_policy: Policy = "error",
    ambiguous_ref_policy: Policy = "error",
    haploid_policy: HaploidPolicy = "duplicate",
    ploidy_policy: PloidyPolicy = "error",
    variant_classes: frozenset[str] | None = None,
    biallelic_only: bool = False,
    mixed_record_policy: str = "exclude",
    chain_lookback: int = 0,
    overlap_policy: OverlapPolicy = "first",
    pass_only: bool = False,
    min_qual: float | None = None,
    min_af: float | None = None,
    max_af: float | None = None,
    af_source: AFSource = "auto",
    mask: np.ndarray | None = None,
    strict: bool = False,
) -> VariantData:
    """
    Direct indexed regional VCF/BCF extraction via cyvcf2.

    Filtering order is intentionally cheap-to-expensive:
      1. FILTER / QUAL / INFO-AF site filters
      2. REF/ALT shape and alphabet
      3. mask checks and per-haplotype position conflict resolution
      4. FASTA validation
      5. genotype decoding
      6. sample-derived allele-frequency filter, after assembly

    ``strict`` escalates the empty-region and partial-multiallelic
    warnings to errors.
    """
    validate_random_assignment(mode, unphased_policy, random_seed)
    _validate_policies(
        mode,
        missing_policy,
        unphased_policy,
        ref_mismatch_policy,
        ambiguous_ref_policy,
        haploid_policy,
        ploidy_policy,
        af_source,
    )
    for name, value in (
        ("pass_only", pass_only),
        ("biallelic_only", biallelic_only),
        ("strict", strict),
    ):
        boolean(name, value)
    check_choice("mixed_record_policy", mixed_record_policy, MIXED_RECORD_POLICIES)
    validate_site_options(
        threads=threads,
        samples=samples,
        min_qual=min_qual,
        min_af=min_af,
        max_af=max_af,
    )

    expected = end - start + 1
    if len(reference_seq) != expected:
        raise ValueError(
            f"reference_seq length {len(reference_seq)} != expected {expected}"
        )
    if mask is not None and len(mask) != expected:
        raise ValueError(f"mask length {len(mask)} != region length {expected}")

    kwargs = {"gts012": False, "strict_gt": True, "lazy": True}
    if samples is not None:
        kwargs["samples"] = list(samples)
    if threads is not None:
        kwargs["threads"] = int(threads)

    vcf = VCF(str(vcf_path), **kwargs)
    # cyvcf2 returns genotype columns in VCF-header order; sample_names is
    # the order the caller asked for. Diagnostics must use the former.
    vcf_sample_names = list(vcf.samples)
    sample_names, column_order = resolve_sample_selection(vcf_sample_names, samples)
    region = f"{chrom}:{max(1, start - max(0, int(chain_lookback)))}-{end}"

    check_choice("overlap_policy", overlap_policy, OVERLAP_POLICIES)
    if variant_classes is None:
        variant_classes = frozenset({"snv"})
    structural_requested = bool(variant_classes & LENGTH_CHANGING)
    if structural_requested and mode != "haplotypes":
        raise ValueError(
            "Length-changing variant classes "
            f"({', '.join(sorted(variant_classes & LENGTH_CHANGING))}) "
            "are only meaningful in --mode haplotypes: a deletion or "
            "insertion has no definition when two sequences are averaged "
            "position by position."
        )

    filtering_af = min_af is not None or max_af is not None
    sample_af_filter = filtering_af and af_source in {"auto", "samples"}

    stats = VariantStats()
    random_assignments: list[dict] = []
    variants_by_pos: dict[int, tuple] = {}
    duplicate_positions: set[int] = set()
    record_positions: set[int] = set()
    substitution_groups: dict[int, list[tuple]] = {}
    rejected_positions: set[int] = set()
    ploidy_seen = np.zeros(len(vcf_sample_names), dtype=np.uint8)
    kept_records: list[tuple[int, int, int, np.ndarray]] = []
    structural_pending: list[tuple[int, list[int], np.ndarray, list[Edit]]] = []

    try:
        _check_region_contig(vcf, chrom)
        for var in vcf(region):
            stats.records_seen += 1
            pos = int(var.POS)

            if pass_only and var.FILTER is not None:
                stats.filtered_not_pass += 1
                rejected_positions.add(pos)
                continue

            if min_qual is not None:
                qual = var.QUAL
                if qual is None or not np.isfinite(qual) or qual < min_qual:
                    stats.filtered_low_qual += 1
                    rejected_positions.add(pos)
                    continue

            info_af_used = False
            if filtering_af and af_source in {"auto", "info"}:
                info_af = _record_allele_frequency(var)
                if info_af is not None:
                    info_af_used = True
                    if (min_af is not None and info_af < min_af) or (
                        max_af is not None and info_af > max_af
                    ):
                        stats.filtered_allele_frequency += 1
                        rejected_positions.add(pos)
                        continue
                elif af_source == "info":
                    raise ValueError(
                        f"{chrom}:{pos}: --af-source info was requested but "
                        "the record has no INFO/AF field."
                    )

            ref = var.REF
            alts = list(var.ALT)

            if not alts:
                stats.non_snv_records += 1
                rejected_positions.add(pos)
                continue

            if biallelic_only and len(alts) != 1:
                stats.filtered_multiallelic += 1
                continue

            if len(alts) > MAX_ALT_ALLELES:
                stats.too_many_alt_alleles += 1
                rejected_positions.add(pos)
                msg = (
                    f"{chrom}:{pos}: {len(alts)} ALT alleles; hottools "
                    f"supports at most {MAX_ALT_ALLELES}."
                )
                if strict:
                    raise ValueError(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                continue

            # The VCF specification calls for upper-case bases, but
            # htslib tolerates lower case and so do we. The counter is
            # incremented only if the record is ultimately kept.
            was_lowercase = ref.islower() or any(
                a.islower() and not a.startswith("<") for a in alts
            )
            if was_lowercase:
                ref = ref.upper()
                alts = [a if a.startswith("<") else a.upper() for a in alts]

            classified = _classify_record(
                pos, ref, alts, var, variant_classes, stats, mixed_record_policy
            )
            if classified is None:
                rejected_positions.add(pos)
                continue
            (
                allele_tables,
                record_edits,
                _has_substitution,
                claim_start,
                claim_end,
                changed,
                excluded_alleles,
            ) = classified

            rel = pos - start
            span = len(ref)
            # Two spans, and they are not the same thing. The REF span
            # is the stretch of FASTA the record quotes and is what gets
            # validated; the event span also covers what a symbolic
            # <DEL> removes, whose END can reach far past a one-base
            # REF, and is what decides whether the record is in scope.
            event_end = rel + span
            for edit in record_edits:
                event_end = max(event_end, edit.end - start)

            if claim_end <= start:
                # Entirely upstream of the window, but still part of the
                # overlap chain: bcftools decides what to skip by
                # walking the whole file, so a record just left of the
                # window can suppress one inside it. Nothing is encoded
                # from it; it only contributes its claim to the chain.
                decoded = _decode_genotypes(
                    var,
                    vcf_sample_names,
                    len(alts),
                    haploid_policy,
                    ploidy_policy,
                    VariantStats(),
                )
                if decoded is not None:
                    chain_codes = _project_excluded_alleles(
                        decoded[0], excluded_alleles
                    )
                    chain_assignments: list[dict] = []
                    chain_codes = _apply_missing_and_phase(
                        var, chain_codes, decoded[2], missing_policy,
                        unphased_policy, VariantStats(),
                        check_phase=(mode == "haplotypes"),
                        random_seed=random_seed, sample_names=vcf_sample_names,
                        assignments=chain_assignments, original_codes=decoded[0],
                    )
                    # Apply the same sample-AF policy before this upstream
                    # record can claim bases in the overlap sweep.
                    if sample_af_filter and not info_af_used:
                        frequency = _record_alt_frequency(chain_codes)
                        if (min_af is not None and frequency < min_af) or (
                            max_af is not None and frequency > max_af
                        ):
                            stats.filtered_allele_frequency += 1
                            continue
                    random_assignments.extend(chain_assignments)
                    stats.randomly_assigned_genotypes += len(chain_assignments)
                    kept_records.append((pos, claim_start, claim_end, chain_codes))
                continue

            # A record is in scope when its event span *intersects* the
            # window: a multi-base event can straddle either edge, and
            # the part that falls inside is real sequence for this
            # sample. (With SNVs only, both spans are 1 and this reduces
            # to the old containment test.)
            if min(len(reference_seq), event_end) <= max(0, rel):
                stats.outside_region += 1
                continue
            if rel < 0 or event_end > len(reference_seq):
                stats.clipped_at_edge += 1

            lo = max(0, rel)
            hi = min(len(reference_seq), rel + span)

            allele_tables = [
                (position, table)
                for position, table in allele_tables
                if start <= position <= end
            ]
            if not allele_tables and not record_edits:
                stats.outside_region += 1
                continue
            # Position collisions are resolved after genotype decoding/filtering,
            # per haplotype and original record order, rather than dropping both.
            block_positions = [position for position, _ in allele_tables]

            if mask is not None and _touches_mask(mask, changed, start):
                stats.masked_out += 1
                continue

            # When the quoted REF bases lie entirely outside the window
            # there is nothing to compare against; the event still
            # applies through its clipped interval.
            fasta_ref = reference_seq[lo:hi].upper() if hi > lo else ""
            vcf_ref = ref[lo - rel : hi - rel] if hi > lo else ""

            if any(b not in VALID_BASES for b in fasta_ref):
                stats.ambiguous_fasta += 1
                rejected_positions.add(pos)
                msg = (
                    f"Ambiguous FASTA base in "
                    f"{chrom}:{start + lo}-{start + hi - 1}: {fasta_ref!r}"
                )
                if ambiguous_ref_policy == "error":
                    raise ValueError(msg)
                if ambiguous_ref_policy == "warn":
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                continue

            if fasta_ref != vcf_ref:
                stats.ref_mismatch += 1
                rejected_positions.add(pos)
                msg = (
                    f"REF mismatch at {chrom}:{pos}: "
                    f"VCF REF={vcf_ref}, FASTA REF={fasta_ref}"
                )
                if ref_mismatch_policy == "error":
                    raise ValueError(msg)
                if ref_mismatch_policy == "warn":
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                continue

            decoded = _decode_genotypes(
                var,
                vcf_sample_names,
                len(alts),
                haploid_policy,
                ploidy_policy,
                stats,
            )
            if decoded is None:
                rejected_positions.add(pos)
                continue
            codes, ploidy, phased = decoded
            original_codes = codes
            pending_assignments: list[dict] = []
            codes = _project_excluded_alleles(codes, excluded_alleles)
            ploidy_seen = np.maximum(ploidy_seen, ploidy)

            codes = _apply_missing_and_phase(
                var,
                codes,
                phased,
                missing_policy,
                unphased_policy,
                stats,
                check_phase=(mode == "haplotypes"),
                random_seed=random_seed,
                sample_names=vcf_sample_names,
                assignments=pending_assignments,
                original_codes=original_codes,
            )

            # Sample-derived allele frequency, per record. Deciding it here
            # rather than on the assembled matrix means a structural-only
            # record (which has no row there) is filtered too, the whole
            # record goes or stays together, and a filtered record never
            # suppresses another one it overlaps. Under 'auto' a record
            # whose INFO/AF already decided is not filtered twice.
            if sample_af_filter and not info_af_used:
                alt_freq = _record_alt_frequency(codes)
                if (min_af is not None and alt_freq < min_af) or (
                    max_af is not None and alt_freq > max_af
                ):
                    stats.filtered_allele_frequency += 1
                    rejected_positions.add(pos)
                    # A filtered site does not exist for the duplicate
                    # rule, exactly as one rejected by INFO/AF.
                    continue

            random_assignments.extend(pending_assignments)
            stats.randomly_assigned_genotypes += len(pending_assignments)
            if pos in record_positions:
                duplicate_positions.add(pos)
                stats.duplicate_records += 1
            record_positions.add(pos)
            for position, table in allele_tables:
                row = (table, len(alts), codes)
                if position in variants_by_pos:
                    group = substitution_groups.setdefault(
                        position, [variants_by_pos[position]]
                    )
                    group.append(row)
                else:
                    variants_by_pos[position] = row
            kept_records.append((pos, claim_start, claim_end, codes))
            stats.records_genotyped += 1
            if len(alts) > 1:
                stats.multiallelic_records += 1
            if was_lowercase:
                stats.lowercase_alleles += 1
            if record_edits:
                structural_pending.append((pos, block_positions, codes, record_edits))
                stats.structural_records += 1
                for edit in record_edits:
                    if edit.variant_class == "del":
                        stats.deletion_edits += 1
                    elif edit.variant_class == "ins":
                        stats.insertion_edits += 1
                    else:
                        stats.symbolic_deletion_edits += 1
    finally:
        vcf.close()

    # Keep the first applicable ALT independently on each haplotype. A REF
    # call on an earlier record does not suppress a later called alternate.
    if any(end - begin != 1 for _, begin, end, _ in kept_records):
        _suppress_overlapping_records(kept_records, stats)
    elif duplicate_positions:
        # Unique SNVs cannot overlap: only scan the colliding loci, retaining
        # the fast path for large windows with a few duplicate records.
        _suppress_overlapping_records(
            [row for row in kept_records if row[0] in duplicate_positions], stats
        )
    for position, rows in substitution_groups.items():
        variants_by_pos[position] = _merge_substitution_rows(rows, missing_policy)
    substitution_groups.clear()
    stats.duplicate_positions_resolved = len(duplicate_positions)

    # The sweep patched the genotype rows in place; drop these extra
    # references so the rows really are released once the matrix exists.
    kept_records.clear()

    partial = rejected_positions.intersection(variants_by_pos)
    stats.multiallelic_sites_partial = len(partial)
    if partial:
        shown = ", ".join(f"{chrom}:{p}" for p in sorted(partial)[:5])
        msg = (
            f"{len(partial)} retained SNV(s) share a position with a "
            f"discarded record (split multiallelic site); only the "
            f"retained allele(s) are encoded: {shown}"
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, MultiallelicWarning, stacklevel=2)

    positions = np.asarray(sorted(variants_by_pos), dtype=np.int32)
    n_variants = len(positions)
    n_samples = len(sample_names)

    allele_bases = np.full((n_variants, N_ALLELES), NO_ALLELE, dtype=np.uint8)
    n_alt = np.zeros(n_variants, dtype=np.uint8)
    genotypes = np.full((n_variants, n_samples), GENOTYPE_UNKNOWN, dtype=np.uint8)
    # cyvcf2 returns samples in VCF order. The requested order is
    # restored row by row while filling the matrix, which avoids building
    # a second full matrix just to permute its columns.
    order = None
    if column_order is not None and column_order != list(range(n_samples)):
        order = np.asarray(column_order, dtype=np.intp)
    for i, pos in enumerate(positions.tolist()):
        allele_row, alt_count, codes = variants_by_pos[pos]
        allele_bases[i] = allele_row
        n_alt[i] = alt_count
        if order is None:
            genotypes[i] = codes
        else:
            np.take(codes, order, out=genotypes[i])

    # Release the per-variant genotype rows now that the contiguous
    # matrix exists; holding both doubles the compact genotype array.
    variants_by_pos.clear()

    # The sample allele-frequency filter ran per record during the scan.
    # Frequencies for het_policy='major' are computed lazily from the
    # final matrix on first use.
    allele_freq = None

    structural = None
    if structural_requested and structural_pending:
        # A length-changing record can legitimately contribute no entry
        # to the substitution table - a deletion whose POS lies just
        # before the window is one - so surviving edits are tracked by
        # record rather than inferred from kept positions.
        rows = [
            (p, codes, record_edits)
            for p, block, codes, record_edits in structural_pending
        ]
        if rows:
            structural_codes = np.stack([r[1] for r in rows])
            if column_order is not None:
                structural_codes = structural_codes[
                    :, np.asarray(column_order, dtype=np.intp)
                ]
            pairs = [
                (record_index, edit)
                for record_index, (_, _, record_edits) in enumerate(rows)
                for edit in record_edits
            ]
            structural = build_structural_edits(
                pairs,
                np.asarray([r[0] for r in rows], dtype=np.int32),
                structural_codes,
            )

    stats.kept = n_variants

    if overlap_policy == "error" and (
        stats.overlapping_records_suppressed or stats.overlapping_blocks_dropped
    ):
        raise ValueError(
            f"{stats.overlapping_records_suppressed} genotype(s) and "
            f"{stats.overlapping_blocks_dropped} record(s) in "
            f"{chrom}:{start}-{end} claim reference bases another record "
            "already claims on the same haplotype (overlap_policy='error'). "
            "Use --overlap-policy first to apply the first record in POS "
            "order, as bcftools consensus does."
        )
    if stats.overlapping_records_suppressed or stats.overlapping_blocks_dropped:
        warnings.warn(
            f"{stats.overlapping_records_suppressed} genotype(s) and "
            f"{stats.overlapping_blocks_dropped} record(s) were set aside "
            "because two records claimed the same reference bases. The "
            "first record in POS order is applied, as bcftools consensus "
            "does. Where such records run in long chains, which one wins "
            "near a window edge can differ from reading the whole file.",
            StructuralEditWarning,
            stacklevel=2,
        )

    if stats.haploid_genotypes and haploid_policy in {"duplicate", "blank"}:
        rendered = (
            "duplicated onto both haplotype tracks"
            if haploid_policy == "duplicate"
            else "left blank on the second haplotype track"
        )
        warnings.warn(
            f"{stats.haploid_genotypes} haploid genotype(s) across "
            f"{stats.haploid_records} record(s) in {region}; the single "
            f"allele was {rendered}. Per-sample ploidy is recorded in the "
            "metadata as sample_ploidy.",
            HaploidWarning,
            stacklevel=2,
        )

    if n_variants == 0 and (structural is None or structural.n_edits == 0):
        msg = (
            f"No usable variants in {region} "
            f"({stats.records_seen} record(s) seen). Output will be the "
            "reference sequence for every sample. Check the region, the "
            "chromosome-naming convention, and the filter policies."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, EmptyRegionWarning, stacklevel=2)

    if column_order is not None:
        ploidy_seen = ploidy_seen[np.asarray(column_order, dtype=np.intp)]

    return VariantData(
        mode=mode,
        samples=sample_names,
        positions=positions,
        allele_bases=allele_bases,
        n_alt=n_alt,
        genotypes=genotypes,
        sample_ploidy=ploidy_seen,
        stats=stats,
        random_assignments=random_assignments,
        structural=structural,
        _allele_freq=allele_freq,
    )
