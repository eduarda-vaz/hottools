from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from .config_validation import (
    boolean,
    finite_number,
    nonempty_path,
    positive_integer,
    validate_random_assignment,
    validate_site_options,
)
from .constants import (
    AF_SOURCES,
    ANCHOR_DELETION_POLICIES,
    DTYPES,
    FORMATS,
    HAPLOID_POLICIES,
    HET_POLICIES,
    LAYOUTS,
    MISSING_POLICIES,
    MIXED_RECORD_POLICIES,
    MODES,
    OVERLAP_POLICIES,
    OVERWRITE_POLICIES,
    PLOIDY_POLICIES,
    RECORD_POLICIES,
    UNPHASED_POLICIES,
    AfSource,
    AnchorDeletionPolicy,
    HaploidPolicy,
    HetPolicy,
    MissingPolicy,
    MixedRecordPolicy,
    Mode,
    OutputLayout,
    OverlapPolicy,
    OverwritePolicy,
    PloidyPolicy,
    RecordPolicy,
    UnphasedPolicy,
    check_choice,
)
from .contigs import alias_key, resolve_input_contigs, validate_alias_policy
from .edits import LENGTH_CHANGING, parse_variant_classes
from .encoding import (
    _build_variant_tables,
    _encode_haplotype_tracks,
    _encode_single_track,
    _table_bytes,
)
from .memory import (
    choose_batch_size,
    full_output_shape,
    human_bytes,
    peak_multiplier_for,
)
from .memory import (
    sequence_axis as sequence_axis_for,
)
from .reference import ReferenceReader, one_hot_reference
from .regions import Region, read_mask
from .transforms import apply_transform, compile_all, haplotype_signatures
from .variants import (
    StructuralEditWarning,
    extract_variants,
    find_chain_start,
    scan_structural_span,
)
from .writers import (
    check_output_plan,
    make_writer,
    plan_outputs,
    previous_run_outputs,
    safe_sample_names,
)

#: Kept for callers that imported them from here; the definitions live
#: in :mod:`hottools.constants`.
VALID_MODES = MODES
VALID_DTYPES = DTYPES
VALID_LAYOUTS = LAYOUTS
VALID_FORMATS = FORMATS

#: Per-variant Python object overhead during extraction (row array,
#: allele table, dictionary and list entries), fitted to tracemalloc
#: peaks. With it, the reported estimate was 0.90-1.16x the measured
#: peak across 50-2,504 samples and 1,000-3,000 variants.
EXTRACTION_BYTES_PER_VARIANT = 512


@dataclass(slots=True)
class RunConfig:
    vcf_path: str
    fasta_path: str
    region: Region
    output_path: str

    mode: Mode = "dosage"
    output_format: str = "npy"
    per_sample: bool = False
    dtype: str = "float32"
    output_layout: OutputLayout = "samples_first"

    batch_size: int | str | None = "auto"
    memory_fraction: float = 0.40
    threads: int | None = None
    contig_alias_policy: str = "exact"

    samples: Sequence[str] | None = None

    missing_policy: MissingPolicy = "zero"
    unphased_policy: UnphasedPolicy = "error"
    random_seed: int | None = None
    ref_mismatch_policy: RecordPolicy = "error"
    ambiguous_ref_policy: RecordPolicy = "error"
    haploid_policy: HaploidPolicy = "duplicate"
    ploidy_policy: PloidyPolicy = "error"
    #: Used by ``mode='consensus'`` only.
    het_policy: HetPolicy = "alternate"

    # Site filters
    pass_only: bool = False
    min_qual: float | None = None
    min_af: float | None = None
    max_af: float | None = None
    af_source: AfSource = "auto"
    mask_path: str | None = None

    h5_compression: str | None = None
    h5_chunk_samples: int | None = None
    h5_chunk_length: int | None = None

    #: Comma-separated variant classes to apply; None means SNVs only.
    variant_classes: str | None = None
    biallelic_only: bool = False
    mixed_record_policy: MixedRecordPolicy = "exclude"
    #: Reference position pinned to a fixed output offset: "start",
    #: "center", or an explicit 1-based coordinate.
    anchor: str = "start"
    #: What to do when a deletion removes the anchor base itself.
    anchor_deletion_policy: AnchorDeletionPolicy = "right"
    #: What to do when records claim the same bases on one haplotype.
    overlap_policy: OverlapPolicy = "first"

    metadata_json: bool = True
    dry_run: bool = False
    strict: bool = False
    overwrite: OverwritePolicy = "error"

    #: Optional shared FASTA handle, reused across regions.
    reference_reader: ReferenceReader | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def reverse_complement(self) -> bool:
        """
        Derived from the region, never stored separately.

        Keeping one source of truth is what stops the sidecar metadata
        from disagreeing with the orientation of the data.
        """
        return self.region.reverse_complement

    @property
    def strand(self) -> str:
        return self.region.strand

    def resolve_anchor(self) -> tuple[int, int]:
        """
        Return ``(reference_position, output_offset)`` for the anchor.

        The anchor is the one position guaranteed to land at a fixed
        offset in the output, so an upstream deletion cannot slide the
        biological target within the model input.
        """
        region = self.region
        spec = str(self.anchor).strip()
        if spec.lower() == "start":
            return region.start, 0
        if spec.lower() == "center":
            offset = (region.length - 1) // 2
            return region.start + offset, offset
        if ":" in spec:
            chrom, coordinate = spec.rsplit(":", 1)
            if chrom != region.chrom and not (
                self.contig_alias_policy == "auto"
                and alias_key(chrom) == alias_key(region.chrom)
            ):
                raise ValueError(
                    f"anchor contig {chrom!r} does not match region "
                    f"contig {region.chrom!r}"
                )
        else:
            coordinate = spec
        try:
            position = int(coordinate.replace(",", ""))
        except ValueError as exc:
            raise ValueError(
                "anchor must be 'start', 'center', or a 1-based position"
            ) from exc
        if not region.start <= position <= region.end:
            raise ValueError(
                f"anchor {position} lies outside the region {region.label}"
            )
        return position, position - region.start

    def validate(self) -> np.dtype:
        """Check every choice-valued field before any file is opened."""
        validate_alias_policy(self.contig_alias_policy)
        if not isinstance(self.region, Region):
            raise ValueError("region must be a Region instance")
        for name in ("vcf_path", "fasta_path", "output_path"):
            nonempty_path(name, getattr(self, name))
        if self.mask_path is not None:
            nonempty_path("mask_path", self.mask_path)
        for name, value, choices in (
            ("dtype", self.dtype, DTYPES),
            ("output_layout", self.output_layout, LAYOUTS),
            ("output_format", str(self.output_format).lower(), FORMATS),
            ("mode", self.mode, MODES),
            ("missing_policy", self.missing_policy, MISSING_POLICIES),
            ("mixed_record_policy", self.mixed_record_policy, MIXED_RECORD_POLICIES),
            ("unphased_policy", self.unphased_policy, UNPHASED_POLICIES),
            ("ref_mismatch_policy", self.ref_mismatch_policy, RECORD_POLICIES),
            ("ambiguous_ref_policy", self.ambiguous_ref_policy, RECORD_POLICIES),
            ("haploid_policy", self.haploid_policy, HAPLOID_POLICIES),
            ("ploidy_policy", self.ploidy_policy, PLOIDY_POLICIES),
            ("het_policy", self.het_policy, HET_POLICIES),
            ("af_source", self.af_source, AF_SOURCES),
            ("overwrite", self.overwrite, OVERWRITE_POLICIES),
            ("overlap_policy", self.overlap_policy, OVERLAP_POLICIES),
            (
                "anchor_deletion_policy",
                self.anchor_deletion_policy,
                ANCHOR_DELETION_POLICIES,
            ),
        ):
            check_choice(name, value, choices)
        validate_random_assignment(self.mode, self.unphased_policy, self.random_seed)
        validate_site_options(
            threads=self.threads,
            samples=self.samples,
            min_qual=self.min_qual,
            min_af=self.min_af,
            max_af=self.max_af,
        )
        finite_number("memory_fraction", self.memory_fraction, upper=1)
        if self.memory_fraction == 0:
            raise ValueError("memory_fraction must be > 0 and <= 1")
        if self.batch_size is not None and self.batch_size not in ("auto", "none"):
            positive_integer("batch_size", self.batch_size)
        for name in (
            "per_sample",
            "pass_only",
            "biallelic_only",
            "metadata_json",
            "dry_run",
            "strict",
        ):
            boolean(name, getattr(self, name))
        if self.h5_compression not in (None, "gzip", "lzf"):
            raise ValueError("h5_compression must be None, gzip, or lzf")
        positive_integer("h5_chunk_samples", self.h5_chunk_samples, optional=True)
        positive_integer("h5_chunk_length", self.h5_chunk_length, optional=True)
        if str(self.output_format).lower() != "h5" and any(
            value is not None
            for value in (
                self.h5_compression,
                self.h5_chunk_samples,
                self.h5_chunk_length,
            )
        ):
            raise ValueError(
                "h5_compression and h5_chunk settings require output_format='h5'"
            )
        if self.mode != "consensus" and self.het_policy != "alternate":
            raise ValueError("het_policy reference/major requires mode='consensus'")
        classes = parse_variant_classes(self.variant_classes)
        if classes & LENGTH_CHANGING and self.mode != "haplotypes":
            raise ValueError(
                "Length-changing variant classes are only meaningful in "
                "--mode haplotypes"
            )
        self.resolve_anchor()
        return np.dtype(self.dtype)


def _apply_reverse_complement(
    batch: np.ndarray,
    *,
    mode: Mode,
    layout: str,
) -> np.ndarray:
    """
    Reverse-complement an encoded batch.

    Channel order is A,C,G,T, so complementing is the channel
    permutation [T,G,C,A] applied together with a reversal of the
    sequence axis. Dosage and consensus are both single-track
    (``B,L,4`` / ``B,4,L``); haplotypes carry an extra axis.
    """
    comp = np.array([3, 2, 1, 0], dtype=np.intp)
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of: {', '.join(VALID_MODES)}")

    if mode == "haplotypes":
        if layout == "samples_first":  # B,2,L,4
            return batch[:, :, ::-1, :][:, :, :, comp]
        return batch[:, :, comp, :][:, :, :, ::-1]  # B,2,4,L

    if layout == "samples_first":  # B,L,4
        return batch[:, ::-1, :][:, :, comp]
    return batch[:, comp, :][:, :, ::-1]  # B,4,L


def _apply_structural(
    batch: np.ndarray,
    *,
    signatures,
    transforms,
    pool: np.ndarray,
    sample_start: int,
    sample_end: int,
    out_length: int,
    layout: str,
) -> np.ndarray:
    """
    Turn an SNV-personalised ``B x 2 x W x 4`` batch into ``B x 2 x L x 4``.

    Haplotypes are grouped by structural signature and each group is
    gathered in one advanced-indexing operation, so the cost scales with
    the number of distinct signatures rather than with the number of
    haplotypes.
    """
    n_samples = sample_end - sample_start
    work = batch.reshape(n_samples * 2, batch.shape[2], 4)
    out = np.zeros((n_samples * 2, out_length, 4), dtype=batch.dtype)

    group = signatures.group[sample_start * 2 : sample_end * 2]
    for g, transform in enumerate(transforms):
        rows = np.flatnonzero(group == g)
        if rows.size:
            apply_transform(work, rows, transform, pool, out)

    out = out.reshape(n_samples, 2, out_length, 4)
    if layout == "channels_first":
        return np.transpose(out, (0, 1, 3, 2))
    return out


def run_hottools(config: RunConfig) -> dict:
    t_total = time.perf_counter()
    region = config.region
    dtype = config.validate()
    output_format = config.output_format.lower()

    timings: dict[str, float] = {}
    variant_classes = parse_variant_classes(config.variant_classes)
    structural_requested = bool(variant_classes & LENGTH_CHANGING)
    if structural_requested and config.mode != "haplotypes":
        # Checked here as well as in the extractor so the run fails
        # before any reference or VCF I/O happens.
        raise ValueError(
            "Length-changing variant classes "
            f"({', '.join(sorted(variant_classes & LENGTH_CHANGING))}) "
            "are only meaningful in --mode haplotypes: a deletion or "
            "insertion has no definition when two sequences are averaged "
            "position by position."
        )
    anchor_ref, anchor_offset = config.resolve_anchor()

    # Under overwrite='error' an earlier run in this output location
    # already decides the outcome, and finding it needs neither the
    # samples nor the batch size - so check before reading anything.
    # The full plan is checked again once file names are known.
    if config.overwrite == "error":
        earlier = previous_run_outputs(
            config.output_path, output_format, config.per_sample
        )
        if earlier:
            shown = ", ".join(str(p) for p in earlier[:5])
            more = f" and {len(earlier) - 5} more" if len(earlier) > 5 else ""
            raise FileExistsError(
                f"Output already exists: {shown}{more}. Nothing was "
                "written; pass --overwrite replace to replace the earlier "
                "run, or choose another output path."
            )

    t0 = time.perf_counter()
    reader = config.reference_reader
    owns_reader = reader is None
    if owns_reader:
        reader = ReferenceReader(config.fasta_path)
    try:
        vcf_chrom = fasta_chrom = region.chrom
        if config.contig_alias_policy == "auto":
            vcf_chrom, fasta_chrom = resolve_input_contigs(
                config.vcf_path, reader, region.chrom
            )
        work_start, work_end = region.start, region.end
        pad_left = pad_right = chain_lookback = 0
        if variant_classes != frozenset({"snv"}):
            # Sizing the working window needs the deletion load, and a
            # deletion just outside the window can pull bases in, so the
            # cheap scan runs twice: once over the region, once over the
            # region it implies.
            contig_length = reader.length(fasta_chrom)
            # Bases pulled in to replace deleted ones can carry deletions
            # of their own, so widen until the deletion load of the
            # window stops growing. The load is an upper bound over all
            # samples and the window only grows, capped at the contig
            # ends, so this terminates.
            removed = -1
            grown = 0
            while grown != removed:
                removed = grown
                grown, chain_lookback = scan_structural_span(
                    vcf_path=config.vcf_path,
                    chrom=vcf_chrom,
                    start=max(1, region.start - removed),
                    end=min(contig_length, region.end + removed),
                    variant_classes=variant_classes,
                    threads=config.threads,
                )
                if (
                    region.start - removed <= 1
                    and region.end + removed >= contig_length
                ):
                    removed = max(removed, grown)
                    break
            chain_start, chain_clean = find_chain_start(
                vcf_path=config.vcf_path,
                chrom=vcf_chrom,
                start=region.start,
                max_lookback=max(64, 16 * chain_lookback),
                variant_classes=variant_classes,
                threads=config.threads,
            )
            chain_lookback = region.start - chain_start
            if not chain_clean:
                warnings.warn(
                    "Overlapping records run continuously up to the start "
                    "of the region, so which of them applies near the left "
                    "edge may differ from a whole-file reading. This "
                    "indicates records that claim the same reference bases.",
                    StructuralEditWarning,
                    stacklevel=2,
                )
            pad_right = removed
            pad_left = removed if anchor_offset > 0 else 0
            work_start = max(1, region.start - pad_left)
            work_end = min(contig_length, region.end + pad_right)
        reference_seq = reader.interval(fasta_chrom, work_start, work_end)
    finally:
        if owns_reader:
            reader.close()
    work_length = work_end - work_start + 1
    ref_onehot, ref_indices = one_hot_reference(reference_seq, dtype=dtype)
    n_ambiguous = int(np.count_nonzero(ref_indices > 3))

    mask = None
    n_masked = 0
    if config.mask_path:
        mask = read_mask(
            config.mask_path, replace(region, start=work_start, end=work_end),
            contig_alias_policy=config.contig_alias_policy,
        )
        n_masked = int(np.count_nonzero(mask))
        ref_onehot[mask] = 0
    timings["reference"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    variants = extract_variants(
        vcf_path=config.vcf_path,
        chrom=vcf_chrom,
        start=work_start,
        end=work_end,
        reference_seq=reference_seq,
        mode=config.mode,
        samples=config.samples,
        threads=config.threads,
        missing_policy=config.missing_policy,
        unphased_policy=config.unphased_policy,
        random_seed=config.random_seed,
        ref_mismatch_policy=config.ref_mismatch_policy,
        ambiguous_ref_policy=config.ambiguous_ref_policy,
        variant_classes=variant_classes,
        biallelic_only=config.biallelic_only,
        mixed_record_policy=config.mixed_record_policy,
        chain_lookback=chain_lookback,
        overlap_policy=config.overlap_policy,
        haploid_policy=config.haploid_policy,
        ploidy_policy=config.ploidy_policy,
        pass_only=config.pass_only,
        min_qual=config.min_qual,
        min_af=config.min_af,
        max_af=config.max_af,
        af_source=config.af_source,
        mask=mask,
        strict=config.strict,
    )
    timings["vcf_extraction"] = time.perf_counter() - t0

    n_samples = len(variants.samples)
    if n_samples == 0:
        raise ValueError("No samples selected from VCF.")

    payload = variants.genotypes
    genotype_bytes = int(payload.nbytes)
    # Estimated, not measured: extract_variants holds one genotype row
    # per variant and then copies them into the contiguous matrix, so the
    # rows and the matrix briefly coexist (2x the compact array), and
    # each variant carries about EXTRACTION_BYTES_PER_VARIANT of Python
    # object overhead on top. The model was fitted to tracemalloc peaks;
    # it is reported because extraction happens before the memory plan
    # exists and is released before encoding starts.
    genotype_peak_estimate = 2 * genotype_bytes + EXTRACTION_BYTES_PER_VARIANT * len(
        variants.positions
    )

    # Offsets into the working window the substitution stage encodes.
    variant_pos0 = variants.positions.astype(np.int64, copy=False) - work_start

    tables = None
    if config.dry_run:
        # A dry run does not allocate the tables, but it should still
        # report what they would cost, or its memory plan understates
        # the real run.
        tables_bytes = _table_bytes(len(variants.positions), config.mode, dtype)
    else:
        # Built once per run. The variant offsets are validated and
        # turned into a plain list here, so no sample batch pays for the
        # conversion again.
        tables = _build_variant_tables(
            allele_bases=variants.allele_bases,
            variant_pos0=variant_pos0,
            sequence_length=work_length,
            dtype=dtype,
            mode=config.mode,
            missing_policy=config.missing_policy,
            haploid_policy=(
                config.haploid_policy
                if config.haploid_policy in ("duplicate", "blank")
                else "duplicate"
            ),
            het_policy=config.het_policy,
            allele_freq=(
                variants.allele_freq
                if config.mode == "consensus" and config.het_policy == "major"
                else None
            ),
        )
        tables_bytes = tables.nbytes

    structural = variants.structural if structural_requested else None
    signatures = transforms_list = insert_pool = None
    n_unfilled = n_skipped = 0
    if structural is not None and structural.n_edits:
        signatures = haplotype_signatures(
            structural,
            haploid_policy=(
                "blank" if config.haploid_policy == "blank" else "duplicate"
            ),
        )
        transforms_list = compile_all(
            signatures,
            structural,
            work_start=work_start,
            work_end=work_end,
            out_length=region.length,
            anchor_ref=anchor_ref,
            anchor_offset=anchor_offset,
            anchor_deletion_policy=config.anchor_deletion_policy,
        )
        insert_pool = structural.insert_onehot(dtype)
        n_unfilled = sum(t.n_unfilled for t in transforms_list)
        n_skipped = sum(t.n_skipped for t in transforms_list)
        if n_skipped and config.overlap_policy == "error":
            raise ValueError(
                f"{n_skipped} length-changing edit(s) overlap an earlier "
                "edit on the same haplotype (overlap_policy='error')."
            )
        if n_skipped:
            warnings.warn(
                f"{n_skipped} length-changing edit(s) overlapped an earlier "
                "edit on the same haplotype and were skipped; two records "
                "claiming the same reference bases are contradictory input.",
                StructuralEditWarning,
                stacklevel=2,
            )
        if n_unfilled:
            warnings.warn(
                f"{n_unfilled} output position(s) had no source base: the "
                "working window ran out of reference, which happens near a "
                "contig edge or with very large deletions. Those positions "
                "are zero-filled.",
                StructuralEditWarning,
                stacklevel=2,
            )
    else:
        structural = None
    # The window may have been widened for a deletion load and then lost
    # every structural edit - all filtered, masked, skipped or duplicate.
    # The identity transform of a padded window is the slice starting at
    # region.start, so that is what each batch is cut back to.
    crop = None
    if structural is None and (work_start, work_end) != (region.start, region.end):
        first = region.start - work_start
        crop = slice(first, first + region.length)

    structural_bytes = (
        0
        if structural is None
        else (
            structural.nbytes
            + sum(t.source.nbytes + t.insert.nbytes for t in transforms_list)
            + (0 if insert_pool is None else insert_pool.nbytes)
            + signatures.carriers.nbytes
            + signatures.unique.nbytes
            + signatures.group.nbytes
        )
    )
    resident_reference_bytes = (
        ref_onehot.nbytes
        + ref_indices.nbytes
        + len(reference_seq)
        + (0 if mask is None else mask.nbytes)
    )
    writer_peak = peak_multiplier_for(
        reverse_complement=config.reverse_complement,
        output_format=output_format,
    )
    if structural is not None:
        # Working canvas + output + largest advanced-index gather. The
        # gather can cover the entire batch when it shares one signature.
        batch_peak = max(writer_peak, work_length / region.length + 2.0)
    elif crop is not None:
        # The cropped view retains its whole working canvas while a
        # reverse-complement/serialization copy can hold an output tensor.
        batch_peak = work_length / region.length + writer_peak - 1.0
    else:
        batch_peak = writer_peak

    plan = choose_batch_size(
        n_samples=n_samples,
        sequence_length=region.length,
        mode=config.mode,
        dtype=dtype,
        requested=config.batch_size,
        memory_fraction=config.memory_fraction,
        layout=config.output_layout,
        peak_multiplier=batch_peak,
        extra_bytes=(
            genotype_bytes + tables_bytes + structural_bytes + resident_reference_bytes
        ),
    )

    full_shape = full_output_shape(
        n_samples,
        region.length,
        config.mode,
        config.output_layout,
    )
    seq_axis = sequence_axis_for(config.mode, config.output_layout)

    n_batches = -(-n_samples // plan.batch_size)
    metadata = {
        "mode": config.mode,
        "chrom": region.chrom,
        "contig_alias_policy": config.contig_alias_policy,
        "requested_chrom": region.chrom,
        "vcf_chrom": vcf_chrom,
        "fasta_chrom": fasta_chrom,
        "start": region.start,
        "end": region.end,
        "length": region.length,
        "strand": region.strand,
        "reverse_complement": config.reverse_complement,
        "samples": n_samples,
        "sample_ids": list(variants.samples),
        "variants_kept": variants.stats.kept,
        "reference_ambiguous_bases": n_ambiguous,
        "masked_bases": n_masked,
        "variant_classes": sorted(variant_classes),
        "biallelic_only": config.biallelic_only,
        "mixed_record_policy": config.mixed_record_policy,
        "anchor_position": anchor_ref,
        "anchor_output_offset": anchor_offset,
        "working_start": work_start,
        "working_end": work_end,
        "working_length": work_length,
        "working_pad_left": pad_left,
        "working_pad_right": pad_right,
        "overlap_chain_lookback": chain_lookback,
        "structural_edits": 0 if structural is None else structural.n_edits,
        "structural_signatures": (0 if signatures is None else signatures.n_groups),
        "unfilled_positions": n_unfilled,
        "overlapping_edits_skipped": n_skipped,
        "sample_ploidy": variants.sample_ploidy.tolist(),
        "het_policy": config.het_policy if config.mode == "consensus" else None,
        "haploid_policy": config.haploid_policy,
        "unphased_policy": config.unphased_policy,
        "random_seed": None if config.random_seed is None else int(config.random_seed),
        "random_assignment_algorithm": (
            "blake2b-call-v1" if config.unphased_policy == "random" else None
        ),
        "random_assignments": variants.random_assignments,
        "random_assignments_scope": (
            "retained records including overlap-chain context, after allele "
            "projection, before overlap resolution"
        ),
        "contains_pseudo_haplotypes": bool(variants.stats.randomly_assigned_genotypes),
        "multiallelic_records": int(variants.stats.multiallelic_records),
        "dtype": str(dtype),
        "output_format": output_format,
        "output_layout": config.output_layout,
        "output_shape": list(full_shape),
        "per_sample": config.per_sample,
        "batch_size": plan.batch_size,
        "n_batches": n_batches,
        "estimated_full_output_bytes": plan.estimated_full_output_bytes,
        "estimated_full_output_human": human_bytes(plan.estimated_full_output_bytes),
        "bytes_per_sample": plan.bytes_per_sample,
        "bytes_per_sample_human": human_bytes(plan.bytes_per_sample),
        "estimated_peak_batch_bytes": plan.estimated_peak_batch_bytes,
        "estimated_peak_batch_human": human_bytes(plan.estimated_peak_batch_bytes),
        "memory_budget_bytes": plan.memory_budget_bytes,
        "available_memory_bytes": plan.available_memory_bytes,
        "memory_constraints": plan.memory_constraints,
        "peak_multiplier": plan.peak_multiplier,
        "genotype_array_bytes": genotype_bytes,
        "estimated_genotype_extraction_peak_bytes": genotype_peak_estimate,
        "variant_table_bytes": tables_bytes,
        "overwrite": config.overwrite,
        **{f"variant_{k}": v for k, v in variants.stats.as_dict().items()},
    }

    output_plan = plan_outputs(
        output_path=config.output_path,
        fmt=output_format,
        per_sample=config.per_sample,
        samples=list(variants.samples),
        batch_size=plan.batch_size,
        metadata_json=config.metadata_json,
        safe_names=(
            safe_sample_names(list(variants.samples)) if config.per_sample else None
        ),
    )
    # Every planned file is known now, so this refuses a conflicting run
    # before a single byte is written - including under --dry-run, which
    # then reports exactly the error the real run would raise.
    check_output_plan(output_plan, config.overwrite)
    planned = [str(p) for p in output_plan.data_files]
    stale = [str(p) for p in output_plan.stale] if config.overwrite == "replace" else []

    if config.dry_run:
        # Nothing is written, so nothing is reported as written.
        metadata["planned_output_files"] = planned
        metadata["planned_stale_removals"] = stale
        timings["total"] = time.perf_counter() - t_total
        metadata["timings"] = timings
        return metadata

    metadata["output_files"] = planned
    metadata["n_output_files"] = len(planned)
    metadata["stale_outputs_removed"] = stale
    if output_plan.metadata_path is not None:
        metadata["metadata_path"] = str(output_plan.metadata_path)

    t0 = time.perf_counter()
    with make_writer(
        output_path=config.output_path,
        fmt=output_format,
        full_shape=full_shape,
        dtype=dtype,
        samples=variants.samples,
        per_sample=config.per_sample,
        metadata=metadata,
        sequence_axis=seq_axis,
        h5_compression=config.h5_compression,
        h5_chunk_samples=config.h5_chunk_samples,
        h5_chunk_length=config.h5_chunk_length,
        overwrite=config.overwrite,
        plan=output_plan,
    ) as writer:
        for start_s in range(0, n_samples, plan.batch_size):
            end_s = min(start_s + plan.batch_size, n_samples)

            common = {
                "ref_onehot": ref_onehot,
                "genotypes": variants.genotypes,
                "tables": tables,
                "sample_start": start_s,
                "sample_end": end_s,
                # The structural stage needs the plain layout to index
                # the sequence axis; transpose after it, not before.
                "output_layout": (
                    "samples_first"
                    if structural is not None or crop is not None
                    else config.output_layout
                ),
            }
            if config.mode == "haplotypes":
                batch = _encode_haplotype_tracks(**common)
            else:
                batch = _encode_single_track(**common)

            if structural is not None:
                batch = _apply_structural(
                    batch,
                    signatures=signatures,
                    transforms=transforms_list,
                    pool=insert_pool,
                    sample_start=start_s,
                    sample_end=end_s,
                    out_length=region.length,
                    layout=config.output_layout,
                )

            elif crop is not None:
                batch = batch[:, :, crop, :] if batch.ndim == 4 else (batch[:, crop, :])
                if config.output_layout == "channels_first":
                    batch = np.transpose(
                        batch, (0, 1, 3, 2) if batch.ndim == 4 else (0, 2, 1)
                    )

            if config.reverse_complement:
                batch = _apply_reverse_complement(
                    batch,
                    mode=config.mode,
                    layout=config.output_layout,
                )

            writer.write(batch, start_s, end_s)
            del batch

        h5_chunks = getattr(writer, "chunks", None)
        if h5_chunks is not None:
            metadata["h5_chunks"] = list(h5_chunks)
        timings["encoding_and_write"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - t_total
        metadata["timings"] = timings
        # The sidecar is part of the same commit as the data, so it never
        # describes a run that did not finish.
        writer.commit(metadata if config.metadata_json else None)

    return metadata
