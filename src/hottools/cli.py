from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import RunConfig, run_hottools
from .reference import ReferenceReader
from .regions import parse_region, read_region_file, region_from_center
from .writers import previous_run_outputs


def _parse_batch_size(value: str):
    lower = value.lower()
    if lower in {"auto", "none"}:
        return lower
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "batch size must be auto, none, or a positive integer"
        ) from exc
    if number < 1:
        raise argparse.ArgumentTypeError("batch size must be >= 1")
    return number


def _load_samples(args) -> list[str] | None:
    if args.samples and args.sample_file:
        raise ValueError("Use either --samples or --sample-file, not both.")

    if args.samples:
        names = [x.strip() for x in args.samples.split(",") if x.strip()]
        if not names:
            raise ValueError("--samples did not contain any sample IDs.")
        return names

    if args.sample_file:
        with open(args.sample_file, encoding="utf-8") as handle:
            names = [
                line.strip()
                for line in handle
                if line.strip() and not line.startswith("#")
            ]
        if not names:
            raise ValueError(f"No sample IDs found in {args.sample_file}")
        return names
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hottools",
        description=(
            "Fast individualized one-hot DNA sequence generation "
            "directly from indexed VCF/BCF."
        ),
    )

    parser.add_argument("vcf", help="Indexed VCF.gz or BCF")
    parser.add_argument("--reference", required=True, help="Reference FASTA")
    parser.add_argument("-o", "--output", required=True)

    region_group = parser.add_mutually_exclusive_group(required=True)
    region_group.add_argument("--region", help="CHROM:START-END, 1-based inclusive")
    region_group.add_argument(
        "--region-file",
        help=(
            "One region per line. --output is treated as a directory. "
            "See --region-file-format."
        ),
    )
    region_group.add_argument(
        "--center",
        help="CHROM:POS; requires --length",
    )
    parser.add_argument(
        "--length",
        type=int,
        help="Window length in bp; only valid with --center",
    )
    parser.add_argument(
        "--region-file-format",
        choices=["auto", "1based", "bed"],
        default="auto",
        help=(
            "Coordinate convention for --region-file. '1based' is "
            "CHROM START END [STRAND], fully closed (same as --region); "
            "'bed' is 0-based half-open with strand in column 6. "
            "'auto' picks 'bed' for .bed/.bed.gz and '1based' otherwise."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=["dosage", "haplotypes", "consensus"],
        default="dosage",
        help=(
            "dosage: S x L x 4 with heterozygotes split 0.5/0.5. "
            "haplotypes: S x 2 x L x 4. "
            "consensus: S x L x 4 carrying one allele per sample, "
            "heterozygotes collapsed per --het-policy."
        ),
    )
    parser.add_argument(
        "--het-policy",
        choices=["reference", "alternate", "major"],
        default="alternate",
        help=(
            "--mode consensus only: which allele to use at heterozygous "
            "sites. 'major' uses the more frequent allele among the "
            "selected samples."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["npy", "npz", "h5", "pt"],
        default="npy",
    )
    parser.add_argument("--per-sample", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float32",
    )
    parser.add_argument(
        "--output-layout",
        choices=["samples_first", "channels_first"],
        default="samples_first",
    )

    parser.add_argument(
        "--batch-size",
        type=_parse_batch_size,
        default="auto",
        help="auto, none, or positive integer",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.40,
        help="Fraction of currently available RAM usable by one output batch.",
    )
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--contig-alias-policy", choices=["exact", "auto"], default="exact",
        help="Require exact contig names (default), or resolve unique standard aliases."
    )

    parser.add_argument("--samples", help="Comma-separated VCF sample IDs")
    parser.add_argument("--sample-file", help="One VCF sample ID per line")

    parser.add_argument(
        "--missing-policy",
        choices=["zero", "reference", "error"],
        default="zero",
    )
    parser.add_argument(
        "--unphased-policy",
        choices=["error", "missing", "random"],
        default="error",
        help=("Unphased heterozygotes in haplotype mode: error, missing, "
              "or seeded random pseudo-phasing."),
    )
    parser.add_argument(
        "--random-seed", type=int,
        help="Required with --unphased-policy random; integer from 0 through 2**64-1.",
    )
    parser.add_argument(
        "--ref-mismatch-policy",
        choices=["error", "skip", "warn"],
        default="error",
    )
    parser.add_argument(
        "--ambiguous-ref-policy",
        choices=["error", "skip", "warn"],
        default="error",
    )
    parser.add_argument(
        "--haploid-policy",
        choices=["duplicate", "blank", "error", "skip"],
        default="duplicate",
        help=(
            "What the second haplotype track carries for a haploid call "
            "(chrX/chrY/chrM). 'duplicate' copies the single allele onto "
            "it, 'blank' leaves it empty; 'error'/'skip' reject the "
            "record. Dosage and consensus are ploidy-aware and are not "
            "affected by duplicate/blank. Either way a message is printed "
            "and per-sample ploidy is recorded in the metadata."
        ),
    )
    parser.add_argument(
        "--ploidy-policy",
        choices=["error", "skip"],
        default="error",
        help="What to do with records whose ploidy is greater than two.",
    )

    site = parser.add_argument_group("site filters")
    site.add_argument(
        "--pass-only",
        action="store_true",
        help="Keep only records whose FILTER column is PASS or '.'.",
    )
    site.add_argument(
        "--min-qual",
        type=float,
        help="Drop records whose QUAL is below this value, or absent.",
    )
    site.add_argument(
        "--min-af",
        type=float,
        help="Drop records whose alternate allele frequency is below this.",
    )
    site.add_argument(
        "--max-af",
        type=float,
        help="Drop records whose alternate allele frequency is above this.",
    )
    site.add_argument(
        "--af-source",
        choices=["auto", "info", "samples"],
        default="auto",
        help=(
            "Where allele frequency comes from: the INFO/AF field, the "
            "selected samples, or 'auto' (INFO/AF when the record has it, "
            "otherwise the samples)."
        ),
    )
    classes = parser.add_argument_group(
        "variant classes",
        "By default only single-nucleotide substitutions are applied. "
        "Length-changing classes require --mode haplotypes.",
    )
    classes.add_argument(
        "--variant-classes",
        default="snv",
        metavar="LIST",
        help=(
            "Comma-separated variant classes to apply: snv, mnv, del, ins, "
            "symbolic-del. Shorthands: 'indel' = del,ins; 'all' = every "
            "supported class. SNVs are always included. Default: snv."
        ),
    )
    classes.add_argument(
        "--biallelic-only",
        action="store_true",
        help="Exclude records with more than one ALT allele.",
    )
    classes.add_argument(
        "--mixed-record-policy",
        choices=("exclude", "apply-selected"),
        default="exclude",
        help=(
            "Exclude records containing unrequested classes (default), "
            "or apply only selected alleles."
        ),
    )
    classes.add_argument(
        "--overlap-policy",
        choices=("first", "error"),
        default="first",
        help=(
            "Records that claim the same reference bases on one haplotype: "
            "'first' applies the first in POS order and sets the rest aside, "
            "as bcftools consensus does (default); 'error' refuses the run "
            "before anything is written."
        ),
    )
    classes.add_argument(
        "--anchor-deletion-policy",
        choices=("right", "error"),
        default="right",
        help=(
            "When a deletion removes the anchor base itself: 'right' pins "
            "the first surviving base after it (default); 'error' refuses "
            "the run."
        ),
    )
    classes.add_argument(
        "--anchor",
        default="start",
        metavar="SPEC",
        help=(
            "Reference position pinned to a fixed output offset once "
            "length-changing edits move sequence around: 'start' (default), "
            "'center', or a 1-based coordinate inside the region."
        ),
    )

    site.add_argument(
        "--mask",
        help=(
            "BED (or 1-based) file of intervals to blank out: their "
            "reference channels are zeroed and variants there are dropped."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=("Escalate empty-region and split-multiallelic warnings to errors."),
    )

    parser.add_argument(
        "--strand",
        choices=["+", "-"],
        default="+",
        help=(
            "Output orientation; '-' reverse-complements. Applies to "
            "regions that do not carry their own strand column."
        ),
    )

    parser.add_argument(
        "--h5-compression",
        choices=["gzip", "lzf", "none"],
        default="none",
    )
    parser.add_argument(
        "--h5-chunk-samples",
        type=int,
        help="Override the automatic HDF5 chunk size along the sample axis.",
    )
    parser.add_argument(
        "--h5-chunk-length",
        type=int,
        help="Override the automatic HDF5 chunk size along the sequence axis.",
    )

    parser.add_argument(
        "--overwrite",
        choices=["error", "replace"],
        default="error",
        help=(
            "What to do when an output file already exists. The default "
            "refuses rather than silently replacing a previous run."
        ),
    )
    parser.add_argument(
        "--no-metadata-json",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Inspect/filter VCF and report expected output/memory "
            "without constructing sequence tensors."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the full traceback instead of a one-line error.",
    )

    return parser


def _regions_from_args(args):
    """Resolve the region selection, with --strand as the default strand."""
    if args.region:
        if args.length is not None:
            raise ValueError("--length is only valid together with --center")
        return [parse_region(args.region, args.strand)]
    if args.region_file:
        if args.length is not None:
            raise ValueError("--length is only valid together with --center")
        return read_region_file(
            args.region_file,
            default_strand=args.strand,
            coordinate_system=args.region_file_format,
        )
    if args.center:
        if args.length is None:
            raise ValueError("--center requires --length")
        return [region_from_center(args.center, args.length, args.strand)]
    raise ValueError("No region supplied.")


def _output_path_for(args, region, as_directory: bool) -> Path:
    if not as_directory:
        return Path(args.output)
    # Not created here: a run that is refused must not leave even an
    # empty directory behind. The writer creates it when it commits.
    out_dir = Path(args.output)
    safe = f"{region.chrom}_{region.start}_{region.end}".replace("/", "_").replace(
        ":", "_"
    )
    if region.strand == "-":
        safe += "_minus"
    if args.per_sample:
        return out_dir / safe
    return out_dir / f"{safe}.{args.format}"


def _preflight_regions(args, regions, outputs) -> None:
    """
    Check every region's output before the first region runs.

    Each region is its own transaction, so without this a region file
    whose third region conflicts would stop after committing the first
    two. Two regions that map to the same output name would also
    overwrite each other.
    """
    seen: dict[str, object] = {}
    for region, output in zip(regions, outputs, strict=True):
        key = str(output).casefold()
        if key in seen:
            raise ValueError(
                f"Regions {seen[key].label} and {region.label} would both "
                f"write {output}; remove the duplicate line from the "
                "region file."
            )
        seen[key] = region
    if args.overwrite != "error":
        return
    conflicts = [
        path
        for output in outputs
        for path in previous_run_outputs(output, args.format, args.per_sample)
    ]
    if conflicts:
        shown = ", ".join(str(p) for p in conflicts[:5])
        more = f" and {len(conflicts) - 5} more" if len(conflicts) > 5 else ""
        raise FileExistsError(
            f"Output already exists: {shown}{more}. No region was written; "
            "pass --overwrite replace to replace the earlier run, or "
            "choose another output path."
        )


def run(args) -> list[dict]:
    """Execute a parsed argument namespace; raises on any failure."""
    samples = _load_samples(args)
    regions = _regions_from_args(args)

    h5_compression = None if args.h5_compression == "none" else args.h5_compression
    # A region file always writes into a directory, even when it holds a
    # single region, so the output layout does not change with its length.
    as_directory = bool(args.region_file)
    outputs = [_output_path_for(args, r, as_directory) for r in regions]
    _preflight_regions(args, regions, outputs)

    configs = []
    for region, output in zip(regions, outputs, strict=True):
        config = RunConfig(
            vcf_path=args.vcf,
            fasta_path=args.reference,
            region=region,
            output_path=str(output),
            mode=args.mode,
            output_format=args.format,
            per_sample=args.per_sample,
            dtype=args.dtype,
            output_layout=args.output_layout,
            batch_size=args.batch_size,
            memory_fraction=args.memory_fraction,
            threads=args.threads,
            contig_alias_policy=args.contig_alias_policy,
            samples=samples,
            missing_policy=args.missing_policy,
            unphased_policy=args.unphased_policy,
            random_seed=args.random_seed,
            ref_mismatch_policy=args.ref_mismatch_policy,
            ambiguous_ref_policy=args.ambiguous_ref_policy,
            haploid_policy=args.haploid_policy,
            ploidy_policy=args.ploidy_policy,
            variant_classes=args.variant_classes,
            biallelic_only=args.biallelic_only,
            mixed_record_policy=args.mixed_record_policy,
            anchor=args.anchor,
            overlap_policy=args.overlap_policy,
            anchor_deletion_policy=args.anchor_deletion_policy,
            het_policy=args.het_policy,
            pass_only=args.pass_only,
            min_qual=args.min_qual,
            min_af=args.min_af,
            max_af=args.max_af,
            af_source=args.af_source,
            mask_path=args.mask,
            h5_compression=h5_compression,
            h5_chunk_samples=args.h5_chunk_samples,
            h5_chunk_length=args.h5_chunk_length,
            metadata_json=not args.no_metadata_json,
            dry_run=args.dry_run,
            strict=args.strict,
            overwrite=args.overwrite,
            reference_reader=None,
        )
        config.validate()
        configs.append(config)
    all_results = []
    with ReferenceReader(args.reference) as reader:
        for config in configs:
            config.reference_reader = reader
            all_results.append(run_hottools(config))
    return all_results


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        all_results = run(args)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        if args.debug:
            raise
        print(f"hottools: error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            all_results[0] if len(all_results) == 1 else all_results,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
