from __future__ import annotations

import argparse
import logging
import sys

from .core import run_hottools
from .inspect import ensure_biallelic_snps

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

def _add_common_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-l", "--log-level",
        choices=LOG_LEVELS.keys(),
        default="INFO",
        help="Logging level (default: INFO).",
    )

# =========================
# Subcommand: run
# =========================

def build_run_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Reconstruct and one-hot encode a region (biallelic SNVs only).",
        description=(
            "Run Hottools on a genomic region and write one-hot encoded sequences.\n\n"
            "Variant support:\n"
            "  • biallelic single-nucleotide variants (SNVs) only (REF/ALT length == 1)\n"
            "  • indels and multiallelic sites are excluded during preprocessing\n\n"
            "Preprocessing:\n"
            "  • filters input to biallelic SNVs using bcftools (if needed)\n"
            "  • detects duplicate positions and keeps the record with lowest missingness\n\n"
            "Genotype encoding:\n"
            "  • haplotype_mode=average (default): dosage-based mixture of REF/ALT\n"
            "  • haplotype_mode=separate: writes separate haplotypes (requires phased GT '|')\n\n"
            "Output shapes:\n"
            "  • average: (S, L, 4)\n"
            "  • separate: (S, 2, L, 4)\n\n"
            "Examples:\n"
            "  hottools run --vcf input.bcf --region 12:31000000-31200000 --fasta hg38.fa \\\n"
            "      --out-dir out --prefix chr12 --format npz\n\n"
            "  hottools run --vcf input.vcf.gz --samples samples.txt --region chr21:33000000-33100000 \\\n"
            "      --fasta hg38.fa --haplotype-mode separate --format hdf5 --max-memory-gb 8\n"
        ),
        epilog=(
            "Notes:\n"
            "  • Requires bcftools on PATH.\n"
            "  • --format hdf5 requires h5py.\n"
            "  • --format pt requires torch.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Input VCF/BCF
    p.add_argument(
        "-i", "--vcf",
        type=str,
        required=True,
        help="Input VCF/BCF file (.vcf, .vcf.gz, .bcf). Requires bcftools on PATH.",
    )
    p.add_argument(
        "-s", "--samples",
        type=str,
        dest="samples_file",
        help="Optional text file with sample IDs (one per line) to subset and set ordering.",
    )
    p.add_argument(
        "-r", "--region",
        type=str,
        required=True,
        help="Genomic region to process: 'chr:start-end' (e.g. chr12:31000000-31200000). "
             "Either 'chr12' or '12' is accepted (auto-normalized to match contigs).",
    )

    # Reference genome
    p.add_argument(
        "-f", "--fasta",
        type=str,
        required=True,
        help="Reference FASTA (e.g. GRCh38.fa). Must be indexed with .fai.",
    )

    # Genotype interpretation
    p.add_argument(
        "-H", "--haplotype-mode",
        choices=["average", "separate"],
        default="average",
        help="average: one sequence per sample (default). separate: output two haplotypes; requires phased GT '|'.",
    )

    # Batching / memory
    p.add_argument(
        "-b", "--batch-size",
        type=int,
        default=None,
        help="Samples per batch. If unset, Hottools tries full run unless --max-memory-gb is provided or OOM fallback is enabled.",
    )
    p.add_argument(
        "-m", "--max-memory-gb",
        type=float,
        default=None,
        help="Approx memory budget (GB) for automatic batch size estimation (proactive or as OOM fallback).",
    )
    p.add_argument(
        "--auto-batch-on-oom",
        dest="auto_batch_on_oom",
        action="store_true",
        default=True,
        help="On OOM, automatically retry with computed batching (default: enabled).",
    )
    p.add_argument(
        "--no-auto-batch-on-oom",
        dest="auto_batch_on_oom",
        action="store_false",
        help="Disable automatic retry with batching on OOM.",
    )
    p.add_argument(
        "--print-estimates",
        dest="print_estimates",
        action="store_true",
        default=True,
        help="Log estimated output array size before encoding (default: enabled).",
    )
    p.add_argument(
        "--no-print-estimates",
        dest="print_estimates",
        action="store_false",
        help="Disable logging of size estimates.",
    )

    # Output
    p.add_argument(
        "-o", "--out-dir",
        type=str,
        default=".",
        help="Output directory (default: current directory).",
    )
    p.add_argument(
        "-p", "--prefix",
        type=str,
        default="hottools",
        help="Prefix for output files (default: hottools).",
    )
    p.add_argument(
        "-F", "--format",
        dest="fmt",
        choices=["npy", "npz", "hdf5", "pt"],
        default="npy",
        help="Output format: npy/npz/hdf5/pt (default: npy). Note: hdf5 requires h5py; pt requires torch.",
    )
    p.add_argument(
        "--per-sample",
        action="store_true",
        help="Write one file per sample (and per haplotype if separate) instead of a combined array.",
    )
    p.add_argument(
        "-d", "--dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Output floating point precision (default: float32).",
    )

    _add_common_logging_args(p)
    p.set_defaults(func=main_run)

def main_run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=LOG_LEVELS[args.log_level], format="[%(levelname)s] %(message)s")

    run_hottools(
        vcf=args.vcf,
        samples_file=args.samples_file,
        region=args.region,
        fasta=args.fasta,
        haplotype_mode=args.haplotype_mode,
        batch_size=args.batch_size,
        max_memory_gb=args.max_memory_gb,
        auto_batch_on_oom=args.auto_batch_on_oom,
        print_estimates=args.print_estimates,
        out_dir=args.out_dir,
        prefix=args.prefix,
        fmt=args.fmt,
        per_sample=args.per_sample,
        dtype=args.dtype,
    )

# =========================
# Subcommand: inspect
# =========================

def build_inspect_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "inspect",
        help="Inspect a VCF/BCF and optionally write a biallelic-SNV-only VCF.",
        description=(
            "Inspect a VCF/BCF using bcftools and report basic properties.\n\n"
            "Variant support (for Hottools encoding): biallelic SNVs only.\n"
            "Non-SNV and multiallelic sites are not supported for encoding.\n\n"
            "This command can:\n"
            "  • estimate fraction of non-SNV/multiallelic records\n"
            "  • detect duplicate positions (CHROM,POS)\n"
            "  • write a filtered VCF restricted to biallelic SNVs (optionally in a region)\n"
            "  • deduplicate by keeping the record with lowest missingness\n\n"
            "Examples:\n"
            "  hottools inspect --vcf input.bcf\n"
            "  hottools inspect --vcf input.vcf.gz --region chr12:31000000-31200000 --out-dir out\n"
        ),
        epilog=(
            "Notes:\n"
            "  • Requires bcftools on PATH.\n"
            "  • Filtering writes a new .vcf.gz file with tabix index.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument("-i", "--vcf", type=str, required=True, help="Input VCF/BCF file (.vcf, .vcf.gz, .bcf).")
    p.add_argument("-r", "--region", type=str, default=None, help="Optional region 'chr:start-end' to restrict inspection/filtering.")
    p.add_argument("-n", "--n-check", type=int, default=2000, help="Number of variant records to inspect (default: 2000).")
    p.add_argument("-t", "--threshold", type=float, default=0.01, help="Filter if fraction non-SNV/multiallelic > threshold (default: 0.01).")
    p.add_argument("-o", "--out-dir", type=str, default=None, help="Directory for filtered VCF (default: same directory as input).")

    _add_common_logging_args(p)
    p.set_defaults(func=main_inspect)

def main_inspect(args: argparse.Namespace) -> None:
    logging.basicConfig(level=LOG_LEVELS[args.log_level], format="[%(levelname)s] %(message)s")

    final_path = ensure_biallelic_snps(
        path=args.vcf,
        out_dir=args.out_dir,
        n_check=args.n_check,
        threshold=args.threshold,
        region=args.region,
    )
    print(final_path)

# =========================
# Top-level entry point
# =========================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hottools",
        description=(
            "Hottools: Fast reconstruction and one-hot encoding of personalized genomic sequences from VCF/BCF.\n\n"
            "⚠ Supported variants: biallelic SNVs only. Indels and multiallelic sites are filtered.\n"
            "Requires bcftools to be installed and available on PATH."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="subcommand",
        metavar="<command>",
        required=True,
    )
    build_run_parser(subparsers)
    build_inspect_parser(subparsers)
    return parser

def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
