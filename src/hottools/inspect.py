# hottools/inspect.py

from __future__ import annotations
from typing import Dict, Literal, Optional
import os
import shutil
import subprocess
import logging
import tempfile

logger = logging.getLogger(__name__)

def _run_bcftools(args: list[str]) -> subprocess.CompletedProcess:
    """
    Run bcftools with given args and return Completed Process.
    """
    if shutil.which("bcftools") is None:
        raise RuntimeError("bcftools not found. Please install it or add it to PATH.")

    cmd = ["bcftools"] + args
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)
    return result

def assert_phased_genotypes(
    vcf_path: str,
    region: str,
    samples_file: Optional[str] = None,
    n_check_variants: int = 2000,
) -> None:
    """
    Raise ValueError if any non-missing GT in the queried region is unphased (contains '/').
    Checks up to n_check_variants records for speed.
    """
    if shutil.which("bcftools") is None:
        raise RuntimeError("bcftools not found. Please install it or add it to PATH.")

    fmt = "[%GT\t]\n"
    cmd = ["bcftools", "query", "-r", region, "-f", fmt]
    if samples_file is not None:
        cmd += ["-S", samples_file]
    cmd.append(vcf_path)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    checked = 0
    try:
        for line in proc.stdout:
            checked += 1
            gts = line.rstrip("\n").split("\t")

            # ignore empty lines
            if not gts or (len(gts) == 1 and gts[0] == ""):
                continue

            for gt in gts:
                gt = gt.strip()
                if gt in (".", "./.", ".|."):
                    continue
                if "/" in gt:
                    raise ValueError(
                        f"Input file must contain phased genotypes ('|') to produce separate haplotypes. "
                        f"Found unphased genotype example: {gt}"
                    )

            if checked >= n_check_variants:
                break

    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()

def inspect_vcf(
    path: str,
    n_check: int = 2000,
    region: Optional[str] = None,
) -> Dict[str, float]:
    """
    Inspect a VCF/BCF file using bcftools.
    - Counts the number of samples.
    - Looks at up to n_check variant records to estimate how many are non-SNV or multi-allelic.
    - If region is provided, file is restricted to that region.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    res = _run_bcftools(["query", "-l", path])
    sample_ids = [line for line in res.stdout.splitlines() if line]
    num_samples = len(sample_ids)

    if shutil.which("bcftools") is None:
        raise RuntimeError("bcftools not found in PATH.")

    cmd = ["bcftools", "view", "-H"]
    if region is not None:
        cmd += ["-r", region]
    cmd.append(path)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    total = 0
    non_snv_or_multiallelic = 0

    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            total += 1
            fields = line.split("\t")
            if len(fields) < 5:
                continue

            ref = fields[3]
            alt = fields[4]
            alt_alleles = alt.split(",")

            # Non-SNV if REF length != 1 or any ALT length != 1.
            # Multi-allelic if more than one ALT allele.
            if len(ref) != 1 or any(len(a) != 1 for a in alt_alleles) or len(alt_alleles) != 1:
                non_snv_or_multiallelic += 1

            if total >= n_check:
                break
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.terminate()
        proc.wait()

    if total == 0:
        frac = 0.0
    else:
        frac = non_snv_or_multiallelic / total

    if region is None:
        region_msg = "whole file"
    else:
        region_msg = f"region {region}"

    logger.info(
        "Inspection of %s (%s): %d samples, %.1f%% non-SNV or multiallelic in first %d variants",
        path,
        region_msg,
        num_samples,
        frac * 100.0,
        total,
    )

    return {
        "num_samples": num_samples,
        "frac_non_snv_or_multiallelic": frac,
    }

def has_duplicate_positions(path: str, region: Optional[str] = None) -> bool:
    """
    Fast check: stream CHROM+POS and see if any repeats.
    Assumes sorted VCF (duplicates adjacent).

    We stop early if we find a duplicate. That can cause bcftools to exit with
    SIGPIPE (-13) or SIGTERM (-15) depending on how the process is stopped.
    Those are not errors in this function.
    """
    cmd = ["bcftools", "query", "-f", "%CHROM\t%POS\n"]
    if region is not None:
        cmd += ["-r", region]
    cmd.append(path)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    prev = None
    found_dup = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            cur = line.strip()
            if cur == prev:
                found_dup = True
                break
            prev = cur
    finally:
        # If we break early, close stdout and terminate bcftools to avoid long running.
        if proc.stdout:
            proc.stdout.close()

        if found_dup:
            try:
                proc.terminate()
            except Exception:
                pass

        stderr = proc.stderr.read() if proc.stderr else ""
        rc = proc.wait()

        # If we intentionally stopped early, accept SIGPIPE/SIGTERM
        if rc != 0 and not (found_dup and rc in (-13, -15)):
            raise RuntimeError(f"bcftools query failed (exit={rc}). stderr:\n{stderr}")

    return found_dup

def drop_duplicate_positions_to_output(
    in_path: str,
    out_path: str,
    region: Optional[str] = None,
) -> None:
    """
    Write a VCF.GZ to out_path where *any* position (CHROM,POS) that appears more than once
    is completely removed (i.e., drop all records at that position).

    Assumes sorted VCF (duplicates adjacent). If region is provided, operation is restricted
    to that region (and output contains only that region as well).
    NOTE: out_path must be different from in_path (no in-place overwrite).
    """
    if os.path.abspath(in_path) == os.path.abspath(out_path):
        raise ValueError("drop_duplicate_positions_to_output requires out_path != in_path (no in-place overwrite).")

    def chrom_pos_key(line: str):
        # CHROM and POS are the first two fields
        parts = line.split("\t", 3)
        return (parts[0], parts[1]) if len(parts) >= 2 else None

    def write_line(fh, s: str):
        fh.write(s if s.endswith("\n") else s + "\n")

    # Header (region-restricted if requested)
    hdr_cmd = ["view", "-h"]
    if region is not None:
        hdr_cmd += ["-r", region]
    hdr_cmd.append(in_path)
    hdr = _run_bcftools(hdr_cmd).stdout

    # Stream body lines (region-restricted if requested)
    body_cmd = ["bcftools", "view", "-H"]
    if region is not None:
        body_cmd += ["-r", region]
    body_cmd.append(in_path)

    proc = subprocess.Popen(body_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as tmp:
            tmp_name = tmp.name
            tmp.write(hdr)

            prev_key = None
            prev_line = None
            prev_is_dup = False  # whether prev_key is duplicated

            assert proc.stdout is not None
            for line in proc.stdout:
                if not line.strip():
                    continue

                key = chrom_pos_key(line)
                if key is None:
                    continue

                if prev_key is None:
                    prev_key = key
                    prev_line = line
                    prev_is_dup = False
                    continue

                if key == prev_key:
                    # We saw the same position again => mark as duplicate
                    prev_is_dup = True
                    continue

                # key changed: decide whether to write previous
                if not prev_is_dup and prev_line is not None:
                    write_line(tmp, prev_line)

                # reset for new key
                prev_key = key
                prev_line = line
                prev_is_dup = False

            # flush last
            if prev_key is not None and not prev_is_dup and prev_line is not None:
                write_line(tmp, prev_line)

    finally:
        if proc.stdout:
            proc.stdout.close()
        stderr = proc.stderr.read() if proc.stderr else ""
        rc = proc.wait()
        if rc != 0:
            if tmp_name:
                try:
                    os.remove(tmp_name)
                except Exception:
                    pass
            raise RuntimeError(f"bcftools view failed (exit={rc}). stderr:\n{stderr}")

    # bgzip + index
    _run_bcftools(["view", "-Oz", "-o", out_path, tmp_name])
    _run_bcftools(["index", "-t", out_path])

    try:
        os.remove(tmp_name)
    except Exception:
        pass


def ensure_biallelic_snps(
    path: str,
    out_dir: Optional[str] = None,
    n_check: int = 5000,
    threshold: float = 0.01,
    region: Optional[str] = None,
) -> str:
    """
    Ensure the VCF/BCF is filtered to biallelic SNVs.
    - Runs inspect_vcf(path, region=region) to estimate fraction of non-SNV
      or multiallelic sites.
    - If fraction <= threshold, returns original path.
    - Otherwise, runs bcftools to filter:
        bcftools view -m2 -M2 -v snps [-r region] -Oz -o <out_path> <path>
        bcftools index -t <out_path>
      and returns the filtered VCF path.
    - Drops duplicate positions by removing all records at 
        any (CHROM,POS) that appears more than once.

    Returns: Path to a file guaranteed to be biallelic SNVs.
    If region is provided, the output will also be restricted to that region.
    """

    # Output name (if new file is written)
    if out_dir is None:
        out_dir = os.path.dirname(path) or "."

    base = os.path.basename(path)
    stem = base

    for suffix in [".vcf.gz", ".vcf.bgz", ".vcf", ".bcf"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    if region is None:
        out_base = stem + ".biallelic_snps.vcf.gz"
    else:
        safe_region = region.replace(":", "_").replace("-", "_")
        out_base = stem + f".{safe_region}.biallelic_snps.vcf.gz"

    out_path = os.path.join(out_dir, out_base)

    # 1) Inspect and decide if SNP/biallelic filter is needed    
    info = inspect_vcf(path, n_check=n_check, region=region)
    frac = info["frac_non_snv_or_multiallelic"]

    # 2) Either keep original or create filtered file
    if frac <= threshold:
        if has_duplicate_positions(path, region=region):
            logger.info(
                "VCF contains duplicate positions. Dropping all records at duplicated (CHROM,POS) sites.",
            )
            drop_duplicate_positions_to_output(path, out_path, region=region)
            return out_path
        else: # returns original file 
            logger.info(
                "VCF appears to be only biallelic SNVs. Using original file.",
            )
            return path 
    
    # Filtering is needed
    logger.warning(
        "VCF contains non-SNV or multiallelic variants. Filtering to biallelic SNVs with bcftools.",
    )

    args = ["view", "-m2", "-M2", "-v", "snps"]
    if region:
        args += ["-r", region]
    args += ["-Oz", "-o", out_path, path]

    # >> Run bcftools
    _run_bcftools(args)
    _run_bcftools(["index", "-t", out_path]) # Index by tabix

    # After filtering, check for duplicates
    if has_duplicate_positions(out_path):
        logger.info(
            "VCF contains duplicate positions. Dropping all records at duplicated (CHROM,POS) sites.",
        )
        tmp_out = out_path + ".tmp_dropdup.vcf.gz"
        drop_duplicate_positions_to_output(out_path, tmp_out, region=None)

        os.replace(tmp_out, out_path)
        if os.path.exists(tmp_out + ".tbi"):
            os.replace(tmp_out + ".tbi", out_path + ".tbi")

    logger.info("Filtered VCF written to %s", out_path)

    return out_path

    