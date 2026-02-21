# hottools/core.py

from __future__ import annotations
from typing import Optional, Literal, List, Tuple
import os
import shutil
import subprocess
import logging
import math

import numpy as np
import pandas as pd
from pyfaidx import Fasta

from .inspect import ensure_biallelic_snps, assert_phased_genotypes

logger = logging.getLogger(__name__)

# =========================
# bcftools helpers
# =========================

def _run_bcftools(args: list[str]) -> subprocess.CompletedProcess:
    """
    Run bcftools with given args and return CompletedProcess.
    """
    if shutil.which("bcftools") is None:
        raise RuntimeError("bcftools not found in PATH. Please install it or add it to PATH.")

    cmd = ["bcftools"] + args
    return subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def _get_sample_ids(vcf_path: str, samples_file: Optional[str]) -> List[str]:
    """
    Return sample IDs in the exact order that will be used:
      - If samples_file is provided: that file's order (bcftools -S respects it).
      - Else: VCF/BCF header order.
    """
    if samples_file is not None:
        ids: List[str] = []
        with open(samples_file) as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                ids.append(s)
        return ids

    res = _run_bcftools(["query", "-l", vcf_path])
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def vcf_uses_chr_prefix(vcf_path: str) -> bool:
    """
    Returns True if the VCF/BCF uses 'chrN' style contigs,
    False if it uses 'N' style.
    """
    cmd = ["bcftools", "view", "-H", vcf_path]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

    for line in proc.stdout:
        if line.strip():
            chrom = line.split("\t")[0]
            proc.terminate()
            return chrom.startswith("chr")

    proc.terminate()
    raise ValueError("No variant lines found in VCF.")


def normalize_chrom_to_vcf(chrom: str, vcf_path: str) -> str:
    """
    Normalize a user-provided chromosome string to match the contig naming
    convention used in the VCF/BCF.
    Accepts inputs like '12' or 'chr12' and returns the VCF-matching form.
    """
    uses_chr = vcf_uses_chr_prefix(vcf_path)
    chrom = chrom.strip()

    if uses_chr and not chrom.startswith("chr"):
        return f"chr{chrom}"
    if (not uses_chr) and chrom.startswith("chr"):
        return chrom[3:]
    return chrom


def _first_fasta_contig_name(fasta_path: str) -> str:
    """
    Return the first contig name in a FASTA, preferring the .fai index if present.
    This is O(1) I/O in practice (reads 1 line), and avoids loading the FASTA.
    """
    fai = fasta_path + ".fai"
    if os.path.exists(fai):
        with open(fai, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    return line.split("\t", 1)[0]
        raise ValueError(f"FASTA index exists but is empty: {fai}")

    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                return line[1:].strip().split()[0]

    raise ValueError(f"No FASTA header lines found in: {fasta_path}")


def normalize_chrom_to_fasta(
    chrom: str,
    fasta_path: str,
    *,
    allow_mito_aliases: bool = True,
) -> str:
    """
    Normalize a user-provided chrom string (e.g. '12' or 'chr12') to match the FASTA naming
    convention (chr-prefixed or not), using only the FASTA's first contig header (or .fai).

    This is intentionally lightweight: it infers the *style* (chr vs no-chr) rather than
    scanning all contigs.

    If allow_mito_aliases=True, it will also map M<->MT (and chrM<->chrMT) when needed.
    """
    chrom = chrom.strip()
    first = _first_fasta_contig_name(fasta_path)
    fasta_has_chr = first.startswith("chr")

    # 1) Normalize chr prefix
    if fasta_has_chr and not chrom.startswith("chr"):
        chrom = "chr" + chrom
    elif (not fasta_has_chr) and chrom.startswith("chr"):
        chrom = chrom[3:]

    # 2) Optional mitochondria alias normalization
    if allow_mito_aliases:
        mito_map = {"M": "MT", "MT": "M"}
        # apply after prefix normalization
        if chrom.startswith("chr"):
            base = chrom[3:]
            if base in mito_map:
                chrom = "chr" + mito_map[base]
        else:
            if chrom in mito_map:
                chrom = mito_map[chrom]

    return chrom


# =========================
# memory / batching helpers
# =========================

def estimate_onehot_bytes(
    num_samples: int,
    seq_len: int,
    dtype: Literal["float32", "float16"],
    haplotype_mode: Literal["average", "separate"],
) -> int:
    bytes_per = 4 if dtype == "float32" else 2
    H = 2 if haplotype_mode == "separate" else 1
    return num_samples * H * seq_len * 4 * bytes_per  # 4 channels


def bytes_to_gb(n_bytes: int) -> float:
    return n_bytes / (1024 ** 3)


def detect_available_memory_gb(default_gb: float = 8.0) -> float:
    """
    Best-effort. If psutil exists, use available RAM; otherwise fall back to default.
    """
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return default_gb


def plan_batch_size_from_budget(
    num_samples: int,
    seq_len: int,
    dtype: Literal["float32", "float16"],
    haplotype_mode: Literal["average", "separate"],
    max_memory_gb: float,
    overhead_factor: float = 2.5,
) -> int:
    """
    Heuristic batch planner:
      (batch_size * per_sample_bytes * overhead_factor) <= budget_bytes
    overhead_factor accounts for temporary arrays/broadcasting.
    """
    bytes_per = 4 if dtype == "float32" else 2
    H = 2 if haplotype_mode == "separate" else 1
    per_sample_bytes = H * seq_len * 4 * bytes_per
    budget_bytes = max_memory_gb * (1024 ** 3)

    if per_sample_bytes <= 0:
        return 1

    bs = int(budget_bytes // max(1.0, per_sample_bytes * overhead_factor))
    bs = max(1, min(num_samples, bs))
    return bs


# =========================
# GT parsing helpers
# =========================

def _gt_str_to_dosage(gt_str: str) -> float:
    """
    Convert a GT string like '0/0', '0/1', '1|1', './.' into alt dosage.
    - REF/REF → 0.0
    - REF/ALT or ALT/REF → 0.5
    - ALT/ALT → 1.0
    - MISSING → 0.0
    """
    gt_str = gt_str.strip()
    if not gt_str or gt_str in (".", "./.", ".|."):
        return 0.0

    sep = "|" if "|" in gt_str else "/"
    parts = gt_str.split(sep)
    if len(parts) != 2:
        return 0.0

    a1s, a2s = parts
    if a1s == "." or a2s == ".":
        return 0.0

    try:
        a1 = int(a1s)
        a2 = int(a2s)
    except ValueError:
        return 0.0

    alt_count = a1 + a2  # 0, 1, or 2 (biallelic)
    if alt_count <= 0:
        return 0.0
    if alt_count == 1:
        return 0.5
    return 1.0


def _gt_str_to_haps(gt_str: str) -> Tuple[float, float]:
    """
    Convert a phased GT string like '0|1' into per-haplotype alt counts (0.0/1.0).
    Unphased/missing → (0.0, 0.0).
    """
    gt_str = gt_str.strip()
    if not gt_str or gt_str in (".", "./.", ".|."):
        return 0.0, 0.0
    if "|" not in gt_str:
        return 0.0, 0.0

    a1s, a2s = gt_str.split("|")
    if a1s == "." or a2s == ".":
        return 0.0, 0.0

    try:
        a1 = int(a1s)
        a2 = int(a2s)
    except ValueError:
        return 0.0, 0.0

    return (1.0 if a1 == 1 else 0.0), (1.0 if a2 == 1 else 0.0)


# =========================
# Variant extraction
# =========================

def extract_region_with_bcftools(
    vcf_path: str,
    region: str,
    sample_ids: List[str],
    samples_file: Optional[str],
    haplotype_mode: Literal["average", "separate"] = "average",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Use bcftools query to extract variants and per-sample (or per-haplotype) dosages.

    Returns:
      - positions: (V,) int
      - ref_bases: (V,) 'U1'
      - alt_bases: (V,) 'U1'
      - dosage_data:
          * average  -> (V, S)
          * separate -> (2, V, S)
    """
    S = len(sample_ids)
    positions: List[int] = []
    ref_bases: List[str] = []
    alt_bases: List[str] = []

    dosage_rows: List[List[float]] = []
    hap1_rows: List[List[float]] = []
    hap2_rows: List[List[float]] = []

    if shutil.which("bcftools") is None:
        raise RuntimeError("bcftools not found in PATH.")

    fmt = "%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n"
    cmd = ["bcftools", "query", "-r", region, "-f", fmt]
    if samples_file is not None:
        cmd += ["-S", samples_file]
    cmd.append(vcf_path)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            fields = line.split("\t")
            if len(fields) < 5:
                continue

            pos = int(fields[1])
            ref = fields[2]
            alt = fields[3]

            alt_alleles = alt.split(",")
            if len(ref) != 1 or any(len(a) != 1 for a in alt_alleles) or len(alt_alleles) != 1:
                continue

            gts = fields[4:]
            if len(gts) != S:
                raise ValueError(
                    f"Mismatch between sample count ({S}) and GT fields ({len(gts)}) "
                    f"in bcftools output for region {region}."
                )

            if haplotype_mode == "average":
                dosage_rows.append([_gt_str_to_dosage(gt) for gt in gts])
            else:
                h1_row: List[float] = []
                h2_row: List[float] = []
                for gt in gts:
                    a1, a2 = _gt_str_to_haps(gt)
                    h1_row.append(a1)
                    h2_row.append(a2)
                hap1_rows.append(h1_row)
                hap2_rows.append(h2_row)

            positions.append(pos)
            ref_bases.append(ref)
            alt_bases.append(alt_alleles[0])

    finally:
        if proc.stdout:
            proc.stdout.close()

        stderr_txt = proc.stderr.read() if proc.stderr else ""
        ret = proc.wait()

    if ret != 0:
        raise RuntimeError(
            "bcftools query failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr:\n{stderr_txt.strip()}"
        )


    if not positions:
        raise ValueError(f"No variants found in region {region} after filtering.")

        

    positions_arr = np.array(positions, dtype=int)
    ref_arr = np.array(ref_bases, dtype="U1")
    alt_arr = np.array(alt_bases, dtype="U1")

    if haplotype_mode == "average":
        dosage_arr = np.array(dosage_rows, dtype=np.float32)  # (V, S)
        return positions_arr, ref_arr, alt_arr, dosage_arr

    hap1 = np.array(hap1_rows, dtype=np.float32)              # (V, S)
    hap2 = np.array(hap2_rows, dtype=np.float32)              # (V, S)
    hap_dosage_arr = np.stack([hap1, hap2], axis=0)           # (2, V, S)
    return positions_arr, ref_arr, alt_arr, hap_dosage_arr


# =========================
# Base encoding
# =========================

BASE2VEC = {
    "A": np.array([1, 0, 0, 0], dtype=np.float32),
    "C": np.array([0, 1, 0, 0], dtype=np.float32),
    "G": np.array([0, 0, 1, 0], dtype=np.float32),
    "T": np.array([0, 0, 0, 1], dtype=np.float32),
}

def encode_base(base: str, dtype: np.dtype) -> np.ndarray:
    base = base.upper()
    if base == "N":
        base = "A"
    return BASE2VEC.get(base, BASE2VEC["A"]).astype(dtype)


def encode_bases(bases: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return np.stack([encode_base(b, dtype) for b in bases], axis=0)


# =========================
# Reference sequence
# =========================

def load_reference_window(
    fasta_path: str,
    chrom: str,
    start_1based: int,
    end_1based: int,
) -> np.ndarray:
    ref = Fasta(fasta_path)
    start0 = start_1based - 1
    seq = ref[chrom][start0:end_1based].seq
    return np.array(list(seq), dtype="U1")


# =========================
# Output writers
# =========================

def save_onehot(
    seq: np.ndarray,            # (S,L,4) or (S,2,L,4)
    sample_ids: List[str],
    out_dir: str,
    prefix: str,
    fmt: Literal["npy", "npz", "hdf5", "pt"],
    compress: Optional[Literal["gzip", "lzf"]] = "gzip",
    per_sample: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    if seq.ndim == 3:
        S, L, C = seq.shape
        H = 1
    elif seq.ndim == 4:
        S, H, L, C = seq.shape
    else:
        raise ValueError(f"Unexpected seq.ndim={seq.ndim}; expected 3 or 4.")

    # per-sample mode
    if per_sample:
        for i, sid in enumerate(sample_ids):
            if H == 1:
                suffixes = [sid]
                arrays = [seq[i]]
            else:
                suffixes = [f"{sid}_hap1", f"{sid}_hap2"]
                arrays = [seq[i, 0], seq[i, 1]]

            for suf, arr in zip(suffixes, arrays):
                base_prefix = f"{prefix}.{suf}"
                if fmt == "npy":
                    np.save(os.path.join(out_dir, f"{base_prefix}.onehot.npy"), arr)
                elif fmt == "pt":
                    import torch
                    torch.save(torch.from_numpy(arr), os.path.join(out_dir, f"{base_prefix}.onehot.pt"))
                elif fmt == "npz":
                    np.savez_compressed(os.path.join(out_dir, f"{base_prefix}.onehot.npz"), onehot=arr)
                else:
                    raise NotImplementedError(f"per-sample not implemented for format {fmt}")
        return

    # combined mode
    if fmt == "npy":
        np.save(os.path.join(out_dir, f"{prefix}.onehot.npy"), seq)
    elif fmt == "npz":
        np.savez_compressed(
            os.path.join(out_dir, f"{prefix}.onehot.npz"),
            onehot=seq,
            sample_ids=np.array(sample_ids),
        )
    elif fmt == "pt":
        try:
            import torch
        except ImportError:
            raise ImportError(
                "Saving in PyTorch format requires the 'torch' package, "
                "but it is not installed in the current environment."
            )
        torch.save(
            {"onehot": torch.from_numpy(seq), "sample_ids": np.array(sample_ids)}, 
            os.path.join(out_dir, f"{prefix}.onehot.pt")
            )
    elif fmt == "hdf5":
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "Saving in HDF5 format requires the 'h5py' package, "
                "but it is not installed in the current environment."
            )
        with h5py.File(os.path.join(out_dir, f"{prefix}.onehot.h5"), "w") as h5:
            h5.create_dataset("onehot", data=seq, compression=compress or "gzip")
            h5.create_dataset("sample_ids", data=np.array(sample_ids, dtype="S"))
    else:
        raise ValueError(f"Unsupported format: {fmt}")


# =========================
# Main entry point
# =========================

def run_hottools(
    # Input VCF/BCF
    vcf: Optional[str] = None,
    samples_file: Optional[str] = None,
    region: Optional[str] = None,

    # Reference genome
    fasta: str = "",
    fai: Optional[str] = None,   # pyfaidx uses .fai automatically

    # Genotype interpretation
    haplotype_mode: Literal["average", "separate"] = "average",

    # Batching / memory
    batch_size: Optional[int] = None,            # explicit override
    max_memory_gb: Optional[float] = None,       # proactive auto-batching budget
    auto_batch_on_oom: bool = True,              # try full first; fallback on MemoryError
    print_estimates: bool = True,                # log estimated sizes

    # Output
    out_dir: str = ".",
    prefix: str = "hottools",
    fmt: Literal["npy", "npz", "hdf5", "pt"] = "npy",
    per_sample: bool = False,

    # Dtype
    dtype: Literal["float32", "float16"] = "float32",
) -> None:
    if fasta == "":
        raise ValueError("You must provide --fasta.")

    # Resolve input path
    input_path = vcf
    if input_path is None:
        raise ValueError("You must provide --vcf.")

    # Parse region
    chrom_user, coords = region.split(":")
    start_str, end_str = coords.replace(",", "").split("-")
    start_1 = int(start_str)
    end_1 = int(end_str)

    chrom = normalize_chrom_to_vcf(chrom_user, input_path)
    chrom_fa  = normalize_chrom_to_fasta(chrom_user, fasta)
    
    region_str = f"{chrom}:{start_1}-{end_1}"
    L = end_1 - start_1 + 1

    # 1) Ensure we are working with a biallelic SNV VCF (and dedup if needed)
    filtered_vcf_path = ensure_biallelic_snps(input_path, out_dir=out_dir, region=region_str)

    # 2) If separate haplotypes, require phased
    if haplotype_mode == "separate":
        assert_phased_genotypes(
            vcf_path=filtered_vcf_path,
            region=region_str,
            samples_file=samples_file,
            n_check_variants=2000,
        )

    # 3) Sample IDs
    sample_ids = _get_sample_ids(filtered_vcf_path, samples_file)
    # Always save the sample order actually used
    sample_list_out = os.path.join(out_dir, f"{prefix}.sample_list.txt")
    with open(sample_list_out, "w") as f:
        for sid in sample_ids:
            f.write(f"{sid}\n")
    logger.info("Saved sample order to %s", sample_list_out)

    S_total = len(sample_ids)
    logger.info("Processing %d samples from %s", S_total, filtered_vcf_path)

    # dtype
    target_dtype = np.float16 if dtype == "float16" else np.float32

    # estimates
    est_bytes = estimate_onehot_bytes(S_total, L, dtype, haplotype_mode)
    if print_estimates:
        logger.info(
            "Estimated one-hot size (array only, no overhead): %.2f GB (S=%d, L=%d, dtype=%s, mode=%s)",
            bytes_to_gb(est_bytes), S_total, L, dtype, haplotype_mode
        )

    # Decide batch size:
    # - If user gave batch_size: use it
    # - Else if user gave max_memory_gb: compute proactive optimal
    # - Else: default is NO batching (try full)
    if batch_size is not None:
        bs = max(1, min(int(batch_size), S_total))
    elif max_memory_gb is not None:
        bs = plan_batch_size_from_budget(S_total, L, dtype, haplotype_mode, max_memory_gb=max_memory_gb)
    else:
        bs = S_total

    logger.info("Using batch_size=%d", bs)

    # 4) Extract variants + dosage
    positions, ref_bases, alt_bases, dosage_data = extract_region_with_bcftools(
        vcf_path=filtered_vcf_path,
        region=region_str,
        sample_ids=sample_ids,
        samples_file=samples_file,
        haplotype_mode=haplotype_mode,
    )
    V = positions.shape[0]
    logger.info("Region %s: %d variants, window length %d bp", region_str, V, L)

    # 5) Load reference and precompute matrices
    ref_window = load_reference_window(fasta, chrom_fa, start_1, end_1)  # (L,)
    ref_onehot = encode_bases(ref_window, target_dtype)               # (L,4)
    alt_mat = encode_bases(alt_bases, target_dtype)                  # (V,4)

    rel_pos = positions - start_1
    if (rel_pos < 0).any() or (rel_pos >= L).any():
        raise ValueError("Some variant positions lie outside the requested window.")

    ref_at_var = ref_onehot[rel_pos]  # (V,4)

    def _encode_and_save_chunk(start_s: int, end_s: int, out_prefix: str) -> None:
        batch_ids = sample_ids[start_s:end_s]
        S_batch = end_s - start_s

        if haplotype_mode == "average":
            # (V, S_total) -> (S_batch, V)
            dosage_sv = dosage_data[:, start_s:end_s].T  # (S_batch, V)

            seq_batch = np.broadcast_to(ref_onehot[None, :, :], (S_batch, L, 4)).copy().astype(target_dtype)

            D = dosage_sv.astype(target_dtype)[:, :, None]  # (S_batch, V, 1)
            R = np.broadcast_to(ref_at_var[None, :, :], (S_batch, V, 4))
            A = np.broadcast_to(alt_mat[None, :, :], (S_batch, V, 4))
            seq_batch[:, rel_pos, :] = (1.0 - D) * R + D * A

        else:
            # dosage_data: (2, V, S_total)
            hap_dos = dosage_data[:, :, start_s:end_s]      # (2, V, S_batch)
            hap_dos_sv = np.transpose(hap_dos, (2, 0, 1))   # (S_batch, 2, V)

            seq_batch = np.broadcast_to(ref_onehot[None, None, :, :], (S_batch, 2, L, 4)).copy().astype(target_dtype)

            D = hap_dos_sv.astype(target_dtype)[:, :, :, None]  # (S_batch, 2, V, 1)
            R = np.broadcast_to(ref_at_var[None, None, :, :], (S_batch, 2, V, 4))
            A = np.broadcast_to(alt_mat[None, None, :, :], (S_batch, 2, V, 4))
            seq_batch[:, :, rel_pos, :] = (1.0 - D) * R + D * A

        save_onehot(
            seq=seq_batch,
            sample_ids=batch_ids,
            out_dir=out_dir,
            prefix=out_prefix,
            fmt=fmt,
            per_sample=per_sample,
            compress="gzip",
        )

    # 6) Run: default try full; fallback on MemoryError if enabled
    try:
        if bs == S_total:
            _encode_and_save_chunk(0, S_total, prefix)
        else:
            for start_s in range(0, S_total, bs):
                end_s = min(S_total, start_s + bs)
                _encode_and_save_chunk(start_s, end_s, f"{prefix}.batch{start_s}")

    except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
        if not auto_batch_on_oom:
            raise

        logger.warning("Out of memory during encoding (%s). Falling back to automatic batching.", type(e).__name__)

        budget = max_memory_gb if max_memory_gb is not None else detect_available_memory_gb()
        bs2 = plan_batch_size_from_budget(S_total, L, dtype, haplotype_mode, max_memory_gb=budget)

        logger.warning("Auto-batch: budget≈%.2f GB -> batch_size=%d", budget, bs2)

        for start_s in range(0, S_total, bs2):
            end_s = min(S_total, start_s + bs2)
            _encode_and_save_chunk(start_s, end_s, f"{prefix}.batch{start_s}")

    logger.info("Hottools run completed.")
