# tests/test_integration_hottools_vs_bcftools.py

from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
import os

def _need(exe: str):
    if shutil.which(exe) is None:
        pytest.skip(f"{exe} not available")

def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True).stdout

def _fasta_to_onehot(path: Path) -> np.ndarray:
    seq = "".join(l.strip() for l in path.read_text().splitlines() if not l.startswith(">"))
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq.upper()):
        if b == "A":
            arr[0, i] = 1
        elif b == "C":
            arr[1, i] = 1
        elif b == "G":
            arr[2, i] = 1
        elif b == "T":
            arr[3, i] = 1
    return arr

def _load_npy(out_dir: Path, prefix: str) -> np.ndarray:
    p = out_dir / f"{prefix}.npy"
    if not p.exists():
        hits = sorted(out_dir.glob(f"{prefix}*.npy"))
        assert hits, f"Missing npy for prefix={prefix} in {out_dir}"
        p = hits[0]
    return np.load(p, allow_pickle=False)

def _as_N4L(x: np.ndarray) -> np.ndarray:
    # (N,4,L) or (N,L,4)
    if x.ndim != 3:
        raise AssertionError(f"expected 3D, got {x.shape}")
    if x.shape[1] == 4:
        return x
    if x.shape[2] == 4:
        return np.transpose(x, (0, 2, 1))
    raise AssertionError(f"cannot interpret onehot shape {x.shape}")

def _as_N24L(x: np.ndarray, n: int) -> np.ndarray:
    # (N,2,4,L) or (N,2,L,4) or (2N,4,L) or (2N,L,4)
    if x.ndim == 4:
        if x.shape[2] == 4:
            return x
        if x.shape[3] == 4:
            return np.transpose(x, (0, 1, 3, 2))
        raise AssertionError(f"cannot interpret hap shape {x.shape}")
    if x.ndim == 3:
        flat = _as_N4L(x)  # (2N,4,L)
        assert flat.shape[0] == 2 * n, f"expected 2*N rows, got {flat.shape}"
        return flat.reshape(n, 2, 4, flat.shape[2])
    raise AssertionError(f"cannot interpret hap shape {x.shape}")


@pytest.mark.integration
def test_hottools_vs_bcftools(
    tmp_path: Path,
    synthetic_phased_vcf: Path,
    synthetic_unphased_vcf: Path,
    synthetic_vcf_samples: Path,
    grch38_test_fasta: Path,
):
    _need("hottools")
    _need("bcftools")
    _need("samtools")
    _need("tabix")

    region = "chr12:1-61"
    samples = [l.strip() for l in synthetic_vcf_samples.read_text().splitlines() if l.strip()]
    N = len(samples)

    hot_dir = tmp_path / "hottools"
    bcf_dir = tmp_path / "bcftools"
    hot_dir.mkdir()
    bcf_dir.mkdir()

    fasta_bcftools = grch38_test_fasta

    from collections import Counter

def _filtered_vcf_for_bcftools(in_vcf: Path, tag: str) -> Path:
    out_vcf = bcf_dir / f"bcftools.{tag}.vcf.gz"
    dup_txt = bcf_dir / f"dup_positions.{tag}.txt"

    pos_text = _run(
        [
            "bcftools",
            "query",
            "-r",
            region,
            "-m2",
            "-M2",
            "-v",
            "snps",
            "-f",
            "%CHROM\t%POS\n",
            str(in_vcf),
        ]
    )

    pairs = [tuple(ln.split("\t")) for ln in pos_text.splitlines() if ln.strip()]
    counts = Counter(pairs)
    dup_sites = sorted([k for k, v in counts.items() if v > 1], key=lambda x: (x[0], int(x[1])))

    dup_txt.write_text("".join(f"{c}\t{p}\n" for c, p in dup_sites))

    if not dup_sites:
        _run(
            [
                "bcftools",
                "view",
                "-r",
                region,
                "-m2",
                "-M2",
                "-v",
                "snps",
                "-Oz",
                "-o",
                str(out_vcf),
                str(in_vcf),
            ]
        )
    else:
        dup_bed = bcf_dir / f"dup_sites.{tag}.bed"
        dup_bed.write_text("".join(f"{c}\t{int(p)-1}\t{p}\n" for c, p in dup_sites))

        _run(
            [
                "bcftools",
                "view",
                "-r",
                region,
                "-m2",
                "-M2",
                "-v",
                "snps",
                "-T",
                f"^{dup_bed}",
                "-Oz",
                "-o",
                str(out_vcf),
                str(in_vcf),
            ]
        )

    _run(["tabix", "-f", "-p", "vcf", str(out_vcf)])
    return out_vcf

    def _bcftools_haps(vcf_filtered: Path, tag: str) -> np.ndarray:
        """
        Replicates:
          bcftools consensus -H 1/2 on OUTPUT_BCFTOOLS_FILE
          then fasta_to_one_hot per sample
        Returns (N,2,4,L)
        """
        h1, h2 = [], []
        for s in samples:
            hap1_fa = bcf_dir / f"bcftools_{tag}_{s}_hap1.fa"
            hap2_fa = bcf_dir / f"bcftools_{tag}_{s}_hap2.fa"

            hap1_fa.write_text(
                _run(["bcftools", "consensus", "-s", s, "-f", str(fasta_bcftools), "-H", "1", str(vcf_filtered)])
            )
            hap2_fa.write_text(
                _run(["bcftools", "consensus", "-s", s, "-f", str(fasta_bcftools), "-H", "2", str(vcf_filtered)])
            )

            h1.append(_fasta_to_onehot(hap1_fa))
            h2.append(_fasta_to_onehot(hap2_fa))

        return np.stack([np.stack(h1), np.stack(h2)], axis=1)  # (N,2,4,L)

    # ---------------- phased: hottools avg + separate; bcftools filtered + consensus ----------------

    _run(
        [
            "hottools",
            "run",
            "--vcf",
            str(synthetic_phased_vcf),
            "--samples",
            str(synthetic_vcf_samples),
            "--region",
            region,
            "--fasta",
            str(grch38_test_fasta),
            "--format",
            "npy",
            "--out-dir",
            str(hot_dir),
            "--prefix",
            "hottools.avg.synthetic.phased",
        ]
    )

    # hottools separate
    _run(
        [
            "hottools",
            "run",
            "--vcf",
            str(synthetic_phased_vcf),
            "--samples",
            str(synthetic_vcf_samples),
            "--region",
            region,
            "--fasta",
            str(grch38_test_fasta),
            "--haplotype-mode",
            "separate",
            "--format",
            "npy",
            "--out-dir",
            str(hot_dir),
            "--prefix",
            "hottools.sep.synthetic.phased",
        ]
    )

    phased_filtered = _filtered_vcf_for_bcftools(synthetic_phased_vcf, tag="synthetic.phased")
    bcf_phased = _bcftools_haps(phased_filtered, tag="phased")

    hot_phased_sep = _as_N24L(_load_npy(hot_dir, "hottools.sep.synthetic.phased"), N)

    assert hot_phased_sep.shape == bcf_phased.shape
    assert np.array_equal(hot_phased_sep, bcf_phased)

    # ---------------- unphased: hottools avg; bcftools filtered + consensus ----------------

    _run(
        [
            "hottools",
            "run",
            "--vcf",
            str(synthetic_unphased_vcf),
            "--samples",
            str(synthetic_vcf_samples),
            "--region",
            region,
            "--fasta",
            str(grch38_test_fasta),
            "--format",
            "npy",
            "--out-dir",
            str(hot_dir),
            "--prefix",
            "hottools.avg.synthetic.unphased",
        ]
    )

    unphased_filtered = _filtered_vcf_for_bcftools(
        synthetic_unphased_vcf, tag="synthetic.unphased"
    )

    # bcftools still produces hap1/hap2 even if GT is unphased; we average them
    bcf_unphased_haps = _bcftools_haps(unphased_filtered, tag="unphased")
    bcf_unphased_avg = ((bcf_unphased_haps[:, 0] + bcf_unphased_haps[:, 1]) / 2.0).astype(
        np.float32
    )

    hot_unphased_avg = _as_N4L(
        _load_npy(hot_dir, "hottools.avg.synthetic.unphased")
    ).astype(np.float32)

    assert hot_unphased_avg.shape == bcf_unphased_avg.shape
    assert np.array_equal(hot_unphased_avg, bcf_unphased_avg)
