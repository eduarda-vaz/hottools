# Hottools

**Fast reconstruction and one-hot encoding of personalized genomic sequences from VCF/BCF files.**

Hottools extracts biallelic single-nucleotide variants, reconstructs sample-specific sequences relative to a reference genome, and produces memory-efficient one-hot encoded arrays suitable for deep learning models (e.g., Enformer, Borzoi).

### ⚠ Supported Variants

Hottools operates **exclusively on biallelic single-nucleotide variants (SNVs)**.

- REF and ALT must both be length 1  
- Indels are excluded  
- Multiallelic sites are excluded  
- Duplicate positions are excluded  

If necessary, Hottools filters the input VCF/BCF to biallelic SNVs before encoding.

---

## Features

- Fast region extraction via `bcftools`
- Automatic biallelic SNV filtering
- Dosage-based diploid encoding
- Optional phased haplotype separation
- Memory-aware batching
- Multiple output formats: `.npy`, `.npz`, `.hdf5` and `.pt` (PyTorch)

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/eduarda-vaz/hottools.git
```

---

## Requirements

- Python ≥ 3.9
- `bcftools` (must be installed and on PATH)
```bash
conda install -c bioconda bcftools
```
- Indexed reference FASTA (`.fai` required)


## Run test suite
```bash
pip install -e .[test]
pytest
```

## Basic Usage

```bash
hottools run \
  --vcf input.bcf \
  --region chr12:31000000-31200000 \
  --fasta hg38.fa \
  --out-dir output \
  --prefix chr12_block \
  --format npy
```

---

## Encoding Modes

### Average Mode (Default)

Combines diploid genotype into a dosage-based encoding.

Genotype mapping:

| GT | Encoding |
|----|----------|
| 0/0 | REF |
| 0/1 | 0.5 REF + 0.5 ALT |
| 1/1 | ALT |

Output shape is (S, L, 4), where **S** is the number of samples, **L** is the window length and **4** equals to A/C/G/T channels.

### Separate Haplotype Mode

Requires phased genotypes (`|` separator).

```bash
--haplotype-mode separate
```

Output shape is (S, 2, L, 4). Each sample produces two haplotype sequences.

## Output Formats

| Format | Description | Extra Dependency |
|--------|------------|-----------------|
| `npy`  | NumPy array | None |
| `npz`  | Compressed NumPy archive | None |
| `hdf5` | HDF5 dataset | `h5py` |
| `pt`   | PyTorch tensor | `torch` |


## Memory and Batching

Large regions and large cohorts can require significant memory.

Hottools provides:

- `--batch-size`
- `--max-memory-gb`
- `--auto-batch-on-oom`

Example:

```bash
hottools run \
  --vcf input.bcf \
  --region chr12:30000000-30500000 \
  --fasta hg38.fa \
  --max-memory-gb 8
```

Hottools will automatically compute a safe batch size. If an out-of-memory error occurs, it can retry automatically.

## Sample Subsetting

To restrict processing to specific samples:

```bash
--samples samples.txt
```

File format:
```
sample_1
sample_2
sample_3
```

Order in this file defines output ordering.

---

## Performance

**Runtime Comparison**
Total runtime for producing 524,288 bp sequences for 838 GTEx genomes (both haplotypes).

| Method                         | Time (s) | Relative Speed |
|--------------------------------|----------|----------------|
| Bcftools consensus (.fa)       | 1459     | 133× slower    |
| Hottools (.npy)                | 11       | 1× (baseline)  |


**Output Size Comparison**
Output size for 524,288 bp sequences for 838 GTEx genomes (float32).

| Format | Size    | Additional Runtime |
|--------|---------|-------------------|
| hdf5   | 450 MB  | +42 s             |
| npy    | 7.03 GB | 0 s               |
| npz    | 281 MB  | +150 s            |
| pt     | 7.03 GB | +5 s              |

---

## Limitations

- Only biallelic SNVs supported
- Indels are excluded
- Separate haplotype mode requires phased genotypes
- Structural variants are not supported

---

## Command Line Help

Top-level help:

```bash
hottools --help
```

Subcommand help:

```bash
hottools run --help
hottools inspect --help
```

---

## License

MIT License © 2026 Eduarda Vaz

See `LICENSE` file for full text.

---

## Citation

If you use Hottools in academic work, please cite:

```
Vaz, E. (2026). Hottools: Fast one-hot encoding of personalized genomic sequences.
```