# Hottools

**Fast reconstruction and one-hot encoding of personalized genomic sequences from a reference FASTA and VCF/BCF files.**

Hottools reconstructs personalized genomic sequences for selected samples and intervals, producing encoded arrays ready for downstream analysis and sequence models. It supports biallelic and multiallelic SNVs, multi-nucleotide substitutions, explicit insertions and deletions, and symbolic deletions with known endpoints. Three encoding modes are available: allele dosage, paired haplotype tracks, and a single consensus track per sample. Reconstruction and encoding proceed directly from the reference FASTA and indexed VCF/BCF, without generating intermediate FASTA files.

Version **1.0.0** · Python **3.10+** · [MIT license](LICENSE)

## Contents

- [Runtime example](#runtime-example)
- [Installation](#installation)
- [Inputs and coordinates](#inputs-and-coordinates)
- [Modes and output arrays](#modes-and-output-arrays)
- [Quick start](#quick-start)
- [Variant selection and reconstruction rules](#variant-selection-and-reconstruction-rules)
- [Filtering, masking and orientation](#filtering-masking-and-orientation)
- [Output files and memory](#output-files-and-memory)
- [Command-line examples](#command-line-examples)
- [Python examples](#python-examples)
- [All command-line options](#all-command-line-options)
- [Testing and documentation](#testing-and-documentation)

## Runtime example

Runtime for producing 49,152 bp paired-haplotype arrays in a high-density MAGE-derived region.

| Method | Allocated CPUs | 5,000 samples | 20,000 samples |
| --- | --- | --- | --- |
| Hottools, unbatched | 1 | 24.9 s | 114.7 s |
| bcftools consensus | 8 | 735.2 s | 4.71 h |
| bcftools consensus | 16 | 386.9 s | 2.41 h |

bcftools consensus used parallel sample workers, with one CPU allocated per worker. Reported times include reconstruction and array output; bcftools totals also include FASTA-to-array encoding.

## Installation

Install directly from GitHub:

```bash
python -m pip install "hottools[mem] @ git+https://github.com/eduarda-vaz/hottools.git"
```
NumPy, cyvcf2 and pyfaidx are installed automatically. The mem extra installs psutil, which is required for the default automatic batch planning.

Optional extras enable HDF5 output and PyTorch output:

```bash   
python -m pip install "hottools[mem,hdf5] @ git+https://github.com/eduarda-vaz/hottools.git"
python -m pip install "hottools[mem,torch] @ git+https://github.com/eduarda-vaz/hottools.git"
python -m pip install "hottools[all] @ git+https://github.com/eduarda-vaz/hottools.git"
```

A base installation is also available:
```bash
python -m pip install git+https://github.com/eduarda-vaz/hottools.git
```

Without the mem dependency, supply an explicit batch size or use --batch-size none. These commands install the repository’s default branch.

## Inputs and coordinates

Hottools requires a reference FASTA and a position-sorted, indexed variant file in bgzip-compressed VCF (.vcf.gz) or BCF format. Both files must use the same genome assembly. 

Chromosome names must match by default (`--contig-alias-policy exact`). Use `--contig-alias-policy auto` to resolve unique standard aliases such as `1` and `chr1`. Exact matches take priority. Ambiguous mappings and known chromosome-length disagreements raise errors. This option does not modify the input files or convert coordinates between genome assemblies.

The VCF must contain sample genotypes in the `GT` field. Hottools reconstructs sequences from these allele calls; it does not infer biological phase or use dosage estimates from fields such as `DS`.

If indexes are missing, these external commands can prepare them:

```bash
samtools faidx reference.fa
# For bgzip-compressed VCF:
bcftools index -t cohort.vcf.gz
# For BCF:
bcftools index cohort.bcf
```

Samtools and bcftools are useful for input preparation and validation but are not required by the Hottools reconstruction pipeline. Prepare the FASTA index in advance if its directory is read-only.

### Selecting regions 

Hottools requires a region selection; it does not reconstruct the whole VCF by default. Use `--region CHROM:START-END` for one interval with 1-based inclusive coordinates; `--center CHROM:POS --length L` for a window around a reference position; or `--region-file FILE` for multiple intervals processed separately in one run. These options cannot be used together. The interval coordinates must refer to the same genome assembly as the FASTA and VCF/BCF.

Region files ending in `.bed`, `.bed.gz` or `.bed.bgz` are automatically read as 0-based, half-open BED intervals. Other filenames default to 1-based inclusive coordinates. Use `--region-file-format bed` or `--region-file-format 1based` to specify the convention explicitly. 

Plain-text region files accept:
```bash
chr1:10001-59152
chr2:20001-69152
```

Alternatively, use `CHROM START END [STRAND]`. BED files use the standard optional strand field in column 6; plain-text intervals use column 4.

Regions may have different lengths, but share the same sample selection, filters and encoding settings. Each interval must lie within one reference contig. With `--region-file`, `--output` specifies a directory.

## Modes and output arrays

Channels always follow **A, C, G, T**. Let `S` be sample count and `L` output length.

| Mode | Default shape | Interpretation |
| --- | --- | --- |
| `dosage` (default) | `S × L × 4` | Mean allele contribution; diploid A/G is `[0.5, 0, 0.5, 0]`. Phase is not needed. |
| `haplotypes` | `S × 2 × L × 4` | One track per haplotype. Heterozygotes require phase unless an explicit unphased policy is selected. |
| `consensus` | `S × L × 4` | One chosen allele per sample; heterozygotes follow `--het-policy`. |

### Pseudo-haplotypes
In `haplotypes` mode, a VCF may contain both phased and unphased genotypes. Phased calls retain their allele order. Unphased heterozygotes raise an error by default; `--unphased-policy missing` treats them as missing and applies `--missing-policy`, while `--unphased-policy random --random-seed 42` assigns their alleles reproducibly between the two tracks. 

When random assignment is used, these mixed tracks contain known phase where available and arbitrary assignments elsewhere, so they should be described as pseudo-haplotypes, not fully phased biological sequences. Hottools does not infer phase or establish continuity between separate phase blocks.

### Consensus encoding policy
Consensus produces one sequence per sample, rather than preserving both haplotypes. At heterozygous sites, `--het-policy alternate` (default) selects ALT when present, `--het-policy reference` selects REF, and `--het-policy major` selects the more frequent of the called alleles among the selected samples. This output does not preserve both haplotypes.

Frequency ties choose the lower allele index. For a `1|2` genotype, both reference and alternate select the lower-indexed ALT because no REF allele is present.

### Unknown or missing bases 
An all-zero vector `[0, 0, 0, 0]` represents unknown or masked sequence, not a fifth nucleotide. Missing alleles contribute zeros by default. Add `--missing-policy reference` when filling missing calls with the REF is appropriate for the analysis.

### Output layout
The default, `--output-layout samples_first`, produces `S × L × 4` or `S × 2 × L × 4`. Use `--output-layout channels_first` to produce `S × 4 × L` or `S × 2 × 4 × L`.

The default dtype is `--dtype float32`; `--dtype float16` is also supported.

## Quick start

For an unphased cohort, construct allele-dosage tensors:

```bash
hottools cohort.vcf.gz \
  --reference reference.fa \
  --region chr1:10001-59152 \
  --mode dosage \
  --output dosage.npy
```

For a phased cohort, reconstruct both haplotypes:

```bash
hottools phased.vcf.gz \
  --reference reference.fa \
  --region chr1:10001-59152 \
  --mode haplotypes \
  --output haplotypes.npy
```

These examples apply biallelic and multiallelic SNV-only records. Multiallelic records can also be excluded altogether (`--biallelic-only`). 

## Variant selection and reconstruction rules

### SNVs and multiallelic records

The default is `--variant-classes snv --mixed-record-policy exclude`. This requires a literal single A/C/G/T base in REF and every ALT. Biallelic and multiallelic SNV-only records are supported, with at most three ALT alleles. Records exceeding that limit are excluded.

The following table assumes **SNV-only** reconstruction:

| Record | Default | With `--biallelic-only` | With `--mixed-record-policy apply-selected` |
| --- | --- | --- | --- |
| `A → C` | Apply | Apply | Apply |
| `A → C,G` | Apply | Exclude whole record | Apply |
| `A → C,AT` | Exclude whole record | Exclude whole record | Apply C; project AT calls to REF |
| `A → C,<DEL>` | Exclude whole record | Exclude whole record | Apply C; project deletion calls to REF |
| `AT → AC` | Exclude in literal SNV-only mode | Exclude in literal SNV-only mode | Exclude in literal SNV-only mode |

For a mixed record such as `REF=A, ALT=C,AT`, a SNV-only run normally excludes the entire record. With `--mixed-record-policy apply-selected`, Hottools retains C and treats calls selecting the excluded insertion AT as reference A. If `--biallelic-only` is also enabled, the entire record is excluded because the original record contains two ALT alleles, even if projection would retain only one.

### Other supported variants

SNVs are always included when another class is requested.

| `--variant-classes` value | Applies | Modes |
| --- | --- | --- |
| `snv` | Literal SNVs under the default selection rule | All |
| `snv,mnv` | SNVs and equal-length multi-base substitutions | All |
| `snv,del,ins` or `indel` | SNVs, explicit deletions and insertions | Haplotype only |
| `snv,symbolic-del` | SNVs and symbolic `<DEL>` records with `INFO/END` | Haplotype only |
| `all` | All supported classes above | Haplotype only |

For mixed records, like `REF=A, ALT=C,<INV>`, where the SNV is supported but not the symbolic inversion, the default is exclusion. However, once again, `--mixed-record-policy apply-selected` can be used to retain supported alleles and treat unsupported allele calls as REF. In the previous case, the sample carrying `ALT=C` for that allele would be reconstructed.

Breakends, symbolic inversions/duplications/insertions and * spanning-deletion alleles are unsupported. Records containing unsupported alleles are excluded by default. Explicit `apply-selected` can retain other supported, requested alleles while treating excluded ALT calls as REF.

**Fixed-length outputs and anchors**

Length-changing edits, such as insertions and deletions require haplotype mode. They change the reconstructed sequence, but the output window retains the requested length by cropping excess sequence or drawing additional bases from the reference. 

By default, the reconstructed sequence starts at the beginning of your requested region (`--anchor start`). For a region starting at position 100, reference base 100 stays at the first output position. `--anchor center` keeps the central reference base at its original output position. You can also specify a coordinate, such as `--anchor 34576` or `--anchor chr1:34576`, to keep that base at its original offset within the region. 

The coordinate must lie inside the requested region. If a deletion removes the anchor, `--anchor-deletion-policy right` uses the next surviving base; `error` stops the run instead. Positions that cannot be filled from available reference sequence are zero-filled, counted and reported with a warning.

### Duplicates and overlaps

Hottools resolves duplicate positions and overlapping edits separately for each sample and haplotype. By default (`--overlap-policy first`), it retains the first applicable conflicting ALT in file order and skips later conflicts. Use `--overlap-policy error` to stop the run instead.

Reference call does not prevent a later ALT from being applied, and variants on different haplotypes can both be retained. Suppose two records at the same position have **REF=A**: the first has **ALT=C, GT=1|0**, and the second has **ALT=G, GT=0|1**. The first record applies C to haplotype 1, while its reference call on haplotype 2 leaves that track available for the second record’s G. The result is **C|G**, so both variants are retained. If the second record instead called G on haplotype 1, it would conflict with the earlier C and be skipped on that haplotype under the default `--overlap-policy first`. Use `--overlap-policy error` to stop the run instead of resolving conflicts. Records excluded by filters do not compete with retained records; warnings and metadata counters report skipped conflicts. 

This ordering rule is consistent with the sample/haplotype behavior tested against bcftools consensus, but it does not determine which conflicting variant is biologically correct. 

Agreement also depends on using matching variant filters and genotype policies. Because overlapping variants before the requested region can affect which record wins inside it, Hottools checks upstream records and warns when that context may be insufficient.

### Missing calls, phase and reference checks

**Missing alleles** 

By default, each missing allele contributes `[0,0,0,0]`. Use `--missing-policy reference` to substitute the reference allele, or `--missing-policy error` to stop when a missing call is encountered.

With zero filling, a partially called diploid genotype such as `GT=0/.` has two allele copies but one unknown call. If REF is A, its dosage contribution is `[0.5,0,0,0]`.

**Unphased heterozygotes** 

Accepted in dosage and consensus modes; rejected by default in haplotype mode. See #pseudo-haplotypes for alternative policies and their limitations.

**Ploidy** 

Hottools supports genotypes containing one allele (haploid, such as GT=1) or two alleles (diploid, such as GT=0|1). If any selected sample has more than two allele copies in a genotype, such as GT=0/1/1, Hottools stops by default. With `--ploidy-policy skip`, it instead excludes that variant record for all selected samples, not just the sample with the unsupported ploidy.

For a haploid genotype, Hottools encodes the single called allele at full weight. For example, an A allele becomes `[1,0,0,0]` in dosage and consensus modes. In haplotype mode, which always produces two tracks, the allele is copied into both tracks by default (`--haploid-policy duplicate`). Using `--haploid-policy blank` keeps the allele in the first track and fills the second with zeros. The duplicated track is an output convention; it does not mean the sample biologically carries two copies.

**Reference checks** 

Hottools checks each candidate variant’s REF against the FASTA. It stops by default if they disagree, or if the FASTA base is ambiguous. Use `--ref-mismatch-policy skip` to exclude records with a disagreement, or `--ambiguous-ref-policy skip` to exclude records whose FASTA contains an uncertain base, such as N. Replacing `skip` with `warn` prints a warning while the record is still excluded. Ambiguous bases elsewhere in the reference are encoded as `[0,0,0,0]`.

**No usable variants** 

If no variants remain after selection and filtering, Hottools writes reference-derived sequences without applying any sample-specific variants, with any masks still applied, and issues a warning. Use `--strict` to make this an error instead.

## Filtering, masking and orientation

### Reverse complement

`--strand` controls the orientation of the reconstructed sequence. The default, `--strand +`, follows the reference FASTA and `--strand -` produces the reverse complement: it reverses the sequence and exchanges A <-> T and C <-> G. Coordinates still refer to the forward reference genome.

A strand specified in a region file overrides `--strand` for that interval.

### Masking

Use `--mask FILE` to mark regions as unknown; their bases become `[0,0,0,0]`, and variants affecting those regions are not applied. Masks use reference coordinates, so insertions or deletions may shift masked bases within the final output. 

Accepted formats are BED (.bed, .bed.gz, .bed.bgz) with 0-based, half-open coordinates, or plain text (.txt, .txt.gz) containing CHROM:START-END or CHROM START END with 1-based inclusive coordinates.

### Sample selection

Hottools processes all samples in the VCF by default. Select a subset with `--samples SAMPLE_A,SAMPLE_B` for comma-separated IDs, or `--sample-file samples.txt` for a file containing one ID per line. Output rows preserve the requested order. 

With `--af-source samples`, only the selected samples determine the filtering frequencies.

### Quality and allele-frequency filters

Hottools applies no FILTER, QUAL or AF thresholds unless requested.

Use `--pass-only` to retain FILTER `PASS` or `.`, `--min-qual Q` to retain QUAL ≥Q and exclude missing QUAL, and `--min-af`/`--max-af` to set inclusive alternate-frequency bounds.

These filters apply to entire records, not individual genotypes. Dedicated GQ or DP filters are not provided.

By default, `--af-source auto` uses INFO/AF when available and otherwise calculates frequencies from the selected samples. Use `--af-source samples` to always calculate frequencies from the selected samples after genotype policies are applied, or `--af-source info` to require INFO/AF. 

For multiallelic records, filtering uses the largest individual ALT frequency, not the sum. These are AF filters, not direct MAF filters. For biallelic records, `--min-af 0.01 --max-af 0.99` corresponds to MAF ≥0.01.

Excluded variants leave reference sequence at their positions unless those positions are masked.

## Output files and memory

### Batching

Batching controls how many samples are encoded at once. The default, `--batch-size auto`, requires the optional mem dependency and estimates a batch size from available memory. Use a positive integer, such as `--batch-size 200`, to set it explicitly, or `--batch-size none` to encode all selected samples together. 

Automatic planning considers sequence length, mode, dtype, resident data and temporary arrays. It checks available system memory and readable cgroup/Slurm constraints. `--memory-fraction` defaults to 0.40.

Planning is an estimate, not a hard memory limit. Genotype extraction occurs before batch planning. Batching reduces temporary encoding memory but does not reduce final output size.

### Formats

| Format | Batching Behavior | Optional dependency | Loading |
| --- | --- | --- | --- | 
| `npy` | Batches are written into one memory-mapped array | None | `np.load(path, mmap_mode="r")` |
| `npz` | Compressed arrays; separate files if multiple batches are needed | None | `np.load(path)`; use `.files` for stored keys |
| `h5` | Batches are written into one chunked HDF5 file | `hdf5` extra | `h5py.File(path, "r")` |
| `pt` | PyTorch tensors; separate files if multiple batches are needed | `torch` extra | `torch.load(path)` |

Use `--format` to select the format; changing the filename extension alone does not select it.

For HDF5, use `--h5-compression gzip` or `lzf` to enable compression. Chunk sizes can be configured with `--h5-chunk-samples` and `--h5-chunk-length`.

### Per-sample output

By default, samples share a combined output. Use `--per-sample` to write one file per sample; `--output` then specifies a directory. 

### Metadata and existing outputs

Hottools writes a JSON metadata containing sample order, genomic coordinates, reconstruction policies, filtering counters, memory estimates and output filenames. Keep it with the arrays to preserve their interpretation. Use `--no-metadata-json` to suppress it.

Existing outputs are protected by default (--overwrite error). --overwrite replace replaces a previous run and removes its obsolete Hottools outputs. Replacement temporarily requires space for both old and new data.

`--threads` controls cyvcf2/HTSlib input threads, not parallel sample-reconstruction workers.

## Command-line examples

### Biallelic SNVs only, with selected samples and site filters

```bash
hottools cohort.vcf.gz \
  --reference reference.fa \
  --region chr1:10001-59152 \
  --samples SAMPLE_A,SAMPLE_B \
  --mode dosage --biallelic-only --pass-only --min-qual 20 \
  --min-af 0.01 --max-af 0.99 --af-source samples \
  --output selected.npy
```

### Retain SNV alleles from mixed records

```bash
hottools cohort.vcf.gz --reference reference.fa \
  --region chr1:10001-59152 --mode dosage \
  --mixed-record-policy apply-selected --output projected_snvs.npy
```

### SNVs, insertions and deletions in phased samples

```bash
hottools phased.vcf.gz --reference reference.fa \
  --center chr1:34576 --length 49152 --mode haplotypes \
  --variant-classes all --anchor center --overlap-policy first \
  --output haplotypes_with_edits.npy
```

### Inspect a run before constructing tensors

```bash
hottools cohort.vcf.gz --reference reference.fa --region chr1:10001-59152 \
  --mode dosage --dry-run --output planned.npy
```

Dry run reads and filters the variants, validates the configuration and reports metadata and the memory plan. It does not construct or write sequence tensors; it is not a zero-cost check.

## Python examples

### Run the complete pipeline

```python
from hottools import RunConfig, parse_region, run_hottools

config = RunConfig(
    vcf_path="cohort.vcf.gz",
    fasta_path="reference.fa",
    region=parse_region("chr1:10001-59152"),
    output_path="dosage.npy",
    mode="dosage",
    samples=["SAMPLE_A", "SAMPLE_B"],
    variant_classes="snv",
    biallelic_only=True,
    mixed_record_policy="exclude",
    batch_size="auto",
    missing_policy="zero",
)
metadata = run_hottools(config)
print(metadata["sample_ids"])
```

run_hottools writes the output and returns metadata, not the sequence array. Load the NPY output with:

```python
sequences = np.load("dosage.npy", mmap_mode="r")
print(sequences.shape) 
```

CLI names usually become underscore names in `RunConfig`. Important mappings include:
| CLI | Python |
| --- | --- |
| Positional VCF | `vcf_path` |
| `--reference` | `fasta_path` |
| `--output` | `output_path` |
| `--format` | `output_format` |
| `--mask` | `mask_path` |
| `--no-metadata-json` | `metadata_json=False` |

In Python, supply a `Region` and a sample list directly. To process multiple regions, iterate over configurations. `batch_size="none"` disables batching; `None` and `"auto"` enable automatic planning.

### Encode an existing reference string

This Python-only function encodes the supplied string without reading a FASTA or applying variants. Columns follow A, C, G, T; N produces four zeros. Base index 255 marks ambiguity.

```python
from hottools import one_hot_reference

reference, base_indices = one_hot_reference("ACGTN")
print(reference.shape) 
```

## All command-line options

Run `hottools --help` for the parser's complete help. 

### Inputs, regions and samples

| Option | Values / default | Meaning |
| --- | --- | --- |
| `vcf` | Required positional path | Indexed VCF.gz or BCF. |
| `--reference` | Required FASTA path | Matching reference sequence. |
| `-o`, `--output` | Required path | Output file, or directory for per-sample/region-file runs. |
| `--region` | `CHROM:START-END` | One 1-based inclusive interval. |
| `--center` | `CHROM:POS` | Center-based selection; requires `--length`. |
| `--length` | Positive integer | Window length; only with `--center`. |
| `--region-file` | Path | Multiple intervals; output is a directory. |
| `--region-file-format` | `auto` (default), `1based`, `bed` | Region-file coordinate convention. |
| `--samples` | Comma-separated IDs | Sample subset and output order. |
| `--sample-file` | One ID per line | Alternative to `--samples`; blank/comment lines ignored. |
| `--strand` | `+` (default), `-` | Output orientation unless supplied per region. Minus reverse-complements bases and channels. |
| `--contig-alias-policy` | `exact` (default), `auto` | Require exact chromosome names, or resolve unique standard aliases such as `1` <-> `chr1`. |

Choose exactly one of `--region`, `--center`, or `--region-file`, and use at most one of `--samples` or `--sample-file`; omitting both selects all samples.

### Encoding and variant selection

| Option | Values / default | Meaning |
| --- | --- | --- |
| `--mode` | `dosage` (default), `haplotypes`, `consensus` | Output interpretation. |
| `--het-policy` | `alternate` (default), `reference`, `major` | Heterozygote choice in consensus mode. |
| `--variant-classes` | `snv` (default); comma-separated `snv,mnv,del,ins,symbolic-del`; `indel`; `all` | Requested supported classes; SNVs always included. |
| `--biallelic-only` | Off by default | Reject records with more than one ALT. |
| `--mixed-record-policy` | `exclude` (default), `apply-selected` | Whole-record exclusion or explicit allele projection. |
| `--overlap-policy` | `first` (default), `error` | Resolve or reject conflicting records. |
| `--anchor` | `start` (default), `center`, position, `CHROM:POS` | Reference base pinned to its original output offset; explicit positions must lie in the region and contigs must match. |
| `--anchor-deletion-policy` | `right` (default), `error` | Behavior when a deletion removes the anchor. |

### Genotype and reference policies

| Option | Values / default | Meaning |
| --- | --- | --- |
| `--missing-policy` | `zero` (default), `reference`, `error` | Unknown allele encoding. |
| `--unphased-policy` | `error` (default), `missing`, `random` | Unphased heterozygotes in haplotype mode. |
| `--random-seed` | Integer `0` through `2**64-1`; unset | Required for `random`, invalid without that policy; random assignment requires haplotype mode. |
| `--haploid-policy` | `duplicate` (default), `blank`, `error`, `skip` | Haplotype-track handling, or record refusal/exclusion. |
| `--ploidy-policy` | `error` (default), `skip` | Records with more than two allele slots. |
| `--ref-mismatch-policy` | `error` (default), `skip`, `warn` | VCF REF disagrees with FASTA; warn also skips. |
| `--ambiguous-ref-policy` | `error` (default), `skip`, `warn` | Ambiguous FASTA at a candidate variant; warn also skips. |
| `--strict` | Off by default | Promote empty-region and split-multiallelic warnings to errors. |

With `--missing-policy zero`, a partially called diploid genotype such as `A/.` (two allele copies exist, but one is unknown) will contribute as `[0.5, 0, 0, 0]` in dosage mode. A haploid A call such as `GT=0` (only one allele copy exists) contributes as `[1, 0, 0, 0]` (when REF is A). In haplotype mode, the haploid allele is copied into both tracks by default; `--haploid-policy blank` leaves the second track zero-filled instead.

### Site filters and masking

| Option | Values / default | Meaning |
| --- | --- | --- |
| `--pass-only` | Off by default | Keep FILTER `PASS` or `.`. |
| `--min-qual` | Number; unset | Drop lower or missing QUAL. |
| `--min-af`, `--max-af` | Frequencies in `[0,1]`; unset | Keep records within inclusive largest-ALT frequency bounds. |
| `--af-source` | `auto` (default), `info`, `samples` | Frequency source; `info` requires usable INFO/AF when an AF filter is used. |
| `--mask` | Interval file; unset | Zero masked reference positions and exclude variants affecting masked bases. BED/1-based format is inferred from filename. |

For deletions the changed bases exclude the VCF anchor. Insertions are excluded
when either flanking base is masked. Masking uses reference coordinates.

### Output and resources

| Option | Values / default | Meaning |
| --- | --- | --- |
| `--format` | `npy` (default), `npz`, `h5`, `pt` | Storage format; h5/pt need optional dependencies. |
| `--per-sample` | Off by default | Write one file per sample. |
| `--dtype` | `float32` (default), `float16` | Channel dtype. |
| `--output-layout` | `samples_first` (default), `channels_first` | Channel-axis position. |
| `--batch-size` | `auto` (default), `none`, positive integer | Samples encoded at once. |
| `--memory-fraction` | `(0,1]`; default `0.40` | Fraction of effective available memory used by automatic planning. |
| `--threads` | Positive integer; library default if unset | cyvcf2/HTSlib input threads; not sample-worker parallelism. |
| `--h5-compression` | `none` (default), `gzip`, `lzf` | HDF5 compression. |
| `--h5-chunk-samples` | Positive integer; automatic | HDF5 chunk size on sample axis. |
| `--h5-chunk-length` | Positive integer; automatic | HDF5 chunk size on sequence axis. |
| `--overwrite` | `error` (default), `replace` | Refuse existing outputs or replace files belonging to the previous run. |
| `--no-metadata-json` | Off by default | Suppress the metadata sidecar. |
| `--dry-run` | Off by default | Inspect inputs and report without constructing tensors. |
| `--debug` | Off by default | Show full tracebacks on errors. |
| `-h`, `--help` | — | Show help and exit. |

HDF5 compression/chunk overrides require `--format h5`. Invalid settings and unsupported combinations raise errors rather than silently selecting another mode. CLI input/configuration errors exit with status 2; successful calls print metadata as JSON and exit with status 0.

## Testing and documentation

```bash
python -m pip install -e '.[dev]'
python -m pytest tests -q -ra
python -m ruff check src tests
```
