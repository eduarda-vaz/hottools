# Correctness test suite

These are controlled correctness tests, separate from the paper's real-cohort
accuracy and runtime experiments. Run them from the repository root:

```bash
conda activate hottools_v2
python -m pip install -e '.[test]'
python -m pytest tests -q
```

The PyTorch writer tests require `torch`; without it, those tests are skipped.
HDF5, indexed VCF/BCF, and native bcftools checks use the test dependencies.
Native comparator tests call compiled bcftools through `pysam.bcftools`; this
is independent of the `bcftools` executable/version on the shell's PATH.
Native tests automatically print the compiled bcftools and pysam versions.
Use `python -m pytest tests -q --junitxml=validation-results.xml` to retain
versions in suite and concordance-case properties. No cohort VCFs or cluster
jobs are needed. In a partial checkout without `validation/`, only five
checkout-comparator regressions skip; independent regressions still run.

The reviewed suite contains 1,628 collected cases from 517 test functions.
Parameterized case counts are not counts of distinct biological behaviors.
The structural-oracle and synthetic-configuration matrices contribute 589
cases (36.2%). `test_review_coverage.py` adds 32 execution/input-format cases,
including BCF, 4/8 threads, per-sample masks and empty consensus runs.

To run only the new fixture-based tests:

```bash
python -m pytest tests/test_synthetic_configuration_matrix.py \
  tests/test_synthetic_cli_bcftools.py \
  tests/test_synthetic_structural_matrix.py -q
```

## Small, readable inputs

`data/snv_edges/reference.fa` contains a 64-base `ACGT` reference. The VCFs
are plain text for review; each test creates its own indexed BGZF copy and
FASTA index in pytest's temporary directory.

| Fixture | Records / samples | Cases |
| --- | --- | --- |
| `phased.vcf` | 11 / 8 | First and last reference positions; two and three ALT SNVs; mixed insertion/SNV with the SNV at ALT index 2; mixed symbolic deletion/SNV; star allele; duplicate positions; half calls, fully missing calls, haploid calls, missing haploid calls, homozygous REF |
| `unphased.vcf` | 1 / 8 | Unphased REF/ALT and ALT/ALT heterozygotes, homozygotes, half calls, haploid and missing calls |
| `structural.vcf` | 7 / 4 | Literal insertion/deletion, MNV, precise symbolic deletion, mixed SNV/insertion, unsupported symbolic insertion and breakend |

## What the new tests check

- **SNV policies:** biallelic-only; all SNV-only records; opt-in SNVs from mixed
  records. Whole-record exclusion is the default. Original ALT numbering is
  respected, including `A -> AT,C` and `A -> *,T`.
- **Encodings:** haplotypes, dosage, and consensus. Independent scalar
  expectations check every output position and channel, including untouched
  reference positions. Reference/alternate consensus policies and cohort-major
  selection are covered; major selection uses the selected cohort, never a
  batch-local frequency.
- **Genotypes:** zero/reference/error missing policies; phased/unphased behavior;
  duplicated/blank haploid tracks; haploid error/skip; rejection before output
  publication. Dosage and consensus remain phase-agnostic.
- **Execution profiles:** full sample run, batch sizes 1 and 3, automatic batches;
  1/2 reader threads; float16/float32; both strand orientations and layouts;
  deliberately reordered sample subsets. Four representative profiles are
  crossed with all SNV selection, mode, missing-fill, and haploid-track policies.
  This is targeted combination coverage, not every possible Cartesian product.
- **Writers and CLI:** NPY, NPZ, HDF5, and PyTorch; aggregate and per-sample outputs;
  biallelic and mixed-record selection flags through the public CLI.
- **Duplicates/masks/strict mode:** compatible calls on opposite haplotypes,
  conflicting calls, a discarded mixed record preceding a clean SNV, masks
  over duplicate sites, strict rejection of partially retained positions and
  acceptance of deliberate biallelic filtering.
- **Structural choices:** SNV-only, SNV+MNV, indel, and all-class configurations;
  whole/biallelic/selected-allele policies; start/center anchoring; a center
  anchor removed by a deletion; forward/reverse strands and full/single-sample
  batches. The existing independent text-edit oracle supplies expectations.
- **Native comparator:** twelve paired tests make 116 compiled bcftools consensus
  calls, with matching eligible variants and sample/haplotype selection. SNV
  projection independently filters whole records or remaps retained ALT/GT
  indices. Haploid duplication is compared to two copies of its one biological
  allele. Structural outputs use the same reference-start crop/padding rule.
- **Validation:** an invalid mixed-record policy fails before VCF/FASTA I/O in
  both the pipeline configuration and direct extraction APIs.

Expected fixture warnings are suppressed locally in these matrices. Existing
focused warning tests continue to assert warning/error behavior.

## Existing coverage retained

The original 1,064 tests remain in place. They include independent randomized
structural oracles, overlapping/nested edits, anchors, reference mismatches and
ambiguous bases, polyploid rejection/skip, QUAL/FILTER/AF filters and AF sources,
region and mask coordinates, indexed BCF input, overwrite/rollback behavior,
metadata, memory planning, and writer validation. Memory arithmetic tests use
a simulated, fixed available-RAM budget rather than the host machine's RAM;
additional checks exercise 512 MiB, 8 GiB and 512 GiB hosts and sample-count
clamping. `test_record_selection.py`
was reformatted for readability without changing its reconstruction assertions.

Passing this suite verifies these specified behaviors. It does not establish
biological haplotype truth, certify all SV representations, or replace the
paper's GTEx/MAGE accuracy experiments.

## Configuration audit (0.7.5)

See `COVERAGE_AUDIT.md` for the complete behavior/parameter inventory, valid
combinations, error rules and fixes. `test_configuration_errors.py` covers
invalid numeric/types/mode/format combinations and data-conditioned errors;
input-independent failures occur before FASTA/VCF opening.

```bash
python -m pytest tests/test_configuration_errors.py tests/test_api_contract.py -q
```


CLI help coverage also checks five terminal widths (COLUMNS 40/60/80/100/120).
The rendered-help assertion joins hyphenated names wrapped across physical
lines, avoiding a formatting-dependent failure. This adds four cases to the
previous 1,596-case suite; the complete 1,600-case suite passes with COLUMNS=100.


## Seeded pseudo-phasing

`test_random_assignment.py` adds 28 cases for the explicit random policy,
phased-call/allele preservation, reproducibility, seed errors, metadata, input
formats and CLI. It also compares reconstructed tracks to native bcftools on a
derived VCF containing the recorded allele assignments. This does not validate
the biological phase of the random choices. See `../RANDOM_ASSIGNMENT.md`.


### Contig alias fixtures

`data/contig_alias/numeric.vcf` uses chromosome `1` and phased SNV, deletion and
insertion calls for two samples. `data/contig_alias/reference.fa` uses `chr1`.
`test_contig_alias.py` indexes those small files in a temporary directory and
checks exact/auto policies, tensor agreement in all modes, batch sizes, strands,
structural anchors, masks, CLI region files/dry run, BCF, unknown/mismatched
lengths, mitochondrial aliases, shared reference handles, ambiguous names,
unsupported scaffold aliases and preserved REF validation. No cohort data are used.
