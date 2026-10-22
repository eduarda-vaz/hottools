# Behavior and configuration audit — Hottools 0.7.5

This is a correctness audit, separate from cohort accuracy/runtime experiments.
The complete suite passes **1,628 tests with zero failures and zero skips** in
the local Python 3.12.2 environment. The earlier configuration audit added 203 cases
beyond the 1,361-test suite; the review follow-up adds 32 targeted cases. Full logs and dependency versions accompany
the downloadable repository. Twelve new paired native-comparator tests execute
116 compiled bcftools 1.24 consensus calls through pysam.

## Behaviors checked

| Category | Behavior and edge cases | Main test files |
| --- | --- | --- |
| SNV eligibility | Literal single-base REF/ALT, biallelic-only, two/three ALT SNVs; whole mixed-record rejection by default; opt-in selected alleles; original ALT ordering; biallelic filter precedes mixed opt-in | `test_record_selection`, `test_snv_record_selection`, `test_synthetic_configuration_matrix`, `test_configuration_errors` |
| Mixed allele projection | Excluded ALT calls become REF genotype codes before phasing, sample AF, major-consensus selection and overlap resolution; REF/ALT consensus agrees with independent VCF projection regardless of ALT order | `test_configuration_errors`, `test_synthetic_cli_bcftools` |
| Unsupported records | Indels/SVs excluded from SNV-only runs; star, ambiguous ALT, symbolic insertion/deletion, breakends, DUP, INV and NON_REF; SNV+SV records excluded unless selected-allele policy is enabled; retained SNVs must have usable GT | `test_variants`, `test_edits`, `test_synthetic_structural_matrix`, `test_configuration_errors` |
| Genotypes | REF/REF, REF/ALT, ALT/ALT; multiple ALT indices; phased/unphased; fully missing and half-called; haploid and missing haploid; triploid rejection or whole-record skip; undeclared allele index errors | `test_essentials`, `test_variants`, `test_features`, `test_review_fixes` |
| Missing and phase interactions | Zero, REF-fill and error policies; haplotypes require phase for projected heterozygotes; unphased-to-missing plus missing-error fails; dosage/consensus are phase-agnostic; excluded records cannot cause genotype errors | `test_synthetic_configuration_matrix`, `test_configuration_errors` |
| Encodings | Exact haplotype tracks, equal diploid allele dosage, full-weight haploid dosage, half-call contributions, REF/ALT/cohort-major consensus and ties; frequencies use selected samples rather than each batch; reverse-complement base/channel order | `test_encoding`, `test_essentials`, `test_features`, `test_synthetic_configuration_matrix` |
| Duplicates and overlaps | First applicable ALT in record order per haplotype; opposite haplotypes can combine; conflicts keep first or raise under overlap-error; excluded ALT does not claim a position; filtering, masks and strict-warning escalation | `test_duplicate_consensus`, `test_sv_scenarios`, `test_configuration_errors` |
| Structural reconstruction | Literal insertions/deletions, MNVs, precise symbolic DEL; variant-class groups and aliases; nested/overlapping edits; event spans at window boundaries; upstream events; start/center/explicit anchors; deleted anchor moves right or errors; contig-end crop/padding | `test_edits`, `test_structural`, `test_structural_edges`, `test_sv_scenarios`, `test_structural_oracle`, `test_synthetic_structural_matrix` |
| Reference validation | REF mismatch, ambiguity in FASTA or REF, skip/warn/error policies, missing contigs, contig-name diagnostics, untouched reference positions and padding zeros | `test_reference`, `test_variants`, `test_structural`, `test_cli` |
| Site filtering | PASS and unfiltered '.', QUAL thresholds and missing QUAL; AF min/max and inclusive boundaries; INFO, samples and per-record auto fallback; simultaneous filters; missing/non-finite INFO AF; invalid frequencies and non-finite QUAL | `test_features`, `test_review_fixes`, `test_configuration_errors` |
| Regions and masks | 1-based closed versus BED half-open coordinates; strand; centered windows; explicit anchors; compressed region/mask files; invalid/empty region files; malformed coordinate types; masked variants and zero channels | `test_regions`, `test_features`, `test_cli`, `test_review_fixes`, `test_configuration_errors` |
| Sample selection and input | Reordered subsets, sample files, empty/duplicate/unknown IDs; malformed API sample requests; indexed VCF and BCF equivalence; missing inputs; reusable reference reader | `test_cli`, `test_variants`, `test_regression_fixes`, `test_pipeline`, `test_configuration_errors` |
| Batching and memory | Explicit, auto, all-sample 'none' and oversized batches; batch invariance; float16/32; both layouts; reader threads; transient copies and genotype memory; constrained/large simulated RAM and sample-count ceiling | `test_memory`, `test_pipeline`, `test_synthetic_configuration_matrix` |
| Writers and transactions | NPY, NPZ, HDF5, Torch; aggregate/per-sample and split batches; HDF5 gzip/lzf/chunks and clamped valid oversize chunks; sample-name sanitization/collisions; replace/error overwrite; rollback on write/commit failure; earlier outputs preserved | `test_writers`, `test_overwrite`, `test_commit_rollback`, `test_review_fixes`, `test_configuration_errors` |
| Metadata and CLI | Tensor dimensions, sample order/count/ploidy, filtering statistics, policy settings, strand, memory/timing estimates; no sidecar option; dry run writes nothing; mutually exclusive selectors; duplicate region outputs; clean exit code 2 versus debug traceback | `test_cli`, `test_pipeline`, `test_overwrite`, `test_api_contract` |
| Low-level array API | Position bounds, duplicates, integer positions, allele/genotype table shape and codes, floating channels, sample ranges, major-consensus frequency shape and finite 0–1 values | `test_api_contract`, `test_encoding` |

## All 41 RunConfig fields

`test_every_config_field_has_a_coverage_category` compares this inventory with the
actual dataclass fields, so adding a new field requires updating the audit.
This checks inventory completeness, not numerical code-coverage percentage.

| Group | Fields |
| --- | --- |
| Inputs | `vcf_path`, `fasta_path`, `region`, `reference_reader`, `samples`, `mask_path` |
| Encoding | `mode`, `dtype`, `output_layout`, `missing_policy`, `unphased_policy`, `random_seed`, `ref_mismatch_policy`, `ambiguous_ref_policy`, `haploid_policy`, `ploidy_policy`, `het_policy` |
| Selection | `variant_classes`, `biallelic_only`, `mixed_record_policy`, `pass_only`, `min_qual`, `min_af`, `max_af`, `af_source`, `strict` |
| Structural | `anchor`, `anchor_deletion_policy`, `overlap_policy` |
| Execution | `batch_size`, `memory_fraction`, `threads`, `dry_run` |
| Output | `output_path`, `output_format`, `per_sample`, `metadata_json`, `overwrite`, `h5_compression`, `h5_chunk_samples`, `h5_chunk_length` |

## Combinations that must produce an error

1. Length-changing classes (`del`, `ins`, `symbolic-del`, `indel`, `all`) with
   dosage or consensus. MNVs remain valid for fixed-length modes.
2. `het_policy='reference'` or `'major'` outside consensus. The default
   `'alternate'` remains accepted in other modes for shared default configs.
3. HDF5 compression/chunk options with a non-HDF5 output format.
4. Unknown policies, mode, dtype, layout, format, compression or variant class.
5. Nonpositive/noninteger threads, batch counts or HDF5 chunk sizes; booleans
   masquerading as integers. Batch size `None`/'auto' means automatic selection;
   the explicit string 'none' means all samples in one batch.
6. Non-finite/out-of-range AF or memory fraction; memory fraction must be
   >0 and <=1; QUAL threshold must be finite and >=0; min_af must be <=max_af.
7. String/numeric values used instead of boolean flags; empty/bad path types;
   a non-Region object supplied as `region`; malformed/noninteger coordinates.
8. A plain string instead of a sequence of sample IDs; empty IDs, duplicate
   selections, unknown samples, or simultaneous CLI --samples/--sample-file.
9. An explicit anchor outside the region; an anchor deleted under the error
   policy; actual overlapping claims under overlap-error.
10. Missing/half-called GT under missing-error; unphased heterozygotes in
    haplotype mode under phase-error; phase-to-missing with missing-error.
11. Haploid GT under haploid-error; greater-than-diploid GT under ploidy-error;
    undeclared allele indices; absent/malformed FORMAT/GT on retained records.
12. INFO-only AF filtering without usable INFO AF; invalid finite-range INFO AF
    when used by the filter. Missing NaN AF follows the missing-field rules;
    auto mode can fall back to the selected samples.
13. Empty/partially retained regions when strict mode escalates their warnings;
    existing output under overwrite-error; duplicate region output names.
14. Invalid low-level array shapes/codes/positions/sample ranges or missing/bad
    frequencies supplied to the major-consensus encoder.

Input-independent configuration errors are checked before opening FASTA/VCF.
Data-dependent errors require reading the record; tests check that failure does
not publish a new tensor. Multi-region runtime failures do not roll back already
completed regions; the output-conflict preflight is checked before any region.

## Unusual combinations that are intentionally valid

- Biallelic-only plus apply-selected: the whole multiallelic record is excluded;
  the mixed opt-in does not bypass the biallelic filter.
- Dosage/consensus on unphased genotypes: there is no haplotype accuracy claim.
- Haploid blank/duplicate options with dosage: either yields full-weight dosage.
- Strict biallelic filtering: deliberate site filtering does not itself fail.
- Equal min/max AF bounds: an inclusive single-frequency filter.
- Oversized positive batch/chunk sizes: clamp to the relevant sample/dimension.
- Zero QUAL threshold, AF endpoints 0/1 and memory_fraction=1.
- Forward/reverse strands with either layout, batch counts or sample ordering.
- HDF5 compression with per-sample output, either layout and small batches.
- Dry run with overwrite-replace: validate/report without writing outputs.

## Issues fixed during this audit

- **Mixed ALT projection:** REF-looking excluded ALT slots were still considered
  ALT genotype codes. This affected reference/alternate/major consensus,
  phase checking and duplicate claiming. Excluded calls are now mapped to
  genotype 0 before those decisions. Tested against independent VCF projection
  and native bcftools for both ALT orders.
- **Literal SNV scope:** selected-SNV mode now also excludes multi-base REF
  representations; broader variant-class mode retains normalized substitutions.
- **Non-finite record values:** NaN INFO/AF could bypass thresholds; NaN QUAL
  could pass minimum quality. Missing AF now follows missing/fallback semantics;
  invalid nonmissing AF raises; non-finite QUAL fails an active quality filter.
- **Configuration validation:** reversed AF bounds, non-finite values, invalid
  counts/flags/coordinates and ignored format/mode options now fail explicitly.
- **Input diagnostics:** absent/malformed GT now raises a contextual ValueError
  rather than an uncaught generic parser exception.
- **Direct helper validation:** extraction shares the site-option checks;
  HDF5 chunk overrides and major-consensus frequency values are validated.

This revision changes selected-mixed-record decisions and strengthens validation.
Previously ignored nondefault consensus/HDF5 options now raise. Use both updated
source and tests, reinstall the editable package, and rerun relevant paper runs
with source version 0.7.5. No GTEx/MAGE paper runs were revalidated in this audit.

## Limits

These tests combine isolated cases, shared fixture matrices, native comparisons
and independent randomized text-edit oracles. They do not enumerate every
Cartesian product, every VCF parser corruption, every external-library version,
or every SV representation. They establish specified reconstruction behavior,
not biological ground-truth phasing or cohort accuracy. Continuous overlap chains
near region edges retain the documented whole-file comparison limitation.


## Review follow-up

The suite contains 517 test functions and 1,628 parameterized cases. The two
large matrix modules contribute 589 cases (36.2%); functions also need not map
one-to-one to distinct behaviors. Report behavior categories separately.
`test_review_coverage.py` covers additional thread counts (4/8), BCF mixed-record
policies and structural edits, mask plus per-sample layouts, zero-variant
consensus, unsupported ploidy policy and cross-contig single-region rejection,
and separate CLI regions on two contigs. Existing structural lookback tests
remain independent of reported working-window metadata.
Native comparator versions are retained automatically in JUnit properties and
terminal output. Checkout-only comparator imports are deferred to a fixture;
missing validation files skip five cases without blocking other collection.


CLI help coverage also checks five terminal widths (COLUMNS 40/60/80/100/120).
The rendered-help assertion joins hyphenated names wrapped across physical
lines, avoiding a formatting-dependent failure. This adds four cases to the
previous 1,596-case suite; the complete 1,600-case suite passes with COLUMNS=100.


## Random-assignment extension

The explicit `random` unphased policy requires a seed and haplotype mode.
`test_random_assignment.py` adds 28 cases, including known-phase and dosage
preservation, batch/thread/sample/window/input-format reproducibility, seed
validation, mixed-allele projection, structural calls, audit metadata and CLI.
Native consensus on a derived VCF with recorded assignments supplies one
additional paired test and six native calls. Random assignments are
pseudo-phasing, not biological haplotype truth.
