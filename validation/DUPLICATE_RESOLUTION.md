# Duplicate reconstruction validation — Hottools 0.7.1

Duplicate coordinates are resolved per sample/haplotype in input order: REF calls
permit later ALTs, the first applicable ALT wins, and later conflicting calls
are skipped with existing overlap warnings/counters. Compatible opposite-track
calls survive. Substitution rows are combined after conflict resolution, with
allele order retained by first appearance. Filters and REF/genotype validation
run before records can suppress another. Pure SNV windows only sweep duplicate
loci, preserving the fast path for unique positions.

The full updated test suite has 1,019 passing tests, including 16 new duplicate
regressions. Assertions requiring the prior drop-all behavior were changed.
Native bcftools 1.24 comparisons passed for 49 duplicate-SNV genotype pairs
(98 full haplotype windows plus dosage outputs) covering fully called, missing
and half-called GTs with missing_policy=reference. Ten same-anchor structural
scenarios (80 full haplotype windows) also matched: insertion/deletion conflicts,
shared SNV/deletion anchors, MNV/SNV overlaps and opposite-track compatibility.

The paper benchmark additionally passed six planted-data joint cases at N=2/49/60,
including donor resampling and all-sample independent duplicate-policy checks.
No real GTEx/MAGE reconstruction was rerun locally. Original HPC cohort
mismatch records were not supplied. These checks validate the stated
reconstruction policy; they do not establish biological correctness of the first
call when the input is contradictory. Missing/phase/ploidy policies remain
explicit and can differ from bcftools outside the matched diploid scope.

Use a fresh paper-benchmark run after installing this version. Existing completed
case results must not be reused as evidence for changed reconstruction code.
