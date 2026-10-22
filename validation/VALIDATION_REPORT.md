> Historical validation: duplicate-position behavior described below predates
> Hottools 0.7.1. Same-position records are now resolved per haplotype.
> See validation/DUPLICATE_RESOLUTION.md for the new validation.

# Cross-validation of hottools against `bcftools consensus`

**Result: 843,194 bases compared across 323 synthetic cases, zero
mismatches** in both haplotype and dosage mode, covering single-base
substitutions through to insertions, deletions and symbolic deletions.
Eight defects were found and fixed in the process; none was reachable
from the unit suite, because each required an independent
implementation to expose.

Environment: hottools 0.6.0, bcftools 1.24 / htslib 1.23.1,
cyvcf2 0.34.0, pyfaidx 0.9.0.4, NumPy 2.5.3.

---

## Length-changing variant classes (0.5.0)

### Method

A deletion inside a fixed-length window pulls sequence in from
downstream, so the two tools cannot be handed the same interval.
hottools reads a *working* window wider than its output by the largest
number of reference bases any one haplotype could lose, and the harness
gives `bcftools consensus` that same larger interval:

```
samtools faidx ref.fa CHROM:1-(WINDOW+PAD) | bcftools consensus -s S -H h
```

The first `WINDOW` bases of bcftools' answer are the expectation. For a
minus-strand case the window is cut from the plus strand first and only
then reverse-complemented, which is the order hottools uses. Dosage mode
is not compared for these cases because hottools refuses to produce it.

### Result

323 concordance cases (23 hand-written + 300 randomised), 843,194 bases
compared, **zero mismatches**. The randomised structural cases draw
their own contig name, window length, sample count and record mix across
all five classes, on either strand, with records spaced so their
normalised intervals stay disjoint.

Named structural cases: `deletion_only`, `insertion_only`, `mnv_blocks`,
`symbolic_deletions`, `every_class_together`, `adjacent_deletions`,
`deletion_reverse_strand`, `structural_centre_anchor`.

### Defects this round found

Cross-validation again found things the unit tests did not, and most of
them were in logic the SNV-only path shares:

1. **A record reaching into the window from outside was dropped.** Only
   a multi-base REF span can straddle an edge, so this was unreachable
   while hottools encoded substitutions alone. A carrier was written as
   reference.
2. **A symbolic `<DEL>` is in scope by its `END`.** Testing scope with
   `len(REF)` discarded a `<DEL>` whose `POS` sat just before the
   window, losing bases it removes inside it.
3. **Overlap is decided on normalised intervals.** Two deletions that
   share only an anchor base do not conflict, and bcftools applies both.
   A raw REF-span test concludes the opposite and suppresses the second.
   This was the single largest source of divergence: it accounted for
   most of a 38%-of-bases disagreement in the dense-overlap regime.
4. **A substitution block keyed off `POS`.** An MNV with shared flanking
   bases was written at the wrong coordinate.
5. **Genotype patches applied after the matrix was assembled.** The
   per-haplotype overlap resolution reached the structural stage but not
   the substitution stage, so an MNV whose carrier should have kept the
   reference was partly applied.
6. **An edit on a record with no substitution entry was dropped.**
   Surviving edits were inferred from kept positions, and a deletion
   whose `POS` lies before the window keeps no position.

### Deliberate divergences

- **`<DEL>` without `INFO/END`**: hottools refuses the record, because
  the deleted interval is undefined. bcftools emits the reference.
- **Length-changing classes in dosage or consensus mode**: hottools
  errors before reading anything. There is no position-by-position
  average of two sequences of different lengths.
- **A second record claiming a position an earlier multi-base block
  already owns**: hottools drops it for every sample rather than
  applying it partially, because one reference position carries one
  allele table. bcftools keeps it for haplotypes that do not carry the
  first record.
- **Overlapping records in continuous chains**: which record wins
  depends on the chain leading up to it. hottools seeds the chain at the
  nearest upstream position no interval straddles - exact whenever
  records are separated by at least one base anywhere upstream - and
  warns when no such gap exists within its lookback.

### Dense-overlap stress test

A separate fuzz run placed records 1-6 bp apart with deletions of up to
10 bp, so their intervals overlap continuously - input no variant caller
produces and which bcftools itself reports as "overlaps with another
variant, skipping". Measured over 160 haplotypes:

| Record spacing | Haplotypes exact | Bases differing |
| --- | --- | --- |
| 6-20 bp apart | 160/160 | 0 of 24,000 |
| 1-6 bp apart | 149/160 | 1,140 of 24,000 (4.8%) |

The residual is localised, not a frame shift, and hottools reports the
condition through `variant_overlapping_records_suppressed`,
`variant_overlapping_blocks_dropped` and a `StructuralEditWarning`.

## 1. What is being compared, and why it is a fair test

`bcftools consensus` is an independent implementation of the same
primitive: apply a sample's alleles to a reference interval. Its
haplotype output is a sequence, hottools' is a one-hot tensor, so the
comparison decodes the tensor back to sequence.

```bash
samtools faidx ref.fa CHROM:START-END \
  | bcftools consensus -s SAMPLE -H {1,2} variants.vcf.gz
```

For every case, every sample, and both haplotypes:

1. **Haplotype mode.** `hottools --mode haplotypes` produces
   `S x 2 x L x 4`. Each `L x 4` track is decoded (`argmax` where the
   row sums to 1, `N` where it sums to 0) and compared character by
   character against the corresponding `bcftools -H` sequence.
2. **Dosage mode.** `hottools --mode dosage` produces `S x L x 4`. It is
   checked against the *same two bcftools haplotypes*: where they agree
   the channel vector must be one-hot on that base; where they differ it
   must be exactly 0.5/0.5 on those two bases and zero elsewhere.

`--missing-policy reference` is used throughout, because that is the
policy whose semantics match bcftools (a missing allele falls back to
REF). The `zero` policy has no bcftools equivalent and is asserted in
the unit suite instead.

Three choices keep the comparison honest rather than circular:

- The expected values come from bcftools, not from a re-implementation
  of hottools' own logic.
- Where hottools deliberately filters records (indels, multiallelics),
  bcftools is handed the matching subset via
  `bcftools view -v snps -m2 -M2`, so both tools see the same records.
- Reverse-strand cases are compared against the reverse complement of
  the bcftools output, computed independently of hottools' transform.

---

## 2. Concordance battery

| case | content | bases | mismatches |
|---|---|---:|---:|
| `basic_phased` | 25 phased SNVs, 6 samples, every zygosity | 3,600 | 0 |
| `naming_unprefixed` | identical content on contig `1` rather than `chr1` | 3,600 | 0 |
| `mixed_separators` | `\|` and `/` in the same file and same sample | 1,200 | 0 |
| `all_substitutions` | all 12 ordered REF→ALT base pairs | 800 | 0 |
| `boundary_positions` | variants at positions 1–3, mid-contig, final base | 720 | 0 |
| `missing_genotypes` | `.\|.`, `./.`, `.\|1`, `1\|.`, `0\|.`, `.\|0` | 1,200 | 0 |
| `many_samples` | 24 samples × 15 phased SNVs | 9,600 | 0 |
| `dense_cluster` | 40 consecutive variant positions | 800 | 0 |
| `soft_masked_reference` | lower-case reference stretch, variants inside | 900 | 0 |
| `ambiguous_reference` | 10 bp N-run, variants outside it | 900 | 0 |
| `reverse_strand` | minus-strand encoding of case 1 | 3,600 | 0 |
| `lowercase_vcf_alleles` | lower-case REF/ALT mixed with upper-case | 960 | 0 |
| `multiallelic_snvs` | two ALT alleles per record, all genotype combinations | 1,680 | 0 |
| `haploid_contig` | every call haploid, as on chrY | 960 | 0 |
| `indels_and_multiallelic` | indels + a multiallelic site among SNVs | 1,200 | 0 |
| **randomised × 200** | see below | **654,632** | **0** |
| **total** | | **686,352** | **0** |

### Randomised sweep

200 cases drawn from a seeded generator, each with its own contig name
(`chr1`, `1`, `chrX`, `scaffold_7`, `NC_000001.11`), length (80–400 bp),
sample count (1–12), variant density (up to one variant per 8 bp),
strand (25% minus), and a 30% chance of a soft-masked stretch. Per
genotype: 10% missing or half-missing, 20% unphased, 70% phased.

Unphased calls in the sweep are constrained to be homozygous. That is
not a convenience — it is the only situation in which `bcftools
consensus -H` is well defined without phase, so comparing unphased
heterozygotes would be comparing against an arbitrary choice rather than
a ground truth. hottools' behaviour on unphased heterozygotes is covered
in §4 instead.

---

## 3. Defects found by cross-validation

### 3.1 Half-called genotypes were inconsistent between modes

`.|1` carries one known ALT allele. Haplotype mode preserved it
(REF/ALT, agreeing with bcftools); dosage mode called the sample
**hom-ref**, silently discarding the ALT.

Root cause: dosage read `cyvcf2.Variant.gt_types`, which with
`strict_gt=True` collapses every partially-called genotype to `UNKNOWN`;
`--missing-policy reference` then rewrote `UNKNOWN` to hom-ref. The
haplotype path worked on allele indices and only replaced the allele
that was actually missing.

Observed on `missing_genotypes`, sample `M1` at position 50
(`.|1`, REF=G, ALT=C):

| | haplotype mode | dosage mode |
|---|---|---|
| before | `(G, C)` — heterozygous | `G: 1.0` — hom-ref |
| after | `(G, C)` | `G: 0.5, C: 0.5` |

**Fix.** Two genotype codes were added, `GT_HALF_REF = 4` and
`GT_HALF_ALT = 5`, refined from the allele array only when `gt_types`
reports `UNKNOWN` — so the common no-missing path is untouched. The
dosage lookup table grew from 4 to 6 rows:

| code | genotype | `--missing-policy zero` | `--missing-policy reference` |
|---|---|---|---|
| 0 | hom-ref | REF | REF |
| 1 | het | ½REF + ½ALT | ½REF + ½ALT |
| 2 | hom-alt | ALT | ALT |
| 3 | both alleles missing | 0 | REF |
| 4 | one REF called, one missing | ½REF | REF |
| 5 | one ALT called, one missing | ½ALT | ½REF + ½ALT |

Each allele contributes 0.5, which makes the dosage tensor the exact
superposition of the two haplotype tensors for *every* genotype. That is
now a unit test
(`test_dosage_is_the_superposition_of_the_two_haplotypes`), run over all
nine allele-pair combinations under both missing policies — it is the
invariant the bcftools comparison depends on, so it is pinned
structurally rather than only observed empirically.

A `half_calls` counter was added to `VariantStats`, and both modes now
report `missing_genotypes` (both alleles absent) and `half_calls`
(exactly one) identically for the same input. `--missing-policy error`
fires on either.

### 3.2 Lower-case REF/ALT was silently dropped

The VCF specification calls for upper-case bases, but htslib tolerates
lower case and `bcftools consensus` applies such records. hottools
tested `ref not in VALID_BASES` against an upper-case-only set, so a
lower-case record fell through to the `non_biallelic_snv` counter with
no specific signal — the variant simply vanished from the output.

This was inconsistent on its own terms: hottools already accepts
soft-masked (lower-case) *reference* sequence as real bases.

**Fix.** REF/ALT are upper-cased before the alphabet check, and a
`lowercase_alleles` counter records how often it happened. A dedicated
concordance case (`lowercase_vcf_alleles`, 960 bases) now matches
bcftools exactly.

---

## 4. Deliberate divergence on malformed input

These cases are not failures of either tool; they are policy
differences, measured rather than assumed. Each was probed across every
sample and both haplotypes, because bcftools only validates a record
when the requested haplotype actually carries the ALT allele — a single
`-s`/`-H` probe silently skips the interesting record.

| case | hottools | bcftools | assessment |
|---|---|---|---|
| unphased heterozygote | rejects, naming the position | applies the alleles in listed order | hottools refuses to invent phase |
| haploid call | **accepts**, pads to two tracks, warns | `-H 1` applies the allele, `-H 2` falls back to the reference | policy difference; see §5 |
| triploid call | rejects, naming the sample | applies alleles 1–2 | hottools encodes at most two alleles |
| duplicate positions | drops every record at the position | applies the first, warns `overlaps with another variant, skipping` | hottools is the more conservative choice |
| `*` spanning deletion | rejected as non-SNV | applied | correct for a one-hot SNV encoder |
| `<DEL>` / `<NON_REF>` | rejected as non-SNV | applied | same |
| IUPAC REF (`R`) | rejected as non-ACGT | **fails 3/6 runs** with a REF-mismatch error | both refuse; hottools refuses deterministically |
| VCF REF ≠ FASTA REF | rejects up front | **fails 3/6 runs**, succeeds on the 3 that never needed the allele | hottools fails for all samples, bcftools only for some |
| variant on reference `N` | rejects, naming the base | **fails 3/6 runs** with a REF-mismatch error | same |
| contig naming mismatch | rejects with a `chr`-prefix hint | `samtools faidx` returns an empty sequence | hottools catches it before any work |
| empty region | warns, `--strict` escalates | returns the reference | see below |
| lower-case REF/ALT | **accepts** (§3.2) | accepts | now concordant |

The three REF-validation rows are the most consequential. bcftools
produced a valid-looking sequence for exactly the three of six
sample/haplotype runs that did not happen to need the mismatching
allele. A pipeline that only ran those three would never learn the
reference was wrong. hottools refuses the whole run.

---

## 5. What the 0.4.0 feature round changed here

Extending hottools invalidated two assumptions baked into this harness.
Both were harness bugs, not package defects, and both are worth
recording because they are the kind of thing a validation suite hides
from itself.

**The pre-filter was stripping the new capability.** The
`indels_and_multiallelic` case handed bcftools
`bcftools view -v snps -m2 -M2`, so that both tools saw the record set
hottools could encode. Once hottools learned to encode multiallelic
SNVs, `-m2 -M2` was removing exactly the records it had just gained,
and the case failed with four haplotype mismatches. The pre-filter is
now `-v snps`, and a dedicated `multiallelic_snvs` case was added.

**`bcftools consensus -H 2` does not duplicate a haploid allele.**
Asked for a second haplotype that does not exist, bcftools returns the
reference base; hottools' `--haploid-policy duplicate` returns the
called allele. Verified directly on a hemizygous ALT call: bcftools
`-H 1` gave the ALT base and `-H 2` gave the reference. This is a
genuine policy difference, not an error on either side, so haploid
cases are compared on haplotype 1 only — which still validates the
haploid decoding — and the divergence is recorded in §4.

## 6. Reproducing

```bash
python validation/compare_with_bcftools.py 200   # 323 cases, ~2.5 min
```

Writes `validation/concordance.json` and `validation/edge_cases.json`,
and exits non-zero if any case mismatches. Requires `bcftools`,
`samtools`, `tabix` and `bgzip` on `PATH`. All inputs are generated from
seeded RNGs; nothing binary is checked in.

`validation/synthetic_cases.py` holds the case definitions and can be
imported on its own to materialise the battery for other purposes.

---

## 7. State after this round

| | before this round | after |
|---|---:|---:|
| unit tests | 180 | **280** |
| line coverage | 95% | **95%** |
| named concordance cases | 13 | **15** |
| bases cross-validated | 683,712 | **686,352** |
| open findings | 0 | 0 |

Every feature has end-to-end tests in `tests/test_features.py`, and the
nine defects found in review of 0.4.0, plus the six found in a second
audit pass, have regression tests in `tests/test_review_fixes.py`.

`ruff check src tests` is clean. The two defects in §3 are fixed, tested
and listed in `CHANGELOG.md`.
