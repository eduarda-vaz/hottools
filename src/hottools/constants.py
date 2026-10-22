from __future__ import annotations

from typing import Literal

import numpy as np

BASES = "ACGT"
VALID_BASES = frozenset(BASES)

# ASCII -> 0..3 for A/C/G/T (upper- and lower-case, so soft-masked FASTA is
# treated as sequence rather than as ambiguity); 255 for everything else.
BASE_TO_INDEX = np.full(256, 255, dtype=np.uint8)
for _idx, _base in enumerate(BASES):
    BASE_TO_INDEX[ord(_base)] = _idx
    BASE_TO_INDEX[ord(_base.lower())] = _idx

AMBIGUOUS_INDEX = np.uint8(255)

# ------------------------------------------------------------------
# Allele slots
# ------------------------------------------------------------------
# One genotype slot holds either an allele index (0 = REF, 1..3 = ALT)
# or one of two distinct "no allele" markers:
#
#   ALLELE_MISSING   the VCF wrote '.' here -- the allele exists but was
#                    not called
#   ALLELE_ABSENT    there is no such allele -- the sample is haploid at
#                    this record
#
# Keeping those two apart is what stops a hemizygous chrX/chrY call from
# being indistinguishable from a half-called diploid genotype.
MAX_ALT_ALLELES = 3
N_ALLELES = 1 + MAX_ALT_ALLELES              # REF + up to three ALTs
ALLELE_MISSING = np.uint8(N_ALLELES)         # 4
ALLELE_ABSENT = np.uint8(N_ALLELES + 1)      # 5
N_ALLELE_SLOTS = N_ALLELES + 2               # 6

#: Marks an unused column in the per-variant allele table.
NO_ALLELE = np.uint8(255)

# A genotype is two slots packed into one byte as ``a1 * 6 + a2``, so the
# compact genotype array stays V x S uint8 no matter how many ALT
# alleles a record carries.
N_GENOTYPE_CODES = N_ALLELE_SLOTS * N_ALLELE_SLOTS   # 36

#: Slot values of the first and second allele, indexed by packed code.
CODE_ALLELE1 = (np.arange(N_GENOTYPE_CODES) // N_ALLELE_SLOTS).astype(np.uint8)
CODE_ALLELE2 = (np.arange(N_GENOTYPE_CODES) % N_ALLELE_SLOTS).astype(np.uint8)

#: Number of allele slots that actually exist, per packed code: 2 for a
#: diploid genotype and 1 for a hemizygous one. This is the divisor the
#: dosage encoding uses, so a hemizygous ALT call is full-weight ALT
#: rather than half-weight -- ``ALLELE_ABSENT`` means the allele does
#: not exist, not that it is unknown.
CODE_PLOIDY = (
    (CODE_ALLELE1 != ALLELE_ABSENT).astype(np.uint8)
    + (CODE_ALLELE2 != ALLELE_ABSENT).astype(np.uint8)
).clip(1)

GENOTYPE_UNKNOWN = np.uint8(
    ALLELE_MISSING * N_ALLELE_SLOTS + ALLELE_MISSING
)


def pack_genotype(allele1, allele2):
    """Pack two allele slots into a single uint8 genotype code."""
    return np.uint8(allele1) * np.uint8(N_ALLELE_SLOTS) + np.uint8(allele2)


# Diploid genotype rows from cyvcf2 are [allele1, allele2, phased]; the
# phase flag is always the last entry, so ploidy is len(row) - 1.
PHASE_FIELDS = 1
DIPLOID_GENOTYPE_WIDTH = 2 + PHASE_FIELDS
HAPLOID_GENOTYPE_WIDTH = 1 + PHASE_FIELDS


# ------------------------------------------------------------------
# Mode and policy vocabularies
# ------------------------------------------------------------------
# Defined once, here, and imported everywhere else. The Literal aliases
# give type checkers the vocabulary; the tuples are what runtime
# validation checks against. Keeping each pair side by side is what
# stops a mode or policy from being accepted in one module and missing
# from another, which is how ``consensus`` was once absent from the
# pipeline's own ``Mode`` type.

Mode = Literal["dosage", "haplotypes", "consensus"]
MODES: tuple[str, ...] = ("dosage", "haplotypes", "consensus")

OutputLayout = Literal["samples_first", "channels_first"]
LAYOUTS: tuple[str, ...] = ("samples_first", "channels_first")

OutputFormat = Literal["npy", "npz", "h5", "pt"]
FORMATS: tuple[str, ...] = ("npy", "npz", "h5", "pt")

#: Floating dtypes the encoders produce. Integer dtypes are refused:
#: dosage is fractional, so an integer tensor would silently truncate a
#: heterozygote to zero.
DTYPES: tuple[str, ...] = ("float32", "float16")

#: ``error`` refuses any missing allele call; ``zero`` leaves it as an
#: all-zero channel vector; ``reference`` fills it with the REF base.
MissingPolicy = Literal["zero", "reference", "error"]
MISSING_POLICIES: tuple[str, ...] = ("zero", "reference", "error")

MixedRecordPolicy = Literal["exclude", "apply-selected"]
MIXED_RECORD_POLICIES: tuple[str, ...] = ("exclude", "apply-selected")

UnphasedPolicy = Literal["error", "missing", "random"]
UNPHASED_POLICIES: tuple[str, ...] = ("error", "missing", "random")

#: For per-record problems: REF mismatch and ambiguous FASTA bases.
RecordPolicy = Literal["error", "skip", "warn"]
RECORD_POLICIES: tuple[str, ...] = ("error", "skip", "warn")

HaploidPolicy = Literal["duplicate", "blank", "error", "skip"]
HAPLOID_POLICIES: tuple[str, ...] = ("duplicate", "blank", "error", "skip")

#: The subset of haploid policies that reach the encoder; ``error`` and
#: ``skip`` are resolved during extraction.
HaploidTrackPolicy = Literal["duplicate", "blank"]
HAPLOID_TRACK_POLICIES: tuple[str, ...] = ("duplicate", "blank")

PloidyPolicy = Literal["error", "skip"]
PLOIDY_POLICIES: tuple[str, ...] = ("error", "skip")

HetPolicy = Literal["reference", "alternate", "major"]
HET_POLICIES: tuple[str, ...] = ("reference", "alternate", "major")

AfSource = Literal["auto", "info", "samples"]
AF_SOURCES: tuple[str, ...] = ("auto", "info", "samples")

OverwritePolicy = Literal["error", "replace"]
OVERWRITE_POLICIES: tuple[str, ...] = ("error", "replace")

#: Records claiming the same reference bases on one haplotype: ``first``
#: applies the first in POS order and sets the rest aside, as bcftools
#: consensus does; ``error`` refuses the run before anything is written.
OverlapPolicy = Literal["first", "error"]
OVERLAP_POLICIES: tuple[str, ...] = ("first", "error")

#: A deletion that removes the anchor base itself: ``right`` pins the
#: first surviving base after it instead; ``error`` refuses the run.
AnchorDeletionPolicy = Literal["right", "error"]
ANCHOR_DELETION_POLICIES: tuple[str, ...] = ("right", "error")


def check_choice(name: str, value, choices: tuple[str, ...]) -> str:
    """
    Return ``value`` if it is one of ``choices``; raise otherwise.

    Matching is exact. A near-miss such as ``"Reference"`` or ``"ref"``
    is an error rather than a silent fallback to some other policy.
    """
    if value not in choices:
        raise ValueError(
            f"{name} must be one of: {', '.join(choices)}; got {value!r}"
        )
    return value

ContigAliasPolicy = Literal["exact", "auto"]
CONTIG_ALIAS_POLICIES: tuple[str, ...] = ("exact", "auto")
