"""
Synthetic VCF/FASTA case battery for validating hottools against
``bcftools consensus``.

Every case is generated from a seeded RNG, so the battery is
reproducible and carries no binary test data. Cases are split into:

``CONCORDANCE``
    Clean enough that hottools and bcftools must agree base-for-base.

``EDGE``
    Malformed, exotic or policy-divergent input. These assert hottools'
    documented behaviour and record what bcftools does for contrast.
"""
from __future__ import annotations

import random
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

BASES = "ACGT"
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
OTHER = {b: [x for x in BASES if x != b] for b in BASES}


# ------------------------------------------------------------------
# file construction
# ------------------------------------------------------------------

def write_fasta(path: Path, contig: str, sequence: str, width: int = 60) -> Path:
    lines = [sequence[i : i + width] for i in range(0, len(sequence), width)]
    path.write_text(f">{contig}\n" + "\n".join(lines) + "\n", encoding="utf-8")
    _run(["samtools", "faidx", str(path)])
    return path


def write_vcf(
    path: Path,
    contig: str,
    contig_length: int,
    samples: list[str],
    rows: list[str],
    extra_header: tuple[str, ...] = (),
) -> Path:
    """Write, bgzip and tabix-index a VCF. ``path`` ends in ``.vcf.gz``."""
    header = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID={contig},length={contig_length}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        *extra_header,
    ]
    columns = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
    if samples:
        columns += "\t" + "\t".join(samples)
    body = "\n".join([*header, columns, *sorted(rows, key=_pos_key)])
    plain = path.parent / path.name[: -len(".gz")]
    plain.write_text(body + "\n", encoding="utf-8")
    _run(["bgzip", "-f", str(plain)])
    _run(["tabix", "-f", "-p", "vcf", str(path)])
    return path


def _pos_key(row_text: str) -> tuple[int, str]:
    fields = row_text.split("\t")
    return (int(fields[1]), fields[4])


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    if shutil.which(cmd[0]) is None:
        raise RuntimeError(f"{cmd[0]} is required to build the case battery")
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def row(
    pos: int, ref: str, alt: str, gts: list[str], info: str = "."
) -> str:
    return (
        f"\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info}\tGT\t"
        + "\t".join(gts)
    )


#: A symbolic deletion needs INFO/END declared in the header.
END_HEADER = [
    '##INFO=<ID=END,Number=1,Type=Integer,Description="End position">'
]


# ------------------------------------------------------------------
# case definition
# ------------------------------------------------------------------

@dataclass
class Case:
    name: str
    description: str
    contig: str
    sequence: str
    samples: list[str]
    rows: list[str]
    #: hottools flags beyond the common set.
    hottools_flags: list[str] = field(default_factory=list)
    #: Pre-filter the VCF with `bcftools view` before handing it to
    #: `bcftools consensus`, so both tools see the same record set.
    bcftools_prefilter: list[str] | None = None
    #: Expected outcome for an EDGE case: "ok" or a substring of the
    #: hottools error message.
    expect: str = "ok"
    #: Why hottools and bcftools are not expected to agree.
    divergence: str = ""
    strand: str = "+"
    #: Contig name to write into the FASTA, when it must differ.
    fasta_contig: str | None = None
    #: Value for ``--variant-classes``; None leaves the SNV-only
    #: default in place.
    variant_classes: str | None = None
    #: A length-changing case. Only haplotype mode is comparable, and
    #: bcftools has to be given the larger working window hottools read.
    structural: bool = False
    #: Extra VCF header lines this case needs.
    extra_header: list[str] = field(default_factory=list)
    #: Output window length. Shorter than the sequence for a
    #: length-changing case, so a deletion has real reference left to
    #: pull downstream bases from.
    out_length: int | None = None
    #: Which bcftools haplotypes are comparable. Haploid data has no
    #: second allele: bcftools -H 2 falls back to the reference, while
    #: hottools duplicates the called allele, so only -H 1 is a fair
    #: comparison there.
    haploid_source: bool = False

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def window(self) -> int:
        """Length of the output window."""
        return self.out_length or self.length

    def materialise(self, root: Path) -> MaterialisedCase:
        directory = root / self.name
        directory.mkdir(parents=True, exist_ok=True)
        fasta = write_fasta(
            directory / "ref.fa", self.fasta_contig or self.contig,
            self.sequence,
        )
        rows = [self.contig + r for r in self.rows]
        vcf = write_vcf(
            directory / "var.vcf.gz", self.contig, self.length,
            self.samples, rows, tuple(self.extra_header),
        )
        return MaterialisedCase(self, directory, fasta, vcf)


@dataclass
class MaterialisedCase:
    case: Case
    directory: Path
    fasta: Path
    vcf: Path

    def consensus_input(self) -> Path:
        """The VCF bcftools should consume (optionally pre-filtered)."""
        if not self.case.bcftools_prefilter:
            return self.vcf
        out = self.directory / "filtered.vcf.gz"
        _run([
            "bcftools", "view", *self.case.bcftools_prefilter,
            "-Oz", "-o", str(out), str(self.vcf),
        ])
        _run(["tabix", "-f", "-p", "vcf", str(out)])
        return out


# ------------------------------------------------------------------
# sequence helpers
# ------------------------------------------------------------------

def random_sequence(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(BASES) for _ in range(length))


def phased_genotypes(rng: random.Random, n: int) -> list[str]:
    return [rng.choice(["0|0", "0|1", "1|0", "1|1"]) for _ in range(n)]


# ------------------------------------------------------------------
# the battery
# ------------------------------------------------------------------

def build_concordance_cases() -> list[Case]:
    rng = random.Random(20240601)
    cases: list[Case] = []

    # 1. dense phased SNVs across a 300 bp window
    seq = random_sequence(rng, 300)
    samples = [f"S{i}" for i in range(1, 7)]
    positions = sorted(rng.sample(range(1, 301), 25))
    rows = [
        row(p, seq[p - 1], rng.choice(OTHER[seq[p - 1]]),
            phased_genotypes(rng, len(samples)))
        for p in positions
    ]
    cases.append(Case(
        "basic_phased",
        "25 phased biallelic SNVs, 6 samples, every zygosity",
        "chr1", seq, samples, rows,
    ))

    # 2. the same content on an unprefixed contig
    cases.append(Case(
        "naming_unprefixed",
        "Identical content with contig '1' instead of 'chr1'",
        "1", seq, samples, list(rows),
    ))

    # 3. mixed phase separators in one file; every '/' call is homozygous,
    #    so phase is irrelevant and haplotype mode must still succeed
    seq3 = random_sequence(rng, 200)
    samples3 = ["A1", "A2", "A3"]
    rows3 = [
        row(20, seq3[19], OTHER[seq3[19]][0], ["0|1", "0/0", "1|1"]),
        row(40, seq3[39], OTHER[seq3[39]][1], ["0/0", "1|0", "1/1"]),
        row(60, seq3[59], OTHER[seq3[59]][2], ["1|1", "0|1", "0/0"]),
        row(80, seq3[79], OTHER[seq3[79]][0], ["0/0", "1/1", "1|0"]),
        row(100, seq3[99], OTHER[seq3[99]][1], ["1|0", "0/0", "0|1"]),
    ]
    cases.append(Case(
        "mixed_separators",
        "'|' and '/' in the same file and the same sample; "
        "all unphased calls are homozygous",
        "chr1", seq3, samples3, rows3,
    ))

    # 4. every ordered REF->ALT substitution
    seq4 = ("ACGT" * 25)[:100]
    rows4 = []
    pos = 5
    for ref in BASES:
        for alt in OTHER[ref]:
            while seq4[pos - 1] != ref:
                pos += 1
            rows4.append(row(pos, ref, alt, ["0|0", "0|1", "1|0", "1|1"]))
            pos += 1
    cases.append(Case(
        "all_substitutions",
        "All 12 ordered REF->ALT base substitutions",
        "chr1", seq4, ["H1", "H2", "H3", "H4"], rows4,
    ))

    # 5. variants at both contig boundaries and in adjacent runs
    seq5 = random_sequence(rng, 120)
    rows5 = [
        row(p, seq5[p - 1], OTHER[seq5[p - 1]][0], ["0|1", "1|1", "0|0"])
        for p in (1, 2, 3, 59, 60, 61, 118, 119, 120)
    ]
    cases.append(Case(
        "boundary_positions",
        "Variants at positions 1-3, mid-contig and the final base",
        "chr1", seq5, ["B1", "B2", "B3"], rows5,
    ))

    # 6. every flavour of missing genotype
    seq6 = random_sequence(rng, 150)
    rows6 = [
        row(25, seq6[24], OTHER[seq6[24]][0], [".|.", "./.", "0|1", "1|1"]),
        row(50, seq6[49], OTHER[seq6[49]][1], [".|1", "1|.", "0|0", "./."]),
        row(75, seq6[74], OTHER[seq6[74]][2], ["0|.", ".|0", "1|0", ".|."]),
        row(100, seq6[99], OTHER[seq6[99]][0], ["0|0", "0|1", "1|1", "1|1"]),
    ]
    cases.append(Case(
        "missing_genotypes",
        "Fully missing, half-missing and phased/unphased missing calls",
        "chr1", seq6, ["M1", "M2", "M3", "M4"], rows6,
    ))

    # 7. many samples, to catch column-ordering errors
    seq7 = random_sequence(rng, 200)
    samples7 = [f"NA{1000 + i}" for i in range(24)]
    rows7 = [
        row(p, seq7[p - 1], rng.choice(OTHER[seq7[p - 1]]),
            phased_genotypes(rng, len(samples7)))
        for p in sorted(rng.sample(range(1, 201), 15))
    ]
    cases.append(Case(
        "many_samples",
        "24 samples x 15 phased SNVs",
        "chr1", seq7, samples7, rows7,
    ))

    # 8. a contiguous run of variant positions
    seq8 = random_sequence(rng, 100)
    rows8 = [
        row(p, seq8[p - 1], OTHER[seq8[p - 1]][p % 3],
            ["0|1", "1|0", "1|1", "0|0"])
        for p in range(30, 70)
    ]
    cases.append(Case(
        "dense_cluster",
        "40 consecutive variant positions",
        "chr1", seq8, ["D1", "D2", "D3", "D4"], rows8,
    ))

    # 9. soft-masked (lower-case) reference
    seq9 = random_sequence(rng, 150)
    seq9 = seq9[:50] + seq9[50:100].lower() + seq9[100:]
    rows9 = [
        row(p, seq9[p - 1].upper(), OTHER[seq9[p - 1].upper()][0],
            ["0|1", "1|1", "0|0"])
        for p in (25, 60, 75, 125)
    ]
    cases.append(Case(
        "soft_masked_reference",
        "Lower-case (repeat-masked) reference stretch with variants inside",
        "chr1", seq9, ["R1", "R2", "R3"], rows9,
    ))

    # 10. ambiguous reference bases away from any variant
    seq10 = random_sequence(rng, 150)
    seq10 = seq10[:60] + "N" * 10 + seq10[70:]
    rows10 = [
        row(p, seq10[p - 1], OTHER[seq10[p - 1]][0], ["0|1", "1|0", "1|1"])
        for p in (20, 40, 100, 140)
    ]
    cases.append(Case(
        "ambiguous_reference",
        "10 bp N-run in the reference, variants outside it",
        "chr1", seq10, ["N1", "N2", "N3"], rows10,
    ))

    # 11. reverse-strand output of case 1
    cases.append(Case(
        "reverse_strand",
        "Case 1 encoded on the minus strand",
        "chr1", seq, samples, list(rows),
        hottools_flags=["--strand", "-"], strand="-",
    ))

    # 12. indels and multiallelics alongside SNVs; bcftools is given the
    #     biallelic-SNV subset, which is what hottools encodes
    seq12 = random_sequence(rng, 200)
    rows12 = [
        row(20, seq12[19], OTHER[seq12[19]][0], ["0|1", "1|1", "0|0"]),
        row(40, seq12[39], seq12[39] + "TTA", ["0|1", "0|0", "1|1"]),
        row(60, seq12[59] + "AC", seq12[59], ["1|0", "0|1", "0|0"]),
        row(80, seq12[79],
            f"{OTHER[seq12[79]][0]},{OTHER[seq12[79]][1]}",
            ["0|1", "0|2", "1|2"]),
        row(100, seq12[99], OTHER[seq12[99]][2], ["1|1", "0|1", "1|0"]),
        row(150, seq12[149], OTHER[seq12[149]][0], ["0|0", "1|1", "0|1"]),
    ]
    # 13. lower-case REF/ALT: out of spec, accepted by htslib, and now
    #     accepted by hottools -- so the two tools must agree
    seq13 = random_sequence(rng, 120)
    rows13 = [
        row(p, seq13[p - 1].lower(), OTHER[seq13[p - 1]][0].lower(),
            ["0|1", "1|1", "0|0", "1|0"])
        for p in (15, 45, 95)
    ]
    rows13.append(
        row(70, seq13[69], OTHER[seq13[69]][1], ["1|1", "0|1", "0|0", "0|1"])
    )
    cases.append(Case(
        "lowercase_vcf_alleles",
        "Lower-case REF/ALT mixed with upper-case records",
        "chr1", seq13, ["L1", "L2", "L3", "L4"], rows13,
    ))

    # 14. multiallelic SNVs on their own, every genotype combination
    seq14 = random_sequence(rng, 140)
    rows14 = []
    for p in (20, 55, 90, 120):
        ref14 = seq14[p - 1]
        a1, a2 = OTHER[ref14][0], OTHER[ref14][1]
        rows14.append(row(
            p, ref14, f"{a1},{a2}",
            ["0|1", "0|2", "1|2", "2|2", "1|1", "0|0"],
        ))
    cases.append(Case(
        "multiallelic_snvs",
        "Two ALT alleles per record, covering 0|1, 0|2, 1|2, 2|2, 1|1, 0|0",
        "chr1", seq14, [f"M{i}" for i in range(1, 7)], rows14,
    ))

    # 15. haploid calls throughout, as on chrY
    seq15 = random_sequence(rng, 120)
    rows15 = [
        row(p, seq15[p - 1], OTHER[seq15[p - 1]][0], ["0", "1", "0", "1"])
        for p in (15, 45, 75, 105)
    ]
    cases.append(Case(
        "haploid_contig",
        "Every call haploid; hottools duplicates onto both tracks",
        "chrY", seq15, ["Y1", "Y2", "Y3", "Y4"], rows15,
        haploid_source=True,
    ))

    cases.append(Case(
        "indels_and_multiallelic",
        "Insertions, deletions and a multiallelic site mixed with SNVs; "
        "bcftools is given the SNV subset (multiallelic SNVs included, "
        "since hottools now encodes them)",
        "chr1", seq12, ["X1", "X2", "X3"], rows12,
        bcftools_prefilter=["-v", "snps"],
    ))


    # ---- length-changing classes, haplotype mode only ----------------
    rng13 = random.Random(1301)
    seq13 = random_sequence(rng13, 220)
    trio13 = ["D1", "D2", "D3"]
    het_hom = ["1|0", "1|1", "0|0"]

    cases.append(Case(
        "deletion_only",
        "Explicit deletions of 1-9 bp, phased het and hom carriers",
        "chr1", seq13, trio13,
        [
            row(20, seq13[19:21], seq13[19], het_hom),
            row(50, seq13[49:54], seq13[49], ["0|1", "1|1", "1|0"]),
            row(90, seq13[89:99], seq13[89], het_hom),
        ],
        variant_classes="del", structural=True, out_length=150,
    ))

    cases.append(Case(
        "insertion_only",
        "Explicit insertions of 1-6 bp pushing sequence out of the window",
        "chr1", seq13, trio13,
        [
            row(25, seq13[24], seq13[24] + "T", het_hom),
            row(60, seq13[59], seq13[59] + "GGCA", ["0|1", "1|1", "1|0"]),
            row(100, seq13[99], seq13[99] + "ACGTAC", het_hom),
        ],
        variant_classes="ins", structural=True, out_length=150,
    ))

    cases.append(Case(
        "mnv_blocks",
        "Multi-base substitutions, including one with shared flanks that "
        "normalises to a single base",
        "chr1", seq13, trio13,
        [
            row(30, seq13[29:32],
                "".join(OTHER[b][0] for b in seq13[29:32]), het_hom),
            row(70, seq13[69:73],
                "".join(OTHER[b][1] for b in seq13[69:73]),
                ["0|1", "1|1", "1|0"]),
            # Shared first and last base: only the middle one changes.
            row(110, seq13[109:112],
                seq13[109] + OTHER[seq13[110]][0] + seq13[111], het_hom),
        ],
        variant_classes="mnv", structural=False, out_length=150,
    ))

    cases.append(Case(
        "symbolic_deletions",
        "Precise <DEL> records whose extent comes from INFO/END",
        "chr1", seq13, trio13,
        [
            row(40, seq13[39], "<DEL>", het_hom, info="END=44"),
            row(80, seq13[79], "<DEL>", ["0|1", "1|1", "1|0"],
                info="END=92"),
        ],
        variant_classes="symbolic-del", structural=True, out_length=150,
        extra_header=END_HEADER,
    ))

    rng14 = random.Random(1401)
    seq14 = random_sequence(rng14, 260)
    quad14 = ["M1", "M2", "M3", "M4"]

    cases.append(Case(
        "every_class_together",
        "SNVs, MNVs, insertions, deletions and a <DEL> in one file",
        "chr1", seq14, quad14,
        [
            row(15, seq14[14], OTHER[seq14[14]][0], ["0|1", "1|1", "0|0", "1|0"]),
            row(35, seq14[34:37],
                "".join(OTHER[b][0] for b in seq14[34:37]),
                ["1|0", "0|1", "1|1", "0|0"]),
            row(55, seq14[54], seq14[54] + "TTG", ["0|0", "1|0", "0|1", "1|1"]),
            row(75, seq14[74:80], seq14[74], ["1|1", "0|0", "1|0", "0|1"]),
            row(100, seq14[99], "<DEL>", ["0|1", "1|0", "0|0", "1|1"],
                info="END=108"),
            row(130, seq14[129], OTHER[seq14[129]][1],
                ["1|1", "1|1", "0|1", "0|0"]),
        ],
        variant_classes="all", structural=True, out_length=180,
        extra_header=END_HEADER,
    ))

    cases.append(Case(
        "adjacent_deletions",
        "Deletions whose trimmed intervals touch but do not overlap, so "
        "both are applied",
        "chr1", seq14, quad14,
        [
            row(40, seq14[39:45], seq14[39], ["1|1"] * 4),
            row(45, seq14[44:52], seq14[44], ["1|1"] * 4),
            row(80, seq14[79:84], seq14[79], ["1|0", "0|1", "1|1", "0|0"]),
        ],
        variant_classes="del", structural=True, out_length=180,
    ))

    cases.append(Case(
        "deletion_reverse_strand",
        "Deletions and insertions read on the minus strand",
        "chr1", seq14, quad14,
        [
            row(30, seq14[29:36], seq14[29], ["1|0", "1|1", "0|0", "0|1"]),
            row(90, seq14[89], seq14[89] + "CCGG",
                ["0|1", "0|0", "1|1", "1|0"]),
        ],
        variant_classes="indel", structural=True, out_length=180,
        strand="-",
        hottools_flags=["--region", "chr1:1-180", "--strand", "-"],
    ))

    cases.append(Case(
        "structural_centre_anchor",
        "Indels with the window pinned on its centre position",
        "chr1", seq14, quad14,
        [
            row(120, seq14[119:125], seq14[119],
                ["1|0", "0|1", "1|1", "0|0"]),
        ],
        variant_classes="del", structural=True, out_length=180,
        hottools_flags=["--region", "chr1:1-180", "--anchor", "center"],
    ))

    return cases


def build_edge_cases() -> list[Case]:
    rng = random.Random(770)
    seq = random_sequence(rng, 120)
    trio = ["E1", "E2", "E3"]
    cases: list[Case] = []

    cases.append(Case(
        "unphased_heterozygote",
        "Unphased heterozygous call in haplotype mode",
        "chr1", seq, trio,
        [row(30, seq[29], OTHER[seq[29]][0], ["0/1", "0|1", "1|1"])],
        expect="unphased heterozygous",
        divergence=(
            "hottools refuses to invent phase; bcftools -H silently takes "
            "the alleles in listed order."
        ),
    ))

    cases.append(Case(
        "haploid_call",
        "Haploid genotype (chrX/chrY/chrM style) in haplotype mode",
        "chr1", seq, trio,
        [row(30, seq[29], OTHER[seq[29]][0], ["0", "0|1", "1|1"])],
        divergence=(
            "Both tools now apply the single allele to both outputs; "
            "hottools additionally warns and records per-sample ploidy."
        ),
    ))

    cases.append(Case(
        "triploid_call",
        "Triploid genotype in haplotype mode",
        "chr1", seq, trio,
        [row(30, seq[29], OTHER[seq[29]][0], ["0|1|1", "0|1", "1|1"])],
        expect="ploidy > 2",
        divergence="bcftools accepts higher ploidy via -H 3.",
    ))

    cases.append(Case(
        "duplicate_positions",
        "Two biallelic SNV records at the same position",
        "chr1", seq, trio,
        [
            row(30, seq[29], OTHER[seq[29]][0], ["0|1", "0|1", "0|1"]),
            row(30, seq[29], OTHER[seq[29]][1], ["1|1", "1|1", "1|1"]),
            row(60, seq[59], OTHER[seq[59]][0], ["0|1", "1|1", "0|0"]),
        ],
        divergence=(
            "Hottools 0.7.1 and bcftools retain the first applicable ALT "
            "per haplotype and warn/count conflicting overlaps."
        ),
    ))

    cases.append(Case(
        "star_allele",
        "Spanning-deletion '*' allele",
        "chr1", seq, trio,
        [
            row(30, seq[29], "*", ["0|1", "0|0", "1|1"]),
            row(60, seq[59], OTHER[seq[59]][0], ["0|1", "1|1", "0|0"]),
        ],
        divergence="hottools rejects '*' as a non-SNV ALT.",
    ))

    cases.append(Case(
        "symbolic_alleles",
        "Symbolic <DEL> and <NON_REF> alleles",
        "chr1", seq, trio,
        [
            row(30, seq[29], "<DEL>", ["0|1", "0|0", "1|1"]),
            row(50, seq[49], "<NON_REF>", ["0|0", "0|1", "0|0"]),
            row(60, seq[59], OTHER[seq[59]][0], ["0|1", "1|1", "0|0"]),
        ],
        divergence="hottools rejects symbolic ALTs as non-SNV.",
    ))

    cases.append(Case(
        "lowercase_vcf_bases",
        "REF/ALT written in lower case in the VCF",
        "chr1", seq, trio,
        [
            row(30, seq[29].lower(), OTHER[seq[29]][0].lower(),
                ["0|1", "1|1", "0|0"]),
            row(60, seq[59], OTHER[seq[59]][0], ["0|1", "1|1", "0|0"]),
        ],
        divergence=(
            "Out of specification but unambiguous; both tools now "
            "upper-case and apply it."
        ),
    ))

    cases.append(Case(
        "iupac_ref_base",
        "VCF REF is an IUPAC ambiguity code",
        "chr1", seq, trio,
        [
            row(30, "R", "A", ["0|1", "1|1", "0|0"]),
            row(60, seq[59], OTHER[seq[59]][0], ["0|1", "1|1", "0|0"]),
        ],
        divergence="hottools rejects non-ACGT REF before FASTA comparison.",
    ))

    cases.append(Case(
        "ref_mismatch",
        "VCF REF disagrees with the FASTA base",
        "chr1", seq, trio,
        [row(30, OTHER[seq[29]][0], seq[29], ["0|1", "1|1", "0|0"])],
        expect="REF mismatch",
        divergence="Both tools refuse; bcftools exits non-zero as well.",
    ))

    cases.append(Case(
        "variant_on_ambiguous_base",
        "Variant sited on an N in the reference",
        "chr1", seq[:49] + "N" + seq[50:], trio,
        [row(50, "A", "C", ["0|1", "1|1", "0|0"])],
        expect="Ambiguous FASTA base",
        divergence="bcftools reports a REF mismatch at the same site.",
    ))

    cases.append(Case(
        "empty_region",
        "No variant records overlap the requested window",
        "chr1", seq, trio,
        [row(110, seq[109], OTHER[seq[109]][0], ["0|1", "1|1", "0|0"])],
        hottools_flags=["--region", "chr1:1-100"],
        divergence="Warning only; output is the reference for every sample.",
    ))

    cases.append(Case(
        "contig_naming_mismatch",
        "VCF uses 'chr1' while the reference FASTA uses '1'",
        "chr1", seq, trio,
        [row(30, seq[29], OTHER[seq[29]][0], ["0|1", "1|1", "0|0"])],
        fasta_contig="1",
        expect="not found in",
        divergence="Caught before any encoding work.",
    ))


    cases.append(Case(
        "symbolic_del_without_end",
        "A <DEL> with no INFO/END: its extent is unknowable",
        "chr1", seq, trio,
        [row(40, seq[39], "<DEL>", ["1|1", "0|1", "0|0"])],
        variant_classes="all", expect="ok",
        divergence=(
            "hottools refuses the record because the deleted interval is "
            "undefined; bcftools emits the reference unchanged."
        ),
    ))

    cases.append(Case(
        "overlapping_deletions",
        "Two deletions claiming the same reference bases on one haplotype",
        "chr1", seq, trio,
        [
            row(30, seq[29:38], seq[29], ["1|1", "1|1", "0|0"]),
            row(34, seq[33:42], seq[33], ["1|1", "0|1", "1|1"]),
        ],
        variant_classes="del", expect="ok",
        divergence=(
            "Both tools apply the first record in POS order and set the "
            "second aside; hottools says so in its counters and a warning."
        ),
    ))

    cases.append(Case(
        "indels_in_dosage_mode",
        "Length-changing classes requested outside haplotype mode",
        "chr1", seq, trio,
        [row(30, seq[29:34], seq[29], ["1|1", "0|1", "0|0"])],
        variant_classes="del",
        hottools_flags=["--region", "chr1:1-120", "--mode", "dosage"],
        expect="only meaningful in --mode haplotypes",
        divergence=(
            "A deletion has no definition when two sequences are averaged "
            "position by position, so hottools refuses rather than guess."
        ),
    ))

    return cases
