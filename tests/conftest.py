"""
Shared fixtures: a small reference FASTA and several indexed VCFs built
from scratch, so the suite needs no checked-in binary data.
"""
from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

BASES = "ACGT"
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}
#: Transition-like ALT choice; any base other than REF would do.
ALT_OF = {"A": "G", "C": "T", "G": "A", "T": "C"}

REF_LENGTH = 200
AMBIGUOUS_POS = 50          # 1-based position of the 'N' in the reference
SAMPLES = ["S1", "S2", "S3", "S4"]


def _bgzip_and_index(vcf_text: str, path: Path) -> Path:
    """Write ``vcf_text`` as a bgzipped, tabix-indexed VCF at ``path``."""
    plain = path.with_suffix("")
    if plain.suffix != ".vcf":
        plain = path.parent / (path.name.replace(".gz", ""))
    plain.write_text(vcf_text, encoding="utf-8")

    try:
        import pysam

        pysam.tabix_compress(str(plain), str(path), force=True)
        pysam.tabix_index(str(path), preset="vcf", force=True)
        return path
    except ImportError:
        pass

    if shutil.which("bgzip") and shutil.which("tabix"):
        subprocess.run(["bgzip", "-f", str(plain)], check=True)
        subprocess.run(["tabix", "-f", "-p", "vcf", str(path)], check=True)
        return path

    pytest.skip("neither pysam nor bgzip/tabix is available")


def _vcf_text(rows: list[str], samples: list[str] = SAMPLES) -> str:
    header = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID=chr1,length={REF_LENGTH}>",
        '##INFO=<ID=END,Number=1,Type=Integer,Description="End position">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(samples),
    ]
    return "\n".join(header + rows) + "\n"


def _row(
    pos: int, ref: str, alt: str, gts: list[str], info: str = "."
) -> str:
    return (
        f"chr1\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info}\tGT\t"
        + "\t".join(gts)
    )


@pytest.fixture(scope="session")
def reference_sequence() -> str:
    rng = random.Random(7)
    seq = "".join(rng.choice(BASES) for _ in range(REF_LENGTH))
    return seq[: AMBIGUOUS_POS - 1] + "N" + seq[AMBIGUOUS_POS:]


@pytest.fixture(scope="session")
def fasta_path(tmp_path_factory, reference_sequence) -> Path:
    directory = tmp_path_factory.mktemp("reference")
    path = directory / "ref.fa"
    lines = [
        reference_sequence[i : i + 60]
        for i in range(0, len(reference_sequence), 60)
    ]
    path.write_text(">chr1\n" + "\n".join(lines) + "\n", encoding="utf-8")

    try:
        import pysam

        pysam.faidx(str(path))
    except ImportError:
        if not shutil.which("samtools"):
            pytest.skip("neither pysam nor samtools is available")
        subprocess.run(["samtools", "faidx", str(path)], check=True)
    return path


@pytest.fixture(scope="session")
def variant_spec(reference_sequence) -> dict:
    """
    The variant layout used by the main VCF, as plain data so tests can
    assert against it without re-deriving coordinates.

    Positions are 1-based.
    """
    seq = reference_sequence
    return {
        "clean_phased": {
            "pos": 10,
            "ref": seq[9],
            "alt": ALT_OF[seq[9]],
            "gts": ["0|0", "0|1", "1|1", ".|."],
        },
        "unphased_het": {
            "pos": 20,
            "ref": seq[19],
            "alt": ALT_OF[seq[19]],
            "gts": ["0/0", "0/1", "1/1", "./."],
        },
        "indel": {"pos": 30, "ref": seq[29], "alt": "AT"},
        "duplicate": {"pos": 40, "ref": seq[39], "alt": ALT_OF[seq[39]]},
        "ambiguous": {"pos": AMBIGUOUS_POS, "ref": "A", "alt": "C"},
        "ref_mismatch": {"pos": 60, "ref": ALT_OF[seq[59]], "alt": seq[59]},
        "clean_phased_2": {
            "pos": 70,
            "ref": seq[69],
            "alt": ALT_OF[seq[69]],
            "gts": ["0|0", "0|1", "1|1", "0|1"],
        },
    }


@pytest.fixture(scope="session")
def vcf_path(tmp_path_factory, variant_spec) -> Path:
    """
    Eight records exercising every filter branch:

    ====  ====================================================
    pos   branch
    ====  ====================================================
    10    clean phased biallelic SNV (kept)
    20    unphased heterozygote (kept; errors in haplotype mode)
    30    insertion -> rejected as non-SNV
    40    two same-allele records -> first applicable ALT per haplotype
    50    FASTA base is 'N' -> ambiguous
    60    VCF REF disagrees with FASTA
    70    clean phased biallelic SNV (kept)
    ====  ====================================================
    """
    v = variant_spec
    rows = [
        _row(v["clean_phased"]["pos"], v["clean_phased"]["ref"],
             v["clean_phased"]["alt"], v["clean_phased"]["gts"]),
        _row(v["unphased_het"]["pos"], v["unphased_het"]["ref"],
             v["unphased_het"]["alt"], v["unphased_het"]["gts"]),
        _row(v["indel"]["pos"], v["indel"]["ref"], v["indel"]["alt"],
             ["0|0", "0|1", "1|1", "0|0"]),
        _row(v["duplicate"]["pos"], v["duplicate"]["ref"],
             v["duplicate"]["alt"], ["0|1"] * 4),
        _row(v["duplicate"]["pos"], v["duplicate"]["ref"],
             v["duplicate"]["alt"], ["1|1"] * 4),
        _row(v["ambiguous"]["pos"], v["ambiguous"]["ref"],
             v["ambiguous"]["alt"], ["0|1"] * 4),
        _row(v["ref_mismatch"]["pos"], v["ref_mismatch"]["ref"],
             v["ref_mismatch"]["alt"], ["0|1"] * 4),
        _row(v["clean_phased_2"]["pos"], v["clean_phased_2"]["ref"],
             v["clean_phased_2"]["alt"], v["clean_phased_2"]["gts"]),
    ]
    directory = tmp_path_factory.mktemp("vcf")
    return _bgzip_and_index(_vcf_text(rows), directory / "var.vcf.gz")


@pytest.fixture(scope="session")
def clean_vcf_path(tmp_path_factory, variant_spec) -> Path:
    """Only the two clean phased SNVs: no policy branch is triggered."""
    v = variant_spec
    rows = [
        _row(v["clean_phased"]["pos"], v["clean_phased"]["ref"],
             v["clean_phased"]["alt"], v["clean_phased"]["gts"]),
        _row(v["clean_phased_2"]["pos"], v["clean_phased_2"]["ref"],
             v["clean_phased_2"]["alt"], v["clean_phased_2"]["gts"]),
    ]
    directory = tmp_path_factory.mktemp("vcf_clean")
    return _bgzip_and_index(_vcf_text(rows), directory / "clean.vcf.gz")


@pytest.fixture(scope="session")
def haploid_vcf_path(tmp_path_factory, variant_spec) -> Path:
    """One record where S1 is haploid, as on chrX/chrY/chrM."""
    v = variant_spec
    rows = [
        _row(v["clean_phased"]["pos"], v["clean_phased"]["ref"],
             v["clean_phased"]["alt"], ["0", "0|1", "1|1", "0|0"]),
        _row(v["clean_phased_2"]["pos"], v["clean_phased_2"]["ref"],
             v["clean_phased_2"]["alt"], ["1", "0|1", "1|1", "0|1"]),
    ]
    directory = tmp_path_factory.mktemp("vcf_haploid")
    return _bgzip_and_index(_vcf_text(rows), directory / "haploid.vcf.gz")


@pytest.fixture(scope="session")
def lowercase_vcf_path(tmp_path_factory, variant_spec) -> Path:
    """REF/ALT written in lower case, which htslib and bcftools accept."""
    v = variant_spec
    rows = [
        _row(v["clean_phased"]["pos"], v["clean_phased"]["ref"].lower(),
             v["clean_phased"]["alt"].lower(), v["clean_phased"]["gts"]),
        _row(v["clean_phased_2"]["pos"], v["clean_phased_2"]["ref"],
             v["clean_phased_2"]["alt"], v["clean_phased_2"]["gts"]),
    ]
    directory = tmp_path_factory.mktemp("vcf_lowercase")
    return _bgzip_and_index(_vcf_text(rows), directory / "lower.vcf.gz")


@pytest.fixture(scope="session")
def halfcall_vcf_path(tmp_path_factory, variant_spec) -> Path:
    """Half-called genotypes: one allele known, one missing."""
    v = variant_spec
    rows = [
        _row(v["clean_phased"]["pos"], v["clean_phased"]["ref"],
             v["clean_phased"]["alt"], [".|1", "1|.", ".|0", ".|."]),
        _row(v["clean_phased_2"]["pos"], v["clean_phased_2"]["ref"],
             v["clean_phased_2"]["alt"], v["clean_phased_2"]["gts"]),
    ]
    directory = tmp_path_factory.mktemp("vcf_halfcall")
    return _bgzip_and_index(_vcf_text(rows), directory / "halfcall.vcf.gz")


@pytest.fixture(scope="session")
def hemizygous_nocall_vcf_path(tmp_path_factory, variant_spec) -> Path:
    """Haploid calls including an uncalled one, as on chrY."""
    v = variant_spec
    rows = [
        _row(v["clean_phased"]["pos"], v["clean_phased"]["ref"],
             v["clean_phased"]["alt"], [".", "0", "1", "."]),
    ]
    directory = tmp_path_factory.mktemp("vcf_hemi_nocall")
    return _bgzip_and_index(_vcf_text(rows), directory / "hemi.vcf.gz")


@pytest.fixture(scope="session")
def bad_allele_index_vcf_path(tmp_path_factory, variant_spec) -> Path:
    """A GT naming an allele index the record does not declare."""
    v = variant_spec
    rows = [
        _row(v["clean_phased"]["pos"], v["clean_phased"]["ref"],
             v["clean_phased"]["alt"], ["0|1", "0|4", "0|0", "0|0"]),
    ]
    directory = tmp_path_factory.mktemp("vcf_bad_index")
    return _bgzip_and_index(_vcf_text(rows), directory / "bad.vcf.gz")


@pytest.fixture(scope="session")
def dropped_multiallelic_vcf_path(tmp_path_factory, reference_sequence) -> Path:
    """A multiallelic, lower-case record that a later check discards."""
    seq = reference_sequence
    ref50 = "A"          # position 50 is 'N' in the reference
    rows = [
        _row(50, ref50.lower(), "c,g", ["0|1", "0|2", "1|2", "0|0"]),
        _row(70, seq[69], ALT_OF[seq[69]], ["0|1", "1|1", "0|0", "0|1"]),
    ]
    directory = tmp_path_factory.mktemp("vcf_dropped_multi")
    return _bgzip_and_index(_vcf_text(rows), directory / "dropped.vcf.gz")


@pytest.fixture(scope="session")
def triploid_vcf_path(tmp_path_factory, variant_spec) -> Path:
    """A record where one sample is triploid."""
    v = variant_spec
    rows = [
        _row(v["clean_phased"]["pos"], v["clean_phased"]["ref"],
             v["clean_phased"]["alt"], ["0|1|1", "0|1", "1|1", "0|0"]),
        _row(v["clean_phased_2"]["pos"], v["clean_phased_2"]["ref"],
             v["clean_phased_2"]["alt"], v["clean_phased_2"]["gts"]),
    ]
    directory = tmp_path_factory.mktemp("vcf_triploid")
    return _bgzip_and_index(_vcf_text(rows), directory / "triploid.vcf.gz")


@pytest.fixture(scope="session")
def triallelic_vcf_path(tmp_path_factory, variant_spec, reference_sequence):
    """One record carrying two ALT alleles."""
    v = variant_spec
    pos = v["clean_phased"]["pos"]
    ref = v["clean_phased"]["ref"]
    alts = [b for b in BASES if b != ref][:2]
    rows = [
        _row(pos, ref, ",".join(alts), ["0|1", "0|2", "1|2", "2|2"]),
        _row(v["clean_phased_2"]["pos"], v["clean_phased_2"]["ref"],
             v["clean_phased_2"]["alt"], v["clean_phased_2"]["gts"]),
    ]
    directory = tmp_path_factory.mktemp("vcf_triallelic")
    return _bgzip_and_index(_vcf_text(rows), directory / "tri.vcf.gz")


@pytest.fixture(scope="session")
def quadallelic_vcf_path(tmp_path_factory, variant_spec):
    """A record with four ALT alleles, beyond what hottools encodes."""
    v = variant_spec
    pos, ref = v["clean_phased"]["pos"], v["clean_phased"]["ref"]
    alts = [b for b in BASES if b != ref] + ["N"]
    rows = [
        _row(pos, ref, ",".join(alts), ["0|1", "0|2", "1|3", "2|4"]),
        _row(v["clean_phased_2"]["pos"], v["clean_phased_2"]["ref"],
             v["clean_phased_2"]["alt"], v["clean_phased_2"]["gts"]),
    ]
    directory = tmp_path_factory.mktemp("vcf_quad")
    return _bgzip_and_index(_vcf_text(rows), directory / "quad.vcf.gz")


@pytest.fixture(scope="session")
def filtered_vcf_path(tmp_path_factory, reference_sequence) -> Path:
    """Records carrying FILTER, QUAL and INFO/AF values to filter on."""
    seq = reference_sequence
    def alt_of(b):
        return {"A": "G", "C": "T", "G": "A", "T": "C"}[b]
    rows = [
        f"chr1\t10\t.\t{seq[9]}\t{alt_of(seq[9])}\t50\tPASS\tAF=0.40\tGT\t"
        + "\t".join(["0|0", "0|1", "1|1", "0|1"]),
        f"chr1\t20\t.\t{seq[19]}\t{alt_of(seq[19])}\t10\tLowQual\tAF=0.02\tGT\t"
        + "\t".join(["0|0", "0|1", "0|0", "0|0"]),
        f"chr1\t70\t.\t{seq[69]}\t{alt_of(seq[69])}\t99\t.\tAF=0.90\tGT\t"
        + "\t".join(["1|1", "1|1", "0|1", "1|1"]),
    ]
    header = (
        '##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">',
        '##FILTER=<ID=LowQual,Description="Low quality">',
    )
    directory = tmp_path_factory.mktemp("vcf_filtered")
    text = _vcf_text(rows)
    text = text.replace(
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        + "\n".join(header),
    )
    return _bgzip_and_index(text, directory / "filtered.vcf.gz")


@pytest.fixture(scope="session")
def mixed_af_vcf_path(tmp_path_factory, reference_sequence) -> Path:
    """One record with INFO/AF, one without."""
    seq = reference_sequence
    rows = [
        f"chr1\t10\t.\t{seq[9]}\t{ALT_OF[seq[9]]}\t99\tPASS\tAF=0.90\tGT\t"
        + "\t".join(["0|1", "1|1", "0|0", "0|0"]),
        f"chr1\t70\t.\t{seq[69]}\t{ALT_OF[seq[69]]}\t99\tPASS\t.\tGT\t"
        + "\t".join(["0|1", "0|0", "0|0", "0|0"]),
    ]
    text = _vcf_text(rows).replace(
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">',
    )
    directory = tmp_path_factory.mktemp("vcf_mixed_af")
    return _bgzip_and_index(text, directory / "mixed_af.vcf.gz")


@pytest.fixture(scope="session")
def collision_vcf_path(tmp_path_factory, variant_spec) -> Path:
    """Sample IDs that sanitise to the same filename."""
    v = variant_spec
    rows = [
        _row(v["clean_phased"]["pos"], v["clean_phased"]["ref"],
             v["clean_phased"]["alt"], ["0|0", "1|1", "0|1"]),
    ]
    directory = tmp_path_factory.mktemp("vcf_collision")
    return _bgzip_and_index(
        _vcf_text(rows, samples=["S/2", "S:2", "other"]),
        directory / "collide.vcf.gz",
    )


@pytest.fixture(scope="session")
def multiallelic_vcf_path(tmp_path_factory, variant_spec, reference_sequence):
    """
    A split multiallelic site at position 80: one indel row and one SNV
    row. The indel is rejected before the duplicate check ever sees it.
    """
    v = variant_spec
    ref80 = reference_sequence[79]
    rows = [
        _row(v["clean_phased"]["pos"], v["clean_phased"]["ref"],
             v["clean_phased"]["alt"], v["clean_phased"]["gts"]),
        _row(80, ref80, "ATTT", ["0|1"] * 4),
        _row(80, ref80, ALT_OF[ref80], ["1|1"] * 4),
    ]
    directory = tmp_path_factory.mktemp("vcf_multi")
    return _bgzip_and_index(_vcf_text(rows), directory / "multi.vcf.gz")


@pytest.fixture(scope="session")
def nochr_vcf_path(tmp_path_factory, variant_spec):
    """Same variants, but the contig is named '1' instead of 'chr1'."""
    v = variant_spec
    text = _vcf_text([
        _row(v["clean_phased"]["pos"], v["clean_phased"]["ref"],
             v["clean_phased"]["alt"], v["clean_phased"]["gts"]),
    ]).replace("chr1", "1")
    directory = tmp_path_factory.mktemp("vcf_nochr")
    return _bgzip_and_index(text, directory / "nochr.vcf.gz")


@pytest.fixture(scope="session")
def reference_onehot(reference_sequence) -> np.ndarray:
    """Independently built reference one-hot, for comparison."""
    out = np.zeros((len(reference_sequence), 4), dtype=np.float32)
    for i, base in enumerate(reference_sequence):
        if base in BASES:
            out[i, BASES.index(base)] = 1.0
    return out


def onehot_of(base: str) -> np.ndarray:
    vec = np.zeros(4, dtype=np.float32)
    vec[BASES.index(base)] = 1.0
    return vec

#: Reference positions used by the structural fixtures, all clear of the
#: ambiguous base at :data:`AMBIGUOUS_POS`.
DEL_POS = 100
INS_POS = 120
MNV_POS = 140
SYMDEL_POS = 160
SYMDEL_END = 165


@pytest.fixture(scope="session")
def indel_vcf_path(tmp_path_factory, reference_sequence):
    """
    One of each length-changing class plus an MNV, all phased.

    S1 is heterozygous on haplotype 1, S2 is homozygous for every
    alternate allele, S3 and S4 are homozygous reference, so a single
    run exercises carrier, non-carrier and mixed signatures.
    """
    seq = reference_sequence
    gts = ["1|0", "1|1", "0|0", "0|0"]
    rows = [
        # A four-base deletion: REF quotes the anchor plus the removed
        # bases, which is how a VCF writes a deletion.
        _row(DEL_POS, seq[DEL_POS - 1:DEL_POS + 3], seq[DEL_POS - 1], gts),
        # A three-base insertion after INS_POS.
        _row(INS_POS, seq[INS_POS - 1], seq[INS_POS - 1] + "GGG", gts),
        # A three-base MNV, every base changed so trimming cannot
        # collapse it.
        _row(
            MNV_POS,
            seq[MNV_POS - 1:MNV_POS + 2],
            "".join(ALT_OF[b] for b in seq[MNV_POS - 1:MNV_POS + 2]),
            gts,
        ),
        # A precise symbolic deletion: POS is the base before the event.
        _row(
            SYMDEL_POS, seq[SYMDEL_POS - 1], "<DEL>", gts,
            info=f"END={SYMDEL_END}",
        ),
    ]
    directory = tmp_path_factory.mktemp("vcf_indel")
    return _bgzip_and_index(_vcf_text(rows), directory / "indel.vcf.gz")


@pytest.fixture(scope="session")
def overlapping_del_vcf_path(tmp_path_factory, reference_sequence):
    """Two deletions claiming overlapping reference bases."""
    seq = reference_sequence
    rows = [
        _row(100, seq[99:106], seq[99], ["1|1"] * 4),
        _row(103, seq[102:110], seq[102], ["1|1"] * 4),
    ]
    directory = tmp_path_factory.mktemp("vcf_overlap")
    return _bgzip_and_index(_vcf_text(rows), directory / "overlap.vcf.gz")


@pytest.fixture(scope="session")
def adjacent_del_vcf_path(tmp_path_factory, reference_sequence):
    """
    Two deletions that share only an anchor base.

    Their trimmed intervals are adjacent, not overlapping, so both are
    applied - the case that distinguishes a trimmed-interval overlap
    test from a naive REF-span test.
    """
    seq = reference_sequence
    rows = [
        _row(100, seq[99:104], seq[99], ["1|1"] * 4),
        _row(104, seq[103:109], seq[103], ["1|1"] * 4),
    ]
    directory = tmp_path_factory.mktemp("vcf_adjacent")
    return _bgzip_and_index(_vcf_text(rows), directory / "adjacent.vcf.gz")


@pytest.fixture(scope="session")
def symbolic_no_end_vcf_path(tmp_path_factory, reference_sequence):
    """A ``<DEL>`` with no INFO/END: its extent is unknowable."""
    rows = [
        _row(100, reference_sequence[99], "<DEL>", ["1|1"] * 4),
    ]
    directory = tmp_path_factory.mktemp("vcf_noend")
    return _bgzip_and_index(_vcf_text(rows), directory / "noend.vcf.gz")


@pytest.fixture(scope="session")
def straddling_del_vcf_path(tmp_path_factory, reference_sequence):
    """
    A deletion whose POS is upstream of a region starting at 100.

    Its REF span reaches into the window, so the bases it removes there
    are real sequence for a carrier even though the record itself sits
    outside.
    """
    seq = reference_sequence
    rows = [_row(95, seq[94:103], seq[94], ["1|0", "1|1", "0|0", "0|0"])]
    directory = tmp_path_factory.mktemp("vcf_straddle")
    return _bgzip_and_index(_vcf_text(rows), directory / "straddle.vcf.gz")


@pytest.fixture(scope="session")
def chain_vcf_path(tmp_path_factory, reference_sequence):
    """
    An upstream record that decides the fate of one inside the window.

    The deletion at 90 claims 91-104; the deletion at 100 claims
    101-110 and so loses to it. A window starting at 100 only sees the
    second record, and has to look left to resolve it the way reading
    the whole file would.
    """
    seq = reference_sequence
    rows = [
        _row(90, seq[89:104], seq[89], ["1|1"] * 4),
        _row(100, seq[99:110], seq[99], ["1|1"] * 4),
    ]
    directory = tmp_path_factory.mktemp("vcf_chain")
    return _bgzip_and_index(_vcf_text(rows), directory / "chain.vcf.gz")


@pytest.fixture(scope="session")
def colliding_mnv_vcf_path(tmp_path_factory, reference_sequence):
    """
    Two MNV blocks that want the same reference position.

    One position carries one allele table, so the second record cannot
    be represented alongside the first.
    """
    seq = reference_sequence
    first = "".join(ALT_OF[b] for b in seq[99:102])
    second = "".join(ALT_OF[b] for b in seq[101:104])
    rows = [
        _row(100, seq[99:102], first, ["1|1"] * 4),
        _row(102, seq[101:104], second, ["0|1"] * 4),
    ]
    directory = tmp_path_factory.mktemp("vcf_mnvcollide")
    return _bgzip_and_index(_vcf_text(rows), directory / "collide.vcf.gz")



# --------------------------------------------------------------------
# essential-case fixtures (tests/test_essentials.py)
# --------------------------------------------------------------------

#: One sample per genotype pattern, so a single run covers all of them.
ESSENTIAL_SAMPLES = [
    "P00", "P01", "P10", "P11", "MISS", "HALF", "HAP0", "HAP1", "MULTI",
]
#: Genotype of each sample at the biallelic site ``ESSENTIAL_SNV_POS``.
ESSENTIAL_GTS = {
    "P00": "0|0", "P01": "0|1", "P10": "1|0", "P11": "1|1",
    "MISS": "./.", "HALF": "0/.", "HAP0": "0", "HAP1": "1", "MULTI": "0|0",
}
ESSENTIAL_SNV_POS = 20
ESSENTIAL_MULTI_POS = 30


def _two_other_bases(base: str) -> tuple[str, str]:
    others = [b for b in "ACGT" if b != base]
    return others[0], others[1]


@pytest.fixture(scope="session")
def essential_vcf_path(tmp_path_factory, reference_sequence):
    """
    Every genotype pattern at one biallelic site, plus a ``1|2`` call at
    a triallelic site.

    Patterns: ``0|0 0|1 1|0 1|1``, uncalled ``./.``, half-called ``0/.``,
    haploid ``0`` and ``1``. Unphased ``./.`` and ``0/.`` are accepted in
    haplotype mode because there is no heterozygote to phase.
    """
    seq = reference_sequence
    snv_ref = seq[ESSENTIAL_SNV_POS - 1]
    multi_ref = seq[ESSENTIAL_MULTI_POS - 1]
    alt1, alt2 = _two_other_bases(multi_ref)
    haploid_or_diploid = {"HAP0": "0", "HAP1": "0"}
    rows = [
        _row(ESSENTIAL_SNV_POS, snv_ref, ALT_OF[snv_ref],
             [ESSENTIAL_GTS[s] for s in ESSENTIAL_SAMPLES]),
        _row(ESSENTIAL_MULTI_POS, multi_ref, f"{alt1},{alt2}",
             [("1|2" if s == "MULTI" else haploid_or_diploid.get(s, "0|0"))
              for s in ESSENTIAL_SAMPLES]),
    ]
    directory = tmp_path_factory.mktemp("vcf_essential")
    return _bgzip_and_index(
        _vcf_text(rows, samples=ESSENTIAL_SAMPLES), directory / "ess.vcf.gz"
    )


@pytest.fixture(scope="session")
def unphased_het_vcf_path(tmp_path_factory, reference_sequence):
    """S1 is an unphased heterozygote ``0/1``; the rest are phased."""
    ref = reference_sequence[ESSENTIAL_SNV_POS - 1]
    rows = [_row(ESSENTIAL_SNV_POS, ref, ALT_OF[ref],
                 ["0/1", "0|1", "1|1", "0|0"])]
    directory = tmp_path_factory.mktemp("vcf_unphased")
    return _bgzip_and_index(_vcf_text(rows), directory / "unph.vcf.gz")


@pytest.fixture(scope="session")
def major_vcf_path(tmp_path_factory, reference_sequence):
    """
    S1 is heterozygous at two sites with opposite majorities.

    At 20 the ALT is carried by 7 of 8 haplotypes, at 40 by 1 of 8, so
    ``het_policy='major'`` must pick ALT at one and REF at the other.
    """
    seq = reference_sequence
    rows = [
        _row(20, seq[19], ALT_OF[seq[19]], ["0|1", "1|1", "1|1", "1|1"]),
        _row(40, seq[39], ALT_OF[seq[39]], ["0|1", "0|0", "0|0", "0|0"]),
    ]
    directory = tmp_path_factory.mktemp("vcf_major")
    return _bgzip_and_index(_vcf_text(rows), directory / "major.vcf.gz")


@pytest.fixture(scope="session")
def missing_ref_alt_vcf_path(tmp_path_factory, reference_sequence):
    """ALT ``.``, REF ``N`` and REF ``.``, plus one valid SNV at 60."""
    seq = reference_sequence
    rows = [
        _row(20, seq[19], ".", ["0|0"] * 4),
        _row(30, "N", ALT_OF[seq[29]], ["0|1"] * 4),
        _row(40, ".", ALT_OF[seq[39]], ["0|1"] * 4),
        _row(60, seq[59], ALT_OF[seq[59]], ["1|1"] * 4),
    ]
    directory = tmp_path_factory.mktemp("vcf_missing_ref_alt")
    return _bgzip_and_index(_vcf_text(rows), directory / "mra.vcf.gz")


@pytest.fixture(scope="session")
def ambiguous_site_vcf_path(tmp_path_factory, reference_sequence):
    """A variant on the FASTA ``N`` at AMBIGUOUS_POS, plus one valid SNV."""
    seq = reference_sequence
    rows = [
        _row(20, seq[19], ALT_OF[seq[19]], ["1|1"] * 4),
        _row(AMBIGUOUS_POS, "A", "G", ["1|1"] * 4),
    ]
    directory = tmp_path_factory.mktemp("vcf_ambig_site")
    return _bgzip_and_index(_vcf_text(rows), directory / "ambig.vcf.gz")


@pytest.fixture(scope="session")
def ref_mismatch_vcf_path(tmp_path_factory, reference_sequence):
    """REF at 20 disagrees with the FASTA; the SNV at 30 is valid."""
    seq = reference_sequence
    wrong = ALT_OF[seq[19]]
    rows = [
        _row(20, wrong, ALT_OF[wrong], ["1|1"] * 4),
        _row(30, seq[29], ALT_OF[seq[29]], ["1|1"] * 4),
    ]
    directory = tmp_path_factory.mktemp("vcf_mismatch")
    return _bgzip_and_index(_vcf_text(rows), directory / "mismatch.vcf.gz")


@pytest.fixture
def native_bcftools(request, record_testsuite_property):
    """Use compiled consensus and attach its version to concordance evidence."""
    module = pytest.importorskip("pysam.bcftools")
    import pysam
    versions = {"pysam_version": pysam.__version__,
                "bcftools_version": pysam.version.__bcftools_version__}
    request.config._hottools_comparator_versions = versions
    for name, value in versions.items():
        request.node.user_properties.append((name, value))
    if not getattr(request.config, "_hottools_comparator_recorded", False):
        for name, value in versions.items():
            record_testsuite_property(name, value)
        request.config._hottools_comparator_recorded = True
    return module


def pytest_terminal_summary(terminalreporter):
    versions = getattr(terminalreporter.config, "_hottools_comparator_versions", None)
    if versions:
        terminalreporter.write_line(
            "Native concordance comparator: bcftools " + versions["bcftools_version"]
            + " (pysam " + versions["pysam_version"] + ")")
