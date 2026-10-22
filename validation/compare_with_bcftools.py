"""
Validate hottools against ``bcftools consensus``.

For each concordance case and each sample, the two haplotype sequences
produced by

    samtools faidx ref.fa REGION | bcftools consensus -s SAMPLE -H {1,2}

are compared base-for-base against hottools' haplotype tensor decoded
back to sequence. The dosage tensor is checked against the same two
haplotypes: where they agree the channel must be one-hot on that base,
where they differ it must be 0.5/0.5 on exactly those two bases.

``--missing-policy reference`` is used throughout, because that is the
policy whose semantics match bcftools (a missing allele falls back to
REF). The zero policy is asserted separately in the unit suite.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_cases import (
    COMPLEMENT,
    MaterialisedCase,
    build_concordance_cases,
    build_edge_cases,
)

BASES = "ACGT"
COMMON_FLAGS = ["--missing-policy", "reference", "--overwrite", "replace"]


# ------------------------------------------------------------------
# running the two tools
# ------------------------------------------------------------------

def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, check=False, **kwargs
    )


def bcftools_haplotype(
    fasta: Path, vcf: Path, region: str, sample: str, hap: int
) -> str | None:
    """One consensus haplotype as a bare sequence string, or None on error."""
    faidx = run(["samtools", "faidx", str(fasta), region])
    if faidx.returncode != 0:
        return None
    cons = run(
        ["bcftools", "consensus", "-s", sample, "-H", str(hap), str(vcf)],
        input=faidx.stdout,
    )
    if cons.returncode != 0:
        return None
    lines = [ln for ln in cons.stdout.splitlines() if not ln.startswith(">")]
    return "".join(lines).upper()


def hottools_run(
    mat: MaterialisedCase, out: Path, mode: str, extra: list[str]
) -> tuple[int, dict | None, str]:
    case = mat.case
    region = f"{case.contig}:1-{case.window}"
    flags = case.hottools_flags or ["--region", region]
    if case.variant_classes:
        flags = [*flags, "--variant-classes", case.variant_classes]
    cmd = [
        "hottools", str(mat.vcf), "--reference", str(mat.fasta),
        "-o", str(out), "--mode", mode, *COMMON_FLAGS, *flags, *extra,
    ]
    if "--region" not in flags:
        cmd += ["--region", region]
    proc = run(cmd)
    meta = None
    if proc.returncode == 0 and proc.stdout.strip():
        try:
            meta = json.loads(proc.stdout)
        except json.JSONDecodeError:
            meta = None
    return proc.returncode, meta, (proc.stderr or "").strip()


# ------------------------------------------------------------------
# decoding
# ------------------------------------------------------------------

def decode_onehot(track: np.ndarray) -> str:
    """L x 4 one-hot -> sequence. All-zero -> 'N', ambiguous -> '?'."""
    sums = track.sum(axis=1)
    out = np.full(track.shape[0], "?", dtype="<U1")
    hot = (sums == 1.0) & np.all((track == 0.0) | (track == 1.0), axis=1)
    out[hot] = np.asarray(list(BASES))[track[hot].argmax(axis=1)]
    out[np.all(track == 0.0, axis=1)] = "N"
    return "".join(out.tolist())


def revcomp(seq: str) -> str:
    return "".join(COMPLEMENT.get(b, "N") for b in reversed(seq))


# ------------------------------------------------------------------
# comparison
# ------------------------------------------------------------------

@dataclass
class CaseResult:
    case: str
    description: str
    samples: int
    variants_kept: int
    bases_compared: int
    haplotype_mismatches: int
    dosage_mismatches: int
    status: str
    note: str = ""
    expected_bases: int = 0
    completed_haplotypes: int = 0
    expected_haplotypes: int = 0


def compare_case(mat: MaterialisedCase, work: Path) -> CaseResult:
    case = mat.case
    region = f"{case.contig}:1-{case.window}"

    hap_out = work / f"{case.name}_hap.npy"
    code, meta, err = hottools_run(mat, hap_out, "haplotypes", [])
    if code != 0:
        return CaseResult(case.name, case.description, len(case.samples),
                          -1, 0, -1, -1, "hottools-error",
                          err.splitlines()[-1] if err else "no diagnostic")
    dose = None
    if not case.structural:
        dose_out = work / f"{case.name}_dose.npy"
        code, _, err = hottools_run(mat, dose_out, "dosage", [])
        if code != 0:
            return CaseResult(case.name, case.description, len(case.samples),
                              -1, 0, -1, -1, "hottools-error",
                              err.splitlines()[-1] if err else "dosage run failed")
        dose = np.load(dose_out)     # S x L x 4

    haps = np.load(hap_out)          # S x 2 x L x 4
    wanted_shape = (len(case.samples), 2, case.window, 4)
    if haps.shape != wanted_shape or (
        dose is not None and dose.shape != (len(case.samples), case.window, 4)
    ):
        return CaseResult(case.name, case.description, len(case.samples),
                          meta["variants_kept"], 0, -1, -1, "invalid-output",
                          f"unexpected tensor shape: haplotypes={haps.shape}")
    consensus_vcf = mat.consensus_input()

    if case.structural:
        # A deletion pulls bases from beyond the output window, so
        # bcftools has to edit the same larger interval hottools read
        # and the first `window` bases of its answer are the comparison.
        pad = int(meta.get("working_pad_right", 0))
        region = f"{case.contig}:1-{case.window + pad}"

    hap_mismatch = dose_mismatch = compared = completed = 0
    expected_haplotypes = 2 * len(case.samples)
    expected_bases = expected_haplotypes * case.window
    notes: list[str] = []
    for s_index, sample in enumerate(case.samples):
        expected = []
        wanted = (1,) if case.haploid_source else (1, 2)
        for hap in wanted:
            seq = bcftools_haplotype(
                mat.fasta, consensus_vcf, region, sample, hap
            )
            if seq is None:
                notes.append(f"bcftools failed for {sample} -H{hap}")
                expected = []
                break
            if case.structural:
                # The window is cut from the *plus* strand first, exactly
                # as hottools does, and only then reverse-complemented.
                seq = seq[: case.window]
            expected.append(revcomp(seq) if case.strand == "-" else seq)
        if not expected:
            continue
        if case.haploid_source:
            # --haploid-policy duplicate puts the single called allele on
            # both tracks, so bcftools -H 1 is the expectation for both.
            expected = [expected[0], expected[0]]

        got = [decode_onehot(haps[s_index, h]) for h in (0, 1)]
        if any(len(seq) != case.window for seq in expected):
            notes.append(
                f"{sample}: consensus lengths {[len(seq) for seq in expected]} "
                f"do not match required window {case.window}"
            )
            continue
        for exp, obs in zip(expected, got, strict=True):
            compared += len(exp)
            completed += 1
            hap_mismatch += sum(
                1 for a, b in zip(exp, obs, strict=True) if a != b
            )

        if dose is None:
            continue

        # dosage must be the superposition of the two haplotypes
        e0, e1 = expected
        for pos, (a, b) in enumerate(zip(e0, e1, strict=True)):
            vec = dose[s_index, pos]
            expected_vec = np.zeros(4, dtype=np.float32)
            for base in (a, b):
                if base in BASES:
                    expected_vec[BASES.index(base)] += 0.5
            ok = np.array_equal(vec, expected_vec)
            if not ok:
                dose_mismatch += 1

    status = "match" if (hap_mismatch == 0 and dose_mismatch == 0
                         and compared == expected_bases
                         and completed == expected_haplotypes
                         and not notes and compared > 0) else "MISMATCH"
    if notes or completed != expected_haplotypes:
        status = "incomplete-comparison"
    if compared == 0 and not notes:
        status = "no-comparison"
    return CaseResult(
        case.name, case.description, len(case.samples),
        meta["variants_kept"], compared, hap_mismatch, dose_mismatch,
        status, "; ".join(notes), expected_bases, completed,
        expected_haplotypes,
    )


@dataclass
class EdgeResult:
    case: str
    description: str
    expectation: str
    hottools_exit: int
    hottools_message: str
    bcftools_exit: int
    bcftools_message: str
    outcome: str
    divergence: str


def probe_edge(mat: MaterialisedCase, work: Path) -> EdgeResult:
    case = mat.case
    region = f"{case.contig}:1-{case.length}"
    out = work / f"{case.name}_edge.npy"
    code, meta, err = hottools_run(mat, out, "haplotypes", [])

    message = ""
    if code != 0:
        message = err.splitlines()[-1] if err else ""
    elif meta is not None:
        message = (
            f"kept={meta['variants_kept']} "
            f"non_snv={meta['variant_non_snv_records']} "
            f"dup_removed={meta['variant_duplicate_positions_removed']} "
            f"haploid={meta['variant_haploid_records']} "
            f"polyploid={meta['variant_polyploid_records']} "
            f"multiallelic={meta['multiallelic_records']}"
        )

    faidx = run(["samtools", "faidx", str(mat.fasta), region])
    if faidx.returncode != 0 or not faidx.stdout.strip():
        first = next(
            (ln.strip() for ln in faidx.stderr.splitlines() if ln.strip()), ""
        )
        return EdgeResult(
            case.name, case.description, case.expect, code, message[:160],
            faidx.returncode or 1,
            f"samtools faidx could not extract {region}: {first}"[:160],
            _outcome(case, code, message), case.divergence,
        )

    # Sweep every sample and both haplotypes: a record is only validated
    # by bcftools when that haplotype actually carries the ALT allele,
    # so a single -s/-H probe can silently skip the interesting record.
    applied = failures = 0
    diagnostics: list[str] = []
    worst_exit = 0
    for sample in case.samples:
        for hap in (1, 2):
            cons = run(
                ["bcftools", "consensus", "-s", sample, "-H", str(hap),
                 str(mat.vcf)],
                input=faidx.stdout,
            )
            produced = any(
                ln and not ln.startswith(">")
                for ln in cons.stdout.splitlines()
            )
            if cons.returncode != 0 or not produced:
                failures += 1
                worst_exit = cons.returncode or 1
            for line in cons.stderr.splitlines():
                text = line.strip()
                if not text:
                    continue
                if text.startswith("Applied"):
                    applied = max(applied, int(text.split()[1]))
                    continue
                if text not in diagnostics:
                    diagnostics.append(text)

    n_runs = len(case.samples) * 2
    summary = f"{n_runs - failures}/{n_runs} runs produced a sequence"
    if applied:
        summary += f", max {applied} variant(s) applied"
    if diagnostics:
        summary += "; " + " | ".join(diagnostics[:3])
    b_msg = summary[:200]
    cons = type("R", (), {"returncode": worst_exit})()

    return EdgeResult(
        case.name, case.description, case.expect, code, message[:160],
        cons.returncode, b_msg, _outcome(case, code, message),
        case.divergence,
    )


def _outcome(case, code: int, message: str) -> str:
    if case.expect == "ok":
        return "handled" if code == 0 else "unexpected-error"
    return (
        "rejected as expected" if code != 0 and case.expect in message
        else "UNEXPECTED"
    )


# ------------------------------------------------------------------
# entry point
# ------------------------------------------------------------------

def build_fuzz_cases(n: int, seed: int = 99) -> list:
    """
    Randomised cases that must still agree with bcftools exactly.

    Each draws its own contig name, length, sample count, variant
    density, phase separators and missing-genotype pattern. Unphased
    calls are constrained to be homozygous, which is the only situation
    where ``bcftools consensus -H`` is well defined without phase.
    """
    import random as _random

    from synthetic_cases import OTHER, Case, random_sequence, row

    rng = _random.Random(seed)
    cases = []
    for i in range(n):
        contig = rng.choice(["chr1", "1", "chrX", "scaffold_7", "NC_000001.11"])
        length = rng.randint(80, 400)
        seq = random_sequence(rng, length)
        if rng.random() < 0.3:                      # soft-masked stretch
            a = rng.randrange(0, max(1, length - 20))
            seq = seq[:a] + seq[a : a + 20].lower() + seq[a + 20 :]
        n_samples = rng.randint(1, 12)
        samples = [f"s{j}" for j in range(n_samples)]
        n_variants = rng.randint(1, max(2, length // 8))
        positions = sorted(rng.sample(range(1, length + 1), n_variants))

        rows = []
        for p in positions:
            ref = seq[p - 1].upper()
            alt = rng.choice(OTHER[ref])
            gts = []
            for _ in samples:
                draw = rng.random()
                if draw < 0.10:
                    gts.append(rng.choice([".|.", "./.", ".|1", "1|.",
                                           ".|0", "0|."]))
                elif draw < 0.30:
                    # unphased, but homozygous so phase is irrelevant
                    gts.append(rng.choice(["0/0", "1/1"]))
                else:
                    gts.append(rng.choice(["0|0", "0|1", "1|0", "1|1"]))
            rows.append(row(p, ref, alt, gts))

        cases.append(Case(
            f"fuzz_{i:03d}",
            f"{contig} {length} bp, {n_samples} samples, {n_variants} SNVs",
            contig, seq, samples, rows,
            strand="-" if rng.random() < 0.25 else "+",
            hottools_flags=(["--strand", "-"] if rng.random() < 0 else []),
        ))
        if cases[-1].strand == "-":
            cases[-1].hottools_flags = ["--strand", "-"]
    return cases


def build_structural_fuzz_cases(n: int, seed: int = 4242) -> list:
    """
    Randomised length-changing cases that must agree with bcftools.

    Records are spaced so their claimed intervals do not overlap, which
    is what a variant caller produces and the only situation where both
    tools have the same record set to apply. Genotypes are fully phased,
    because a length-changing edit only has meaning on a known
    haplotype.
    """
    import random as _random

    from synthetic_cases import (
        BASES,
        END_HEADER,
        OTHER,
        Case,
        random_sequence,
        row,
    )

    rng = _random.Random(seed)
    cases = []
    for i in range(n):
        contig = rng.choice(["chr1", "1", "chrX", "scaffold_3"])
        window = rng.randint(80, 220)
        # Room beyond the window for deletions to pull bases from.
        length = window + rng.randint(60, 160)
        seq = random_sequence(rng, length)
        n_samples = rng.randint(1, 8)
        samples = [f"h{j}" for j in range(n_samples)]

        rows, position, used_end = [], rng.randint(2, 10), False
        while position < window:
            kind = rng.choice(["snv", "mnv", "del", "ins", "symdel"])
            ref_base = seq[position - 1].upper()
            info = "."
            if kind == "snv":
                ref, alt, span = ref_base, rng.choice(OTHER[ref_base]), 1
            elif kind == "mnv":
                width = rng.randint(2, 4)
                ref = seq[position - 1 : position - 1 + width].upper()
                alt = "".join(rng.choice(OTHER[b]) for b in ref)
                span = width
            elif kind == "del":
                width = rng.randint(1, 9)
                ref = seq[position - 1 : position + width].upper()
                alt, span = ref_base, width + 1
            elif kind == "ins":
                width = rng.randint(1, 6)
                ref = ref_base
                alt = ref + "".join(rng.choice(BASES) for _ in range(width))
                span = 1
            else:
                width = rng.randint(2, 10)
                ref, alt = ref_base, "<DEL>"
                info, span, used_end = f"END={position + width}", width + 1, True
            gts = [
                rng.choice(["0|0", "0|1", "1|0", "1|1"]) for _ in samples
            ]
            rows.append(row(position, ref, alt, gts, info=info))
            # A gap of at least one base keeps the claims disjoint.
            position += span + rng.randint(1, 12)

        if not rows:
            continue
        minus = rng.random() < 0.25
        flags = ["--region", f"{contig}:1-{window}"]
        if minus:
            flags += ["--strand", "-"]
        cases.append(Case(
            f"sfuzz_{i:03d}",
            f"{contig} {window} bp window, {n_samples} samples, "
            f"{len(rows)} records of every class",
            contig, seq, samples, rows,
            variant_classes="all", structural=True, out_length=window,
            strand="-" if minus else "+", hottools_flags=flags,
            extra_header=END_HEADER if used_end else [],
        ))
    return cases


def main() -> int:
    n_fuzz = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    root = Path("validation/cases")
    work = Path("validation/work")
    for directory in (root, work):
        directory.mkdir(parents=True, exist_ok=True)

    cases = (
        build_concordance_cases()
        + build_fuzz_cases(n_fuzz)
        + build_structural_fuzz_cases(n_fuzz // 2)
    )
    concordance = [c.materialise(root) for c in cases]
    edges = [c.materialise(root) for c in build_edge_cases()]

    results = [compare_case(m, work) for m in concordance]
    edge_results = [probe_edge(m, work) for m in edges]

    Path("validation/concordance.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2) + "\n"
    )
    Path("validation/edge_cases.json").write_text(
        json.dumps([asdict(r) for r in edge_results], indent=2) + "\n"
    )

    total = sum(r.bases_compared for r in results)
    bad = [r for r in results if r.status != "match"]
    named = [
        r for r in results
        if not r.case.startswith(("fuzz_", "sfuzz_"))
    ]
    fuzz = [
        r for r in results if r.case.startswith(("fuzz_", "sfuzz_"))
    ]
    print(f"concordance cases: {len(results)} "
          f"({len(named)} named + {len(fuzz)} randomised), "
          f"failures: {len(bad)}")
    print(f"bases compared: {total:,}")
    for r in named:
        print(f"  {r.status:>7}  {r.case:<24} "
              f"{r.bases_compared:>7,} bases  "
              f"hap_mm={r.haplotype_mismatches} dose_mm={r.dosage_mismatches}")
    if fuzz:
        print(f"  {'match' if not any(f in bad for f in fuzz) else 'MISMATCH':>7}"
              f"  {'randomised x' + str(len(fuzz)):<24} "
              f"{sum(f.bases_compared for f in fuzz):>7,} bases  "
              f"hap_mm={sum(f.haplotype_mismatches for f in fuzz)} "
              f"dose_mm={sum(f.dosage_mismatches for f in fuzz)}")
    for r in bad:
        print(f"    FAILING: {r.case} {r.status} {r.note}")
    print()
    for e in edge_results:
        print(f"  {e.outcome:>20}  {e.case}")
    bad_edges = [e for e in edge_results if e.outcome not in {
        "handled", "rejected as expected"
    }]
    return int(bool(bad or bad_edges or not results))


if __name__ == "__main__":
    raise SystemExit(main())
