"""
Randomised comparison against a slow, independent model.

The oracle edits the reference as text - sort the edits, copy the
untouched stretches, splice in the new bases - and then cuts the window
around the anchor. It is far too slow for production and shares no code
with hottools, which is what makes it a useful test oracle.

Two levels:

- the transform alone: 1-5 non-overlapping edits on a short reference,
  presented in random order, with a random output length and anchor;
- the whole pipeline: random VCF records of every class over four phased
  samples, a random window, anchor and strand.
"""

from __future__ import annotations

import random

import pytest
from sv_helpers import (
    BASES,
    Rec,
    allele_of,
    haps,
    oracle_window,
    other,
    random_sequence,
    revcomp,
    run_paths,
    write_case,
)
from test_signatures import transform_result

from hottools.edits import Edit

# --------------------------------------------------------------------
# transform level
# --------------------------------------------------------------------

def oracle_from_intervals(seq, intervals, *, anchor, offset, length):
    """Apply (start, end, new) edits as text, then cut the window."""
    entries: list[tuple[int | None, str]] = []
    cursor = 1
    # An insertion at the same coordinate as a deletion goes first: it
    # sits before the base the deletion starts at.
    for start, end, new in sorted(intervals, key=lambda t: (t[0], t[1] - t[0])):
        entries += [(p, seq[p - 1]) for p in range(cursor, start)]
        entries += [(None, b) for b in new]
        cursor = max(cursor, end)
    entries += [(p, seq[p - 1]) for p in range(cursor, len(seq) + 1)]
    index = next((i for i, (p, _) in enumerate(entries)
                  if p is not None and p >= anchor), len(entries))
    lo = index - offset
    return "".join(entries[i][1] if 0 <= i < len(entries) else "-"
                   for i in range(lo, lo + length))


def random_edits(rng: random.Random, n_ref: int) -> list[Edit]:
    edits: list[Edit] = []
    cursor = rng.randint(1, 6)
    last_was_insertion_at = None
    for _ in range(rng.randint(1, 5)):
        kind = rng.choice(["del", "ins", "replace"])
        if kind == "ins":
            if cursor == last_was_insertion_at:
                cursor += 1       # two insertions at one point are ambiguous
            if cursor > n_ref:
                break
            new = "".join(rng.choice(BASES) for _ in range(rng.randint(1, 6)))
            edits.append(Edit(cursor, cursor, new, "ins", 1))
            last_was_insertion_at = cursor
            # The next edit may start right here, even a deletion.
            cursor += rng.choice([0, 0, 1, 3])
            continue
        size = rng.randint(1, 8)
        if cursor + size > n_ref + 1:
            break
        if kind == "del":
            edits.append(Edit(cursor, cursor + size, "", "del", 1))
        else:
            new_len = rng.choice([n for n in range(0, 10) if n != size])
            new = "".join(rng.choice(BASES) for _ in range(new_len))
            cls = "del" if new_len < size else "ins"
            edits.append(Edit(cursor, cursor + size, new, cls, 1))
        cursor += size + rng.choice([0, 1, 2, 5])
        last_was_insertion_at = None
    return edits


@pytest.mark.parametrize("seed", range(300))
def test_transform_matches_the_oracle(seed):
    rng = random.Random(seed)
    seq = random_sequence(seed, rng.randint(20, 60))
    edits = random_edits(rng, len(seq))
    if not edits:
        pytest.skip("no room for an edit")
    shuffled = edits[:]
    rng.shuffle(shuffled)                      # input order must not matter
    out_length = rng.randint(5, len(seq) + 15)
    anchor = rng.randint(1, len(seq))
    offset = rng.randint(0, out_length - 1)
    got, transform = transform_result(seq, shuffled, out_length,
                                      anchor=anchor, offset=offset)
    want = oracle_from_intervals(
        seq, [(e.start, e.end, e.alt) for e in edits],
        anchor=anchor, offset=offset, length=out_length,
    )
    assert got == want, (edits, anchor, offset)
    assert transform.n_skipped == 0
    assert transform.n_unfilled == want.count("-")


# --------------------------------------------------------------------
# pipeline level
# --------------------------------------------------------------------

def random_records(rng: random.Random, seq: str, n_samples: int) -> list[Rec]:
    """
    Biallelic records of every class at distinct POS whose normalised
    intervals never overlap (they may touch). Overlap resolution has its
    own hand-written tests; here the subject is coordinates.
    """
    records: list[Rec] = []
    last_pos, claim_end = 0, 1
    n = len(seq)
    while True:
        kind = rng.choice(["snv", "mnv", "del", "ins", "symdel"])
        size = rng.randint(1, 9)
        # Earliest POS whose normalised start is not before claim_end.
        earliest = claim_end if kind in {"snv", "mnv"} else claim_end - 1
        pos = max(last_pos + 1, earliest) + rng.choice([0, 0, 1, 2, 4, 8])
        if pos + size + 1 > n:
            break
        base = seq[pos - 1]
        gts = tuple(rng.choice(["0|0", "0|1", "1|0", "1|1"])
                    for _ in range(n_samples))
        if kind == "snv":
            rec, end = Rec(pos, base, other(base, rng.randint(0, 2)), gts), pos + 1
        elif kind == "mnv":
            size = max(2, size // 2)
            ref = seq[pos - 1 : pos - 1 + size]
            rec = Rec(pos, ref, "".join(other(b) for b in ref), gts)
            end = pos + size
        elif kind == "del":
            rec = Rec(pos, seq[pos - 1 : pos + size], base, gts)
            end = pos + size + 1
        elif kind == "ins":
            new = "".join(rng.choice(BASES) for _ in range(size))
            rec, end = Rec(pos, base, base + new, gts), pos + 1
        else:
            rec = Rec(pos, base, "<DEL>", gts, f"END={pos + size}")
            end = pos + size + 1
        records.append(rec)
        last_pos, claim_end = pos, end
    return records


@pytest.mark.parametrize("seed", range(60))
def test_pipeline_matches_the_oracle(tmp_path, seed):
    rng = random.Random(1000 + seed)
    seq = random_sequence(1000 + seed, 320)
    samples = ["A", "B", "C", "D"]
    records = random_records(rng, seq, len(samples))
    start = rng.randint(1, 120)
    length = rng.randint(40, 150)
    end = min(len(seq), start + length - 1)
    length = end - start + 1
    anchor_mode = rng.choice(["start", "center"])
    strand = rng.choice(["+", "-"])
    offset = 0 if anchor_mode == "start" else (length - 1) // 2

    fasta, vcf = write_case(tmp_path, seq, records, samples)
    t, meta = run_paths(tmp_path, fasta, vcf, start=start, end=end,
                        strand=strand, anchor=anchor_mode,
                        batch_size=rng.choice([1, 3, 4]))
    assert meta["variant_overlapping_records_suppressed"] == 0
    for s, (h1, h2) in enumerate(haps(t)):
        for hap, got in enumerate((h1, h2)):
            alleles = [allele_of(r.gts[s], hap) for r in records]
            want = oracle_window(seq, records, alleles, anchor=start + offset,
                                 offset=offset, length=length)
            if strand == "-":
                want = revcomp(want)
            assert got == want, (samples[s], hap, anchor_mode, strand)


def test_the_generators_cover_every_class():
    """Guard against the random generators silently narrowing."""
    rng = random.Random(0)
    seen = set()
    for seed in range(30):
        seq = random_sequence(seed, 320)
        for rec in random_records(rng, seq, 1):
            if rec.alt == "<DEL>":
                seen.add("symdel")
            elif len(rec.ref) == len(rec.alt):
                seen.add("snv" if len(rec.ref) == 1 else "mnv")
            else:
                seen.add("del" if len(rec.ref) > len(rec.alt) else "ins")
    assert seen == {"snv", "mnv", "del", "ins", "symdel"}
    kinds = set()
    for seed in range(50):
        for e in random_edits(random.Random(seed), 60):
            kinds.add("ins" if e.start == e.end else
                      "del" if not e.alt else "replace")
    assert kinds == {"ins", "del", "replace"}
