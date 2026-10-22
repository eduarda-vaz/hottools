"""
Signature grouping: which haplotypes share a compiled transform.

Signatures are packed with ``np.packbits`` and grouped with
``np.unique``, so these tests probe exactly the places such code tends
to break: every carrier pattern of three edits, bit patterns that cross
byte boundaries (9, 17 and 65 edits), and the same signature landing in
different sample batches.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from sv_helpers import (
    deletion,
    haps,
    oracle_window,
    random_sequence,
    run_paths,
    write_case,
)

from hottools.constants import N_ALLELE_SLOTS
from hottools.edits import Edit, build_structural_edits, parse_variant_classes
from hottools.transforms import (
    apply_transform,
    compile_transform,
    haplotype_signatures,
)
from hottools.variants import extract_variants

SEQ = random_sequence(4242, 420)


def pack(a1: int, a2: int) -> int:
    return a1 * N_ALLELE_SLOTS + a2


def edits_from_patterns(patterns: list[tuple[int, ...]]):
    """
    One one-base deletion per edit, one haplotype per pattern.

    ``patterns[h][e]`` is 1 when haplotype ``h`` carries edit ``e``;
    haplotypes pair into samples in order.
    """
    n_edits = len(patterns[0])
    pairs = [(e, Edit(10 + 5 * e, 11 + 5 * e, "", "del", 1))
             for e in range(n_edits)]
    genotypes = np.array(
        [[pack(patterns[2 * s][e], patterns[2 * s + 1][e])
          for s in range(len(patterns) // 2)]
         for e in range(n_edits)],
        dtype=np.uint8,
    )
    return build_structural_edits(
        pairs, np.array([p.start - 1 for _, p in pairs], dtype=np.int32),
        genotypes,
    )


def assert_grouping_matches(patterns, signatures):
    """Equal patterns share a group, distinct patterns never do."""
    carriers = np.asarray(patterns, dtype=bool)
    np.testing.assert_array_equal(signatures.carriers, carriers)
    for h in range(len(patterns)):
        np.testing.assert_array_equal(
            signatures.unique[signatures.group[h]], carriers[h]
        )
    for a, b in itertools.combinations(range(len(patterns)), 2):
        same = patterns[a] == patterns[b]
        assert (signatures.group[a] == signatures.group[b]) == same, (a, b)
    assert signatures.n_groups == len(set(patterns))


# --------------------------------------------------------------------
# 23. every carrier pattern of three edits
# --------------------------------------------------------------------

THREE = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
         (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]


def test_all_eight_patterns_get_distinct_groups():
    assert_grouping_matches(THREE, haplotype_signatures(
        edits_from_patterns(THREE)
    ))


def test_all_eight_patterns_end_to_end(tmp_path):
    """Each haplotype gets exactly its own deletions, nothing else."""
    def gt(e, s):
        return f"{THREE[2 * s][e]}|{THREE[2 * s + 1][e]}"

    records = [
        deletion(SEQ, pos, size, [gt(e, s) for s in range(4)])
        for e, (pos, size) in enumerate([(40, 2), (100, 3), (160, 4)])
    ]
    fasta, vcf = write_case(tmp_path, SEQ, records)
    t, meta = run_paths(tmp_path, fasta, vcf)
    assert meta["structural_signatures"] == 8
    for s, (h1, h2) in enumerate(haps(t)):
        for hap, got in enumerate((h1, h2)):
            alleles = list(THREE[2 * s + hap])
            assert got == oracle_window(SEQ, records, alleles, anchor=1,
                                        offset=0, length=200), (s, hap)


# --------------------------------------------------------------------
# 24. patterns that cross packbits byte boundaries
# --------------------------------------------------------------------

@pytest.mark.parametrize("n_edits", [8, 9, 16, 17, 64, 65])
def test_grouping_across_byte_boundaries(n_edits):
    none = (0,) * n_edits
    last_only = (0,) * (n_edits - 1) + (1,)
    all_but_last = (1,) * (n_edits - 1) + (0,)
    every = (1,) * n_edits
    # Bits 7 and 8 sit either side of the first byte boundary.
    bit7 = tuple(int(e == 7) for e in range(n_edits))
    bit8 = tuple(int(e == 8) for e in range(n_edits)) if n_edits > 8 else bit7
    patterns = [none, last_only, all_but_last, every,
                last_only, bit7, bit8, none]
    assert_grouping_matches(patterns, haplotype_signatures(
        edits_from_patterns(patterns)
    ))


def test_sixty_five_edits_end_to_end(tmp_path):
    """The bit in the ninth packed byte must reach the sequence."""
    n = 65
    positions = [10 + 6 * e for e in range(n)]           # 10 .. 394
    hap_patterns = [
        [int(e == n - 1) for e in range(n)],             # only the 65th
        [1] * n,                                         # all
        [1] * (n - 1) + [0],                             # all but the 65th
        [int(e == 0) for e in range(n)],                 # only the first
    ]
    records = [
        deletion(SEQ, pos, 1, [f"{hap_patterns[0][e]}|{hap_patterns[1][e]}",
                               f"{hap_patterns[2][e]}|{hap_patterns[3][e]}"])
        for e, pos in enumerate(positions)
    ]
    fasta, vcf = write_case(tmp_path, SEQ, records, samples=["A", "B"])
    t, meta = run_paths(tmp_path, fasta, vcf, end=410)
    assert meta["structural_edits"] == n
    assert meta["structural_signatures"] == 4
    decoded = [h for pair in haps(t) for h in pair]
    for h, pattern in enumerate(hap_patterns):
        assert decoded[h] == oracle_window(SEQ, records, pattern, anchor=1,
                                           offset=0, length=410), h
    # The two that differ only in the 65th deletion really differ.
    assert decoded[1] != decoded[2]


# --------------------------------------------------------------------
# 25. one signature across several sample batches
# --------------------------------------------------------------------

def test_a_signature_split_across_batches_uses_one_transform(tmp_path):
    shared = ["1|0", "0|1", "1|1"]
    gts = {
        "S1": shared, "S2": ["0|0", "1|1", "0|0"], "S3": ["1|1", "0|0", "0|1"],
        "S4": ["0|1", "0|0", "1|0"], "S5": ["0|0", "0|0", "0|0"], "S6": shared,
    }
    samples = list(gts)
    records = [
        deletion(SEQ, pos, size, [gts[s][e] for s in samples])
        for e, (pos, size) in enumerate([(40, 3), (90, 5), (150, 2)])
    ]
    fasta, vcf = write_case(tmp_path, SEQ, records, samples=samples)

    edits = extract_variants(
        vcf_path=vcf, chrom="chr1", start=1, end=len(SEQ), reference_seq=SEQ,
        mode="haplotypes", variant_classes=parse_variant_classes("all"),
    ).structural
    groups = haplotype_signatures(edits).group
    # S1 (batch 0) and S6 (batch 2 at batch_size 2) share both groups.
    assert groups[0] == groups[10] and groups[1] == groups[11]

    t, _ = run_paths(tmp_path, fasta, vcf, batch_size=2)
    np.testing.assert_array_equal(t[0], t[5])
    for s in (0, 5):
        for hap in (0, 1):
            alleles = [int(gts[samples[s]][e].split("|")[hap])
                       for e in range(3)]
            assert haps(t)[s][hap] == oracle_window(
                SEQ, records, alleles, anchor=1, offset=0, length=200
            )


# --------------------------------------------------------------------
# transform-level boundary semantics
# --------------------------------------------------------------------

BASES = "ACGT"


def _onehot(seq: str) -> np.ndarray:
    out = np.zeros((len(seq), 4), dtype=np.float32)
    out[np.arange(len(seq)), [BASES.index(b) for b in seq]] = 1
    return out


def _decode(rows: np.ndarray) -> str:
    return "".join(BASES[int(r.argmax())] if r.sum() else "-" for r in rows)


def transform_result(seq, edit_list, out_length, anchor=1, offset=0,
                     policy="right"):
    pairs = list(enumerate(edit_list))
    genotypes = np.full((len(edit_list), 1), pack(1, 1), dtype=np.uint8)
    edits = build_structural_edits(
        pairs, np.array([e.start for e in edit_list], dtype=np.int32),
        genotypes,
    )
    transform = compile_transform(
        np.arange(len(edit_list)), edits, work_start=1, work_end=len(seq),
        out_length=out_length, anchor_ref=anchor, anchor_offset=offset,
        anchor_deletion_policy=policy,
    )
    out = np.zeros((1, out_length, 4), dtype=np.float32)
    apply_transform(_onehot(seq)[None], np.array([0]), transform,
                    edits.insert_onehot(np.dtype(np.float32)), out)
    return _decode(out[0]), transform


@pytest.mark.parametrize("order", ["insertion first", "deletion first"])
def test_insertion_at_a_deletions_start_applies_in_either_order(order):
    seq = random_sequence(5, 40)
    ins = Edit(15, 15, "GG", "ins", 1)
    dele = Edit(15, 20, "", "del", 1)
    edit_list = [ins, dele] if order == "insertion first" else [dele, ins]
    got, transform = transform_result(seq, edit_list, 30)
    assert transform.n_skipped == 0
    assert got == (seq[:14] + "GG" + seq[19:])[:30]


def test_a_deletion_through_the_anchor_to_the_end_keeps_left_context():
    """
    Nothing survives after the deleted anchor, so the anchor sits just
    past the last surviving base: the five positions left of it are
    bases 5-9, and everything from the anchor on is empty. The old
    fallback put the anchor at index 0 instead, shifting the whole window
    upstream ('-----' then bases 1-9).
    """
    seq = random_sequence(6, 40)
    got, transform = transform_result(seq, [Edit(10, 41, "", "del", 1)], 20,
                                      anchor=15, offset=5)
    assert got == seq[4:9] + "-" * 15
    assert transform.n_unfilled == 15
    from sv_helpers import Rec, oracle_window
    rec = Rec(9, seq[8:40], seq[8], ("1|1",))          # deletes 10-40
    assert got == oracle_window(seq, [rec], [1], anchor=15, offset=5,
                                length=20)


def test_anchor_deletion_policy_error_at_transform_level():
    seq = random_sequence(6, 40)
    with pytest.raises(ValueError, match="anchor"):
        transform_result(seq, [Edit(10, 20, "", "del", 1)], 20, anchor=15,
                         offset=5, policy="error")


# An insertion exactly at the window's first base is inside the window,
# just as one after its last base is. It used to be dropped because its
# end equals ``work_start``, the test that rightly drops a deletion
# ending there.
@pytest.mark.parametrize("offset, want_prefix", [(2, "GG"), (0, "")])
def test_an_insertion_at_the_window_start_is_kept(offset, want_prefix):
    seq = random_sequence(9, 30)
    got, transform = transform_result(
        seq, [Edit(1, 1, "GG", "ins", 1)], 10, anchor=1, offset=offset,
    )
    assert transform.n_skipped == 0
    # With context left of the anchor the inserted bases fill it; with
    # none, they fall before the window, as any left-of-anchor insertion.
    want = "-" * (offset - len(want_prefix)) + want_prefix + seq
    assert got == want[:10]

