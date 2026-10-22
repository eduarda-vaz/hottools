import pytest

from hottools.regions import (
    Region,
    parse_region,
    read_region_file,
    region_from_center,
    resolve_coordinate_system,
)


def test_region_geometry_and_strand():
    r = Region("chr1", 100, 199)
    assert r.length == 100
    assert r.label == "chr1:100-199"
    assert r.strand == "+"
    assert r.reverse_complement is False
    assert r.with_strand("-").reverse_complement is True


@pytest.mark.parametrize(
    "chrom,start,end,strand",
    [("chr1", 0, 10, "+"), ("chr1", 10, 9, "+"), ("chr1", 1, 10, "?"),
     ("", 1, 10, "+")],
)
def test_region_rejects_invalid_construction(chrom, start, end, strand):
    with pytest.raises(ValueError):
        Region(chrom, start, end, strand)


def test_parse_region_variants():
    assert parse_region("chr1:100-200") == Region("chr1", 100, 200, "+")
    assert parse_region("chr1:1,000-2,000") == Region("chr1", 1000, 2000)
    assert parse_region("chr1:1-2", "-").strand == "-"
    # A colon inside the contig name must not break parsing.
    assert parse_region("HLA:A*01:01:5-9").chrom == "HLA:A*01:01"


@pytest.mark.parametrize("text", ["chr1", "chr1:abc-def", "chr1:10", "chr1:20-10"])
def test_parse_region_rejects_malformed(text):
    with pytest.raises(ValueError):
        parse_region(text)


@pytest.mark.parametrize(
    "length,expected",
    [(1, (100, 100)), (2, (100, 101)), (3, (99, 101)), (5, (98, 102))],
)
def test_region_from_center_is_symmetric_for_odd_lengths(length, expected):
    region = region_from_center("chr1:100", length)
    assert (region.start, region.end) == expected
    assert region.length == length


def test_region_from_center_rejects_underflow():
    with pytest.raises(ValueError, match="before coordinate 1"):
        region_from_center("chr1:5", 100)


def test_resolve_coordinate_system():
    assert resolve_coordinate_system("x.bed") == "bed"
    assert resolve_coordinate_system("x.BED.gz") == "bed"
    assert resolve_coordinate_system("x.txt") == "1based"
    assert resolve_coordinate_system("x.bed", "1based") == "1based"


def test_region_file_1based_with_strand_column(tmp_path):
    path = tmp_path / "regions.txt"
    path.write_text(
        "# comment\n"
        "chr1:1-100\n"
        "chr2 5 15\n"
        "chr3 5 15 -\n"
        "\n",
        encoding="utf-8",
    )
    regions = read_region_file(path)
    assert regions == [
        Region("chr1", 1, 100, "+"),
        Region("chr2", 5, 15, "+"),
        Region("chr3", 5, 15, "-"),
    ]
    assert regions[1].length == 11


def test_region_file_bed_is_zero_based_half_open(tmp_path):
    path = tmp_path / "regions.bed"
    path.write_text(
        "track name=x\n"
        "chr2\t4\t15\n"
        "chr3\t4\t15\tname\t0\t-\n",
        encoding="utf-8",
    )
    regions = read_region_file(path)
    # BED [4,15) is 1-based 5..15 -> 11 bp, matching the 1-based file above.
    assert regions == [
        Region("chr2", 5, 15, "+"),
        Region("chr3", 5, 15, "-"),
    ]


def test_region_file_default_strand_applies_only_without_column(tmp_path):
    path = tmp_path / "regions.txt"
    path.write_text("chr1 1 10\nchr2 1 10 +\n", encoding="utf-8")
    regions = read_region_file(path, default_strand="-")
    assert [r.strand for r in regions] == ["-", "+"]


def test_region_file_reports_offending_line(tmp_path):
    path = tmp_path / "regions.txt"
    path.write_text("chr1 1 10\nchr2 oops\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_region_file(path)


def test_region_file_rejects_empty(tmp_path):
    path = tmp_path / "regions.txt"
    path.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No regions"):
        read_region_file(path)
