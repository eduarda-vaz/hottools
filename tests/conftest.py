from pathlib import Path
import pytest

@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"

@pytest.fixture
def phased_vcf(data_dir: Path) -> Path:
    return data_dir / "phased.vcf.gz"

@pytest.fixture
def unphased_vcf(data_dir: Path) -> Path:
    return data_dir / "unphased.vcf.gz"

@pytest.fixture
def duplicates_vcf(data_dir: Path) -> Path:
    return data_dir / "duplicates.vcf.gz"

@pytest.fixture
def multiallelic_vcf(data_dir: Path) -> Path:
    return data_dir / "multiallelic.vcf.gz"

@pytest.fixture
def indel_vcf(data_dir: Path) -> Path:
    return data_dir / "indel.vcf.gz"

@pytest.fixture
def chr_prefix_vcf(data_dir: Path) -> Path:
    return data_dir / "chr_prefix.vcf.gz"

@pytest.fixture
def no_chr_prefix_vcf(data_dir: Path) -> Path:
    return data_dir / "no_chr_prefix.vcf.gz"

@pytest.fixture
def unsorted_vcf(data_dir: Path) -> Path:
    return data_dir / "unsorted.vcf.gz"

@pytest.fixture(scope="session")
def synthetic_phased_vcf(data_dir: Path) -> Path:
    return data_dir / "synthetic.phased.chr12_1_61.vcf.gz"

@pytest.fixture(scope="session")
def synthetic_unphased_vcf(data_dir: Path) -> Path:
    return data_dir / "synthetic.unphased.chr12_1_61.vcf.gz"

@pytest.fixture(scope="session")
def synthetic_vcf_samples(data_dir: Path) -> Path:
    return data_dir / "synthetic.10samples.txt"

@pytest.fixture(scope="session")
def grch38_test_fasta(data_dir: Path) -> Path:
    return data_dir / "GRCh38_chr12_1_61.fa"
