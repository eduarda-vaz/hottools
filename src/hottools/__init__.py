"""Hottools: fast individualized DNA sequence encoding from VCF/BCF."""

from .encoding import (
    encode_consensus_batch,
    encode_dosage_batch,
    encode_haplotype_batch,
    encode_haplotypes,
)
from .memory import MemoryBudgetWarning, MemoryPlan, choose_batch_size
from .pipeline import RunConfig, run_hottools
from .reference import ReferenceReader, one_hot_reference, read_reference_interval
from .regions import (
    Region,
    parse_region,
    read_mask,
    read_region_file,
    region_from_center,
)
from .variants import (
    EmptyRegionWarning,
    HaploidWarning,
    MultiallelicWarning,
    StructuralEditWarning,
    VariantData,
    VariantStats,
    extract_variants,
)
from .writers import SampleNameCollisionWarning

#: Warning categories are exported too, so a caller can promote any of
#: them to an error with ``warnings.simplefilter``.
__all__ = [
    "EmptyRegionWarning",
    "HaploidWarning",
    "MemoryBudgetWarning",
    "MemoryPlan",
    "MultiallelicWarning",
    "ReferenceReader",
    "Region",
    "RunConfig",
    "SampleNameCollisionWarning",
    "StructuralEditWarning",
    "VariantData",
    "VariantStats",
    "choose_batch_size",
    "encode_consensus_batch",
    "encode_dosage_batch",
    "encode_haplotype_batch",
    "encode_haplotypes",
    "extract_variants",
    "one_hot_reference",
    "parse_region",
    "read_mask",
    "read_reference_interval",
    "read_region_file",
    "region_from_center",
    "run_hottools",
]
__version__ = "1.0.0"
