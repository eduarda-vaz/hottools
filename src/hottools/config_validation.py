"""Small input-independent checks shared by configuration and extraction."""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Sequence
from numbers import Integral, Real


def positive_integer(name, value, *, optional=False):
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def finite_number(name, value, *, lower=0, upper=None, optional=False):
    if optional and value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    if value < lower or (upper is not None and value > upper):
        raise ValueError(
            f"{name} must be between {lower} and {upper}"
            if upper is not None
            else f"{name} must be >= {lower}"
        )


def validate_sample_request(samples):
    if samples is None:
        return
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise ValueError("samples must be a sequence of sample IDs, not a string")
    if not samples:
        raise ValueError("Sample selection is empty.")
    if any(not isinstance(name, str) or not name.strip() for name in samples):
        raise ValueError("samples must contain nonempty string IDs")
    duplicates = sorted(name for name, n in Counter(samples).items() if n > 1)
    if duplicates:
        raise ValueError(f"Duplicate sample ID(s) requested: {', '.join(duplicates)}")


def validate_site_options(
    *, threads=None, samples=None, min_qual=None, min_af=None, max_af=None
):
    positive_integer("threads", threads, optional=True)
    validate_sample_request(samples)
    finite_number("min_qual", min_qual, optional=True)
    finite_number("min_af", min_af, upper=1, optional=True)
    finite_number("max_af", max_af, upper=1, optional=True)
    if min_af is not None and max_af is not None and min_af > max_af:
        raise ValueError("min_af must be <= max_af")


def boolean(name, value):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def nonempty_path(name, value):
    try:
        path = os.fspath(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty filesystem path") from exc
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{name} must be a nonempty filesystem path")


def validate_random_assignment(mode, unphased_policy, random_seed):
    """Random pseudo-phasing requires an explicit, portable integer seed."""
    if unphased_policy == "random":
        if mode != "haplotypes":
            raise ValueError("unphased_policy='random' requires mode='haplotypes'")
        if (isinstance(random_seed, bool) or not isinstance(random_seed, Integral)
                or not 0 <= random_seed <= 2**64 - 1):
            raise ValueError(
                "random_seed must be an explicit integer between 0 and 2**64-1"
            )
    elif random_seed is not None:
        raise ValueError("random_seed requires unphased_policy='random'")
