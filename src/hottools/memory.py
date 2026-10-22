from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .constants import Mode
from .memory_limits import effective_memory_available

#: Floor held back from the batch budget for the interpreter, HTSlib,
#: writer buffers and transient NumPy scratch.
DEFAULT_SAFETY_BYTES = 256 * 1024**2


class MemoryBudgetWarning(RuntimeWarning):
    """Available memory is too small for a comfortable batch."""


@dataclass(slots=True)
class MemoryPlan:
    batch_size: int
    bytes_per_sample: int
    estimated_full_output_bytes: int
    memory_budget_bytes: int | None
    available_memory_bytes: int | None
    peak_multiplier: float = 1.0
    extra_bytes: int = 0
    safety_bytes: int = 0
    memory_constraints: dict[str, int] = field(default_factory=dict)

    @property
    def estimated_peak_batch_bytes(self) -> int:
        """Transient RAM for one batch, including in-place copies."""
        return int(
            self.batch_size * self.bytes_per_sample * self.peak_multiplier
        )


def output_shape_per_sample(
    sequence_length: int,
    mode: Mode,
    layout: str = "samples_first",
) -> tuple[int, ...]:
    if layout not in {"samples_first", "channels_first"}:
        raise ValueError(
            "layout must be 'samples_first' or 'channels_first'"
        )
    if mode in ("dosage", "consensus"):
        return (
            (sequence_length, 4)
            if layout == "samples_first"
            else (4, sequence_length)
        )
    if mode == "haplotypes":
        return (
            (2, sequence_length, 4)
            if layout == "samples_first"
            else (2, 4, sequence_length)
        )
    raise ValueError(
        f"mode must be 'dosage', 'haplotypes' or 'consensus'; got {mode!r}"
    )


def full_output_shape(
    n_samples: int,
    sequence_length: int,
    mode: Mode,
    layout: str = "samples_first",
) -> tuple[int, ...]:
    """Shape of the complete output tensor for all samples."""
    return (n_samples, *output_shape_per_sample(sequence_length, mode, layout))


def sequence_axis(mode: Mode, layout: str = "samples_first") -> int:
    """
    Axis of the full output tensor that indexes genomic position.

    Sample axis is 0; haplotype mode inserts a track axis at 1; the
    channel axis comes before the sequence axis in channels_first.
    """
    shape = output_shape_per_sample(1, mode, layout)   # validates both
    axis = 1 + (1 if mode == "haplotypes" else 0)
    return axis + (1 if layout == "channels_first" else 0) if shape else axis


def bytes_per_sample(
    sequence_length: int,
    mode: Mode,
    dtype: np.dtype | str,
    layout: str = "samples_first",
) -> int:
    shape = output_shape_per_sample(sequence_length, mode, layout)
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize


def peak_multiplier_for(
    *,
    reverse_complement: bool = False,
    output_format: str = "npy",
) -> float:
    """
    How many copies of one batch tensor are live at peak.

    The encoder itself holds exactly one batch. Reverse complement adds
    a fancy-index copy, and the compressed/serialising writers buffer a
    contiguous copy of a possibly non-contiguous batch.
    """
    multiplier = 1.0
    if reverse_complement:
        multiplier += 1.0
    if output_format.lower() in {"npz", "pt"}:
        multiplier += 1.0
    return multiplier


def choose_batch_size(
    *,
    n_samples: int,
    sequence_length: int,
    mode: Mode,
    dtype: np.dtype | str,
    requested: int | str | None = "auto",
    memory_fraction: float = 0.40,
    layout: str = "samples_first",
    peak_multiplier: float = 1.0,
    extra_bytes: int = 0,
    safety_bytes: int = DEFAULT_SAFETY_BYTES,
) -> MemoryPlan:
    """
    Choose batch size before allocating the final tensor.

    requested:
      "auto" or None -> memory-aware automatic batch
      positive int   -> explicit batch size
      "none"         -> all samples in one batch

    The automatic budget is
    ``available * memory_fraction - extra_bytes - safety_bytes``, and the
    batch is sized so that ``batch * bytes_per_sample * peak_multiplier``
    fits inside it. ``extra_bytes`` is for arrays that are already
    resident (the compact genotype matrix); ``peak_multiplier`` accounts
    for transient copies made after encoding (see
    :func:`peak_multiplier_for`).
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if not (0 < memory_fraction <= 1):
        raise ValueError("memory_fraction must be in (0, 1]")
    if peak_multiplier < 1:
        raise ValueError("peak_multiplier must be >= 1")

    bps = bytes_per_sample(sequence_length, mode, dtype, layout)
    full = bps * n_samples

    if requested == "none":
        return MemoryPlan(
            batch_size=n_samples,
            bytes_per_sample=bps,
            estimated_full_output_bytes=full,
            memory_budget_bytes=None,
            available_memory_bytes=None,
            peak_multiplier=peak_multiplier,
        )

    if isinstance(requested, bool):
        raise ValueError("batch_size must be 'auto', 'none', None, or an integer")

    if isinstance(requested, (int, np.integer)):
        if requested < 1:
            raise ValueError("batch_size integer must be >= 1")
        return MemoryPlan(
            batch_size=min(int(requested), n_samples),
            bytes_per_sample=bps,
            estimated_full_output_bytes=full,
            memory_budget_bytes=None,
            available_memory_bytes=None,
            peak_multiplier=peak_multiplier,
        )

    if requested not in {"auto", None}:
        raise ValueError("batch_size must be 'auto', 'none', None, or an integer")

    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError(
            "Automatic memory-aware batching requires psutil. "
            "Install hottools[mem] or set an explicit --batch-size."
        ) from exc

    available, constraints = effective_memory_available(
        int(psutil.virtual_memory().available),
        process_rss=int(psutil.Process().memory_info().rss),
    )
    safety = min(int(safety_bytes), max(0, int(available * 0.25)))
    budget = int(available * memory_fraction) - int(extra_bytes) - safety

    cost_per_sample = max(1, int(bps * peak_multiplier))
    if budget < cost_per_sample:
        warnings.warn(
            "Available memory is too small for the requested encoding: "
            f"{available / 1024**3:.2f} GiB available, "
            f"{bps * peak_multiplier / 1024**2:.1f} MiB needed per sample. "
            "Falling back to a single-sample batch; consider "
            "--dtype float16, a shorter region, or --format npy.",
            MemoryBudgetWarning,
            stacklevel=2,
        )
    batch = max(1, min(n_samples, budget // cost_per_sample))

    return MemoryPlan(
        batch_size=int(batch),
        bytes_per_sample=bps,
        estimated_full_output_bytes=full,
        memory_budget_bytes=max(0, budget),
        available_memory_bytes=available,
        peak_multiplier=peak_multiplier,
        extra_bytes=int(extra_bytes),
        safety_bytes=int(safety),
        memory_constraints=constraints,
    )


def human_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    x = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or unit == "TiB":
            return f"{x:.2f} {unit}"
        x /= 1024
    return f"{x:.2f} TiB"
