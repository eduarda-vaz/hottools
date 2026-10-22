from types import SimpleNamespace

import numpy as np
import pytest

from hottools.memory import (
    MemoryBudgetWarning,
    bytes_per_sample,
    choose_batch_size,
    full_output_shape,
    human_bytes,
    output_shape_per_sample,
    peak_multiplier_for,
    sequence_axis,
)


@pytest.fixture(autouse=True)
def isolate_host_arithmetic(monkeypatch):
    # Controller/scheduler handling is tested independently in test_memory_limits.
    monkeypatch.setattr(
        "hottools.memory.effective_memory_available",
        lambda available, **kwargs: (available, {"host_available": available}),
    )


@pytest.fixture
def fixed_available_memory(monkeypatch):
    """Test memory arithmetic independently of host RAM and transient usage."""
    import psutil

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=8 * 1024**3)
    )


def test_bytes_per_sample_float32():
    # 10 bp * 4 channels * 4 bytes
    assert bytes_per_sample(10, "dosage", "float32") == 160
    assert bytes_per_sample(10, "haplotypes", "float32") == 320


def test_bytes_per_sample_float16_halves():
    assert bytes_per_sample(10, "dosage", "float16") == 80


def test_bytes_per_sample_is_layout_invariant():
    assert bytes_per_sample(
        99, "dosage", "float32", "samples_first"
    ) == bytes_per_sample(99, "dosage", "float32", "channels_first")


@pytest.mark.parametrize(
    "mode,layout,expected",
    [
        ("dosage", "samples_first", (7, 4)),
        ("dosage", "channels_first", (4, 7)),
        ("haplotypes", "samples_first", (2, 7, 4)),
        ("haplotypes", "channels_first", (2, 4, 7)),
    ],
)
def test_output_shape_per_sample(mode, layout, expected):
    assert output_shape_per_sample(7, mode, layout) == expected
    assert full_output_shape(3, 7, mode, layout) == (3, *expected)


@pytest.mark.parametrize(
    "mode,layout,axis",
    [
        ("dosage", "samples_first", 1),
        ("dosage", "channels_first", 2),
        ("haplotypes", "samples_first", 2),
        ("haplotypes", "channels_first", 3),
    ],
)
def test_sequence_axis(mode, layout, axis):
    assert sequence_axis(mode, layout) == axis
    assert full_output_shape(3, 999, mode, layout)[axis] == 999


def test_invalid_mode_and_layout():
    with pytest.raises(ValueError):
        output_shape_per_sample(10, "nonsense")
    with pytest.raises(ValueError):
        output_shape_per_sample(10, "dosage", "sideways")


def test_peak_multiplier():
    assert peak_multiplier_for() == 1.0
    assert peak_multiplier_for(reverse_complement=True) == 2.0
    assert peak_multiplier_for(output_format="npz") == 2.0
    assert peak_multiplier_for(reverse_complement=True, output_format="pt") == 3.0


def test_explicit_batch_size_is_clamped():
    plan = choose_batch_size(
        n_samples=4,
        sequence_length=10,
        mode="dosage",
        dtype="float32",
        requested=99,
    )
    assert plan.batch_size == 4


def test_batch_size_none_uses_all_samples():
    plan = choose_batch_size(
        n_samples=17,
        sequence_length=10,
        mode="dosage",
        dtype="float32",
        requested="none",
    )
    assert plan.batch_size == 17
    assert plan.memory_budget_bytes is None


@pytest.mark.parametrize("bad", [0, -1, True, "many", 1.5])
def test_invalid_batch_size_requests(bad):
    with pytest.raises(ValueError):
        choose_batch_size(
            n_samples=4,
            sequence_length=10,
            mode="dosage",
            dtype="float32",
            requested=bad,
        )


def test_auto_batch_respects_budget_and_headroom(fixed_available_memory):
    """The chosen batch must fit the budget *including* transient copies."""
    plan = choose_batch_size(
        n_samples=100_000,
        sequence_length=100_000,
        mode="dosage",
        dtype="float32",
        requested="auto",
        peak_multiplier=2.0,
    )
    assert plan.available_memory_bytes > 0
    assert plan.safety_bytes > 0
    assert plan.estimated_peak_batch_bytes <= plan.memory_budget_bytes
    assert plan.peak_multiplier == 2.0


def test_reverse_complement_halves_the_batch(fixed_available_memory):
    kwargs = dict(
        n_samples=100_000,
        sequence_length=100_000,
        mode="dosage",
        dtype="float32",
        requested="auto",
    )
    plain = choose_batch_size(**kwargs, peak_multiplier=1.0)
    rc = choose_batch_size(**kwargs, peak_multiplier=2.0)
    assert rc.batch_size <= plain.batch_size // 2 + 1


def test_resident_genotype_array_shrinks_the_budget(fixed_available_memory):
    kwargs = dict(
        n_samples=100_000,
        sequence_length=100_000,
        mode="dosage",
        dtype="float32",
        requested="auto",
    )
    small = choose_batch_size(**kwargs, extra_bytes=0)
    large = choose_batch_size(**kwargs, extra_bytes=200 * 1024**2)
    assert large.batch_size < small.batch_size
    assert large.extra_bytes == 200 * 1024**2


def test_impossible_budget_warns_and_falls_back_to_one(fixed_available_memory):
    with pytest.warns(MemoryBudgetWarning):
        plan = choose_batch_size(
            n_samples=10,
            sequence_length=10**9,
            mode="haplotypes",
            dtype="float32",
            requested="auto",
        )
    assert plan.batch_size == 1


def test_memory_fraction_bounds():
    with pytest.raises(ValueError):
        choose_batch_size(
            n_samples=1,
            sequence_length=10,
            mode="dosage",
            dtype="float32",
            memory_fraction=0,
        )
    with pytest.raises(ValueError):
        choose_batch_size(
            n_samples=1,
            sequence_length=10,
            mode="dosage",
            dtype="float32",
            memory_fraction=1.5,
        )


def test_human_bytes():
    assert human_bytes(None) == "n/a"
    assert human_bytes(512) == "512.00 B"
    assert human_bytes(1024) == "1.00 KiB"
    assert human_bytes(5 * 1024**3) == "5.00 GiB"


def test_full_output_bytes_matches_numpy():
    plan = choose_batch_size(
        n_samples=6,
        sequence_length=25,
        mode="haplotypes",
        dtype="float16",
        requested="none",
    )
    shape = full_output_shape(6, 25, "haplotypes")
    assert plan.estimated_full_output_bytes == np.prod(shape) * 2


@pytest.mark.parametrize("available", [512 * 1024**2, 8 * 1024**3, 512 * 1024**3])
def test_auto_batch_formula_across_host_sizes(monkeypatch, available):
    import psutil

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=available)
    )
    plans = [
        choose_batch_size(
            n_samples=100_000,
            sequence_length=100_000,
            mode="dosage",
            dtype="float32",
            requested="auto",
            peak_multiplier=multiplier,
        )
        for multiplier in [1.0, 2.0]
    ]
    for plan in plans:
        budget = max(0, int(available * 0.4) - plan.safety_bytes)
        expected = min(
            100_000, max(1, int(budget // (1_600_000 * plan.peak_multiplier)))
        )
        assert plan.batch_size == expected
    # At high RAM the plain run can hit the sample-count ceiling: the
    # reverse-complement batch need not be half that clamped count.
    assert plans[1].batch_size <= plans[0].batch_size
